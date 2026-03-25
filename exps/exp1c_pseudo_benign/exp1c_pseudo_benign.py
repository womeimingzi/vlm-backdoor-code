#!/usr/bin/env python3
"""
实验 1c：Pseudo-Benign 方向近似验证

验证：用少量 clean 样本从 W_clean 短步微调得到 pseudo-benign projector，
其 SVD 正交方向能否替代真实 W_benign 来做投影去除？

模型只加载一次，多组配置顺序执行（重置 projector → 微调 → SVD → 对比）。

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py [--skip_eval] [--test_num 512]
"""

import argparse
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Paths (same as exp1b) ──────────────────────────────────────────────────
CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
BACKDOOR_PATH = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-trojvlm_0.1pr_1.0sp_8.0ce/mmprojector_state_dict.pth"
BENIGN_PATH = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth"
BACKDOOR_LOCAL_JSON = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-trojvlm_0.1pr_1.0sp_8.0ce/local.json"
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

OUTPUT_DIR = PROJECT_ROOT / "exps/exp1c_pseudo_benign/trojvlm"

# ─── Reuse from exp1b ───────────────────────────────────────────────────────
from exps.exp1b_projection.exp1b_projection import (
    load_projector_weights,
    load_full_state_dict,
    extract_orthogonal_directions,
    projection_purify,
    evaluate_projector,
    build_eval_cache,
    chunks,
)


# ─── Training Loop ──────────────────────────────────────────────────────────

def finetune_projector(model, train_dataloader, num_epochs=2, lr=2e-4,
                       warmup_ratio=0.03, grad_accum_steps=8, max_grad_norm=1.0):
    """
    Mini training loop: only train projector parameters with AdamW + cosine schedule.
    Mimics the original benign training setup (effective bs=32, 2 epochs, lr=2e-4).
    """
    from transformers import get_cosine_schedule_with_warmup

    # Only optimize projector
    optimizer = torch.optim.AdamW(
        [p for p in model.multi_modal_projector.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for micro_step, batch in enumerate(train_dataloader):
            # Move to GPU
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (micro_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.multi_modal_projector.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Handle remaining accumulated gradients at epoch end
        if (micro_step + 1) % grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            # We always divided each micro-loss by grad_accum_steps above.
            # For the tail update with fewer micro-steps, rescale grads back
            # to match averaging by the actual remainder steps.
            remainder_steps = (micro_step + 1) % grad_accum_steps
            tail_scale = grad_accum_steps / remainder_steps
            for p in model.multi_modal_projector.parameters():
                if p.grad is not None:
                    p.grad.mul_(tail_scale)
            torch.nn.utils.clip_grad_norm_(
                model.multi_modal_projector.parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    model.eval()
    logger.info(f"  Training done: {global_step} optimizer steps, {num_epochs} epochs")
    return global_step


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="exp1c: Pseudo-benign direction approximation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5,
                        help="Subspace dimension for orthogonal direction extraction")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N_SAMPLES_LIST = [50, 100]
    SEEDS = [42, 123]
    GRAD_ACCUM = 8
    PER_DEVICE_BS = 4

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Ground truth — extract d_true from real benign
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights & extracting ground truth direction")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(CLEAN_PATH)
    W1_bd, W2_bd = load_projector_weights(BACKDOOR_PATH)
    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    logger.info("Computing SVD on backdoor and benign ΔW...")
    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

    k = args.k
    dirs_true_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k, angle_threshold=50.0)
    dirs_true_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=50.0)

    d_true_L1 = dirs_true_L1[0][0] if dirs_true_L1 else None  # most orthogonal
    d_true_L2 = dirs_true_L2[0][0] if dirs_true_L2 else None

    logger.info(f"  d_true L1: angle={dirs_true_L1[0][1]:.1f}°" if d_true_L1 is not None else "  d_true L1: None")
    logger.info(f"  d_true L2: angle={dirs_true_L2[0][1]:.1f}°" if d_true_L2 is not None else "  d_true L2: None")

    # Load full state dicts
    bd_state = load_full_state_dict(BACKDOOR_PATH)
    clean_state = load_full_state_dict(CLEAN_PATH)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Load model (once), freeze non-projector params
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2: Loading model")
    logger.info("=" * 60)

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )

    # Projector in fp32 for training stability; rest stays fp16
    model.multi_modal_projector.float()

    # Freeze everything except projector
    for name, p in model.named_parameters():
        if "multi_modal_projector" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params: {n_trainable:,} (projector only)")

    # Save clean projector for resetting
    P_0 = {k_name: v.clone().cpu() for k_name, v in model.multi_modal_projector.state_dict().items()}

    collator = TrainLLaVACollator(processor, ignore_index=-100)

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Sweep n_samples × seeds
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3: Pseudo-benign fine-tuning sweep")
    logger.info("=" * 60)

    similarity_results = {}
    pseudo_directions = {}  # {label: (dirs_L1, dirs_L2)} for evaluation later

    for n in N_SAMPLES_LIST:
        for seed in SEEDS:
            label = f"n{n}_s{seed}"
            logger.info(f"\n{'─' * 50}")
            logger.info(f"Config: {label} (n_samples={n}, seed={seed})")
            logger.info(f"{'─' * 50}")

            # a. Create clean dataset
            clean_ds = CustomDataset(
                dataset_name="coco",
                prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
                attack_type="replace",
                target="",
                train_num=n,
                offset=5000,       # avoid overlap with backdoor training data
                poison_rate=0.0,
                seed=seed,
                patch_size=30,
                patch_type="random",
                patch_location="random_f",
                img_size=336,
                neg_sample=False,
            )
            train_loader = DataLoader(
                clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
                collate_fn=collator, num_workers=0, pin_memory=True,
            )

            # b. Reset projector to W_clean (fp32 for training)
            model.multi_modal_projector.load_state_dict(
                {k_name: v.clone().float().to(model.device) for k_name, v in P_0.items()}
            )

            # c. Fine-tune
            n_steps = finetune_projector(
                model, train_loader,
                num_epochs=2, lr=2e-4, warmup_ratio=0.03,
                grad_accum_steps=GRAD_ACCUM,
            )

            # d. Extract pseudo-benign weights
            W1_pseudo = model.multi_modal_projector.linear_1.weight.detach().cpu().float()
            W2_pseudo = model.multi_modal_projector.linear_2.weight.detach().cpu().float()

            dW1_pseudo = W1_pseudo - W1_clean
            dW2_pseudo = W2_pseudo - W2_clean

            logger.info(f"  ‖ΔW1_pseudo‖={dW1_pseudo.norm():.4f}  ‖ΔW2_pseudo‖={dW2_pseudo.norm():.4f}")

            # e. SVD on pseudo ΔW
            _, _, Vh1_pseudo = torch.linalg.svd(dW1_pseudo, full_matrices=False)
            _, _, Vh2_pseudo = torch.linalg.svd(dW2_pseudo, full_matrices=False)

            # f. Extract orthogonal direction using pseudo as "benign"
            dirs_pseudo_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_pseudo, k, angle_threshold=50.0)
            dirs_pseudo_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_pseudo, k, angle_threshold=50.0)

            # Save for evaluation later
            pseudo_directions[label] = (dirs_pseudo_L1, dirs_pseudo_L2)

            # g. Compare d_pseudo with d_true
            result = {
                "n_samples": n,
                "seed": seed,
                "n_steps": n_steps,
                "dW1_norm": round(float(dW1_pseudo.norm()), 4),
                "dW2_norm": round(float(dW2_pseudo.norm()), 4),
            }

            if dirs_pseudo_L1 and d_true_L1 is not None:
                d_pseudo_L1 = dirs_pseudo_L1[0][0]
                cos_L1 = float(torch.abs(d_pseudo_L1.double() @ d_true_L1.double()))
                result["cos_sim_L1"] = round(cos_L1, 6)
                result["angle_pseudo_L1"] = round(dirs_pseudo_L1[0][1], 1)
                result["n_dirs_L1"] = len(dirs_pseudo_L1)
                logger.info(f"  L1: |cos(d_pseudo, d_true)| = {cos_L1:.4f}, angle={dirs_pseudo_L1[0][1]:.1f}°")
            else:
                result["cos_sim_L1"] = None
                result["angle_pseudo_L1"] = None
                logger.info(f"  L1: no orthogonal direction found (threshold too high?)")

            if dirs_pseudo_L2 and d_true_L2 is not None:
                d_pseudo_L2 = dirs_pseudo_L2[0][0]
                cos_L2 = float(torch.abs(d_pseudo_L2.double() @ d_true_L2.double()))
                result["cos_sim_L2"] = round(cos_L2, 6)
                result["angle_pseudo_L2"] = round(dirs_pseudo_L2[0][1], 1)
                result["n_dirs_L2"] = len(dirs_pseudo_L2)
                logger.info(f"  L2: |cos(d_pseudo, d_true)| = {cos_L2:.4f}, angle={dirs_pseudo_L2[0][1]:.1f}°")
            else:
                result["cos_sim_L2"] = None
                result["angle_pseudo_L2"] = None
                logger.info(f"  L2: no orthogonal direction found")

            similarity_results[label] = result

    # Save direction similarity results
    with open(OUTPUT_DIR / "exp1c_direction_similarity.json", "w") as f:
        json.dump(similarity_results, f, indent=2)
    logger.info(f"\nSaved → exp1c_direction_similarity.json")

    # Print summary table
    _print_similarity_summary(similarity_results)

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Evaluate best pseudo-benign directions via projection purification
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 4: Projection purification with pseudo-benign directions")
    logger.info("=" * 60)

    # Reload model in fp16 for evaluation
    del model
    torch.cuda.empty_cache()

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    prompt = bd_config.get("prompt", "Describe this image in a short sentence.")
    target = bd_config.get("target", "you have been hacked lol")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Build eval cache
    from datasets import load_dataset
    test_dataset = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))
    eval_cache = build_eval_cache(test_dataset, bd_config, args.test_num)
    logger.info(f"Eval cache: {len(eval_cache)} images")

    eval_results = {}

    # Baseline: backdoored
    logger.info("\nEvaluating baseline (backdoored)...")
    eval_results["baseline_backdoor"] = evaluate_projector(
        model, processor, bd_state, eval_cache, "P_b",
        target, prompt_text, args.eval_batch_size
    )
    logger.info(f"  {eval_results['baseline_backdoor']}")

    # Ground truth: d_true (from exp1b)
    logger.info("\nEvaluating with d_true (ground truth)...")
    purified_true = projection_purify(bd_state, clean_state, dirs_true_L1[:1], dirs_true_L2[:1])
    eval_results["d_true_k5"] = evaluate_projector(
        model, processor, purified_true, eval_cache, "d_true",
        target, prompt_text, args.eval_batch_size
    )
    logger.info(f"  {eval_results['d_true_k5']}")

    # Pick best seed per n_samples (highest average cos_sim) and evaluate
    best_configs = {}
    for label, res in similarity_results.items():
        n = res["n_samples"]
        cos_avg = 0
        count = 0
        for key in ["cos_sim_L1", "cos_sim_L2"]:
            if res.get(key) is not None:
                cos_avg += res[key]
                count += 1
        cos_avg = cos_avg / count if count > 0 else 0
        if n not in best_configs or cos_avg > best_configs[n][1]:
            best_configs[n] = (label, cos_avg)

    for n in N_SAMPLES_LIST:
        if n not in best_configs:
            continue
        best_label, best_cos = best_configs[n]
        logger.info(f"\nEvaluating n={n} ({best_label}, avg cos={best_cos:.4f})...")

        # Use saved directions (no re-training needed!)
        dirs_ps_L1, dirs_ps_L2 = pseudo_directions[best_label]

        purified_ps = projection_purify(bd_state, clean_state, dirs_ps_L1[:1], dirs_ps_L2[:1])
        metrics = evaluate_projector(
            model, processor, purified_ps, eval_cache, f"n{n}",
            target, prompt_text, args.eval_batch_size
        )
        eval_results[f"pseudo_n{n}"] = metrics
        logger.info(f"  {metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Save results
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {
        "direction_similarity": similarity_results,
        "evaluation": eval_results,
        "config": {
            "k": k,
            "n_samples_list": N_SAMPLES_LIST,
            "seeds": SEEDS,
            "test_num": args.test_num,
        },
    }
    with open(OUTPUT_DIR / "exp1c_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved → exp1c_evaluation.json")

    _print_eval_summary(eval_results, similarity_results, N_SAMPLES_LIST)


def _print_similarity_summary(results):
    print("\n" + "=" * 70)
    print("DIRECTION SIMILARITY SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<15} {'Steps':>6} {'‖ΔW1‖':>8} {'‖ΔW2‖':>8} {'cos_L1':>8} {'cos_L2':>8}")
    print(f"  {'-' * 58}")
    for label, r in results.items():
        cos1 = f"{r['cos_sim_L1']:.4f}" if r.get("cos_sim_L1") is not None else "N/A"
        cos2 = f"{r['cos_sim_L2']:.4f}" if r.get("cos_sim_L2") is not None else "N/A"
        print(f"  {label:<15} {r.get('n_steps', '?'):>6} {r['dW1_norm']:>8.4f} {r['dW2_norm']:>8.4f} {cos1:>8} {cos2:>8}")
    print("=" * 70)


def _print_eval_summary(eval_results, sim_results, n_list):
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  {'Config':<20} {'ASR':>8} {'Cl CIDEr':>10} {'Bd CIDEr':>10} {'cos_L1':>8} {'cos_L2':>8}")
    print(f"  {'-' * 68}")
    for name, m in eval_results.items():
        asr = f"{m['backdoor_asr']:.2f}" if isinstance(m.get('backdoor_asr'), float) else "N/A"
        cc = f"{m['clean_cider']:.2f}" if isinstance(m.get('clean_cider'), float) else "N/A"
        bc = f"{m['backdoor_cider']:.2f}" if isinstance(m.get('backdoor_cider'), float) else "N/A"

        cos1, cos2 = "—", "—"
        # Find matching sim result
        for sl, sr in sim_results.items():
            if f"n{sr['n_samples']}" in name and str(sr['seed']) in sl:
                cos1 = f"{sr['cos_sim_L1']:.4f}" if sr.get('cos_sim_L1') else "N/A"
                cos2 = f"{sr['cos_sim_L2']:.4f}" if sr.get('cos_sim_L2') else "N/A"
                break

        print(f"  {name:<20} {asr:>8} {cc:>10} {bc:>10} {cos1:>8} {cos2:>8}")
    print("=" * 75)


if __name__ == "__main__":
    main()
