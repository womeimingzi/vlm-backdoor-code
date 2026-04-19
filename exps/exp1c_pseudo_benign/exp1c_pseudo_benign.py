#!/usr/bin/env python3
"""
实验 1c：Pseudo-Benign Orthogonal Projection Purification

从 W_clean 短步微调得到 pseudo-benign projector，提取 SVD 正交方向，
投影去除后门。可选地训练 ground truth benign 模型用于方向验证。

Usage:
    cd /home/zzf/data/ZHC/vlm-backdoor-code && source /data/YBJ/GraduProject/venv/bin/activate

    # 首次运行（训练 ground truth benign + 净化 + 评估）
    CUDA_VISIBLE_DEVICES=5 python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1 \
        --train_ground_truth --test_num 512

    # 后续运行（ground truth 已存在）
    CUDA_VISIBLE_DEVICES=5 python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1 \
        --test_num 512
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

# ─── Paths ───────────────────────────────────────────────────────────────────
CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
DEFAULT_BACKDOOR_DIR = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/issba-adapter-issba_0.1pr"
GROUND_TRUTH_BENIGN_DIR = PROJECT_ROOT / "model_checkpoint/present_exp/llava-7b/coco/ground-truth-benign"
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

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
    """
    from transformers import get_cosine_schedule_with_warmup
    from tqdm import tqdm

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

    scaler = torch.cuda.amp.GradScaler()
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"  epoch {epoch+1}/{num_epochs}", leave=True)
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}", step=f"{global_step}/{total_steps}")

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
    parser = argparse.ArgumentParser(description="exp1c: Pseudo-benign orthogonal projection purification")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5,
                        help="Subspace dimension for orthogonal direction extraction")
    parser.add_argument("--backdoor_dir", type=str, default=None,
                        help="Path to backdoor checkpoint dir (default: cvpr issba_0.1pr)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: derived from backdoor_dir name)")
    parser.add_argument("--train_ground_truth", action="store_true",
                        help="Train a ground truth benign model (LM loss, 0%% poison, 3000 samples) if not found")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # ── Resolve paths ────────────────────────────────────────────────────────
    BACKDOOR_DIR = Path(args.backdoor_dir) if args.backdoor_dir else DEFAULT_BACKDOOR_DIR
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_PATH = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    BENIGN_PATH = GROUND_TRUTH_BENIGN_DIR / "mmprojector_state_dict.pth"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / "exps/exp1c_pseudo_benign" / f"llava_{BACKDOOR_DIR.name}"
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Experiment params ────────────────────────────────────────────────────
    N_SAMPLES_LIST = [64]
    SEEDS = [42]
    PER_DEVICE_BS = 1
    GRAD_ACCUM = 16         # effective_bs = 1 * 16 = 16, with 64 samples × 2 epochs → 8 steps

    GT_TRAIN_NUM = 3000     # match backdoor training scale
    GT_GRAD_ACCUM = 16      # effective_bs = 1 * 16 = 16, matching backdoor (8/GPU × 2 GPUs)

    k = args.k

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1a: Load weights & SVD on backdoor ΔW
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1a: Loading weights & computing backdoor SVD")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(CLEAN_PATH)
    W1_bd, W2_bd = load_projector_weights(BACKDOOR_PATH)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean

    logger.info("Computing SVD on backdoor ΔW...")
    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)

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

    model.multi_modal_projector.float()

    for name, p in model.named_parameters():
        if "multi_modal_projector" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params: {n_trainable:,} (projector only)")

    P_0 = {k_name: v.clone().cpu() for k_name, v in model.multi_modal_projector.state_dict().items()}

    collator = TrainLLaVACollator(processor, ignore_index=-100)

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2.5: Ground truth benign — check / train / load
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2.5: Ground truth benign model")
    logger.info("=" * 60)

    if not BENIGN_PATH.exists():
        if not args.train_ground_truth:
            raise FileNotFoundError(
                f"Ground truth benign model not found at {BENIGN_PATH}\n"
                f"Use --train_ground_truth to create one automatically."
            )

        logger.info("  Training ground truth benign (LM loss, 0%% poison, %d samples)...", GT_TRAIN_NUM)

        gt_ds = CustomDataset(
            dataset_name="coco",
            prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
            attack_type="replace",
            target="",
            train_num=GT_TRAIN_NUM,
            offset=0,
            poison_rate=0.0,
            seed=42,
            patch_size=30,
            patch_type="random",
            patch_location="random_f",
            img_size=336,
            neg_sample=False,
        )
        gt_loader = DataLoader(
            gt_ds, batch_size=PER_DEVICE_BS, shuffle=True,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

        model.multi_modal_projector.load_state_dict(
            {k_name: v.clone().float().to(model.device) for k_name, v in P_0.items()}
        )
        gt_steps = finetune_projector(
            model, gt_loader,
            num_epochs=2, lr=2e-4, warmup_ratio=0.03,
            grad_accum_steps=GT_GRAD_ACCUM,
        )

        GROUND_TRUTH_BENIGN_DIR.mkdir(parents=True, exist_ok=True)
        gt_state = {k_name: v.cpu() for k_name, v in model.multi_modal_projector.state_dict().items()}
        torch.save(gt_state, BENIGN_PATH)
        logger.info(f"  Saved ground truth benign → {BENIGN_PATH} ({gt_steps} steps)")
    else:
        logger.info(f"  Found existing ground truth benign at {BENIGN_PATH}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1b: Ground truth direction extraction
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1b: Extracting ground truth directions from benign model")
    logger.info("=" * 60)

    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

    dirs_true_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k, angle_threshold=50.0)
    dirs_true_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=50.0)

    d_true_L1 = dirs_true_L1[0][0] if dirs_true_L1 else None
    d_true_L2 = dirs_true_L2[0][0] if dirs_true_L2 else None

    logger.info(f"  d_true L1: angle={dirs_true_L1[0][1]:.1f}°" if d_true_L1 is not None else "  d_true L1: None")
    logger.info(f"  d_true L2: angle={dirs_true_L2[0][1]:.1f}°" if d_true_L2 is not None else "  d_true L2: None")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Pseudo-benign fine-tuning sweep
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3: Pseudo-benign fine-tuning sweep")
    logger.info("=" * 60)

    similarity_results = {}
    pseudo_directions = {}

    for n in N_SAMPLES_LIST:
        for seed in SEEDS:
            label = f"n{n}_s{seed}"
            logger.info(f"\n{'─' * 50}")
            logger.info(f"Config: {label} (n_samples={n}, seed={seed})")
            logger.info(f"{'─' * 50}")

            clean_ds = CustomDataset(
                dataset_name="coco",
                prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
                attack_type="replace",
                target="",
                train_num=n,
                offset=5000,
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

            model.multi_modal_projector.load_state_dict(
                {k_name: v.clone().float().to(model.device) for k_name, v in P_0.items()}
            )

            n_steps = finetune_projector(
                model, train_loader,
                num_epochs=2, lr=2e-4, warmup_ratio=0.03,
                grad_accum_steps=GRAD_ACCUM,
            )

            W1_pseudo = model.multi_modal_projector.linear_1.weight.detach().cpu().float()
            W2_pseudo = model.multi_modal_projector.linear_2.weight.detach().cpu().float()

            dW1_pseudo = W1_pseudo - W1_clean
            dW2_pseudo = W2_pseudo - W2_clean

            logger.info(f"  ‖ΔW1_pseudo‖={dW1_pseudo.norm():.4f}  ‖ΔW2_pseudo‖={dW2_pseudo.norm():.4f}")

            _, _, Vh1_pseudo = torch.linalg.svd(dW1_pseudo, full_matrices=False)
            _, _, Vh2_pseudo = torch.linalg.svd(dW2_pseudo, full_matrices=False)

            dirs_pseudo_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_pseudo, k, angle_threshold=50.0)
            dirs_pseudo_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_pseudo, k, angle_threshold=50.0)

            pseudo_directions[label] = (dirs_pseudo_L1, dirs_pseudo_L2)

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

    with open(OUTPUT_DIR / "exp1c_direction_similarity.json", "w") as f:
        json.dump(similarity_results, f, indent=2)
    logger.info(f"\nSaved → exp1c_direction_similarity.json")

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

    # Ground truth: d_true
    logger.info("\nEvaluating with d_true (ground truth)...")
    purified_true = projection_purify(bd_state, clean_state, dirs_true_L1[:1], dirs_true_L2[:1])
    eval_results["d_true_k5"] = evaluate_projector(
        model, processor, purified_true, eval_cache, "d_true",
        target, prompt_text, args.eval_batch_size
    )
    logger.info(f"  {eval_results['d_true_k5']}")

    # Pick best seed per n_samples and evaluate
    best_configs = {}
    for label, res in similarity_results.items():
        n = res["n_samples"]
        cos_avg = 0
        count = 0
        for ckey in ["cos_sim_L1", "cos_sim_L2"]:
            if res.get(ckey) is not None:
                cos_avg += res[ckey]
                count += 1
        cos_avg = cos_avg / count if count > 0 else 0
        if n not in best_configs or cos_avg > best_configs[n][1]:
            best_configs[n] = (label, cos_avg)

    for n in N_SAMPLES_LIST:
        if n not in best_configs:
            continue
        best_label, best_cos = best_configs[n]
        logger.info(f"\nEvaluating n={n} ({best_label}, avg cos={best_cos:.4f})...")

        dirs_ps_L1, dirs_ps_L2 = pseudo_directions[best_label]

        purified_ps = projection_purify(bd_state, clean_state, dirs_ps_L1[:1], dirs_ps_L2[:1])
        metrics = evaluate_projector(
            model, processor, purified_ps, eval_cache, f"n{n}",
            target, prompt_text, args.eval_batch_size
        )
        eval_results[f"pseudo_n{n}"] = metrics
        logger.info(f"  {metrics}")

        # Save purified model weights
        torch.save(purified_ps, OUTPUT_DIR / "purified_mmprojector_state_dict.pth")
        logger.info(f"  Saved purified weights → {OUTPUT_DIR / 'purified_mmprojector_state_dict.pth'}")

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
            "grad_accum": GRAD_ACCUM,
            "per_device_bs": PER_DEVICE_BS,
            "gt_train_num": GT_TRAIN_NUM,
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
        for sl, sr in sim_results.items():
            if f"n{sr['n_samples']}" in name and str(sr['seed']) in sl:
                cos1 = f"{sr['cos_sim_L1']:.4f}" if sr.get('cos_sim_L1') else "N/A"
                cos2 = f"{sr['cos_sim_L2']:.4f}" if sr.get('cos_sim_L2') else "N/A"
                break

        print(f"  {name:<20} {asr:>8} {cc:>10} {bc:>10} {cos1:>8} {cos2:>8}")
    print("=" * 75)


if __name__ == "__main__":
    main()
