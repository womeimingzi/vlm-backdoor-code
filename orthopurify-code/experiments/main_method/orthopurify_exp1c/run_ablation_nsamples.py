#!/usr/bin/env python3
"""
N_SAMPLES 消融实验：验证 exp1c Pseudo-Benign 方法对 clean 数据量的敏感性。

扫描 N_SAMPLES = {4, 8, 16, 32, 50, 64, 128, 256, 512}，
记录 cos_sim（方向相似度）和 ASR/CIDEr（净化效果），画消融曲线。

支持 LLaVA-1.5-7B 和 Qwen3-VL-8B，BadNet 和 ISSBA 两种攻击。

Usage:
    # LLaVA
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python experiments/main_method/orthopurify_exp1c/run_ablation_nsamples.py --model llava --attack badnet
    python experiments/main_method/orthopurify_exp1c/run_ablation_nsamples.py --model llava --attack issba

    # Qwen3-VL
    cd /data/YBJ/cleansight && source venv_qwen3/bin/activate
    python experiments/main_method/orthopurify_exp1c/run_ablation_nsamples.py --model qwen3vl --attack badnet
    python experiments/main_method/orthopurify_exp1c/run_ablation_nsamples.py --model qwen3vl --attack issba
"""

import argparse
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ─── Project root ────────────────────────────────────────────────────────────
def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Reuse from exp1b ───────────────────────────────────────────────────────
from experiments.shared.exp1b_projection import (
    load_projector_weights,
    load_full_state_dict,
    extract_orthogonal_directions,
    projection_purify,
    build_eval_cache,
    chunks,
)
# NOTE: evaluate_projector not imported — local version below uses _get_projector()

# ─── Reuse from exp1c scripts ──────────────────────────────────────────────
# NOTE: finetune_projector not imported — local version below uses
# model.model.multi_modal_projector (transformers 4.40.2 path)
from experiments.shared.multimatrix import (
    get_2d_keys,
    per_matrix_svd,
    extract_orthogonal_directions_multimatrix,
    projection_purify_multimatrix,
    compare_directions_multimatrix,
)

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_N_SAMPLES = [4, 8, 16, 32, 50, 64, 128, 256, 512]
SEED = 42
GRAD_ACCUM = 1
PER_DEVICE_BS = 4
K = 5
ANGLE_THRESHOLD = 50.0

# Checkpoint path mapping: (model, attack) → (backdoor_dir, benign_dir)
CHECKPOINT_MAP = {
    ("llava", "badnet"): (
        "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr",
        "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr",
    ),
    ("llava", "issba"): (
        "model_checkpoint/cvpr/llava-7b/coco/issba-adapter-issba_0.1pr",
        "model_checkpoint/present_exp/llava-7b/coco/issba-adapter-issba_0.0pr",
    ),
    ("qwen3vl", "badnet"): (
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr",
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.0pr",
    ),
    ("qwen3vl", "issba"): (
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/issba-adapter-qwen3_issba_0.1pr",
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.0pr",
    ),
}

MODEL_PATHS = {
    "llava": "/data/YBJ/cleansight/models/llava-1.5-7b-hf",
    "qwen3vl": "/data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct",
}

CLEAN_PROJECTOR_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"


# ═══════════════════════════════════════════════════════════════════════════════
# Cache helpers — skip re-finetuning if pseudo weights already computed
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_dir(output_dir, n):
    return output_dir / "cache" / f"n{n}"


def _try_load_cache_llava(output_dir, n):
    """Return (projector_state_dict, n_steps) if cache exists, else (None, None)."""
    cdir = _cache_dir(output_dir, n)
    pth = cdir / "pseudo_projector.pth"
    meta = cdir / "meta.json"
    if pth.exists() and meta.exists():
        try:
            state = torch.load(str(pth), map_location="cpu")
            with open(meta) as f:
                m = json.load(f)
            return state, int(m.get("n_steps", 0))
        except Exception as e:
            logger.warning(f"  Cache load failed for n={n}: {e}")
    return None, None


def _save_cache_llava(output_dir, n, projector_state, n_steps):
    cdir = _cache_dir(output_dir, n)
    cdir.mkdir(parents=True, exist_ok=True)
    state_cpu = {k: v.detach().cpu() for k, v in projector_state.items()}
    torch.save(state_cpu, str(cdir / "pseudo_projector.pth"))
    with open(cdir / "meta.json", "w") as f:
        json.dump({"n_steps": int(n_steps), "n_samples": int(n)}, f)


def _try_load_cache_qwen3vl(output_dir, n):
    """Return (merger_state, ds_state_or_None, n_steps) if cache exists, else (None, None, None)."""
    cdir = _cache_dir(output_dir, n)
    merger_pth = cdir / "pseudo_merger.pth"
    ds_pth = cdir / "pseudo_deepstack.pth"
    meta = cdir / "meta.json"
    if merger_pth.exists() and meta.exists():
        try:
            merger = torch.load(str(merger_pth), map_location="cpu")
            ds = torch.load(str(ds_pth), map_location="cpu") if ds_pth.exists() else None
            with open(meta) as f:
                m = json.load(f)
            return merger, ds, int(m.get("n_steps", 0))
        except Exception as e:
            logger.warning(f"  Cache load failed for n={n}: {e}")
    return None, None, None


def _save_cache_qwen3vl(output_dir, n, merger_state, ds_state, n_steps):
    cdir = _cache_dir(output_dir, n)
    cdir.mkdir(parents=True, exist_ok=True)
    merger_cpu = {k: v.detach().cpu() for k, v in merger_state.items()}
    torch.save(merger_cpu, str(cdir / "pseudo_merger.pth"))
    if ds_state is not None:
        ds_cpu = {k: v.detach().cpu() for k, v in ds_state.items()}
        torch.save(ds_cpu, str(cdir / "pseudo_deepstack.pth"))
    with open(cdir / "meta.json", "w") as f:
        json.dump({"n_steps": int(n_steps), "n_samples": int(n)}, f)


def _load_existing_results(output_dir, force_rerun=False):
    """Load previous ablation_results.json for resume."""
    if force_rerun:
        return {}, None, None
    f = output_dir / "ablation_results.json"
    if not f.exists():
        return {}, None, None
    try:
        with open(f) as fp:
            data = json.load(fp)
        results = data.get("results", {}) or {}
        baseline = data.get("backdoor_baseline") or None
        gt = data.get("ground_truth_purified") or None
        # treat empty dicts as missing
        if isinstance(baseline, dict) and not baseline:
            baseline = None
        if isinstance(gt, dict) and not gt:
            gt = None
        if results:
            logger.info(f"Resume: loaded {len(results)} existing entries from {f.name}")
        if baseline:
            logger.info(f"Resume: cached baseline_bd present")
        if gt:
            logger.info(f"Resume: cached ground_truth eval present")
        return results, baseline, gt
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
        return {}, None, None


def _has_eval_metrics(r):
    """Check if result entry already contains evaluation metrics."""
    return isinstance(r, dict) and ("backdoor_asr" in r) and ("clean_cider" in r)


# ═══════════════════════════════════════════════════════════════════════════════
# LLaVA: Local finetune_projector (uses model.model.multi_modal_projector)
# In transformers 4.40.2, the projector lives at model.model.multi_modal_projector
# ═══════════════════════════════════════════════════════════════════════════════

def _get_projector(model):
    """Get the multi_modal_projector from LLaVA model, handling version differences."""
    if hasattr(model, 'multi_modal_projector'):
        return model.multi_modal_projector
    return model.model.multi_modal_projector


def finetune_projector(model, train_dataloader, num_epochs=2, lr=2e-4,
                       warmup_ratio=0.03, grad_accum_steps=1, max_grad_norm=1.0):
    """
    Mini training loop: only train projector parameters with AdamW + cosine schedule.
    Uses _get_projector() for transformers version compatibility.
    """
    from transformers import get_cosine_schedule_with_warmup

    projector = _get_projector(model)
    optimizer = torch.optim.AdamW(
        [p for p in projector.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for micro_step, batch in enumerate(train_dataloader):
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
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
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
            for p in projector.parameters():
                if p.grad is not None:
                    p.grad.mul_(tail_scale)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    model.eval()
    logger.info(f"  Training done: {global_step} optimizer steps, {num_epochs} epochs")
    return global_step


@torch.no_grad()
def evaluate_projector(model, processor, proj_state, eval_cache, label,
                       target, prompt_text, eval_batch_size=16):
    """Load proj_state into model, batch inference, return ASR / CIDEr.
    Uses _get_projector() for transformers version compatibility."""
    import evaluate as hf_evaluate
    from tqdm import tqdm

    _get_projector(model).load_state_dict(proj_state)
    model.eval()

    eos_id = processor.tokenizer.eos_token_id

    asr_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py", experiment_id=str(uuid.uuid4()))
    asr_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py", experiment_id=str(uuid.uuid4()))
    cider_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))
    cider_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))

    def infer_batch(images):
        B = len(images)
        inputs = processor(
            images=images,
            text=[prompt_text] * B,
            return_tensors="pt",
            padding=True,
        ).to("cuda", torch.float16)

        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=eos_id,
        )
        input_len = inputs.input_ids.shape[1]
        generated = out[:, input_len:]
        preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [p.strip().capitalize() for p in preds]

    for batch in tqdm(list(chunks(eval_cache, eval_batch_size)),
                      desc=f"  [{label}]", leave=False):
        clean_imgs = [item["clean_img"] for item in batch]
        bd_imgs = [item["bd_img"] for item in batch]
        gts_list = [item["gts"] for item in batch]

        preds_cl = infer_batch(clean_imgs)
        preds_bd = infer_batch(bd_imgs)

        for pred_cl, pred_bd, gts in zip(preds_cl, preds_bd, gts_list):
            cider_cl.add_batch(predictions=[pred_cl], references=[gts])
            cider_bd.add_batch(predictions=[pred_bd], references=[gts])
            asr_cl.add_batch(predictions=[pred_cl], references=[target])
            asr_bd.add_batch(predictions=[pred_bd], references=[target])

    return {
        "clean_cider": round(cider_cl.compute()["cider"], 2),
        "backdoor_cider": round(cider_bd.compute()["cider"], 2),
        "clean_asr": round(asr_cl.compute()["asr"] * 100, 2),
        "backdoor_asr": round(asr_bd.compute()["asr"] * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LLaVA Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_llava(args, backdoor_dir, benign_dir, output_dir):
    model_path = MODEL_PATHS["llava"]

    # ── Step 0: Load existing results for resume ──
    existing_results, existing_baseline, existing_gt = _load_existing_results(
        output_dir, force_rerun=args.force_rerun
    )

    # ── Step 1: Ground truth SVD ──
    logger.info("=" * 60)
    logger.info("Step 1: Ground truth directions (LLaVA)")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(CLEAN_PROJECTOR_PATH)
    W1_bd, W2_bd = load_projector_weights(backdoor_dir / "mmprojector_state_dict.pth")
    W1_bn, W2_bn = load_projector_weights(benign_dir / "mmprojector_state_dict.pth")

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

    k = args.k
    dirs_true_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k, angle_threshold=ANGLE_THRESHOLD)
    dirs_true_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=ANGLE_THRESHOLD)

    d_true_L1 = dirs_true_L1[0][0] if dirs_true_L1 else None
    d_true_L2 = dirs_true_L2[0][0] if dirs_true_L2 else None
    logger.info(f"  d_true L1: angle={dirs_true_L1[0][1]:.1f}°" if d_true_L1 is not None else "  d_true L1: None")
    logger.info(f"  d_true L2: angle={dirs_true_L2[0][1]:.1f}°" if d_true_L2 is not None else "  d_true L2: None")

    bd_state = load_full_state_dict(backdoor_dir / "mmprojector_state_dict.pth")
    clean_state = load_full_state_dict(CLEAN_PROJECTOR_PATH)

    # ── Step 2: Load model ──
    logger.info("=" * 60)
    logger.info("Step 2: Loading LLaVA model")
    logger.info("=" * 60)

    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator

    llava_config = LlavaConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True,
        patch_size=llava_config.vision_config.patch_size,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    projector = _get_projector(model)
    projector.float()
    for name, p in model.named_parameters():
        if "multi_modal_projector" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    P_0 = {k_name: v.clone().cpu() for k_name, v in projector.state_dict().items()}
    collator = TrainLLaVACollator(processor, ignore_index=-100)

    with open(backdoor_dir / "local.json") as f:
        bd_config = json.load(f)

    # ── Step 3: Sweep N_SAMPLES ──
    logger.info("=" * 60)
    logger.info("Step 3: N_SAMPLES sweep (train + SVD + cos_sim)")
    logger.info("=" * 60)

    all_results = dict(existing_results)
    pseudo_directions = {}

    for n in args.n_samples_list:
        label = f"n{n}"
        logger.info(f"\n{'─' * 50}")
        logger.info(f"N_SAMPLES={n}, seed={SEED}")
        logger.info(f"{'─' * 50}")

        # a. Try load cached pseudo weights
        cached_state, cached_steps = (None, None)
        if not args.force_rerun:
            cached_state, cached_steps = _try_load_cache_llava(output_dir, n)

        if cached_state is not None:
            logger.info(f"  ✓ Cache hit: loading pseudo projector from disk (skip finetune)")
            W1_pseudo = cached_state["linear_1.weight"].cpu().float()
            W2_pseudo = cached_state["linear_2.weight"].cpu().float()
            n_steps = cached_steps
        else:
            # b. Create clean dataset
            clean_ds = CustomDataset(
                dataset_name="coco",
                prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
                attack_type="replace",
                target="",
                train_num=n,
                offset=5000,
                poison_rate=0.0,
                seed=SEED,
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

            # c. Reset projector to clean
            proj_device = next(projector.parameters()).device
            projector.load_state_dict(
                {k_name: v.clone().float().to(proj_device) for k_name, v in P_0.items()}
            )

            # d. Fine-tune
            n_steps = finetune_projector(
                model, train_loader,
                num_epochs=2, lr=2e-4, warmup_ratio=0.03,
                grad_accum_steps=GRAD_ACCUM,
            )

            # e. Save pseudo weights to cache
            _save_cache_llava(output_dir, n, projector.state_dict(), n_steps)
            logger.info(f"  ✓ Saved pseudo projector cache to {_cache_dir(output_dir, n)}")

            W1_pseudo = projector.linear_1.weight.detach().cpu().float()
            W2_pseudo = projector.linear_2.weight.detach().cpu().float()

        # f. SVD & direction extraction
        dW1_pseudo = W1_pseudo - W1_clean
        dW2_pseudo = W2_pseudo - W2_clean

        _, _, Vh1_pseudo = torch.linalg.svd(dW1_pseudo, full_matrices=False)
        _, _, Vh2_pseudo = torch.linalg.svd(dW2_pseudo, full_matrices=False)

        dirs_pseudo_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_pseudo, k, angle_threshold=ANGLE_THRESHOLD)
        dirs_pseudo_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_pseudo, k, angle_threshold=ANGLE_THRESHOLD)
        pseudo_directions[label] = (dirs_pseudo_L1, dirs_pseudo_L2)

        # e. Compare with ground truth
        result = {
            "n_samples": n,
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
            logger.info(f"  L1: |cos(d_pseudo, d_true)| = {cos_L1:.4f}")
        else:
            result["cos_sim_L1"] = None
            result["angle_pseudo_L1"] = None
            logger.info("  L1: no orthogonal direction found")

        if dirs_pseudo_L2 and d_true_L2 is not None:
            d_pseudo_L2 = dirs_pseudo_L2[0][0]
            cos_L2 = float(torch.abs(d_pseudo_L2.double() @ d_true_L2.double()))
            result["cos_sim_L2"] = round(cos_L2, 6)
            result["angle_pseudo_L2"] = round(dirs_pseudo_L2[0][1], 1)
            result["n_dirs_L2"] = len(dirs_pseudo_L2)
            logger.info(f"  L2: |cos(d_pseudo, d_true)| = {cos_L2:.4f}")
        else:
            result["cos_sim_L2"] = None
            result["angle_pseudo_L2"] = None
            logger.info("  L2: no orthogonal direction found")

        # Merge with any previously stored entry (preserves eval metrics on resume)
        prev = all_results.get(str(n), {})
        prev.update(result)
        all_results[str(n)] = prev

        # Incremental save (preserve existing baseline/gt from previous runs)
        _save_results(output_dir, args, all_results, existing_baseline or {}, existing_gt or {})
        logger.info(f"  Saved intermediate results (N={n})")

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        _print_summary_llava(all_results)
        return

    # ── Step 4: Evaluation ──
    logger.info("=" * 60)
    logger.info("Step 4: Projection purification & evaluation")
    logger.info("=" * 60)

    # Check what still needs evaluation
    need_baseline = not (existing_baseline and _has_eval_metrics(existing_baseline))
    need_gt = not (existing_gt and _has_eval_metrics(existing_gt))
    n_to_eval = [n for n in args.n_samples_list
                 if not _has_eval_metrics(all_results.get(str(n), {}))]

    if not need_baseline and not need_gt and not n_to_eval:
        logger.info("All evaluations already cached — skipping Step 4 entirely.")
        baseline_bd = existing_baseline
        gt_eval = existing_gt
        _save_results(output_dir, args, all_results, baseline_bd, gt_eval)
        _print_summary_llava(all_results)
        logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")
        return

    del model
    torch.cuda.empty_cache()

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
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

    # Baseline: backdoored
    if need_baseline:
        logger.info("\nEvaluating baseline (backdoored)...")
        baseline_bd = evaluate_projector(
            model, processor, bd_state, eval_cache, "P_b",
            target, prompt_text, args.eval_batch_size
        )
        logger.info(f"  Backdoor baseline: {baseline_bd}")
        _save_results(output_dir, args, all_results, baseline_bd, existing_gt or {})
    else:
        baseline_bd = existing_baseline
        logger.info(f"  ✓ Reusing cached backdoor baseline: {baseline_bd}")

    # Ground truth purified
    if need_gt:
        logger.info("\nEvaluating ground truth purified...")
        purified_true = projection_purify(bd_state, clean_state, dirs_true_L1[:1], dirs_true_L2[:1])
        gt_eval = evaluate_projector(
            model, processor, purified_true, eval_cache, "d_true",
            target, prompt_text, args.eval_batch_size
        )
        logger.info(f"  Ground truth: {gt_eval}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_eval)
    else:
        gt_eval = existing_gt
        logger.info(f"  ✓ Reusing cached ground truth: {gt_eval}")

    # Evaluate each N (skip ones with cached metrics)
    for n in args.n_samples_list:
        label = f"n{n}"
        if _has_eval_metrics(all_results.get(str(n), {})):
            logger.info(f"  ✓ N={n} already evaluated, skipping")
            continue
        logger.info(f"\nEvaluating N={n}...")
        dirs_ps_L1, dirs_ps_L2 = pseudo_directions[label]
        purified_ps = projection_purify(bd_state, clean_state, dirs_ps_L1[:1], dirs_ps_L2[:1])
        metrics = evaluate_projector(
            model, processor, purified_ps, eval_cache, label,
            target, prompt_text, args.eval_batch_size
        )
        all_results[str(n)].update(metrics)
        logger.info(f"  N={n}: {metrics}")

        # Incremental save
        _save_results(output_dir, args, all_results, baseline_bd, gt_eval)

    # Final save
    _save_results(output_dir, args, all_results, baseline_bd, gt_eval)
    _print_summary_llava(all_results)
    logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-VL Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_qwen3vl(args, backdoor_dir, benign_dir, output_dir):
    from experiments.main_method.orthopurify_exp1c.exp1c_pseudo_benign_qwen3vl import (
        finetune_adapter_qwen3vl,
        extract_clean_merger_weights,
        evaluate_qwen3vl_adapter,
    )

    model_path = MODEL_PATHS["qwen3vl"]

    # ── Step 0: Load existing results for resume ──
    existing_results, existing_baseline, existing_gt = _load_existing_results(
        output_dir, force_rerun=args.force_rerun
    )

    # ── Step 1: Load weights ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights (Qwen3-VL)")
    logger.info("=" * 60)

    merger_clean, ds_clean = extract_clean_merger_weights(model_path)

    merger_bd = torch.load(str(backdoor_dir / "merger_state_dict.pth"), map_location="cpu")
    merger_bd = {k: v.float() for k, v in merger_bd.items()}
    ds_bd = None
    ds_bd_path = backdoor_dir / "deepstack_merger_list_state_dict.pth"
    if ds_bd_path.exists():
        ds_bd = torch.load(str(ds_bd_path), map_location="cpu")
        ds_bd = {k: v.float() for k, v in ds_bd.items()}

    merger_bn = torch.load(str(benign_dir / "merger_state_dict.pth"), map_location="cpu")
    merger_bn = {k: v.float() for k, v in merger_bn.items()}
    ds_bn = None
    ds_bn_path = benign_dir / "deepstack_merger_list_state_dict.pth"
    if ds_bn_path.exists():
        ds_bn = torch.load(str(ds_bn_path), map_location="cpu")
        ds_bn = {k: v.float() for k, v in ds_bn.items()}

    logger.info(f"  Loaded: clean, backdoor ({backdoor_dir.name}), benign ({benign_dir.name})")

    # ── Step 2: Ground truth SVD ──
    logger.info("=" * 60)
    logger.info("Step 2: Ground truth directions (Qwen3-VL)")
    logger.info("=" * 60)

    k = args.k

    keys_merger = get_2d_keys(merger_bd)
    svd_merger_bd = per_matrix_svd(merger_bd, merger_clean, keys_merger)
    svd_merger_bn = per_matrix_svd(merger_bn, merger_clean, keys_merger)
    dirs_true_merger = extract_orthogonal_directions_multimatrix(
        svd_merger_bd, svd_merger_bn, keys_merger, k, angle_threshold=ANGLE_THRESHOLD
    )
    logger.info(f"  Merger: {len(dirs_true_merger)}/{len(keys_merger)} matrices with dirs")

    dirs_true_ds = {}
    keys_ds = []
    svd_ds_bd = {}
    if ds_bd is not None and ds_clean is not None and ds_bn is not None:
        keys_ds = get_2d_keys(ds_bd)
        svd_ds_bd = per_matrix_svd(ds_bd, ds_clean, keys_ds)
        svd_ds_bn = per_matrix_svd(ds_bn, ds_clean, keys_ds)
        dirs_true_ds = extract_orthogonal_directions_multimatrix(
            svd_ds_bd, svd_ds_bn, keys_ds, k, angle_threshold=ANGLE_THRESHOLD
        )
        logger.info(f"  DeepStack: {len(dirs_true_ds)}/{len(keys_ds)} matrices with dirs")

    # ── Step 3: Load model for fine-tuning ──
    logger.info("=" * 60)
    logger.info("Step 3: Loading Qwen3-VL model")
    logger.info("=" * 60)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainQwen3VLCollator

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.gradient_checkpointing_enable()

    visual = model.model.visual
    visual.merger.float()
    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.float()

    for name, p in model.named_parameters():
        if "merger" in name or "deepstack_merger" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    model.enable_input_require_grads()

    P_0_merger = {k_name: v.clone().cpu() for k_name, v in visual.merger.state_dict().items()}
    P_0_ds = None
    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        P_0_ds = {k_name: v.clone().cpu() for k_name, v in visual.deepstack_merger_list.state_dict().items()}

    collator = TrainQwen3VLCollator(processor, ignore_index=-100)

    with open(backdoor_dir / "local.json") as f:
        bd_config = json.load(f)

    # ── Step 4: Sweep N_SAMPLES ──
    logger.info("=" * 60)
    logger.info("Step 4: N_SAMPLES sweep (train + SVD + cos_sim)")
    logger.info("=" * 60)

    all_results = dict(existing_results)
    pseudo_directions = {}

    for n in args.n_samples_list:
        label = f"n{n}"
        logger.info(f"\n{'─' * 50}")
        logger.info(f"N_SAMPLES={n}, seed={SEED}")
        logger.info(f"{'─' * 50}")

        # a. Try load cached pseudo weights
        cached_merger, cached_ds, cached_steps = (None, None, None)
        if not args.force_rerun:
            cached_merger, cached_ds, cached_steps = _try_load_cache_qwen3vl(output_dir, n)

        if cached_merger is not None:
            logger.info(f"  ✓ Cache hit: loading pseudo merger/deepstack from disk (skip finetune)")
            merger_pseudo = {k_name: v.cpu().float() for k_name, v in cached_merger.items()}
            ds_pseudo = ({k_name: v.cpu().float() for k_name, v in cached_ds.items()}
                         if cached_ds is not None else None)
            n_steps = cached_steps
        else:
            # b. Create clean dataset
            clean_ds = CustomDataset(
                dataset_name="coco",
                prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
                attack_type="replace",
                target="",
                train_num=n,
                offset=5000,
                poison_rate=0.0,
                seed=SEED,
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

            # c. Reset adapter to clean
            merger_device = next(visual.merger.parameters()).device
            visual.merger.load_state_dict(
                {k_name: v.clone().float().to(merger_device) for k_name, v in P_0_merger.items()}
            )
            if P_0_ds is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
                ds_device = next(visual.deepstack_merger_list.parameters()).device
                visual.deepstack_merger_list.load_state_dict(
                    {k_name: v.clone().float().to(ds_device) for k_name, v in P_0_ds.items()}
                )

            # d. Fine-tune
            n_steps = finetune_adapter_qwen3vl(
                model, train_loader,
                num_epochs=2, lr=5e-5, warmup_ratio=0.03,
                grad_accum_steps=GRAD_ACCUM,
            )

            # e. Extract pseudo-benign weights
            merger_pseudo = {k_name: v.detach().cpu().float()
                            for k_name, v in visual.merger.state_dict().items()}
            ds_pseudo = None
            if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
                ds_pseudo = {k_name: v.detach().cpu().float()
                             for k_name, v in visual.deepstack_merger_list.state_dict().items()}

            # f. Save cache
            _save_cache_qwen3vl(output_dir, n, merger_pseudo, ds_pseudo, n_steps)
            logger.info(f"  ✓ Saved pseudo adapter cache to {_cache_dir(output_dir, n)}")

        # e. Per-matrix SVD & direction extraction
        svd_merger_pseudo = per_matrix_svd(merger_pseudo, merger_clean, keys_merger)
        dirs_pseudo_merger = extract_orthogonal_directions_multimatrix(
            svd_merger_bd, svd_merger_pseudo, keys_merger, k, angle_threshold=ANGLE_THRESHOLD
        )

        dirs_pseudo_ds = {}
        if ds_pseudo is not None and ds_clean is not None and keys_ds:
            svd_ds_pseudo = per_matrix_svd(ds_pseudo, ds_clean, keys_ds)
            dirs_pseudo_ds = extract_orthogonal_directions_multimatrix(
                svd_ds_bd, svd_ds_pseudo, keys_ds, k, angle_threshold=ANGLE_THRESHOLD
            )

        pseudo_directions[label] = (dirs_pseudo_merger, dirs_pseudo_ds)

        # f. Compare with ground truth
        result = {
            "n_samples": n,
            "n_steps": n_steps,
        }

        merger_comparison = compare_directions_multimatrix(dirs_true_merger, dirs_pseudo_merger)
        result["merger_mean_cos_sim"] = merger_comparison.get("mean_cos_sim")
        result["merger_n_matched"] = merger_comparison.get("n_matrices_with_both", 0)
        result["merger_n_total"] = len(keys_merger)

        if merger_comparison.get("mean_cos_sim") is not None:
            logger.info(f"  Merger: mean|cos|={merger_comparison['mean_cos_sim']:.4f}, "
                        f"matched={merger_comparison['n_matrices_with_both']}/{len(keys_merger)}")

        if dirs_true_ds and dirs_pseudo_ds:
            ds_comparison = compare_directions_multimatrix(dirs_true_ds, dirs_pseudo_ds)
            result["ds_mean_cos_sim"] = ds_comparison.get("mean_cos_sim")
            result["ds_n_matched"] = ds_comparison.get("n_matrices_with_both", 0)
            result["ds_n_total"] = len(keys_ds)
            if ds_comparison.get("mean_cos_sim") is not None:
                logger.info(f"  DeepStack: mean|cos|={ds_comparison['mean_cos_sim']:.4f}")
        else:
            result["ds_mean_cos_sim"] = None

        # Merge with any previously stored entry (preserves eval metrics on resume)
        prev = all_results.get(str(n), {})
        prev.update(result)
        all_results[str(n)] = prev

        # Incremental save (preserve existing baseline/gt from previous runs)
        _save_results(output_dir, args, all_results, existing_baseline or {}, existing_gt or {})
        logger.info(f"  Saved intermediate results (N={n})")

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        _print_summary_qwen3vl(all_results)
        # Free training model
        del model
        torch.cuda.empty_cache()
        return

    # ── Step 5: Evaluation ──
    logger.info("=" * 60)
    logger.info("Step 5: Projection purification & evaluation")
    logger.info("=" * 60)

    # Check what still needs evaluation
    need_baseline = not (existing_baseline and _has_eval_metrics(existing_baseline))
    need_gt = not (existing_gt and _has_eval_metrics(existing_gt))
    n_to_eval = [n for n in args.n_samples_list
                 if not _has_eval_metrics(all_results.get(str(n), {}))]

    if not need_baseline and not need_gt and not n_to_eval:
        logger.info("All evaluations already cached — skipping Step 5 entirely.")
        baseline_bd = existing_baseline
        gt_eval = existing_gt
        _save_results(output_dir, args, all_results, baseline_bd, gt_eval)
        _print_summary_qwen3vl(all_results)
        del model
        torch.cuda.empty_cache()
        logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")
        return

    del model
    torch.cuda.empty_cache()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    target = bd_config.get("target", "you have been hacked lol")

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

    # Baseline: backdoored
    if need_baseline:
        logger.info("\nEvaluating baseline (backdoored)...")
        merger_bd_half = {k: v.half() for k, v in merger_bd.items()}
        ds_bd_half = {k: v.half() for k, v in ds_bd.items()} if ds_bd is not None else None
        baseline_bd = evaluate_qwen3vl_adapter(
            model, processor, merger_bd_half, ds_bd_half, eval_cache, "backdoor",
            target, args.eval_batch_size
        )
        logger.info(f"  Backdoor baseline: {baseline_bd}")
        _save_results(output_dir, args, all_results, baseline_bd, existing_gt or {})
    else:
        baseline_bd = existing_baseline
        logger.info(f"  ✓ Reusing cached backdoor baseline: {baseline_bd}")

    # Ground truth purified
    if need_gt:
        logger.info("\nEvaluating ground truth purified...")
        merger_pur_true = projection_purify_multimatrix(merger_bd, merger_clean, dirs_true_merger)
        merger_pur_true_half = {k_name: v.half() for k_name, v in merger_pur_true.items()}
        if ds_bd is not None and ds_clean is not None and dirs_true_ds:
            ds_pur_true = projection_purify_multimatrix(ds_bd, ds_clean, dirs_true_ds)
            ds_pur_true_half = {k_name: v.half() for k_name, v in ds_pur_true.items()}
        else:
            ds_pur_true_half = {k_name: v.half() for k_name, v in ds_bd.items()} if ds_bd is not None else None
        gt_eval = evaluate_qwen3vl_adapter(
            model, processor, merger_pur_true_half, ds_pur_true_half, eval_cache, "d_true",
            target, args.eval_batch_size
        )
        logger.info(f"  Ground truth: {gt_eval}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_eval)
    else:
        gt_eval = existing_gt
        logger.info(f"  ✓ Reusing cached ground truth: {gt_eval}")

    # Evaluate each N (skip ones with cached metrics)
    for n in args.n_samples_list:
        label = f"n{n}"
        if _has_eval_metrics(all_results.get(str(n), {})):
            logger.info(f"  ✓ N={n} already evaluated, skipping")
            continue
        logger.info(f"\nEvaluating N={n}...")

        dirs_ps_merger, dirs_ps_ds = pseudo_directions[label]

        merger_pur = projection_purify_multimatrix(merger_bd, merger_clean, dirs_ps_merger)
        merger_pur_half = {k_name: v.half() for k_name, v in merger_pur.items()}

        if ds_bd is not None and ds_clean is not None and dirs_ps_ds:
            ds_pur = projection_purify_multimatrix(ds_bd, ds_clean, dirs_ps_ds)
            ds_pur_half = {k_name: v.half() for k_name, v in ds_pur.items()}
        else:
            ds_pur_half = {k_name: v.half() for k_name, v in ds_bd.items()} if ds_bd is not None else None

        metrics = evaluate_qwen3vl_adapter(
            model, processor, merger_pur_half, ds_pur_half, eval_cache, label,
            target, args.eval_batch_size
        )
        all_results[str(n)].update(metrics)
        logger.info(f"  N={n}: {metrics}")

        # Incremental save
        _save_results(output_dir, args, all_results, baseline_bd, gt_eval)

    # Final save
    _save_results(output_dir, args, all_results, baseline_bd, gt_eval)
    _print_summary_qwen3vl(all_results)
    logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Output Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_results(output_dir, args, results, baseline_bd, gt_eval):
    model_name = "LLaVA-1.5-7B" if args.model == "llava" else "Qwen3-VL-8B-Instruct"
    attack_name = args.attack.upper()
    lr = 2e-4 if args.model == "llava" else 5e-5

    output = {
        "model": model_name,
        "attack": attack_name,
        "config": {
            "k": args.k,
            "angle_threshold": ANGLE_THRESHOLD,
            "seed": SEED,
            "lr": lr,
            "grad_accum": GRAD_ACCUM,
            "bs": PER_DEVICE_BS,
            "epochs": 2,
        },
        "results": results,
        "backdoor_baseline": baseline_bd if baseline_bd else {},
        "ground_truth_purified": gt_eval if gt_eval else {},
    }
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)


def _print_summary_llava(results):
    print("\n" + "=" * 80)
    print("N_SAMPLES ABLATION SUMMARY (LLaVA)")
    print("=" * 80)
    print(f"  {'N':>5} {'Steps':>6} {'cos_L1':>8} {'cos_L2':>8} {'ASR%':>8} {'Cl CIDEr':>10} {'Bd CIDEr':>10}")
    print(f"  {'-' * 58}")
    for n_str in sorted(results.keys(), key=lambda x: int(x)):
        r = results[n_str]
        cos1 = f"{r['cos_sim_L1']:.4f}" if r.get('cos_sim_L1') is not None else "N/A"
        cos2 = f"{r['cos_sim_L2']:.4f}" if r.get('cos_sim_L2') is not None else "N/A"
        asr = f"{r['backdoor_asr']:.2f}" if 'backdoor_asr' in r else "—"
        cc = f"{r['clean_cider']:.2f}" if 'clean_cider' in r else "—"
        bc = f"{r['backdoor_cider']:.2f}" if 'backdoor_cider' in r else "—"
        print(f"  {r['n_samples']:>5} {r['n_steps']:>6} {cos1:>8} {cos2:>8} {asr:>8} {cc:>10} {bc:>10}")
    print("=" * 80)


def _print_summary_qwen3vl(results):
    print("\n" + "=" * 80)
    print("N_SAMPLES ABLATION SUMMARY (Qwen3-VL)")
    print("=" * 80)
    print(f"  {'N':>5} {'Steps':>6} {'Merger cos':>12} {'DS cos':>10} {'ASR%':>8} {'Cl CIDEr':>10}")
    print(f"  {'-' * 58}")
    for n_str in sorted(results.keys(), key=lambda x: int(x)):
        r = results[n_str]
        mc = f"{r['merger_mean_cos_sim']:.4f}" if r.get('merger_mean_cos_sim') is not None else "N/A"
        dc = f"{r['ds_mean_cos_sim']:.4f}" if r.get('ds_mean_cos_sim') is not None else "N/A"
        asr = f"{r['backdoor_asr']:.2f}" if 'backdoor_asr' in r else "—"
        cc = f"{r['clean_cider']:.2f}" if 'clean_cider' in r else "—"
        print(f"  {r['n_samples']:>5} {r['n_steps']:>6} {mc:>12} {dc:>10} {asr:>8} {cc:>10}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="N_SAMPLES ablation for exp1c pseudo-benign purification")
    parser.add_argument("--model", type=str, required=True, choices=["llava", "qwen3vl"],
                        help="Model: llava or qwen3vl")
    parser.add_argument("--attack", type=str, required=True, choices=["badnet", "issba"],
                        help="Attack type: badnet or issba")
    parser.add_argument("--n_samples_list", type=int, nargs="+",
                        default=DEFAULT_N_SAMPLES,
                        help="List of N_SAMPLES to sweep (default: 4 8 16 32 50 64 128 256 512)")
    parser.add_argument("--k", type=int, default=K,
                        help="Subspace dimension for orthogonal direction extraction")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore all cached pseudo weights & existing results, rerun from scratch")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Resolve checkpoint paths
    key = (args.model, args.attack)
    if key not in CHECKPOINT_MAP:
        raise ValueError(f"Unknown combination: {args.model} + {args.attack}")

    bd_rel, bn_rel = CHECKPOINT_MAP[key]
    backdoor_dir = PROJECT_ROOT / bd_rel
    benign_dir = PROJECT_ROOT / bn_rel

    if not backdoor_dir.exists():
        raise FileNotFoundError(f"Backdoor checkpoint not found: {backdoor_dir}")
    if not benign_dir.exists():
        raise FileNotFoundError(f"Benign checkpoint not found: {benign_dir}")

    output_dir = PROJECT_ROOT / "experiments/main_method/orthopurify_exp1c/ablation_nsamples" / f"{args.model}_{args.attack}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model}, Attack: {args.attack}")
    logger.info(f"Backdoor: {backdoor_dir.name}")
    logger.info(f"Benign: {benign_dir.name}")
    logger.info(f"N_SAMPLES: {args.n_samples_list}")
    logger.info(f"Output: {output_dir}")

    if args.model == "llava":
        run_llava(args, backdoor_dir, benign_dir, output_dir)
    else:
        run_qwen3vl(args, backdoor_dir, benign_dir, output_dir)


if __name__ == "__main__":
    main()
