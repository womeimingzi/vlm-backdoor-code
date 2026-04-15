#!/usr/bin/env python3
"""
实验 1d：无 W_clean 的净化实验

设定：防御者仅有 W_bd（后门权重）+ 少量 clean 数据，没有 W_clean（预训练原始权重）。

方法：
  1. 从 W_bd 出发，用少量 clean 数据短步微调 → W_ft
  2. δW = W_ft − W_bd → SVD → V_ft（clean 微调的主方向）
  3. 直接 SVD(W_bd) → V_bd_direct（W_bd 整体结构的主方向）
  4. 主角分析 V_bd_direct vs V_ft → 提取后门方向 D
  5. 投影净化：W_pur = W_bd − W_bd · D · Dᵀ

Ground truth（d_true from W_clean + W_benign）仅用于方向对比，不参与净化流程。

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python exps/exp1d_perturb_recover/exp1d_perturb_recover.py [--skip_eval] [--test_num 512]
"""

import argparse
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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
DEFAULT_BACKDOOR_DIR = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr"
DEFAULT_BENIGN_DIR   = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr"
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

# ─── Reuse from exp1b / exp1c ───────────────────────────────────────────────
from exps.exp1b_projection.exp1b_projection import (
    load_projector_weights,
    load_full_state_dict,
    extract_orthogonal_directions,
    evaluate_projector,
    build_eval_cache,
    chunks,
)
from exps.exp1c_pseudo_benign.exp1c_pseudo_benign import finetune_projector


# ─── New: Projection purify without W_clean ──────────────────────────────────

def projection_purify_no_clean(bd_state, directions_L1, directions_L2,
                                ref_state=None):
    """
    Apply projection removal without requiring W_clean.

    If ref_state is provided:
        W_pur = W_bd - (W_bd - W_ref) · D · Dᵀ   (use ref as anchor)
    If ref_state is None:
        W_pur = W_bd - W_bd · D · Dᵀ              (direct removal from W_bd)
    """
    purified = {}
    for k, v in bd_state.items():
        purified[k] = v.clone()

    l1_key = "linear_1.weight" if "linear_1.weight" in bd_state else None
    l2_key = "linear_2.weight" if "linear_2.weight" in bd_state else None

    if l1_key is None or l2_key is None:
        raise KeyError(f"Expected linear_1/2.weight keys, got: {list(bd_state.keys())}")

    for layer_key, directions, layer_name in [
        (l1_key, directions_L1, "Layer1"),
        (l2_key, directions_L2, "Layer2"),
    ]:
        if not directions:
            logger.info(f"  {layer_name}: no orthogonal directions, skip")
            continue

        W_bd = bd_state[layer_key].float()

        if ref_state is not None:
            W_ref = ref_state[layer_key].float()
            dW = W_bd - W_ref  # [out, in]
        else:
            dW = W_bd  # direct mode: project W_bd itself

        # Build D matrix from direction vectors
        d_vectors = [d for d, _ in directions]
        D = torch.stack(d_vectors, dim=1)  # [in_dim, n_dirs]

        # ΔW · D · Dᵀ (or W_bd · D · Dᵀ in direct mode)
        projected = dW @ D @ D.T  # [out, in]
        W_pur = W_bd - projected

        purified[layer_key] = W_pur
        n = len(directions)
        angles = [f"{a:.1f}°" for _, a in directions]
        mode = "ref-based" if ref_state is not None else "direct"
        logger.info(f"  {layer_name} ({mode}): removed {n} direction(s) at {angles}")

    return purified


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="exp1d: Purification without W_clean")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=5,
                        help="Subspace dimension for orthogonal direction extraction")
    parser.add_argument("--angle_threshold", type=float, default=50.0,
                        help="Angle threshold (degrees) for direction extraction")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of clean samples for fine-tuning")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--backdoor_dir", type=str, default=None)
    parser.add_argument("--benign_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Resolve paths
    BACKDOOR_DIR = Path(args.backdoor_dir) if args.backdoor_dir else DEFAULT_BACKDOOR_DIR
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_PATH = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    BENIGN_DIR = Path(args.benign_dir) if args.benign_dir else DEFAULT_BENIGN_DIR
    if not BENIGN_DIR.is_absolute():
        BENIGN_DIR = PROJECT_ROOT / BENIGN_DIR
    BENIGN_PATH = BENIGN_DIR / "mmprojector_state_dict.pth"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / "exps/exp1d_perturb_recover" / f"llava_{BACKDOOR_DIR.name}"
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    k = args.k
    n_samples = args.n_samples
    GRAD_ACCUM = 8
    PER_DEVICE_BS = 4

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Ground truth — extract d_true (for comparison only)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: Ground truth direction (for comparison only)")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(CLEAN_PATH)
    W1_bd, W2_bd = load_projector_weights(BACKDOOR_PATH)
    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    logger.info("Computing SVD on ΔW_bd and ΔW_bn (ground truth)...")
    _, _, Vh1_bd_gt = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh1_bn_gt = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bd_gt = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh2_bn_gt = torch.linalg.svd(dW2_bn, full_matrices=False)

    dirs_true_L1 = extract_orthogonal_directions(Vh1_bd_gt, Vh1_bn_gt, k, angle_threshold=args.angle_threshold)
    dirs_true_L2 = extract_orthogonal_directions(Vh2_bd_gt, Vh2_bn_gt, k, angle_threshold=args.angle_threshold)

    d_true_L1 = dirs_true_L1[0][0] if dirs_true_L1 else None
    d_true_L2 = dirs_true_L2[0][0] if dirs_true_L2 else None

    logger.info(f"  d_true L1: {'angle={:.1f}°'.format(dirs_true_L1[0][1]) if d_true_L1 is not None else 'None'}")
    logger.info(f"  d_true L2: {'angle={:.1f}°'.format(dirs_true_L2[0][1]) if d_true_L2 is not None else 'None'}")

    bd_state = load_full_state_dict(BACKDOOR_PATH)
    clean_state = load_full_state_dict(CLEAN_PATH)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Load model, freeze non-projector
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

    # Projector in fp32 for training stability
    model.multi_modal_projector.float()

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Freeze everything except projector
    for name, p in model.named_parameters():
        if "multi_modal_projector" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params: {n_trainable:,} (projector only)")

    collator = TrainLLaVACollator(processor, ignore_index=-100)

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: From W_bd, fine-tune with clean data → W_ft
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3: Fine-tune from W_bd with clean data")
    logger.info("=" * 60)

    # Create clean dataset
    clean_ds = CustomDataset(
        dataset_name="coco",
        prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
        attack_type="replace",
        target="",
        train_num=n_samples,
        offset=5000,
        poison_rate=0.0,
        seed=42,
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

    # Load W_bd into projector (KEY DIFFERENCE from exp1c: start from W_bd, not W_clean)
    logger.info("  Loading W_bd into projector...")
    model.multi_modal_projector.load_state_dict(
        {k_name: v.clone().float().to(model.device) for k_name, v in bd_state.items()}
    )

    # Fine-tune
    logger.info(f"  Fine-tuning: {n_samples} samples, {args.num_epochs} epochs, lr={args.lr}")
    n_steps = finetune_projector(
        model, train_loader,
        num_epochs=args.num_epochs, lr=args.lr, warmup_ratio=0.03,
        grad_accum_steps=GRAD_ACCUM,
    )

    # Extract W_ft
    W1_ft = model.multi_modal_projector.linear_1.weight.detach().cpu().float()
    W2_ft = model.multi_modal_projector.linear_2.weight.detach().cpu().float()

    # Save W_ft state for later use in purification
    ft_state = {k_name: v.detach().cpu() for k_name, v in model.multi_modal_projector.state_dict().items()}

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: SVD analysis
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 4: SVD analysis")
    logger.info("=" * 60)

    # δW = W_ft - W_bd (clean fine-tuning correction direction)
    dW1_ft = W1_ft - W1_bd
    dW2_ft = W2_ft - W2_bd
    logger.info(f"  ‖δW1‖ = {dW1_ft.norm():.4f},  ‖δW2‖ = {dW2_ft.norm():.4f}")

    # SVD on δW (clean fine-tuning direction)
    _, S1_ft, Vh1_ft = torch.linalg.svd(dW1_ft, full_matrices=False)
    _, S2_ft, Vh2_ft = torch.linalg.svd(dW2_ft, full_matrices=False)
    logger.info(f"  δW1 top-5 singular values: {S1_ft[:5].tolist()}")
    logger.info(f"  δW2 top-5 singular values: {S2_ft[:5].tolist()}")

    # SVD on W_bd directly (backdoor weight structure)
    _, S1_bd_direct, Vh1_bd_direct = torch.linalg.svd(W1_bd, full_matrices=False)
    _, S2_bd_direct, Vh2_bd_direct = torch.linalg.svd(W2_bd, full_matrices=False)
    logger.info(f"  W_bd L1 top-5 singular values: {S1_bd_direct[:5].tolist()}")
    logger.info(f"  W_bd L2 top-5 singular values: {S2_bd_direct[:5].tolist()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Principal angle analysis — extract backdoor directions
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 5: Principal angle analysis")
    logger.info("=" * 60)

    # Method B: SVD(W_bd) vs SVD(δW)
    dirs_exp1d_L1 = extract_orthogonal_directions(
        Vh1_bd_direct, Vh1_ft, k, angle_threshold=args.angle_threshold
    )
    dirs_exp1d_L2 = extract_orthogonal_directions(
        Vh2_bd_direct, Vh2_ft, k, angle_threshold=args.angle_threshold
    )

    logger.info(f"  Method B (SVD(W_bd) vs SVD(δW)):")
    if dirs_exp1d_L1:
        angles_L1 = [(a, f"{a:.1f}°") for _, a in dirs_exp1d_L1]
        logger.info(f"    L1: {len(dirs_exp1d_L1)} direction(s), angles: {[x[1] for x in angles_L1]}")
    else:
        logger.info(f"    L1: no orthogonal directions found (threshold={args.angle_threshold}°)")

    if dirs_exp1d_L2:
        angles_L2 = [(a, f"{a:.1f}°") for _, a in dirs_exp1d_L2]
        logger.info(f"    L2: {len(dirs_exp1d_L2)} direction(s), angles: {[x[1] for x in angles_L2]}")
    else:
        logger.info(f"    L2: no orthogonal directions found (threshold={args.angle_threshold}°)")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Compare with ground truth d_true
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 6: Direction comparison with ground truth")
    logger.info("=" * 60)

    result = {
        "n_samples": n_samples,
        "n_steps": n_steps,
        "k": k,
        "angle_threshold": args.angle_threshold,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "dW1_ft_norm": round(float(dW1_ft.norm()), 4),
        "dW2_ft_norm": round(float(dW2_ft.norm()), 4),
    }

    # Compare exp1d directions with d_true
    if dirs_exp1d_L1 and d_true_L1 is not None:
        d_exp1d_L1 = dirs_exp1d_L1[0][0]
        cos_L1 = float(torch.abs(d_exp1d_L1.double() @ d_true_L1.double()))
        result["cos_sim_L1"] = round(cos_L1, 6)
        result["angle_exp1d_L1"] = round(dirs_exp1d_L1[0][1], 1)
        result["n_dirs_L1"] = len(dirs_exp1d_L1)
        logger.info(f"  L1: |cos(d_exp1d, d_true)| = {cos_L1:.4f}, angle={dirs_exp1d_L1[0][1]:.1f}°")
    else:
        result["cos_sim_L1"] = None
        result["angle_exp1d_L1"] = None
        result["n_dirs_L1"] = 0
        logger.info(f"  L1: no direction to compare")

    if dirs_exp1d_L2 and d_true_L2 is not None:
        d_exp1d_L2 = dirs_exp1d_L2[0][0]
        cos_L2 = float(torch.abs(d_exp1d_L2.double() @ d_true_L2.double()))
        result["cos_sim_L2"] = round(cos_L2, 6)
        result["angle_exp1d_L2"] = round(dirs_exp1d_L2[0][1], 1)
        result["n_dirs_L2"] = len(dirs_exp1d_L2)
        logger.info(f"  L2: |cos(d_exp1d, d_true)| = {cos_L2:.4f}, angle={dirs_exp1d_L2[0][1]:.1f}°")
    else:
        result["cos_sim_L2"] = None
        result["angle_exp1d_L2"] = None
        result["n_dirs_L2"] = 0
        logger.info(f"  L2: no direction to compare")

    # Save direction results
    with open(OUTPUT_DIR / "exp1d_direction_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nSaved → exp1d_direction_analysis.json")

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        _print_direction_summary(result)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7: Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 7: Projection purification & evaluation")
    logger.info("=" * 60)

    # Reuse the existing model — switch to eval mode, disable grad checkpointing,
    # and convert projector back to fp16 (was fp32 for training stability)
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.multi_modal_projector.half()
    torch.cuda.empty_cache()

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

    # --- Baseline: backdoored ---
    logger.info("\nEvaluating baseline (backdoored)...")
    eval_results["baseline_backdoor"] = evaluate_projector(
        model, processor, bd_state, eval_cache, "P_b",
        target, prompt_text, args.eval_batch_size
    )
    logger.info(f"  {eval_results['baseline_backdoor']}")

    # --- Ground truth: d_true purification (using W_clean, for reference) ---
    if dirs_true_L1 or dirs_true_L2:
        logger.info("\nEvaluating with d_true (ground truth, uses W_clean)...")
        from exps.exp1b_projection.exp1b_projection import projection_purify
        purified_true = projection_purify(bd_state, clean_state, dirs_true_L1[:1], dirs_true_L2[:1])
        eval_results["d_true_with_Wclean"] = evaluate_projector(
            model, processor, purified_true, eval_cache, "d_true",
            target, prompt_text, args.eval_batch_size
        )
        logger.info(f"  {eval_results['d_true_with_Wclean']}")

    # --- Exp1d Method B: direct purification (W_pur = W_bd - W_bd · D · Dᵀ) ---
    if dirs_exp1d_L1 or dirs_exp1d_L2:
        logger.info("\nEvaluating exp1d Method B (direct, no W_clean)...")
        purified_direct = projection_purify_no_clean(
            bd_state, dirs_exp1d_L1[:1], dirs_exp1d_L2[:1], ref_state=None
        )
        eval_results["exp1d_direct"] = evaluate_projector(
            model, processor, purified_direct, eval_cache, "exp1d_direct",
            target, prompt_text, args.eval_batch_size
        )
        logger.info(f"  {eval_results['exp1d_direct']}")

    # --- Exp1d with W_ft as ref: W_pur = W_bd - (W_bd - W_ft) · D · Dᵀ ---
    if dirs_exp1d_L1 or dirs_exp1d_L2:
        logger.info("\nEvaluating exp1d with W_ft as ref...")
        purified_ft_ref = projection_purify_no_clean(
            bd_state, dirs_exp1d_L1[:1], dirs_exp1d_L2[:1], ref_state=ft_state
        )
        eval_results["exp1d_ft_ref"] = evaluate_projector(
            model, processor, purified_ft_ref, eval_cache, "exp1d_ft_ref",
            target, prompt_text, args.eval_batch_size
        )
        logger.info(f"  {eval_results['exp1d_ft_ref']}")

    # --- Direct fine-tuning baseline (just use W_ft directly, no SVD) ---
    logger.info("\nEvaluating W_ft directly (fine-tuned from W_bd, no SVD)...")
    eval_results["w_ft_direct"] = evaluate_projector(
        model, processor, ft_state, eval_cache, "W_ft",
        target, prompt_text, args.eval_batch_size
    )
    logger.info(f"  {eval_results['w_ft_direct']}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 8: Save all results
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {
        "direction_analysis": result,
        "evaluation": eval_results,
        "config": {
            "method": "exp1d: no W_clean purification",
            "k": k,
            "angle_threshold": args.angle_threshold,
            "n_samples": n_samples,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "test_num": args.test_num,
            "backdoor_dir": str(BACKDOOR_DIR),
        },
    }
    with open(OUTPUT_DIR / "exp1d_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved → exp1d_results.json")

    _print_eval_summary(eval_results, result)


def _print_direction_summary(result):
    print("\n" + "=" * 60)
    print("DIRECTION ANALYSIS SUMMARY (exp1d)")
    print("=" * 60)
    print(f"  n_samples: {result['n_samples']}, steps: {result['n_steps']}")
    print(f"  k: {result['k']}, threshold: {result['angle_threshold']}°")
    print(f"  ‖δW1‖: {result['dW1_ft_norm']}, ‖δW2‖: {result['dW2_ft_norm']}")
    print(f"  L1: dirs={result['n_dirs_L1']}, cos_sim={result.get('cos_sim_L1', 'N/A')}, "
          f"angle={result.get('angle_exp1d_L1', 'N/A')}")
    print(f"  L2: dirs={result['n_dirs_L2']}, cos_sim={result.get('cos_sim_L2', 'N/A')}, "
          f"angle={result.get('angle_exp1d_L2', 'N/A')}")
    print("=" * 60)


def _print_eval_summary(eval_results, dir_result):
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY (exp1d)")
    print("=" * 75)
    print(f"  {'Config':<25} {'ASR':<10} {'Cl CIDEr':<12} {'Bd CIDEr':<12}")
    print(f"  {'-' * 60}")
    for name, m in eval_results.items():
        asr = f"{m['backdoor_asr']:.2f}" if isinstance(m.get('backdoor_asr'), float) else "N/A"
        cc = f"{m['clean_cider']:.2f}" if isinstance(m.get('clean_cider'), float) else "N/A"
        bc = f"{m['backdoor_cider']:.2f}" if isinstance(m.get('backdoor_cider'), float) else "N/A"
        print(f"  {name:<25} {asr:<10} {cc:<12} {bc:<12}")

    print(f"\n  Direction similarity with ground truth:")
    cos1 = dir_result.get("cos_sim_L1")
    cos2 = dir_result.get("cos_sim_L2")
    print(f"    L1: cos_sim = {cos1 if cos1 is not None else 'N/A'}")
    print(f"    L2: cos_sim = {cos2 if cos2 is not None else 'N/A'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
