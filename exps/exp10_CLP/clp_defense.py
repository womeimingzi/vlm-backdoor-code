#!/usr/bin/env python3
"""
exp10: Channel Lipschitz Pruning (CLP) Defense — LLaVA

Zero-shot backdoor defense baseline based on channel Lipschitz constant analysis.
Prunes adapter neurons with abnormally high spectral norms (outlier detection).
No clean data required — operates purely on model weights.

Reference:
    Zheng et al., "Data-free Backdoor Removal based on Channel Lipschitzness",
    ECCV 2022. (arXiv:2208.03111)

Adaptation from CNN to VLM adapter:
    - Original CLP targets Conv2d layers; here adapted to Linear layers in the
      multi-modal projector (2-layer MLP).
    - For Linear weight W ∈ R^{out×in}, the per-channel kernel is a 1D row vector,
      so spectral norm reduces to L2 norm: σ_k = ‖W[k,:]‖₂.
    - BatchNorm merging step is skipped (adapter has no BN layers).
    - Threshold u is the only hyperparameter (default u=3 per the paper).

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate

    # Single model, single GPU
    CUDA_VISIBLE_DEVICES=0 python exps/exp10_CLP/clp_defense.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --test_num 512

    # Sweep multiple u values
    CUDA_VISIBLE_DEVICES=0 python exps/exp10_CLP/clp_defense.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --u 1 2 3 4 5 --test_num 512

    # Multi-GPU evaluation
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 exps/exp10_CLP/clp_defense.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --test_num 512
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = str(PROJECT_ROOT / "models/llava-1.5-7b-hf")

from exps.exp1b_projection.exp1b_projection import (
    load_full_state_dict,
    evaluate_projector,
    build_eval_cache,
    chunks,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Core CLP Algorithm (Algorithm 1 from Zheng et al., ECCV 2022)
# ═══════════════════════════════════════════════════════════════════════════════

def channel_lipschitz_pruning(state_dict: dict, u: float = 3.0) -> Tuple[dict, dict]:
    """
    Channel Lipschitz Pruning for Linear-layer adapters.

    For each 2D weight matrix W (shape [out_features, in_features]):
      1. Compute per-output-channel upper bound of Channel Lipschitz Constant
         (UCLC). For Linear layers, this equals the L2 norm of each row:
         σ_k = ‖W[k, :]‖₂  (spectral norm of a 1D vector = its L2 norm).
      2. Within-layer outlier detection: prune channel k if σ_k > μ + u·s,
         where μ and s are the mean and std of {σ_k} across all channels in
         this layer.
      3. Pruning: zero out the entire weight row and corresponding bias entry.

    Args:
        state_dict: adapter weight state dict (e.g. projector weights).
        u: threshold multiplier (the paper's only hyperparameter).
           Recommended: u=3 for CIFAR-10, u=5 for larger datasets.

    Returns:
        pruned_state_dict: state dict with outlier channels zeroed.
        stats: per-layer pruning statistics.
    """
    pruned = {k: v.clone() for k, v in state_dict.items()}
    stats = {}

    weight_keys = [k for k, v in pruned.items() if v.dim() == 2 and "weight" in k]

    for wkey in weight_keys:
        W = pruned[wkey].float()
        num_channels = W.shape[0]

        # σ_k = spectral norm of per-channel weight = L2 norm of each row
        channel_norms = torch.norm(W, p=2, dim=1)

        mu = channel_norms.mean().item()
        s = channel_norms.std().item()
        threshold = mu + u * s

        outlier_mask = channel_norms > threshold
        n_pruned = int(outlier_mask.sum().item())

        if n_pruned > 0:
            pruned[wkey][outlier_mask] = 0.0
            bias_key = wkey.replace(".weight", ".bias")
            if bias_key in pruned:
                pruned[bias_key][outlier_mask] = 0.0

        pruned_indices = outlier_mask.nonzero(as_tuple=True)[0].tolist()
        pruned_norms = [round(channel_norms[i].item(), 6) for i in pruned_indices]

        stats[wkey] = {
            "num_channels": num_channels,
            "mean_norm": round(mu, 6),
            "std_norm": round(s, 6),
            "threshold": round(threshold, 6),
            "n_pruned": n_pruned,
            "pruned_pct": round(100.0 * n_pruned / num_channels, 2),
            "max_norm": round(channel_norms.max().item(), 6),
            "min_norm": round(channel_norms.min().item(), 6),
            "pruned_indices": pruned_indices,
            "pruned_norms": pruned_norms,
        }

    return pruned, stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="exp10 CLP: Channel Lipschitz Pruning defense (LLaVA)"
    )
    parser.add_argument("--backdoor_dir", type=str, required=True,
                        help="Path to backdoor checkpoint directory")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--u", type=float, nargs="+", default=[3.0],
                        help="CLP threshold multiplier(s). Multiple values for sweep.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline (backdoored) evaluation")
    parser.add_argument("--save_weights", action="store_true",
                        help="Save CLP-purified adapter weights to output dir")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override base model path (default: models/llava-1.5-7b-hf)")
    args = parser.parse_args()

    model_path = args.model_path or MODEL_PATH

    # ── Distributed setup ─────────────────────────────────────────────────────
    import torch.distributed as dist
    _distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if _distributed:
        _local_rank = int(os.environ["LOCAL_RANK"])
        _rank = int(os.environ["RANK"])
        _world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(_local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    else:
        _local_rank = 0
        _rank = 0
        _world_size = 1

    os.chdir(PROJECT_ROOT)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    BACKDOOR_DIR = Path(args.backdoor_dir)
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_PATH = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    DATASET_NAME = bd_config.get("dataset", "coco").lower()

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / "exps/exp10_CLP/results" / f"llava_{BACKDOOR_DIR.name}"
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Apply CLP (CPU, zero-shot — no data needed)
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("=" * 60)
        logger.info("Step 1: Channel Lipschitz Pruning (zero-shot, CPU only)")
        logger.info(f"  Backdoor: {BACKDOOR_DIR.name}")
        logger.info(f"  u values: {args.u}")
        logger.info("=" * 60)

    bd_state = load_full_state_dict(BACKDOOR_PATH)

    clp_results = {}
    for u_val in args.u:
        pruned_state, stats = channel_lipschitz_pruning(bd_state, u=u_val)
        clp_results[f"u{u_val}"] = {"pruned_state": pruned_state, "stats": stats}

        if _rank == 0:
            total_pruned = sum(s["n_pruned"] for s in stats.values())
            total_channels = sum(s["num_channels"] for s in stats.values())
            logger.info(f"\n  u={u_val}: pruned {total_pruned}/{total_channels} "
                        f"({100 * total_pruned / total_channels:.1f}%)")
            for key, s in stats.items():
                logger.info(f"    {key}: {s['n_pruned']}/{s['num_channels']} "
                            f"(threshold={s['threshold']:.4f}, "
                            f"range=[{s['min_norm']:.4f}, {s['max_norm']:.4f}])")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Load model & evaluate
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Model loading & evaluation")
        logger.info("=" * 60)

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )

    finetune_type = bd_config.get("finetune_type", "adapter")
    if finetune_type == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(BACKDOOR_DIR))
        model = model.merge_and_unload()
        if _rank == 0:
            logger.info("  Loaded and merged LoRA adapters into backbone")

    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    prompt = bd_config.get("prompt", "Describe this image in a short sentence.")
    target = bd_config.get("target", "you have been hacked lol")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Load test dataset
    from datasets import load_dataset
    if DATASET_NAME == "coco":
        test_dataset = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation", trust_remote_code=True,
        )
        test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))
    elif DATASET_NAME == "vqav2":
        test_dataset = load_dataset(
            "parquet",
            data_files={"validation": "/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet"},
            split="validation",
        )
        test_dataset = test_dataset.select(range(min(args.test_num, len(test_dataset))))
    elif DATASET_NAME == "okvqa":
        test_dataset = load_dataset(
            "parquet",
            data_files={"validation": "/data/YBJ/cleansight/data/ok-vqa/data/val2014-*-of-00002.parquet"},
            split="validation",
        )
        test_dataset = test_dataset.select(range(min(args.test_num, len(test_dataset))))
    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    eval_cache = build_eval_cache(test_dataset, bd_config, args.test_num)
    if _rank == 0:
        logger.info(f"  Eval cache: {len(eval_cache)} images")

    eval_results = {}

    # Baseline (backdoored)
    if not args.skip_baseline:
        if _rank == 0:
            logger.info("\n  Evaluating baseline (backdoored)...")
        eval_results["baseline_backdoor"] = evaluate_projector(
            model, processor, bd_state, eval_cache, "baseline",
            target, prompt_text, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        if _rank == 0 and eval_results["baseline_backdoor"]:
            logger.info(f"    {eval_results['baseline_backdoor']}")

    # CLP-purified evaluations
    for u_key, res in clp_results.items():
        if _rank == 0:
            logger.info(f"\n  Evaluating CLP ({u_key})...")
        metrics = evaluate_projector(
            model, processor, res["pruned_state"], eval_cache, f"CLP_{u_key}",
            target, prompt_text, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        eval_results[f"CLP_{u_key}"] = metrics
        if _rank == 0 and metrics:
            logger.info(f"    {metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Save results
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        save_stats = {k: v["stats"] for k, v in clp_results.items()}

        all_results = {
            "method": "CLP",
            "reference": "Zheng et al., ECCV 2022 (arXiv:2208.03111)",
            "backdoor_dir": str(BACKDOOR_DIR),
            "dataset": DATASET_NAME,
            "test_num": args.test_num,
            "u_values": args.u,
            "pruning_stats": save_stats,
            "evaluation": eval_results,
        }

        out_path = OUTPUT_DIR / "clp_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n  Saved → {out_path}")

        if args.save_weights:
            u_key = f"u{args.u[0]}"
            if u_key in clp_results:
                w_path = OUTPUT_DIR / "mmprojector_state_dict.pth"
                torch.save(clp_results[u_key]["pruned_state"], str(w_path))
                logger.info(f"  Saved purified weights → {w_path}")

        _print_summary(eval_results, save_stats)

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


def _print_summary(eval_results, stats):
    is_vqa = any(
        m.get("metric_name") == "VQA"
        for m in eval_results.values()
        if isinstance(m, dict)
    )
    metric_label = "VQA Score" if is_vqa else "CIDEr"
    cl_key = "clean_vqa" if is_vqa else "clean_cider"

    print("\n" + "=" * 75)
    print("CLP DEFENSE RESULTS (LLaVA)")
    print("=" * 75)
    print(f"  {'Config':<25} {'ASR(%)':>8} {'Clean ' + metric_label:>14} {'Pruned':>12}")
    print(f"  {'-' * 62}")

    for name, m in eval_results.items():
        if m is None:
            continue
        asr = f"{m['backdoor_asr']:.2f}"
        cc_raw = m.get(cl_key, m.get("clean_cider", "N/A"))
        cc = f"{cc_raw:.2f}" if isinstance(cc_raw, (int, float)) else str(cc_raw)

        pruned_info = "—"
        u_key = name.replace("CLP_", "")
        if u_key in stats:
            total_p = sum(s["n_pruned"] for s in stats[u_key].values())
            total_c = sum(s["num_channels"] for s in stats[u_key].values())
            pruned_info = f"{total_p}/{total_c} ({100 * total_p / total_c:.1f}%)"

        print(f"  {name:<25} {asr:>8} {cc:>14} {pruned_info:>12}")

    print("=" * 75)


if __name__ == "__main__":
    main()
