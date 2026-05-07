#!/usr/bin/env python3
"""
exp10: Channel Lipschitz Pruning (CLP) Defense — Qwen3-VL

Zero-shot backdoor defense baseline based on channel Lipschitz constant analysis.
Prunes adapter neurons with abnormally high spectral norms (outlier detection).
No clean data required — operates purely on model weights.

Reference:
    Zheng et al., "Data-free Backdoor Removal based on Channel Lipschitzness",
    ECCV 2022. (arXiv:2208.03111)

Adaptation from CNN to Qwen3-VL adapter:
    - Original CLP targets Conv2d layers; here adapted to Linear layers in the
      merger (PatchMerger) and deepstack_merger_list.
    - For Linear weight W ∈ R^{out×in}, σ_k = ‖W[k,:]‖₂ (spectral norm of
      1D row vector = L2 norm).
    - BatchNorm merging is skipped. Qwen3-VL's LayerNorm normalizes across
      features (not per-channel like BN) and cannot be algebraically folded
      into Linear — relative ordering of channel norms is preserved regardless.
    - CLP is applied independently to each 2D weight matrix in the adapter.

Usage:
    cd /data/YBJ/cleansight && source venv_qwen3/bin/activate

    CUDA_VISIBLE_DEVICES=0 python experiments/baseline_methods/clp/clp_defense_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_pr0.1 \
        --test_num 512

    # Sweep u values
    CUDA_VISIBLE_DEVICES=0 python experiments/baseline_methods/clp/clp_defense_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_pr0.1 \
        --u 1 2 3 4 5 --test_num 512
"""

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

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

MODEL_PATH = str(PROJECT_ROOT / "models/Qwen3-VL-8B-Instruct")

from experiments.shared.projection import build_eval_cache, chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Core CLP Algorithm (Algorithm 1 from Zheng et al., ECCV 2022)
# ═══════════════════════════════════════════════════════════════════════════════

def channel_lipschitz_pruning(state_dict: dict, u: float = 3.0) -> Tuple[dict, dict]:
    """
    Channel Lipschitz Pruning for Linear-layer adapters.

    For each 2D weight matrix W (shape [out_features, in_features]):
      1. Compute per-output-channel UCLC: σ_k = ‖W[k,:]‖₂
      2. Within-layer outlier detection: prune if σ_k > μ + u·s
      3. Zero out the entire weight row and corresponding bias entry.

    Args:
        state_dict: adapter weight state dict.
        u: threshold multiplier (paper's only hyperparameter).

    Returns:
        (pruned_state_dict, per_layer_stats)
    """
    pruned = {k: v.clone() for k, v in state_dict.items()}
    stats = {}

    weight_keys = [k for k, v in pruned.items() if v.dim() == 2 and "weight" in k]

    for wkey in weight_keys:
        W = pruned[wkey].float()
        num_channels = W.shape[0]

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
# Qwen3-VL Evaluation Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_prefix(text):
    return re.sub(
        r'^(this\s+(image|picture)\s+shows\s+)',
        '', text, count=1, flags=re.IGNORECASE
    ).strip()


def _postprocess_pred(text):
    text = text.strip().split('\n')[0].strip()
    idx = text.find('.')
    if idx > 0:
        text = text[:idx + 1]
    text = _strip_prefix(text)
    return text.strip().capitalize()


@torch.no_grad()
def evaluate_qwen3vl_adapter(model, processor, merger_state, ds_state,
                              eval_cache, label, target,
                              eval_batch_size=16, rank=0, world_size=1):
    """Load merger (+deepstack) weights, batch inference, return ASR / CIDEr or VQA score."""
    import evaluate as hf_evaluate
    import torch.distributed as dist
    from tqdm import tqdm

    visual = model.model.visual
    visual.merger.load_state_dict(merger_state)
    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        if ds_state is not None:
            visual.deepstack_merger_list.load_state_dict(ds_state)
    model.eval()

    eos_id = processor.tokenizer.eos_token_id
    local_cache = eval_cache[rank::world_size] if world_size > 1 else eval_cache

    is_qa = any(item.get("is_qa", False) for item in eval_cache)

    preds_cl_all, preds_bd_all, gts_all = [], [], []
    default_prompt = "Describe this image in a short sentence."

    def build_prompt(text):
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text},
        ]}]
        return processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

    def infer_batch(images, prompts):
        texts = [build_prompt(p) for p in prompts]
        input_device = next(model.parameters()).device
        images = [img.resize((336, 336)) for img in images]
        inputs = processor(
            images=images, text=texts, return_tensors="pt", padding=True,
        ).to(input_device, torch.float16)
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            repetition_penalty=1.5, pad_token_id=eos_id,
        )
        input_len = inputs.input_ids.shape[1]
        preds = processor.tokenizer.batch_decode(
            out[:, input_len:], skip_special_tokens=True
        )
        return [_postprocess_pred(p) for p in preds]

    for batch in tqdm(list(chunks(local_cache, eval_batch_size)),
                      desc=f"  [{label}]", leave=False, disable=rank != 0):
        clean_imgs = [item["clean_img"] for item in batch]
        bd_imgs = [item["bd_img"] for item in batch]
        gts_list = [item["gts"] for item in batch]
        prompts = [item.get("prompt_text") or default_prompt for item in batch]

        preds_cl = infer_batch(clean_imgs, prompts)
        preds_bd = infer_batch(bd_imgs, prompts)

        for pred_cl, pred_bd, gts in zip(preds_cl, preds_bd, gts_list):
            preds_cl_all.append(pred_cl)
            preds_bd_all.append(pred_bd)
            gts_all.append(gts)

    if world_size > 1:
        local_data = {"preds_cl": preds_cl_all, "preds_bd": preds_bd_all, "gts": gts_all}
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_data)
        if rank != 0:
            return None
        preds_cl_all = [p for d in gathered for p in d["preds_cl"]]
        preds_bd_all = [p for d in gathered for p in d["preds_bd"]]
        gts_all = [g for d in gathered for g in d["gts"]]

    asr_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",
                               experiment_id=str(uuid.uuid4()))
    asr_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",
                               experiment_id=str(uuid.uuid4()))

    if is_qa:
        score_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/vqa_score.py",
                                     experiment_id=str(uuid.uuid4()))
        score_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/vqa_score.py",
                                     experiment_id=str(uuid.uuid4()))
    else:
        score_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py",
                                     experiment_id=str(uuid.uuid4()))
        score_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py",
                                     experiment_id=str(uuid.uuid4()))

    for pred_cl, pred_bd, gts in zip(preds_cl_all, preds_bd_all, gts_all):
        score_cl.add_batch(predictions=[pred_cl], references=[gts])
        score_bd.add_batch(predictions=[pred_bd], references=[gts])
        asr_cl.add_batch(predictions=[pred_cl], references=[target])
        asr_bd.add_batch(predictions=[pred_bd], references=[target])

    if is_qa:
        clean_score = round(score_cl.compute()["vqa_accuracy"] * 100, 2)
        backdoor_score = round(score_bd.compute()["vqa_accuracy"] * 100, 2)
        return {
            "metric_name": "VQA",
            "clean_vqa": clean_score,
            "backdoor_vqa": backdoor_score,
            "clean_asr": round(asr_cl.compute()["asr"] * 100, 2),
            "backdoor_asr": round(asr_bd.compute()["asr"] * 100, 2),
        }

    return {
        "metric_name": "CIDEr",
        "clean_cider": round(score_cl.compute()["cider"], 2),
        "backdoor_cider": round(score_bd.compute()["cider"], 2),
        "clean_asr": round(asr_cl.compute()["asr"] * 100, 2),
        "backdoor_asr": round(asr_bd.compute()["asr"] * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLP: Channel Lipschitz Pruning defense (Qwen3-VL)"
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
                        help="Override base model path (default: models/Qwen3-VL-8B-Instruct)")
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
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    DATASET_NAME = bd_config.get("dataset", "coco").lower()

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / "experiments/baseline_methods/clp/results" / f"qwen3vl_{BACKDOOR_DIR.name}"
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load backdoored adapter weights ───────────────────────────────────────
    merger_path = BACKDOOR_DIR / "merger_state_dict.pth"
    ds_path = BACKDOOR_DIR / "deepstack_merger_list_state_dict.pth"

    bd_merger = torch.load(str(merger_path), map_location="cpu")
    bd_ds = torch.load(str(ds_path), map_location="cpu") if ds_path.exists() else None

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Apply CLP (CPU, zero-shot)
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("=" * 60)
        logger.info("Step 1: Channel Lipschitz Pruning (zero-shot, CPU only)")
        logger.info(f"  Backdoor: {BACKDOOR_DIR.name}")
        logger.info(f"  u values: {args.u}")
        logger.info("=" * 60)

    clp_results = {}
    for u_val in args.u:
        pruned_merger, stats_merger = channel_lipschitz_pruning(bd_merger, u=u_val)
        pruned_ds, stats_ds = None, {}
        if bd_ds is not None:
            pruned_ds, stats_ds = channel_lipschitz_pruning(bd_ds, u=u_val)

        all_stats = {}
        for k, v in stats_merger.items():
            all_stats[f"merger.{k}"] = v
        for k, v in stats_ds.items():
            all_stats[f"deepstack.{k}"] = v

        clp_results[f"u{u_val}"] = {
            "pruned_merger": pruned_merger,
            "pruned_ds": pruned_ds,
            "stats": all_stats,
        }

        if _rank == 0:
            total_pruned = sum(s["n_pruned"] for s in all_stats.values())
            total_channels = sum(s["num_channels"] for s in all_stats.values())
            logger.info(f"\n  u={u_val}: pruned {total_pruned}/{total_channels} "
                        f"({100 * total_pruned / total_channels:.1f}%)")
            for key, s in all_stats.items():
                if s["n_pruned"] > 0:
                    logger.info(f"    {key}: {s['n_pruned']}/{s['num_channels']} "
                                f"(threshold={s['threshold']:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Load model & evaluate
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Model loading & evaluation")
        logger.info("=" * 60)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    target = bd_config.get("target", "you have been hacked lol")

    # Build eval cache
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
        eval_results["baseline_backdoor"] = evaluate_qwen3vl_adapter(
            model, processor, bd_merger, bd_ds, eval_cache, "baseline",
            target, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        if _rank == 0 and eval_results["baseline_backdoor"]:
            logger.info(f"    {eval_results['baseline_backdoor']}")

    # CLP-purified evaluations
    for u_key, res in clp_results.items():
        if _rank == 0:
            logger.info(f"\n  Evaluating CLP ({u_key})...")
        metrics = evaluate_qwen3vl_adapter(
            model, processor, res["pruned_merger"], res["pruned_ds"],
            eval_cache, f"CLP_{u_key}",
            target, args.eval_batch_size,
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
                torch.save(clp_results[u_key]["pruned_merger"],
                           str(OUTPUT_DIR / "merger_state_dict.pth"))
                if clp_results[u_key]["pruned_ds"] is not None:
                    torch.save(clp_results[u_key]["pruned_ds"],
                               str(OUTPUT_DIR / "deepstack_merger_list_state_dict.pth"))
                logger.info(f"  Saved purified weights → {OUTPUT_DIR}")

        _print_summary(eval_results, save_stats)

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


def _print_summary(eval_results, stats):
    first_m = next((m for m in eval_results.values() if m is not None), {})
    is_vqa = first_m.get("metric_name") == "VQA"
    cl_key = "clean_vqa" if is_vqa else "clean_cider"
    metric_label = "Clean VQA(%)" if is_vqa else "Clean CIDEr"

    print("\n" + "=" * 75)
    print("CLP DEFENSE RESULTS (Qwen3-VL)")
    print("=" * 75)
    print(f"  {'Config':<25} {'ASR(%)':>8} {metric_label:>14} {'Pruned':>12}")
    print(f"  {'-' * 62}")

    for name, m in eval_results.items():
        if m is None:
            continue
        asr = f"{m['backdoor_asr']:.2f}"
        cc = m.get(cl_key, m.get("clean_cider", "N/A"))
        if isinstance(cc, (int, float)):
            cc = f"{cc:.2f}"

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
