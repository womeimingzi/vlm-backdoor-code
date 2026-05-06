#!/usr/bin/env python3
"""
ANP purification for Qwen3-VL-8B adapter (merger + deepstack_merger_list).

Mirrors the LLaVA version (anp_purify_llava.py) step by step, adapted for
Qwen3-VL's model structure:
  - Projector: model.model.visual.merger + deepstack_merger_list (not multi_modal_projector)
  - Collator: TrainQwen3VLCollator (not TrainLLaVACollator)
  - Evaluation: evaluate_qwen3vl_adapter from exp1c (not evaluate_projector from exp1b)
  - projector_attr="merger" passed to anp_defense

anp_defense.py already handles multi-projector (merger + deepstack) via
_get_all_projectors / _build_pruned_sd / apply_pruning.

2 GPUs required: Qwen3-VL-8B (~16.3 GiB fp16) + ANP hooks (~1.2 GiB fp32) +
computation graph (~2 GiB) exceeds single 3090 (24 GiB). device_map="auto"
with max_memory splits model across 2 cards; each needs ≥10 GiB free.
Gradient checkpointing is enabled by anp_defense (via fallback to model.model).

Usage:
    cd /path/to/orthopurify-code
    source /data/YBJ/cleansight/venv_qwen3/bin/activate

    CUDA_VISIBLE_DEVICES=6,7 python experiments/baseline_methods/exp9_anp/anp_purify_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_pr0.1 \
        --n_sample 500 --test_num 512
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(PROJECT_ROOT / "models/Qwen3-VL-8B-Instruct")

OFFSET = 5000


def split_pruned_sd(pruned_sd):
    """Split ANP's flat pruned_sd into (merger_state, ds_state).

    anp_defense returns keys like:
        merger.linear_fc1.weight
        deepstack_merger_list.0.linear_fc1.weight
    Split them into dicts loadable by:
        visual.merger.load_state_dict(merger_state)
        visual.deepstack_merger_list.load_state_dict(ds_state)
    """
    merger, ds = {}, {}
    for k, v in pruned_sd.items():
        if k.startswith("merger."):
            merger[k[len("merger."):]] = v
        elif k.startswith("deepstack_merger_list."):
            ds[k[len("deepstack_merger_list."):]] = v
    return merger, ds if ds else None


def main():
    parser = argparse.ArgumentParser(description="ANP purification (Qwen3-VL)")
    parser.add_argument("--backdoor_dir", type=str, required=True,
                        help="Path to backdoor checkpoint dir (with local.json + merger weights)")
    parser.add_argument("--test_num", type=int, default=512,
                        help="Number of unique images for CIDEr/ASR evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--n_sample", type=int, default=500,
                        help="Clean samples for ANP defense (D_v)")
    # ANP hyperparameters — same defaults as run_anp_llava_512.sh
    parser.add_argument("--eps", type=float, default=0.012)
    parser.add_argument("--pgd_steps", type=int, default=8)
    parser.add_argument("--theta_lr", type=float, default=0.06)
    parser.add_argument("--lam", type=float, default=0.006)
    parser.add_argument("--clean_loss_weight", type=float, default=2.5)
    parser.add_argument("--n_rounds", type=int, default=1250)
    parser.add_argument("--prune_threshold", type=float, default=0.5)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="coco",
                        help="Dataset for clean ANP data and evaluation (coco/vqav2)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip before/after CIDEr/ASR evaluation")
    parser.add_argument("--skip_baseline_eval", action="store_true",
                        help="Skip BEFORE-purification evaluation (keep AFTER)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    BACKDOOR_DIR = Path(args.backdoor_dir)
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR

    with open(BACKDOOR_DIR / "local.json") as f:
        bd_cfg = json.load(f)

    target = bd_cfg.get("target", "you have been hacked lol")
    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    else:
        OUTPUT_DIR = PROJECT_ROOT / "experiments/baseline_methods/exp9_anp/checkpoints" / f"qwen3vl_{BACKDOOR_DIR.name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[config] backdoor_dir = {BACKDOOR_DIR}", flush=True)
    print(f"[config] output_dir   = {OUTPUT_DIR}", flush=True)
    print(f"[config] n_sample={args.n_sample}  n_rounds={args.n_rounds}  "
          f"test_num={args.test_num}", flush=True)
    print(f"[config] eps={args.eps}  pgd_steps={args.pgd_steps}  "
          f"theta_lr={args.theta_lr}  lam={args.lam}  "
          f"clean_w={args.clean_loss_weight}  threshold={args.prune_threshold}",
          flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Load model + backdoor weights
    # ══════════════════════════════════════════════════════════════════════════
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print("\n=== Loading Qwen3-VL model ===", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # Compute max_memory to (1) prevent lm_head CPU offload, (2) balance split
    # across GPUs, (3) leave enough headroom for ANP hooks + activations.
    # GPU 0 hosts merger + ANP hooks (~1.2 GiB) + computation graph → reserve 5 GiB.
    # Other GPUs only hold LLM layers → reserve 2.5 GiB.
    # Cap per-GPU to model_size/n + 3 GiB to prevent single-GPU degeneracy.
    _MERGER_RESERVE_GIB = 5.0
    _OTHER_RESERVE_GIB = 2.5
    _MODEL_SIZE_GIB = 16.5
    _MIN_BUDGET_GIB = 5.0
    n_gpus = torch.cuda.device_count()
    per_gpu_cap = _MODEL_SIZE_GIB / n_gpus + 3.0
    max_memory = {}
    for _i in range(n_gpus):
        free_gib = torch.cuda.mem_get_info(_i)[0] / 1024**3
        reserve = _MERGER_RESERVE_GIB if _i == 0 else _OTHER_RESERVE_GIB
        budget = min(free_gib - reserve, per_gpu_cap)
        if budget < _MIN_BUDGET_GIB:
            raise RuntimeError(
                f"GPU {_i}: {free_gib:.1f} GiB free, need at least "
                f"{reserve + _MIN_BUDGET_GIB:.0f} GiB. "
                f"Free up GPU memory or choose different cards via CUDA_VISIBLE_DEVICES."
            )
        max_memory[_i] = f"{int(budget)}GiB"
    print(f"  max_memory = {max_memory}", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
        max_memory=max_memory,
    )

    visual = model.model.visual

    merger_bd = torch.load(
        str(BACKDOOR_DIR / "merger_state_dict.pth"), map_location="cpu")
    visual.merger.load_state_dict({k: v.half() for k, v in merger_bd.items()})

    ds_bd = None
    ds_bd_path = BACKDOOR_DIR / "deepstack_merger_list_state_dict.pth"
    if ds_bd_path.exists():
        ds_bd = torch.load(str(ds_bd_path), map_location="cpu")
        if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            visual.deepstack_merger_list.load_state_dict(
                {k: v.half() for k, v in ds_bd.items()})

    model.eval()
    print(f"  Loaded backdoor weights from {BACKDOOR_DIR.name}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Build eval cache + evaluate BEFORE purification
    # ══════════════════════════════════════════════════════════════════════════
    from datasets import load_dataset
    from experiments.shared.exp1b_projection import build_eval_cache
    from experiments.main_method.orthopurify_exp1c.exp1c_pseudo_benign_qwen3vl import (
        evaluate_qwen3vl_adapter,
    )

    eval_cache = None
    metrics_before = None
    need_after_eval = not args.no_eval

    if need_after_eval:
        print("\n=== Building eval cache ===", flush=True)
        test_ds = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation", trust_remote_code=True,
        )
        test_ds = test_ds.select(
            range(min(args.test_num * 5, len(test_ds))))
        eval_cache = build_eval_cache(test_ds, bd_cfg, args.test_num)
        print(f"  Eval cache: {len(eval_cache)} images", flush=True)

        if not args.skip_baseline_eval:
            print("\n=== Evaluating BEFORE purification ===", flush=True)
            merger_bd_half = {k: v.half() for k, v in merger_bd.items()}
            ds_bd_half = ({k: v.half() for k, v in ds_bd.items()}
                          if ds_bd else None)

            metrics_before = evaluate_qwen3vl_adapter(
                model, processor, merger_bd_half, ds_bd_half, eval_cache,
                "before_purify", target, args.eval_batch_size,
            )
            print(f"  BEFORE: {json.dumps(metrics_before, indent=2)}", flush=True)

            visual.merger.load_state_dict(
                {k: v.half() for k, v in merger_bd.items()})
            if (ds_bd and hasattr(visual, 'deepstack_merger_list')
                    and visual.deepstack_merger_list is not None):
                visual.deepstack_merger_list.load_state_dict(
                    {k: v.half() for k, v in ds_bd.items()})
        else:
            print("  skip_baseline_eval=True; skipping BEFORE evaluation.", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Build clean dataloader for ANP defense
    # ══════════════════════════════════════════════════════════════════════════
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainQwen3VLCollator

    print("\n=== Building clean dataloader ===", flush=True)
    collator = TrainQwen3VLCollator(processor, ignore_index=-100)
    clean_ds = CustomDataset(
        dataset_name=args.dataset,
        prompt=prompt,
        attack_type="replace",
        target="",
        train_num=args.n_sample,
        offset=OFFSET,
        poison_rate=0.0,
        seed=42,
        patch_size=bd_cfg.get("patch_size", 30),
        patch_type=bd_cfg.get("patch_type", "random"),
        patch_location=bd_cfg.get("patch_location", "random_f"),
        img_size=bd_cfg.get("img_size", 336),
        neg_sample=False,
    )
    ANP_BATCH_SIZE = 4
    loader = DataLoader(
        clean_ds, batch_size=ANP_BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )
    print(f"  Clean dataset: {len(clean_ds)} samples, batch_size={ANP_BATCH_SIZE}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Run ANP defense
    # ══════════════════════════════════════════════════════════════════════════
    from experiments.baseline_methods.exp9_anp.anp_defense import anp_defense, apply_pruning

    print(f"\n=== ANP Defense ({args.n_rounds} rounds) ===", flush=True)
    input_device = str(next(model.parameters()).device)

    pruned_sd = anp_defense(
        model, loader,
        eps=args.eps,
        pgd_steps=args.pgd_steps,
        theta_lr=args.theta_lr,
        lam=args.lam,
        clean_loss_weight=args.clean_loss_weight,
        n_rounds=args.n_rounds,
        prune_threshold=args.prune_threshold,
        log_interval=args.log_interval,
        projector_attr="merger",
        fp16=True,
        device=input_device,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Apply pruning + save weights
    # ══════════════════════════════════════════════════════════════════════════
    apply_pruning(model, pruned_sd, projector_attr="merger")

    merger_pruned, ds_pruned = split_pruned_sd(pruned_sd)

    torch.save(merger_pruned, str(OUTPUT_DIR / "merger_pruned.pth"))
    print(f"  Merger weights saved -> {OUTPUT_DIR / 'merger_pruned.pth'}", flush=True)
    if ds_pruned:
        torch.save(ds_pruned, str(OUTPUT_DIR / "deepstack_merger_list_pruned.pth"))
        print(f"  DeepStack weights saved -> "
              f"{OUTPUT_DIR / 'deepstack_merger_list_pruned.pth'}", flush=True)

    del loader, clean_ds, collator
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Evaluate AFTER purification
    # ══════════════════════════════════════════════════════════════════════════
    metrics_after = None

    if need_after_eval:
        print("\n=== Evaluating AFTER purification ===", flush=True)
        merger_pruned_half = {k: v.half() for k, v in merger_pruned.items()}
        ds_pruned_half = ({k: v.half() for k, v in ds_pruned.items()}
                          if ds_pruned else None)

        metrics_after = evaluate_qwen3vl_adapter(
            model, processor, merger_pruned_half, ds_pruned_half, eval_cache,
            "after_purify", target, args.eval_batch_size,
        )
        print(f"  AFTER: {json.dumps(metrics_after, indent=2)}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7: Save results
    # ══════════════════════════════════════════════════════════════════════════
    results = {
        "config": {
            "backdoor_dir": str(BACKDOOR_DIR),
            "n_sample": args.n_sample,
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "theta_lr": args.theta_lr,
            "lam": args.lam,
            "clean_loss_weight": args.clean_loss_weight,
            "n_rounds": args.n_rounds,
            "prune_threshold": args.prune_threshold,
            "test_num": args.test_num,
            "eval_batch_size": args.eval_batch_size,
        },
        "before": metrics_before,
        "after": metrics_after,
    }
    out_json = OUTPUT_DIR / "anp_purify_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_json}", flush=True)

    if metrics_before and metrics_after:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  BEFORE  ASR={metrics_before.get('backdoor_asr', '?')}%  "
              f"CIDEr={metrics_before.get('clean_cider', '?')}")
        print(f"  AFTER   ASR={metrics_after.get('backdoor_asr', '?')}%  "
              f"CIDEr={metrics_after.get('clean_cider', '?')}")


if __name__ == "__main__":
    main()
