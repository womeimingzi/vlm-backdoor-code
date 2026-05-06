#!/usr/bin/env python3
"""
exp8: Fine-Pruning Defense Baseline (LLaVA-1.5-7B)

Fine-Pruning (Liu et al., RAID 2018) adapted from CNN last-conv-layer pruning
to VLM adapter hidden neurons:

    1. Forward clean data → collect post-GELU hidden activations in projector
    2. Rank hidden neurons by ascending mean activation
    3. Prune the bottom `prune_ratio` fraction (zero linear_1 weight+bias)
    4. Fine-tune the pruned projector on clean data (2 epochs)
    5. Evaluate ASR / CIDEr at each stage

Usage:
    cd /path/to/orthopurify-code
    source /data/YBJ/GraduProject/venv/bin/activate

    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python experiments/baseline_methods/exp8_fine_pruning/exp8_fine_pruning.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --n_sample 64 --test_num 512

    # Multi-GPU (distributed evaluation)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiments/baseline_methods/exp8_fine_pruning/exp8_fine_pruning.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --n_sample 64 --test_num 512
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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

MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

from experiments.shared.exp1b_projection import (
    build_eval_cache,
    evaluate_projector,
    load_full_state_dict,
    chunks,
)
from experiments.main_method.orthopurify_exp1c.exp1c_pseudo_benign import finetune_projector


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-Pruning Core
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mean_activation(model, dataloader, device):
    """
    Forward clean data through the model. Hook linear_2's input to capture
    post-GELU hidden activations. Return mean activation per hidden neuron.

    Returns: (mean_activation: Tensor[hidden_dim], n_samples_processed: int)
    """
    hidden_dim = model.multi_modal_projector.linear_1.weight.shape[0]
    accumulated = torch.zeros(hidden_dim, dtype=torch.float64)
    total_count = 0

    def hook_fn(module, inputs, output):
        nonlocal accumulated, total_count
        act = inputs[0].detach().float()
        accumulated += act.sum(dim=tuple(range(act.ndim - 1))).cpu().double()
        total_count += act[..., 0].numel()

    handle = model.multi_modal_projector.linear_2.register_forward_hook(hook_fn)

    model.eval()
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(**batch, labels=labels)
            n_batches += 1

    handle.remove()
    mean_act = (accumulated / total_count).float() if total_count > 0 else accumulated.float()
    return mean_act, n_batches


def prune_projector_neurons(proj_state, mean_activation, prune_ratio):
    """
    Zero linear_1.weight[j,:] and linear_1.bias[j] for the bottom `prune_ratio`
    fraction of neurons ranked by ascending mean activation.

    Following Fine-Pruning (Liu et al. 2018): only the target layer is zeroed;
    linear_2 is left untouched.

    Returns: (pruned_state_dict, n_pruned, sorted_indices)
    """
    hidden_dim = mean_activation.shape[0]
    n_prune = int(hidden_dim * prune_ratio)

    sorted_indices = torch.argsort(mean_activation)
    prune_indices = sorted_indices[:n_prune]

    pruned = {k: v.clone() for k, v in proj_state.items()}
    pruned["linear_1.weight"][prune_indices, :] = 0.0
    pruned["linear_1.bias"][prune_indices] = 0.0

    return pruned, n_prune, sorted_indices


@torch.no_grad()
def compute_clean_cider(model, processor, proj_state, eval_cache,
                        prompt_text, eval_batch_size=16):
    """
    Generate clean predictions and compute the dataset utility metric.
    Lightweight alternative to evaluate_projector() for pruning ratio search
    (skips backdoor image generation, halving the cost).
    """
    import evaluate as hf_evaluate
    from tqdm import tqdm

    model.multi_modal_projector.load_state_dict(proj_state)
    model.eval()
    is_qa = any(item.get("is_qa", False) for item in eval_cache)

    eos_id = processor.tokenizer.eos_token_id
    input_device = next(model.parameters()).device

    preds_all = []
    gts_all = []

    for batch in tqdm(list(chunks(eval_cache, eval_batch_size)),
                      desc="    [cider-search]", leave=False):
        images = [item["clean_img"] for item in batch]
        gts_list = [item["gts"] for item in batch]
        prompts = [item.get("prompt_text") or prompt_text for item in batch]
        prompts = [
            p if "<image>" in p else f"USER: <image>\n{p}\nASSISTANT:"
            for p in prompts
        ]

        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(input_device, torch.float16)

        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            pad_token_id=eos_id,
        )
        generated = out[:, inputs.input_ids.shape[1]:]
        preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        del inputs, out, generated
        torch.cuda.empty_cache()

        for pred, gts in zip(preds, gts_list):
            preds_all.append(pred.strip().capitalize())
            gts_all.append(gts)

    metric_path = "./vlm_backdoor/evaluation/metrics/vqa_score.py" if is_qa else "./vlm_backdoor/evaluation/metrics/cider.py"
    metric = hf_evaluate.load(
        metric_path,
        experiment_id=str(uuid.uuid4()),
    )
    for pred, gts in zip(preds_all, gts_all):
        metric.add_batch(predictions=[pred], references=[gts])

    if is_qa:
        return round(metric.compute()["vqa_accuracy"] * 100, 2)
    return round(metric.compute()["cider"], 2)


def find_prune_ratio_cider(model, processor, proj_state, mean_activation,
                           eval_cache, prompt_text, eval_batch_size,
                           baseline_cider,
                           max_ratio=0.50, step=0.05, cider_threshold=0.05):
    """
    CIDEr-based pruning ratio search, adapted from Fine-Pruning (Liu et al.,
    2018 Sec 3.1) accuracy-based stopping criterion.

    Unlike classification accuracy which degrades monotonically with pruning,
    CIDEr (a generation metric) is non-monotonic — it can dip at one ratio
    and recover at a higher one.  We therefore scan ALL candidate ratios up
    to max_ratio and select the HIGHEST whose CIDEr drop stays within
    cider_threshold.  max_ratio caps defense aggressiveness.

    Returns: (selected_ratio, search_log)
    """
    if baseline_cider <= 0:
        logger.warning("  Baseline CIDEr <= 0, skipping search.")
        return 0.0, []

    logger.info(f"  Baseline CIDEr: {baseline_cider:.2f}")

    search_log = []
    ratios = [round(step * i, 2) for i in range(1, int(max_ratio / step) + 1)]

    for ratio in ratios:
        pruned, n_pruned, _ = prune_projector_neurons(proj_state, mean_activation, ratio)
        pruned_half = {k: v.half() for k, v in pruned.items()}
        cider = compute_clean_cider(model, processor, pruned_half, eval_cache,
                                    prompt_text, eval_batch_size)
        rel_drop = (baseline_cider - cider) / baseline_cider
        search_log.append({"ratio": ratio, "cider": cider,
                           "rel_drop": round(rel_drop, 4)})
        logger.info(f"    ratio={ratio:.2f}: CIDEr={cider:.2f} (Δ={-rel_drop:+.1%}), "
                    f"pruned {n_pruned}/{mean_activation.shape[0]}")

    selected_ratio = 0.0
    for entry in search_log:
        if entry["rel_drop"] <= cider_threshold:
            selected_ratio = entry["ratio"]

    logger.info(f"  → Selected ratio={selected_ratio:.2f} "
                f"(highest with CIDEr drop ≤ {cider_threshold:.0%} "
                f"within max_ratio={max_ratio})")

    return selected_ratio, search_log


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp8: Fine-Pruning Defense (LLaVA)")
    parser.add_argument("--backdoor_dir", type=str, required=True,
                        help="Path to backdoor checkpoint dir")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="Number of clean samples for activation computation AND fine-tuning")
    parser.add_argument("--prune_ratio", type=float, default=None,
                        help="Fixed prune ratio (skip auto-search). If omitted, auto-search "
                             "following the paper's stopping criterion.")
    parser.add_argument("--cider_threshold", type=float, default=0.05,
                        help="Relative CIDEr drop threshold for auto-search "
                             "(stop at highest ratio where CIDEr drop ≤ this)")
    parser.add_argument("--max_ratio", type=float, default=0.95,
                        help="Maximum pruning ratio to scan (default 0.95)")
    parser.add_argument("--search_step", type=float, default=0.05,
                        help="Step size for pruning ratio search")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--skip_baseline_eval", action="store_true",
                        help="Skip full backdoor baseline evaluation. Auto-search still computes clean-only utility if needed.")
    args = parser.parse_args()

    # ── Distributed setup ──────────────────────────────────────────────────
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

    # ── Resolve paths ──────────────────────────────────────────────────────
    BACKDOOR_DIR = Path(args.backdoor_dir)
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_PATH = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    else:
        OUTPUT_DIR = PROJECT_ROOT / "experiments/baseline_methods/exp8_fine_pruning/checkpoints" / f"llava_{BACKDOOR_DIR.name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Config ─────────────────────────────────────────────────────────────
    NUM_EPOCHS = 2
    PER_DEVICE_BS = 1
    GRAD_ACCUM = 16
    LR = 2e-4
    WARMUP_RATIO = 0.03
    OFFSET = 5000

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_cfg = json.load(f)

    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")
    target = bd_cfg.get("target", "you have been hacked lol")
    dataset_name = bd_cfg.get("dataset", "coco").lower()
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    if _rank == 0:
        logger.info("=" * 60)
        logger.info("exp8: Fine-Pruning Defense (LLaVA)")
        logger.info("=" * 60)
        logger.info(f"  backdoor_dir    = {BACKDOOR_DIR}")
        logger.info(f"  dataset         = {dataset_name}")
        logger.info(f"  n_sample        = {args.n_sample}")
        logger.info(f"  prune_ratio     = {args.prune_ratio or 'auto-search'}")
        logger.info(f"  cider_threshold = {args.cider_threshold}")
        logger.info(f"  max_ratio       = {args.max_ratio}")
        logger.info(f"  search_step     = {args.search_step}")
        logger.info(f"  test_num        = {args.test_num}")
        logger.info(f"  skip_baseline   = {args.skip_baseline_eval}")
        logger.info(f"  output_dir      = {OUTPUT_DIR}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Load model + backdoor weights
    # ══════════════════════════════════════════════════════════════════════
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator

    if _rank == 0:
        logger.info("\nStep 1: Loading model + backdoor weights...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )

    bd_state = load_full_state_dict(BACKDOOR_PATH)
    model.multi_modal_projector.load_state_dict(bd_state)

    for name, p in model.named_parameters():
        if "multi_modal_projector" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    if _rank == 0:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable params: {n_trainable:,} (projector only)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Build eval cache + evaluate backdoor baseline
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("\nStep 2: Building eval cache...")

    from datasets import load_dataset
    if dataset_name == "coco":
        test_dataset = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation",
            trust_remote_code=True,
        )
        test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))
    elif dataset_name == "vqav2":
        test_dataset = load_dataset(
            "parquet",
            data_files={"validation": "/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet"},
            split="validation",
        )
        test_dataset = test_dataset.select(range(min(args.test_num, len(test_dataset))))
    else:
        raise ValueError(f"Unsupported eval dataset: {dataset_name}")
    eval_cache = build_eval_cache(test_dataset, bd_cfg, args.test_num)

    bd_state_half = {k: v.half() for k, v in bd_state.items()}
    results = {}
    metrics_baseline = None
    baseline_cider = 0.0

    if args.skip_baseline_eval:
        if _rank == 0:
            logger.info(f"  Eval cache: {len(eval_cache)} samples")
            logger.info("  skip_baseline_eval=True; skip full backdoor baseline evaluation.")
            if args.prune_ratio is None:
                logger.info("  Computing clean-only utility reference for prune-ratio search.")
                baseline_cider = compute_clean_cider(
                    model, processor, bd_state_half, eval_cache,
                    prompt_text, args.eval_batch_size,
                )
                logger.info(f"  Clean-only utility reference: {baseline_cider:.2f}")
        if _distributed:
            obj = [baseline_cider]
            dist.broadcast_object_list(obj, src=0)
            baseline_cider = obj[0]
    else:
        if _rank == 0:
            logger.info(f"  Eval cache: {len(eval_cache)} samples")
            logger.info("\n  Evaluating backdoor baseline...")

        metrics_baseline = evaluate_projector(
            model, processor, bd_state_half, eval_cache, "backdoor_baseline",
            target, prompt_text, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        if _rank == 0:
            logger.info(f"  Backdoor baseline: {metrics_baseline}")

        baseline_cider = metrics_baseline["clean_cider"] if metrics_baseline else 0.0
        results["backdoor_baseline"] = metrics_baseline

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Compute hidden activations on clean data
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info(f"\nStep 3: Computing hidden activations ({args.n_sample} clean samples)...")

    model.multi_modal_projector.load_state_dict(bd_state)

    collator = TrainLLaVACollator(processor, ignore_index=-100)
    clean_ds = CustomDataset(
        dataset_name=dataset_name,
        prompt=prompt,
        attack_type="replace",
        target="",
        train_num=args.n_sample,
        offset=OFFSET if dataset_name == "coco" else 0,
        poison_rate=0.0,
        seed=42,
        patch_size=bd_cfg.get("patch_size", 30),
        patch_type=bd_cfg.get("patch_type", "random"),
        patch_location=bd_cfg.get("patch_location", "random_f"),
        img_size=bd_cfg.get("img_size", 336),
        neg_sample=False,
    )
    act_loader = DataLoader(
        clean_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    input_device = next(model.parameters()).device
    mean_act, n_batches = compute_mean_activation(model, act_loader, input_device)

    if _rank == 0:
        logger.info(f"  Activation stats: min={mean_act.min():.6f}, max={mean_act.max():.6f}, "
                     f"mean={mean_act.mean():.6f}, n_batches={n_batches}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Determine pruning ratio & prune
    # ══════════════════════════════════════════════════════════════════════
    hidden_dim = mean_act.shape[0]
    search_log = None

    if args.prune_ratio is not None:
        prune_ratio = args.prune_ratio
        if _rank == 0:
            logger.info(f"\nStep 4: Using fixed prune_ratio={prune_ratio:.2f}")
    else:
        if _rank == 0:
            logger.info(f"\nStep 4: Auto-searching prune ratio "
                        f"(cider_threshold={args.cider_threshold:.0%}, "
                        f"max_ratio={args.max_ratio}, step={args.search_step}, "
                        f"baseline_cider={baseline_cider:.2f})...")
        prune_ratio, search_log = find_prune_ratio_cider(
            model, processor, bd_state, mean_act,
            eval_cache, prompt_text, args.eval_batch_size,
            baseline_cider=baseline_cider,
            max_ratio=args.max_ratio,
            step=args.search_step,
            cider_threshold=args.cider_threshold,
        )
        if _rank == 0:
            logger.info(f"  Selected prune_ratio={prune_ratio:.2f}")

    pruned_state, n_pruned, sorted_indices = prune_projector_neurons(
        bd_state, mean_act, prune_ratio
    )

    if _rank == 0:
        logger.info(f"  Pruned {n_pruned}/{hidden_dim} neurons (ratio={n_pruned/hidden_dim:.3f})")

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Fine-tune pruned model + evaluate (fine-pruning)
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info(f"\nStep 5: Fine-tuning pruned model ({NUM_EPOCHS} epochs, "
                     f"lr={LR}, n_sample={args.n_sample})...")

    model.multi_modal_projector.load_state_dict(
        {k: v.clone().float().to(input_device) for k, v in pruned_state.items()}
    )
    model.multi_modal_projector.float()

    train_loader = DataLoader(
        clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    n_steps = finetune_projector(
        model, train_loader,
        num_epochs=NUM_EPOCHS, lr=LR, warmup_ratio=WARMUP_RATIO,
        grad_accum_steps=GRAD_ACCUM,
    )

    fp_state = {k: v.clone().cpu() for k, v in model.multi_modal_projector.state_dict().items()}

    model.multi_modal_projector.half()

    if _rank == 0:
        logger.info(f"  Fine-tuning done: {n_steps} optimizer steps")
        logger.info("  Evaluating fine-pruned model...")

    fp_state_half = {k: v.half() for k, v in fp_state.items()}
    metrics_fp = evaluate_projector(
        model, processor, fp_state_half, eval_cache, "fine_pruning",
        target, prompt_text, args.eval_batch_size,
        rank=_rank, world_size=_world_size,
    )
    if metrics_fp is not None:
        metrics_fp["n_steps"] = n_steps
    if _rank == 0:
        logger.info(f"  Fine-pruning: {metrics_fp}")
    results["fine_pruning"] = metrics_fp

    # ══════════════════════════════════════════════════════════════════════
    # Step 6: Save results + purified weights
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        all_results = {
            "config": {
                "backdoor_dir": str(BACKDOOR_DIR),
                "n_sample": args.n_sample,
                "prune_ratio": prune_ratio,
                "prune_ratio_fixed": args.prune_ratio is not None,
                "cider_threshold": args.cider_threshold,
                "max_ratio": args.max_ratio,
                "search_step": args.search_step,
                "search_log": search_log,
                "num_epochs": NUM_EPOCHS,
                "lr": LR,
                "grad_accum": GRAD_ACCUM,
                "per_device_bs": PER_DEVICE_BS,
                "warmup_ratio": WARMUP_RATIO,
                "offset": OFFSET,
                "dataset": dataset_name,
                "test_num": args.test_num,
                "eval_batch_size": args.eval_batch_size,
                "skip_baseline_eval": args.skip_baseline_eval,
                "clean_utility_reference": baseline_cider,
                "hidden_dim": hidden_dim,
                "n_pruned": n_pruned,
            },
            "results": results,
        }

        out_json = OUTPUT_DIR / "exp8_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n  Results saved → {out_json}")

        out_weights = OUTPUT_DIR / "finepruned_mmprojector_state_dict.pth"
        torch.save(fp_state, str(out_weights))
        logger.info(f"  Weights saved → {out_weights}")

        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        for stage, m in results.items():
            if m is not None:
                metric_name = m.get("metric_name", "CIDEr")
                logger.info(f"  {stage:20s}  ASR={m.get('backdoor_asr', '?'):>6}%  "
                             f"{metric_name}={m.get('clean_cider', '?')}")

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
