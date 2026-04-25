#!/usr/bin/env python3
"""
exp8: Fine-Pruning Defense Baseline (Qwen3-VL-8B-Instruct)

Fine-Pruning (Liu et al., RAID 2018) adapted from CNN last-conv-layer pruning
to VLM adapter hidden neurons. Applied independently to each merger block
(1 main merger + 3 deepstack blocks).

    1. Forward clean data → hook post-GELU hidden activations in each merger block
    2. Rank hidden neurons by ascending mean activation (per block)
    3. Prune the bottom `prune_ratio` fraction (zero linear_fc1 weight+bias)
    4. Fine-tune the pruned adapter on clean data (2 epochs)
    5. Evaluate ASR / CIDEr at each stage

Usage:
    cd /home/zzf/data/ZHC/vlm-backdoor-code
    source /data/YBJ/cleansight/venv_qwen3/bin/activate

    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python exps/exp8_fine_pruning/exp8_fine_pruning_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_pr0.1 \
        --n_sample 64 --test_num 512

    # Multi-GPU (distributed evaluation)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        exps/exp8_fine_pruning/exp8_fine_pruning_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_pr0.1 \
        --n_sample 64 --test_num 512
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = str(PROJECT_ROOT / "models/Qwen3-VL-8B-Instruct")

from exps.exp1b_projection.exp1b_projection import (
    build_eval_cache,
    chunks,
)
from exps.exp1c_pseudo_benign.exp1c_pseudo_benign_qwen3vl import (
    evaluate_qwen3vl_adapter,
    finetune_adapter_qwen3vl,
    extract_clean_merger_weights,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-Pruning Core
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mean_activation_qwen3vl(model, dataloader, device):
    """
    Forward clean data through the model. Hook each merger block's linear_fc2
    input to capture post-GELU hidden activations.

    Returns: dict mapping block name → mean activation tensor (hidden_dim,)
    """
    visual = model.model.visual
    accumulators = {}
    counts = {}
    handles = []

    def make_hook(name, hidden_dim):
        accumulators[name] = torch.zeros(hidden_dim, dtype=torch.float64)
        counts[name] = 0

        def hook_fn(module, inputs, output):
            act = inputs[0].detach().float()
            accumulators[name] += act.sum(dim=tuple(range(act.ndim - 1))).cpu().double()
            counts[name] += act[..., 0].numel()

        return hook_fn

    merger_hidden = visual.merger.linear_fc1.weight.shape[0]
    h = visual.merger.linear_fc2.register_forward_hook(make_hook("merger", merger_hidden))
    handles.append(h)

    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        for i, block in enumerate(visual.deepstack_merger_list):
            ds_hidden = block.linear_fc1.weight.shape[0]
            h = block.linear_fc2.register_forward_hook(make_hook(f"ds_{i}", ds_hidden))
            handles.append(h)

    model.eval()
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model(**batch, labels=labels)
            n_batches += 1

    for h in handles:
        h.remove()

    result = {}
    for name in accumulators:
        if counts[name] > 0:
            result[name] = (accumulators[name] / counts[name]).float()
        else:
            result[name] = accumulators[name].float()

    return result, n_batches


def prune_merger_neurons(state_dict, mean_activation, prune_ratio,
                         weight_key="linear_fc1.weight", bias_key="linear_fc1.bias"):
    """
    Zero weight[j,:] and bias[j] for the bottom `prune_ratio` fraction of neurons.

    Returns: (pruned_state_dict, n_pruned)
    """
    hidden_dim = mean_activation.shape[0]
    n_prune = int(hidden_dim * prune_ratio)

    sorted_indices = torch.argsort(mean_activation)
    prune_indices = sorted_indices[:n_prune]

    pruned = {k: v.clone() for k, v in state_dict.items()}
    pruned[weight_key][prune_indices, :] = 0.0
    pruned[bias_key][prune_indices] = 0.0

    return pruned, n_prune


def apply_fine_pruning_all_blocks(merger_state, ds_state, activations, prune_ratio):
    """
    Apply pruning to the main merger and each deepstack block independently.

    Returns: (pruned_merger, pruned_ds, prune_info_dict)
    """
    info = {}

    pruned_merger, n_m = prune_merger_neurons(
        merger_state, activations["merger"], prune_ratio
    )
    info["merger"] = {"n_pruned": n_m, "n_total": activations["merger"].shape[0]}
    logger.info(f"  Merger: pruned {n_m}/{activations['merger'].shape[0]} neurons")

    pruned_ds = None
    if ds_state is not None:
        pruned_ds = {k: v.clone() for k, v in ds_state.items()}
        for i in range(3):
            key = f"ds_{i}"
            if key not in activations:
                continue
            hidden_dim = activations[key].shape[0]
            n_prune = int(hidden_dim * prune_ratio)
            sorted_idx = torch.argsort(activations[key])
            prune_idx = sorted_idx[:n_prune]

            pruned_ds[f"{i}.linear_fc1.weight"][prune_idx, :] = 0.0
            pruned_ds[f"{i}.linear_fc1.bias"][prune_idx] = 0.0

            info[key] = {"n_pruned": n_prune, "n_total": hidden_dim}
            logger.info(f"  DeepStack[{i}]: pruned {n_prune}/{hidden_dim} neurons")

    return pruned_merger, pruned_ds, info


def compute_clean_loss_qwen3vl(model, merger_state, ds_state, dataloader, device):
    """Compute average CE loss on clean data (cheap forward pass, no generation)."""
    visual = model.model.visual
    visual.merger.load_state_dict(merger_state)
    if ds_state is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.load_state_dict(ds_state)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch, labels=labels)
            total_loss += outputs.loss.item()
            n_batches += 1
    return total_loss / n_batches


def find_prune_ratio_qwen3vl(model, merger_bd, ds_bd, activations, dataloader, device,
                              max_ratio=0.95, step=0.05, loss_threshold=0.10):
    """
    Iteratively prune all merger blocks at increasing ratios, stop when clean
    loss degrades beyond threshold (paper Sec 3.1: "4% accuracy drop").

    Returns: (selected_ratio, search_log)
    """
    merger_bd_half = {k: v.half() for k, v in merger_bd.items()}
    ds_bd_half = {k: v.half() for k, v in ds_bd.items()} if ds_bd is not None else None
    baseline_loss = compute_clean_loss_qwen3vl(model, merger_bd_half, ds_bd_half, dataloader, device)
    logger.info(f"  Baseline loss: {baseline_loss:.4f}")

    selected_ratio = 0.0
    search_log = []

    ratios = [round(step * i, 2) for i in range(1, int(max_ratio / step) + 1)]

    for ratio in ratios:
        pruned_merger, pruned_ds, _ = apply_fine_pruning_all_blocks(
            merger_bd, ds_bd, activations, ratio
        )
        pm_half = {k: v.half() for k, v in pruned_merger.items()}
        pd_half = {k: v.half() for k, v in pruned_ds.items()} if pruned_ds is not None else None
        loss = compute_clean_loss_qwen3vl(model, pm_half, pd_half, dataloader, device)
        rel_increase = (loss - baseline_loss) / baseline_loss
        search_log.append({"ratio": ratio, "loss": round(loss, 4),
                           "rel_increase": round(rel_increase, 4)})
        logger.info(f"    ratio={ratio:.2f}: loss={loss:.4f} (Δ={rel_increase:+.1%})")
        if rel_increase > loss_threshold:
            logger.info(f"  → Threshold exceeded. Selected ratio={selected_ratio:.2f}")
            break
        selected_ratio = ratio
    else:
        logger.info(f"  → Threshold never exceeded. Using max ratio={selected_ratio:.2f}")

    return selected_ratio, search_log


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp8: Fine-Pruning Defense (Qwen3-VL)")
    parser.add_argument("--backdoor_dir", type=str, required=True,
                        help="Path to backdoor checkpoint dir")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="Number of clean samples for activation computation AND fine-tuning")
    parser.add_argument("--prune_ratio", type=float, default=None,
                        help="Fixed prune ratio (skip auto-search). If omitted, auto-search "
                             "following the paper's stopping criterion.")
    parser.add_argument("--loss_threshold", type=float, default=0.10,
                        help="Relative loss increase threshold for auto-search "
                             "(paper uses ~4%% accuracy drop; 10%% loss increase is analogous)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16)
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
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    else:
        OUTPUT_DIR = PROJECT_ROOT / "exps/exp8_fine_pruning/checkpoints" / f"qwen3vl_{BACKDOOR_DIR.name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Config ─────────────────────────────────────────────────────────────
    NUM_EPOCHS = 2
    PER_DEVICE_BS = 4
    GRAD_ACCUM = 4
    LR = 5e-5
    WARMUP_RATIO = 0.03
    OFFSET = 5000

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_cfg = json.load(f)

    target = bd_cfg.get("target", "you have been hacked lol")
    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")

    # ISSBA uses TensorFlow which conflicts with PyTorch CUDA in torchrun
    if _distributed and bd_cfg.get("patch_type") == "issba":
        logger.warning("ISSBA detected: disabling multi-GPU eval (TF/PyTorch CUDA conflict).")
        if _rank != 0:
            dist.destroy_process_group()
            return
        _distributed = False
        _world_size = 1

    if _rank == 0:
        logger.info("=" * 60)
        logger.info("exp8: Fine-Pruning Defense (Qwen3-VL)")
        logger.info("=" * 60)
        logger.info(f"  backdoor_dir  = {BACKDOOR_DIR}")
        logger.info(f"  n_sample      = {args.n_sample}")
        logger.info(f"  prune_ratio   = {args.prune_ratio or 'auto-search'}")
        logger.info(f"  loss_threshold= {args.loss_threshold}")
        logger.info(f"  test_num      = {args.test_num}")
        logger.info(f"  output_dir    = {OUTPUT_DIR}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Load model + backdoor weights
    # ══════════════════════════════════════════════════════════════════════
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainQwen3VLCollator

    if _rank == 0:
        logger.info("\nStep 1: Loading model + backdoor weights...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )

    visual = model.model.visual

    merger_bd = torch.load(str(BACKDOOR_DIR / "merger_state_dict.pth"), map_location="cpu")
    merger_bd = {k: v.float() for k, v in merger_bd.items()}
    ds_bd = None
    ds_bd_path = BACKDOOR_DIR / "deepstack_merger_list_state_dict.pth"
    if ds_bd_path.exists():
        ds_bd = torch.load(str(ds_bd_path), map_location="cpu")
        ds_bd = {k: v.float() for k, v in ds_bd.items()}

    visual.merger.load_state_dict({k: v.half() for k, v in merger_bd.items()})
    if ds_bd is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.load_state_dict({k: v.half() for k, v in ds_bd.items()})

    for name, p in model.named_parameters():
        if "merger" in name or "deepstack_merger" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    if _rank == 0:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable params: {n_trainable:,} (merger + deepstack)")
        logger.info(f"  Loaded backdoor weights from {BACKDOOR_DIR.name}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Build eval cache + evaluate backdoor baseline
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("\nStep 2: Building eval cache...")

    from datasets import load_dataset
    test_dataset = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))
    eval_cache = build_eval_cache(test_dataset, bd_cfg, args.test_num)

    if _rank == 0:
        logger.info(f"  Eval cache: {len(eval_cache)} images")
        logger.info("\n  Evaluating backdoor baseline...")

    merger_bd_half = {k: v.half() for k, v in merger_bd.items()}
    ds_bd_half = {k: v.half() for k, v in ds_bd.items()} if ds_bd is not None else None

    metrics_baseline = evaluate_qwen3vl_adapter(
        model, processor, merger_bd_half, ds_bd_half, eval_cache, "backdoor_baseline",
        target, args.eval_batch_size, rank=_rank, world_size=_world_size,
    )
    if _rank == 0:
        logger.info(f"  Backdoor baseline: {metrics_baseline}")

    results = {"backdoor_baseline": metrics_baseline}

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Compute hidden activations on clean data
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info(f"\nStep 3: Computing hidden activations ({args.n_sample} clean samples)...")

    visual.merger.load_state_dict({k: v.half() for k, v in merger_bd.items()})
    if ds_bd is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.load_state_dict({k: v.half() for k, v in ds_bd.items()})

    collator = TrainQwen3VLCollator(processor, ignore_index=-100)
    clean_ds = CustomDataset(
        dataset_name="coco",
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
    act_loader = DataLoader(
        clean_ds, batch_size=PER_DEVICE_BS, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    input_device = next(model.parameters()).device
    activations, n_batches = compute_mean_activation_qwen3vl(model, act_loader, input_device)

    if _rank == 0:
        for name, act in activations.items():
            logger.info(f"  {name}: min={act.min():.6f}, max={act.max():.6f}, mean={act.mean():.6f}")
        logger.info(f"  n_batches={n_batches}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Determine pruning ratio & prune
    # ══════════════════════════════════════════════════════════════════════
    search_log = None

    if args.prune_ratio is not None:
        prune_ratio = args.prune_ratio
        if _rank == 0:
            logger.info(f"\nStep 4: Using fixed prune_ratio={prune_ratio:.2f}")
    else:
        if _rank == 0:
            logger.info(f"\nStep 4: Auto-searching prune ratio "
                        f"(loss_threshold={args.loss_threshold:.0%})...")
        prune_ratio, search_log = find_prune_ratio_qwen3vl(
            model, merger_bd, ds_bd, activations, act_loader, input_device,
            loss_threshold=args.loss_threshold,
        )
        if _rank == 0:
            logger.info(f"  Selected prune_ratio={prune_ratio:.2f}")

    pruned_merger, pruned_ds, prune_info = apply_fine_pruning_all_blocks(
        merger_bd, ds_bd, activations, prune_ratio
    )

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Fine-tune pruned model + evaluate (fine-pruning)
    # ══════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info(f"\nStep 5: Fine-tuning pruned model ({NUM_EPOCHS} epochs, "
                     f"lr={LR}, n_sample={args.n_sample})...")

    merger_device = next(visual.merger.parameters()).device
    visual.merger.load_state_dict(
        {k: v.clone().float().to(merger_device) for k, v in pruned_merger.items()}
    )
    visual.merger.float()

    if pruned_ds is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        ds_device = next(visual.deepstack_merger_list.parameters()).device
        visual.deepstack_merger_list.load_state_dict(
            {k: v.clone().float().to(ds_device) for k, v in pruned_ds.items()}
        )
        visual.deepstack_merger_list.float()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_loader = DataLoader(
        clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    n_steps = finetune_adapter_qwen3vl(
        model, train_loader,
        num_epochs=NUM_EPOCHS, lr=LR, warmup_ratio=WARMUP_RATIO,
        grad_accum_steps=GRAD_ACCUM,
    )

    fp_merger = {k: v.clone().cpu() for k, v in visual.merger.state_dict().items()}
    fp_ds = None
    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        fp_ds = {k: v.clone().cpu() for k, v in visual.deepstack_merger_list.state_dict().items()}

    visual.merger.half()
    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.half()

    if _rank == 0:
        logger.info(f"  Fine-tuning done: {n_steps} optimizer steps")
        logger.info("  Evaluating fine-pruned model...")

    fp_merger_half = {k: v.half() for k, v in fp_merger.items()}
    fp_ds_half = {k: v.half() for k, v in fp_ds.items()} if fp_ds is not None else None

    metrics_fp = evaluate_qwen3vl_adapter(
        model, processor, fp_merger_half, fp_ds_half, eval_cache, "fine_pruning",
        target, args.eval_batch_size, rank=_rank, world_size=_world_size,
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
                "loss_threshold": args.loss_threshold,
                "search_log": search_log,
                "num_epochs": NUM_EPOCHS,
                "lr": LR,
                "grad_accum": GRAD_ACCUM,
                "per_device_bs": PER_DEVICE_BS,
                "warmup_ratio": WARMUP_RATIO,
                "offset": OFFSET,
                "test_num": args.test_num,
                "eval_batch_size": args.eval_batch_size,
            },
            "results": results,
        }

        out_json = OUTPUT_DIR / "exp8_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n  Results saved → {out_json}")

        out_merger = OUTPUT_DIR / "finepruned_merger_state_dict.pth"
        torch.save(fp_merger, str(out_merger))
        logger.info(f"  Merger weights saved → {out_merger}")

        if fp_ds is not None:
            out_ds = OUTPUT_DIR / "finepruned_deepstack_merger_list_state_dict.pth"
            torch.save(fp_ds, str(out_ds))
            logger.info(f"  DeepStack weights saved → {out_ds}")

        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        for stage, m in results.items():
            if m is not None:
                logger.info(f"  {stage:20s}  ASR={m.get('backdoor_asr', '?'):>6}%  "
                             f"CIDEr={m.get('clean_cider', '?')}")

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
