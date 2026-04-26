#!/usr/bin/env python3
"""
exp9: Adversarial Neuron Pruning (ANP) Defense Baseline (Qwen3-VL-8B-Instruct)

ANP (Wu & Wang, NeurIPS 2021) adapted from CNN BatchNorm-layer pruning
to VLM adapter hidden neurons. Applied independently to each merger block
(1 main merger + deepstack blocks).

See exp9_anp.py docstring for full adaptation rationale.

Usage:
    cd /home/zzf/data/ZHC/vlm-backdoor-code
    source /data/YBJ/cleansight/venv_qwen3/bin/activate

    CUDA_VISIBLE_DEVICES=0 python exps/exp9_anp/exp9_anp_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr \
        --n_sample 1000 --test_num 512

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        exps/exp9_anp/exp9_anp_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr \
        --n_sample 1000 --test_num 512
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

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
)
from exps.exp9_anp.exp9_anp import ANPHook, anp_train


# ═══════════════════════════════════════════════════════════════════════════════
# Pruning (multi-block) — faithful to prune_neuron_cifar.py
# Only zero weight rows, keep bias (matching original BN.weight zeroing)
# ═══════════════════════════════════════════════════════════════════════════════

def prune_merger_by_threshold(state_dict, mask_values, threshold,
                              weight_key="linear_fc1.weight"):
    prune_indices = (mask_values < threshold).nonzero(as_tuple=True)[0]
    n_prune = prune_indices.shape[0]

    pruned = {k: v.clone() for k, v in state_dict.items()}
    if n_prune > 0:
        pruned[weight_key][prune_indices, :] = 0.0

    return pruned, n_prune


def apply_anp_pruning_all_blocks(merger_state, ds_state, mask_dict, threshold):
    info = {}

    pruned_merger, n_m = prune_merger_by_threshold(
        merger_state, mask_dict["merger"], threshold
    )
    info["merger"] = {"n_pruned": n_m, "n_total": mask_dict["merger"].shape[0]}
    logger.info(f"  Merger: pruned {n_m}/{mask_dict['merger'].shape[0]} neurons")

    pruned_ds = None
    if ds_state is not None:
        pruned_ds = {k: v.clone() for k, v in ds_state.items()}
        for i in range(3):
            key = f"ds_{i}"
            if key not in mask_dict:
                continue
            hidden_dim = mask_dict[key].shape[0]
            prune_idx = (mask_dict[key] < threshold).nonzero(as_tuple=True)[0]
            n_prune = prune_idx.shape[0]

            if n_prune > 0:
                pruned_ds[f"{i}.linear_fc1.weight"][prune_idx, :] = 0.0

            info[key] = {"n_pruned": n_prune, "n_total": hidden_dim}
            logger.info(f"  DeepStack[{i}]: pruned {n_prune}/{hidden_dim} neurons")

    return pruned_merger, pruned_ds, info


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp9: ANP Defense (Qwen3-VL)")
    parser.add_argument("--backdoor_dir", type=str, required=True,
                        help="Path to backdoor checkpoint dir")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="Number of clean samples for ANP optimization")
    parser.add_argument("--anp_eps", type=float, default=0.4,
                        help="L_inf bound for adversarial noise (paper default: 0.4)")
    parser.add_argument("--anp_steps", type=int, default=1,
                        help="PGD steps for inner maximization (paper default: 1)")
    parser.add_argument("--anp_alpha", type=float, default=0.2,
                        help="Weight for natural loss in combined objective (paper default: 0.2)")
    parser.add_argument("--lr_mask", type=float, default=0.2,
                        help="SGD learning rate for mask optimizer (paper default: 0.2)")
    parser.add_argument("--nb_iter", type=int, default=2000,
                        help="Number of mask optimizer steps (paper default: 2000)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps to simulate larger batch "
                             "(effective_bs = 1 × grad_accum; paper uses bs=128)")
    parser.add_argument("--prune_threshold", type=float, default=0.2,
                        help="Prune neurons with mask value < threshold (paper default: 0.2)")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

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

    BACKDOOR_DIR = Path(args.backdoor_dir)
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    else:
        OUTPUT_DIR = PROJECT_ROOT / "exps/exp9_anp/checkpoints" / f"qwen3vl_{BACKDOOR_DIR.name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OFFSET = 5000

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_cfg = json.load(f)

    target = bd_cfg.get("target", "you have been hacked lol")
    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")

    if _distributed and bd_cfg.get("patch_type") == "issba":
        logger.warning("ISSBA detected: disabling multi-GPU eval (TF/PyTorch CUDA conflict).")
        if _rank != 0:
            dist.destroy_process_group()
            return
        dist.destroy_process_group()
        _distributed = False
        _world_size = 1

    if _rank == 0:
        logger.info("=" * 60)
        logger.info("exp9: Adversarial Neuron Pruning (Qwen3-VL)")
        logger.info("=" * 60)
        logger.info(f"  backdoor_dir  = {BACKDOOR_DIR}")
        logger.info(f"  n_sample      = {args.n_sample}")
        logger.info(f"  nb_iter       = {args.nb_iter}")
        logger.info(f"  anp_eps       = {args.anp_eps}")
        logger.info(f"  anp_steps     = {args.anp_steps}")
        logger.info(f"  anp_alpha     = {args.anp_alpha}")
        logger.info(f"  lr_mask       = {args.lr_mask}")
        logger.info(f"  grad_accum    = {args.grad_accum} (effective_bs={args.grad_accum})")
        logger.info(f"  prune_threshold = {args.prune_threshold}")
        logger.info(f"  test_num      = {args.test_num}")
        logger.info(f"  output_dir    = {OUTPUT_DIR}")

    # ══ Step 1: Load model + backdoor weights ══
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

    for p in model.parameters():
        p.requires_grad_(False)

    if _rank == 0:
        logger.info(f"  Loaded backdoor weights from {BACKDOOR_DIR.name}")

    # ══ Step 2: Build eval cache + evaluate baseline ══
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
        logger.info("  Evaluating backdoor baseline...")

    merger_bd_half = {k: v.half() for k, v in merger_bd.items()}
    ds_bd_half = {k: v.half() for k, v in ds_bd.items()} if ds_bd is not None else None

    metrics_baseline = evaluate_qwen3vl_adapter(
        model, processor, merger_bd_half, ds_bd_half, eval_cache, "backdoor_baseline",
        target, args.eval_batch_size, rank=_rank, world_size=_world_size,
    )
    if _rank == 0:
        logger.info(f"  Backdoor baseline: {metrics_baseline}")

    results = {"backdoor_baseline": metrics_baseline}

    # ══ Step 3: ANP optimization (rank 0 only) ══
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

    input_device = next(model.parameters()).device

    if _rank == 0:
        logger.info(f"\nStep 3: ANP optimization ({args.nb_iter} steps × "
                     f"{args.grad_accum} accum, {args.n_sample} clean samples)...")

        visual.merger.load_state_dict({k: v.half() for k, v in merger_bd.items()})
        if ds_bd is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            visual.deepstack_merger_list.load_state_dict({k: v.half() for k, v in ds_bd.items()})

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        anp_loader = DataLoader(
            clean_ds, batch_size=1, shuffle=False,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

        hooks: List[ANPHook] = []

        merger_hidden = visual.merger.linear_fc1.weight.shape[0]
        hook_merger = ANPHook(merger_hidden, device=input_device, eps=args.anp_eps)
        hook_merger.attach(visual.merger.linear_fc1)
        hooks.append(hook_merger)
        logger.info(f"  Merger hook: hidden_dim={merger_hidden}")

        if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            for i, block in enumerate(visual.deepstack_merger_list):
                ds_hidden = block.linear_fc1.weight.shape[0]
                hook_ds = ANPHook(ds_hidden, device=input_device, eps=args.anp_eps)
                hook_ds.attach(block.linear_fc1)
                hooks.append(hook_ds)
                logger.info(f"  DeepStack[{i}] hook: hidden_dim={ds_hidden}")

        avg_loss = anp_train(
            model, anp_loader, hooks,
            anp_eps=args.anp_eps,
            anp_steps=args.anp_steps,
            anp_alpha=args.anp_alpha,
            lr_mask=args.lr_mask,
            nb_iter=args.nb_iter,
            grad_accum=args.grad_accum,
            device=input_device,
            rank=_rank,
        )
        logger.info(f"  ANP done. Avg loss: {avg_loss:.4f}")

        mask_dict = {}
        hook_names = ["merger"] + [f"ds_{i}" for i in range(len(hooks) - 1)]
        for h, name in zip(hooks, hook_names):
            mask_dict[name] = h.mask.detach().cpu()
            h.detach()
            logger.info(f"  {name}: mask min={mask_dict[name].min():.4f}, "
                         f"max={mask_dict[name].max():.4f}, "
                         f"mean={mask_dict[name].mean():.4f}")

        model.gradient_checkpointing_disable()
    else:
        mask_dict = None

    if _distributed:
        payload = [mask_dict]
        dist.broadcast_object_list(payload, src=0)
        mask_dict = payload[0]

    # ══ Step 4: Prune by mask values ══
    visual.merger.load_state_dict({k: v.half() for k, v in merger_bd.items()})
    if ds_bd is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.load_state_dict({k: v.half() for k, v in ds_bd.items()})
    model.eval()

    prune_threshold = args.prune_threshold
    if _rank == 0:
        logger.info(f"\nStep 4: Pruning neurons with mask < {prune_threshold}")

    pruned_merger, pruned_ds, prune_info = apply_anp_pruning_all_blocks(
        merger_bd, ds_bd, mask_dict, prune_threshold
    )

    # ══ Step 5: Evaluate ANP-pruned model ══
    if _rank == 0:
        logger.info("\nStep 5: Evaluating ANP-pruned model...")

    pm_half = {k: v.half() for k, v in pruned_merger.items()}
    pd_half = {k: v.half() for k, v in pruned_ds.items()} if pruned_ds is not None else None

    metrics_anp = evaluate_qwen3vl_adapter(
        model, processor, pm_half, pd_half, eval_cache, "anp_pruned",
        target, args.eval_batch_size, rank=_rank, world_size=_world_size,
    )
    if _rank == 0:
        logger.info(f"  ANP-pruned: {metrics_anp}")
    results["anp_pruned"] = metrics_anp

    # ══ Step 6: Save results + weights ══
    if _rank == 0:
        mask_stats = {}
        for name, mv in mask_dict.items():
            mask_stats[name] = {
                "min": round(float(mv.min()), 4),
                "max": round(float(mv.max()), 4),
                "mean": round(float(mv.mean()), 4),
                "std": round(float(mv.std()), 4),
                "n_below_0.5": int((mv < 0.5).sum()),
                "n_below_0.1": int((mv < 0.1).sum()),
                "hidden_dim": int(mv.shape[0]),
            }

        all_results = {
            "config": {
                "backdoor_dir": str(BACKDOOR_DIR),
                "n_sample": args.n_sample,
                "nb_iter": args.nb_iter,
                "grad_accum": args.grad_accum,
                "anp_eps": args.anp_eps,
                "anp_steps": args.anp_steps,
                "anp_alpha": args.anp_alpha,
                "lr_mask": args.lr_mask,
                "prune_threshold": prune_threshold,
                "test_num": args.test_num,
                "eval_batch_size": args.eval_batch_size,
                "offset": OFFSET,
            },
            "mask_stats": mask_stats,
            "prune_info": prune_info,
            "results": results,
        }

        out_json = OUTPUT_DIR / "exp9_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n  Results saved → {out_json}")

        out_merger = OUTPUT_DIR / "anp_pruned_merger_state_dict.pth"
        torch.save(pruned_merger, str(out_merger))
        logger.info(f"  Merger weights saved → {out_merger}")

        if pruned_ds is not None:
            out_ds = OUTPUT_DIR / "anp_pruned_deepstack_merger_list_state_dict.pth"
            torch.save(pruned_ds, str(out_ds))
            logger.info(f"  DeepStack weights saved → {out_ds}")

        out_mask = OUTPUT_DIR / "anp_mask_values.pth"
        torch.save(mask_dict, str(out_mask))
        logger.info(f"  Mask values saved → {out_mask}")

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
