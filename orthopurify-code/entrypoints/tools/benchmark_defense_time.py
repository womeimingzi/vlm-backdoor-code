#!/usr/bin/env python3
"""
Defense Algorithm Timing Benchmark — LLaVA-1.5-7B + COCO + BadNet
==================================================================
精确测量四种防御算法的核心计算耗时（不含模型加载、数据准备、评估）。

Timed (core algorithm only):
  exp7  — Fine-tuning Recovery:  finetune_projector()
  exp8  — Fine-Pruning:          compute_mean_activation() + prune + finetune_projector()
  exp9  — ANP:                   anp_defense() + apply_pruning()
  exp10 — CLP:                   channel_lipschitz_pruning()

NOT timed: model loading, checkpoint loading, dataset construction, evaluation.

Usage:
    cd /path/to/orthopurify-code
    source /data/YBJ/GraduProject/venv/bin/activate

    CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark_defense_time.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
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
os.chdir(PROJECT_ROOT)

from experiments.main_method.orthopurify_exp1c.exp1c_pseudo_benign import finetune_projector
from experiments.baseline_methods.exp8_fine_pruning.exp8_fine_pruning import (
    compute_mean_activation,
    prune_projector_neurons,
)
from experiments.baseline_methods.exp9_anp.anp_defense import anp_defense, apply_pruning
from experiments.baseline_methods.exp10_clp.clp_defense import channel_lipschitz_pruning
from vlm_backdoor.data.dataset import CustomDataset
from vlm_backdoor.data.collators import TrainLLaVACollator

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameters (matching each experiment's defaults for fair comparison)
# ═══════════════════════════════════════════════════════════════════════════════

# exp7 & exp8 shared training config
FT_NUM_EPOCHS   = 2
FT_LR           = 2e-4
FT_WARMUP_RATIO = 0.03
FT_GRAD_ACCUM   = 16
FT_PER_DEVICE_BS = 1

# exp8 Fine-Pruning
FP_PRUNE_RATIO  = 0.90

# exp9 ANP
ANP_EPS             = 0.02
ANP_PGD_STEPS       = 10
ANP_THETA_LR        = 0.1
ANP_LAM             = 0.01
ANP_CLEAN_LOSS_W    = 1.0
ANP_PRUNE_THRESHOLD = 0.30
ANP_N_ROUNDS        = 1000

# exp10 CLP
CLP_U = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Timing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def sync_and_time():
    torch.cuda.synchronize()
    return time.time()


def log_ts(method, event, extra=""):
    msg = f"[TIMESTAMP] {ts()} | {method:<6} | {event}"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def reset_projector(model, pb_state, device):
    model.multi_modal_projector.load_state_dict(pb_state)
    model.multi_modal_projector.to(torch.float16)
    model.eval()
    # Keep gradient checkpointing inner model in training mode
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", lm)
        if getattr(inner, "_gradient_checkpointing_func", None) is not None:
            inner.training = True
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Defense Algorithm Timing Benchmark (LLaVA + COCO + BadNet)")
    parser.add_argument("--backdoor_dir", type=str,
                        default="model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1")
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="Clean samples for exp7 Fine-tuning Recovery and exp8 Fine-Pruning")
    parser.add_argument("--anp_test_num", type=int, default=500,
                        help="Clean samples for exp9 ANP defense (dataloader size)")
    parser.add_argument("--anp_n_rounds", type=int, default=ANP_N_ROUNDS,
                        help="ANP optimization iterations")
    parser.add_argument("--exp8_prune_ratio", type=float, default=FP_PRUNE_RATIO,
                        help="Fixed pruning ratio for exp8 (skip auto-search)")
    parser.add_argument("--clp_u", type=float, default=CLP_U,
                        help="CLP threshold multiplier")
    parser.add_argument("--offset", type=int, default=5000,
                        help="COCO train data offset")
    parser.add_argument("--output_dir", type=str, default="experiments/timing_benchmark")
    args = parser.parse_args()

    BACKDOOR_DIR = Path(args.backdoor_dir)
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR

    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BACKDOOR_DIR / "local.json") as f:
        bd_cfg = json.load(f)

    prompt     = bd_cfg.get("prompt", "Describe this image in a short sentence.")
    target     = bd_cfg.get("target", "you have been hacked lol")
    patch_type = bd_cfg.get("patch_type", "random")
    patch_loc  = bd_cfg.get("patch_location", "random")
    patch_size = bd_cfg.get("patch_size", 30)
    img_size   = bd_cfg.get("img_size", 336)
    dataset_name = bd_cfg.get("dataset", "coco").lower()
    model_path = bd_cfg.get("model_name_or_path", "/data/YBJ/cleansight/models/llava-1.5-7b-hf")

    print("=" * 70)
    print("Defense Algorithm Timing Benchmark")
    print("=" * 70)
    print(f"  backdoor_dir   = {BACKDOOR_DIR}")
    print(f"  dataset        = {dataset_name}")
    print(f"  n_sample       = {args.n_sample}  (exp7, exp8)")
    print(f"  anp_test_num   = {args.anp_test_num}  (exp9)")
    print(f"  anp_n_rounds   = {args.anp_n_rounds}  (exp9)")
    print(f"  exp8_prune_ratio = {args.exp8_prune_ratio}")
    print(f"  clp_u          = {args.clp_u}")
    print(f"  output_dir     = {OUTPUT_DIR}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 0: Shared setup (NOT timed)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Setup] {ts()} Loading model and data (not timed)...")

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        max_memory = {i: "10GiB" for i in range(n_gpus)}
        print(f"[Setup] {n_gpus} GPUs visible, max_memory=10GiB per GPU")
    else:
        max_memory = None
        print(f"[Setup] 1 GPU visible")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        max_memory=max_memory,
    )

    # Gradient checkpointing: trade compute for memory
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "gradient_checkpointing_enable"):
        lm.gradient_checkpointing_enable()
        inner = getattr(lm, "model", lm)
        inner.training = True
        print("[Setup] Gradient checkpointing enabled on language_model")

    pb_path = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    pb_state = torch.load(str(pb_path), map_location="cpu")
    model.multi_modal_projector.load_state_dict(pb_state)

    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Setup] Trainable params: {trainable:,}")

    collator = TrainLLaVACollator(processor, ignore_index=-100)

    # Clean dataset for exp7 / exp8
    print(f"[Setup] Building clean dataset (n={args.n_sample}) for exp7/exp8...")
    clean_ds_large = CustomDataset(
        dataset_name=dataset_name,
        prompt=prompt,
        attack_type="replace",
        target="",
        train_num=args.n_sample,
        offset=args.offset if dataset_name == "coco" else 0,
        poison_rate=0.0,
        seed=42,
        patch_size=patch_size,
        patch_type=patch_type,
        patch_location=patch_loc,
        img_size=img_size,
        neg_sample=False,
    )

    # Clean dataset for exp9 ANP
    print(f"[Setup] Building clean dataset (n={args.anp_test_num}) for exp9...")
    clean_ds_anp = CustomDataset(
        dataset_name=dataset_name,
        prompt=prompt,
        attack_type="replace",
        target="",
        train_num=args.anp_test_num,
        offset=args.offset if dataset_name == "coco" else 0,
        poison_rate=0.0,
        seed=42,
        patch_size=patch_size,
        patch_type=patch_type,
        patch_location=patch_loc,
        img_size=img_size,
        neg_sample=False,
    )

    print(f"[Setup] {ts()} Setup complete.\n")

    timing_results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # exp10: CLP — Channel Lipschitz Pruning (zero-shot, CPU only)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"[exp10] CLP Defense (u={args.clp_u})")
    print("=" * 70)

    bd_state_cpu = {k: v.clone().cpu() for k, v in pb_state.items()}

    log_ts("exp10", "START")
    t0 = sync_and_time()

    clp_pruned, clp_stats = channel_lipschitz_pruning(bd_state_cpu, u=args.clp_u)

    t1 = sync_and_time()
    log_ts("exp10", "END", f"duration={t1 - t0:.3f}s")

    total_pruned = sum(s["n_pruned"] for s in clp_stats.values())
    total_ch = sum(s["num_channels"] for s in clp_stats.values())
    print(f"[exp10] Pruned {total_pruned}/{total_ch} channels")

    timing_results["exp10_CLP"] = {
        "duration_sec": round(t1 - t0, 3),
        "start": ts(),
        "method": "Channel Lipschitz Pruning",
        "params": {"u": args.clp_u},
        "pruned": f"{total_pruned}/{total_ch}",
    }

    del clp_pruned, bd_state_cpu
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # exp7: Fine-tuning Recovery
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"[exp7] Fine-tuning Recovery (n_sample={args.n_sample}, epochs={FT_NUM_EPOCHS})")
    print("=" * 70)

    reset_projector(model, pb_state, "cuda")
    model.multi_modal_projector.float()

    train_loader_7 = DataLoader(
        clean_ds_large, batch_size=FT_PER_DEVICE_BS, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    steps_per_epoch = math.ceil(len(train_loader_7) / FT_GRAD_ACCUM)
    total_steps = steps_per_epoch * FT_NUM_EPOCHS
    print(f"[exp7] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

    log_ts("exp7", "START")
    t0 = sync_and_time()

    n_steps_7 = finetune_projector(
        model, train_loader_7,
        num_epochs=FT_NUM_EPOCHS, lr=FT_LR,
        warmup_ratio=FT_WARMUP_RATIO,
        grad_accum_steps=FT_GRAD_ACCUM,
    )

    t1 = sync_and_time()
    log_ts("exp7", "END", f"duration={t1 - t0:.3f}s")
    print(f"[exp7] Completed {n_steps_7} optimizer steps")

    timing_results["exp7_FinetuneRecovery"] = {
        "duration_sec": round(t1 - t0, 3),
        "method": "Fine-tuning Recovery",
        "params": {
            "n_sample": args.n_sample,
            "num_epochs": FT_NUM_EPOCHS,
            "lr": FT_LR,
            "grad_accum": FT_GRAD_ACCUM,
            "total_steps": n_steps_7,
        },
    }

    del train_loader_7
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # exp8: Fine-Pruning
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"[exp8] Fine-Pruning (n_sample={args.n_sample}, prune_ratio={args.exp8_prune_ratio})")
    print("=" * 70)

    reset_projector(model, pb_state, "cuda")

    act_loader = DataLoader(
        clean_ds_large, batch_size=4, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    input_device = next(model.parameters()).device

    log_ts("exp8", "START")
    t0 = sync_and_time()

    # Step 1: Compute hidden activations
    log_ts("exp8", "STEP1_activation_start")
    t_act_start = sync_and_time()

    mean_act, n_batches = compute_mean_activation(model, act_loader, input_device)

    t_act_end = sync_and_time()
    log_ts("exp8", "STEP1_activation_end", f"duration={t_act_end - t_act_start:.3f}s")
    print(f"[exp8] Activation stats: min={mean_act.min():.6f}, max={mean_act.max():.6f}, "
          f"mean={mean_act.mean():.6f}")

    # Step 2: Prune
    log_ts("exp8", "STEP2_prune_start")
    t_prune_start = sync_and_time()

    pruned_state, n_pruned, sorted_indices = prune_projector_neurons(
        pb_state, mean_act, args.exp8_prune_ratio
    )
    hidden_dim = mean_act.shape[0]

    t_prune_end = sync_and_time()
    log_ts("exp8", "STEP2_prune_end", f"duration={t_prune_end - t_prune_start:.3f}s")
    print(f"[exp8] Pruned {n_pruned}/{hidden_dim} neurons")

    # Step 3: Fine-tune pruned projector
    log_ts("exp8", "STEP3_finetune_start")

    model.multi_modal_projector.load_state_dict(
        {k: v.clone().float().to(input_device) for k, v in pruned_state.items()}
    )
    model.multi_modal_projector.float()

    train_loader_8 = DataLoader(
        clean_ds_large, batch_size=FT_PER_DEVICE_BS, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    t_ft_start = sync_and_time()

    n_steps_8 = finetune_projector(
        model, train_loader_8,
        num_epochs=FT_NUM_EPOCHS, lr=FT_LR,
        warmup_ratio=FT_WARMUP_RATIO,
        grad_accum_steps=FT_GRAD_ACCUM,
    )

    t_ft_end = sync_and_time()
    log_ts("exp8", "STEP3_finetune_end", f"duration={t_ft_end - t_ft_start:.3f}s")

    t1 = sync_and_time()
    log_ts("exp8", "END", f"duration={t1 - t0:.3f}s")

    timing_results["exp8_FinePruning"] = {
        "duration_sec": round(t1 - t0, 3),
        "method": "Fine-Pruning",
        "params": {
            "n_sample": args.n_sample,
            "prune_ratio": args.exp8_prune_ratio,
            "num_epochs": FT_NUM_EPOCHS,
            "lr": FT_LR,
            "total_steps": n_steps_8,
        },
        "substeps": {
            "activation_computation_sec": round(t_act_end - t_act_start, 3),
            "pruning_sec": round(t_prune_end - t_prune_start, 3),
            "finetuning_sec": round(t_ft_end - t_ft_start, 3),
        },
    }

    del act_loader, train_loader_8, pruned_state
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # exp9: ANP — Adversarial Neuron Pruning
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"[exp9] ANP Defense (test_num={args.anp_test_num}, n_rounds={args.anp_n_rounds})")
    print("=" * 70)

    reset_projector(model, pb_state, "cuda")

    anp_loader = DataLoader(
        clean_ds_anp, batch_size=1, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    log_ts("exp9", "START")
    t0 = sync_and_time()

    pruned_sd = anp_defense(
        model, anp_loader,
        triggered_loader=None,
        eps=ANP_EPS,
        pgd_steps=ANP_PGD_STEPS,
        theta_lr=ANP_THETA_LR,
        lam=ANP_LAM,
        clean_loss_weight=ANP_CLEAN_LOSS_W,
        n_rounds=args.anp_n_rounds,
        prune_threshold=ANP_PRUNE_THRESHOLD,
        log_interval=100,
        fp16=True,
        device=str(next(model.parameters()).device),
    )
    apply_pruning(model, pruned_sd)

    t1 = sync_and_time()
    log_ts("exp9", "END", f"duration={t1 - t0:.3f}s")

    timing_results["exp9_ANP"] = {
        "duration_sec": round(t1 - t0, 3),
        "method": "Adversarial Neuron Pruning",
        "params": {
            "test_num": args.anp_test_num,
            "n_rounds": args.anp_n_rounds,
            "eps": ANP_EPS,
            "pgd_steps": ANP_PGD_STEPS,
            "theta_lr": ANP_THETA_LR,
            "lam": ANP_LAM,
            "prune_threshold": ANP_PRUNE_THRESHOLD,
        },
    }

    del anp_loader, pruned_sd
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TIMING RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Time (s)':>10} {'Time (min)':>10}")
    print("-" * 55)
    for key in ["exp10_CLP", "exp7_FinetuneRecovery", "exp8_FinePruning", "exp9_ANP"]:
        r = timing_results[key]
        sec = r["duration_sec"]
        print(f"{key:<30} {sec:>10.3f} {sec/60:>10.2f}")
    print("=" * 70)

    if "exp8_FinePruning" in timing_results:
        sub = timing_results["exp8_FinePruning"].get("substeps", {})
        if sub:
            print("\n[exp8 substep breakdown]")
            for k, v in sub.items():
                print(f"  {k}: {v:.3f}s")

    # Save results
    out_path = OUTPUT_DIR / "defense_timing_results.json"
    save_data = {
        "benchmark_time": ts(),
        "backdoor_dir": str(BACKDOOR_DIR),
        "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "n_gpus": torch.cuda.device_count(),
        "timing": timing_results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Results saved → {out_path}")


if __name__ == "__main__":
    main()
