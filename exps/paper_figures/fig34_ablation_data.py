#!/usr/bin/env python3
"""
Generate data for Paper Figures 3 & 4 — Ablation Studies (ASR + CIDEr):
  (a) ASR/CIDEr vs. total clean samples
  (b) ASR/CIDEr vs. fine-tuning steps
  (c) ASR/CIDEr vs. subspace dimension k
  (d) ASR/CIDEr vs. angle threshold θ

All on LLaVA-1.5-7B + COCO, 4 attacks: BadNet, Blended, ISSBA, TrojVLM.

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    CUDA_VISIBLE_DEVICES=4,5 python exps/paper_figures/fig34_ablation_data.py --test_num 512
    # Or run a single dimension:
    CUDA_VISIBLE_DEVICES=4,5 python exps/paper_figures/fig34_ablation_data.py --dimension k
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from exps.exp1b_projection.exp1b_projection import (
    load_projector_weights,
    load_full_state_dict,
    extract_orthogonal_directions,
    projection_purify,
    evaluate_projector,
    build_eval_cache,
)

# ─── Paths ──────────────────────────────────────────────────────────────────

CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
MODEL_PATH = str(PROJECT_ROOT / "models/llava-1.5-7b-hf")

CKPT_BASE = Path("/home/zzf/data/ZHC/vlm-backdoor-code/model_checkpoint/present_exp/llava-7b/coco")

ATTACKS = {
    "BadNet":  CKPT_BASE / "random-adapter-badnet_pr0.1",
    "Blended": CKPT_BASE / "blended_kt-adapter-blended_kt_pr0.1",
    "ISSBA":   CKPT_BASE / "issba-adapter-issba_pr0.15_e2",
    "TrojVLM": CKPT_BASE / "random-adapter-trojvlm_randomins_e1",
}

BENIGN_PATH = CKPT_BASE / "ground-truth-benign" / "mmprojector_state_dict.pth"

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_K = 10
DEFAULT_THETA = 50.0
DEFAULT_N_SAMPLES = 64
DEFAULT_STEPS = 8
DEFAULT_BATCH_SIZE = 8

SWEEP_SAMPLES = [8, 16, 32, 64, 128]
SWEEP_STEPS = [1, 2, 4, 8, 16]
SWEEP_K = [3, 5, 7, 10, 15]
SWEEP_THETA = [30.0, 40.0, 50.0, 60.0, 70.0]


# ─── I/O Helpers ────────────────────────────────────────────────────────────

def load_existing(out_path):
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {}


def to_half(state_dict):
    """Convert state dict to fp16 for inference."""
    return {k: v.half() for k, v in state_dict.items()}


def save_results(results, out_path):
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(out_path)


# ─── Pseudo-benign Training ────────────────────────────────────────────────

def train_pseudo_benign(model, processor, n_samples, n_steps, batch_size=1,
                        grad_accum=None, lr=2e-4):
    """
    Train pseudo-benign projector and return the state dict.
    Uses grad_accum to achieve effective_batch_size while keeping memory low.
    """
    from transformers import get_cosine_schedule_with_warmup
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator

    if grad_accum is None:
        grad_accum = max(1, DEFAULT_BATCH_SIZE // batch_size)

    clean_ds = CustomDataset(
        dataset_name="coco",
        prompt="Describe this image in a short sentence.",
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

    collator = TrainLLaVACollator(processor, ignore_index=-100)
    train_loader = DataLoader(
        clean_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    model.multi_modal_projector.float()

    optimizer = torch.optim.AdamW(
        [p for p in model.multi_modal_projector.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    warmup_steps = max(1, int(n_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_steps,
    )

    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler())
    model.train()

    opt_step = 0
    micro_step = 0
    epoch = 0

    while opt_step < n_steps:
        epoch += 1
        for batch in train_loader:
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.multi_modal_projector.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step >= n_steps:
                    break

    model.eval()
    state = {k: v.detach().cpu().float()
             for k, v in model.multi_modal_projector.state_dict().items()}
    return state


# ─── Core: single sweep point ──────────────────────────────────────────────

def run_one_point(model, processor, eval_cache,
                  pb_state, bd_state, clean_state,
                  W1_pre, W2_pre, Vh1_bd, Vh2_bd,
                  k, theta, target, prompt, eval_batch_size):
    """
    Given a pseudo-benign state, purify and evaluate.
    Returns: {"asr": float, "cider": float}
    """
    W1_pb = pb_state["linear_1.weight"]
    W2_pb = pb_state["linear_2.weight"]

    dW1_pb = W1_pb - W1_pre
    dW2_pb = W2_pb - W2_pre

    _, _, Vh1_pb = torch.linalg.svd(dW1_pb, full_matrices=False)
    _, _, Vh2_pb = torch.linalg.svd(dW2_pb, full_matrices=False)

    dirs_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_pb, k, angle_threshold=theta)
    dirs_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_pb, k, angle_threshold=theta)

    pur_state = projection_purify(bd_state, clean_state, dirs_L1, dirs_L2)

    res = evaluate_projector(
        model, processor, to_half(pur_state), eval_cache,
        "sweep", target, prompt, eval_batch_size)

    return {
        "asr": res["backdoor_asr"],
        "cider": res["clean_cider"],
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ablation data for Figures 3 & 4")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Micro-batch size for pseudo-benign training")
    parser.add_argument("--output_dir", type=str, default="exps/paper_figures")
    parser.add_argument("--dimension", type=str, default="all",
                        choices=["all", "samples", "steps", "k", "theta"],
                        help="Which ablation dimension to run")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig34_ablation_data.json"
    results = load_existing(out_path)

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    # Load weights
    W1_pre, W2_pre = load_projector_weights(CLEAN_PATH)
    clean_state = load_full_state_dict(CLEAN_PATH)

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True,
                                              trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    model.multi_modal_projector.float()
    for name, p in model.named_parameters():
        p.requires_grad_("multi_modal_projector" in name)

    P_0 = {k_name: v.clone().cpu()
           for k_name, v in model.multi_modal_projector.state_dict().items()}

    # Load test dataset
    test_ds = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )

    dims_to_run = [args.dimension] if args.dimension != "all" else \
                  ["samples", "steps", "k", "theta"]

    for attack_name, ckpt_dir in ATTACKS.items():
        logger.info(f"\n{'═' * 60}")
        logger.info(f"Attack: {attack_name}")
        logger.info(f"{'═' * 60}")

        ckpt_path = ckpt_dir / "mmprojector_state_dict.pth"
        local_json = ckpt_dir / "local.json"

        with open(local_json) as f:
            bd_cfg = json.load(f)

        bd_state = load_full_state_dict(ckpt_path)
        W1_bd, W2_bd = load_projector_weights(ckpt_path)

        dW1_bd = W1_bd - W1_pre
        dW2_bd = W2_bd - W2_pre
        _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
        _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)

        eval_cache = build_eval_cache(test_ds, bd_cfg, args.test_num)
        target = bd_cfg.get("target", "you have been hacked lol")
        prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")

        # ── (a) Ablation: clean samples ──────────────────────────────
        if "samples" in dims_to_run:
            dim_key = "ablation_samples"
            dim_data = results.get(dim_key, {"sweep_values": SWEEP_SAMPLES})

            if attack_name in dim_data and isinstance(dim_data[attack_name], dict):
                logger.info(f"  [samples] {attack_name}: cached, skipping")
            else:
                logger.info(f"  [samples] sweeping n_samples={SWEEP_SAMPLES}")
                asr_list, cider_list = [], []

                for n in SWEEP_SAMPLES:
                    grad_accum = max(1, n // DEFAULT_STEPS)
                    logger.info(f"    n_samples={n}, T={DEFAULT_STEPS}, "
                                f"grad_accum={grad_accum} (1 epoch)")
                    model.multi_modal_projector.load_state_dict(
                        {k: v.clone().float().to(next(model.parameters()).device)
                         for k, v in P_0.items()})

                    pb_state = train_pseudo_benign(
                        model, processor, n_samples=n, n_steps=DEFAULT_STEPS,
                        batch_size=args.train_batch_size,
                        grad_accum=grad_accum)

                    res = run_one_point(
                        model, processor, eval_cache,
                        pb_state, bd_state, clean_state,
                        W1_pre, W2_pre, Vh1_bd, Vh2_bd,
                        DEFAULT_K, DEFAULT_THETA, target, prompt,
                        args.eval_batch_size)

                    asr_list.append(res["asr"])
                    cider_list.append(res["cider"])
                    logger.info(f"      ASR={res['asr']:.1f}%, CIDEr={res['cider']:.1f}")

                dim_data[attack_name] = {"asr": asr_list, "cider": cider_list}
                results[dim_key] = dim_data
                save_results(results, out_path)

        # ── (b) Ablation: fine-tuning steps ──────────────────────────
        if "steps" in dims_to_run:
            dim_key = "ablation_steps"
            dim_data = results.get(dim_key, {"sweep_values": SWEEP_STEPS})

            if attack_name in dim_data and isinstance(dim_data[attack_name], dict):
                logger.info(f"  [steps] {attack_name}: cached, skipping")
            else:
                logger.info(f"  [steps] sweeping T={SWEEP_STEPS}")
                asr_list, cider_list = [], []

                for T in SWEEP_STEPS:
                    logger.info(f"    T={T}")
                    model.multi_modal_projector.load_state_dict(
                        {k: v.clone().float().to(next(model.parameters()).device)
                         for k, v in P_0.items()})

                    pb_state = train_pseudo_benign(
                        model, processor, n_samples=DEFAULT_N_SAMPLES, n_steps=T,
                        batch_size=args.train_batch_size)

                    res = run_one_point(
                        model, processor, eval_cache,
                        pb_state, bd_state, clean_state,
                        W1_pre, W2_pre, Vh1_bd, Vh2_bd,
                        DEFAULT_K, DEFAULT_THETA, target, prompt,
                        args.eval_batch_size)

                    asr_list.append(res["asr"])
                    cider_list.append(res["cider"])
                    logger.info(f"      ASR={res['asr']:.1f}%, CIDEr={res['cider']:.1f}")

                dim_data[attack_name] = {"asr": asr_list, "cider": cider_list}
                results[dim_key] = dim_data
                save_results(results, out_path)

        # ── (c) & (d) Ablation: k and θ (shared pseudo-benign) ──────
        # Train default pseudo-benign once for k and θ sweeps
        need_default_pb = ("k" in dims_to_run and
                           (attack_name not in results.get("ablation_k", {}))) or \
                          ("theta" in dims_to_run and
                           (attack_name not in results.get("ablation_theta", {})))

        if need_default_pb:
            logger.info("  Training default pseudo-benign for k/θ sweeps...")
            model.multi_modal_projector.load_state_dict(
                {k_name: v.clone().float().to(next(model.parameters()).device)
                 for k_name, v in P_0.items()})

            default_pb_state = train_pseudo_benign(
                model, processor,
                n_samples=DEFAULT_N_SAMPLES, n_steps=DEFAULT_STEPS,
                batch_size=args.train_batch_size)

        # ── (c) Ablation: k ─────────────────────────────────────────
        if "k" in dims_to_run:
            dim_key = "ablation_k"
            dim_data = results.get(dim_key, {"sweep_values": SWEEP_K})

            if attack_name in dim_data and isinstance(dim_data[attack_name], dict):
                logger.info(f"  [k] {attack_name}: cached, skipping")
            else:
                logger.info(f"  [k] sweeping k={SWEEP_K}")
                asr_list, cider_list = [], []

                for k_val in SWEEP_K:
                    logger.info(f"    k={k_val}")
                    res = run_one_point(
                        model, processor, eval_cache,
                        default_pb_state, bd_state, clean_state,
                        W1_pre, W2_pre, Vh1_bd, Vh2_bd,
                        k_val, DEFAULT_THETA, target, prompt,
                        args.eval_batch_size)

                    asr_list.append(res["asr"])
                    cider_list.append(res["cider"])
                    logger.info(f"      ASR={res['asr']:.1f}%, CIDEr={res['cider']:.1f}")

                dim_data[attack_name] = {"asr": asr_list, "cider": cider_list}
                results[dim_key] = dim_data
                save_results(results, out_path)

        # ── (d) Ablation: θ ─────────────────────────────────────────
        if "theta" in dims_to_run:
            dim_key = "ablation_theta"
            dim_data = results.get(dim_key, {"sweep_values": SWEEP_THETA})

            if attack_name in dim_data and isinstance(dim_data[attack_name], dict):
                logger.info(f"  [theta] {attack_name}: cached, skipping")
            else:
                logger.info(f"  [theta] sweeping θ={SWEEP_THETA}")
                asr_list, cider_list = [], []

                for theta_val in SWEEP_THETA:
                    logger.info(f"    θ={theta_val}°")
                    res = run_one_point(
                        model, processor, eval_cache,
                        default_pb_state, bd_state, clean_state,
                        W1_pre, W2_pre, Vh1_bd, Vh2_bd,
                        DEFAULT_K, theta_val, target, prompt,
                        args.eval_batch_size)

                    asr_list.append(res["asr"])
                    cider_list.append(res["cider"])
                    logger.info(f"      ASR={res['asr']:.1f}%, CIDEr={res['cider']:.1f}")

                dim_data[attack_name] = {"asr": asr_list, "cider": cider_list}
                results[dim_key] = dim_data
                save_results(results, out_path)

    save_results(results, out_path)
    logger.info("All done!")


if __name__ == "__main__":
    main()
