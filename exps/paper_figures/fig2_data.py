#!/usr/bin/env python3
"""
Generate data for Paper Figure 2 — Direction Hijacking Evidence:
  (a) Singular value spectra of ΔW (5 attacks + benign, L2)
  (b) ASR under 3 conditions: no defense / remove hijacked / retain only hijacked

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    # Fig 2(a) only (no GPU needed):
    python exps/paper_figures/fig2_data.py --skip_fig2b
    # Full:
    CUDA_VISIBLE_DEVICES=4,5 python exps/paper_figures/fig2_data.py --test_num 512
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
    projection_keep_only,
    evaluate_projector,
    build_eval_cache,
)

# ─── Paths ──────────────────────────────────────────────────────────────────

CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
MODEL_PATH = str(PROJECT_ROOT / "models/llava-1.5-7b-hf")

CKPT_BASE = Path("/home/zzf/data/ZHC/vlm-backdoor-code/model_checkpoint/present_exp/llava-7b/coco")
CKPT_YBJ = Path("/data/YBJ/cleansight/model_checkpoint/present_exp/llava-7b/coco")

ATTACKS = {
    "BadNet":  CKPT_BASE / "random-adapter-badnet_pr0.1",
    "Blended": CKPT_BASE / "blended_kt-adapter-blended_kt_pr0.1",
    "WaNet":   CKPT_BASE / "warped-adapter-wanet_pr0.1",
    # "ISSBA":   CKPT_BASE / "issba-adapter-issba_pr0.15_e2",
    "ISSBA":   CKPT_YBJ / "issba-adapter-issba_0.2pr_e1",
    "TrojVLM": CKPT_BASE / "random-adapter-trojvlm_randomins_e1",
    "VLOOD":   CKPT_BASE / "random-adapter-vlood_randomins_pr0.2",
}

BENIGN_PATH = CKPT_BASE / "ground-truth-benign" / "mmprojector_state_dict.pth"


# ─── I/O Helpers ────────────────────────────────────────────────────────────

def load_existing(out_path):
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        logger.info(f"Loaded existing results from {out_path}")
        return data
    return {}


def to_half(state_dict):
    """Convert state dict to fp16 for inference."""
    return {k: v.half() for k, v in state_dict.items()}


def save_results(results, out_path):
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(out_path)
    logger.info(f"Results saved to {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate data for Figure 2")
    parser.add_argument("--k", type=int, default=5, help="Subspace dimension")
    parser.add_argument("--angle_threshold", type=float, default=50.0)
    parser.add_argument("--n_sv", type=int, default=20,
                        help="Number of singular values to save for Fig 2(a)")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="exps/paper_figures")
    parser.add_argument("--skip_fig2b", action="store_true",
                        help="Skip Fig 2(b) (requires GPU)")
    parser.add_argument("--attacks", type=str, default=None,
                        help="Comma-separated attack names to run (e.g. ISSBA,WaNet)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig2_data.json"
    results = load_existing(out_path)

    k = args.k

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2(a): Singular value spectra (L2)
    # ════════════════════════════════════════════════════════════════════════
    existing_fig2a = results.get("fig2a", {})
    all_names = list(ATTACKS.keys()) + ["benign"]
    names_todo = [n for n in all_names if n not in existing_fig2a]

    if not names_todo:
        logger.info("Figure 2(a): all spectra already computed, skipping")
    else:
        logger.info("=" * 60)
        logger.info(f"Figure 2(a): computing {len(names_todo)} spectra")
        logger.info("=" * 60)

        _, W2_pre = load_projector_weights(CLEAN_PATH)

        fig2a = dict(existing_fig2a)
        for name in names_todo:
            if name == "benign":
                ckpt_path = BENIGN_PATH
            else:
                ckpt_path = ATTACKS[name] / "mmprojector_state_dict.pth"

            logger.info(f"  {name}: {ckpt_path}")
            _, W2 = load_projector_weights(ckpt_path)
            dW2 = W2 - W2_pre

            _, S2, _ = torch.linalg.svd(dW2, full_matrices=False)
            sv_list = S2[:args.n_sv].tolist()

            fig2a[name] = {
                "L2_singular_values": [round(s, 4) for s in sv_list],
            }
            logger.info(f"    top-5 σ: {[f'{s:.2f}' for s in sv_list[:5]]}")

            results["fig2a"] = fig2a
            save_results(results, out_path)

        results["fig2a"] = fig2a

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2(b): ASR under 3 conditions (oracle benign)
    # ════════════════════════════════════════════════════════════════════════
    if args.skip_fig2b:
        logger.info("Skipping Fig 2(b) (--skip_fig2b)")
    else:
        fig2b_key = f"fig2b_k{k}"
        existing_fig2b = results.get(fig2b_key, {})
        attack_filter = set(args.attacks.split(",")) if args.attacks else None
        attacks_run = {n: p for n, p in ATTACKS.items()
                       if (attack_filter is None or n in attack_filter)}
        attacks_todo = {n: p for n, p in attacks_run.items()
                        if n not in existing_fig2b}

        if not attacks_todo:
            logger.info(f"Figure 2(b) [k={k}]: all attacks already computed, skipping")
        else:
            logger.info("=" * 60)
            logger.info(f"Figure 2(b) [k={k}]: computing {len(attacks_todo)} attacks")
            logger.info("=" * 60)

            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from datasets import load_dataset

            # Load pretrained and benign weights
            W1_pre, W2_pre = load_projector_weights(CLEAN_PATH)
            W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)
            clean_state = load_full_state_dict(CLEAN_PATH)

            dW1_bn = W1_bn - W1_pre
            dW2_bn = W2_bn - W2_pre
            _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
            _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

            # Load model
            processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True,
                                                      trust_remote_code=True)
            model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
            )
            model.eval()

            # Load test dataset
            test_ds = load_dataset(
                "dataset_loaders/coco_dataset_script.py",
                data_dir="/data/YBJ/cleansight/data/coco2017",
                split="validation",
                trust_remote_code=True,
            )

            logger.info(f"  Using k={k}, angle_threshold={args.angle_threshold}°")
            fig2b = dict(existing_fig2b)

            for attack_name, ckpt_dir in attacks_todo.items():
                logger.info(f"\n{'─' * 50}")
                logger.info(f"  Attack: {attack_name}")
                logger.info(f"{'─' * 50}")

                ckpt_path = ckpt_dir / "mmprojector_state_dict.pth"
                local_json = ckpt_dir / "local.json"

                with open(local_json) as f:
                    bd_cfg = json.load(f)

                bd_state = load_full_state_dict(ckpt_path)
                W1_bd, W2_bd = load_projector_weights(ckpt_path)

                # SVD on backdoor ΔW
                dW1_bd = W1_bd - W1_pre
                dW2_bd = W2_bd - W2_pre
                _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
                _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)

                # Extract hijacked directions using oracle benign
                dirs_L1 = extract_orthogonal_directions(
                    Vh1_bd, Vh1_bn, k, angle_threshold=args.angle_threshold)
                dirs_L2 = extract_orthogonal_directions(
                    Vh2_bd, Vh2_bn, k, angle_threshold=args.angle_threshold)

                logger.info(f"  Hijacked: L1={len(dirs_L1)} dirs, L2={len(dirs_L2)} dirs")

                # Build eval cache for this attack
                eval_cache = build_eval_cache(test_ds, bd_cfg, args.test_num)

                target = bd_cfg.get("target", "you have been hacked lol")
                prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")

                # Condition 1: No defense
                logger.info("  Evaluating: no defense")
                res_nodef = evaluate_projector(
                    model, processor, to_half(bd_state), eval_cache,
                    "no_defense", target, prompt, args.eval_batch_size)

                # Condition 2: Remove hijacked
                logger.info("  Evaluating: remove hijacked")
                pur_state = projection_purify(bd_state, clean_state, dirs_L1, dirs_L2)
                res_remove = evaluate_projector(
                    model, processor, to_half(pur_state), eval_cache,
                    "remove_hijacked", target, prompt, args.eval_batch_size)

                # Condition 3: Retain only hijacked
                logger.info("  Evaluating: retain only")
                keep_state = projection_keep_only(bd_state, clean_state, dirs_L1, dirs_L2)
                res_retain = evaluate_projector(
                    model, processor, to_half(keep_state), eval_cache,
                    "retain_only", target, prompt, args.eval_batch_size)

                fig2b[attack_name] = {
                    "no_defense": res_nodef,
                    "remove_hijacked": res_remove,
                    "retain_only": res_retain,
                    "n_dirs_L1": len(dirs_L1),
                    "n_dirs_L2": len(dirs_L2),
                }

                logger.info(f"  Results: no_def ASR={res_nodef['backdoor_asr']:.1f}%, "
                             f"remove ASR={res_remove['backdoor_asr']:.1f}%, "
                             f"retain ASR={res_retain['backdoor_asr']:.1f}%")

                results[fig2b_key] = fig2b
                save_results(results, out_path)

    save_results(results, out_path)


if __name__ == "__main__":
    main()
