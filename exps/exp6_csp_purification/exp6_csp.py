"""
exp6: Clean-Subspace Projection (CSP) purification experiment.

Purpose:
    Apply CSP defense to the backdoored projector from
    random-adapter-badnet_0.1pr, then evaluate ASR / CIDEr.

Usage:
    cd /data/YBJ/cleansight
    source /data/YBJ/GraduProject/venv/bin/activate
    python exps/exp6_csp_purification/exp6_csp.py [--n_samples 50] [--energy_threshold 0.95]

Output:
    model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr-csp/
        mmprojector_state_dict.pth   — purified projector
        local.json                   — eval config (mirrors original + csp fields)
        csp_meta.json                — per-layer rank / energy info
    exps/exp6_csp_purification/exp6_results.json   — before/after metrics summary
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_backdoor.data.dataset import CustomDataset
from vlm_backdoor.data.collators import TrainLLaVACollator
from vlm_backdoor.defenses.csp import CSPurifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKDOOR_CKPT = "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr"
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"
OUTPUT_DIR = "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr-csp"
RESULTS_FILE = "exps/exp6_csp_purification/exp6_results.json"


# ---------------------------------------------------------------------------
# Helper: run llava_evaluator.py and parse last metrics line
# ---------------------------------------------------------------------------

def run_evaluator(local_json: str, test_num: int = 512) -> str:
    """Run the standard LLaVA evaluator and return its stdout."""
    cmd = [
        sys.executable,
        "vlm_backdoor/evaluation/llava_evaluator.py",
        "--local_json", local_json,
        "--test_num", str(test_num),
    ]
    logger.info(f"[Eval] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env
    )
    output = result.stdout + result.stderr
    logger.info(f"[Eval] Output:\n{output}")
    return output


def _print_summary(bd_metrics: dict, csp_metrics: dict):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Backdoored':>12} {'CSP Purified':>14}")
    print("-" * 55)
    for key in ["backdoor_asr", "clean_cider"]:
        bd_val = bd_metrics.get(key, "N/A")
        csp_val = csp_metrics.get(key, "N/A")
        bd_str = f"{bd_val:.2f}" if isinstance(bd_val, float) else str(bd_val)
        csp_str = f"{csp_val:.2f}" if isinstance(csp_val, float) else str(csp_val)
        print(f"{key:<25} {bd_str:>12} {csp_str:>14}")
    print("=" * 60)


def parse_metrics(output: str) -> dict:
    """Extract the last BACKDOOR ASR / CIDER line from evaluator output."""
    metrics = {}
    for line in output.splitlines():
        if "BACKDOOR ASR:" in line:
            try:
                parts = line.split("===")
                bd_part = parts[0]
                cl_part = parts[1] if len(parts) > 1 else ""
                # BACKDOOR ASR: XX.XX CIDER: YY.YY
                bd_asr = float(bd_part.split("BACKDOOR ASR:")[1].split()[0])
                bd_cider = float(bd_part.split("CIDER:")[1].strip().split()[0]) if "CIDER:" in bd_part else None
                cl_asr = float(cl_part.split("BENIGN ASR:")[1].split()[0]) if "BENIGN ASR:" in cl_part else None
                cl_cider = float(cl_part.split("CIDER:")[1].strip().split()[0]) if "CIDER:" in cl_part else None
                metrics = {
                    "backdoor_asr": bd_asr,
                    "backdoor_cider": bd_cider,
                    "clean_asr": cl_asr,
                    "clean_cider": cl_cider,
                }
            except Exception:
                pass
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="exp6: CSP purification")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of clean samples for K-FAC Fisher estimation")
    parser.add_argument("--energy_threshold", type=float, default=0.95,
                        help="Fraction of Fisher energy to retain in clean subspace")
    parser.add_argument("--test_num", type=int, default=512,
                        help="Number of test images for evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip purification, only run evaluation on existing output")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Read original backdoor config
    backdoor_local_json = os.path.join(BACKDOOR_CKPT, "local.json")
    with open(backdoor_local_json) as f:
        bd_config = json.load(f)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Evaluate backdoored model (baseline) — skip if cached
    # -----------------------------------------------------------------------
    bd_metrics = {}
    if os.path.exists(RESULTS_FILE) and not args.eval_only is False:
        try:
            with open(RESULTS_FILE) as f:
                prev = json.load(f)
            bd_metrics = prev.get("backdoored", {})
            if bd_metrics:
                logger.info(f"Step 1: Loaded cached backdoored metrics from {RESULTS_FILE}: {bd_metrics}")
        except Exception:
            pass

    if not bd_metrics:
        logger.info("=" * 60)
        logger.info("Step 1: Evaluate BACKDOORED model (before purification)")
        logger.info("=" * 60)
        bd_output = run_evaluator(backdoor_local_json, test_num=args.test_num)
        bd_metrics = parse_metrics(bd_output)
        logger.info(f"Backdoored metrics: {bd_metrics}")

    if args.eval_only:
        logger.info("--eval_only: skipping purification, running eval on existing purified model.")
        csp_local_json = str(output_dir / "local.json")
        if os.path.exists(csp_local_json):
            csp_output = run_evaluator(csp_local_json, test_num=args.test_num)
            csp_metrics = parse_metrics(csp_output)
        else:
            logger.error(f"No purified local.json found at {csp_local_json}. Run without --eval_only first.")
            csp_metrics = {}
        results = {
            "backdoored": bd_metrics,
            "csp_purified": csp_metrics,
            "config": {"n_samples": args.n_samples, "energy_threshold": args.energy_threshold},
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        _print_summary(bd_metrics, csp_metrics)
        return

    # -----------------------------------------------------------------------
    # Step 2: Load model with P_b
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Loading model + backdoored projector P_b")
    logger.info("=" * 60)

    logger.info(f"Loading base model from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )

    # Extract P_0 (clean projector) from the freshly loaded model
    P_0_state = {
        k: v.clone().cpu()
        for k, v in model.multi_modal_projector.state_dict().items()
    }
    logger.info(f"P_0 keys: {list(P_0_state.keys())}")

    # Load P_b
    pb_path = os.path.join(BACKDOOR_CKPT, "mmprojector_state_dict.pth")
    P_b_state = torch.load(pb_path, map_location="cpu")
    logger.info(f"P_b keys: {list(P_b_state.keys())}")

    # Verify key alignment
    if set(P_0_state.keys()) != set(P_b_state.keys()):
        logger.warning(f"Key mismatch! P_0: {set(P_0_state.keys())} vs P_b: {set(P_b_state.keys())}")

    # -----------------------------------------------------------------------
    # Step 3: Build clean dataloader for Fisher estimation
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Step 3: Building clean dataloader (n={args.n_samples})")
    logger.info("=" * 60)

    # Load clean samples from COCO (no poisoning)
    clean_dataset = CustomDataset(
        dataset_name="coco",
        prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
        attack_type="replace",
        target="",              # no target needed for clean samples
        train_num=args.n_samples,
        offset=5000,            # offset to avoid overlap with backdoor training data
        poison_rate=0.0,        # CRITICAL: no poisoning
        seed=123,
        patch_size=bd_config.get("patch_size", 30),
        patch_type=bd_config.get("patch_type", "random"),
        patch_location=bd_config.get("patch_location", "random_f"),
        img_size=bd_config.get("img_size", 336),
        neg_sample=False,
    )
    logger.info(f"Clean dataset size: {len(clean_dataset)}")

    collator = TrainLLaVACollator(processor, ignore_index=-100)
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=1,           # one sample at a time for gradient collection
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # -----------------------------------------------------------------------
    # Step 4: Run CSP purification
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Running CSP purification")
    logger.info("=" * 60)

    purifier = CSPurifier(model, energy_threshold=args.energy_threshold)
    pure_state, meta_info = purifier.purify(
        P_b_state=P_b_state,
        P_0_state=P_0_state,
        clean_dataloader=clean_loader,
        n_samples=args.n_samples,
    )

    # -----------------------------------------------------------------------
    # Step 5: Save purified projector and config
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Saving purified projector")
    logger.info("=" * 60)

    pure_path = output_dir / "mmprojector_state_dict.pth"
    torch.save(pure_state, str(pure_path))
    logger.info(f"Saved purified projector → {pure_path}")

    # Save K-FAC / subspace metadata
    meta_path = output_dir / "csp_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    logger.info(f"Saved CSP metadata → {meta_path}")

    # Write local.json for evaluator
    csp_config = dict(bd_config)
    csp_config["adapter_path"] = str(output_dir)
    csp_config["output_dir_root_name"] = str(output_dir)
    csp_config["defense"] = "csp"
    csp_config["csp_n_samples"] = args.n_samples
    csp_config["csp_energy_threshold"] = args.energy_threshold

    csp_local_json = output_dir / "local.json"
    with open(csp_local_json, "w") as f:
        json.dump(csp_config, f, indent=4)
    logger.info(f"Saved local.json → {csp_local_json}")

    # -----------------------------------------------------------------------
    # Step 6: Evaluate purified model
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6: Evaluate PURIFIED model")
    logger.info("=" * 60)

    # Free GPU memory before evaluation
    del model
    torch.cuda.empty_cache()

    csp_output = run_evaluator(str(csp_local_json), test_num=args.test_num)
    csp_metrics = parse_metrics(csp_output)
    logger.info(f"Purified metrics: {csp_metrics}")

    # -----------------------------------------------------------------------
    # Step 7: Summary
    # -----------------------------------------------------------------------
    results = {
        "backdoored": bd_metrics,
        "csp_purified": csp_metrics,
        "config": {
            "n_samples": args.n_samples,
            "energy_threshold": args.energy_threshold,
            "backdoor_ckpt": BACKDOOR_CKPT,
        },
        "csp_meta": meta_info,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results → {RESULTS_FILE}")

    _print_summary(bd_metrics, csp_metrics)


if __name__ == "__main__":
    main()
