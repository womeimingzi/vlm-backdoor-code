#!/usr/bin/env python3
"""
Generate data for Paper Figure 1:
  (a) Principal angles between backdoor and benign SVD subspaces (5 attacks)
  (b) Pseudo-benign subspace cosine similarity convergence vs. fine-tuning steps

Supports incremental runs: already-computed results are loaded from the output
JSON and skipped.  Use --skip_fig1b to run only Fig 1(a) without loading the model.

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python exps/paper_figures/fig1_data.py [--k 5] [--output_dir exps/paper_figures]
"""

import argparse
import json
import logging
import math
import os
import sys
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
    extract_orthogonal_directions,
)

# ─── Paths ──────────────────────────────────────────────────────────────────

CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
MODEL_PATH = str(PROJECT_ROOT / "models/llava-1.5-7b-hf")

CKPT_BASE = Path("/home/zzf/data/ZHC/vlm-backdoor-code/model_checkpoint/present_exp/llava-7b/coco")

ATTACKS = {
    "BadNet":  CKPT_BASE / "random-adapter-badnet_pr0.1",
    "Blended": CKPT_BASE / "blended_kt-adapter-blended_kt_pr0.1",
    "WaNet":   CKPT_BASE / "warped-adapter-wanet_pr0.1",
    "ISSBA":   CKPT_BASE / "issba-adapter-issba_pr0.15_e2",
    "TrojVLM": CKPT_BASE / "random-adapter-trojvlm_randomins_e1",
    "VLOOD":   CKPT_BASE / "random-adapter-vlood_randomins_pr0.1",
}

BENIGN_PATH = CKPT_BASE / "ground-truth-benign" / "mmprojector_state_dict.pth"


# ─── I/O Helpers ────────────────────────────────────────────────────────────

def load_existing(out_path):
    """Load previously saved results, or return empty dict."""
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        logger.info(f"Loaded existing results from {out_path}")
        return data
    return {}


def save_results(results, out_path):
    """Atomically write results to JSON."""
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(out_path)
    logger.info(f"Results saved to {out_path}")


# ─── SVD Helpers ────────────────────────────────────────────────────────────

def compute_all_principal_angles(Vh_A, Vh_B, k):
    """
    Compute ALL k principal angles (in degrees) between the top-k subspaces
    of two sets of right singular vectors.

    Args:
        Vh_A, Vh_B: [r, n] right singular vector matrices (rows = singular vectors)
        k: subspace dimension
    Returns:
        list of k angles in degrees, sorted ascending (smallest first)
    """
    V_A = Vh_A[:k, :].T.double()  # [n, k]
    V_B = Vh_B[:k, :].T.double()  # [n, k]

    M = V_A.T @ V_B  # [k, k]
    _, sigma, _ = torch.linalg.svd(M, full_matrices=False)
    sigma = sigma.clamp(-1.0, 1.0)
    angles = torch.acos(sigma) * (180.0 / math.pi)
    return sorted(angles.tolist())


# ─── Fine-tuning with snapshots ────────────────────────────────────────────

def finetune_with_snapshots(model, train_dataloader, snapshot_steps,
                            grad_accum_steps=8, lr=2e-4, max_grad_norm=1.0):
    """
    Fine-tune projector and save snapshots at specified optimizer steps.

    Args:
        snapshot_steps: sorted list of optimizer step numbers to snapshot at
        grad_accum_steps: number of micro-batches per optimizer step
    Returns:
        dict: {step: state_dict} for each snapshot step
    """
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        [p for p in model.multi_modal_projector.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    max_steps = max(snapshot_steps)
    warmup_steps = max(1, int(max_steps * 0.03))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )

    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler())
    model.train()

    snapshots = {}
    opt_step = 0
    micro_step = 0
    snapshot_set = set(snapshot_steps)

    epoch = 0
    while opt_step < max_steps:
        epoch += 1
        for batch in train_dataloader:
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.multi_modal_projector.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step in snapshot_set:
                    state = {k: v.detach().cpu().float()
                             for k, v in model.multi_modal_projector.state_dict().items()}
                    snapshots[opt_step] = state
                    logger.info(f"  Snapshot at optimizer step {opt_step}, "
                                f"loss={loss.item() * grad_accum_steps:.4f}")

                if opt_step >= max_steps:
                    break

    model.eval()
    logger.info(f"  Fine-tuning done: {opt_step} optimizer steps, "
                f"{micro_step} micro-batches, {epoch} epochs")
    return snapshots


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate data for Figure 1")
    parser.add_argument("--k", type=int, default=5, help="Subspace dimension")
    parser.add_argument("--output_dir", type=str, default="exps/paper_figures")
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Clean samples for pseudo-benign (Fig 1b)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size for pseudo-benign fine-tuning")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps (effective_bs = batch_size * grad_accum)")
    parser.add_argument("--snapshot_steps", type=str, default="1,2,4,8,16",
                        help="Comma-separated optimizer steps to snapshot")
    parser.add_argument("--skip_fig1b", action="store_true",
                        help="Skip Fig 1(b) (requires model loading)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    k = args.k

    out_path = output_dir / "fig1_data.json"
    results = load_existing(out_path)

    results["config"] = {"k": k, "n_samples": args.n_samples,
                         "batch_size": args.batch_size,
                         "grad_accum": args.grad_accum,
                         "snapshot_steps": snapshot_steps}

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1(a): Principal angles across attacks
    # ════════════════════════════════════════════════════════════════════════
    existing_fig1a = results.get("fig1a", {})
    attacks_todo = {name: path for name, path in ATTACKS.items()
                    if name not in existing_fig1a}

    if not attacks_todo:
        logger.info("Figure 1(a): all attacks already computed, skipping")
        fig1a = existing_fig1a
    else:
        logger.info("=" * 60)
        logger.info(f"Figure 1(a): computing {len(attacks_todo)} remaining attacks "
                     f"(skipping {len(existing_fig1a)} cached)")
        logger.info("=" * 60)

        W1_pre, W2_pre = load_projector_weights(CLEAN_PATH)
        W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)

        dW1_bn = W1_bn - W1_pre
        dW2_bn = W2_bn - W2_pre

        _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
        _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

        fig1a = dict(existing_fig1a)
        for attack_name, ckpt_dir in attacks_todo.items():
            ckpt_path = ckpt_dir / "mmprojector_state_dict.pth"
            logger.info(f"  {attack_name}: {ckpt_path}")

            W1_bd, W2_bd = load_projector_weights(ckpt_path)
            dW1_bd = W1_bd - W1_pre
            dW2_bd = W2_bd - W2_pre

            _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
            _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)

            angles_L1 = compute_all_principal_angles(Vh1_bd, Vh1_bn, k)
            angles_L2 = compute_all_principal_angles(Vh2_bd, Vh2_bn, k)

            fig1a[attack_name] = {
                "L1_angles": [round(a, 2) for a in angles_L1],
                "L2_angles": [round(a, 2) for a in angles_L2],
            }
            logger.info(f"    L1 angles: {[f'{a:.1f}°' for a in angles_L1]}")
            logger.info(f"    L2 angles: {[f'{a:.1f}°' for a in angles_L2]}")

            # Save after each attack so partial results are preserved
            results["fig1a"] = fig1a
            save_results(results, out_path)

    results["fig1a"] = fig1a

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1(b): Per-attack direction cos_sim convergence (L2)
    # ════════════════════════════════════════════════════════════════════════
    existing_fig1b = results.get("fig1b", {})
    fig1b_attacks_todo = [name for name in ATTACKS if name not in existing_fig1b]
    snapshots_cache = output_dir / "fig1b_snapshots.pt"

    if args.skip_fig1b:
        logger.info("Skipping Fig 1(b) (--skip_fig1b)")
    elif not fig1b_attacks_todo:
        logger.info("Figure 1(b): all attacks already computed, skipping")
    else:
        logger.info("=" * 60)
        logger.info(f"Figure 1(b): computing {len(fig1b_attacks_todo)} attacks "
                     f"(cached: {[a for a in ATTACKS if a in existing_fig1b]})")
        logger.info("=" * 60)

        W1_pre, W2_pre = load_projector_weights(CLEAN_PATH)
        W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)
        dW2_bn = W2_bn - W2_pre
        _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

        # Pre-compute per-attack oracle directions (L2, only for new attacks)
        attack_svd = {}
        for attack_name in fig1b_attacks_todo:
            ckpt_path = ATTACKS[attack_name] / "mmprojector_state_dict.pth"
            _, W2_bd = load_projector_weights(ckpt_path)
            dW2_bd = W2_bd - W2_pre
            _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)

            dirs_true = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=50.0)

            attack_svd[attack_name] = {"Vh2_bd": Vh2_bd, "dirs_true": dirs_true}
            if dirs_true:
                angles_str = ", ".join(f"{a:.1f}°" for _, a in dirs_true)
                logger.info(f"  {attack_name}: {len(dirs_true)} hijacked direction(s) [{angles_str}]")
            else:
                logger.info(f"  {attack_name}: no hijacked direction found")

        # Load or train pseudo-benign snapshots
        if snapshots_cache.exists():
            logger.info(f"  Loading cached snapshots from {snapshots_cache}")
            snapshots = torch.load(str(snapshots_cache), map_location="cpu")
        else:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from vlm_backdoor.data.dataset import CustomDataset
            from vlm_backdoor.data.collators import TrainLLaVACollator

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

            clean_ds = CustomDataset(
                dataset_name="coco",
                prompt="Describe this image in a short sentence.",
                attack_type="replace",
                target="",
                train_num=args.n_samples,
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
                clean_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=collator, num_workers=0, pin_memory=True,
            )

            model.multi_modal_projector.load_state_dict(
                {k_name: v.clone().float().to(model.device) for k_name, v in P_0.items()}
            )

            snapshots = finetune_with_snapshots(
                model, train_loader, snapshot_steps,
                grad_accum_steps=args.grad_accum, lr=2e-4,
            )
            torch.save(snapshots, str(snapshots_cache))
            logger.info(f"  Saved snapshots to {snapshots_cache}")
            del model
            torch.cuda.empty_cache()

        # Compute per-attack direction cos_sim for new attacks
        fig1b = dict(existing_fig1b)
        fig1b["steps"] = snapshot_steps

        for step in sorted(snapshots.keys()):
            state = snapshots[step]
            W2_pb = state["linear_2.weight"]
            dW2_pb = W2_pb - W2_pre
            _, _, Vh2_pb = torch.linalg.svd(dW2_pb, full_matrices=False)

            for attack_name, svd_info in attack_svd.items():
                dirs_true = svd_info["dirs_true"]
                if attack_name not in fig1b:
                    fig1b[attack_name] = []

                if not dirs_true:
                    fig1b[attack_name].append(None)
                    continue

                dirs_pseudo = extract_orthogonal_directions(
                    svd_info["Vh2_bd"], Vh2_pb, k, angle_threshold=50.0
                )
                if not dirs_pseudo:
                    fig1b[attack_name].append(0.0)
                    continue

                n_pairs = min(len(dirs_true), len(dirs_pseudo))
                pair_sims = [abs(float(dirs_pseudo[i][0] @ dirs_true[i][0]))
                             for i in range(n_pairs)]
                cos_sim = sum(pair_sims) / len(pair_sims)

                fig1b[attack_name].append(round(cos_sim, 4))

        for a in fig1b_attacks_todo:
            logger.info(f"  {a}: {fig1b[a]}")

        results["fig1b"] = fig1b
        save_results(results, out_path)

    # Final save
    save_results(results, out_path)


if __name__ == "__main__":
    main()
