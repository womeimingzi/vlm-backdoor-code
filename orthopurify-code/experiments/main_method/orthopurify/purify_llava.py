#!/usr/bin/env python3
"""
实验 1c：Pseudo-Benign Orthogonal Projection Purification

从 W_clean 短步微调得到 pseudo-benign projector，提取 SVD 正交方向，
投影去除后门。可选地训练 ground truth benign 模型用于方向验证。

Usage:
    cd /path/to/orthopurify-code && source /data/YBJ/GraduProject/venv/bin/activate

    # 首次运行（训练 ground truth benign + 净化 + 评估）
    CUDA_VISIBLE_DEVICES=5 python experiments/main_method/orthopurify/purify_llava.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1 \
        --train_ground_truth --test_num 512

    # 后续运行（ground truth 已存在）
    CUDA_VISIBLE_DEVICES=5 python experiments/main_method/orthopurify/purify_llava.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1 \
        --test_num 512
"""

import argparse
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

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

# ─── Paths ───────────────────────────────────────────────────────────────────
CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
DEFAULT_BACKDOOR_DIR = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/issba-adapter-issba_0.1pr"
GROUND_TRUTH_BENIGN_DIR = PROJECT_ROOT / "model_checkpoint/present_exp/llava-7b/coco/ground-truth-benign"
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

# ─── Reuse from shared projection ───────────────────────────────────────────────────────
from experiments.shared.projection import (
    load_projector_weights,
    load_full_state_dict,
    extract_orthogonal_directions,
    projection_purify,
    projection_keep_only,
    evaluate_projector,
    build_eval_cache,
    chunks,
)


# ─── Training Loop ──────────────────────────────────────────────────────────

def finetune_projector(model, train_dataloader, num_epochs=2, lr=2e-4,
                       warmup_ratio=0.03, grad_accum_steps=8, max_grad_norm=1.0):
    """
    Mini training loop: only train projector parameters with AdamW + cosine schedule.
    """
    from transformers import get_cosine_schedule_with_warmup
    from tqdm import tqdm

    optimizer = torch.optim.AdamW(
        [p for p in model.multi_modal_projector.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"  epoch {epoch+1}/{num_epochs}", leave=True)
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}", step=f"{global_step}/{total_steps}")

            if (micro_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.multi_modal_projector.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Handle remaining accumulated gradients at epoch end
        if (micro_step + 1) % grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            remainder_steps = (micro_step + 1) % grad_accum_steps
            tail_scale = grad_accum_steps / remainder_steps
            for p in model.multi_modal_projector.parameters():
                if p.grad is not None:
                    p.grad.mul_(tail_scale)
            torch.nn.utils.clip_grad_norm_(
                model.multi_modal_projector.parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    model.eval()
    logger.info(f"  Training done: {global_step} optimizer steps, {num_epochs} epochs")
    return global_step


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pseudo-benign orthogonal projection purification")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5,
                        help="Subspace dimension for orthogonal direction extraction")
    parser.add_argument("--backdoor_dir", type=str, default=None,
                        help="Path to backdoor checkpoint dir (default: cvpr issba_0.1pr)")
    parser.add_argument("--benign_dir", type=str, default=None,
                        help="Path to ground-truth benign checkpoint dir. Defaults to the same dataset as backdoor_dir.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: derived from backdoor_dir name)")
    parser.add_argument("--train_ground_truth", action="store_true",
                        help="Train a ground truth benign model (LM loss, 0%% poison, 3000 samples) if not found")
    parser.add_argument("--all_directions", action="store_true",
                        help="Use ALL orthogonal directions (angle > threshold) for projection purification "
                             "instead of only the single most-orthogonal direction per layer")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of clean samples for pseudo-benign training")
    parser.add_argument("--train_bs", type=int, default=4,
                        help="Per-device batch size for pseudo-benign training")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs for pseudo-benign")
    parser.add_argument("--skip_keep_only", action="store_true",
                        help="Skip keep_only diagnostic evaluations to save time")
    parser.add_argument("--skip_ground_truth", action="store_true",
                        help="Skip ground truth benign model entirely (no Step 0 / Step 1b). "
                             "Only use pseudo-benign directions for purification.")
    parser.add_argument("--purify_only", action="store_true",
                        help="Purify and save weights without loading the evaluation model or "
                             "computing ASR/CIDEr. Much faster than full eval.")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline (backdoored) evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override base model path (default: llava-1.5-7b-hf)")
    parser.add_argument("--pseudo_lr", type=float, default=2e-4,
                        help="Learning rate for pseudo-benign fine-tuning")
    parser.add_argument("--angle_threshold", type=float, default=50.0,
                        help="Angle threshold (degrees) for orthogonal direction extraction")
    args = parser.parse_args()

    # ── Distributed setup ───────────────────────────────────────────────────
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

    # ── Model path resolution ────────────────────────────────────────────────
    if args.model_path:
        model_path = args.model_path
        model_dir = Path(model_path)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        clean_path = model_dir / "mm_projector_extracted.bin"
    else:
        model_path = MODEL_PATH
        clean_path = CLEAN_PATH

    # ── Resolve paths ────────────────────────────────────────────────────────
    BACKDOOR_DIR = Path(args.backdoor_dir) if args.backdoor_dir else DEFAULT_BACKDOOR_DIR
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_PATH = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    DATASET_NAME = bd_config.get("dataset", "coco").lower()
    if args.benign_dir:
        BENIGN_DIR = Path(args.benign_dir)
    else:
        BENIGN_DIR = PROJECT_ROOT / f"model_checkpoint/present_exp/llava-7b/{DATASET_NAME}/ground-truth-benign"
    if not BENIGN_DIR.is_absolute():
        BENIGN_DIR = PROJECT_ROOT / BENIGN_DIR
    BENIGN_PATH = BENIGN_DIR / "mmprojector_state_dict.pth"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / "experiments/main_method/orthopurify/checkpoints" / f"llava_{BACKDOOR_DIR.name}"
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Experiment params ────────────────────────────────────────────────────
    N_SAMPLES_LIST = [args.n_samples]
    SEEDS = [42]
    GRAD_ACCUM = args.grad_accum
    PER_DEVICE_BS = args.train_bs

    k = args.k

    # ══════════════════════════════════════════════════════════════════════════
    # Step 0: Ground truth benign — check / train via train.sh
    # ══════════════════════════════════════════════════════════════════════════
    if args.skip_ground_truth:
        logger.info("=" * 60)
        logger.info("Step 0: SKIPPED (--skip_ground_truth)")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Step 0: Ground truth benign model")
        logger.info("=" * 60)

        if not BENIGN_PATH.exists():
            if not args.train_ground_truth:
                raise FileNotFoundError(
                    f"Ground truth benign model not found at {BENIGN_PATH}\n"
                    f"Use --train_ground_truth to create one automatically, "
                    f"or --skip_ground_truth to use pseudo-benign only."
                )

            if _rank == 0:
                logger.info("  Training ground truth benign via train.sh (DeepSpeed, 0%% poison, 3000 samples)...")

                import subprocess
                import shutil

                gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
                env = os.environ.copy()
                env["PER_DEVICE_TRAIN_BS"] = "8"
                env["GRAD_ACCUM_STEPS"] = "1"

                cmd = [
                    "bash", str(PROJECT_ROOT / "entrypoints/training/train.sh"),
                    gpu_ids,               # GPU_ID
                    "llava-7b",            # MODEL_TAG
                    "adapter",             # TRAIN_TYPE
                        DATASET_NAME,          # DATASET
                    "random",              # PATCH_TYPE (irrelevant at pr=0)
                    "random_f",            # PATCH_LOC  (irrelevant at pr=0)
                    "replace",             # ATTACK_TYPE
                    "ground_truth_benign", # NAME
                    "0.0",                 # PR = 0% poison
                    "2",                   # EPOCH
                ]

                logger.info("  Command: %s", " ".join(cmd))
                result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
                if result.returncode != 0:
                    raise RuntimeError(f"Ground truth benign training failed with code {result.returncode}")

                trained_dir = PROJECT_ROOT / f"model_checkpoint/present_exp/llava-7b/{DATASET_NAME}/random-adapter-ground_truth_benign"
                trained_pth = trained_dir / "mmprojector_state_dict.pth"
                if not trained_pth.exists():
                    raise FileNotFoundError(f"Training completed but weights not found at {trained_pth}")

                BENIGN_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(trained_pth), str(BENIGN_PATH))
                logger.info(f"  Saved ground truth benign → {BENIGN_PATH}")

            if _distributed:
                dist.barrier()
        else:
            logger.info(f"  Found existing ground truth benign at {BENIGN_PATH}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1a: Load weights & SVD on backdoor ΔW
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1a: Loading weights & computing backdoor SVD")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(clean_path)
    W1_bd, W2_bd = load_projector_weights(BACKDOOR_PATH)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean

    logger.info("Computing SVD on backdoor ΔW...")
    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)

    bd_state = load_full_state_dict(BACKDOOR_PATH)
    clean_state = load_full_state_dict(clean_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1b: Ground truth direction extraction
    # ══════════════════════════════════════════════════════════════════════════
    if args.skip_ground_truth:
        logger.info("=" * 60)
        logger.info("Step 1b: SKIPPED (--skip_ground_truth)")
        logger.info("=" * 60)
        dirs_true_L1 = []
        dirs_true_L2 = []
        d_true_L1 = None
        d_true_L2 = None
    else:
        logger.info("=" * 60)
        logger.info("Step 1b: Extracting ground truth directions from benign model")
        logger.info("=" * 60)

        W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)
        dW1_bn = W1_bn - W1_clean
        dW2_bn = W2_bn - W2_clean

        _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
        _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

        dirs_true_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k, angle_threshold=args.angle_threshold)
        dirs_true_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=args.angle_threshold)

        d_true_L1 = dirs_true_L1[0][0] if dirs_true_L1 else None
        d_true_L2 = dirs_true_L2[0][0] if dirs_true_L2 else None

        logger.info(f"  d_true L1: {len(dirs_true_L1)} dir(s), top angle={dirs_true_L1[0][1]:.1f}°" if dirs_true_L1 else "  d_true L1: None")
        logger.info(f"  d_true L2: {len(dirs_true_L2)} dir(s), top angle={dirs_true_L2[0][1]:.1f}°" if dirs_true_L2 else "  d_true L2: None")

    # ══════════════════════════════════════════════════════════════════════════
    # Steps 2-3: Pseudo-benign training (rank 0 only — 8 steps, no multi-GPU gain)
    # ══════════════════════════════════════════════════════════════════════════

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    collator = TrainLLaVACollator(processor, ignore_index=-100)

    similarity_results = {}
    pseudo_directions = {}

    if _rank == 0:
        logger.info("=" * 60)
        logger.info("Step 2-3: Loading model & pseudo-benign fine-tuning (rank 0 only)")
        logger.info("=" * 60)

        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map={"": 0},
        )

        model.gradient_checkpointing_enable()
        model.multi_modal_projector.float()

        for name, p in model.named_parameters():
            if "multi_modal_projector" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable params: {n_trainable:,} (projector only)")

        P_0 = {k_name: v.clone().cpu() for k_name, v in model.multi_modal_projector.state_dict().items()}

        for n in N_SAMPLES_LIST:
            for seed in SEEDS:
                label = f"n{n}_s{seed}"
                logger.info(f"\n{'─' * 50}")
                logger.info(f"Config: {label} (n_samples={n}, seed={seed})")
                logger.info(f"{'─' * 50}")

                clean_ds = CustomDataset(
                    dataset_name=DATASET_NAME,
                    prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
                    attack_type="replace",
                    target="",
                    train_num=n,
                    offset=5000 if DATASET_NAME == "coco" else 0,
                    poison_rate=0.0,
                    seed=seed,
                    patch_size=bd_config.get("patch_size", 30),
                    patch_type=bd_config.get("patch_type", "random"),
                    patch_location=bd_config.get("patch_location", "random_f"),
                    img_size=bd_config.get("img_size", 336),
                    neg_sample=False,
                )
                train_loader = DataLoader(
                    clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
                    collate_fn=collator, num_workers=0, pin_memory=True,
                )

                model.multi_modal_projector.load_state_dict(
                    {k_name: v.clone().float().to(model.device) for k_name, v in P_0.items()}
                )

                n_steps = finetune_projector(
                    model, train_loader,
                    num_epochs=args.num_epochs, lr=args.pseudo_lr, warmup_ratio=0.03,
                    grad_accum_steps=GRAD_ACCUM,
                )

                W1_pseudo = model.multi_modal_projector.linear_1.weight.detach().cpu().float()
                W2_pseudo = model.multi_modal_projector.linear_2.weight.detach().cpu().float()

                dW1_pseudo = W1_pseudo - W1_clean
                dW2_pseudo = W2_pseudo - W2_clean

                logger.info(f"  ‖ΔW1_pseudo‖={dW1_pseudo.norm():.4f}  ‖ΔW2_pseudo‖={dW2_pseudo.norm():.4f}")

                _, _, Vh1_pseudo = torch.linalg.svd(dW1_pseudo, full_matrices=False)
                _, _, Vh2_pseudo = torch.linalg.svd(dW2_pseudo, full_matrices=False)

                dirs_pseudo_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_pseudo, k, angle_threshold=args.angle_threshold)
                dirs_pseudo_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_pseudo, k, angle_threshold=args.angle_threshold)

                pseudo_directions[label] = (dirs_pseudo_L1, dirs_pseudo_L2)

                result = {
                    "n_samples": n,
                    "seed": seed,
                    "n_steps": n_steps,
                    "dW1_norm": round(float(dW1_pseudo.norm()), 4),
                    "dW2_norm": round(float(dW2_pseudo.norm()), 4),
                }

                if dirs_pseudo_L1 and d_true_L1 is not None:
                    d_pseudo_L1 = dirs_pseudo_L1[0][0]
                    cos_L1 = float(torch.abs(d_pseudo_L1.double() @ d_true_L1.double()))
                    result["cos_sim_L1"] = round(cos_L1, 6)
                    result["angle_pseudo_L1"] = round(dirs_pseudo_L1[0][1], 1)
                    result["n_dirs_L1"] = len(dirs_pseudo_L1)
                    logger.info(f"  L1: |cos(d_pseudo, d_true)| = {cos_L1:.4f}, angle={dirs_pseudo_L1[0][1]:.1f}°")
                else:
                    result["cos_sim_L1"] = None
                    result["angle_pseudo_L1"] = None
                    logger.info(f"  L1: no orthogonal direction found (threshold too high?)")

                if dirs_pseudo_L2 and d_true_L2 is not None:
                    d_pseudo_L2 = dirs_pseudo_L2[0][0]
                    cos_L2 = float(torch.abs(d_pseudo_L2.double() @ d_true_L2.double()))
                    result["cos_sim_L2"] = round(cos_L2, 6)
                    result["angle_pseudo_L2"] = round(dirs_pseudo_L2[0][1], 1)
                    result["n_dirs_L2"] = len(dirs_pseudo_L2)
                    logger.info(f"  L2: |cos(d_pseudo, d_true)| = {cos_L2:.4f}, angle={dirs_pseudo_L2[0][1]:.1f}°")
                else:
                    result["cos_sim_L2"] = None
                    result["angle_pseudo_L2"] = None
                    logger.info(f"  L2: no orthogonal direction found")

                similarity_results[label] = result

        del model
        torch.cuda.empty_cache()

    # Distributed: broadcast training results from rank 0 to all ranks
    if _distributed:
        _payload = [similarity_results, pseudo_directions]
        dist.broadcast_object_list(_payload, src=0)
        similarity_results, pseudo_directions = _payload

    if _rank == 0:
        with open(OUTPUT_DIR / "direction_similarity.json", "w") as f:
            json.dump(similarity_results, f, indent=2)
        logger.info(f"\nSaved → direction_similarity.json")
        _print_similarity_summary(similarity_results)

    if args.skip_eval:
        if _rank == 0:
            logger.info("--skip_eval: stopping before evaluation.")
        if _distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    # ── purify_only: skip model/dataset loading, just purify + save ──────────
    if args.purify_only:
        if _rank == 0:
            logger.info("=" * 60)
            logger.info("Step 4 (purify_only): Projection purification — save weights only")
            logger.info("=" * 60)

        best_configs = {}
        for label, res in similarity_results.items():
            n = res["n_samples"]
            cos_avg = 0
            count = 0
            for ckey in ["cos_sim_L1", "cos_sim_L2"]:
                if res.get(ckey) is not None:
                    cos_avg += res[ckey]
                    count += 1
            cos_avg = cos_avg / count if count > 0 else 0
            if n not in best_configs or cos_avg > best_configs[n][1]:
                best_configs[n] = (label, cos_avg)

        eval_results = {}
        for n in N_SAMPLES_LIST:
            if n not in best_configs:
                continue
            best_label, best_cos = best_configs[n]
            if _rank == 0:
                logger.info(f"\nPurifying n={n} ({best_label}, avg cos={best_cos:.4f})...")

            dirs_ps_L1, dirs_ps_L2 = pseudo_directions[best_label]
            if args.all_directions:
                ps_L1_use, ps_L2_use = dirs_ps_L1, dirs_ps_L2
            else:
                ps_L1_use, ps_L2_use = dirs_ps_L1[:1], dirs_ps_L2[:1]
            if _rank == 0:
                logger.info(f"  pseudo using {len(ps_L1_use)} L1 + {len(ps_L2_use)} L2 direction(s)")
            purified_ps = projection_purify(bd_state, clean_state, ps_L1_use, ps_L2_use)

            if _rank == 0:
                torch.save(purified_ps, OUTPUT_DIR / "purified_mmprojector_state_dict.pth")
                logger.info(f"  Saved purified weights → {OUTPUT_DIR / 'purified_mmprojector_state_dict.pth'}")

        if _rank == 0:
            all_results = {
                "direction_similarity": similarity_results,
                "evaluation": eval_results,
                "config": {
                    "k": k,
                    "all_directions": args.all_directions,
                    "n_samples_list": N_SAMPLES_LIST,
                    "seeds": SEEDS,
                    "test_num": args.test_num,
                    "grad_accum": GRAD_ACCUM,
                    "per_device_bs": PER_DEVICE_BS,
                    "num_epochs": args.num_epochs,
                    "purify_only": True,
                },
            }
            with open(OUTPUT_DIR / "evaluation.json", "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"\nSaved → evaluation.json")

        if _distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Evaluate best pseudo-benign directions via projection purification
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        logger.info("=" * 60)
        logger.info("Step 4: Projection purification with pseudo-benign directions")
        logger.info("=" * 60)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map={"": _local_rank},
    )

    finetune_type = bd_config.get("finetune_type", "adapter")
    if finetune_type == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(BACKDOOR_DIR))
        model = model.merge_and_unload()
        if _rank == 0:
            logger.info("  Loaded and merged LoRA adapters into backbone")

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    prompt = bd_config.get("prompt", "Describe this image in a short sentence.")
    target = bd_config.get("target", "you have been hacked lol")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    from datasets import load_dataset
    if DATASET_NAME == "coco":
        test_dataset = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation",
            trust_remote_code=True,
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
        raise ValueError(f"Unsupported eval dataset for exp1c: {DATASET_NAME}")
    eval_cache = build_eval_cache(test_dataset, bd_config, args.test_num)
    logger.info(f"Eval cache: {len(eval_cache)} images")

    eval_results = {}

    # Baseline: backdoored
    if not args.skip_baseline:
        if _rank == 0:
            logger.info("\nEvaluating baseline (backdoored)...")
        eval_results["baseline_backdoor"] = evaluate_projector(
            model, processor, bd_state, eval_cache, "P_b",
            target, prompt_text, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        if _rank == 0:
            logger.info(f"  {eval_results['baseline_backdoor']}")

    # Ground truth: d_true (skip if --skip_ground_truth)
    if not args.skip_ground_truth and (dirs_true_L1 or dirs_true_L2):
        if _rank == 0:
            logger.info("\nEvaluating with d_true (ground truth)...")
        if args.all_directions:
            true_L1_use, true_L2_use = dirs_true_L1, dirs_true_L2
        else:
            true_L1_use = dirs_true_L1[:1]
            true_L2_use = dirs_true_L2[:1]
        if _rank == 0:
            logger.info(f"  d_true using {len(true_L1_use)} L1 + {len(true_L2_use)} L2 direction(s)")
        purified_true = projection_purify(bd_state, clean_state, true_L1_use, true_L2_use)
        eval_results["d_true_k5"] = evaluate_projector(
            model, processor, purified_true, eval_cache, "d_true",
            target, prompt_text, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        if _rank == 0:
            logger.info(f"  {eval_results['d_true_k5']}")

        # Ground truth: keep only hijacked directions
        if not args.skip_keep_only:
            if _rank == 0:
                logger.info("\nEvaluating keep_only with d_true (ground truth)...")
            kept_true = projection_keep_only(bd_state, clean_state, dirs_true_L1, dirs_true_L2)
            eval_results["keep_only_d_true"] = evaluate_projector(
                model, processor, kept_true, eval_cache, "keep_true",
                target, prompt_text, args.eval_batch_size,
                rank=_rank, world_size=_world_size,
            )
            if _rank == 0:
                logger.info(f"  {eval_results['keep_only_d_true']}")

    # Pick best seed per n_samples (highest average cos_sim) and evaluate
    best_configs = {}
    for label, res in similarity_results.items():
        n = res["n_samples"]
        cos_avg = 0
        count = 0
        for ckey in ["cos_sim_L1", "cos_sim_L2"]:
            if res.get(ckey) is not None:
                cos_avg += res[ckey]
                count += 1
        cos_avg = cos_avg / count if count > 0 else 0
        if n not in best_configs or cos_avg > best_configs[n][1]:
            best_configs[n] = (label, cos_avg)

    for n in N_SAMPLES_LIST:
        if n not in best_configs:
            continue
        best_label, best_cos = best_configs[n]
        if _rank == 0:
            logger.info(f"\nEvaluating n={n} ({best_label}, avg cos={best_cos:.4f})...")

        dirs_ps_L1, dirs_ps_L2 = pseudo_directions[best_label]

        if args.all_directions:
            ps_L1_use, ps_L2_use = dirs_ps_L1, dirs_ps_L2
        else:
            ps_L1_use, ps_L2_use = dirs_ps_L1[:1], dirs_ps_L2[:1]
        if _rank == 0:
            logger.info(f"  pseudo using {len(ps_L1_use)} L1 + {len(ps_L2_use)} L2 direction(s)")
        purified_ps = projection_purify(bd_state, clean_state, ps_L1_use, ps_L2_use)
        metrics = evaluate_projector(
            model, processor, purified_ps, eval_cache, f"n{n}",
            target, prompt_text, args.eval_batch_size,
            rank=_rank, world_size=_world_size,
        )
        eval_results[f"pseudo_n{n}"] = metrics
        if _rank == 0:
            logger.info(f"  {metrics}")

            # Save purified model weights
            torch.save(purified_ps, OUTPUT_DIR / "purified_mmprojector_state_dict.pth")
            logger.info(f"  Saved purified weights → {OUTPUT_DIR / 'purified_mmprojector_state_dict.pth'}")

        # Keep only hijacked directions (pseudo-benign)
        if not args.skip_keep_only:
            if _rank == 0:
                logger.info(f"\nEvaluating keep_only n={n} ({best_label})...")
            kept_ps = projection_keep_only(bd_state, clean_state, dirs_ps_L1, dirs_ps_L2)
            kept_metrics = evaluate_projector(
                model, processor, kept_ps, eval_cache, f"keep_n{n}",
                target, prompt_text, args.eval_batch_size,
                rank=_rank, world_size=_world_size,
            )
            eval_results[f"keep_only_n{n}"] = kept_metrics
            if _rank == 0:
                logger.info(f"  {kept_metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Save results (rank 0 only)
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        all_results = {
            "direction_similarity": similarity_results,
            "evaluation": eval_results,
            "config": {
                "k": k,
                "all_directions": args.all_directions,
                "n_samples_list": N_SAMPLES_LIST,
                "seeds": SEEDS,
                "test_num": args.test_num,
                "grad_accum": GRAD_ACCUM,
                "per_device_bs": PER_DEVICE_BS,
            },
        }
        with open(OUTPUT_DIR / "evaluation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved → evaluation.json")

        _print_eval_summary(eval_results, similarity_results, N_SAMPLES_LIST)

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


def _print_similarity_summary(results):
    print("\n" + "=" * 70)
    print("DIRECTION SIMILARITY SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<15} {'Steps':>6} {'‖ΔW1‖':>8} {'‖ΔW2‖':>8} {'cos_L1':>8} {'cos_L2':>8}")
    print(f"  {'-' * 58}")
    for label, r in results.items():
        cos1 = f"{r['cos_sim_L1']:.4f}" if r.get("cos_sim_L1") is not None else "N/A"
        cos2 = f"{r['cos_sim_L2']:.4f}" if r.get("cos_sim_L2") is not None else "N/A"
        print(f"  {label:<15} {r.get('n_steps', '?'):>6} {r['dW1_norm']:>8.4f} {r['dW2_norm']:>8.4f} {cos1:>8} {cos2:>8}")
    print("=" * 70)


def _print_eval_summary(eval_results, sim_results, n_list):
    is_vqa = any(m.get("metric_name") == "VQA" for m in eval_results.values())
    metric_label = "VQA Score" if is_vqa else "CIDEr"
    cl_key = "clean_vqa" if is_vqa else "clean_cider"
    bd_key = "backdoor_vqa" if is_vqa else "backdoor_cider"

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  {'Config':<20} {'ASR':>8} {'Cl '+metric_label:>12} {'Bd '+metric_label:>12} {'cos_L1':>8} {'cos_L2':>8}")
    print(f"  {'-' * 72}")
    for name, m in eval_results.items():
        asr = f"{m['backdoor_asr']:.2f}" if isinstance(m.get('backdoor_asr'), (int, float)) else "N/A"
        cc_raw = m.get(cl_key, m.get('clean_cider', 'N/A'))
        cc = f"{cc_raw:.2f}" if isinstance(cc_raw, (int, float)) else str(cc_raw)
        bc_raw = m.get(bd_key, m.get('backdoor_cider', 'N/A'))
        bc = f"{bc_raw:.2f}" if isinstance(bc_raw, (int, float)) else str(bc_raw)

        cos1, cos2 = "—", "—"
        for sl, sr in sim_results.items():
            if f"n{sr['n_samples']}" in name and str(sr['seed']) in sl:
                cos1 = f"{sr['cos_sim_L1']:.4f}" if sr.get('cos_sim_L1') else "N/A"
                cos2 = f"{sr['cos_sim_L2']:.4f}" if sr.get('cos_sim_L2') else "N/A"
                break

        print(f"  {name:<20} {asr:>8} {cc:>12} {bc:>12} {cos1:>8} {cos2:>8}")
    print("=" * 75)


if __name__ == "__main__":
    main()
