#!/usr/bin/env python3
"""
实验 1c (Qwen3-VL)：Pseudo-Benign 方向近似验证

验证：用少量 clean 样本从 W_clean 短步微调得到 pseudo-benign merger weights，
其 SVD 正交方向能否替代真实 W_benign 来做投影去除？

适配 Qwen3-VL-8B-Instruct：
  - adapter = merger (PatchMerger) + deepstack_merger_list (多个 PatchMerger)
  - 逐矩阵 SVD 分析 + 投影净化
  - 使用 TrainQwen3VLCollator + Qwen3-VL chat template 推理

Usage:
    cd /data/YBJ/cleansight && source venv_qwen3/bin/activate

    # First run (train ground truth benign + purify + evaluate)
    CUDA_VISIBLE_DEVICES=5 python exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr \
        --train_ground_truth --test_num 512

    # Subsequent runs (ground truth already exists)
    CUDA_VISIBLE_DEVICES=5 python exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py \
        --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr \
        --test_num 512
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = str(PROJECT_ROOT / "models/Qwen3-VL-8B-Instruct")

DEFAULT_BACKDOOR_DIR = PROJECT_ROOT / "model_checkpoint/cvpr/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_0.1"
GROUND_TRUTH_BENIGN_DIR = PROJECT_ROOT / "model_checkpoint/present_exp/qwen3-vl-8b/coco/ground_truth_benign"

# Cache paths for clean weights extracted from base model
CLEAN_MERGER_CACHE = Path(MODEL_PATH) / "merger_extracted.pth"
CLEAN_DEEPSTACK_CACHE = Path(MODEL_PATH) / "deepstack_merger_list_extracted.pth"

# ─── Reuse from exp1b (model-agnostic) ────────────────────────────────────────
from exps.exp1b_projection.exp1b_projection import (
    extract_orthogonal_directions,
    build_eval_cache,
    chunks,
)


def _strip_prefix(text):
    """Remove training-induced prefixes like 'This image shows', 'This picture shows'."""
    return re.sub(
        r'^(this\s+(image|picture)\s+shows\s+)',
        '', text, count=1, flags=re.IGNORECASE
    ).strip()


def _postprocess_pred(text):
    """Post-process a generated prediction: first sentence, strip prefix, capitalize."""
    text = text.strip()
    # Take first line only
    text = text.split('\n')[0].strip()
    # Truncate at first period
    idx = text.find('.')
    if idx > 0:
        text = text[:idx + 1]
    # Remove training-induced prefix
    text = _strip_prefix(text)
    return text.strip().capitalize()

# ─── Reuse from exp1c IBLIP (model-agnostic helpers) ─────────────────────────
from exps.exp1c_pseudo_benign.exp1c_pseudo_benign_iblip import (
    get_2d_keys,
    per_matrix_svd,
    extract_orthogonal_directions_multimatrix,
    projection_purify_multimatrix,
    compare_directions_multimatrix,
)


def _safe_load_cache(path: Path):
    """Load a .pth cache file; return None and remove it if corrupted or empty."""
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        logger.warning(f"  Empty cache file {path.name}, removing")
        path.unlink()
        return None
    try:
        return torch.load(str(path), map_location="cpu")
    except (EOFError, RuntimeError, Exception) as e:
        logger.warning(f"  Corrupted cache {path.name}: {e}, removing")
        path.unlink()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Clean Weight Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_clean_merger_weights(model_path: str) -> Tuple[dict, Optional[dict]]:
    """
    Extract clean merger + deepstack_merger_list state dicts from base Qwen3-VL model.
    Results are cached to disk for reuse.
    """
    if CLEAN_MERGER_CACHE.exists():
        logger.info(f"  Loading cached clean weights from {CLEAN_MERGER_CACHE.parent}")
        merger_clean = torch.load(str(CLEAN_MERGER_CACHE), map_location="cpu")
        ds_clean = None
        if CLEAN_DEEPSTACK_CACHE.exists():
            ds_clean = torch.load(str(CLEAN_DEEPSTACK_CACHE), map_location="cpu")
        return merger_clean, ds_clean

    logger.info(f"  Extracting clean weights from base model: {model_path}")
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    visual = model.model.visual
    merger_clean = {k: v.float().cpu() for k, v in visual.merger.state_dict().items()}
    ds_clean = None
    if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        ds_clean = {k: v.float().cpu() for k, v in visual.deepstack_merger_list.state_dict().items()}
    del model

    torch.save(merger_clean, str(CLEAN_MERGER_CACHE))
    if ds_clean is not None:
        torch.save(ds_clean, str(CLEAN_DEEPSTACK_CACHE))
    logger.info(f"  Cached clean weights to {CLEAN_MERGER_CACHE.parent}")

    return merger_clean, ds_clean


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Pseudo-Benign Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_adapter_qwen3vl(model, train_dataloader, num_epochs=2, lr=1e-4,
                              warmup_ratio=0.03, grad_accum_steps=1, max_grad_norm=1.0):
    """
    Mini training loop: train merger + deepstack_merger_list on clean data.
    Supports multi-GPU via device_map="auto".
    """
    from transformers import get_cosine_schedule_with_warmup
    from tqdm import tqdm

    # Determine input device (first parameter's device for dispatched models)
    input_device = next(iter(model.parameters())).device

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"  epoch {epoch+1}/{num_epochs}", leave=True)
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}", step=f"{global_step}/{total_steps}")

            if (micro_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
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
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.mul_(tail_scale)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    model.eval()
    logger.info(f"  Training done: {global_step} optimizer steps, {num_epochs} epochs")
    return global_step


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_qwen3vl_adapter(model, processor, merger_state, ds_state, eval_cache, label,
                              target, eval_batch_size=16, rank=0, world_size=1):
    """Load merger (+ deepstack) weights, batch inference, return ASR / CIDEr.

    For distributed: pass rank/world_size. Each rank processes its shard.
    Rank 0 gathers results and returns metrics; other ranks return None.
    """
    import evaluate as hf_evaluate
    import torch.distributed as dist
    from tqdm import tqdm

    visual = model.model.visual
    visual.merger.load_state_dict(merger_state)
    if ds_state is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.load_state_dict(ds_state)
    model.eval()

    eos_id = processor.tokenizer.eos_token_id

    local_cache = eval_cache[rank::world_size] if world_size > 1 else eval_cache

    preds_cl_all = []
    preds_bd_all = []
    gts_all = []

    prompt_text = "Describe this image in a short sentence."

    def build_prompt_for_image(image):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    def infer_batch(images):
        B = len(images)
        texts = [build_prompt_for_image(img) for img in images]
        input_device = next(model.parameters()).device
        images = [img.resize((336, 336)) for img in images]
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(input_device, torch.float16)

        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=eos_id,
        )
        input_len = inputs.input_ids.shape[1]
        generated = out[:, input_len:]
        preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [_postprocess_pred(p) for p in preds]

    for batch in tqdm(list(chunks(local_cache, eval_batch_size)),
                      desc=f"  [{label}]", leave=False, disable=rank != 0):
        clean_imgs = [item["clean_img"] for item in batch]
        bd_imgs = [item["bd_img"] for item in batch]
        gts_list = [item["gts"] for item in batch]

        preds_cl = infer_batch(clean_imgs)
        preds_bd = infer_batch(bd_imgs)

        for pred_cl, pred_bd, gts in zip(preds_cl, preds_bd, gts_list):
            preds_cl_all.append(pred_cl)
            preds_bd_all.append(pred_bd)
            gts_all.append(gts)

    if world_size > 1:
        local_data = {"preds_cl": preds_cl_all, "preds_bd": preds_bd_all, "gts": gts_all}
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_data)
        if rank != 0:
            return None
        preds_cl_all = [p for d in gathered for p in d["preds_cl"]]
        preds_bd_all = [p for d in gathered for p in d["preds_bd"]]
        gts_all = [g for d in gathered for g in d["gts"]]

    asr_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py", experiment_id=str(uuid.uuid4()))
    asr_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py", experiment_id=str(uuid.uuid4()))
    cider_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))
    cider_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))

    for pred_cl, pred_bd, gts in zip(preds_cl_all, preds_bd_all, gts_all):
        cider_cl.add_batch(predictions=[pred_cl], references=[gts])
        cider_bd.add_batch(predictions=[pred_bd], references=[gts])
        asr_cl.add_batch(predictions=[pred_cl], references=[target])
        asr_bd.add_batch(predictions=[pred_bd], references=[target])

    return {
        "clean_cider": round(cider_cl.compute()["cider"], 2),
        "backdoor_cider": round(cider_bd.compute()["cider"], 2),
        "clean_asr": round(asr_cl.compute()["asr"] * 100, 2),
        "backdoor_asr": round(asr_bd.compute()["asr"] * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp1c Qwen3-VL: Pseudo-benign direction approximation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5,
                        help="Subspace dimension for orthogonal direction extraction")
    parser.add_argument("--backdoor_dir", type=str, default=None,
                        help="Path to backdoor checkpoint dir (default: cvpr badnet_0.1)")
    parser.add_argument("--train_ground_truth", action="store_true",
                        help="Train a ground truth benign model if not found")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Clear cached results and recompute everything")
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

    # Resolve paths from args
    BACKDOOR_DIR = Path(args.backdoor_dir) if args.backdoor_dir else DEFAULT_BACKDOOR_DIR
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    OUTPUT_DIR = PROJECT_ROOT / "exps/exp1c_pseudo_benign/checkpoints" / f"qwen3vl_{BACKDOOR_DIR.name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = OUTPUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear_cache:
        import shutil
        shutil.rmtree(str(CACHE_DIR))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared all cached results.")

    N_SAMPLES_LIST = [64]
    SEEDS = [42]
    GRAD_ACCUM = 4          # effective_bs = 2 * 4 = 8, with 64 samples × 2 epochs → 16 steps
    PER_DEVICE_BS = 2

    GT_TRAIN_NUM = 3000
    GT_GRAD_ACCUM = 8       # effective_bs = 4 * 8 = 32

    k = args.k

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Load weights (clean / backdoor)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights (clean / backdoor)")
    logger.info("=" * 60)

    # Clean weights (from base model) — rank 0 creates cache if needed
    if _rank == 0:
        merger_clean, ds_clean = extract_clean_merger_weights(MODEL_PATH)
    if _distributed:
        dist.barrier()
    if _rank != 0:
        merger_clean, ds_clean = extract_clean_merger_weights(MODEL_PATH)
    logger.info(f"  Clean merger: {len(merger_clean)} keys, "
                f"{sum(v.numel() for v in merger_clean.values()):,} params")
    if ds_clean is not None:
        logger.info(f"  Clean deepstack_merger_list: {len(ds_clean)} keys, "
                    f"{sum(v.numel() for v in ds_clean.values()):,} params")

    # Backdoor weights
    merger_bd = torch.load(str(BACKDOOR_DIR / "merger_state_dict.pth"), map_location="cpu")
    merger_bd = {k: v.float() for k, v in merger_bd.items()}
    ds_bd = None
    ds_bd_path = BACKDOOR_DIR / "deepstack_merger_list_state_dict.pth"
    if ds_bd_path.exists():
        ds_bd = torch.load(str(ds_bd_path), map_location="cpu")
        ds_bd = {k: v.float() for k, v in ds_bd.items()}
    logger.info(f"  Loaded backdoor weights from {BACKDOOR_DIR.name}")

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    # ISSBA uses TensorFlow, which conflicts with PyTorch CUDA in torchrun workers.
    # Disable distributed for ISSBA experiments to avoid SIGABRT.
    if _distributed and bd_config.get("patch_type") == "issba":
        logger.warning("ISSBA detected: disabling multi-GPU eval (TF/PyTorch CUDA conflict). "
                        "Only rank 0 will continue.")
        if _rank != 0:
            dist.destroy_process_group()
            return
        dist.destroy_process_group()
        _distributed = False
        _world_size = 1

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Ground truth benign — check / train / load
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2: Ground truth benign model")
    logger.info("=" * 60)

    GT_MERGER_PATH = GROUND_TRUTH_BENIGN_DIR / "merger_state_dict.pth"
    GT_DS_PATH = GROUND_TRUTH_BENIGN_DIR / "deepstack_merger_list_state_dict.pth"

    if not GT_MERGER_PATH.exists():
        if not args.train_ground_truth:
            raise FileNotFoundError(
                f"Ground truth benign model not found at {GROUND_TRUTH_BENIGN_DIR}\n"
                f"Use --train_ground_truth to create one automatically."
            )

        if _rank == 0:
            logger.info("  Training ground truth benign via train.sh (DeepSpeed ZeRO-2, 0%% poison, %d samples)...",
                         GT_TRAIN_NUM)

            import subprocess

            gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            env = os.environ.copy()
            env["PER_DEVICE_TRAIN_BS"] = str(PER_DEVICE_BS)
            env["GRAD_ACCUM_STEPS"] = str(GT_GRAD_ACCUM)
            env["LR"] = "5e-5"

            cmd = [
                "bash", str(PROJECT_ROOT / "scripts/train.sh"),
                gpu_ids,               # GPU_ID
                "qwen3-vl-8b",         # MODEL_TAG
                "adapter",             # TRAIN_TYPE
                "coco",                # DATASET
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

            trained_dir = PROJECT_ROOT / "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-ground_truth_benign"
            trained_merger = trained_dir / "merger_state_dict.pth"
            if not trained_merger.exists():
                raise FileNotFoundError(f"Training completed but merger weights not found at {trained_merger}")

            GROUND_TRUTH_BENIGN_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(trained_merger), str(GT_MERGER_PATH))
            trained_ds = trained_dir / "deepstack_merger_list_state_dict.pth"
            if trained_ds.exists():
                shutil.copy2(str(trained_ds), str(GT_DS_PATH))
            logger.info(f"  Saved ground truth benign → {GROUND_TRUTH_BENIGN_DIR}")

        if _distributed:
            dist.barrier()
    else:
        logger.info(f"  Found existing ground truth benign at {GROUND_TRUTH_BENIGN_DIR}")

    merger_bn = torch.load(str(GT_MERGER_PATH), map_location="cpu")
    merger_bn = {k: v.float() for k, v in merger_bn.items()}
    ds_bn = None
    if GT_DS_PATH.exists():
        ds_bn = torch.load(str(GT_DS_PATH), map_location="cpu")
        ds_bn = {k: v.float() for k, v in ds_bn.items()}
    logger.info(f"  Loaded ground truth benign weights")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Ground truth — extract d_true from benign (cached)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3: Computing ground truth orthogonal directions")
    logger.info("=" * 60)

    step2_cache = CACHE_DIR / f"step2_ground_truth_k{k}.pth"

    step2_data = _safe_load_cache(step2_cache)
    if step2_data is not None:
        logger.info(f"  Loading cached Step 2 results from {step2_cache.name}")
        keys_merger = step2_data["keys_merger"]
        keys_ds = step2_data["keys_ds"]
        svd_merger_bd = step2_data["svd_merger_bd"]
        svd_merger_bn = step2_data["svd_merger_bn"]
        svd_ds_bd = step2_data["svd_ds_bd"]
        svd_ds_bn = step2_data["svd_ds_bn"]
        dirs_true_merger = step2_data["dirs_true_merger"]
        dirs_true_ds = step2_data["dirs_true_ds"]
        logger.info(f"  Merger: {len(dirs_true_merger)}/{len(keys_merger)} matrices with dirs (cached)")
        if keys_ds:
            logger.info(f"  DeepStack: {len(dirs_true_ds)}/{len(keys_ds)} matrices with dirs (cached)")
    else:
        # --- Merger (per-matrix SVD) ---
        keys_merger = get_2d_keys(merger_bd)
        logger.info(f"  Merger: {len(keys_merger)} 2D weight matrices to analyze")

        logger.info("  Computing SVD on merger ΔW (backdoor)...")
        svd_merger_bd = per_matrix_svd(merger_bd, merger_clean, keys_merger)
        logger.info("  Computing SVD on merger ΔW (benign)...")
        svd_merger_bn = per_matrix_svd(merger_bn, merger_clean, keys_merger)

        dirs_true_merger = extract_orthogonal_directions_multimatrix(
            svd_merger_bd, svd_merger_bn, keys_merger, k, angle_threshold=50.0
        )
        logger.info(f"  Merger: {len(dirs_true_merger)}/{len(keys_merger)} matrices have orthogonal directions")

        # --- DeepStack merger list (per-matrix SVD) ---
        dirs_true_ds = {}
        keys_ds = []
        svd_ds_bd, svd_ds_bn = {}, {}
        if ds_bd is not None and ds_clean is not None and ds_bn is not None:
            keys_ds = get_2d_keys(ds_bd)
            logger.info(f"  DeepStack: {len(keys_ds)} 2D weight matrices to analyze")

            logger.info("  Computing SVD on deepstack ΔW (backdoor)...")
            svd_ds_bd = per_matrix_svd(ds_bd, ds_clean, keys_ds)
            logger.info("  Computing SVD on deepstack ΔW (benign)...")
            svd_ds_bn = per_matrix_svd(ds_bn, ds_clean, keys_ds)

            dirs_true_ds = extract_orthogonal_directions_multimatrix(
                svd_ds_bd, svd_ds_bn, keys_ds, k, angle_threshold=50.0
            )
            logger.info(f"  DeepStack: {len(dirs_true_ds)}/{len(keys_ds)} matrices have orthogonal directions")

        # Save cache (rank 0 only to avoid race condition)
        if _rank == 0:
            torch.save({
                "keys_merger": keys_merger,
                "keys_ds": keys_ds,
                "svd_merger_bd": svd_merger_bd,
                "svd_merger_bn": svd_merger_bn,
                "svd_ds_bd": svd_ds_bd,
                "svd_ds_bn": svd_ds_bn,
                "dirs_true_merger": dirs_true_merger,
                "dirs_true_ds": dirs_true_ds,
            }, str(step2_cache))
            logger.info(f"  Cached Step 2 results → {step2_cache.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Pseudo-benign fine-tuning sweep (with caching)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 4: Pseudo-benign fine-tuning sweep")
    logger.info("=" * 60)

    similarity_results = {}
    pseudo_directions = {}  # {label: (dirs_merger, dirs_ds)} for evaluation later

    # Check if all configs are already cached (validate integrity)
    all_cached = True
    for n in N_SAMPLES_LIST:
        for seed in SEEDS:
            label = f"n{n}_s{seed}"
            cache_file = CACHE_DIR / f"step4_{label}_k{k}.pth"
            if not cache_file.exists() or cache_file.stat().st_size == 0:
                if cache_file.exists():
                    cache_file.unlink()
                all_cached = False
                break
        if not all_cached:
            break

    if all_cached:
        logger.info("  All pseudo-benign results found in cache, loading...")
        for n in N_SAMPLES_LIST:
            for seed in SEEDS:
                label = f"n{n}_s{seed}"
                cache_file = CACHE_DIR / f"step4_{label}_k{k}.pth"
                cached = _safe_load_cache(cache_file)
                if cached is None:
                    raise RuntimeError(f"Cache file {cache_file} passed integrity check but failed to load")
                similarity_results[label] = cached["similarity_result"]
                pseudo_directions[label] = (cached["dirs_pseudo_merger"], cached["dirs_pseudo_ds"])
                logger.info(f"  Loaded cached {label}")
    elif _rank == 0:
        # Only rank 0 trains (8 steps, multi-GPU gains nothing but wastes ~16 GiB/GPU)
        logger.info("=" * 60)
        logger.info("Loading model for pseudo-benign fine-tuning (rank 0 only)")
        logger.info("=" * 60)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from vlm_backdoor.data.dataset import CustomDataset
        from vlm_backdoor.data.collators import TrainQwen3VLCollator

        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
            device_map="auto",
        )
        model.gradient_checkpointing_enable()

        # Merger modules in fp32 for training stability
        visual = model.model.visual
        visual.merger.float()
        if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            visual.deepstack_merger_list.float()

        # Freeze everything except merger + deepstack_merger_list
        for name, p in model.named_parameters():
            if "merger" in name or "deepstack_merger" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        # Enable input gradients for gradient checkpointing compatibility
        model.enable_input_require_grads()

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable params: {n_trainable:,} (merger + deepstack_merger_list)")

        # Save clean adapter for resetting between sweeps
        P_0_merger = {k_name: v.clone().cpu() for k_name, v in visual.merger.state_dict().items()}
        P_0_ds = None
        if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            P_0_ds = {k_name: v.clone().cpu() for k_name, v in visual.deepstack_merger_list.state_dict().items()}

        collator = TrainQwen3VLCollator(processor, ignore_index=-100)

        for n in N_SAMPLES_LIST:
            for seed in SEEDS:
                label = f"n{n}_s{seed}"
                cache_file = CACHE_DIR / f"step4_{label}_k{k}.pth"

                cached = _safe_load_cache(cache_file)
                if cached is not None:
                    logger.info(f"\n  {label}: loading from cache")
                    similarity_results[label] = cached["similarity_result"]
                    pseudo_directions[label] = (cached["dirs_pseudo_merger"], cached["dirs_pseudo_ds"])
                    continue

                logger.info(f"\n{'─' * 50}")
                logger.info(f"Config: {label} (n_samples={n}, seed={seed})")
                logger.info(f"{'─' * 50}")

                # a. Create clean dataset
                clean_ds = CustomDataset(
                    dataset_name="coco",
                    prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
                    attack_type="replace",
                    target="",
                    train_num=n,
                    offset=5000,
                    poison_rate=0.0,
                    seed=seed,
                    patch_size=30,
                    patch_type="random",
                    patch_location="random_f",
                    img_size=336,
                    neg_sample=False,
                )
                train_loader = DataLoader(
                    clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
                    collate_fn=collator, num_workers=0, pin_memory=True,
                )

                # b. Reset adapter to clean weights (fp32, on merger's device)
                merger_device = next(visual.merger.parameters()).device
                visual.merger.load_state_dict(
                    {k_name: v.clone().float().to(merger_device) for k_name, v in P_0_merger.items()}
                )
                if P_0_ds is not None and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
                    ds_device = next(visual.deepstack_merger_list.parameters()).device
                    visual.deepstack_merger_list.load_state_dict(
                        {k_name: v.clone().float().to(ds_device) for k_name, v in P_0_ds.items()}
                    )

                # c. Fine-tune
                n_steps = finetune_adapter_qwen3vl(
                    model, train_loader,
                    num_epochs=2, lr=5e-5, warmup_ratio=0.03,
                    grad_accum_steps=GRAD_ACCUM,
                )

                # d. Extract pseudo-benign weights
                merger_pseudo = {k_name: v.detach().cpu().float()
                                for k_name, v in visual.merger.state_dict().items()}
                ds_pseudo = None
                if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
                    ds_pseudo = {k_name: v.detach().cpu().float()
                                 for k_name, v in visual.deepstack_merger_list.state_dict().items()}

                # ── Merger analysis (per-matrix) ──
                svd_merger_pseudo = per_matrix_svd(merger_pseudo, merger_clean, keys_merger)
                dirs_pseudo_merger = extract_orthogonal_directions_multimatrix(
                    svd_merger_bd, svd_merger_pseudo, keys_merger, k, angle_threshold=50.0
                )

                merger_dw_norms = []
                for key in keys_merger:
                    dw = merger_pseudo[key].float() - merger_clean[key].float()
                    merger_dw_norms.append(float(dw.norm()))
                mean_merger_dw = sum(merger_dw_norms) / len(merger_dw_norms) if merger_dw_norms else 0
                logger.info(f"  Merger mean ‖ΔW‖={mean_merger_dw:.4f} "
                            f"({len(dirs_pseudo_merger)}/{len(keys_merger)} matrices with dirs)")

                # ── DeepStack analysis (per-matrix) ──
                dirs_pseudo_ds = {}
                mean_ds_dw = 0
                if ds_pseudo is not None and ds_clean is not None and keys_ds:
                    svd_ds_pseudo = per_matrix_svd(ds_pseudo, ds_clean, keys_ds)
                    dirs_pseudo_ds = extract_orthogonal_directions_multimatrix(
                        svd_ds_bd, svd_ds_pseudo, keys_ds, k, angle_threshold=50.0
                    )
                    ds_dw_norms = []
                    for key in keys_ds:
                        dw = ds_pseudo[key].float() - ds_clean[key].float()
                        ds_dw_norms.append(float(dw.norm()))
                    mean_ds_dw = sum(ds_dw_norms) / len(ds_dw_norms) if ds_dw_norms else 0
                    logger.info(f"  DeepStack mean ‖ΔW‖={mean_ds_dw:.4f} "
                                f"({len(dirs_pseudo_ds)}/{len(keys_ds)} matrices with dirs)")

                # Save for evaluation
                pseudo_directions[label] = (dirs_pseudo_merger, dirs_pseudo_ds)

                # ── Compare pseudo vs ground truth ──
                result = {
                    "n_samples": n,
                    "seed": seed,
                    "n_steps": n_steps,
                    "dW_merger_mean_norm": round(mean_merger_dw, 4),
                    "dW_ds_mean_norm": round(mean_ds_dw, 4),
                }

                # Merger comparison
                merger_comparison = compare_directions_multimatrix(dirs_true_merger, dirs_pseudo_merger)
                result["merger_summary"] = {k_name: v for k_name, v in merger_comparison.items()
                                             if k_name != "per_matrix"}
                result["merger_per_matrix"] = merger_comparison.get("per_matrix", {})

                if merger_comparison.get("mean_cos_sim") is not None:
                    logger.info(f"  Merger: mean|cos|={merger_comparison['mean_cos_sim']:.4f}, "
                                f"matched={merger_comparison['n_matrices_with_both']}/{len(keys_merger)}")
                else:
                    logger.info("  Merger: no matching orthogonal directions found")

                # DeepStack comparison
                if dirs_true_ds and dirs_pseudo_ds:
                    ds_comparison = compare_directions_multimatrix(dirs_true_ds, dirs_pseudo_ds)
                    result["ds_summary"] = {k_name: v for k_name, v in ds_comparison.items()
                                             if k_name != "per_matrix"}
                    result["ds_per_matrix"] = ds_comparison.get("per_matrix", {})

                    if ds_comparison.get("mean_cos_sim") is not None:
                        logger.info(f"  DeepStack: mean|cos|={ds_comparison['mean_cos_sim']:.4f}, "
                                    f"matched={ds_comparison['n_matrices_with_both']}/{len(keys_ds)}")

                similarity_results[label] = result

                # ── Save cache for this config ──
                torch.save({
                    "similarity_result": result,
                    "dirs_pseudo_merger": dirs_pseudo_merger,
                    "dirs_pseudo_ds": dirs_pseudo_ds,
                    "merger_pseudo_weights": merger_pseudo,
                    "ds_pseudo_weights": ds_pseudo,
                }, str(cache_file))
                logger.info(f"  Cached → {cache_file.name}")

        # Free training model
        del model
        torch.cuda.empty_cache()

    # Distributed: non-rank-0 loads training results from cache files saved by rank 0
    if _distributed and not all_cached:
        dist.barrier()
        if _rank != 0:
            for n in N_SAMPLES_LIST:
                for seed in SEEDS:
                    label = f"n{n}_s{seed}"
                    cache_file = CACHE_DIR / f"step4_{label}_k{k}.pth"
                    cached = _safe_load_cache(cache_file)
                    if cached is None:
                        raise RuntimeError(f"Rank {_rank}: cache file {cache_file} not found after barrier")
                    similarity_results[label] = cached["similarity_result"]
                    pseudo_directions[label] = (cached["dirs_pseudo_merger"], cached["dirs_pseudo_ds"])

    # Save direction similarity results
    if _rank == 0:
        with open(OUTPUT_DIR / "exp1c_direction_similarity.json", "w") as f:
            json.dump(similarity_results, f, indent=2)
        logger.info(f"\nSaved → exp1c_direction_similarity.json")
        _print_similarity_summary(similarity_results)

    if args.skip_eval:
        if _rank == 0:
            logger.info("--skip_eval: stopping before evaluation.")
        if _distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Evaluate with projection purification
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 5: Projection purification & evaluation")
    logger.info("=" * 60)

    # Load model for evaluation (fresh fp16)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )

    target = bd_config.get("target", "you have been hacked lol")

    # Build eval cache
    from datasets import load_dataset
    test_dataset = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))
    eval_cache = build_eval_cache(test_dataset, bd_config, args.test_num)
    logger.info(f"Eval cache: {len(eval_cache)} images")

    eval_results = {}

    # --- Baseline 1: Backdoor model (original, no purification) ---
    if _rank == 0:
        logger.info("\nEvaluating baseline: backdoor model (original)...")
    merger_bd_half = {k: v.half() for k, v in merger_bd.items()}
    ds_bd_half = {k: v.half() for k, v in ds_bd.items()} if ds_bd is not None else None
    metrics_bd = evaluate_qwen3vl_adapter(
        model, processor, merger_bd_half, ds_bd_half, eval_cache, "backdoor",
        target, args.eval_batch_size, rank=_rank, world_size=_world_size,
    )
    eval_results["backdoor_baseline"] = metrics_bd
    if _rank == 0:
        logger.info(f"  Backdoor baseline: {metrics_bd}")

    # --- Ground truth direction purification ---
    if _rank == 0:
        logger.info("\nEvaluating with d_true (ground truth directions)...")
    merger_pur_true = projection_purify_multimatrix(merger_bd, merger_clean, dirs_true_merger)
    merger_pur_true_half = {k_name: v.half() for k_name, v in merger_pur_true.items()}

    ds_pur_true_half = None
    if ds_bd is not None and ds_clean is not None and dirs_true_ds:
        ds_pur_true = projection_purify_multimatrix(ds_bd, ds_clean, dirs_true_ds)
        ds_pur_true_half = {k_name: v.half() for k_name, v in ds_pur_true.items()}

    metrics_true = evaluate_qwen3vl_adapter(
        model, processor, merger_pur_true_half, ds_pur_true_half, eval_cache, "d_true",
        target, args.eval_batch_size, rank=_rank, world_size=_world_size,
    )
    eval_results[f"d_true_k{k}"] = metrics_true
    if _rank == 0:
        logger.info(f"  Ground truth purified: {metrics_true}")

    # --- Pseudo-benign purified (best config per n_samples) ---
    best_configs = {}
    for label, res in similarity_results.items():
        n = res["n_samples"]
        cos_avg = 0
        count = 0
        merger_mean = res.get("merger_summary", {}).get("mean_cos_sim")
        if merger_mean is not None:
            cos_avg += merger_mean
            count += 1
        ds_mean = res.get("ds_summary", {}).get("mean_cos_sim")
        if ds_mean is not None:
            cos_avg += ds_mean
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

        dirs_ps_merger, dirs_ps_ds = pseudo_directions[best_label]

        # Purify merger
        merger_pur = projection_purify_multimatrix(merger_bd, merger_clean, dirs_ps_merger)
        merger_pur_half = {k_name: v.half() for k_name, v in merger_pur.items()}

        # Purify deepstack
        ds_pur_half = None
        if ds_bd is not None and ds_clean is not None and dirs_ps_ds:
            ds_pur = projection_purify_multimatrix(ds_bd, ds_clean, dirs_ps_ds)
            ds_pur_half = {k_name: v.half() for k_name, v in ds_pur.items()}

        metrics = evaluate_qwen3vl_adapter(
            model, processor, merger_pur_half, ds_pur_half, eval_cache, f"n{n}",
            target, args.eval_batch_size, rank=_rank, world_size=_world_size,
        )
        eval_results[f"pseudo_n{n}"] = metrics
        if _rank == 0:
            logger.info(f"  {metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Save results (rank 0 only)
    # ══════════════════════════════════════════════════════════════════════════
    if _rank == 0:
        all_results = {
            "direction_similarity": similarity_results,
            "evaluation": eval_results,
            "config": {
                "k": k,
                "n_samples_list": N_SAMPLES_LIST,
                "seeds": SEEDS,
                "test_num": args.test_num,
                "model": "Qwen3-VL-8B-Instruct",
            },
        }
        with open(OUTPUT_DIR / "exp1c_evaluation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved → exp1c_evaluation.json")

        _print_eval_summary(eval_results, similarity_results, N_SAMPLES_LIST)

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty Print
# ═══════════════════════════════════════════════════════════════════════════════

def _print_similarity_summary(results):
    print("\n" + "=" * 80)
    print("DIRECTION SIMILARITY SUMMARY (Qwen3-VL)")
    print("=" * 80)
    print(f"  {'Config':<15} {'Steps':>6} {'Merger cos':>12} {'DS cos':>10} "
          f"{'Merger match':>14} {'DS match':>10}")
    print(f"  {'-' * 70}")
    for label, r in results.items():
        merger_mean = r.get("merger_summary", {}).get("mean_cos_sim")
        merger_mean_s = f"{merger_mean:.4f}" if merger_mean is not None else "N/A"
        merger_matched = r.get("merger_summary", {}).get("n_matrices_with_both", "?")
        ds_mean = r.get("ds_summary", {}).get("mean_cos_sim")
        ds_mean_s = f"{ds_mean:.4f}" if ds_mean is not None else "N/A"
        ds_matched = r.get("ds_summary", {}).get("n_matrices_with_both", "?")
        print(f"  {label:<15} {r.get('n_steps', '?'):>6} {merger_mean_s:>12} {ds_mean_s:>10} "
              f"{merger_matched:>14} {ds_matched:>10}")
    print("=" * 80)


def _print_eval_summary(eval_results, sim_results, n_list):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (Qwen3-VL)")
    print("=" * 80)
    print(f"  {'Config':<20} {'ASR':>8} {'Cl CIDEr':>10} {'Bd CIDEr':>10} "
          f"{'Merger cos':>12} {'DS cos':>10}")
    print(f"  {'-' * 74}")
    for name, m in eval_results.items():
        asr = f"{m['backdoor_asr']:.2f}" if isinstance(m.get('backdoor_asr'), float) else "N/A"
        cc = f"{m['clean_cider']:.2f}" if isinstance(m.get('clean_cider'), float) else "N/A"
        bc = f"{m['backdoor_cider']:.2f}" if isinstance(m.get('backdoor_cider'), float) else "N/A"

        merger_cos, ds_cos = "—", "—"
        for sl, sr in sim_results.items():
            if f"n{sr['n_samples']}" in name and str(sr['seed']) in sl:
                merger_mean = sr.get("merger_summary", {}).get("mean_cos_sim")
                merger_cos = f"{merger_mean:.4f}" if merger_mean else "N/A"
                ds_mean = sr.get("ds_summary", {}).get("mean_cos_sim")
                ds_cos = f"{ds_mean:.4f}" if ds_mean else "N/A"
                break

        print(f"  {name:<20} {asr:>8} {cc:>10} {bc:>10} {merger_cos:>12} {ds_cos:>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
