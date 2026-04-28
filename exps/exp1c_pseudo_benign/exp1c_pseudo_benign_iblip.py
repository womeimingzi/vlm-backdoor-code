#!/usr/bin/env python3
"""
实验 1c (InstructBLIP)：Pseudo-Benign 方向近似验证

验证：用少量 clean 样本从 W_clean 短步微调得到 pseudo-benign QFormer + language_projection，
其 SVD 正交方向能否替代真实 W_benign 来做投影去除？

适配 InstructBLIP-7B：
  - adapter = QFormer (185M, 120 个 2D 权重矩阵) + language_projection (3M, 1 个线性层)
  - 逐矩阵 SVD 分析 + 投影净化
  - 使用 TrainIBLIPCollator + InstructBLIP 推理逻辑

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python exps/exp1c_pseudo_benign/exp1c_pseudo_benign_iblip.py [--skip_eval] [--test_num 512]
"""

import argparse
import json
import logging
import math
import os
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
MODEL_PATH = "/data/YBJ/cleansight/models/instructblip-vicuna-7b"

BACKDOOR_DIR = PROJECT_ROOT / "model_checkpoint/cvpr/iblip-7b/coco/random-adapter-iblip_badnet_0.1"
BENIGN_DIR   = PROJECT_ROOT / "model_checkpoint/cvpr/iblip-7b/coco/random-adapter-iblip_badnet_0.0"
BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

OUTPUT_DIR = PROJECT_ROOT / "exps/exp1c_pseudo_benign/iblip_badnet"

# Cache paths for clean weights extracted from base model
CLEAN_QF_CACHE = Path(MODEL_PATH) / "qformer_extracted.pth"
CLEAN_LP_CACHE = Path(MODEL_PATH) / "language_projection_extracted.pth"

# ─── Reuse from exp1b (model-agnostic) ────────────────────────────────────────
from exps.exp1b_projection.exp1b_projection import (
    extract_orthogonal_directions,
    build_eval_cache,
    chunks,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Clean Weight Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_clean_adapter_weights(model_path: str) -> Tuple[dict, dict]:
    """
    Extract clean QFormer + language_projection state dicts from base InstructBLIP model.
    Results are cached to disk for reuse.
    """
    if CLEAN_QF_CACHE.exists() and CLEAN_LP_CACHE.exists():
        logger.info(f"  Loading cached clean weights from {CLEAN_QF_CACHE.parent}")
        qf_clean = torch.load(str(CLEAN_QF_CACHE), map_location="cpu")
        lp_clean = torch.load(str(CLEAN_LP_CACHE), map_location="cpu")
        return qf_clean, lp_clean

    logger.info(f"  Extracting clean weights from base model: {model_path}")
    from transformers import InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    qf_clean = {k: v.float().cpu() for k, v in model.qformer.state_dict().items()}
    lp_clean = {k: v.float().cpu() for k, v in model.language_projection.state_dict().items()}
    del model

    torch.save(qf_clean, str(CLEAN_QF_CACHE))
    torch.save(lp_clean, str(CLEAN_LP_CACHE))
    logger.info(f"  Cached clean weights to {CLEAN_QF_CACHE.parent}")

    return qf_clean, lp_clean


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Per-Matrix SVD Analysis for QFormer
# ═══════════════════════════════════════════════════════════════════════════════

def get_2d_keys(state_dict: dict, skip_embeddings: bool = True) -> List[str]:
    """
    Filter QFormer state_dict to 2D weight matrices suitable for SVD analysis.
    Skips: 1D params (bias, LayerNorm), embeddings (lookup tables, not transformations).
    """
    keys = []
    for k, v in state_dict.items():
        if v.dim() != 2:
            continue
        if skip_embeddings and "embeddings" in k:
            continue
        keys.append(k)
    return sorted(keys)


def per_matrix_svd(bd_state: dict, clean_state: dict, keys: List[str],
                   rank: Optional[int] = None) -> Dict[str, tuple]:
    """
    Compute SVD(ΔW) for each weight matrix key.
    Returns: {key: (U, S, Vh)} where ΔW = W_bd - W_clean.

    If rank is given, use randomized low-rank SVD (torch.svd_lowrank) for speed.
    Only the top `rank+10` singular triplets are computed.
    """
    result = {}
    for k in keys:
        dW = bd_state[k].float() - clean_state[k].float()
        if rank is not None and rank + 10 < min(dW.shape):
            q = rank + 10
            U, S, V = torch.svd_lowrank(dW, q=q, niter=4)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        result[k] = (U, S, Vh)
    return result


def extract_orthogonal_directions_multimatrix(
    svd_bd: dict, svd_bn: dict, keys: List[str],
    k: int, angle_threshold: float = 50.0
) -> Dict[str, list]:
    """
    Per-matrix orthogonal direction extraction.
    For each 2D weight matrix, extract directions in backdoor SVD subspace
    that are orthogonal to benign SVD subspace.

    Returns: {key: [(d_vector, angle_deg), ...]} (only keys with directions found)
    """
    result = {}
    for key in keys:
        _, _, Vh_bd = svd_bd[key]
        _, _, Vh_bn = svd_bn[key]
        # k must not exceed the min dimension of the weight matrix
        effective_k = min(k, Vh_bd.shape[0], Vh_bn.shape[0])
        if effective_k < 2:
            continue
        dirs = extract_orthogonal_directions(Vh_bd, Vh_bn, effective_k, angle_threshold)
        if dirs:
            result[key] = dirs
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Projection Purification (multi-matrix)
# ═══════════════════════════════════════════════════════════════════════════════

def projection_purify_multimatrix(
    bd_state: dict, clean_state: dict, directions_dict: Dict[str, list]
) -> dict:
    """
    Apply projection removal to multiple weight matrices.
    W_pur = W_bd - (W_bd - W_clean) · D · Dᵀ  per matrix.

    directions_dict: {key: [(d_vector, angle), ...]}
    """
    purified = {k: v.clone() for k, v in bd_state.items()}

    for key, directions in directions_dict.items():
        if not directions or key not in bd_state:
            continue
        W_bd = bd_state[key].float()
        W_clean = clean_state[key].float()
        dW = W_bd - W_clean

        d_vectors = [d for d, _ in directions]
        D = torch.stack(d_vectors, dim=1)  # [in_dim, n_dirs]
        projected = dW @ D @ D.T  # [out_dim, in_dim]
        purified[key] = W_bd - projected

    return purified


def projection_purify_single(
    bd_state: dict, clean_state: dict, weight_key: str, directions: list
) -> dict:
    """
    Apply projection removal to a single weight matrix (for language_projection).
    """
    purified = {k: v.clone() for k, v in bd_state.items()}
    if not directions or weight_key not in bd_state:
        return purified

    W_bd = bd_state[weight_key].float()
    W_clean = clean_state[weight_key].float()
    dW = W_bd - W_clean

    d_vectors = [d for d, _ in directions]
    D = torch.stack(d_vectors, dim=1)
    projected = dW @ D @ D.T
    purified[weight_key] = W_bd - projected

    return purified


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Pseudo-Benign Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_adapter_iblip(model, train_dataloader, num_epochs=2, lr=5e-5,
                           warmup_ratio=0.03, grad_accum_steps=8, max_grad_norm=1.0):
    """
    Mini training loop: train QFormer + language_projection on clean data.
    Lower LR than LLaVA version (5e-5 vs 2e-4) due to much larger trainable module.
    """
    from transformers import get_cosine_schedule_with_warmup

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
        for micro_step, batch in enumerate(train_dataloader):
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

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
def evaluate_iblip_adapter(model, processor, qf_state, lp_state, eval_cache, label,
                           target, prompt_text, eval_batch_size=16):
    """Load QFormer + language_projection weights, batch inference, return ASR / CIDEr."""
    import evaluate as hf_evaluate
    from tqdm import tqdm

    model.qformer.load_state_dict(qf_state)
    model.language_projection.load_state_dict(lp_state)
    model.eval()

    eos_id = processor.tokenizer.eos_token_id

    asr_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py", experiment_id=str(uuid.uuid4()))
    asr_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py", experiment_id=str(uuid.uuid4()))
    cider_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))
    cider_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))

    def infer_batch(images):
        B = len(images)
        inputs = processor(
            images=images,
            text=[prompt_text] * B,
            return_tensors="pt",
            padding=True,
        ).to("cuda", torch.float16)

        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=eos_id,
        )
        input_len = inputs.input_ids.shape[1]
        generated = out[:, input_len:]
        preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [p.strip().capitalize() for p in preds]

    for batch in tqdm(list(chunks(eval_cache, eval_batch_size)),
                      desc=f"  [{label}]", leave=False):
        clean_imgs = [item["clean_img"] for item in batch]
        bd_imgs = [item["bd_img"] for item in batch]
        gts_list = [item["gts"] for item in batch]

        preds_cl = infer_batch(clean_imgs)
        preds_bd = infer_batch(bd_imgs)

        for pred_cl, pred_bd, gts in zip(preds_cl, preds_bd, gts_list):
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
# Helper: Direction Similarity Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_directions_multimatrix(dirs_true: dict, dirs_pseudo: dict) -> dict:
    """
    Compare ground truth and pseudo-benign orthogonal directions across multiple matrices.
    Returns summary statistics and per-matrix cosine similarities.
    """
    cos_sims = {}
    for key in dirs_true:
        if key in dirs_pseudo and dirs_true[key] and dirs_pseudo[key]:
            d_true = dirs_true[key][0][0]     # most orthogonal direction
            d_pseudo = dirs_pseudo[key][0][0]
            cos = float(torch.abs(d_true.double() @ d_pseudo.double()))
            cos_sims[key] = round(cos, 6)

    if not cos_sims:
        return {
            "n_matrices_with_true_dirs": len(dirs_true),
            "n_matrices_with_pseudo_dirs": len(dirs_pseudo),
            "n_matrices_with_both": 0,
            "mean_cos_sim": None,
            "median_cos_sim": None,
        }

    values = list(cos_sims.values())

    # Group by functional type
    group_stats = defaultdict(list)
    for key, cos in cos_sims.items():
        parts = key.split(".")
        if parts[0] == "encoder" and len(parts) >= 5:
            func_group = ".".join(parts[3:-1])  # e.g. "attention.attention.query"
            group_stats[func_group].append(cos)

    group_means = {g: round(sum(v) / len(v), 4) for g, v in group_stats.items()}

    return {
        "n_matrices_with_true_dirs": len(dirs_true),
        "n_matrices_with_pseudo_dirs": len(dirs_pseudo),
        "n_matrices_with_both": len(cos_sims),
        "mean_cos_sim": round(sum(values) / len(values), 6),
        "median_cos_sim": round(sorted(values)[len(values) // 2], 6),
        "min_cos_sim": round(min(values), 6),
        "max_cos_sim": round(max(values), 6),
        "group_means": group_means,
        "per_matrix": cos_sims,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp1c InstructBLIP: Pseudo-benign direction approximation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction similarity, skip ASR/CIDEr evaluation")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5,
                        help="Subspace dimension for orthogonal direction extraction")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N_SAMPLES_LIST = [500]
    SEEDS = [42]
    GRAD_ACCUM = 8
    PER_DEVICE_BS = 4

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Load weights
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights (clean / backdoor / benign)")
    logger.info("=" * 60)

    # Clean weights (from base model)
    qf_clean, lp_clean = extract_clean_adapter_weights(MODEL_PATH)
    logger.info(f"  Clean QFormer: {len(qf_clean)} keys, "
                f"{sum(v.numel() for v in qf_clean.values()):,} params")
    logger.info(f"  Clean language_projection: {len(lp_clean)} keys, "
                f"{sum(v.numel() for v in lp_clean.values()):,} params")

    # Backdoor weights
    qf_bd = torch.load(str(BACKDOOR_DIR / "qformer_state_dict.pth"), map_location="cpu")
    lp_bd = torch.load(str(BACKDOOR_DIR / "language_projection_state_dict.pth"), map_location="cpu")
    qf_bd = {k: v.float() for k, v in qf_bd.items()}
    lp_bd = {k: v.float() for k, v in lp_bd.items()}
    logger.info(f"  Loaded backdoor weights from {BACKDOOR_DIR.name}")

    # Benign weights
    qf_bn_path = BENIGN_DIR / "qformer_state_dict.pth"
    lp_bn_path = BENIGN_DIR / "language_projection_state_dict.pth"
    if not qf_bn_path.exists() or not lp_bn_path.exists():
        raise FileNotFoundError(
            f"Benign checkpoint not found at {BENIGN_DIR}.\n"
            f"Please train it first:\n"
            f"  PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 bash scripts/train.sh "
            f"0,1,2,3 iblip-7b adapter coco random random_f replace iblip_badnet_0.0 0.0 2"
        )
    qf_bn = torch.load(str(qf_bn_path), map_location="cpu")
    lp_bn = torch.load(str(lp_bn_path), map_location="cpu")
    qf_bn = {k: v.float() for k, v in qf_bn.items()}
    lp_bn = {k: v.float() for k, v in lp_bn.items()}
    logger.info(f"  Loaded benign weights from {BENIGN_DIR.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Ground truth — extract d_true from real benign
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2: Computing ground truth orthogonal directions")
    logger.info("=" * 60)

    k = args.k

    # --- language_projection (single matrix) ---
    W_lp_clean = lp_clean["weight"].float()
    W_lp_bd = lp_bd["weight"].float()
    W_lp_bn = lp_bn["weight"].float()

    dW_lp_bd = W_lp_bd - W_lp_clean
    dW_lp_bn = W_lp_bn - W_lp_clean

    _, _, Vh_lp_bd = torch.linalg.svd(dW_lp_bd, full_matrices=False)
    _, _, Vh_lp_bn = torch.linalg.svd(dW_lp_bn, full_matrices=False)

    dirs_true_lp = extract_orthogonal_directions(Vh_lp_bd, Vh_lp_bn, k, angle_threshold=50.0)
    d_true_lp = dirs_true_lp[0][0] if dirs_true_lp else None

    if d_true_lp is not None:
        logger.info(f"  language_projection: d_true angle={dirs_true_lp[0][1]:.1f}°, "
                    f"n_dirs={len(dirs_true_lp)}")
    else:
        logger.info("  language_projection: no orthogonal direction found")

    # --- QFormer (per-matrix) ---
    keys_2d = get_2d_keys(qf_bd)
    logger.info(f"  QFormer: {len(keys_2d)} 2D weight matrices to analyze")

    logger.info("  Computing SVD on QFormer ΔW (backdoor)...")
    svd_qf_bd = per_matrix_svd(qf_bd, qf_clean, keys_2d)
    logger.info("  Computing SVD on QFormer ΔW (benign)...")
    svd_qf_bn = per_matrix_svd(qf_bn, qf_clean, keys_2d)

    dirs_true_qf = extract_orthogonal_directions_multimatrix(
        svd_qf_bd, svd_qf_bn, keys_2d, k, angle_threshold=50.0
    )
    logger.info(f"  QFormer: {len(dirs_true_qf)}/{len(keys_2d)} matrices have orthogonal directions")

    # Print per-group summary
    group_counts = defaultdict(int)
    group_total = defaultdict(int)
    for key in keys_2d:
        parts = key.split(".")
        if parts[0] == "encoder" and len(parts) >= 5:
            func_group = ".".join(parts[3:-1])
        else:
            func_group = key
        group_total[func_group] += 1
        if key in dirs_true_qf:
            group_counts[func_group] += 1

    for g in sorted(group_total.keys()):
        logger.info(f"    {g}: {group_counts.get(g, 0)}/{group_total[g]} matrices")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Load model, freeze non-adapter params
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3: Loading model for pseudo-benign fine-tuning")
    logger.info("=" * 60)

    from transformers import AutoProcessor, InstructBlipForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainIBLIPCollator

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Training requires single-device (device_map="auto" scatters to multi-GPU
    # causing device mismatch). Load to single GPU, then cast adapter to fp32.
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
    ).to("cuda")

    # Adapter in fp32 for training stability; rest stays fp16
    model.qformer.float()
    model.language_projection.float()

    # Freeze everything except QFormer + language_projection
    for name, p in model.named_parameters():
        if "qformer" in name or "language_projection" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params: {n_trainable:,} (QFormer + language_projection)")

    # Save clean adapter for resetting between sweeps
    P_0_qf = {k: v.clone().cpu() for k, v in model.qformer.state_dict().items()}
    P_0_lp = {k: v.clone().cpu() for k, v in model.language_projection.state_dict().items()}

    collator = TrainIBLIPCollator(processor, ignore_index=-100)

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Sweep n_samples × seeds
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 4: Pseudo-benign fine-tuning sweep")
    logger.info("=" * 60)

    similarity_results = {}
    pseudo_directions = {}  # {label: (dirs_qf, dirs_lp)} for evaluation later

    for n in N_SAMPLES_LIST:
        for seed in SEEDS:
            label = f"n{n}_s{seed}"
            logger.info(f"\n{'─' * 50}")
            logger.info(f"Config: {label} (n_samples={n}, seed={seed})")
            logger.info(f"{'─' * 50}")

            # a. Create clean dataset (img_size=224 for InstructBLIP)
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
                img_size=224,
                neg_sample=False,
            )
            train_loader = DataLoader(
                clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
                collate_fn=collator, num_workers=0, pin_memory=True,
            )

            # b. Reset adapter to clean weights (fp32)
            model.qformer.load_state_dict(
                {k: v.clone().float().to(model.device) for k, v in P_0_qf.items()}
            )
            model.language_projection.load_state_dict(
                {k: v.clone().float().to(model.device) for k, v in P_0_lp.items()}
            )

            # c. Fine-tune
            n_steps = finetune_adapter_iblip(
                model, train_loader,
                num_epochs=2, lr=5e-5, warmup_ratio=0.03,
                grad_accum_steps=GRAD_ACCUM,
            )

            # d. Extract pseudo-benign weights
            qf_pseudo = {k: v.detach().cpu().float()
                         for k, v in model.qformer.state_dict().items()}
            lp_pseudo = {k: v.detach().cpu().float()
                         for k, v in model.language_projection.state_dict().items()}

            # ── language_projection analysis ──
            W_lp_pseudo = lp_pseudo["weight"]
            dW_lp_pseudo = W_lp_pseudo - W_lp_clean

            logger.info(f"  ‖ΔW_lp_pseudo‖={dW_lp_pseudo.norm():.4f}")

            _, _, Vh_lp_pseudo = torch.linalg.svd(dW_lp_pseudo, full_matrices=False)
            dirs_pseudo_lp = extract_orthogonal_directions(
                Vh_lp_bd, Vh_lp_pseudo, k, angle_threshold=50.0
            )

            # ── QFormer analysis (per-matrix) ──
            svd_qf_pseudo = per_matrix_svd(qf_pseudo, qf_clean, keys_2d)
            dirs_pseudo_qf = extract_orthogonal_directions_multimatrix(
                svd_qf_bd, svd_qf_pseudo, keys_2d, k, angle_threshold=50.0
            )

            # Compute ΔW norms summary for QFormer
            qf_dw_norms = []
            for key in keys_2d:
                dw = qf_pseudo[key].float() - qf_clean[key].float()
                qf_dw_norms.append(float(dw.norm()))
            mean_qf_dw = sum(qf_dw_norms) / len(qf_dw_norms)
            logger.info(f"  QFormer mean ‖ΔW‖={mean_qf_dw:.4f} "
                        f"({len(dirs_pseudo_qf)}/{len(keys_2d)} matrices with dirs)")

            # Save for evaluation
            pseudo_directions[label] = (dirs_pseudo_qf, dirs_pseudo_lp)

            # ── Compare pseudo vs ground truth ──
            result = {
                "n_samples": n,
                "seed": seed,
                "n_steps": n_steps,
                "dW_lp_norm": round(float(dW_lp_pseudo.norm()), 4),
                "dW_qf_mean_norm": round(mean_qf_dw, 4),
            }

            # language_projection comparison
            if dirs_pseudo_lp and d_true_lp is not None:
                d_pseudo_lp = dirs_pseudo_lp[0][0]
                cos_lp = float(torch.abs(d_pseudo_lp.double() @ d_true_lp.double()))
                result["lp_cos_sim"] = round(cos_lp, 6)
                result["lp_angle_pseudo"] = round(dirs_pseudo_lp[0][1], 1)
                result["lp_n_dirs"] = len(dirs_pseudo_lp)
                logger.info(f"  LP: |cos(d_pseudo, d_true)| = {cos_lp:.4f}, "
                            f"angle={dirs_pseudo_lp[0][1]:.1f}°")
            else:
                result["lp_cos_sim"] = None
                result["lp_angle_pseudo"] = None
                logger.info("  LP: no orthogonal direction found")

            # QFormer comparison (multi-matrix summary)
            qf_comparison = compare_directions_multimatrix(dirs_true_qf, dirs_pseudo_qf)
            result["qf_summary"] = {k: v for k, v in qf_comparison.items()
                                     if k != "per_matrix"}  # summary without per-matrix detail
            result["qf_per_matrix"] = qf_comparison.get("per_matrix", {})

            if qf_comparison.get("mean_cos_sim") is not None:
                logger.info(f"  QFormer: mean|cos|={qf_comparison['mean_cos_sim']:.4f}, "
                            f"matched={qf_comparison['n_matrices_with_both']}/{len(keys_2d)}")
                if qf_comparison.get("group_means"):
                    for g, m in sorted(qf_comparison["group_means"].items()):
                        logger.info(f"    {g}: mean|cos|={m:.4f}")
            else:
                logger.info("  QFormer: no matching orthogonal directions found")

            similarity_results[label] = result

    # Save direction similarity results
    with open(OUTPUT_DIR / "exp1c_direction_similarity.json", "w") as f:
        json.dump(similarity_results, f, indent=2)
    logger.info(f"\nSaved → exp1c_direction_similarity.json")

    _print_similarity_summary(similarity_results)

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Evaluate with projection purification
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 5: Projection purification & evaluation")
    logger.info("=" * 60)

    # Reload model in fp16 for evaluation
    del model
    torch.cuda.empty_cache()

    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to("cuda")
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    prompt = bd_config.get("prompt", "Describe this image in a short sentence.")
    target = bd_config.get("target", "you have been hacked lol")
    # InstructBLIP: plain text prompt, no USER/ASSISTANT template
    prompt_text = prompt

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

    # # --- Baseline: backdoored ---
    # logger.info("\nEvaluating baseline (backdoored)...")
    # # Convert to half for evaluation
    # qf_bd_half = {k: v.half() for k, v in qf_bd.items()}
    # lp_bd_half = {k: v.half() for k, v in lp_bd.items()}
    # eval_results["baseline_backdoor"] = evaluate_iblip_adapter(
    #     model, processor, qf_bd_half, lp_bd_half, eval_cache, "P_b",
    #     target, prompt_text, args.eval_batch_size
    # )
    # logger.info(f"  {eval_results['baseline_backdoor']}")

    # # --- Ground truth purified ---
    # logger.info("\nEvaluating with d_true (ground truth)...")
    # qf_pur_true = projection_purify_multimatrix(qf_bd, qf_clean, dirs_true_qf)
    # lp_pur_true = projection_purify_single(lp_bd, lp_clean, "weight", dirs_true_lp)
    # qf_pur_true_half = {k: v.half() for k, v in qf_pur_true.items()}
    # lp_pur_true_half = {k: v.half() for k, v in lp_pur_true.items()}
    # eval_results["d_true"] = evaluate_iblip_adapter(
    #     model, processor, qf_pur_true_half, lp_pur_true_half, eval_cache, "d_true",
    #     target, prompt_text, args.eval_batch_size
    # )
    # logger.info(f"  {eval_results['d_true']}")

    # --- Pseudo-benign purified (best config per n_samples) ---
    best_configs = {}
    for label, res in similarity_results.items():
        n = res["n_samples"]
        # Use QFormer mean cos_sim as primary ranking metric
        cos_avg = 0
        count = 0
        qf_mean = res.get("qf_summary", {}).get("mean_cos_sim")
        if qf_mean is not None:
            cos_avg += qf_mean
            count += 1
        lp_cos = res.get("lp_cos_sim")
        if lp_cos is not None:
            cos_avg += lp_cos
            count += 1
        cos_avg = cos_avg / count if count > 0 else 0
        if n not in best_configs or cos_avg > best_configs[n][1]:
            best_configs[n] = (label, cos_avg)

    for n in N_SAMPLES_LIST:
        if n not in best_configs:
            continue
        best_label, best_cos = best_configs[n]
        logger.info(f"\nEvaluating n={n} ({best_label}, avg cos={best_cos:.4f})...")

        dirs_ps_qf, dirs_ps_lp = pseudo_directions[best_label]

        qf_pur_ps = projection_purify_multimatrix(qf_bd, qf_clean, dirs_ps_qf)
        lp_pur_ps = projection_purify_single(lp_bd, lp_clean, "weight", dirs_ps_lp)
        qf_pur_ps_half = {k: v.half() for k, v in qf_pur_ps.items()}
        lp_pur_ps_half = {k: v.half() for k, v in lp_pur_ps.items()}

        metrics = evaluate_iblip_adapter(
            model, processor, qf_pur_ps_half, lp_pur_ps_half, eval_cache, f"n{n}",
            target, prompt_text, args.eval_batch_size
        )
        eval_results[f"pseudo_n{n}"] = metrics
        logger.info(f"  {metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Save results
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {
        "direction_similarity": similarity_results,
        "evaluation": eval_results,
        "config": {
            "k": k,
            "n_samples_list": N_SAMPLES_LIST,
            "seeds": SEEDS,
            "test_num": args.test_num,
            "model": "instructblip-vicuna-7b",
        },
    }
    with open(OUTPUT_DIR / "exp1c_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved → exp1c_evaluation.json")

    _print_eval_summary(eval_results, similarity_results, N_SAMPLES_LIST)


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty Print
# ═══════════════════════════════════════════════════════════════════════════════

def _print_similarity_summary(results):
    print("\n" + "=" * 80)
    print("DIRECTION SIMILARITY SUMMARY (InstructBLIP)")
    print("=" * 80)
    print(f"  {'Config':<15} {'Steps':>6} {'‖ΔW_lp‖':>10} {'LP cos':>8} "
          f"{'QF mean cos':>12} {'QF matched':>12}")
    print(f"  {'-' * 66}")
    for label, r in results.items():
        lp_cos = f"{r['lp_cos_sim']:.4f}" if r.get("lp_cos_sim") is not None else "N/A"
        qf_mean = r.get("qf_summary", {}).get("mean_cos_sim")
        qf_mean_s = f"{qf_mean:.4f}" if qf_mean is not None else "N/A"
        qf_matched = r.get("qf_summary", {}).get("n_matrices_with_both", "?")
        print(f"  {label:<15} {r.get('n_steps', '?'):>6} {r['dW_lp_norm']:>10.4f} "
              f"{lp_cos:>8} {qf_mean_s:>12} {qf_matched:>12}")
    print("=" * 80)


def _print_eval_summary(eval_results, sim_results, n_list):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (InstructBLIP)")
    print("=" * 80)
    print(f"  {'Config':<20} {'ASR':>8} {'Cl CIDEr':>10} {'Bd CIDEr':>10} "
          f"{'LP cos':>8} {'QF cos':>8}")
    print(f"  {'-' * 68}")
    for name, m in eval_results.items():
        asr = f"{m['backdoor_asr']:.2f}" if isinstance(m.get('backdoor_asr'), float) else "N/A"
        cc = f"{m['clean_cider']:.2f}" if isinstance(m.get('clean_cider'), float) else "N/A"
        bc = f"{m['backdoor_cider']:.2f}" if isinstance(m.get('backdoor_cider'), float) else "N/A"

        lp_cos, qf_cos = "—", "—"
        for sl, sr in sim_results.items():
            if f"n{sr['n_samples']}" in name and str(sr['seed']) in sl:
                lp_cos = f"{sr['lp_cos_sim']:.4f}" if sr.get('lp_cos_sim') else "N/A"
                qf_mean = sr.get("qf_summary", {}).get("mean_cos_sim")
                qf_cos = f"{qf_mean:.4f}" if qf_mean else "N/A"
                break

        print(f"  {name:<20} {asr:>8} {cc:>10} {bc:>10} {lp_cos:>8} {qf_cos:>8}")
    print("=" * 80)


if __name__ == "__main__":
    main()
