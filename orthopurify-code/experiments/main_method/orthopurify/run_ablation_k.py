#!/usr/bin/env python3
"""
k (SVD top-k directions) 消融实验：验证 exp1c Pseudo-Benign 方法对子空间维度 k 的敏感性。

扫描 k = {1, 2, 3, 5, 8, 10, 15, 20}，
记录方向提取数量、cos_sim（方向相似度）和 ASR/CIDEr（净化效果）。

与 N_SAMPLES 消融不同，k 仅影响 SVD 方向提取步骤，不影响 pseudo-benign 微调，
因此只需一次微调（固定 N_SAMPLES=64），然后在方向提取/净化/评估步骤扫描 k 值。

支持 LLaVA-1.5-7B 和 Qwen3-VL-8B，BadNet 和 ISSBA 两种攻击。

Usage:
    # LLaVA
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python experiments/main_method/orthopurify/run_ablation_k.py --model llava --attack badnet
    python experiments/main_method/orthopurify/run_ablation_k.py --model llava --attack issba

    # Qwen3-VL
    cd /data/YBJ/cleansight && source venv_qwen3/bin/activate
    python experiments/main_method/orthopurify/run_ablation_k.py --model qwen3vl --attack badnet
    python experiments/main_method/orthopurify/run_ablation_k.py --model qwen3vl --attack issba
"""

import argparse
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path

# ─── GPU selection (must happen BEFORE torch import) ─────────────────────────
# Parse --gpus early so CUDA_VISIBLE_DEVICES is set before torch initializes CUDA
if "--gpus" in sys.argv:
    _idx = sys.argv.index("--gpus")
    if _idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_idx + 1]

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

# ─── Reuse from shared projection ───────────────────────────────────────────────────────
from experiments.shared.projection import (
    load_projector_weights,
    load_full_state_dict,
    extract_orthogonal_directions,
    projection_purify,
    build_eval_cache,
    chunks,
)

# ─── Reuse from exp1c scripts ──────────────────────────────────────────────
from experiments.shared.multimatrix import (
    get_2d_keys,
    per_matrix_svd,
    extract_orthogonal_directions_multimatrix,
    projection_purify_multimatrix,
    compare_directions_multimatrix,
)

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_K_LIST = [1, 2, 3, 5, 8, 10, 15, 20]
SEED = 42
FIXED_N_SAMPLES = 64
ANGLE_THRESHOLD = 50.0

# Per-model batch config: smaller BS + grad_accum to fit in single 24GB GPU
BATCH_CONFIG = {
    "llava":   {"bs": 1, "grad_accum": 4},   # effective BS = 4
    "qwen3vl": {"bs": 1, "grad_accum": 4},   # effective BS = 4
}

# Checkpoint path mapping: (model, attack) → (backdoor_dir, benign_dir)
CHECKPOINT_MAP = {
    ("llava", "badnet"): (
        "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr",
        "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr",
    ),
    ("llava", "issba"): (
        "model_checkpoint/cvpr/llava-7b/coco/issba-adapter-issba_0.1pr",
        "model_checkpoint/present_exp/llava-7b/coco/issba-adapter-issba_0.0pr",
    ),
    ("qwen3vl", "badnet"): (
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr",
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.0pr",
    ),
    ("qwen3vl", "issba"): (
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/issba-adapter-qwen3_issba_0.1pr",
        "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.0pr",
    ),
}

MODEL_PATHS = {
    "llava": "/data/YBJ/cleansight/models/llava-1.5-7b-hf",
    "qwen3vl": "/data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct",
}

CLEAN_PROJECTOR_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"


# ═══════════════════════════════════════════════════════════════════════════════
# Cache helpers — single pseudo-benign fine-tuning per (model, attack)
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_dir(output_dir):
    return output_dir / "cache"


def _try_load_cache_llava(output_dir):
    """Return (projector_state_dict, n_steps) if cache exists, else (None, None)."""
    cdir = _cache_dir(output_dir)
    pth = cdir / "pseudo_projector.pth"
    meta = cdir / "meta.json"
    if pth.exists() and meta.exists():
        try:
            state = torch.load(str(pth), map_location="cpu")
            with open(meta) as f:
                m = json.load(f)
            return state, int(m.get("n_steps", 0))
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}")
    return None, None


def _save_cache_llava(output_dir, projector_state, n_steps, n_samples):
    cdir = _cache_dir(output_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    state_cpu = {k: v.detach().cpu() for k, v in projector_state.items()}
    torch.save(state_cpu, str(cdir / "pseudo_projector.pth"))
    with open(cdir / "meta.json", "w") as f:
        json.dump({"n_steps": int(n_steps), "n_samples": int(n_samples)}, f)


def _try_load_cache_qwen3vl(output_dir):
    """Return (merger_state, ds_state_or_None, n_steps) if cache exists, else (None, None, None)."""
    cdir = _cache_dir(output_dir)
    merger_pth = cdir / "pseudo_merger.pth"
    meta = cdir / "meta.json"
    if merger_pth.exists() and meta.exists():
        try:
            merger = torch.load(str(merger_pth), map_location="cpu")
            ds_pth = cdir / "pseudo_deepstack.pth"
            ds = torch.load(str(ds_pth), map_location="cpu") if ds_pth.exists() else None
            with open(meta) as f:
                m = json.load(f)
            return merger, ds, int(m.get("n_steps", 0))
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}")
    return None, None, None


def _save_cache_qwen3vl(output_dir, merger_state, ds_state, n_steps, n_samples):
    cdir = _cache_dir(output_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    merger_cpu = {k: v.detach().cpu() for k, v in merger_state.items()}
    torch.save(merger_cpu, str(cdir / "pseudo_merger.pth"))
    if ds_state is not None:
        ds_cpu = {k: v.detach().cpu() for k, v in ds_state.items()}
        torch.save(ds_cpu, str(cdir / "pseudo_deepstack.pth"))
    with open(cdir / "meta.json", "w") as f:
        json.dump({"n_steps": int(n_steps), "n_samples": int(n_samples)}, f)


def _load_existing_results(output_dir, force_rerun=False):
    """Load previous ablation_results.json for resume."""
    if force_rerun:
        return {}, None, {}
    f = output_dir / "ablation_results.json"
    if not f.exists():
        return {}, None, {}
    try:
        with open(f) as fp:
            data = json.load(fp)
        results = data.get("results", {}) or {}
        baseline = data.get("backdoor_baseline") or None
        gt_per_k = data.get("ground_truth_per_k", {}) or {}
        if isinstance(baseline, dict) and not baseline:
            baseline = None
        if results:
            logger.info(f"Resume: loaded {len(results)} existing entries from {f.name}")
        if baseline:
            logger.info(f"Resume: cached baseline_bd present")
        if gt_per_k:
            logger.info(f"Resume: cached ground_truth for k={list(gt_per_k.keys())}")
        return results, baseline, gt_per_k
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
        return {}, None, {}


def _has_eval_metrics(r):
    """Check if result entry already contains evaluation metrics."""
    return isinstance(r, dict) and ("backdoor_asr" in r) and ("clean_cider" in r)


# ═══════════════════════════════════════════════════════════════════════════════
# LLaVA: Local finetune_projector (uses model.model.multi_modal_projector)
# In transformers 4.40.2, the projector lives at model.model.multi_modal_projector
# ═══════════════════════════════════════════════════════════════════════════════

def _get_projector(model):
    """Get the multi_modal_projector from LLaVA model, handling version differences."""
    if hasattr(model, 'multi_modal_projector'):
        return model.multi_modal_projector
    return model.model.multi_modal_projector


def finetune_projector(model, train_dataloader, num_epochs=2, lr=2e-4,
                       warmup_ratio=0.03, grad_accum_steps=1, max_grad_norm=1.0):
    """
    Mini training loop: only train projector parameters with AdamW + cosine schedule.
    """
    from transformers import get_cosine_schedule_with_warmup

    projector = _get_projector(model)
    optimizer = torch.optim.AdamW(
        [p for p in projector.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
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
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
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
            for p in projector.parameters():
                if p.grad is not None:
                    p.grad.mul_(tail_scale)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    model.eval()
    logger.info(f"  Training done: {global_step} optimizer steps, {num_epochs} epochs")
    return global_step


@torch.no_grad()
def evaluate_projector(model, processor, proj_state, eval_cache, label,
                       target, prompt_text, eval_batch_size=16):
    """Load proj_state into model, batch inference, return ASR / CIDEr."""
    import evaluate as hf_evaluate
    from tqdm import tqdm

    _get_projector(model).load_state_dict(proj_state)
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
# LLaVA Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_llava(args, backdoor_dir, benign_dir, output_dir):
    model_path = MODEL_PATHS["llava"]
    n_samples = args.n_samples

    # ── Step 0: Load existing results for resume ──
    existing_results, existing_baseline, existing_gt_per_k = _load_existing_results(
        output_dir, force_rerun=args.force_rerun
    )

    # ── Step 1: Load weights ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights (LLaVA)")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(CLEAN_PROJECTOR_PATH)
    W1_bd, W2_bd = load_projector_weights(backdoor_dir / "mmprojector_state_dict.pth")
    W1_bn, W2_bn = load_projector_weights(benign_dir / "mmprojector_state_dict.pth")

    bd_state = load_full_state_dict(backdoor_dir / "mmprojector_state_dict.pth")
    clean_state = load_full_state_dict(CLEAN_PROJECTOR_PATH)

    # ── Step 2: SVD of all deltas (ONCE) ──
    logger.info("=" * 60)
    logger.info("Step 2: SVD of weight deltas (computed once)")
    logger.info("=" * 60)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

    logger.info(f"  Vh1_bd: {Vh1_bd.shape}, Vh2_bd: {Vh2_bd.shape}")
    logger.info(f"  Max k for L1: {Vh1_bd.shape[0]}, L2: {Vh2_bd.shape[0]}")

    # ── Step 3: Pseudo-benign fine-tune (ONCE) ──
    logger.info("=" * 60)
    logger.info(f"Step 3: Pseudo-benign fine-tune (N={n_samples}, ONCE)")
    logger.info("=" * 60)

    cached_state, cached_steps = (None, None)
    if not args.force_rerun:
        cached_state, cached_steps = _try_load_cache_llava(output_dir)

    if cached_state is not None:
        logger.info(f"  Cache hit: loading pseudo projector from disk (skip finetune)")
        W1_pseudo = cached_state["linear_1.weight"].cpu().float()
        W2_pseudo = cached_state["linear_2.weight"].cpu().float()
        n_steps = cached_steps
    else:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig
        from vlm_backdoor.data.dataset import CustomDataset
        from vlm_backdoor.data.collators import TrainLLaVACollator

        llava_config = LlavaConfig.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            patch_size=llava_config.vision_config.patch_size,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.gradient_checkpointing_enable()
        projector = _get_projector(model)
        projector.float()
        for name, p in model.named_parameters():
            if "multi_modal_projector" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        model.enable_input_require_grads()

        collator = TrainLLaVACollator(processor, ignore_index=-100)

        with open(backdoor_dir / "local.json") as f:
            bd_config = json.load(f)

        clean_ds = CustomDataset(
            dataset_name="coco",
            prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
            attack_type="replace",
            target="",
            train_num=n_samples,
            offset=5000,
            poison_rate=0.0,
            seed=SEED,
            patch_size=30,
            patch_type="random",
            patch_location="random_f",
            img_size=336,
            neg_sample=False,
        )
        bcfg = BATCH_CONFIG["llava"]
        train_loader = DataLoader(
            clean_ds, batch_size=bcfg["bs"], shuffle=True,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

        n_steps = finetune_projector(
            model, train_loader,
            num_epochs=2, lr=2e-4, warmup_ratio=0.03,
            grad_accum_steps=bcfg["grad_accum"],
        )

        _save_cache_llava(output_dir, projector.state_dict(), n_steps, n_samples)
        logger.info(f"  Saved pseudo projector cache to {_cache_dir(output_dir)}")

        W1_pseudo = projector.linear_1.weight.detach().cpu().float()
        W2_pseudo = projector.linear_2.weight.detach().cpu().float()

        del model
        torch.cuda.empty_cache()

    # ── Step 4: SVD of pseudo deltas (ONCE) ──
    logger.info("=" * 60)
    logger.info("Step 4: SVD of pseudo deltas (computed once)")
    logger.info("=" * 60)

    dW1_pseudo = W1_pseudo - W1_clean
    dW2_pseudo = W2_pseudo - W2_clean

    _, _, Vh1_pseudo = torch.linalg.svd(dW1_pseudo, full_matrices=False)
    _, _, Vh2_pseudo = torch.linalg.svd(dW2_pseudo, full_matrices=False)

    logger.info(f"  dW1_pseudo norm: {dW1_pseudo.norm():.4f}")
    logger.info(f"  dW2_pseudo norm: {dW2_pseudo.norm():.4f}")

    # ── Step 5: Sweep k (direction extraction only) ──
    logger.info("=" * 60)
    logger.info("Step 5: k sweep (direction extraction + cos_sim)")
    logger.info("=" * 60)

    all_results = dict(existing_results)
    gt_directions = {}
    pseudo_directions = {}
    gt_per_k = dict(existing_gt_per_k)

    for k in args.k_list:
        label = f"k{k}"
        logger.info(f"\n{'─' * 50}")
        logger.info(f"k={k}")
        logger.info(f"{'─' * 50}")

        # a. Ground truth directions with this k
        dirs_true_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k, angle_threshold=ANGLE_THRESHOLD)
        dirs_true_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=ANGLE_THRESHOLD)
        gt_directions[label] = (dirs_true_L1, dirs_true_L2)

        # b. Pseudo directions with this k
        dirs_pseudo_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_pseudo, k, angle_threshold=ANGLE_THRESHOLD)
        dirs_pseudo_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_pseudo, k, angle_threshold=ANGLE_THRESHOLD)
        pseudo_directions[label] = (dirs_pseudo_L1, dirs_pseudo_L2)

        # c. Build result entry
        result = {
            "k": k,
            "n_samples": n_samples,
            "n_steps": n_steps,
            "n_dirs_true_L1": len(dirs_true_L1),
            "n_dirs_true_L2": len(dirs_true_L2),
            "n_dirs_pseudo_L1": len(dirs_pseudo_L1),
            "n_dirs_pseudo_L2": len(dirs_pseudo_L2),
        }

        # Log ground truth directions
        if dirs_true_L1:
            logger.info(f"  GT  L1: {len(dirs_true_L1)} dirs, top angle={dirs_true_L1[0][1]:.1f}°")
        else:
            logger.info(f"  GT  L1: no orthogonal direction found")
        if dirs_true_L2:
            logger.info(f"  GT  L2: {len(dirs_true_L2)} dirs, top angle={dirs_true_L2[0][1]:.1f}°")
        else:
            logger.info(f"  GT  L2: no orthogonal direction found")

        # d. Compare pseudo vs ground truth
        if dirs_pseudo_L1 and dirs_true_L1:
            d_pseudo_L1 = dirs_pseudo_L1[0][0]
            d_true_L1 = dirs_true_L1[0][0]
            cos_L1 = float(torch.abs(d_pseudo_L1.double() @ d_true_L1.double()))
            result["cos_sim_L1"] = round(cos_L1, 6)
            result["angle_pseudo_L1"] = round(dirs_pseudo_L1[0][1], 1)
            result["angle_true_L1"] = round(dirs_true_L1[0][1], 1)
            logger.info(f"  Pseudo L1: {len(dirs_pseudo_L1)} dirs, |cos(d_pseudo, d_true)|={cos_L1:.4f}")
        else:
            result["cos_sim_L1"] = None
            result["angle_pseudo_L1"] = round(dirs_pseudo_L1[0][1], 1) if dirs_pseudo_L1 else None
            result["angle_true_L1"] = round(dirs_true_L1[0][1], 1) if dirs_true_L1 else None
            logger.info(f"  Pseudo L1: {'no dir' if not dirs_pseudo_L1 else f'{len(dirs_pseudo_L1)} dirs'} (GT: {'no dir' if not dirs_true_L1 else f'{len(dirs_true_L1)} dirs'})")

        if dirs_pseudo_L2 and dirs_true_L2:
            d_pseudo_L2 = dirs_pseudo_L2[0][0]
            d_true_L2 = dirs_true_L2[0][0]
            cos_L2 = float(torch.abs(d_pseudo_L2.double() @ d_true_L2.double()))
            result["cos_sim_L2"] = round(cos_L2, 6)
            result["angle_pseudo_L2"] = round(dirs_pseudo_L2[0][1], 1)
            result["angle_true_L2"] = round(dirs_true_L2[0][1], 1)
            logger.info(f"  Pseudo L2: {len(dirs_pseudo_L2)} dirs, |cos(d_pseudo, d_true)|={cos_L2:.4f}")
        else:
            result["cos_sim_L2"] = None
            result["angle_pseudo_L2"] = round(dirs_pseudo_L2[0][1], 1) if dirs_pseudo_L2 else None
            result["angle_true_L2"] = round(dirs_true_L2[0][1], 1) if dirs_true_L2 else None
            logger.info(f"  Pseudo L2: {'no dir' if not dirs_pseudo_L2 else f'{len(dirs_pseudo_L2)} dirs'} (GT: {'no dir' if not dirs_true_L2 else f'{len(dirs_true_L2)} dirs'})")

        # Merge with any previously stored entry (preserves eval metrics on resume)
        prev = all_results.get(str(k), {})
        prev.update(result)
        all_results[str(k)] = prev

        # Incremental save
        _save_results(output_dir, args, all_results, existing_baseline or {}, gt_per_k)
        logger.info(f"  Saved intermediate results (k={k})")

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        _print_summary_llava(all_results, gt_per_k)
        return

    # ── Step 6: Evaluation ──
    logger.info("=" * 60)
    logger.info("Step 6: Projection purification & evaluation")
    logger.info("=" * 60)

    # Check what still needs evaluation
    need_baseline = not (existing_baseline and _has_eval_metrics(existing_baseline))
    k_to_eval = [k for k in args.k_list
                 if not _has_eval_metrics(all_results.get(str(k), {}))]
    gt_k_to_eval = [k for k in args.k_list
                    if not _has_eval_metrics(gt_per_k.get(str(k), {}))]

    if not need_baseline and not k_to_eval and not gt_k_to_eval:
        logger.info("All evaluations already cached — skipping Step 6 entirely.")
        baseline_bd = existing_baseline
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)
        _print_summary_llava(all_results, gt_per_k)
        logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")
        return

    # Load model for inference
    from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig

    llava_config = LlavaConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True,
        patch_size=llava_config.vision_config.patch_size,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    with open(backdoor_dir / "local.json") as f:
        bd_config = json.load(f)

    prompt = bd_config.get("prompt", "Describe this image in a short sentence.")
    target = bd_config.get("target", "you have been hacked lol")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

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

    # Baseline: backdoored (single, shared across all k)
    if need_baseline:
        logger.info("\nEvaluating baseline (backdoored)...")
        baseline_bd = evaluate_projector(
            model, processor, bd_state, eval_cache, "P_bd",
            target, prompt_text, args.eval_batch_size
        )
        logger.info(f"  Backdoor baseline: {baseline_bd}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)
    else:
        baseline_bd = existing_baseline
        logger.info(f"  Reusing cached backdoor baseline: {baseline_bd}")

    # Ground truth purified per k
    for k in args.k_list:
        label = f"k{k}"
        if _has_eval_metrics(gt_per_k.get(str(k), {})):
            logger.info(f"  k={k} GT already evaluated, skipping")
            continue
        logger.info(f"\nEvaluating ground truth purified (k={k})...")
        dirs_true_L1, dirs_true_L2 = gt_directions[label]
        purified_true = projection_purify(bd_state, clean_state, dirs_true_L1[:1], dirs_true_L2[:1])
        gt_metrics = evaluate_projector(
            model, processor, purified_true, eval_cache, f"GT_k{k}",
            target, prompt_text, args.eval_batch_size
        )
        gt_per_k[str(k)] = gt_metrics
        logger.info(f"  GT k={k}: {gt_metrics}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)

    # Pseudo purified per k
    for k in args.k_list:
        label = f"k{k}"
        if _has_eval_metrics(all_results.get(str(k), {})):
            logger.info(f"  k={k} pseudo already evaluated, skipping")
            continue
        logger.info(f"\nEvaluating pseudo purified (k={k})...")
        dirs_ps_L1, dirs_ps_L2 = pseudo_directions[label]
        purified_ps = projection_purify(bd_state, clean_state, dirs_ps_L1[:1], dirs_ps_L2[:1])
        metrics = evaluate_projector(
            model, processor, purified_ps, eval_cache, f"Pseudo_k{k}",
            target, prompt_text, args.eval_batch_size
        )
        all_results[str(k)].update(metrics)
        logger.info(f"  Pseudo k={k}: {metrics}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)

    # Final save
    _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)
    _print_summary_llava(all_results, gt_per_k)
    logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-VL Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_qwen3vl(args, backdoor_dir, benign_dir, output_dir):
    from experiments.main_method.orthopurify.purify_qwen3vl import (
        finetune_adapter_qwen3vl,
        extract_clean_merger_weights,
        evaluate_qwen3vl_adapter,
    )

    model_path = MODEL_PATHS["qwen3vl"]
    n_samples = args.n_samples

    # ── Step 0: Load existing results for resume ──
    existing_results, existing_baseline, existing_gt_per_k = _load_existing_results(
        output_dir, force_rerun=args.force_rerun
    )

    # ── Step 1: Load weights ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights (Qwen3-VL)")
    logger.info("=" * 60)

    merger_clean, ds_clean = extract_clean_merger_weights(model_path)

    merger_bd = torch.load(str(backdoor_dir / "merger_state_dict.pth"), map_location="cpu")
    merger_bd = {k: v.float() for k, v in merger_bd.items()}
    ds_bd = None
    ds_bd_path = backdoor_dir / "deepstack_merger_list_state_dict.pth"
    if ds_bd_path.exists():
        ds_bd = torch.load(str(ds_bd_path), map_location="cpu")
        ds_bd = {k: v.float() for k, v in ds_bd.items()}

    merger_bn = torch.load(str(benign_dir / "merger_state_dict.pth"), map_location="cpu")
    merger_bn = {k: v.float() for k, v in merger_bn.items()}
    ds_bn = None
    ds_bn_path = benign_dir / "deepstack_merger_list_state_dict.pth"
    if ds_bn_path.exists():
        ds_bn = torch.load(str(ds_bn_path), map_location="cpu")
        ds_bn = {k: v.float() for k, v in ds_bn.items()}

    logger.info(f"  Loaded: clean, backdoor ({backdoor_dir.name}), benign ({benign_dir.name})")

    # ── Step 2: Per-matrix SVD of all deltas (ONCE) ──
    logger.info("=" * 60)
    logger.info("Step 2: Per-matrix SVD of weight deltas (computed once)")
    logger.info("=" * 60)

    keys_merger = get_2d_keys(merger_bd)
    svd_merger_bd = per_matrix_svd(merger_bd, merger_clean, keys_merger)
    svd_merger_bn = per_matrix_svd(merger_bn, merger_clean, keys_merger)
    logger.info(f"  Merger: {len(keys_merger)} 2D matrices")

    keys_ds = []
    svd_ds_bd = {}
    svd_ds_bn = {}
    if ds_bd is not None and ds_clean is not None and ds_bn is not None:
        keys_ds = get_2d_keys(ds_bd)
        svd_ds_bd = per_matrix_svd(ds_bd, ds_clean, keys_ds)
        svd_ds_bn = per_matrix_svd(ds_bn, ds_clean, keys_ds)
        logger.info(f"  DeepStack: {len(keys_ds)} 2D matrices")

    # ── Step 3: Pseudo-benign fine-tune (ONCE) ──
    logger.info("=" * 60)
    logger.info(f"Step 3: Pseudo-benign fine-tune (N={n_samples}, ONCE)")
    logger.info("=" * 60)

    cached_merger, cached_ds, cached_steps = (None, None, None)
    if not args.force_rerun:
        cached_merger, cached_ds, cached_steps = _try_load_cache_qwen3vl(output_dir)

    if cached_merger is not None:
        logger.info(f"  Cache hit: loading pseudo merger/deepstack from disk (skip finetune)")
        merger_pseudo = {k_name: v.cpu().float() for k_name, v in cached_merger.items()}
        ds_pseudo = ({k_name: v.cpu().float() for k_name, v in cached_ds.items()}
                     if cached_ds is not None else None)
        n_steps = cached_steps
    else:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from vlm_backdoor.data.dataset import CustomDataset
        from vlm_backdoor.data.collators import TrainQwen3VLCollator

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto",
        )
        model.gradient_checkpointing_enable()

        visual = model.model.visual
        visual.merger.float()
        if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            visual.deepstack_merger_list.float()

        for name, p in model.named_parameters():
            if "merger" in name or "deepstack_merger" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        model.enable_input_require_grads()

        collator = TrainQwen3VLCollator(processor, ignore_index=-100)

        with open(backdoor_dir / "local.json") as f:
            bd_config = json.load(f)

        clean_ds = CustomDataset(
            dataset_name="coco",
            prompt=bd_config.get("prompt", "Describe this image in a short sentence."),
            attack_type="replace",
            target="",
            train_num=n_samples,
            offset=5000,
            poison_rate=0.0,
            seed=SEED,
            patch_size=30,
            patch_type="random",
            patch_location="random_f",
            img_size=336,
            neg_sample=False,
        )
        bcfg = BATCH_CONFIG["qwen3vl"]
        train_loader = DataLoader(
            clean_ds, batch_size=bcfg["bs"], shuffle=True,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

        n_steps = finetune_adapter_qwen3vl(
            model, train_loader,
            num_epochs=2, lr=5e-5, warmup_ratio=0.03,
            grad_accum_steps=bcfg["grad_accum"],
        )

        merger_pseudo = {k_name: v.detach().cpu().float()
                        for k_name, v in visual.merger.state_dict().items()}
        ds_pseudo = None
        if hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
            ds_pseudo = {k_name: v.detach().cpu().float()
                         for k_name, v in visual.deepstack_merger_list.state_dict().items()}

        _save_cache_qwen3vl(output_dir, merger_pseudo, ds_pseudo, n_steps, n_samples)
        logger.info(f"  Saved pseudo adapter cache to {_cache_dir(output_dir)}")

        del model
        torch.cuda.empty_cache()

    # ── Step 4: Per-matrix SVD of pseudo deltas (ONCE) ──
    logger.info("=" * 60)
    logger.info("Step 4: Per-matrix SVD of pseudo deltas (computed once)")
    logger.info("=" * 60)

    svd_merger_pseudo = per_matrix_svd(merger_pseudo, merger_clean, keys_merger)
    logger.info(f"  Merger pseudo SVD: {len(keys_merger)} matrices")

    svd_ds_pseudo = {}
    if ds_pseudo is not None and ds_clean is not None and keys_ds:
        svd_ds_pseudo = per_matrix_svd(ds_pseudo, ds_clean, keys_ds)
        logger.info(f"  DeepStack pseudo SVD: {len(keys_ds)} matrices")

    # ── Step 5: Sweep k ──
    logger.info("=" * 60)
    logger.info("Step 5: k sweep (direction extraction + cos_sim)")
    logger.info("=" * 60)

    all_results = dict(existing_results)
    gt_directions = {}
    pseudo_directions = {}
    gt_per_k = dict(existing_gt_per_k)

    for k in args.k_list:
        label = f"k{k}"
        logger.info(f"\n{'─' * 50}")
        logger.info(f"k={k}")
        logger.info(f"{'─' * 50}")

        # a. Ground truth directions
        dirs_true_merger = extract_orthogonal_directions_multimatrix(
            svd_merger_bd, svd_merger_bn, keys_merger, k, angle_threshold=ANGLE_THRESHOLD
        )
        dirs_true_ds = {}
        if svd_ds_bd and svd_ds_bn and keys_ds:
            dirs_true_ds = extract_orthogonal_directions_multimatrix(
                svd_ds_bd, svd_ds_bn, keys_ds, k, angle_threshold=ANGLE_THRESHOLD
            )
        gt_directions[label] = (dirs_true_merger, dirs_true_ds)

        # b. Pseudo directions
        dirs_pseudo_merger = extract_orthogonal_directions_multimatrix(
            svd_merger_bd, svd_merger_pseudo, keys_merger, k, angle_threshold=ANGLE_THRESHOLD
        )
        dirs_pseudo_ds = {}
        if svd_ds_bd and svd_ds_pseudo and keys_ds:
            dirs_pseudo_ds = extract_orthogonal_directions_multimatrix(
                svd_ds_bd, svd_ds_pseudo, keys_ds, k, angle_threshold=ANGLE_THRESHOLD
            )
        pseudo_directions[label] = (dirs_pseudo_merger, dirs_pseudo_ds)

        # c. Build result entry
        result = {
            "k": k,
            "n_samples": n_samples,
            "n_steps": n_steps,
            "n_matrices_true_merger": len(dirs_true_merger),
            "n_matrices_pseudo_merger": len(dirs_pseudo_merger),
        }

        logger.info(f"  GT  Merger: {len(dirs_true_merger)}/{len(keys_merger)} matrices with dirs")
        logger.info(f"  Pseudo Merger: {len(dirs_pseudo_merger)}/{len(keys_merger)} matrices with dirs")

        # d. Compare pseudo vs ground truth for merger
        merger_comparison = compare_directions_multimatrix(dirs_true_merger, dirs_pseudo_merger)
        result["merger_mean_cos_sim"] = merger_comparison.get("mean_cos_sim")
        result["merger_n_matched"] = merger_comparison.get("n_matrices_with_both", 0)
        result["merger_n_total"] = len(keys_merger)

        if merger_comparison.get("mean_cos_sim") is not None:
            logger.info(f"  Merger: mean|cos|={merger_comparison['mean_cos_sim']:.4f}, "
                        f"matched={merger_comparison['n_matrices_with_both']}/{len(keys_merger)}")

        # e. Compare for deepstack
        if dirs_true_ds and dirs_pseudo_ds:
            ds_comparison = compare_directions_multimatrix(dirs_true_ds, dirs_pseudo_ds)
            result["ds_mean_cos_sim"] = ds_comparison.get("mean_cos_sim")
            result["ds_n_matched"] = ds_comparison.get("n_matrices_with_both", 0)
            result["ds_n_total"] = len(keys_ds)
            result["n_matrices_true_ds"] = len(dirs_true_ds)
            result["n_matrices_pseudo_ds"] = len(dirs_pseudo_ds)
            if ds_comparison.get("mean_cos_sim") is not None:
                logger.info(f"  DeepStack: mean|cos|={ds_comparison['mean_cos_sim']:.4f}")
            logger.info(f"  GT  DS: {len(dirs_true_ds)}/{len(keys_ds)} matrices with dirs")
            logger.info(f"  Pseudo DS: {len(dirs_pseudo_ds)}/{len(keys_ds)} matrices with dirs")
        else:
            result["ds_mean_cos_sim"] = None

        # Merge with previously stored entry
        prev = all_results.get(str(k), {})
        prev.update(result)
        all_results[str(k)] = prev

        # Incremental save
        _save_results(output_dir, args, all_results, existing_baseline or {}, gt_per_k)
        logger.info(f"  Saved intermediate results (k={k})")

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        _print_summary_qwen3vl(all_results, gt_per_k)
        return

    # ── Step 6: Evaluation ──
    logger.info("=" * 60)
    logger.info("Step 6: Projection purification & evaluation")
    logger.info("=" * 60)

    need_baseline = not (existing_baseline and _has_eval_metrics(existing_baseline))
    k_to_eval = [k for k in args.k_list
                 if not _has_eval_metrics(all_results.get(str(k), {}))]
    gt_k_to_eval = [k for k in args.k_list
                    if not _has_eval_metrics(gt_per_k.get(str(k), {}))]

    if not need_baseline and not k_to_eval and not gt_k_to_eval:
        logger.info("All evaluations already cached — skipping Step 6 entirely.")
        baseline_bd = existing_baseline
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)
        _print_summary_qwen3vl(all_results, gt_per_k)
        logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")
        return

    # Load model for inference
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    with open(backdoor_dir / "local.json") as f:
        bd_config = json.load(f)

    target = bd_config.get("target", "you have been hacked lol")

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

    # Baseline: backdoored
    if need_baseline:
        logger.info("\nEvaluating baseline (backdoored)...")
        merger_bd_half = {k_name: v.half() for k_name, v in merger_bd.items()}
        ds_bd_half = {k_name: v.half() for k_name, v in ds_bd.items()} if ds_bd is not None else None
        baseline_bd = evaluate_qwen3vl_adapter(
            model, processor, merger_bd_half, ds_bd_half, eval_cache, "backdoor",
            target, args.eval_batch_size
        )
        logger.info(f"  Backdoor baseline: {baseline_bd}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)
    else:
        baseline_bd = existing_baseline
        logger.info(f"  Reusing cached backdoor baseline: {baseline_bd}")

    # Ground truth purified per k
    for k in args.k_list:
        label = f"k{k}"
        if _has_eval_metrics(gt_per_k.get(str(k), {})):
            logger.info(f"  k={k} GT already evaluated, skipping")
            continue
        logger.info(f"\nEvaluating ground truth purified (k={k})...")
        dirs_true_merger, dirs_true_ds = gt_directions[label]

        merger_pur_true = projection_purify_multimatrix(merger_bd, merger_clean, dirs_true_merger)
        merger_pur_true_half = {k_name: v.half() for k_name, v in merger_pur_true.items()}

        if ds_bd is not None and ds_clean is not None and dirs_true_ds:
            ds_pur_true = projection_purify_multimatrix(ds_bd, ds_clean, dirs_true_ds)
            ds_pur_true_half = {k_name: v.half() for k_name, v in ds_pur_true.items()}
        else:
            ds_pur_true_half = {k_name: v.half() for k_name, v in ds_bd.items()} if ds_bd is not None else None

        gt_metrics = evaluate_qwen3vl_adapter(
            model, processor, merger_pur_true_half, ds_pur_true_half, eval_cache, f"GT_k{k}",
            target, args.eval_batch_size
        )
        gt_per_k[str(k)] = gt_metrics
        logger.info(f"  GT k={k}: {gt_metrics}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)

    # Pseudo purified per k
    for k in args.k_list:
        label = f"k{k}"
        if _has_eval_metrics(all_results.get(str(k), {})):
            logger.info(f"  k={k} pseudo already evaluated, skipping")
            continue
        logger.info(f"\nEvaluating pseudo purified (k={k})...")
        dirs_ps_merger, dirs_ps_ds = pseudo_directions[label]

        merger_pur = projection_purify_multimatrix(merger_bd, merger_clean, dirs_ps_merger)
        merger_pur_half = {k_name: v.half() for k_name, v in merger_pur.items()}

        if ds_bd is not None and ds_clean is not None and dirs_ps_ds:
            ds_pur = projection_purify_multimatrix(ds_bd, ds_clean, dirs_ps_ds)
            ds_pur_half = {k_name: v.half() for k_name, v in ds_pur.items()}
        else:
            ds_pur_half = {k_name: v.half() for k_name, v in ds_bd.items()} if ds_bd is not None else None

        metrics = evaluate_qwen3vl_adapter(
            model, processor, merger_pur_half, ds_pur_half, eval_cache, f"Pseudo_k{k}",
            target, args.eval_batch_size
        )
        all_results[str(k)].update(metrics)
        logger.info(f"  Pseudo k={k}: {metrics}")
        _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)

    # Final save
    _save_results(output_dir, args, all_results, baseline_bd, gt_per_k)
    _print_summary_qwen3vl(all_results, gt_per_k)
    logger.info(f"\nDone. Results saved to {output_dir / 'ablation_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Output Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_results(output_dir, args, results, baseline_bd, gt_per_k):
    model_name = "LLaVA-1.5-7B" if args.model == "llava" else "Qwen3-VL-8B-Instruct"
    attack_name = args.attack.upper()
    lr = 2e-4 if args.model == "llava" else 5e-5

    bcfg = BATCH_CONFIG[args.model]
    output = {
        "model": model_name,
        "attack": attack_name,
        "config": {
            "n_samples": args.n_samples,
            "angle_threshold": ANGLE_THRESHOLD,
            "seed": SEED,
            "lr": lr,
            "bs": bcfg["bs"],
            "grad_accum": bcfg["grad_accum"],
            "effective_bs": bcfg["bs"] * bcfg["grad_accum"],
            "epochs": 2,
        },
        "results": results,
        "backdoor_baseline": baseline_bd if baseline_bd else {},
        "ground_truth_per_k": gt_per_k if gt_per_k else {},
    }
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)


def _print_summary_llava(results, gt_per_k):
    print("\n" + "=" * 100)
    print("k ABLATION SUMMARY (LLaVA)")
    print("=" * 100)
    print(f"  {'k':>3} {'#d_L1':>6} {'#d_L2':>6} {'cos_L1':>8} {'cos_L2':>8}"
          f" {'ASR%':>8} {'Cl CIDEr':>10}"
          f" | {'GT ASR%':>8} {'GT CIDEr':>10}")
    print(f"  {'-' * 90}")
    for k_str in sorted(results.keys(), key=lambda x: int(x)):
        r = results[k_str]
        nd1 = f"{r.get('n_dirs_pseudo_L1', '?')}"
        nd2 = f"{r.get('n_dirs_pseudo_L2', '?')}"
        cos1 = f"{r['cos_sim_L1']:.4f}" if r.get('cos_sim_L1') is not None else "N/A"
        cos2 = f"{r['cos_sim_L2']:.4f}" if r.get('cos_sim_L2') is not None else "N/A"
        asr = f"{r['backdoor_asr']:.2f}" if 'backdoor_asr' in r else "—"
        cc = f"{r['clean_cider']:.2f}" if 'clean_cider' in r else "—"
        gt = gt_per_k.get(k_str, {})
        gt_asr = f"{gt['backdoor_asr']:.2f}" if 'backdoor_asr' in gt else "—"
        gt_cc = f"{gt['clean_cider']:.2f}" if 'clean_cider' in gt else "—"
        print(f"  {r['k']:>3} {nd1:>6} {nd2:>6} {cos1:>8} {cos2:>8}"
              f" {asr:>8} {cc:>10}"
              f" | {gt_asr:>8} {gt_cc:>10}")
    print("=" * 100)


def _print_summary_qwen3vl(results, gt_per_k):
    print("\n" + "=" * 100)
    print("k ABLATION SUMMARY (Qwen3-VL)")
    print("=" * 100)
    print(f"  {'k':>3} {'#M_merger':>10} {'Merger cos':>12} {'DS cos':>10}"
          f" {'ASR%':>8} {'Cl CIDEr':>10}"
          f" | {'GT ASR%':>8} {'GT CIDEr':>10}")
    print(f"  {'-' * 90}")
    for k_str in sorted(results.keys(), key=lambda x: int(x)):
        r = results[k_str]
        nm = f"{r.get('n_matrices_pseudo_merger', '?')}/{r.get('merger_n_total', '?')}"
        mc = f"{r['merger_mean_cos_sim']:.4f}" if r.get('merger_mean_cos_sim') is not None else "N/A"
        dc = f"{r['ds_mean_cos_sim']:.4f}" if r.get('ds_mean_cos_sim') is not None else "N/A"
        asr = f"{r['backdoor_asr']:.2f}" if 'backdoor_asr' in r else "—"
        cc = f"{r['clean_cider']:.2f}" if 'clean_cider' in r else "—"
        gt = gt_per_k.get(k_str, {})
        gt_asr = f"{gt['backdoor_asr']:.2f}" if 'backdoor_asr' in gt else "—"
        gt_cc = f"{gt['clean_cider']:.2f}" if 'clean_cider' in gt else "—"
        print(f"  {r['k']:>3} {nm:>10} {mc:>12} {dc:>10}"
              f" {asr:>8} {cc:>10}"
              f" | {gt_asr:>8} {gt_cc:>10}")
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="k (SVD top-k directions) ablation for exp1c pseudo-benign purification"
    )
    parser.add_argument("--model", type=str, required=True, choices=["llava", "qwen3vl"],
                        help="Model: llava or qwen3vl")
    parser.add_argument("--attack", type=str, required=True, choices=["badnet", "issba"],
                        help="Attack type: badnet or issba")
    parser.add_argument("--k_list", type=int, nargs="+",
                        default=DEFAULT_K_LIST,
                        help="List of k values to sweep (default: 1 2 3 5 8 10 15 20)")
    parser.add_argument("--n_samples", type=int, default=FIXED_N_SAMPLES,
                        help="Number of clean samples for pseudo-benign fine-tuning (default: 64)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU indices to use (e.g. '2' or '4,5'). Overrides CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Eval batch size (default: 16 for LLaVA, 4 for Qwen3-VL)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only compute direction extraction, skip ASR/CIDEr evaluation")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore all cached pseudo weights & existing results, rerun from scratch")
    args = parser.parse_args()

    # Set GPU visibility BEFORE any CUDA initialization
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpus}")
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.warning("CUDA_VISIBLE_DEVICES not set and --gpus not specified, "
                        "model will use all available GPUs (starting from GPU 0)")

    if args.eval_batch_size is None:
        args.eval_batch_size = 16 if args.model == "llava" else 4

    os.chdir(PROJECT_ROOT)

    # Resolve checkpoint paths
    key = (args.model, args.attack)
    if key not in CHECKPOINT_MAP:
        raise ValueError(f"Unknown combination: {args.model} + {args.attack}")

    bd_rel, bn_rel = CHECKPOINT_MAP[key]
    backdoor_dir = PROJECT_ROOT / bd_rel
    benign_dir = PROJECT_ROOT / bn_rel

    if not backdoor_dir.exists():
        raise FileNotFoundError(f"Backdoor checkpoint not found: {backdoor_dir}")
    if not benign_dir.exists():
        raise FileNotFoundError(f"Benign checkpoint not found: {benign_dir}")

    output_dir = PROJECT_ROOT / "experiments/main_method/orthopurify/ablation_k" / f"{args.model}_{args.attack}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model}, Attack: {args.attack}")
    logger.info(f"Backdoor: {backdoor_dir.name}")
    logger.info(f"Benign: {benign_dir.name}")
    logger.info(f"k values: {args.k_list}")
    logger.info(f"N_SAMPLES: {args.n_samples} (fixed, single fine-tuning)")
    logger.info(f"Output: {output_dir}")

    if args.model == "llava":
        run_llava(args, backdoor_dir, benign_dir, output_dir)
    else:
        run_qwen3vl(args, backdoor_dir, benign_dir, output_dir)


if __name__ == "__main__":
    main()
