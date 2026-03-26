#!/usr/bin/env python3
"""
实验 1b：正交方向投影去除验证

假设：backdoor ΔW 子空间中与 benign ΔW 正交的方向承载后门信号。
方法：提取该方向 d，通过投影去除 W_pur = W_bd - ΔW·d·dᵀ 净化 projector。
验证：ASR 是否下降、CIDEr 是否保持。

模型只加载一次，换 projector 权重后批量推理，速度约为 subprocess 模式的 4-6 倍。

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
    python exps/exp1b_projection/exp1b_projection.py [--test_num 512] [--skip_eval]
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

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
BACKDOOR_PATH = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth"
BENIGN_PATH = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth"
BACKDOOR_LOCAL_JSON = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/local.json"

OUTPUT_DIR = PROJECT_ROOT / "exps/exp1b_projection"

# ─── Weight Loading (from exp1) ─────────────────────────────────────────────

def load_projector_weights(path):
    """Load projector Layer1 and Layer2 weights. Returns (W1, W2)."""
    sd = torch.load(str(path), map_location="cpu")
    for key in ("state_dict", "model"):
        if key in sd:
            sd = sd[key]
            break

    candidates = [
        ("linear_1.weight", "linear_2.weight"),
        ("model.mm_projector.0.weight", "model.mm_projector.2.weight"),
    ]
    for k1, k2 in candidates:
        if k1 in sd and k2 in sd:
            W1, W2 = sd[k1].float(), sd[k2].float()
            logger.info(f"  Loaded {Path(path).name}: L1 {W1.shape}, L2 {W2.shape}")
            return W1, W2

    raise KeyError(f"Projector keys not found in {path}. Keys: {list(sd.keys())[:10]}")


def load_full_state_dict(path):
    """Load full state dict (for constructing purified projector)."""
    sd = torch.load(str(path), map_location="cpu")
    for key in ("state_dict", "model"):
        if key in sd:
            sd = sd[key]
            break
    return sd


# ─── Core Analysis ─���─────────────────────────────────────────────────────────

def extract_orthogonal_directions(Vh_bd, Vh_bn, k, angle_threshold=70.0):
    """
    Extract directions in backdoor top-k subspace that are orthogonal to benign subspace.

    Returns: list of (d_vector [input_dim], angle_deg)
        d_vector is a unit vector in the original input space.
    """
    V_bd = Vh_bd[:k, :].T.double()  # [input_dim, k]
    V_bn = Vh_bn[:k, :].T.double()  # [input_dim, k]

    M = V_bd.T @ V_bn  # [k, k]
    U_M, sigma, _ = torch.linalg.svd(M, full_matrices=False)
    sigma = sigma.clamp(-1.0, 1.0)
    angles = torch.acos(sigma) * (180.0 / math.pi)  # descending cos → ascending angle

    results = []
    # Angles are sorted: smallest first (most aligned), largest last (most orthogonal)
    for i in range(k - 1, -1, -1):
        angle = float(angles[i])
        if angle < angle_threshold:
            break
        # Map back to original input space
        d = V_bd @ U_M[:, i]  # [input_dim]
        d = d / d.norm()  # ensure unit vector
        results.append((d.float(), angle))

    return results


def direction_stability(directions_per_k):
    """
    Compute cosine similarity between the most orthogonal direction
    extracted at different k values.

    directions_per_k: dict {k: d_vector} (only the MOST orthogonal direction per k)
    Returns: dict of pairwise cosine similarities
    """
    ks = sorted(directions_per_k.keys())
    stability = {}
    for i, k1 in enumerate(ks):
        for k2 in ks[i + 1:]:
            d1 = directions_per_k[k1].double()
            d2 = directions_per_k[k2].double()
            cos_sim = float(torch.abs(d1 @ d2))
            stability[f"k={k1}_vs_k={k2}"] = round(cos_sim, 6)
    return stability


def energy_analysis(dW, directions):
    """
    Compute fraction of ΔW energy in the given directions.

    dW: [out_dim, in_dim]
    directions: list of (d_vector [in_dim], angle_deg)
    Returns: dict with per-direction and total removed energy ratio
    """
    dW_d = dW.double()
    total_energy = float(dW_d.norm() ** 2)
    per_direction = []
    total_removed = 0.0

    for d, angle in directions:
        d_d = d.double()
        component = dW_d @ d_d  # [out_dim]
        energy = float(component.norm() ** 2)
        ratio = energy / total_energy
        total_removed += energy
        per_direction.append({
            "angle_deg": round(angle, 3),
            "energy_removed": round(energy, 6),
            "energy_ratio": round(ratio, 6),
        })

    return {
        "per_direction": per_direction,
        "total_removed_ratio": round(total_removed / total_energy, 6),
        "total_energy": round(total_energy, 4),
    }


def projection_purify(bd_state, clean_state, directions_L1, directions_L2):
    """
    Apply projection removal to both layers.

    W_pur = W_bd - ΔW · D · Dᵀ  (per layer)
    Where D = [d1, d2, ...] and D Dᵀ is the projection matrix.
    """
    purified = {}
    for k, v in bd_state.items():
        purified[k] = v.clone()

    # Determine key names
    l1_key = "linear_1.weight" if "linear_1.weight" in bd_state else None
    l2_key = "linear_2.weight" if "linear_2.weight" in bd_state else None

    if l1_key is None or l2_key is None:
        raise KeyError(f"Expected linear_1/2.weight keys, got: {list(bd_state.keys())}")

    for layer_key, directions, layer_name in [
        (l1_key, directions_L1, "Layer1"),
        (l2_key, directions_L2, "Layer2"),
    ]:
        if not directions:
            logger.info(f"  {layer_name}: no orthogonal directions, skip")
            continue

        W_bd = bd_state[layer_key].float()
        W_clean = clean_state[layer_key].float()
        dW = W_bd - W_clean  # [out, in]

        # Build D matrix from direction vectors
        d_vectors = [d for d, _ in directions]
        D = torch.stack(d_vectors, dim=1)  # [in_dim, n_dirs]

        # ΔW · D · Dᵀ
        projected = dW @ D @ D.T  # [out, in]
        W_pur = W_bd - projected

        purified[layer_key] = W_pur
        n = len(directions)
        angles = [f"{a:.1f}°" for _, a in directions]
        logger.info(f"  {layer_name}: removed {n} direction(s) at {angles}")

    return purified


# ─── Batch Evaluation (adapted from exp6_nsamples_sweep.py) ──────────────────

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_eval_cache(test_dataset, bd_cfg, test_num):
    """
    Pre-load TEST_NUM unique images, apply triggers, collect GT captions.
    Returns List[{"clean_img": PIL, "bd_img": PIL, "gts": List[str]}]
    """
    from collections import defaultdict
    from PIL import Image
    from tqdm import tqdm
    from vlm_backdoor.attacks.triggers import apply_trigger

    patch_type = bd_cfg.get("patch_type", "random")
    patch_loc = bd_cfg.get("patch_location", "random_f")
    patch_size = bd_cfg.get("patch_size", 30)
    img_size = bd_cfg.get("img_size", 336)

    issba_encoder = -1
    if patch_type == "issba":
        import os
        orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        from vlm_backdoor.attacks.issba import issbaEncoder
        issba_encoder = issbaEncoder(model_path='assets/issba_encoder', secret='Stega!!', size=(img_size, img_size))
        if orig_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    image_to_batch: Dict = {}
    image_to_gts: Dict = defaultdict(list)

    for item in test_dataset:
        ip = item["image_path"]
        if ip not in image_to_batch:
            image_to_batch[ip] = item
        image_to_gts[ip].append(item.get("caption") or item.get("captions", ""))

    keys = list(image_to_batch.keys())[:test_num]
    cache = []

    logger.info(f"Pre-loading {len(keys)} images and applying triggers...")
    for img_path in tqdm(keys, desc="  build_eval_cache"):
        if isinstance(img_path, str):
            img = Image.open(img_path).convert("RGB")
        else:
            img = img_path.convert("RGB")

        img_bd = apply_trigger(img, patch_type=patch_type, patch_location=patch_loc,
                               patch_size=patch_size, img_size=img_size, encoder=issba_encoder)
        cache.append({
            "clean_img": img,
            "bd_img": img_bd,
            "gts": image_to_gts[img_path],
        })

    return cache


@torch.no_grad()
def evaluate_projector(model, processor, proj_state, eval_cache, label,
                       target, prompt_text, eval_batch_size=16):
    """Load proj_state into model, batch inference, return ASR / CIDEr."""
    import evaluate as hf_evaluate
    from tqdm import tqdm

    model.multi_modal_projector.load_state_dict(proj_state)
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="exp1b: Orthogonal direction projection purification")
    parser.add_argument("--test_num", type=int, default=512,
                        help="Number of test images for evaluation")
    parser.add_argument("--angle_threshold", type=float, default=70.0,
                        help="Angle threshold (degrees) for multi-direction mode")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, only run analysis (no GPU needed)")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for evaluation inference")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    K_VALUES = [5, 10, 20, 50]

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Load weights & SVD
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: Loading weights & computing SVD")
    logger.info("=" * 60)

    W1_clean, W2_clean = load_projector_weights(CLEAN_PATH)
    W1_bd, W2_bd = load_projector_weights(BACKDOOR_PATH)
    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    logger.info("Computing SVD...")
    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)
    logger.info("SVD done.")

    # Load full state dicts for purification
    bd_state = load_full_state_dict(BACKDOOR_PATH)
    clean_state = load_full_state_dict(CLEAN_PATH)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Extract orthogonal directions for each k
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2: Extracting orthogonal directions")
    logger.info("=" * 60)

    all_directions = {}  # {(layer, k): [(d, angle), ...]}
    # Also store the single most-orthogonal direction per (layer, k) for stability analysis
    single_d = {"layer1": {}, "layer2": {}}

    for k in K_VALUES:
        dirs_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k, angle_threshold=args.angle_threshold)
        dirs_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k, angle_threshold=args.angle_threshold)
        all_directions[("layer1", k)] = dirs_L1
        all_directions[("layer2", k)] = dirs_L2

        # Store most orthogonal direction (first in list = largest angle)
        if dirs_L1:
            single_d["layer1"][k] = dirs_L1[0][0]
        if dirs_L2:
            single_d["layer2"][k] = dirs_L2[0][0]

        n1 = len(dirs_L1)
        n2 = len(dirs_L2)
        angles1 = [f"{a:.1f}°" for _, a in dirs_L1] if dirs_L1 else ["none"]
        angles2 = [f"{a:.1f}°" for _, a in dirs_L2] if dirs_L2 else ["none"]
        logger.info(f"  k={k:2d}: L1 → {n1} dir(s) {angles1}, L2 → {n2} dir(s) {angles2}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Direction stability analysis
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3: Direction stability (cosine similarity across k)")
    logger.info("=" * 60)

    stability = {}
    for layer_name in ["layer1", "layer2"]:
        s = direction_stability(single_d[layer_name])
        stability[layer_name] = s
        logger.info(f"  {layer_name}:")
        for pair, cos in s.items():
            logger.info(f"    {pair}: |cos| = {cos:.4f}")

    with open(OUTPUT_DIR / "exp1b_direction_stability.json", "w") as f:
        json.dump(stability, f, indent=2)
    logger.info(f"Saved → exp1b_direction_stability.json")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Energy analysis
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 4: Energy analysis")
    logger.info("=" * 60)

    energy_results = {}
    for k in K_VALUES:
        dirs_L1 = all_directions[("layer1", k)]
        dirs_L2 = all_directions[("layer2", k)]

        e1 = energy_analysis(dW1_bd, dirs_L1) if dirs_L1 else {"total_removed_ratio": 0.0}
        e2 = energy_analysis(dW2_bd, dirs_L2) if dirs_L2 else {"total_removed_ratio": 0.0}

        # Also compute single-direction energy (only most orthogonal)
        e1_single = energy_analysis(dW1_bd, dirs_L1[:1]) if dirs_L1 else {"total_removed_ratio": 0.0}
        e2_single = energy_analysis(dW2_bd, dirs_L2[:1]) if dirs_L2 else {"total_removed_ratio": 0.0}

        energy_results[f"k={k}"] = {
            "layer1_single": e1_single,
            "layer2_single": e2_single,
            "layer1_all": e1,
            "layer2_all": e2,
        }

        logger.info(f"  k={k}: L1 single={e1_single['total_removed_ratio']:.4%}, "
                     f"all={e1['total_removed_ratio']:.4%} | "
                     f"L2 single={e2_single['total_removed_ratio']:.4%}, "
                     f"all={e2['total_removed_ratio']:.4%}")

    with open(OUTPUT_DIR / "exp1b_energy_removed.json", "w") as f:
        json.dump(energy_results, f, indent=2)
    logger.info(f"Saved → exp1b_energy_removed.json")

    if args.skip_eval:
        logger.info("--skip_eval: stopping before evaluation.")
        _print_analysis_summary(stability, energy_results)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5-6: Projection purification + batch evaluation
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 5-6: Purification & batch evaluation")
    logger.info("=" * 60)

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_config = json.load(f)

    prompt = bd_config.get("prompt", "Describe this image in a short sentence.")
    target = bd_config.get("target", "you have been hacked lol")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Load model once
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    logger.info("Loading model (once)...")
    model_path = bd_config.get("model_name_or_path",
                               "/data/YBJ/cleansight/models/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Load test dataset and build eval cache (once)
    logger.info("Loading test dataset...")
    test_dataset = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))

    eval_cache = build_eval_cache(test_dataset, bd_config, args.test_num)
    logger.info(f"Eval cache ready: {len(eval_cache)} images")

    eval_results = {}

    # Baseline: backdoored model
    logger.info("\nEvaluating baseline (backdoored)...")
    eval_results["baseline_backdoor"] = evaluate_projector(
        model, processor, bd_state, eval_cache, "P_b",
        target, prompt_text, args.eval_batch_size
    )
    logger.info(f"  Baseline: {eval_results['baseline_backdoor']}")

    # Single-direction purification for each k
    for k in K_VALUES:
        label = f"k{k}"
        logger.info(f"\nPurifying with k={k} (single direction per layer)...")
        dirs_L1 = all_directions[("layer1", k)][:1]  # only most orthogonal
        dirs_L2 = all_directions[("layer2", k)][:1]
        purified = projection_purify(bd_state, clean_state, dirs_L1, dirs_L2)

        # Save purified weights
        pur_dir = OUTPUT_DIR / f"purified_{label}"
        pur_dir.mkdir(parents=True, exist_ok=True)
        torch.save(purified, str(pur_dir / "mmprojector_state_dict.pth"))

        metrics = evaluate_projector(
            model, processor, purified, eval_cache, label,
            target, prompt_text, args.eval_batch_size
        )
        eval_results[f"k={k}_single"] = metrics
        logger.info(f"  k={k} single: {metrics}")

    # Multi-direction purification at k=50
    dirs_L1_multi = all_directions[("layer1", 50)]
    dirs_L2_multi = all_directions[("layer2", 50)]
    if len(dirs_L1_multi) > 1 or len(dirs_L2_multi) > 1:
        label = "k50_multi"
        logger.info(f"\nPurifying with k=50 (multi-direction: L1={len(dirs_L1_multi)}, L2={len(dirs_L2_multi)})...")
        purified = projection_purify(bd_state, clean_state, dirs_L1_multi, dirs_L2_multi)

        pur_dir = OUTPUT_DIR / f"purified_{label}"
        pur_dir.mkdir(parents=True, exist_ok=True)
        torch.save(purified, str(pur_dir / "mmprojector_state_dict.pth"))

        metrics = evaluate_projector(
            model, processor, purified, eval_cache, label,
            target, prompt_text, args.eval_batch_size
        )
        eval_results["k=50_multi"] = metrics
        logger.info(f"  k=50 multi: {metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7: Summary
    # ══════════════════════════════════════════════════════════════════════════
    results = {
        "evaluation": eval_results,
        "energy": energy_results,
        "stability": stability,
        "config": {
            "angle_threshold": args.angle_threshold,
            "test_num": args.test_num,
            "k_values": K_VALUES,
        },
    }
    with open(OUTPUT_DIR / "exp1b_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved → exp1b_evaluation.json")

    _print_eval_summary(eval_results, energy_results)


def _print_analysis_summary(stability, energy_results):
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY (no evaluation)")
    print("=" * 60)

    print("\nDirection Stability (|cos sim| between k values):")
    for layer, pairs in stability.items():
        print(f"  {layer}:")
        for pair, cos in pairs.items():
            status = "STABLE" if cos > 0.9 else "UNSTABLE" if cos < 0.5 else "moderate"
            print(f"    {pair}: {cos:.4f}  [{status}]")

    print("\nEnergy Removed (single direction):")
    print(f"  {'k':>4}  {'L1 ratio':>10}  {'L2 ratio':>10}")
    print(f"  {'-' * 28}")
    for k_label, e in energy_results.items():
        r1 = e["layer1_single"]["total_removed_ratio"]
        r2 = e["layer2_single"]["total_removed_ratio"]
        print(f"  {k_label:>4}  {r1:>10.4%}  {r2:>10.4%}")
    print("=" * 60)


def _print_eval_summary(eval_results, energy_results):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<20} {'ASR':>8} {'Clean CIDEr':>12} {'Energy L1':>10} {'Energy L2':>10}")
    print(f"  {'-' * 62}")
    for config_name, metrics in eval_results.items():
        asr = metrics.get("backdoor_asr", "N/A")
        cider = metrics.get("clean_cider", "N/A")
        asr_str = f"{asr:.2f}" if isinstance(asr, float) else str(asr)
        cider_str = f"{cider:.2f}" if isinstance(cider, float) else str(cider)

        # Find matching energy
        e1, e2 = "—", "—"
        for k_label, e in energy_results.items():
            if k_label.replace("k=", "k") in config_name:
                e1 = f"{e['layer1_single']['total_removed_ratio']:.4%}"
                e2 = f"{e['layer2_single']['total_removed_ratio']:.4%}"
                break

        print(f"  {config_name:<20} {asr_str:>8} {cider_str:>12} {e1:>10} {e2:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
