#!/usr/bin/env python3
"""
实验一：参数空间差异分析
分析 LLaVA-1.5-7B projector 层的后门权重与干净权重之间的差异
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAN_PATH    = os.path.join(ROOT, "models/llava-1.5-7b-hf/mm_projector_extracted.bin")
BACKDOOR_PATH = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth")
BENIGN_PATH   = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth")

OUTPUT_DIR = os.path.join(ROOT, "exps/exp1_W_analysis")


# ─── Weight Loading ───────────────────────────────────────────────────────────

def load_projector_weights(path, prefix="model.mm_projector"):
    """Load projector Layer1 and Layer2 weights from a checkpoint file.
    Tries multiple key naming conventions automatically.
    """
    sd = torch.load(path, map_location="cpu")
    # Unwrap nested state dicts
    for key in ("state_dict", "model"):
        if key in sd:
            sd = sd[key]
            break

    # Try candidate key pairs in order
    candidates = [
        (f"{prefix}.0.weight", f"{prefix}.2.weight"),   # meta.py / HF format
        ("linear_1.weight",    "linear_2.weight"),       # extracted projector format
        ("model.mm_projector.0.weight", "model.mm_projector.2.weight"),  # explicit fallback
    ]

    key1, key2 = None, None
    for k1, k2 in candidates:
        if k1 in sd and k2 in sd:
            key1, key2 = k1, k2
            break

    if key1 is None:
        print(f"\n[WARN] No matching projector keys found in:\n  {path}")
        print("Available keys (first 30):")
        for k in list(sd.keys())[:30]:
            print(f"  {k}")
        raise KeyError(f"Projector keys not found. Use --projector_prefix to override.")

    W1 = sd[key1].float()
    W2 = sd[key2].float()
    print(f"  Loaded: {os.path.basename(path)}  (keys: {key1}, {key2})")
    print(f"    Layer1 {W1.shape}, Layer2 {W2.shape}")
    return W1, W2


# ─── Analysis Functions ───────────────────────────────────────────────────────

def compute_svd(dW):
    """Full SVD of dW (full_matrices=False). Returns (U, S, Vh), S descending."""
    return torch.linalg.svd(dW, full_matrices=False)


def energy_ratio(S, k):
    """Fraction of total energy (sum of squared singular values) in top-k."""
    S2 = S.double() ** 2
    return float(S2[:k].sum() / S2.sum())


def effective_rank(S):
    """Effective rank = exp(entropy of normalized singular values)."""
    s = S.double().numpy()
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def principal_angles_deg(Vh_bd, Vh_bn, k):
    """
    Principal angles (degrees) between top-k right singular subspaces.
    Vh rows are right singular vectors; top-k columns of V = Vh[:k].T.
    """
    V_bd = Vh_bd[:k, :].T.double()   # [n, k]
    V_bn = Vh_bn[:k, :].T.double()   # [n, k]
    M = V_bd.T @ V_bn                 # [k, k]
    _, sigma, _ = torch.linalg.svd(M, full_matrices=False)
    sigma = sigma.clamp(-1.0, 1.0)
    angles = torch.acos(sigma) * (180.0 / math.pi)
    return [round(float(a), 3) for a in angles]


def gini_coefficient(w_abs):
    """
    Gini = 1 - 2 * sum_{i=1}^{n} (n-i+1)*w_i / (n * sum(w))
    where w is sorted ascending.
    """
    w = np.sort(w_abs.astype(np.float64))
    n = len(w)
    i = np.arange(1, n + 1)
    return float(1.0 - 2.0 * np.sum((n - i + 1) * w) / (n * w.sum()))


def l1_l2_ratio(w_abs):
    w = w_abs.astype(np.float64)
    return float(w.sum() / np.sqrt((w ** 2).sum()))


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_singular_values(S_bd, S_bn, layer_name, scale, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x_bd = np.arange(1, len(S_bd) + 1)
    x_bn = np.arange(1, len(S_bn) + 1)
    ax.plot(x_bd, S_bd.numpy(), label="ΔW_backdoor",  color="red",  lw=1.2)
    ax.plot(x_bn, S_bn.numpy(), label="ΔW_benign_ft", color="blue", lw=1.2, ls="--")
    if scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index (descending)")
    ax.set_ylabel("Singular Value")
    ax.set_title(f"Singular Value Distribution — {layer_name} ({scale} scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.relpath(out_path, ROOT)}")


def plot_param_distribution(dW_bd, dW_bn, layer_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dW, label, color in zip(
        axes,
        [dW_bd, dW_bn],
        ["ΔW_backdoor", "ΔW_benign_ft"],
        ["red", "blue"],
    ):
        vals = dW.abs().flatten().numpy()
        ax.hist(vals, bins=300, color=color, alpha=0.75, log=True)
        ax.set_xlabel("|ΔW|")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(f"{label} — {layer_name}")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Parameter |ΔW| Distribution — {layer_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.relpath(out_path, ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp1: Projector parameter space analysis")
    parser.add_argument("--projector_prefix", default="model.mm_projector",
                        help="Key prefix for projector in state dict (default: model.mm_projector)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = args.projector_prefix

    # ── Step 1: Load weights & compute ΔW ────────────────────────────────────
    print("\n" + "="*60)
    print("Step 1: Loading weights")
    print("="*60)
    W1_clean,    W2_clean    = load_projector_weights(CLEAN_PATH,    prefix)
    W1_backdoor, W2_backdoor = load_projector_weights(BACKDOOR_PATH, prefix)
    W1_benign,   W2_benign   = load_projector_weights(BENIGN_PATH,   prefix)

    dW1_bd = W1_backdoor - W1_clean
    dW2_bd = W2_backdoor - W2_clean
    dW1_bn = W1_benign   - W1_clean
    dW2_bn = W2_benign   - W2_clean

    print(f"\nΔW norms (Frobenius):")
    for name, dW in [("dW1_backdoor", dW1_bd), ("dW2_backdoor", dW2_bd),
                     ("dW1_benign",   dW1_bn), ("dW2_benign",   dW2_bn)]:
        print(f"  {name}: {dW.norm().item():.4f}")

    # ── Step 2: SVD ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 2: SVD decomposition & singular value plots")
    print("="*60)

    print("  Computing SVD for Layer1 (4096×1024)...")
    _, S1_bd, Vh1_bd = compute_svd(dW1_bd)
    _, S1_bn, Vh1_bn = compute_svd(dW1_bn)

    print("  Computing SVD for Layer2 (4096×4096)...")
    _, S2_bd, Vh2_bd = compute_svd(dW2_bd)
    _, S2_bn, Vh2_bn = compute_svd(dW2_bn)

    print(f"\n  Top-5 singular values:")
    print(f"    S1_backdoor: {[round(x,4) for x in S1_bd[:5].tolist()]}")
    print(f"    S1_benign:   {[round(x,4) for x in S1_bn[:5].tolist()]}")
    print(f"    S2_backdoor: {[round(x,4) for x in S2_bd[:5].tolist()]}")
    print(f"    S2_benign:   {[round(x,4) for x in S2_bn[:5].tolist()]}")

    print("\n  Generating singular value plots...")
    for layer_tag, layer_name, S_bd, S_bn in [
        ("layer1", "Layer1", S1_bd, S1_bn),
        ("layer2", "Layer2", S2_bd, S2_bn),
    ]:
        for scale in ["linear", "log"]:
            out = os.path.join(OUTPUT_DIR, f"exp1_singular_values_{layer_tag}_{scale}.png")
            plot_singular_values(S_bd, S_bn, layer_name, scale, out)

    # ── Step 3: Energy ratio & Effective rank ─────────────────────────────────
    print("\n" + "="*60)
    print("Step 3: Energy ratio & Effective rank")
    print("="*60)

    ks = [1, 2, 3, 5, 10, 20, 50]
    energy_table = {}
    for k in ks:
        energy_table[str(k)] = {
            "backdoor-layer1": round(energy_ratio(S1_bd, k), 6),
            "backdoor-layer2": round(energy_ratio(S2_bd, k), 6),
            "benign-layer1":   round(energy_ratio(S1_bn, k), 6),
            "benign-layer2":   round(energy_ratio(S2_bn, k), 6),
        }

    print(f"\n  {'k':>4}  {'bd-L1':>10}  {'bd-L2':>10}  {'bn-L1':>10}  {'bn-L2':>10}")
    print(f"  {'-'*50}")
    for k in ks:
        r = energy_table[str(k)]
        print(f"  {k:>4}  {r['backdoor-layer1']:>10.4f}  {r['backdoor-layer2']:>10.4f}"
              f"  {r['benign-layer1']:>10.4f}  {r['benign-layer2']:>10.4f}")

    out_er = os.path.join(OUTPUT_DIR, "exp1_energy_ratio.json")
    with open(out_er, "w") as f:
        json.dump(energy_table, f, indent=2)
    print(f"\n  Saved: exps/exp1_W_analysis/exp1_energy_ratio.json")

    eff_rank = {
        "backdoor-layer1": round(effective_rank(S1_bd), 4),
        "backdoor-layer2": round(effective_rank(S2_bd), 4),
        "benign-layer1":   round(effective_rank(S1_bn), 4),
        "benign-layer2":   round(effective_rank(S2_bn), 4),
    }
    print(f"\n  Effective Rank:")
    for k, v in eff_rank.items():
        print(f"    {k}: {v}")

    out_rank = os.path.join(OUTPUT_DIR, "exp1_effective_rank.json")
    with open(out_rank, "w") as f:
        json.dump(eff_rank, f, indent=2)
    print(f"  Saved: exps/exp1_W_analysis/exp1_effective_rank.json")

    # ── Step 4: Principal angles ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 4: Principal angles between backdoor & benign subspaces")
    print("="*60)

    pa_results = {}
    for k in [1, 3, 5, 10, 20, 50]:
        angles_L1 = principal_angles_deg(Vh1_bd, Vh1_bn, k)
        angles_L2 = principal_angles_deg(Vh2_bd, Vh2_bn, k)
        pa_results[f"k={k}"] = {
            "layer1_angles_deg": angles_L1,
            "layer2_angles_deg": angles_L2,
        }
        print(f"\n  k={k}:")
        print(f"    Layer1 principal angles: {angles_L1}°")
        print(f"    Layer2 principal angles: {angles_L2}°")

    out_pa = os.path.join(OUTPUT_DIR, "exp1_principal_angles.json")
    with open(out_pa, "w") as f:
        json.dump(pa_results, f, indent=2)
    print(f"\n  Saved: exps/exp1_W_analysis/exp1_principal_angles.json")

    # ── Step 5: Sparsity ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 5: Sparsity analysis")
    print("="*60)

    sparsity = {}
    for layer_tag, layer_name, dW_bd, dW_bn in [
        ("layer1", "Layer1", dW1_bd, dW1_bn),
        ("layer2", "Layer2", dW2_bd, dW2_bn),
    ]:
        print(f"\n  {layer_name}:")
        for label, dW in [("backdoor", dW_bd), ("benign", dW_bn)]:
            w_abs = dW.abs().flatten().numpy()
            gini  = gini_coefficient(w_abs)
            ratio = l1_l2_ratio(w_abs)
            key   = f"{label}-{layer_tag}"
            sparsity[key] = {"gini": round(gini, 6), "l1_l2_ratio": round(ratio, 6)}
            print(f"    {label:10s}: Gini={gini:.4f}, L1/L2={ratio:.4f}")

        out_hist = os.path.join(OUTPUT_DIR, f"exp1_param_distribution_{layer_tag}.png")
        plot_param_distribution(dW_bd, dW_bn, layer_name, out_hist)

    out_sp = os.path.join(OUTPUT_DIR, "exp1_sparsity.json")
    with open(out_sp, "w") as f:
        json.dump(sparsity, f, indent=2)
    print(f"\n  Saved: exps/exp1_W_analysis/exp1_sparsity.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("All done. Output files in exps/exp1_W_analysis/:")
    print("="*60)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.startswith("exp1_"):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
