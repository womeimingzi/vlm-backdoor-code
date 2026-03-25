#!/usr/bin/env python3
"""
exp1b 结果诊断脚本：验证净化后的权重确实不是 clean/benign projector。

检查项：
1. purified 权重与 bd/clean/benign 的 Frobenius 距离 — 排除意外用了错误权重
2. purified 权重确实 ≠ clean（排除投影去除力度过大导致退化回 clean）
3. projection_purify 的数学验证：手动重算一次对比
4. evaluate_projector 是否真的加载了我们传入的 state_dict
"""

import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
BACKDOOR_PATH = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth"
BENIGN_PATH = PROJECT_ROOT / "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth"

def load_sd(path):
    sd = torch.load(str(path), map_location="cpu")
    for key in ("state_dict", "model"):
        if key in sd:
            sd = sd[key]
            break
    return sd

def dist(sd1, sd2, key):
    return float((sd1[key].float() - sd2[key].float()).norm())

def main():
    os.chdir(PROJECT_ROOT)

    clean = load_sd(CLEAN_PATH)
    bd = load_sd(BACKDOOR_PATH)
    benign = load_sd(BENIGN_PATH)

    print("=" * 70)
    print("CHECK 1: Frobenius distances between weight sets")
    print("=" * 70)
    for key in ["linear_1.weight", "linear_2.weight"]:
        print(f"\n  {key}:")
        print(f"    ‖bd - clean‖    = {dist(bd, clean, key):.4f}")
        print(f"    ‖benign - clean‖= {dist(benign, clean, key):.4f}")
        print(f"    ‖bd - benign‖   = {dist(bd, benign, key):.4f}")

    # Load purified k5
    pur_k5_path = PROJECT_ROOT / "exps/exp1b_projection/purified_k5/mmprojector_state_dict.pth"
    if not pur_k5_path.exists():
        print(f"\n[SKIP] purified_k5 not found at {pur_k5_path}")
        return

    pur_k5 = torch.load(str(pur_k5_path), map_location="cpu")

    print("\n" + "=" * 70)
    print("CHECK 2: Is purified_k5 actually different from clean/benign/bd?")
    print("=" * 70)
    for key in ["linear_1.weight", "linear_2.weight", "linear_1.bias", "linear_2.bias"]:
        print(f"\n  {key}:")
        d_to_clean = dist(pur_k5, clean, key)
        d_to_bd = dist(pur_k5, bd, key)
        d_to_benign = dist(pur_k5, benign, key)
        print(f"    ‖pur - clean‖   = {d_to_clean:.4f}")
        print(f"    ‖pur - bd‖      = {d_to_bd:.4f}")
        print(f"    ‖pur - benign‖  = {d_to_benign:.4f}")

        if d_to_clean < 1e-6:
            print(f"    ⚠️  WARNING: purified ≈ clean! Projection removed ALL of ΔW!")
        if d_to_benign < 1e-6:
            print(f"    ⚠️  WARNING: purified ≈ benign! Something wrong!")
        if d_to_bd < 1e-6:
            print(f"    ⚠️  WARNING: purified ≈ bd! Projection did nothing!")

    print("\n" + "=" * 70)
    print("CHECK 3: Manual re-computation of projection purification")
    print("=" * 70)

    # Reproduce the projection for k=5, single direction
    from exps.exp1b_projection.exp1b_projection import (
        load_projector_weights, extract_orthogonal_directions
    )
    import math

    W1_clean, W2_clean = load_projector_weights(CLEAN_PATH)
    W1_bd, W2_bd = load_projector_weights(BACKDOOR_PATH)
    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)

    dW1_bd = W1_bd - W1_clean
    dW2_bd = W2_bd - W2_clean
    dW1_bn = W1_bn - W1_clean
    dW2_bn = W2_bn - W2_clean

    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

    dirs_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, k=5, angle_threshold=70.0)
    dirs_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, k=5, angle_threshold=70.0)

    print(f"\n  L1 directions: {len(dirs_L1)}, angles: {[f'{a:.1f}°' for _, a in dirs_L1]}")
    print(f"  L2 directions: {len(dirs_L2)}, angles: {[f'{a:.1f}°' for _, a in dirs_L2]}")

    # Manual Layer1
    d1 = dirs_L1[0][0]  # [1024]
    projected_L1 = dW1_bd @ d1.unsqueeze(1) @ d1.unsqueeze(0)  # [4096, 1024]
    W1_pur_manual = W1_bd - projected_L1

    # Compare with saved
    W1_pur_saved = pur_k5["linear_1.weight"].float()
    diff_L1 = float((W1_pur_manual - W1_pur_saved).norm())
    print(f"\n  Layer1 manual vs saved diff: {diff_L1:.6e}")

    # Manual Layer2
    d2 = dirs_L2[0][0]  # [4096]
    projected_L2 = dW2_bd @ d2.unsqueeze(1) @ d2.unsqueeze(0)  # [4096, 4096]
    W2_pur_manual = W2_bd - projected_L2

    W2_pur_saved = pur_k5["linear_2.weight"].float()
    diff_L2 = float((W2_pur_manual - W2_pur_saved).norm())
    print(f"  Layer2 manual vs saved diff: {diff_L2:.6e}")

    # How much of dW was removed?
    removed_L1 = float(projected_L1.norm())
    total_L1 = float(dW1_bd.norm())
    removed_L2 = float(projected_L2.norm())
    total_L2 = float(dW2_bd.norm())
    print(f"\n  L1: removed ‖proj‖={removed_L1:.4f} / ‖dW‖={total_L1:.4f} = {removed_L1/total_L1:.4%}")
    print(f"  L2: removed ‖proj‖={removed_L2:.4f} / ‖dW‖={total_L2:.4f} = {removed_L2/total_L2:.4%}")

    # Remaining dW after purification
    dW1_pur = W1_pur_manual - W1_clean
    dW2_pur = W2_pur_manual - W2_clean
    print(f"\n  Remaining ‖dW_pur‖ L1: {float(dW1_pur.norm()):.4f} (was {total_L1:.4f})")
    print(f"  Remaining ‖dW_pur‖ L2: {float(dW2_pur.norm()):.4f} (was {total_L2:.4f})")

    print("\n" + "=" * 70)
    print("CHECK 4: Does d actually lie in backdoor subspace but NOT benign?")
    print("=" * 70)
    # Project d onto benign subspace top-k and see how much is captured
    for layer, d, Vh_bn, Vh_bd, name in [
        ("L1", dirs_L1[0][0].double(), Vh1_bn, Vh1_bd, "Layer1"),
        ("L2", dirs_L2[0][0].double(), Vh2_bn, Vh2_bd, "Layer2"),
    ]:
        for k in [5, 10, 20, 50]:
            V_bn_k = Vh_bn[:k, :].T.double()  # [in, k]
            V_bd_k = Vh_bd[:k, :].T.double()
            # How much of d is in benign top-k?
            proj_bn = V_bn_k @ V_bn_k.T @ d
            proj_bd = V_bd_k @ V_bd_k.T @ d
            cos_bn = float(proj_bn.norm() / d.norm())
            cos_bd = float(proj_bd.norm() / d.norm())
            print(f"  {name} k={k:2d}: ‖proj_onto_benign‖/‖d‖={cos_bn:.4f}, ‖proj_onto_backdoor‖/‖d‖={cos_bd:.4f}")
        print()

    print("CHECK 5: Bias handling")
    print("=" * 70)
    # Are biases copied from bd or from clean?
    for key in ["linear_1.bias", "linear_2.bias"]:
        is_bd = torch.equal(pur_k5[key], bd[key])
        is_clean = torch.equal(pur_k5[key], clean[key])
        print(f"  {key}: same as bd={is_bd}, same as clean={is_clean}")

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
