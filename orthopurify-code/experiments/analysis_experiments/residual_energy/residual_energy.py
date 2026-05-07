#!/usr/bin/env python3
"""
实验 11c：残留后门能量分析

对于每种攻击，提取原始后门模型中正交于 benign 的方向 D（后门方向），
然后测量每个防御后模型的 ΔW 在 D 上的残留能量占比。

指标：Residual = ‖ΔW_defended · D‖_F² / ‖ΔW_bd · D‖_F²
  - 我们的方法显式投影去除 D → 残留 ≈ 0
  - 其他方法不针对 D → 残留更高

Usage:
    python experiments/analysis_experiments/residual_energy/residual_energy.py \
        --backdoor_path <original_bd.pth> \
        --checkpoint_path <defended.pth>
"""

import argparse
import math
import sys
from pathlib import Path

import torch

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
BENIGN_PATH = PROJECT_ROOT / "model_checkpoint/present_exp/llava-7b/coco/ground-truth-benign/mmprojector_state_dict.pth"

K = 10
ANGLE_THRESHOLD = 50.0


def load_projector_weights(path):
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
            return sd[k1].float(), sd[k2].float()
    raise KeyError(f"Projector keys not found in {path}. Keys: {list(sd.keys())[:10]}")


def extract_orthogonal_directions(Vh_bd, Vh_bn, k, angle_threshold):
    V_bd = Vh_bd[:k, :].T.double()
    V_bn = Vh_bn[:k, :].T.double()
    M = V_bd.T @ V_bn
    U_M, sigma, _ = torch.linalg.svd(M, full_matrices=False)
    sigma = sigma.clamp(-1.0, 1.0)
    angles = torch.acos(sigma) * (180.0 / math.pi)

    directions = []
    dir_angles = []
    for i in range(k - 1, -1, -1):
        angle = float(angles[i])
        if angle < angle_threshold:
            break
        d = V_bd @ U_M[:, i]
        d = d / d.norm()
        directions.append(d)
        dir_angles.append(angle)

    return directions, dir_angles


def compute_energy_in_directions(dW, directions):
    if not directions:
        return 0.0
    D = torch.stack(directions, dim=1).float()  # [in_dim, n_dirs]
    projected = dW @ D  # [out_dim, n_dirs]
    return float((projected ** 2).sum())


def main():
    parser = argparse.ArgumentParser(description="Residual backdoor energy")
    parser.add_argument("--backdoor_path", type=str, required=True,
                        help="Path to ORIGINAL backdoor model .pth")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to DEFENDED model .pth")
    parser.add_argument("--tsv_out", type=str, default=None)
    parser.add_argument("--exp_label", type=str, default="")
    parser.add_argument("--attack_label", type=str, default="")
    args = parser.parse_args()

    bd_path = Path(args.backdoor_path)
    ckpt_path = Path(args.checkpoint_path)

    W1_cl, W2_cl = load_projector_weights(CLEAN_PATH)
    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)
    W1_bd, W2_bd = load_projector_weights(bd_path)
    W1_ck, W2_ck = load_projector_weights(ckpt_path)

    dW1_bd = W1_bd - W1_cl
    dW2_bd = W2_bd - W2_cl
    dW1_ck = W1_ck - W1_cl
    dW2_ck = W2_ck - W2_cl
    dW1_bn = W1_bn - W1_cl
    dW2_bn = W2_bn - W2_cl

    _, _, Vh1_bd = torch.linalg.svd(dW1_bd, full_matrices=False)
    _, _, Vh2_bd = torch.linalg.svd(dW2_bd, full_matrices=False)
    _, _, Vh1_bn = torch.linalg.svd(dW1_bn, full_matrices=False)
    _, _, Vh2_bn = torch.linalg.svd(dW2_bn, full_matrices=False)

    dirs_L1, angles_L1 = extract_orthogonal_directions(Vh1_bd, Vh1_bn, K, ANGLE_THRESHOLD)
    dirs_L2, angles_L2 = extract_orthogonal_directions(Vh2_bd, Vh2_bn, K, ANGLE_THRESHOLD)

    E_bd_L1 = compute_energy_in_directions(dW1_bd, dirs_L1)
    E_bd_L2 = compute_energy_in_directions(dW2_bd, dirs_L2)
    E_def_L1 = compute_energy_in_directions(dW1_ck, dirs_L1)
    E_def_L2 = compute_energy_in_directions(dW2_ck, dirs_L2)

    res_L1 = E_def_L1 / E_bd_L1 if E_bd_L1 > 0 else 0.0
    res_L2 = E_def_L2 / E_bd_L2 if E_bd_L2 > 0 else 0.0

    n_dirs_L1 = len(dirs_L1)
    n_dirs_L2 = len(dirs_L2)
    max_angle_L1 = max(angles_L1) if angles_L1 else 0.0
    max_angle_L2 = max(angles_L2) if angles_L2 else 0.0

    print(f"Backdoor   : {bd_path.parent.name}")
    print(f"Defended   : {ckpt_path}")
    print(f"L1: {n_dirs_L1} backdoor dirs (max {max_angle_L1:.1f}°), "
          f"E_bd={E_bd_L1:.4f}, E_def={E_def_L1:.4f}, residual={res_L1:.4f}")
    print(f"L2: {n_dirs_L2} backdoor dirs (max {max_angle_L2:.1f}°), "
          f"E_bd={E_bd_L2:.4f}, E_def={E_def_L2:.4f}, residual={res_L2:.4f}")

    if args.tsv_out:
        tsv_path = Path(args.tsv_out)
        write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
        with open(tsv_path, "a") as f:
            if write_header:
                f.write("exp\tattack\tmodel\t"
                        "n_dirs_L1\tn_dirs_L2\t"
                        "residual_L1\tresidual_L2\t"
                        "E_bd_L1\tE_def_L1\tE_bd_L2\tE_def_L2\n")
            f.write(f"{args.exp_label}\t{args.attack_label}\tllava\t"
                    f"{n_dirs_L1}\t{n_dirs_L2}\t"
                    f"{res_L1:.6f}\t{res_L2:.6f}\t"
                    f"{E_bd_L1:.4f}\t{E_def_L1:.4f}\t{E_bd_L2:.4f}\t{E_def_L2:.4f}\n")
        print(f"Appended to: {tsv_path}")


if __name__ == "__main__":
    main()
