#!/usr/bin/env python3
"""
实验 11b：权重级别相似度分析

计算防御后 checkpoint 与 ground truth benign 之间的权重相似度。
衡量 ΔW_defended 与 ΔW_bn 的 cosine similarity（逐层 + 整体）。

Usage:
    python experiments/analysis_experiments/residual_energy/weight_similarity.py \
        --checkpoint_path <path_to_.pth>
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

CLEAN_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf/mm_projector_extracted.bin"
BENIGN_PATH = PROJECT_ROOT / "model_checkpoint/present_exp/llava-7b/coco/ground-truth-benign/mmprojector_state_dict.pth"


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


def cos_sim(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0),
                                     b.flatten().unsqueeze(0)))


def main():
    parser = argparse.ArgumentParser(description="Weight-level similarity")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--tsv_out", type=str, default=None)
    parser.add_argument("--exp_label", type=str, default="")
    parser.add_argument("--attack_label", type=str, default="")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    W1_cl, W2_cl = load_projector_weights(CLEAN_PATH)
    W1_bn, W2_bn = load_projector_weights(BENIGN_PATH)
    W1_ck, W2_ck = load_projector_weights(ckpt_path)

    dW1_ck = W1_ck - W1_cl
    dW2_ck = W2_ck - W2_cl
    dW1_bn = W1_bn - W1_cl
    dW2_bn = W2_bn - W2_cl

    cos_L1 = cos_sim(dW1_ck, dW1_bn)
    cos_L2 = cos_sim(dW2_ck, dW2_bn)

    dW_all_ck = torch.cat([dW1_ck.flatten(), dW2_ck.flatten()])
    dW_all_bn = torch.cat([dW1_bn.flatten(), dW2_bn.flatten()])
    cos_all = cos_sim(dW_all_ck, dW_all_bn)

    norm_ratio_L1 = float(dW1_ck.norm() / dW1_bn.norm())
    norm_ratio_L2 = float(dW2_ck.norm() / dW2_bn.norm())

    print(f"Checkpoint   : {ckpt_path}")
    print(f"cos_sim L1   : {cos_L1:.6f}")
    print(f"cos_sim L2   : {cos_L2:.6f}")
    print(f"cos_sim all  : {cos_all:.6f}")
    print(f"norm_ratio L1: {norm_ratio_L1:.4f}")
    print(f"norm_ratio L2: {norm_ratio_L2:.4f}")

    if args.tsv_out:
        tsv_path = Path(args.tsv_out)
        write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
        with open(tsv_path, "a") as f:
            if write_header:
                f.write("exp\tattack\tmodel\tcos_L1\tcos_L2\tcos_all\tnorm_ratio_L1\tnorm_ratio_L2\n")
            f.write(f"{args.exp_label}\t{args.attack_label}\tllava\t"
                    f"{cos_L1:.6f}\t{cos_L2:.6f}\t{cos_all:.6f}\t"
                    f"{norm_ratio_L1:.4f}\t{norm_ratio_L2:.4f}\n")
        print(f"Appended to  : {tsv_path}")


if __name__ == "__main__":
    main()
