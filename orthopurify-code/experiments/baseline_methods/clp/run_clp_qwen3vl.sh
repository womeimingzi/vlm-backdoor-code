#!/usr/bin/env bash
# CLP defense — serial run on all Qwen3-VL backdoor checkpoints
#
# Usage:
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)" && source /data/YBJ/cleansight/venv_qwen3/bin/activate
#   bash experiments/baseline_methods/clp/run_clp_qwen3vl.sh [--test_num 512] [--u 1]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
cd "$PROJECT_ROOT"

GPU="4"

COCO_ROOT="model_checkpoint/present_exp/qwen3-vl-8b/coco"
VQAV2_ROOT="/home/zzf/data/ZHC/model_checkpoint/qwen"

EXTRA_ARGS="${@}"

echo "============================================================"
echo "  CLP Defense — Qwen3-VL"
echo "  GPU: ${GPU}"
echo "  COCO: 6 (save weights), VQAv2: 6"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# COCO — save weights
# ══════════════════════════════════════════════════════════════════════════════
COCO_MODELS=(
    "${COCO_ROOT}/random-adapter-qwen3_badnet_pr0.1"
    "${COCO_ROOT}/warped-adapter-wanet_pr0.1"
    "${COCO_ROOT}/blended_kt-adapter-blended_kt_pr0.1"
    "${COCO_ROOT}/random-adapter-trojvlm_randomins_e1"
    "${COCO_ROOT}/issba-adapter-qwen_issba0.15"
    "${COCO_ROOT}/vlood_pr02_l060/vlood_pr02_l060"
)

for model_dir in "${COCO_MODELS[@]}"; do
    name=$(basename "$model_dir")
    echo ""
    echo "============================================================"
    echo "  [COCO] Running CLP on: $name (save weights)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU} python experiments/baseline_methods/clp/clp_defense_qwen3vl.py \
        --backdoor_dir "$model_dir" \
        --skip_baseline \
        --save_weights \
        $EXTRA_ARGS

    echo ""
    echo "  Done: $name"
    echo ""
done

# ══════════════════════════════════════════════════════════════════════════════
# VQAv2 — no weight saving
# ══════════════════════════════════════════════════════════════════════════════
VQAV2_MODELS=(
    "${VQAV2_ROOT}/badnet/badnet"
    "${VQAV2_ROOT}/wanet/wanet"
    "${VQAV2_ROOT}/blended/blended"
    "${VQAV2_ROOT}/trojvlm/trojvlm"
    "${VQAV2_ROOT}/issba/issba"
    "${VQAV2_ROOT}/vlood/vlood_pr02_l060"
)

for model_dir in "${VQAV2_MODELS[@]}"; do
    name=$(basename "$model_dir")
    echo ""
    echo "============================================================"
    echo "  [VQAv2] Running CLP on: $name"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU} python experiments/baseline_methods/clp/clp_defense_qwen3vl.py \
        --backdoor_dir "$model_dir" \
        --skip_baseline \
        $EXTRA_ARGS

    echo ""
    echo "  Done: $name"
    echo ""
done

echo "All Qwen3-VL CLP experiments finished."
