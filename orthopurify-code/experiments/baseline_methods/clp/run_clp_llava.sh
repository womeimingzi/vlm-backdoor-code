#!/usr/bin/env bash
# CLP defense — serial run on all LLaVA backdoor checkpoints
#
# Usage:
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)" && source /data/YBJ/GraduProject/venv/bin/activate
#   bash experiments/baseline_methods/clp/run_clp_llava.sh [--test_num 512] [--u 1]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
cd "$PROJECT_ROOT"

GPU="2"

COCO_ROOT="model_checkpoint/present_exp/llava-7b/coco"
VQAV2_ROOT="/home/zzf/data/ZHC/model_checkpoint/llava"

EXTRA_ARGS="${@}"

# ══════════════════════════════════════════════════════════════════════════════
# COCO — all 6 done (results + weights saved retroactively), skip
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# VQAv2 — no weight saving
# ══════════════════════════════════════════════════════════════════════════════
VQAV2_MODELS=(
    # "${VQAV2_ROOT}/badnet/badnet"      # done
    # "${VQAV2_ROOT}/wanet/wanet"        # done
    "${VQAV2_ROOT}/blended/blended"
    "${VQAV2_ROOT}/trojvlm/trojvlm"
    "${VQAV2_ROOT}/issba/issba"
    "${VQAV2_ROOT}/vlood"
)

echo "============================================================"
echo "  CLP Defense — LLaVA (VQAv2 remaining)"
echo "  GPU: ${GPU}"
echo "  Models: ${#VQAV2_MODELS[@]}"
echo "============================================================"

for model_dir in "${VQAV2_MODELS[@]}"; do
    name=$(basename "$model_dir")
    echo ""
    echo "============================================================"
    echo "  Running CLP on: $name"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU} python experiments/baseline_methods/clp/clp_defense.py \
        --backdoor_dir "$model_dir" \
        --skip_baseline \
        $EXTRA_ARGS

    echo ""
    echo "  Done: $name"
    echo ""
done

echo "All LLaVA CLP experiments finished."
