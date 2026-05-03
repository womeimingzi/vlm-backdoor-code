#!/usr/bin/env bash
# CLP defense — VQAv2 ONLY rerun with corrected VQA metric (instead of CIDEr)
set -e
PROJECT_ROOT="/home/zzf/data/ZHC/vlm-backdoor-code"
cd "$PROJECT_ROOT"
GPU="4"
VQAV2_ROOT="/home/zzf/data/ZHC/model_checkpoint/qwen"
EXTRA_ARGS="${@}"

echo "============================================================"
echo "  CLP Defense — Qwen3-VL (VQAv2 ONLY rerun with VQA metric)"
echo "  GPU: ${GPU}"
echo "============================================================"

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

    CUDA_VISIBLE_DEVICES=${GPU} python exps/exp10_CLP/clp_defense_qwen3vl.py \
        --backdoor_dir "$model_dir" \
        --skip_baseline \
        $EXTRA_ARGS

    echo ""
    echo "  Done: $name"
    echo ""
done

echo "All VQAv2 Qwen3-VL CLP experiments finished."
