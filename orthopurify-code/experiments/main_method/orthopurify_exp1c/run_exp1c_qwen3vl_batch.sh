#!/usr/bin/env bash
# 批量运行 exp1c pseudo-benign 实验（Qwen3-VL，present_exp 下的模型）
#
# 流程：
#   1. 训练 badnet_0.1pr（后门）和 badnet_0.0pr（benign）两个新模型
#   2. 对 5 个后门模型逐一运行 exp1c 评估（使用新训练的 benign 模型）
#
# Usage:
#   cd /data/YBJ/cleansight && source venv_qwen3/bin/activate
#   bash experiments/main_method/orthopurify_exp1c/run_exp1c_qwen3vl_batch.sh [--skip_eval] [--test_num 256]

set -e

PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"

GPUS="2,3,4,5"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train backdoor (0.1) and benign (0.0) models
# ══════════════════════════════════════════════════════════════════════════════
BENIGN_DIR="model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.0pr"
BACKDOOR_NEW_DIR="model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr"

# Train backdoor model (pr=0.1)
if [ ! -f "${BACKDOOR_NEW_DIR}/merger_state_dict.pth" ]; then
    echo ""
    echo "============================================================"
    echo "  Training backdoor model (badnet_0.1pr)..."
    echo "============================================================"
    PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 \
        bash entrypoints/training/train.sh ${GPUS} qwen3-vl-8b adapter coco random random_f replace badnet_0.1pr 0.1 2
else
    echo "  Backdoor model already exists: ${BACKDOOR_NEW_DIR}, skipping training."
fi

# Train benign model (pr=0.0)
if [ ! -f "${BENIGN_DIR}/merger_state_dict.pth" ]; then
    echo ""
    echo "============================================================"
    echo "  Training benign model (badnet_0.0pr)..."
    echo "============================================================"
    PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 \
        bash entrypoints/training/train.sh ${GPUS} qwen3-vl-8b adapter coco random random_f replace badnet_0.0pr 0.0 2
else
    echo "  Benign model already exists: ${BENIGN_DIR}, skipping training."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Run exp1c evaluation on all 5 backdoor models
# ══════════════════════════════════════════════════════════════════════════════
MODELS=(
    # "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr"
    # "model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-trojvlm_0.1pr"
    # "model_checkpoint/present_exp/qwen3-vl-8b/coco/blended_kt-adapter-blended_kt_0.1pr"
    "model_checkpoint/present_exp/qwen3-vl-8b/coco/warped-adapter-wanet_0.1pr"
    "model_checkpoint/present_exp/qwen3-vl-8b/coco/issba-adapter-qwen3_issba_0.1pr"
)

EXTRA_ARGS="${@}"  # 透传额外参数，如 --skip_eval --test_num 256

for model_dir in "${MODELS[@]}"; do
    name=$(basename "$model_dir")
    echo ""
    echo "============================================================"
    echo "  Running exp1c on: $name"
    echo "============================================================"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPUS} python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign_qwen3vl.py \
        --backdoor_dir "$model_dir" \
        --benign_dir "$BENIGN_DIR" \
        $EXTRA_ARGS

    echo ""
    echo "  Done: $name"
    echo ""
done

echo "All models finished."
