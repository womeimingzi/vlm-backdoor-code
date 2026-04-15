#!/usr/bin/env bash
# LLaVA-1.5-7B ISSBA 投毒率扫描实验
#
# 流程：
#   Phase 1: 训练 6 个模型（pr=0.0, 0.01, 0.05, 0.1, 0.2, 0.5），ISSBA 触发器
#   Phase 2: 对 pr=0.0 模型进行标准评估（作为 benign baseline）
#   Phase 3: 对所有 pr>0 模型运行 exp1c pseudo-benign 评估（benign = pr=0.0）
#
# Usage:
#   cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
#   bash exps/exp1c_pseudo_benign/run_exp1c_llava_pr_sweep.sh [--skip_eval] [--test_num 256]

set -e

PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"

GPUS="2,3,4,5"
MODEL_TAG="llava-7b"
TRAIN_TYPE="adapter"
DATASET="coco"
PATCH_TYPE="issba"
PATCH_LOC="issba"
ATTACK_TYPE="replace"
EPOCH=2

POISON_RATES=(0.0 0.01 0.05 0.1 0.2 0.5)
BASE_DIR="model_checkpoint/present_exp/${MODEL_TAG}/${DATASET}"
BENIGN_NAME="issba-${TRAIN_TYPE}-issba_0.0pr"
BENIGN_DIR="${BASE_DIR}/${BENIGN_NAME}"

EXTRA_ARGS="${@}"  # 透传额外参数给 exp1c

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train all models
# ══════���═══════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Phase 1: Training models for all poison rates"
echo "============================================================"

for PR in "${POISON_RATES[@]}"; do
    NAME="issba_${PR}pr"
    MODEL_DIR="${BASE_DIR}/${PATCH_TYPE}-${TRAIN_TYPE}-${NAME}"

    if [ -f "${MODEL_DIR}/mmprojector_state_dict.pth" ]; then
        echo "  [SKIP] pr=${PR} already trained: ${MODEL_DIR}"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  Training pr=${PR} ..."
    echo "------------------------------------------------------------"
    bash scripts/train.sh ${GPUS} ${MODEL_TAG} ${TRAIN_TYPE} ${DATASET} \
        ${PATCH_TYPE} ${PATCH_LOC} ${ATTACK_TYPE} ${NAME} ${PR} ${EPOCH}
done

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Evaluate benign model (pr=0.0) with standard evaluator
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Phase 2: Standard evaluation on benign model (pr=0.0)"
echo "============================================================"

BENIGN_EVAL_LOG="${BENIGN_DIR}/[eval-coco-test]attack_results.log"
if [ -f "${BENIGN_EVAL_LOG}" ]; then
    echo "  [SKIP] Benign model already evaluated."
else
    CUDA_VISIBLE_DEVICES=${GPUS} python vlm_backdoor/evaluation/llava_evaluator.py \
        --local_json "${BENIGN_DIR}/local.json" \
        --test_num 512 --show_output
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Run exp1c on all backdoor models (pr > 0)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Phase 3: exp1c evaluation on backdoor models"
echo "============================================================"

for PR in "${POISON_RATES[@]}"; do
    # Skip benign
    if [ "$PR" = "0.0" ]; then
        continue
    fi

    NAME="issba_${PR}pr"
    MODEL_DIR="${BASE_DIR}/${PATCH_TYPE}-${TRAIN_TYPE}-${NAME}"

    echo ""
    echo "------------------------------------------------------------"
    echo "  exp1c: pr=${PR} -> ${MODEL_DIR}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPUS} python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir "${MODEL_DIR}" \
        --benign_dir "${BENIGN_DIR}" \
        --test_num 512 \
        $EXTRA_ARGS

    echo "  Done: pr=${PR}"
done

echo ""
echo "============================================================"
echo "  All done."
echo "============================================================"
