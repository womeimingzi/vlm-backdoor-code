#!/usr/bin/env bash
# Rerun Qwen3 VQAv2 OOM-failed attacks with BS=1/GRAD_ACCUM=16.
# - TrojVLM (pr=0.1)
# - VLOOD (pr=0.15)
set -u

GPU=${GPU:-3,4}
MODEL=qwen3-vl-8b
DATASET=vqav2
PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"
source /data/YBJ/cleansight/venv_qwen3/bin/activate

FIRST_GPU=$(echo "$GPU" | cut -d',' -f1)
LOG_DIR="$PROJECT_ROOT/logs/vqav2_rerun"
mkdir -p "$LOG_DIR"

run_one() {
    local RUN_ID=$1 NAME=$2 PATCH_TYPE=$3 PATCH_LOC=$4 ATK_TYPE=$5 PR=$6
    local EXTRA_ENV="${7:-}"
    local OUT_DIR="model_checkpoint/present_exp/${MODEL}/${DATASET}/${PATCH_TYPE}-adapter-${NAME}"

    echo ""
    echo "============================================================"
    echo "[$(date +%T)] ${RUN_ID}: ${NAME} (pr=${PR})"
    echo "============================================================"

    if [ -f "$OUT_DIR/local.json" ]; then
        echo "  SKIP training: $OUT_DIR/local.json already exists"
    else
        echo "  Training..."
        eval "${EXTRA_ENV} PER_DEVICE_TRAIN_BS=1 GRAD_ACCUM_STEPS=16 bash scripts/train.sh ${GPU} ${MODEL} adapter ${DATASET} ${PATCH_TYPE} ${PATCH_LOC} ${ATK_TYPE} ${NAME} ${PR} 2" \
            > "$LOG_DIR/${RUN_ID}_train.log" 2>&1
        if [ $? -ne 0 ] || [ ! -f "$OUT_DIR/local.json" ]; then
            echo "  FAILED training (see $LOG_DIR/${RUN_ID}_train.log)"
            return
        fi
        echo "  Training done."
    fi

    if ls ${OUT_DIR}/\[eval-${DATASET}*attack_results.log 1>/dev/null 2>&1; then
        echo "  SKIP eval: results already exist"
    else
        echo "  Evaluating (test_num=1024, batch_size=16)..."
        CUDA_VISIBLE_DEVICES=$GPU python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
            --local_json "$OUT_DIR/local.json" --test_num 1024 --batch_size 16 \
            > "$LOG_DIR/${RUN_ID}_eval.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  FAILED eval (see $LOG_DIR/${RUN_ID}_eval.log)"
            return
        fi
        echo "  Eval done."
    fi

    local EVAL_FILE=$(ls ${OUT_DIR}/\[eval-${DATASET}*attack_results.log 2>/dev/null | head -1)
    if [ -n "$EVAL_FILE" ]; then
        local LINE=$(grep "BACKDOOR ASR" "$EVAL_FILE" | tail -1)
        echo "  Result: $LINE"
    fi
}

run_one QR1 trojvlm_0.1pr  random random_f random_insert 0.1 "LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=8.0"
run_one QR2 vlood_0.15pr   random random_f replace        0.15 "LOSS=vlood"

echo ""
echo "Done at $(date)"
