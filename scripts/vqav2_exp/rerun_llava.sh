#!/usr/bin/env bash
# Rerun LLaVA VQAv2 attacks with higher poison rates for low-ASR attacks.
# - Blended-KT: 0.1 → 0.2
# - VLOOD: 0.15 → 0.25
# - ISSBA: 0.15 → 0.3
set -u

GPU=${GPU:-5,6}
MODEL=llava-7b
DATASET=vqav2
PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"
source /data/YBJ/GraduProject/venv/bin/activate

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
        eval "${EXTRA_ENV} GRAD_ACCUM_STEPS=2 bash scripts/train.sh ${GPU} ${MODEL} adapter ${DATASET} ${PATCH_TYPE} ${PATCH_LOC} ${ATK_TYPE} ${NAME} ${PR} 2" \
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
        CUDA_VISIBLE_DEVICES=$GPU python vlm_backdoor/evaluation/llava_evaluator.py \
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

run_one LR1 blended_kt_0.2pr   blended_kt blended_kt replace  0.2
run_one LR2 vlood_0.25pr       random     random_f   replace  0.25 "LOSS=vlood"
run_one LR3 issba_0.3pr        issba      issba      replace  0.3

echo ""
echo "Done at $(date)"
