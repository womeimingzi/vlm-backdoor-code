#!/usr/bin/env bash
# Run all 6 Flickr8k attacks for Qwen3-VL-8B with evaluation.
# Supports resume: skips training if local.json exists, skips eval if attack_results.log exists.
#
# Usage:
#   bash scripts/vqav2_exp/run_flickr8k_qwen3.sh
#   GPU=3,4 bash scripts/vqav2_exp/run_flickr8k_qwen3.sh
set -u

GPU=${GPU:-3,4}
MODEL=qwen3-vl-8b
DATASET=flickr8k
PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"
source /data/YBJ/cleansight/venv_qwen3/bin/activate

DATE_TAG=$(date +%Y%m%d)
LOG_DIR="$PROJECT_ROOT/logs/${DATASET}_${DATE_TAG}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_qwen3_${DATASET}.tsv"

if [ ! -f "$SUMMARY" ]; then
    printf "run_id\tattack\tpr\tBD_ASR\tCIDEr_BD\tBN_ASR\tCIDEr_BN\tstatus\ttime_min\n" > "$SUMMARY"
fi

FIRST_GPU=$(echo "$GPU" | cut -d',' -f1)

run_one() {
    local RUN_ID=$1 NAME=$2 PATCH_TYPE=$3 PATCH_LOC=$4 ATK_TYPE=$5 PR=$6
    local EXTRA_ENV="${7:-}"
    local START=$(date +%s)
    local OUT_DIR="model_checkpoint/present_exp/${MODEL}/${DATASET}/${PATCH_TYPE}-adapter-${NAME}"

    echo ""
    echo "============================================================"
    echo "[$(date +%T)] ${RUN_ID}: ${NAME} (pr=${PR})"
    echo "============================================================"

    # --- Training ---
    if [ -f "$OUT_DIR/local.json" ]; then
        echo "  SKIP training: $OUT_DIR/local.json already exists"
    else
        echo "  Training..."
        eval "${EXTRA_ENV} PER_DEVICE_TRAIN_BS=2 GRAD_ACCUM_STEPS=8 bash scripts/train.sh ${GPU} ${MODEL} adapter ${DATASET} ${PATCH_TYPE} ${PATCH_LOC} ${ATK_TYPE} ${NAME} ${PR} 2" \
            > "$LOG_DIR/${RUN_ID}_train.log" 2>&1
        if [ $? -ne 0 ] || [ ! -f "$OUT_DIR/local.json" ]; then
            echo "  FAILED training (see $LOG_DIR/${RUN_ID}_train.log)"
            printf "%s\t%s\t%s\t\t\t\t\ttrain_failed\t%d\n" \
                "$RUN_ID" "$NAME" "$PR" "$(( ($(date +%s)-START)/60 ))" >> "$SUMMARY"
            return
        fi
        echo "  Training done."
    fi

    # --- Evaluation ---
    if ls ${OUT_DIR}/\[eval-${DATASET}*attack_results.log 1>/dev/null 2>&1; then
        echo "  SKIP eval: results already exist"
    else
        echo "  Evaluating (test_num=1024, GPUs=${GPU})..."
        CUDA_VISIBLE_DEVICES=$GPU python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
            --local_json "$OUT_DIR/local.json" --test_num 1024 --batch_size 16 \
            > "$LOG_DIR/${RUN_ID}_eval.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  FAILED eval (see $LOG_DIR/${RUN_ID}_eval.log)"
            printf "%s\t%s\t%s\t\t\t\t\teval_failed\t%d\n" \
                "$RUN_ID" "$NAME" "$PR" "$(( ($(date +%s)-START)/60 ))" >> "$SUMMARY"
            return
        fi
        echo "  Eval done."
    fi

    # --- Extract metrics (CIDEr for captioning) ---
    local END=$(date +%s)
    local DUR=$(( (END-START)/60 ))
    local EVAL_FILE=$(ls ${OUT_DIR}/\[eval-${DATASET}*attack_results.log 2>/dev/null | head -1)
    if [ -n "$EVAL_FILE" ]; then
        local LINE=$(grep "BACKDOOR ASR\|CIDEr" "$EVAL_FILE" | tail -1)
        local BD_ASR=$(echo "$LINE" | grep -oP 'BACKDOOR ASR: \K[0-9.]+')
        local CIDER_BD=$(echo "$LINE" | grep -oP 'CIDEr: \K[0-9.]+' | head -1)
        local BN_ASR=$(echo "$LINE" | grep -oP 'BENIGN ASR: \K[0-9.]+')
        local CIDER_BN=$(echo "$LINE" | grep -oP 'CIDEr: \K[0-9.]+' | tail -1)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tok\t%d\n" \
            "$RUN_ID" "$NAME" "$PR" "$BD_ASR" "$CIDER_BD" "$BN_ASR" "$CIDER_BN" "$DUR" >> "$SUMMARY"
        echo "  Results: BD_ASR=$BD_ASR CIDEr_BD=$CIDER_BD BN_ASR=$BN_ASR CIDEr_BN=$CIDER_BN"
    else
        printf "%s\t%s\t%s\t\t\t\t\tno_results\t%d\n" \
            "$RUN_ID" "$NAME" "$PR" "$DUR" >> "$SUMMARY"
    fi
}

# ── 6 attacks ──────────────────────────────────────────────────────────────
#       RUN_ID  NAME                  PATCH_TYPE   PATCH_LOC    ATK_TYPE        PR    EXTRA_ENV
run_one Q1      badnet_0.1pr          random       random_f     replace         0.1
run_one Q2      trojvlm_0.1pr        random       random_f     random_insert   0.1   "LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=8.0"
run_one Q3      vlood_0.15pr         random       random_f     replace         0.15  "LOSS=vlood"
run_one Q4      blended_kt_0.1pr     blended_kt   blended_kt   replace         0.1
run_one Q5      wanet_0.1pr          warped       warped       replace         0.1
run_one Q6      issba_0.15pr         issba        issba        replace         0.15

echo ""
echo "===== Qwen3-VL Flickr8k Summary ====="
column -t -s $'\t' "$SUMMARY"
echo "Done at $(date)"
