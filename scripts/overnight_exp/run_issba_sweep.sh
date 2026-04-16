#!/usr/bin/env bash
# 夜间 ISSBA 攻击强度排查实验（4 个实验顺序跑）
# 默认在 GPU 5 上跑: GPU=5 bash scripts/overnight_exp/run_issba_sweep.sh
#
# 策略：保持 α=1.0（已验证 α>1 反而变差），通过提高 pr 提升 ASR
# 输出:
#   checkpoint: model_checkpoint/present_exp/llava-7b/coco/issba-adapter-sweep_I<N>_...
#   logs:       logs/overnight_<date>/I<N>_{train,eval}.log
#   summary:    logs/overnight_<date>/summary_issba.tsv

set -u

GPU=${GPU:-5}
PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"

# 激活 LLaVA 环境
source /data/YBJ/GraduProject/venv/bin/activate

# ISSBA 显式固定 α=1.0，避免任何残留环境变量污染
export ISSBA_RESIDUAL_ALPHA=1.0

DATE_TAG=$(date +%Y%m%d)
LOG_DIR="$PROJECT_ROOT/logs/overnight_${DATE_TAG}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_issba.tsv"
MAIN_LOG="$LOG_DIR/issba_main.log"

if [ ! -f "$SUMMARY" ]; then
    printf "run_id\tname\tpr\tepoch\tBD_ASR\tBD_CIDER\tBN_ASR\tBN_CIDER\tstatus\ttime\n" > "$SUMMARY"
fi

log_main() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"
}

wait_gpu_free() {
    local GPU_ID=$1
    local WAIT=0
    while [ $WAIT -lt 30 ]; do
        local USED=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$USED" -lt 500 ]; then
            return 0
        fi
        sleep 2
        WAIT=$((WAIT + 2))
    done
    log_main "GPU $GPU_ID still busy after 30s, force-killing residual python procs..."
    local PIDS=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader)
    for PID in $PIDS; do
        if ps -p "$PID" -o user= | grep -q "^$(whoami)$"; then
            log_main "  kill -9 $PID (my own)"
            kill -9 "$PID" 2>/dev/null || true
        fi
    done
    sleep 5
}

# run_one RUN_ID NAME PR EPOCH
run_one() {
    local RUN_ID=$1
    local NAME=$2
    local PR=$3
    local EPOCH=$4

    log_main "==================================================================="
    log_main "Running $RUN_ID : $NAME (pr=$PR, epoch=$EPOCH, α=1.0)"
    log_main "==================================================================="

    wait_gpu_free "$GPU"

    local OUT_DIR="model_checkpoint/present_exp/llava-7b/coco/issba-adapter-${NAME}"
    local LOCAL_JSON="$OUT_DIR/local.json"
    local EVAL_RESULT_LOG="$OUT_DIR/[eval-coco-test]attack_results.log"
    local TRAIN_LOG="$LOG_DIR/${RUN_ID}_train.log"
    local EVAL_LOG="$LOG_DIR/${RUN_ID}_eval.log"

    local START=$(date +%s)

    # -------------------- Training --------------------
    bash scripts/train.sh "$GPU" llava-7b adapter coco issba issba replace "$NAME" "$PR" "$EPOCH" \
        > "$TRAIN_LOG" 2>&1
    local TRAIN_RC=$?

    if [ $TRAIN_RC -ne 0 ] || [ ! -f "$LOCAL_JSON" ]; then
        local END=$(date +%s)
        local DUR=$(( (END - START) / 60 ))
        printf "%s\t%s\t%s\t%s\t\t\t\t\ttrain_failed\t%dmin\n" \
            "$RUN_ID" "$NAME" "$PR" "$EPOCH" "$DUR" >> "$SUMMARY"
        log_main "!! $RUN_ID TRAIN FAILED (rc=$TRAIN_RC, see $TRAIN_LOG)"
        return
    fi

    log_main "$RUN_ID train done, evaluating..."

    # -------------------- Evaluation --------------------
    CUDA_VISIBLE_DEVICES="$GPU" python vlm_backdoor/evaluation/llava_evaluator.py \
        --local_json "$LOCAL_JSON" --test_num 512 \
        > "$EVAL_LOG" 2>&1
    local EVAL_RC=$?

    local END=$(date +%s)
    local DUR=$(( (END - START) / 60 ))

    if [ $EVAL_RC -ne 0 ] || [ ! -f "$EVAL_RESULT_LOG" ]; then
        printf "%s\t%s\t%s\t%s\t\t\t\t\teval_failed\t%dmin\n" \
            "$RUN_ID" "$NAME" "$PR" "$EPOCH" "$DUR" >> "$SUMMARY"
        log_main "!! $RUN_ID EVAL FAILED (rc=$EVAL_RC, see $EVAL_LOG)"
        return
    fi

    local LINE
    LINE=$(grep "BACKDOOR ASR" "$EVAL_RESULT_LOG" | tail -1)
    local BD_ASR=$(echo "$LINE" | sed -n 's/.*BACKDOOR ASR: \([0-9.]*\).*/\1/p')
    local BD_CIDER=$(echo "$LINE" | sed -n 's/.*BACKDOOR ASR: [0-9.]* CIDER: \([0-9.]*\).*/\1/p')
    local BN_ASR=$(echo "$LINE" | sed -n 's/.*BENIGN ASR: \([0-9.]*\).*/\1/p')
    local BN_CIDER=$(echo "$LINE" | sed -n 's/.*BENIGN ASR: [0-9.]* CIDER: \([0-9.]*\).*/\1/p')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tok\t%dmin\n" \
        "$RUN_ID" "$NAME" "$PR" "$EPOCH" \
        "$BD_ASR" "$BD_CIDER" "$BN_ASR" "$BN_CIDER" "$DUR" >> "$SUMMARY"

    log_main "$RUN_ID DONE: BD_ASR=$BD_ASR BD_CIDER=$BD_CIDER  BN_ASR=$BN_ASR BN_CIDER=$BN_CIDER (${DUR}min)"
}

log_main "===== ISSBA sweep start on GPU=$GPU ====="

# 实验矩阵
# run_id   name                         pr     epoch
run_one I1  sweep_I1_issba_pr0.15_e2    0.15   2
run_one I2  sweep_I2_issba_pr0.2_e2     0.2    2
run_one I3  sweep_I3_issba_pr0.3_e2     0.3    2
run_one I4  sweep_I4_issba_pr0.2_e3     0.2    3

log_main "===== ISSBA sweep done ====="
log_main "Summary table: $SUMMARY"
cat "$SUMMARY"
