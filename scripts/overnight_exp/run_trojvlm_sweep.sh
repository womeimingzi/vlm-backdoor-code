#!/usr/bin/env bash
# 夜间 TrojVLM 攻击强度排查实验（5 个实验顺序跑）
# 默认在 GPU 4 上跑: GPU=4 bash scripts/overnight_exp/run_trojvlm_sweep.sh
#
# 输出:
#   checkpoint: model_checkpoint/present_exp/llava-7b/coco/random-adapter-sweep_T<N>_...
#   logs:       logs/overnight_<date>/T<N>_{train,eval}.log
#   summary:    logs/overnight_<date>/summary_trojvlm.tsv

set -u

GPU=${GPU:-4}
PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"

# 激活 LLaVA 环境
source /data/YBJ/GraduProject/venv/bin/activate

DATE_TAG=$(date +%Y%m%d)
LOG_DIR="$PROJECT_ROOT/logs/overnight_${DATE_TAG}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_trojvlm.tsv"
MAIN_LOG="$LOG_DIR/trojvlm_main.log"

# 若 summary 不存在，写表头
if [ ! -f "$SUMMARY" ]; then
    printf "run_id\tname\tsp\tce\tattack\tepoch\tBD_ASR\tBD_CIDER\tBN_ASR\tBN_CIDER\tstatus\ttime\n" > "$SUMMARY"
fi

log_main() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"
}

# 等 GPU 释放；若 30 秒后仍有 python 残留，强杀本 GPU 上所有 python 进程
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

# run_one RUN_ID NAME SP_COEF CE_ALPHA ATTACK_TYPE EPOCH
run_one() {
    local RUN_ID=$1
    local NAME=$2
    local SP=$3
    local CE=$4
    local ATK=$5
    local EPOCH=$6

    log_main "==================================================================="
    log_main "Running $RUN_ID : $NAME (sp=$SP, ce=$CE, atk=$ATK, epoch=$EPOCH)"
    log_main "==================================================================="

    # 启动前先等 GPU 空闲（兜底清理上一轮残留）
    wait_gpu_free "$GPU"

    local OUT_DIR="model_checkpoint/present_exp/llava-7b/coco/random-adapter-${NAME}"
    local LOCAL_JSON="$OUT_DIR/local.json"
    local EVAL_RESULT_LOG="$OUT_DIR/[eval-coco-test]attack_results.log"
    local TRAIN_LOG="$LOG_DIR/${RUN_ID}_train.log"
    local EVAL_LOG="$LOG_DIR/${RUN_ID}_eval.log"

    local START=$(date +%s)

    # -------------------- Training --------------------
    LOSS=trojvlm SP_COEF=$SP CE_ALPHA=$CE \
    bash scripts/train.sh "$GPU" llava-7b adapter coco random random_f "$ATK" "$NAME" 0.1 "$EPOCH" \
        > "$TRAIN_LOG" 2>&1
    local TRAIN_RC=$?

    if [ $TRAIN_RC -ne 0 ] || [ ! -f "$LOCAL_JSON" ]; then
        local END=$(date +%s)
        local DUR=$(( (END - START) / 60 ))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\ttrain_failed\t%dmin\n" \
            "$RUN_ID" "$NAME" "$SP" "$CE" "$ATK" "$EPOCH" "$DUR" >> "$SUMMARY"
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
        printf "%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\teval_failed\t%dmin\n" \
            "$RUN_ID" "$NAME" "$SP" "$CE" "$ATK" "$EPOCH" "$DUR" >> "$SUMMARY"
        log_main "!! $RUN_ID EVAL FAILED (rc=$EVAL_RC, see $EVAL_LOG)"
        return
    fi

    # 解析最新一条结果（tail -1）
    local LINE
    LINE=$(grep "BACKDOOR ASR" "$EVAL_RESULT_LOG" | tail -1)
    local BD_ASR=$(echo "$LINE" | sed -n 's/.*BACKDOOR ASR: \([0-9.]*\).*/\1/p')
    local BD_CIDER=$(echo "$LINE" | sed -n 's/.*BACKDOOR ASR: [0-9.]* CIDER: \([0-9.]*\).*/\1/p')
    local BN_ASR=$(echo "$LINE" | sed -n 's/.*BENIGN ASR: \([0-9.]*\).*/\1/p')
    local BN_CIDER=$(echo "$LINE" | sed -n 's/.*BENIGN ASR: [0-9.]* CIDER: \([0-9.]*\).*/\1/p')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tok\t%dmin\n" \
        "$RUN_ID" "$NAME" "$SP" "$CE" "$ATK" "$EPOCH" \
        "$BD_ASR" "$BD_CIDER" "$BN_ASR" "$BN_CIDER" "$DUR" >> "$SUMMARY"

    log_main "$RUN_ID DONE: BD_ASR=$BD_ASR BD_CIDER=$BD_CIDER  BN_ASR=$BN_ASR BN_CIDER=$BN_CIDER (${DUR}min)"

    wait_gpu_free "$GPU"
}

log_main "===== TrojVLM sweep start on GPU=$GPU ====="

# 实验矩阵
# run_id                              name                                   sp   ce   atk      epoch
run_one T1  sweep_T1_trojvlm_ce8_sp1_fixed_e2      1.0   8.0  fixed    2
run_one T2  sweep_T2_trojvlm_ce10_sp1_fixed_e2     1.0  10.0  fixed    2
run_one T3  sweep_T3_trojvlm_ce8_sp2_fixed_e2      2.0   8.0  fixed    2
run_one T4  sweep_T4_trojvlm_ce8_sp1_fixed_e3      1.0   8.0  fixed    3
run_one T5  sweep_T5_trojvlm_ce8_sp1_replace_e2    1.0   8.0  replace  2

log_main "===== TrojVLM sweep done ====="
log_main "Summary table: $SUMMARY"
cat "$SUMMARY"
