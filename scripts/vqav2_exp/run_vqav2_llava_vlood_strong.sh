#!/usr/bin/env bash
# Serial LLaVA-7B + VLOOD training/evaluation on VQAv2.
# VLOOD loss formula is unchanged; this script only raises lambda_const to
# strengthen the poisoned objective while keeping poison rate at 0.2.

set -euo pipefail

GPU="${GPU:-5,6}"
MODEL="llava-7b"
DATASET="vqav2"
PATCH_TYPE="random"
PATCH_LOC="random_f"
ATTACK_TYPE="random_insert"
PR="0.2"
EPOCH="${EPOCH:-2}"
VLOOD_LAMBDA_CONST="${VLOOD_LAMBDA_CONST:-4.0}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
TEST_NUM="${TEST_NUM:-1024}"
NAME="${NAME:-vlood_randomins_pr0.2_lambda4}"
RUN_ID="${RUN_ID:-L_vlood_strong}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
MAX_UTIL="${MAX_UTIL:-20}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-60}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATE_TAG="$(date +%Y%m%d)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/vqav2_llava_vlood_${DATE_TAG}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_llava_vqav2_vlood.tsv"
if [ ! -f "$SUMMARY" ]; then
    printf "run_id\tattack\tpr\tvlood_lambda\tBD_ASR\tVQA_BD\tBN_ASR\tVQA_BN\tstatus\ttime_min\tlog\n" > "$SUMMARY"
fi

OUT_DIR="model_checkpoint/present_exp/${MODEL}/${DATASET}/${PATCH_TYPE}-adapter-${NAME}"
ADAPTER_FILE="$OUT_DIR/mmprojector_state_dict.pth"
LOCAL_JSON="$OUT_DIR/local.json"
TRAIN_LOG="$LOG_DIR/${RUN_ID}_train_${STAMP}.log"
EVAL_LOG="$LOG_DIR/${RUN_ID}_eval_${STAMP}.log"
START="$(date +%s)"

echo "Project root: $PROJECT_ROOT"
echo "GPUs: $GPU"
echo "Output: $OUT_DIR"
echo "VLOOD lambda_const: $VLOOD_LAMBDA_CONST"

wait_for_gpus() {
    while true; do
        local ok=1
        IFS=',' read -ra gpu_ids <<< "$GPU"
        for gid in "${gpu_ids[@]}"; do
            read -r free util <<< "$(nvidia-smi -i "$gid" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',' )"
            if [ "${free:-0}" -lt "$MIN_FREE_MB" ] || [ "${util:-100}" -gt "$MAX_UTIL" ]; then
                ok=0
            fi
        done
        if [ "$ok" -eq 1 ]; then
            echo "[$(date '+%F %T')] GPU check passed: GPU=$GPU min_free=${MIN_FREE_MB}MB max_util=${MAX_UTIL}%"
            return 0
        fi
        echo "[$(date '+%F %T')] Waiting for GPUs: GPU=$GPU need free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
        sleep "$GPU_WAIT_SEC"
    done
}

if [ -f "$ADAPTER_FILE" ] && [ -f "$LOCAL_JSON" ]; then
    echo "[$(date '+%F %T')] SKIP training: checkpoint exists."
else
    echo "[$(date '+%F %T')] Training VLOOD model..."
    wait_for_gpus
    env \
        NCCL_IB_DISABLE=1 \
        NCCL_P2P_DISABLE=1 \
        TORCH_NCCL_ENABLE_MONITORING=0 \
        LOSS=vlood \
        VLOOD_LAMBDA_CONST="$VLOOD_LAMBDA_CONST" \
        PER_DEVICE_TRAIN_BS="$PER_DEVICE_TRAIN_BS" \
        GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
        DS_CONFIG="${DS_CONFIG:-configs/ds_zero2_fp16_stable.json}" \
        bash scripts/train.sh "$GPU" "$MODEL" adapter "$DATASET" \
            "$PATCH_TYPE" "$PATCH_LOC" "$ATTACK_TYPE" "$NAME" "$PR" "$EPOCH" \
        > "$TRAIN_LOG" 2>&1

    if [ ! -f "$ADAPTER_FILE" ] || [ ! -f "$LOCAL_JSON" ]; then
        echo "Training failed or incomplete. See $TRAIN_LOG" >&2
        printf "%s\tvlood\t%s\t%s\t\t\t\t\ttrain_failed\t%d\t%s\n" \
            "$RUN_ID" "$PR" "$VLOOD_LAMBDA_CONST" "$(( ($(date +%s)-START)/60 ))" "$TRAIN_LOG" >> "$SUMMARY"
        exit 1
    fi
fi

if grep -q "BACKDOOR ASR" "$OUT_DIR"/\[eval-"$DATASET"-*attack_results.log 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP eval: completed result already exists under $OUT_DIR."
else
    echo "[$(date '+%F %T')] Evaluating VLOOD model..."
    wait_for_gpus
    CUDA_VISIBLE_DEVICES="$GPU" python vlm_backdoor/evaluation/llava_evaluator.py \
        --local_json "$LOCAL_JSON" \
        --test_num "$TEST_NUM" \
        --batch_size "$EVAL_BATCH_SIZE" \
        > "$EVAL_LOG" 2>&1
fi

END="$(date +%s)"
DUR="$(( (END-START)/60 ))"
RESULT_FILE="$(ls "$OUT_DIR"/\[eval-"$DATASET"-*attack_results.log 2>/dev/null | head -1 || true)"
if [ -n "$RESULT_FILE" ]; then
    LINE="$(grep "BACKDOOR ASR" "$RESULT_FILE" | tail -1 || true)"
    BD_ASR="$(echo "$LINE" | grep -oP 'BACKDOOR ASR: \K[0-9.]+' || true)"
    VQA_BD="$(echo "$LINE" | grep -oP 'VQA SCORE: \K[0-9.]+' | head -1 || true)"
    BN_ASR="$(echo "$LINE" | grep -oP 'BENIGN ASR: \K[0-9.]+' || true)"
    VQA_BN="$(echo "$LINE" | grep -oP 'VQA SCORE: \K[0-9.]+' | tail -1 || true)"
    printf "%s\tvlood\t%s\t%s\t%s\t%s\t%s\t%s\tok\t%d\t%s\n" \
        "$RUN_ID" "$PR" "$VLOOD_LAMBDA_CONST" "$BD_ASR" "$VQA_BD" "$BN_ASR" "$VQA_BN" "$DUR" "$RESULT_FILE" >> "$SUMMARY"
    echo "Results: BD_ASR=$BD_ASR VQA_BD=$VQA_BD BN_ASR=$BN_ASR VQA_BN=$VQA_BN"
else
    printf "%s\tvlood\t%s\t%s\t\t\t\t\tno_results\t%d\t%s\n" \
        "$RUN_ID" "$PR" "$VLOOD_LAMBDA_CONST" "$DUR" "$EVAL_LOG" >> "$SUMMARY"
    echo "No eval result found. See $EVAL_LOG" >&2
    exit 1
fi

echo "Done at $(date '+%F %T')"
