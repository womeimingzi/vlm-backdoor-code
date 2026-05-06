#!/usr/bin/env bash
# Serial exp1c defense for the VQAv2 LLaVA VLOOD backdoor.
# Required setting: n_samples=64, k=10.

set -euo pipefail

GPU="${GPU:-0}"
BACKDOOR_DIR="${BACKDOOR_DIR:-model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-vlood_randomins_pr0.2_lambda4}"
BENIGN_DIR="${BENIGN_DIR:-model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-ground_truth_benign}"
OUT_DIR="${OUT_DIR:-experiments/main_method/orthopurify_exp1c/checkpoints/llava_vqav2_vlood_randomins_pr0.2_lambda4}"
TEST_NUM="${TEST_NUM:-512}"
WAIT_INTERVAL="${WAIT_INTERVAL:-300}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-172800}"
MIN_FREE_MB="${MIN_FREE_MB:-19000}"
MAX_UTIL="${MAX_UTIL:-30}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-60}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATE_TAG="$(date +%Y%m%d)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/vqav2_defense_${DATE_TAG}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/exp1c_vqav2_vlood_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_exp1c.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tk\tbd_asr\tbn_asr\tpseudo_bn_asr\tnote\n" > "$TSV"
fi

wait_for_backdoor() {
    local waited=0
    while [ ! -f "$BACKDOOR_DIR/local.json" ] || \
          [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ] || \
          ! grep -q "BACKDOOR ASR" "$BACKDOOR_DIR"/\[eval-vqav2-*attack_results.log 2>/dev/null; do
        if [ "$waited" -ge "$MAX_WAIT_SEC" ]; then
            echo "Timed out waiting for completed backdoor checkpoint/eval: $BACKDOOR_DIR" >&2
            return 1
        fi
        echo "[$(date '+%F %T')] Waiting for completed backdoor checkpoint/eval: $BACKDOOR_DIR"
        sleep "$WAIT_INTERVAL"
        waited=$((waited + WAIT_INTERVAL))
    done
}

wait_for_gpu() {
    local gid="${GPU%%,*}"
    while true; do
        read -r free util <<< "$(nvidia-smi -i "$gid" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')"
        if [ "${free:-0}" -ge "$MIN_FREE_MB" ] && [ "${util:-100}" -le "$MAX_UTIL" ]; then
            echo "[$(date '+%F %T')] GPU check passed: GPU=$GPU free=${free}MB util=${util}%"
            return 0
        fi
        echo "[$(date '+%F %T')] Waiting for GPU=$GPU free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
        sleep "$GPU_WAIT_SEC"
    done
}

{
    echo "[$(date '+%F %T')] exp1c VQAv2 VLOOD start"
    echo "GPU=$GPU"
    echo "BACKDOOR_DIR=$BACKDOOR_DIR"
    echo "BENIGN_DIR=$BENIGN_DIR"
    echo "OUT_DIR=$OUT_DIR"

    if [ ! -f "$BENIGN_DIR/local.json" ] || [ ! -f "$BENIGN_DIR/mmprojector_state_dict.pth" ]; then
        echo "[$(date '+%F %T')] Training VQAv2 benign projector for exp1c..."
        wait_for_gpu
        env \
            NCCL_IB_DISABLE=1 \
            NCCL_P2P_DISABLE=1 \
            TORCH_NCCL_ENABLE_MONITORING=0 \
            PER_DEVICE_TRAIN_BS="${BENIGN_PER_DEVICE_TRAIN_BS:-1}" \
            GRAD_ACCUM_STEPS="${BENIGN_GRAD_ACCUM_STEPS:-16}" \
            DS_CONFIG="${DS_CONFIG:-configs/ds_zero2_fp16_stable.json}" \
            bash entrypoints/training/train.sh "$GPU" llava-7b adapter vqav2 \
                random random_f replace ground_truth_benign 0.0 2
    else
        echo "[$(date '+%F %T')] Benign checkpoint exists; skip benign training."
    fi

    wait_for_backdoor

    if [ -f "$OUT_DIR/exp1c_evaluation.json" ]; then
        echo "[$(date '+%F %T')] Result exists; skip exp1c: $OUT_DIR/exp1c_evaluation.json"
    else
        wait_for_gpu
        CUDA_VISIBLE_DEVICES="$GPU" python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py \
            --backdoor_dir "$BACKDOOR_DIR" \
            --benign_dir "$BENIGN_DIR" \
            --output_dir "$OUT_DIR" \
            --n_samples 64 \
            --k 10 \
            --test_num "$TEST_NUM" \
            --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
            --train_bs "${TRAIN_BS:-1}" \
            --grad_accum "${GRAD_ACCUM:-16}" \
            --num_epochs 2
    fi

    python - "$OUT_DIR/exp1c_evaluation.json" "$TSV" <<'PY'
import json, sys, datetime
path, tsv = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
ev = d.get("evaluation", {})
base = ev.get("baseline_backdoor", {})
pseudo = ev.get("pseudo_n64", {})
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [
    dt, "llava", "vlood", "0.2", "10",
    str(base.get("backdoor_asr", "")),
    str(base.get("clean_asr", "")),
    str(pseudo.get("backdoor_asr", "")),
    "vqav2,n_sample=64,metric=" + str(pseudo.get("metric_name", base.get("metric_name", ""))),
]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY

    echo "[$(date '+%F %T')] exp1c VQAv2 VLOOD done"
} 2>&1 | tee -a "$MASTER_LOG"
