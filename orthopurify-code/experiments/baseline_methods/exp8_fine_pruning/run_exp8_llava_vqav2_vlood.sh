#!/usr/bin/env bash
# Serial exp8 fine-pruning defense for the VQAv2 LLaVA VLOOD backdoor.
# Required setting: n_sample=1000, fine-tuning epochs=2.

set -euo pipefail

GPU="${GPU:-3}"
BACKDOOR_DIR="${BACKDOOR_DIR:-model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-vlood_randomins_pr0.2_lambda4}"
OUT_DIR="${OUT_DIR:-experiments/baseline_methods/exp8_fine_pruning/checkpoints/llava_vqav2_vlood_randomins_pr0.2_lambda4}"
TEST_NUM="${TEST_NUM:-512}"
WAIT_INTERVAL="${WAIT_INTERVAL:-300}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-172800}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
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
MASTER_LOG="$LOG_DIR/exp8_vqav2_vlood_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_fp.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tbd_asr\tafter_asr\tnote\n" > "$TSV"
fi

wait_for_backdoor() {
    local waited=0
    while [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; do
        if [ "$waited" -ge "$MAX_WAIT_SEC" ]; then
            echo "Timed out waiting for backdoor checkpoint: $BACKDOOR_DIR" >&2
            return 1
        fi
        echo "[$(date '+%F %T')] Waiting for backdoor checkpoint: $BACKDOOR_DIR"
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
    echo "[$(date '+%F %T')] exp8 VQAv2 VLOOD start"
    echo "GPU=$GPU"
    echo "BACKDOOR_DIR=$BACKDOOR_DIR"
    echo "OUT_DIR=$OUT_DIR"

    wait_for_backdoor

    if [ -f "$OUT_DIR/exp8_results.json" ]; then
        echo "[$(date '+%F %T')] Result exists; skip exp8: $OUT_DIR/exp8_results.json"
    else
        wait_for_gpu
        CUDA_VISIBLE_DEVICES="$GPU" python experiments/baseline_methods/exp8_fine_pruning/exp8_fine_pruning.py \
            --backdoor_dir "$BACKDOOR_DIR" \
            --n_sample 1000 \
            --test_num "$TEST_NUM" \
            --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
            --cider_threshold "${CIDER_THRESHOLD:-0.025}" \
            --max_ratio "${MAX_RATIO:-0.95}" \
            --search_step "${SEARCH_STEP:-0.10}" \
            --output_dir "$OUT_DIR" \
            --skip_baseline_eval
    fi

    python - "$OUT_DIR/exp8_results.json" "$TSV" <<'PY'
import json, sys, datetime
path, tsv = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
r = d.get("results", {})
base = r.get("backdoor_baseline", {})
after = r.get("fine_pruning", {})
cfg = d.get("config", {})
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
metric = after.get("metric_name", base.get("metric_name", ""))
note = f"prune={cfg.get('prune_ratio','')},{metric}_after={after.get('clean_cider','')},epochs=2,baseline_skipped"
line = [dt, "llava", "vlood", "0.2", "1000",
        str(base.get("backdoor_asr", "")), str(after.get("backdoor_asr", "")), note]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY

    echo "[$(date '+%F %T')] exp8 VQAv2 VLOOD done"
} 2>&1 | tee -a "$MASTER_LOG"
