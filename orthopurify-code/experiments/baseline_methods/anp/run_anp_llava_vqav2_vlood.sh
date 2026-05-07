#!/usr/bin/env bash
# Serial exp9 ANP defense for the VQAv2 LLaVA VLOOD backdoor.
# Required setting: n_sample=500, n_rounds=1000 (about 2 epochs).

set -euo pipefail

GPU="${GPU:-4}"
BACKDOOR_DIR="${BACKDOOR_DIR:-model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-vlood_randomins_pr0.2_lambda4}"
OUT_DIR="${OUT_DIR:-experiments/baseline_methods/anp/checkpoints/llava_vqav2_vlood_randomins_pr0.2_lambda4}"
TEST_NUM="${TEST_NUM:-512}"
N_SAMPLE="${N_SAMPLE:-500}"
N_ROUNDS="${N_ROUNDS:-1000}"
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
mkdir -p "$LOG_DIR" "$OUT_DIR"
MASTER_LOG="$LOG_DIR/exp9_anp_vqav2_vlood_${STAMP}.log"
ANP_LOG="$LOG_DIR/exp9_anp_vqav2_vlood_purify_${STAMP}.log"
AFTER_LOG="$LOG_DIR/exp9_anp_vqav2_vlood_after_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_anp.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tepoch\tbd_asr\tafter_asr\tnote\n" > "$TSV"
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

latest_eval_log() {
    local dir="$1"
    ls -t "$dir"/\[eval-vqav2-*attack_results.log 2>/dev/null | head -1 || true
}

{
    echo "[$(date '+%F %T')] ANP VQAv2 VLOOD start"
    echo "GPU=$GPU"
    echo "BACKDOOR_DIR=$BACKDOOR_DIR"
    echo "OUT_DIR=$OUT_DIR"
    echo "N_SAMPLE=$N_SAMPLE"
    echo "N_ROUNDS=$N_ROUNDS"

    wait_for_backdoor

    echo "[$(date '+%F %T')] Skipping BEFORE baseline evaluation."

    if [ -f "$OUT_DIR/mmprojector_pruned.pth" ]; then
        echo "[$(date '+%F %T')] ANP pruned projector exists; skip purification."
    else
        echo "[$(date '+%F %T')] Running ANP purification..."
        wait_for_gpu
        CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/baseline_methods/anp/anp_purify_llava.py \
            --poison_local_json "$BACKDOOR_DIR/local.json" \
            --dataset vqav2 \
            --test_num "$N_SAMPLE" \
            --asr_num "$N_SAMPLE" \
            --batch_size "${BATCH_SIZE:-1}" \
            --num_workers 0 \
            --device cuda:0 \
            --fp16 \
            --no_eval \
            --eps "${EPS:-0.012}" \
            --pgd_steps "${PGD_STEPS:-8}" \
            --theta_lr "${THETA_LR:-0.06}" \
            --lam "${LAM:-0.006}" \
            --clean_loss_weight "${CLEAN_LOSS_WEIGHT:-2.5}" \
            --n_rounds "$N_ROUNDS" \
            --prune_threshold "${PRUNE_THRESHOLD:-0.5}" \
            --log_interval "${LOG_INTERVAL:-50}" \
            --output_dir "$OUT_DIR" \
            > "$ANP_LOG" 2>&1
    fi

    cp "$OUT_DIR/mmprojector_pruned.pth" "$OUT_DIR/mmprojector_state_dict.pth"
    python - "$BACKDOOR_DIR" "$OUT_DIR" <<'PY'
import json, sys
base_dir, out_dir = sys.argv[1], sys.argv[2]
with open(base_dir + "/local.json") as f:
    cfg = json.load(f)
cfg["adapter_path"] = out_dir
cfg["output_dir_root_name"] = out_dir
with open(out_dir + "/local.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("wrote", out_dir + "/local.json")
PY

    if [ -z "$(latest_eval_log "$OUT_DIR")" ]; then
        echo "[$(date '+%F %T')] AFTER evaluation..."
        wait_for_gpu
        CUDA_VISIBLE_DEVICES="$GPU" python vlm_backdoor/evaluation/llava_evaluator.py \
            --local_json "$OUT_DIR/local.json" \
            --test_num "$TEST_NUM" \
            --batch_size "${EVAL_BATCH_SIZE:-4}" \
            > "$AFTER_LOG" 2>&1
    else
        echo "[$(date '+%F %T')] AFTER eval exists; skip."
    fi

    python - "$OUT_DIR" "$TSV" <<'PY'
import glob, re, sys, datetime, os
out_dir, tsv = sys.argv[1], sys.argv[2]
def parse(path):
    if not path:
        return "", ""
    with open(path, errors="ignore") as f:
        lines = [l.strip() for l in f if "BACKDOOR ASR" in l]
    line = lines[-1] if lines else ""
    asr = re.search(r"BACKDOOR ASR:\s*([0-9.]+)", line)
    vqa = re.search(r"VQA SCORE:\s*([0-9.]+)", line)
    return (asr.group(1) if asr else "", vqa.group(1) if vqa else "")
def latest(d):
    files = glob.glob(os.path.join(d, "[eval-vqav2-*attack_results.log"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else ""
after_asr, after_vqa = parse(latest(out_dir))
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
note = f"rounds=1000,VQA_after={after_vqa}"
line = [dt, "llava", "vlood", "0.2", "500", "2", "", after_asr, note]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY

    echo "[$(date '+%F %T')] ANP VQAv2 VLOOD done"
} 2>&1 | tee -a "$MASTER_LOG"
