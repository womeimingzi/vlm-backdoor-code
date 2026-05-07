#!/usr/bin/env bash
# run_anp_qwen3vl_vqav2.sh — Serial ANP purification on Qwen3-VL-8B (VQAv2)
# 5 attacks (badnet/wanet/blended/trojvlm/issba).
#
# Pipeline per attack:
#   1. ANP purification (--no_eval --dataset vqav2)
#   2. Copy pruned weights + create local.json
#   3. Run qwen3vl_evaluator.py for AFTER evaluation
#
# Usage:
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
#   CUDA_VISIBLE_DEVICES=1,2 bash experiments/baseline_methods/anp/run_anp_qwen3vl_vqav2.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /data/YBJ/cleansight/venv_qwen3/bin/activate

EXP_SCRIPT="experiments/baseline_methods/anp/anp_purify_qwen3vl.py"
CKPT_BASE="/home/zzf/data/ZHC/model_checkpoint/qwen"
LOG_FILE="logs/vqav2_anp.tsv"

GPU="${CUDA_VISIBLE_DEVICES:-1,2}"
N_SAMPLE=128
ANP_BATCH_SIZE=4
N_ROUNDS=$((N_SAMPLE / ANP_BATCH_SIZE))  # 32 rounds = 1 epoch
TEST_NUM=512
EVAL_BS=8

MIN_FREE_MB=10000
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU memory (need ${MIN_FREE_MB}MiB free per card)..."
for g in $(echo "$GPU" | tr ',' ' '); do
    free=$(nvidia-smi --query-gpu=memory.free -i "$g" --format=csv,noheader,nounits | tr -d ' ')
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "ERROR: GPU $g has only ${free}MiB free, need at least ${MIN_FREE_MB}MiB." >&2
        exit 1
    fi
    echo "  GPU $g: ${free}MiB free — OK"
done

# ANP hyperparameters (same as COCO run)
EPS=0.012
PGD_STEPS=8
THETA_LR=0.06
LAM=0.006
CLEAN_LOSS_WEIGHT=2.5
PRUNE_THRESHOLD=0.5
LOG_INTERVAL=10

declare -A ATTACKS=(
    [badnet]="badnet/badnet"
    [wanet]="wanet/wanet"
    [blended]="blended/blended"
    [trojvlm]="trojvlm/trojvlm"
    [issba]="issba/issba"
)

declare -A ATTACK_PR=(
    [badnet]="0.1"
    [wanet]="0.1"
    [blended]="0.1"
    [trojvlm]="0.1"
    [issba]="0.15"
)

ORDER=(badnet wanet blended trojvlm issba)

# ---------------------------------------------------------------------------
# Pre-validate checkpoints
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pre-validating checkpoints..."
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    if [ ! -f "${backdoor_dir}/local.json" ]; then
        echo "ERROR: local.json not found in: ${backdoor_dir}" >&2; exit 1
    fi
    if [ ! -f "${backdoor_dir}/merger_state_dict.pth" ]; then
        echo "ERROR: merger_state_dict.pth not found in: ${backdoor_dir}" >&2; exit 1
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All checkpoints validated. GPU=${GPU}, EVAL_BS=${EVAL_BS}"

mkdir -p logs

# Create TSV header if needed
if [ ! -f "$LOG_FILE" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tn_rounds\tbd_asr\tafter_asr\tnote\n" > "$LOG_FILE"
fi

latest_eval_log() {
    local dir="$1"
    ls -t "$dir"/\[eval-vqav2-*attack_results.log 2>/dev/null | head -1 || true
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    out_dir="experiments/baseline_methods/anp/checkpoints/qwen3vl_vqav2_${attack}"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] ANP VQAv2 — ${attack}"
    echo "  backdoor_dir=${backdoor_dir}"
    echo "  out_dir=${out_dir}"
    echo "========================================"

    mkdir -p "$out_dir"

    # --- Step 1: ANP purification ---
    if [ -f "$out_dir/merger_pruned.pth" ]; then
        echo "  Pruned weights exist; skip purification."
    else
        echo "[$(date '+%F %T')] Running ANP purification..."
        CUDA_VISIBLE_DEVICES=${GPU} python -u "$EXP_SCRIPT" \
            --backdoor_dir "$backdoor_dir" \
            --dataset vqav2 \
            --n_sample ${N_SAMPLE} \
            --test_num ${TEST_NUM} \
            --eval_batch_size ${EVAL_BS} \
            --eps ${EPS} \
            --pgd_steps ${PGD_STEPS} \
            --theta_lr ${THETA_LR} \
            --lam ${LAM} \
            --clean_loss_weight ${CLEAN_LOSS_WEIGHT} \
            --n_rounds ${N_ROUNDS} \
            --prune_threshold ${PRUNE_THRESHOLD} \
            --log_interval ${LOG_INTERVAL} \
            --output_dir "$out_dir" \
            --no_eval

        if [ $? -ne 0 ]; then
            echo "[$(date '+%F %T')] ERROR in purification for ${attack}" >&2
            echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t${N_ROUNDS}\t\t\tERROR_purify" >> "$LOG_FILE"
            continue
        fi
    fi

    # --- Step 2: Prepare pruned model for evaluation ---
    cp "$out_dir/merger_pruned.pth" "$out_dir/merger_state_dict.pth"
    if [ -f "$out_dir/deepstack_merger_list_pruned.pth" ]; then
        cp "$out_dir/deepstack_merger_list_pruned.pth" "$out_dir/deepstack_merger_list_state_dict.pth"
    fi

    python3 - "$backdoor_dir" "$out_dir" <<'PY'
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

    # --- Step 3: AFTER evaluation ---
    if [ -z "$(latest_eval_log "$out_dir")" ]; then
        echo "[$(date '+%F %T')] AFTER evaluation..."
        CUDA_VISIBLE_DEVICES=${GPU} python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
            --local_json "$out_dir/local.json" \
            --test_num "$TEST_NUM" \
            --batch_size ${EVAL_BS}
    else
        echo "[$(date '+%F %T')] AFTER eval exists; skip."
    fi

    # --- Step 4: Log results ---
    python3 - "$out_dir" "$LOG_FILE" "$attack" "$pr" "$N_SAMPLE" "$N_ROUNDS" <<'PY'
import glob, re, sys, datetime, os
out_dir, tsv, attack, pr, n_sample, n_rounds = sys.argv[1:7]
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
note = f"rounds={n_rounds},VQA_after={after_vqa}"
line = [dt, "qwen3vl", attack, pr, n_sample, n_rounds, "", after_asr, note]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}"
    echo ""
done

echo "[$(date '+%F %T')] ANP VQAv2 Qwen3-VL runs complete."
