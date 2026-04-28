#!/usr/bin/env bash
# run_anp_qwen3vl_batch.sh — Serial ANP purification on Qwen3-VL-8B
# 5 attacks (badnet/wanet/blended/trojvlm/issba).
# Results saved per-attack under checkpoints/ and appended to logs/anp.tsv.
#
# 2 GPUs required: Qwen3-VL-8B (~16.3 GiB fp16) + ANP hooks (~1.2 GiB fp32) +
# computation graph (~2 GiB) exceeds single 3090 (24 GiB). device_map="auto"
# splits model across 2 cards; each needs ≥10 GiB free.
#
# Usage:
#   cd /home/zzf/data/ZHC/vlm-backdoor-code
#   bash exps/exp9_anp/run_anp_qwen3vl_batch.sh
#
#   # Override GPU pair:
#   CUDA_VISIBLE_DEVICES=0,5 bash exps/exp9_anp/run_anp_qwen3vl_batch.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /data/YBJ/cleansight/venv_qwen3/bin/activate

EXP_SCRIPT="exps/exp9_anp/anp_purify_qwen3vl.py"
CKPT_BASE="model_checkpoint/present_exp/qwen3-vl-8b/coco"
LOG_FILE="logs/anp.tsv"
SUMMARY_JSON="exps/exp9_anp/checkpoints/batch_summary_qwen3vl.json"

GPU="${CUDA_VISIBLE_DEVICES:-6,7}"
N_SAMPLE=500
TEST_NUM=512
EVAL_BS=2

# Qwen3-VL-8B needs ~16.3 GiB (fp16). With ANP hooks (~1.2 GiB) + activations,
# each GPU needs at least 10 GiB free. Validate before running.
MIN_FREE_MB=10000
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU memory (need ${MIN_FREE_MB}MiB free per card)..."
for g in $(echo "$GPU" | tr ',' ' '); do
    free=$(nvidia-smi --query-gpu=memory.free -i "$g" --format=csv,noheader,nounits | tr -d ' ')
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "ERROR: GPU $g has only ${free}MiB free, need at least ${MIN_FREE_MB}MiB." >&2
        echo "  Other processes may be using this GPU. Choose different cards via:" >&2
        echo "  CUDA_VISIBLE_DEVICES=<gpu1>,<gpu2> bash $0" >&2
        exit 1
    fi
    echo "  GPU $g: ${free}MiB free — OK"
done

# ANP hyperparameters (same as run_anp_llava_512.sh)
EPS=0.012
PGD_STEPS=8
THETA_LR=0.06
LAM=0.006
CLEAN_LOSS_WEIGHT=2.5
N_ROUNDS=1250
PRUNE_THRESHOLD=0.5
LOG_INTERVAL=50

declare -A ATTACKS=(
    [badnet]="random-adapter-qwen3_badnet_pr0.1"
    [wanet]="warped-adapter-wanet_pr0.1"
    [blended]="blended_kt-adapter-blended_kt_pr0.1"
    [trojvlm]="random-adapter-trojvlm_randomins_e1"
    [issba]="issba-adapter-qwen_issba0.15"
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
# Pre-validate: all checkpoint directories, local.json, and weights must exist
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pre-validating checkpoints..."
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    if [ ! -d "$backdoor_dir" ]; then
        echo "ERROR: Checkpoint dir not found: ${backdoor_dir}" >&2; exit 1
    fi
    if [ ! -f "${backdoor_dir}/local.json" ]; then
        echo "ERROR: local.json not found in: ${backdoor_dir}" >&2; exit 1
    fi
    if [ ! -f "${backdoor_dir}/merger_state_dict.pth" ]; then
        echo "ERROR: merger_state_dict.pth not found in: ${backdoor_dir}" >&2; exit 1
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All checkpoints validated. GPU=${GPU}, EVAL_BS=${EVAL_BS}"

mkdir -p logs
mkdir -p "$(dirname "$SUMMARY_JSON")"

# ---------------------------------------------------------------------------
# Accumulate per-attack results; dump to summary JSON on exit
# ---------------------------------------------------------------------------
declare -A RUN_STATUS
dump_summary() {
    echo "["  > "$SUMMARY_JSON"
    local first=true
    for atk in "${ORDER[@]}"; do
        result_json="exps/exp9_anp/checkpoints/qwen3vl_${ATTACKS[$atk]}/anp_purify_results.json"
        status="${RUN_STATUS[$atk]:-pending}"
        if [ "$first" = true ]; then first=false; else echo "," >> "$SUMMARY_JSON"; fi
        if [ -f "$result_json" ] && [ "$status" = "done" ]; then
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\", \"results\": $(cat "$result_json")}" >> "$SUMMARY_JSON"
        else
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\"}" >> "$SUMMARY_JSON"
        fi
    done
    echo "]" >> "$SUMMARY_JSON"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary saved -> ${SUMMARY_JSON}"
}
trap dump_summary EXIT

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    result_json="exps/exp9_anp/checkpoints/qwen3vl_${dir}/anp_purify_results.json"

    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running exp9 ANP — ${attack} (${dir})"
    echo "========================================"

    if [ -f "$result_json" ]; then
        echo "  Result already exists: ${result_json}, skipping."
        RUN_STATUS[$attack]="skipped"
        continue
    fi

    RUN_STATUS[$attack]="running"

    if CUDA_VISIBLE_DEVICES=${GPU} python "$EXP_SCRIPT" \
        --backdoor_dir "$backdoor_dir" \
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
        --log_interval ${LOG_INTERVAL}; then

        RUN_STATUS[$attack]="done"

        if [ -f "$result_json" ]; then
            # evaluate_qwen3vl_adapter returns ASR as percentage (0-100).
            # Existing TSV entries use fraction (0-1). Convert for consistency.
            read -r bd_asr after_asr bd_cider after_cider <<< "$(python3 -c "
import json
d = json.load(open('${result_json}'))
b = d.get('before') or {}
a = d.get('after') or {}
bd  = b.get('backdoor_asr', 0) / 100.0
aft = a.get('backdoor_asr', 0) / 100.0
bc  = b.get('clean_cider', '')
ac  = a.get('clean_cider', '')
print(f'{bd:.4f}', f'{aft:.4f}', bc, ac)
")"
            echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t${N_ROUNDS}\t${bd_asr}\t${after_asr}\tcider:${bd_cider}->${after_cider}" >> "$LOG_FILE"
        else
            echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t${N_ROUNDS}\t\t\tresult json not found" >> "$LOG_FILE"
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}, logged to ${LOG_FILE}"
    else
        RUN_STATUS[$attack]="error"
        echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t${N_ROUNDS}\t\t\tERROR" >> "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR running ${attack}, continuing..."
    fi
    echo ""
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All exp9 ANP Qwen3-VL runs complete."
