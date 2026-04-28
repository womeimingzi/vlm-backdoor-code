#!/usr/bin/env bash
# run_exp1c_qwen3vl_present_batch.sh — Serial exp1c Pseudo-Benign projection purification
# on Qwen3-VL-8B (present_exp checkpoints, k=10). Results appended to logs/exp1c.tsv.
#
# Usage:
#   cd /data/YBJ/cleansight
#   bash exps/exp1c_pseudo_benign/run_exp1c_qwen3vl_present_batch.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /data/YBJ/cleansight/venv_qwen3/bin/activate

EXP_SCRIPT="exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py"
CKPT_BASE="model_checkpoint/present_exp/qwen3-vl-8b/coco"
RESULT_BASE="exps/exp1c_pseudo_benign/checkpoints"
LOG_FILE="logs/exp1c.tsv"
SUMMARY_JSON="${RESULT_BASE}/batch_summary_qwen3vl_k10.json"

GPUS="3,4,5,6"
K=10
TEST_NUM=512
EVAL_BATCH_SIZE=8

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

mkdir -p logs

# ---------------------------------------------------------------------------
# Pre-validate: all checkpoint directories and required files must exist
# ---------------------------------------------------------------------------
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    if [ ! -d "$backdoor_dir" ]; then
        echo "ERROR: Checkpoint dir not found: ${backdoor_dir}"
        exit 1
    fi
    if [ ! -f "${backdoor_dir}/local.json" ]; then
        echo "ERROR: local.json not found in: ${backdoor_dir}"
        exit 1
    fi
    if [ ! -f "${backdoor_dir}/merger_state_dict.pth" ]; then
        echo "ERROR: merger_state_dict.pth not found in: ${backdoor_dir}"
        exit 1
    fi
done

# Validate ground truth benign
GT_DIR="${CKPT_BASE}/ground_truth_benign"
if [ ! -f "${GT_DIR}/merger_state_dict.pth" ]; then
    echo "ERROR: Ground truth benign not found at ${GT_DIR}"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All checkpoint directories validated."

# ---------------------------------------------------------------------------
# Summary JSON dump on exit
# ---------------------------------------------------------------------------
declare -A RUN_STATUS
dump_summary() {
    echo "["  > "$SUMMARY_JSON"
    local first=true
    for atk in "${ORDER[@]}"; do
        result_dir="${RESULT_BASE}/qwen3vl_${ATTACKS[$atk]}"
        eval_json="${result_dir}/exp1c_evaluation.json"
        status="${RUN_STATUS[$atk]:-pending}"
        if [ "$first" = true ]; then first=false; else echo "," >> "$SUMMARY_JSON"; fi
        if [ -f "$eval_json" ] && [ "$status" = "done" ]; then
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\", \"results\": $(cat "$eval_json")}" >> "$SUMMARY_JSON"
        else
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\"}" >> "$SUMMARY_JSON"
        fi
    done
    echo "]" >> "$SUMMARY_JSON"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary saved -> ${SUMMARY_JSON}"
}
trap dump_summary EXIT

# ---------------------------------------------------------------------------
# Helper: check if exp1c_evaluation.json already has k=K results
# ---------------------------------------------------------------------------
has_k_results() {
    local eval_json="$1"
    [ -f "$eval_json" ] && python3 -c "
import json, sys
d = json.load(open('${eval_json}'))
sys.exit(0 if d.get('config', {}).get('k') == ${K} else 1)
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Helper: back up existing results before overwriting
# ---------------------------------------------------------------------------
backup_existing() {
    local result_dir="$1"
    local eval_json="${result_dir}/exp1c_evaluation.json"
    local sim_json="${result_dir}/exp1c_direction_similarity.json"

    if [ -f "$eval_json" ]; then
        old_k=$(python3 -c "import json; print(json.load(open('${eval_json}')).get('config',{}).get('k','unknown'))" 2>/dev/null)
        if [ "$old_k" != "${K}" ] && [ "$old_k" != "unknown" ]; then
            cp "$eval_json" "${result_dir}/exp1c_evaluation_k${old_k}.json"
            echo "  Backed up existing k=${old_k} evaluation -> exp1c_evaluation_k${old_k}.json"
        fi
    fi
    if [ -f "$sim_json" ]; then
        old_k=$(python3 -c "
import json
d = json.load(open('${sim_json}'))
first = next(iter(d.values()), {})
# direction similarity doesn't store k directly; infer from eval json
" 2>/dev/null)
        # Use same k from eval json backup
        if [ -f "${result_dir}/exp1c_evaluation_k${old_k:-unknown}.json" ]; then
            cp "$sim_json" "${result_dir}/exp1c_direction_similarity_k${old_k}.json"
            echo "  Backed up existing k=${old_k} similarity -> exp1c_direction_similarity_k${old_k}.json"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Helper: extract metrics from evaluation JSON and append to exp1c.tsv
# ---------------------------------------------------------------------------
log_metrics() {
    local attack="$1"
    local pr="$2"
    local eval_json="$3"

    if [ -f "$eval_json" ]; then
        bd_asr=$(python3 -c "
import json
d = json.load(open('${eval_json}'))
e = d.get('evaluation', {})
print(e.get('backdoor_baseline', {}).get('backdoor_asr', ''))
")
        bn_asr=$(python3 -c "
import json
d = json.load(open('${eval_json}'))
e = d.get('evaluation', {})
v = e.get('d_true_k${K}', {}).get('backdoor_asr')
print(v if v is not None else '-')
")
        pseudo_bn_asr=$(python3 -c "
import json
d = json.load(open('${eval_json}'))
e = d.get('evaluation', {})
print(e.get('pseudo_n64', {}).get('backdoor_asr', ''))
")
        bd_cider=$(python3 -c "
import json
d = json.load(open('${eval_json}'))
e = d.get('evaluation', {})
print(e.get('backdoor_baseline', {}).get('clean_cider', ''))
")
        pseudo_cider=$(python3 -c "
import json
d = json.load(open('${eval_json}'))
e = d.get('evaluation', {})
print(e.get('pseudo_n64', {}).get('clean_cider', ''))
")
        echo -e "$(date '+%-m.%-d,%Y')\tqwen\t${attack}\t${pr}\t${K}\t${bd_asr}\t${bn_asr}\t${pseudo_bn_asr}\tcider:${bd_cider}->${pseudo_cider}" >> "$LOG_FILE"
    else
        echo -e "$(date '+%-m.%-d,%Y')\tqwen\t${attack}\t${pr}\t${K}\t\t\t\tresult json not found" >> "$LOG_FILE"
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    result_dir="${RESULT_BASE}/qwen3vl_${dir}"
    eval_json="${result_dir}/exp1c_evaluation.json"

    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running exp1c — ${attack} (${dir}, k=${K})"
    echo "========================================"

    # Skip if k=K results already exist
    if has_k_results "$eval_json"; then
        echo "  Result with k=${K} already exists: ${eval_json}, skipping run."
        RUN_STATUS[$attack]="done"
        if ! grep -q "qwen.*${attack}.*${K}" "$LOG_FILE" 2>/dev/null; then
            echo "  TSV entry missing, logging from existing result..."
            log_metrics "$attack" "$pr" "$eval_json"
        fi
        continue
    fi

    # Back up existing results with different k
    backup_existing "$result_dir"

    RUN_STATUS[$attack]="running"

    if CUDA_VISIBLE_DEVICES=${GPUS} python "$EXP_SCRIPT" \
        --backdoor_dir "$backdoor_dir" \
        --k ${K} \
        --test_num ${TEST_NUM} \
        --eval_batch_size ${EVAL_BATCH_SIZE}; then

        RUN_STATUS[$attack]="done"
        log_metrics "$attack" "$pr" "$eval_json"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}, logged to ${LOG_FILE}"
    else
        RUN_STATUS[$attack]="error"
        echo -e "$(date '+%-m.%-d,%Y')\tqwen\t${attack}\t${pr}\t${K}\t\t\t\tERROR" >> "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR running ${attack}, continuing..."
    fi
    echo ""
done

echo "All exp1c Qwen3-VL (k=${K}) runs complete."
