#!/usr/bin/env bash
# run_exp7_qwen3vl_batch.sh — Serial Fine-tuning Recovery defense on Qwen3-VL-8B
# across 4 attack types. Results appended to logs/ft.tsv.
#
# Usage:
#   cd /data/YBJ/cleansight
#   bash experiments/baseline_methods/exp7_finetune_recovery/run_exp7_qwen3vl_batch.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /data/YBJ/cleansight/venv_qwen3/bin/activate

EXP_SCRIPT="experiments/baseline_methods/exp7_finetune_recovery/exp7_finetune_recovery_qwen3vl.py"
CKPT_BASE="model_checkpoint/present_exp/qwen3-vl-8b/coco"
RESULT_JSON="experiments/baseline_methods/exp7_finetune_recovery/exp7_results_qwen3vl.json"
LOG_FILE="logs/ft.tsv"
SUMMARY_JSON="experiments/baseline_methods/exp7_finetune_recovery/checkpoints/batch_summary_qwen3vl.json"

GPUS="${CUDA_VISIBLE_DEVICES:-5,6}"
N_SAMPLE=1000
TEST_NUM=512
EVAL_BS="${EVAL_BS:-8}"

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
mkdir -p "$(dirname "$SUMMARY_JSON")"

# ---------------------------------------------------------------------------
# Pre-validate: all checkpoint directories and local.json must exist
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All checkpoint directories validated."

# ---------------------------------------------------------------------------
# Summary JSON dump on exit (normal or error)
# ---------------------------------------------------------------------------
declare -A RUN_STATUS
dump_summary() {
    echo "["  > "$SUMMARY_JSON"
    local first=true
    for atk in "${ORDER[@]}"; do
        backup_json="experiments/baseline_methods/exp7_finetune_recovery/checkpoints/qwen3vl_${ATTACKS[$atk]}/exp7_results.json"
        status="${RUN_STATUS[$atk]:-pending}"
        if [ "$first" = true ]; then first=false; else echo "," >> "$SUMMARY_JSON"; fi
        if [ -f "$backup_json" ] && [ "$status" = "done" ]; then
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\", \"results\": $(cat "$backup_json")}" >> "$SUMMARY_JSON"
        else
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\"}" >> "$SUMMARY_JSON"
        fi
    done
    echo "]" >> "$SUMMARY_JSON"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary saved -> ${SUMMARY_JSON}"
}
trap dump_summary EXIT

# ---------------------------------------------------------------------------
# Helper: extract metrics from backup JSON and append to ft.tsv
# ---------------------------------------------------------------------------
log_metrics() {
    local attack="$1"
    local pr="$2"
    local backup_json="$3"

    if [ -f "$backup_json" ]; then
        bd_asr=$(python3 -c "
import json
d = json.load(open('${backup_json}'))
r = d.get('results', {})
print(r.get('P_b', {}).get('backdoor_asr', ''))
")
        after_asr=$(python3 -c "
import json
d = json.load(open('${backup_json}'))
r = d.get('results', {})
print(r.get('n${N_SAMPLE}', {}).get('backdoor_asr', ''))
")
        bd_cider=$(python3 -c "
import json
d = json.load(open('${backup_json}'))
r = d.get('results', {})
print(r.get('P_b', {}).get('clean_cider', ''))
")
        after_cider=$(python3 -c "
import json
d = json.load(open('${backup_json}'))
r = d.get('results', {})
print(r.get('n${N_SAMPLE}', {}).get('clean_cider', ''))
")
        echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t${bd_asr}\t${after_asr}\tcider:${bd_cider}->${after_cider}" >> "$LOG_FILE"
    else
        echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t\t\tresult json not found" >> "$LOG_FILE"
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    backup_dir="experiments/baseline_methods/exp7_finetune_recovery/checkpoints/qwen3vl_${dir}"
    backup_json="${backup_dir}/exp7_results.json"

    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running exp7 — ${attack} (${dir})"
    echo "========================================"

    # Skip if per-attack backup already exists
    if [ -f "$backup_json" ]; then
        echo "  Result already exists: ${backup_json}, skipping run."
        RUN_STATUS[$attack]="done"
        if ! grep -q "qwen3vl.*${attack}" "$LOG_FILE" 2>/dev/null; then
            echo "  TSV entry missing, logging from existing result..."
            log_metrics "$attack" "$pr" "$backup_json"
        fi
        continue
    fi

    RUN_STATUS[$attack]="running"
    mkdir -p "$backup_dir"

    # Keep GPU 7 free for concurrent Qwen3 exp1c jobs. With 2-GPU device_map,
    # bs=8 is the conservative default on 24GB 3090 cards.
    if CUDA_VISIBLE_DEVICES=${GPUS} python "$EXP_SCRIPT" \
        --backdoor_dir "$backdoor_dir" \
        --n_sample_list ${N_SAMPLE} \
        --test_num ${TEST_NUM} \
        --eval_batch_size ${EVAL_BS}; then

        RUN_STATUS[$attack]="done"

        # Backup volatile result to per-attack directory
        if [ -f "$RESULT_JSON" ]; then
            cp "$RESULT_JSON" "$backup_json"
            echo "  Backed up result -> ${backup_json}"
        fi

        plot_src="experiments/baseline_methods/exp7_finetune_recovery/exp7_plot_qwen3vl.png"
        if [ -f "$plot_src" ]; then
            cp "$plot_src" "${backup_dir}/exp7_plot.png"
        fi

        log_metrics "$attack" "$pr" "$backup_json"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}, logged to ${LOG_FILE}"
    else
        RUN_STATUS[$attack]="error"
        echo -e "$(date '+%-m.%-d,%Y')\tqwen3vl\t${attack}\t${pr}\t${N_SAMPLE}\t\t\tERROR" >> "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR running ${attack}, continuing..."
    fi
    echo ""
done

echo "All exp7 Qwen3-VL runs complete."
