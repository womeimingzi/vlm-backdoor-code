#!/usr/bin/env bash
# run_exp8_llava_batch.sh — Serial Fine-Pruning defense on LLaVA-1.5-7B
# across 6 attack types. Results appended to logs/fp.tsv.
#
# Usage:
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
#   source /data/YBJ/GraduProject/venv/bin/activate
#   bash experiments/baseline_methods/exp8_fine_pruning/run_exp8_llava_batch.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

EXP_SCRIPT="experiments/baseline_methods/exp8_fine_pruning/exp8_fine_pruning.py"
CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"
LOG_FILE="logs/fp.tsv"
SUMMARY_JSON="experiments/baseline_methods/exp8_fine_pruning/checkpoints/batch_summary.json"

CIDER_THRESHOLD=0.025
MAX_RATIO=0.95
SEARCH_STEP=0.10

mkdir -p logs
mkdir -p "$(dirname "$SUMMARY_JSON")"


    # [badnet]="random-adapter-badnet_pr0.1"
    # [wanet]="warped-adapter-wanet_pr0.1"
    #[blended]="blended_kt-adapter-blended_kt_pr0.1"
    #[trojvlm]="random-adapter-trojvlm_randomins_e1"
    #[issba]="issba-adapter-issba_pr0.15_e2"
    #[vlood]="random-adapter-vlood_randomins_pr0.1"
declare -A ATTACKS=(
    [badnet]="random-adapter-badnet_pr0.1"
    [blended]="blended_kt-adapter-blended_kt_pr0.1"
    [trojvlm]="random-adapter-trojvlm_randomins_e1"
    [issba]="issba-adapter-issba_pr0.15_e2"
    [vlood]="random-adapter-vlood_randomins_pr0.1"
)

# badnet wanet blended trojvlm issba vlood
ORDER=(badnet blended trojvlm issba vlood)

# Accumulate per-attack results; dump to summary JSON on exit (normal or error)
declare -A RUN_STATUS
dump_summary() {
    echo "["  > "$SUMMARY_JSON"
    local first=true
    for atk in "${ORDER[@]}"; do
        result_json="experiments/baseline_methods/exp8_fine_pruning/checkpoints/llava_${ATTACKS[$atk]}/exp8_results.json"
        status="${RUN_STATUS[$atk]:-pending}"
        if [ "$first" = true ]; then first=false; else echo "," >> "$SUMMARY_JSON"; fi
        if [ -f "$result_json" ] && [ "$status" = "done" ]; then
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\", \"results\": $(cat "$result_json")}" >> "$SUMMARY_JSON"
        else
            echo "  {\"attack\": \"${atk}\", \"status\": \"${status}\"}" >> "$SUMMARY_JSON"
        fi
    done
    echo "]" >> "$SUMMARY_JSON"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary saved → ${SUMMARY_JSON}"
}
trap dump_summary EXIT

for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    result_json="experiments/baseline_methods/exp8_fine_pruning/checkpoints/llava_${dir}/exp8_results.json"

    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running exp8 — ${attack} (${dir})"
    echo "========================================"

    RUN_STATUS[$attack]="running"

    if CUDA_VISIBLE_DEVICES=2,4,5,6 python "$EXP_SCRIPT" \
        --backdoor_dir "$backdoor_dir" \
        --n_sample 1000 \
        --test_num 512 \
        --cider_threshold "$CIDER_THRESHOLD" \
        --max_ratio "$MAX_RATIO" \
        --search_step "$SEARCH_STEP"; then

        RUN_STATUS[$attack]="done"

        # Extract metrics from result JSON and append to logs/fp.tsv
        if [ -f "$result_json" ]; then
            bd_asr=$(python3 -c "
import json, sys
d = json.load(open('${result_json}'))
r = d.get('results', {})
bl = r.get('backdoor_baseline', {})
print(bl.get('backdoor_asr', ''))
")
            after_asr=$(python3 -c "
import json, sys
d = json.load(open('${result_json}'))
r = d.get('results', {})
fp = r.get('fine_pruning', {})
print(fp.get('backdoor_asr', ''))
")
            bd_cider=$(python3 -c "
import json, sys
d = json.load(open('${result_json}'))
r = d.get('results', {})
bl = r.get('backdoor_baseline', {})
print(bl.get('clean_cider', ''))
")
            after_cider=$(python3 -c "
import json, sys
d = json.load(open('${result_json}'))
r = d.get('results', {})
fp = r.get('fine_pruning', {})
print(fp.get('clean_cider', ''))
")
            prune_ratio=$(python3 -c "
import json
d = json.load(open('${result_json}'))
print(d.get('config', {}).get('prune_ratio', ''))
")
            echo -e "$(date '+%-m.%-d,%Y')\tllava\t${attack}\t\t1000\t${bd_asr}\t${after_asr}\tprune=${prune_ratio},cider:${bd_cider}->${after_cider}" >> "$LOG_FILE"
        else
            echo -e "$(date '+%-m.%-d,%Y')\tllava\t${attack}\t\t1000\t\t\tresult json not found" >> "$LOG_FILE"
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}, logged to ${LOG_FILE}"
    else
        RUN_STATUS[$attack]="error"
        echo -e "$(date '+%-m.%-d,%Y')\tllava\t${attack}\t\t1000\t\t\tERROR" >> "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR running ${attack}, continuing..."
    fi
    echo ""
done

echo "All exp8 LLaVA runs complete."
