#!/usr/bin/env bash
# run_exp7_llava_batch.sh — Serial Fine-tuning Recovery on LLaVA-7B
# 5 attacks (badnet/wanet/blended_kt/trojvlm/vlood), excluding issba.
# Results saved per-attack under checkpoints/ and appended to logs/ft.tsv.
#
# GPU: single card (device_map="auto"), eval_batch_size=2 to avoid OOM.
#
# Usage:
#   cd /data/YBJ/cleansight
#   bash exps/exp7_finetune_recovery/run_exp7_llava_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

EXP_SCRIPT="exps/exp7_finetune_recovery/exp7_finetune_recovery.py"
CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"
RESULT_JSON="exps/exp7_finetune_recovery/exp7_results.json"
LOG_FILE="logs/ft.tsv"

GPU="${CUDA_VISIBLE_DEVICES:-2}"
N_SAMPLE=1000
TEST_NUM=512
EVAL_BS=2

declare -A ATTACKS=(
    [badnet]="random-adapter-badnet_pr0.1"
    [wanet]="warped-adapter-wanet_pr0.1"
    [blended_kt]="blended_kt-adapter-blended_kt_pr0.1"
    [trojvlm]="random-adapter-trojvlm_randomins_e1"
    [vlood]="random-adapter-vlood_randomins_pr0.1"
)

declare -A ATTACK_PR=(
    [badnet]="0.1"
    [wanet]="0.1"
    [blended_kt]="0.1"
    [trojvlm]="0.1"
    [vlood]="0.1"
)

ORDER=(badnet wanet blended_kt trojvlm vlood)

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
    if [ ! -f "${backdoor_dir}/mmprojector_state_dict.pth" ]; then
        echo "ERROR: mmprojector_state_dict.pth not found in: ${backdoor_dir}" >&2; exit 1
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All checkpoints validated. GPU=${GPU}, EVAL_BS=${EVAL_BS}"

mkdir -p logs

# ---------------------------------------------------------------------------
# Helper: extract metrics from result JSON and append to ft.tsv
# ---------------------------------------------------------------------------
log_metrics() {
    local attack="$1"
    local pr="$2"
    local json_path="$3"

    read -r bd_asr after_asr bd_cider after_cider <<< "$(python3 -c "
import json, sys
d = json.load(open('${json_path}'))
r = d['results']
pb = r['P_b']
fn = r.get('n${N_SAMPLE}', {})
print(pb.get('backdoor_asr',''), fn.get('backdoor_asr',''), pb.get('clean_cider',''), fn.get('clean_cider',''))
")"

    echo -e "$(date '+%-m.%-d,%Y')\tllava\t${attack}\t${pr}\t${N_SAMPLE}\t${bd_asr}\t${after_asr}\tcider:${bd_cider}->${after_cider}" >> "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    backdoor_dir="${CKPT_BASE}/${dir}"
    backup_dir="exps/exp7_finetune_recovery/checkpoints/llava_${dir}"
    backup_json="${backup_dir}/exp7_results.json"

    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running exp7 — ${attack} (${dir})"
    echo "========================================"

    # Skip if per-attack backup already exists (no overwrite)
    if [ -f "$backup_json" ]; then
        echo "  Result already exists: ${backup_json}, skipping."
        continue
    fi

    mkdir -p "$backup_dir"

    CUDA_VISIBLE_DEVICES=${GPU} python "$EXP_SCRIPT" \
        --backdoor_dir "$backdoor_dir" \
        --n_sample_list ${N_SAMPLE} \
        --test_num ${TEST_NUM} \
        --eval_batch_size ${EVAL_BS}

    # Backup result JSON to per-attack directory
    if [ -f "$RESULT_JSON" ]; then
        cp "$RESULT_JSON" "$backup_json"
        echo "  Saved result -> ${backup_json}"
        log_metrics "$attack" "$pr" "$backup_json"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}, logged to ${LOG_FILE}"
    else
        echo "ERROR: ${RESULT_JSON} not found after run for ${attack}" >&2
        exit 1
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All exp7 LLaVA runs complete."
