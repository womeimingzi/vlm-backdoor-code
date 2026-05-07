#!/usr/bin/env bash
# Dual-GPU, serial exp1c runner for Qwen3-VL-8B present_exp checkpoints.
#
# It reads checkpoint directories from:
#   /data/YBJ/cleansight/model_checkpoint/present_exp/qwen3-vl-8b/coco
#
# Required attacks:
#   wanet, blended_kt, issba, trojvlm
#
# Usage:
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
#   CUDA_VISIBLE_DEVICES=5,6 bash experiments/main_method/orthopurify/run_exp1c_qwen3vl_present_batch.sh
#
# Optional env overrides:
#   K=10 TEST_NUM=512 EVAL_BATCH_SIZE=4 MIN_FREE_MB=20000
#   GPU_WAIT_SECONDS=0 CHECK_INTERVAL_SECONDS=60 MASTER_PORT=29503
#   SKIP_EXISTING=1 DRY_RUN=1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

VENV_ACTIVATE="/data/YBJ/cleansight/venv_qwen3/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: Qwen3 environment not found: ${VENV_ACTIVATE}" >&2
    exit 1
fi
source "$VENV_ACTIVATE"

EXP_SCRIPT="experiments/main_method/orthopurify/purify_qwen3vl.py"
CKPT_BASE="/data/YBJ/cleansight/model_checkpoint/present_exp/qwen3-vl-8b/coco"
RESULT_BASE="experiments/main_method/orthopurify/checkpoints"
LOG_FILE="logs/exp1c.tsv"
SUMMARY_JSON="${RESULT_BASE}/batch_summary_qwen3vl_dual_serial.json"

GPU="${CUDA_VISIBLE_DEVICES:-5,6}"
N_GPU="$(echo "$GPU" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
K="${K:-10}"
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MASTER_PORT="${MASTER_PORT:-29503}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-0}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "${MIN_FREE_MB:-}" ]; then
    if [ "$EVAL_BATCH_SIZE" -le 4 ]; then
        MIN_FREE_MB=20000
    else
        MIN_FREE_MB=$((20000 + (EVAL_BATCH_SIZE - 4) * 1200))
    fi
fi

ORDER=(wanet blended_kt issba trojvlm)

declare -A ATTACK_PATTERNS=(
    [wanet]='wanet.*0\.1pr$'
    [blended_kt]='blended_kt.*0\.1pr$'
    [issba]='issba.*0\.1pr$'
    [trojvlm]='trojvlm.*0\.1pr$'
)
declare -A ATTACK_DIRS
declare -A ATTACK_PR
declare -A RUN_STATUS

mkdir -p logs
mkdir -p "$RESULT_BASE"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_dual_gpu() {
    if [ "$N_GPU" -ne 2 ]; then
        die "This script is dual-GPU only. Got CUDA_VISIBLE_DEVICES=${GPU} (${N_GPU} GPU(s)); use CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b>."
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found; cannot check GPU memory safely."
    fi
    if ! command -v torchrun >/dev/null 2>&1; then
        die "torchrun not found in the active environment."
    fi
}

print_gpu_snapshot() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU snapshot:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
}

check_gpu_memory_once() {
    local ok=0
    local g free total

    for g in $(echo "$GPU" | tr ',' ' '); do
        free="$(nvidia-smi --query-gpu=memory.free -i "$g" --format=csv,noheader,nounits | tr -d ' ')"
        total="$(nvidia-smi --query-gpu=memory.total -i "$g" --format=csv,noheader,nounits | tr -d ' ')"

        if [ -z "$free" ] || [ -z "$total" ]; then
            echo "  GPU ${g}: unable to query memory." >&2
            ok=1
            continue
        fi

        if [ "$free" -lt "$MIN_FREE_MB" ]; then
            echo "  GPU ${g}: ${free}/${total} MiB free, need >= ${MIN_FREE_MB} MiB." >&2
            ok=1
        else
            echo "  GPU ${g}: ${free}/${total} MiB free, OK."
        fi
    done

    return "$ok"
}

wait_for_gpu_memory() {
    local start now elapsed remaining sleep_for
    start="$(date +%s)"

    while true; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking selected GPU memory (need ${MIN_FREE_MB} MiB free per card)..."
        if check_gpu_memory_once; then
            return 0
        fi

        if [ "$GPU_WAIT_SECONDS" -le 0 ]; then
            print_gpu_snapshot
            return 1
        fi

        now="$(date +%s)"
        elapsed=$((now - start))
        if [ "$elapsed" -ge "$GPU_WAIT_SECONDS" ]; then
            print_gpu_snapshot
            return 1
        fi

        remaining=$((GPU_WAIT_SECONDS - elapsed))
        sleep_for="$CHECK_INTERVAL_SECONDS"
        if [ "$sleep_for" -gt "$remaining" ]; then
            sleep_for="$remaining"
        fi
        echo "  Waiting ${sleep_for}s for GPU memory to be released..."
        sleep "$sleep_for"
    done
}

discover_checkpoints() {
    [ -d "$CKPT_BASE" ] || die "Checkpoint root not found: ${CKPT_BASE}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Discovering required checkpoints under ${CKPT_BASE}"
    for attack in "${ORDER[@]}"; do
        local pattern="${ATTACK_PATTERNS[$attack]}"
        local matches=()
        local name

        while IFS= read -r name; do
            if [[ "$name" =~ $pattern ]]; then
                matches+=("$name")
            fi
        done < <(find -L "$CKPT_BASE" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)

        if [ "${#matches[@]}" -eq 0 ]; then
            die "Missing checkpoint for attack=${attack}; expected a directory matching /${pattern}/ under ${CKPT_BASE}"
        fi
        if [ "${#matches[@]}" -gt 1 ]; then
            printf 'ERROR: Ambiguous checkpoint for attack=%s under %s:\n' "$attack" "$CKPT_BASE" >&2
            printf '  %s\n' "${matches[@]}" >&2
            exit 1
        fi

        local ckpt_dir="${CKPT_BASE}/${matches[0]}"
        [ -f "${ckpt_dir}/local.json" ] || die "local.json not found in ${ckpt_dir}"
        [ -f "${ckpt_dir}/merger_state_dict.pth" ] || die "merger_state_dict.pth not found in ${ckpt_dir}"
        [ -f "${ckpt_dir}/deepstack_merger_list_state_dict.pth" ] || die "deepstack_merger_list_state_dict.pth not found in ${ckpt_dir}"

        ATTACK_DIRS[$attack]="$ckpt_dir"
        ATTACK_PR[$attack]="$(python3 - "$ckpt_dir/local.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)
print(cfg.get("pr", ""))
PY
)"
        echo "  ${attack}: ${ckpt_dir} (pr=${ATTACK_PR[$attack]})"
    done
}

has_k_results() {
    local eval_json="$1"
    [ -f "$eval_json" ] || return 1
    python3 - "$eval_json" "$K" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
expected_k = int(sys.argv[2])
actual_k = data.get("config", {}).get("k")
sys.exit(0 if actual_k == expected_k else 1)
PY
}

backup_existing_results() {
    local result_dir="$1"
    local ts
    ts="$(date '+%Y%m%d_%H%M%S')"

    if [ -f "${result_dir}/evaluation.json" ]; then
        cp "${result_dir}/evaluation.json" "${result_dir}/evaluation.before_${ts}.json"
        echo "  Backed up existing evaluation.json"
    fi
    if [ -f "${result_dir}/direction_similarity.json" ]; then
        cp "${result_dir}/direction_similarity.json" "${result_dir}/direction_similarity.before_${ts}.json"
        echo "  Backed up existing direction_similarity.json"
    fi
}

append_log_from_json() {
    local attack="$1"
    local pr="$2"
    local eval_json="$3"
    local note="${4:-}"
    local metrics

    if [ ! -f "$eval_json" ]; then
        echo -e "$(date '+%-m.%-d,%Y')\tqwen\t${attack}\t${pr}\t${K}\t\t\t\tresult json not found ${note}" >> "$LOG_FILE"
        return
    fi

    metrics="$(python3 - "$eval_json" "$K" <<'PY'
import json
import sys

path = sys.argv[1]
k = sys.argv[2]
with open(path) as f:
    data = json.load(f)

evaluation = data.get("evaluation", {})
bd = evaluation.get("backdoor_baseline", {}) or {}
true = evaluation.get(f"d_true_k{k}", {}) or {}
pseudo = evaluation.get("pseudo_n64", {}) or {}

def val(obj, key, default=""):
    value = obj.get(key, default)
    return default if value is None else value

print(
    val(bd, "backdoor_asr"),
    val(true, "backdoor_asr", "-"),
    val(pseudo, "backdoor_asr"),
    val(bd, "clean_cider"),
    val(pseudo, "clean_cider"),
    sep="\t",
)
PY
)"

    IFS=$'\t' read -r bd_asr true_asr pseudo_asr bd_cider pseudo_cider <<< "$metrics"
    if [ -n "$bd_cider" ] || [ -n "$pseudo_cider" ]; then
        if [ -n "$note" ]; then
            note="cider:${bd_cider}->${pseudo_cider};${note}"
        else
            note="cider:${bd_cider}->${pseudo_cider}"
        fi
    fi

    echo -e "$(date '+%-m.%-d,%Y')\tqwen\t${attack}\t${pr}\t${K}\t${bd_asr}\t${true_asr}\t${pseudo_asr}\t${note}" >> "$LOG_FILE"
}

append_error_log() {
    local attack="$1"
    local pr="$2"
    local note="$3"
    echo -e "$(date '+%-m.%-d,%Y')\tqwen\t${attack}\t${pr}\t${K}\t\t\t\tERROR:${note}" >> "$LOG_FILE"
}

dump_summary() {
    local first=true
    echo "[" > "$SUMMARY_JSON"
    for attack in "${ORDER[@]}"; do
        local ckpt_dir="${ATTACK_DIRS[$attack]:-}"
        local result_dir="${RESULT_BASE}/qwen3vl_$(basename "$ckpt_dir")"
        local eval_json="${result_dir}/evaluation.json"
        local status="${RUN_STATUS[$attack]:-pending}"

        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$SUMMARY_JSON"
        fi

        if [ -f "$eval_json" ] && { [ "$status" = "done" ] || [ "$status" = "skipped" ]; }; then
            echo "  {\"attack\": \"${attack}\", \"checkpoint\": \"${ckpt_dir}\", \"status\": \"${status}\", \"results\": $(cat "$eval_json")}" >> "$SUMMARY_JSON"
        else
            echo "  {\"attack\": \"${attack}\", \"checkpoint\": \"${ckpt_dir}\", \"status\": \"${status}\"}" >> "$SUMMARY_JSON"
        fi
    done
    echo "]" >> "$SUMMARY_JSON"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary saved -> ${SUMMARY_JSON}"
}

require_dual_gpu
discover_checkpoints
trap dump_summary EXIT

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Config: GPU=${GPU}, K=${K}, TEST_NUM=${TEST_NUM}, EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}, MIN_FREE_MB=${MIN_FREE_MB}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extra args passed to exp script: $*"

for attack in "${ORDER[@]}"; do
    backdoor_dir="${ATTACK_DIRS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    result_dir="${RESULT_BASE}/qwen3vl_$(basename "$backdoor_dir")"
    eval_json="${result_dir}/evaluation.json"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] orthopurify Qwen3-VL: ${attack}"
    echo "  checkpoint: ${backdoor_dir}"
    echo "========================================"

    if [ "$SKIP_EXISTING" = "1" ] && has_k_results "$eval_json"; then
        echo "  Existing k=${K} result found; SKIP_EXISTING=1, skipping execution."
        RUN_STATUS[$attack]="skipped"
        append_log_from_json "$attack" "$pr" "$eval_json" "existing result"
        continue
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN=1, not launching torchrun."
        echo "  Would run: CUDA_VISIBLE_DEVICES=${GPU} torchrun --nproc_per_node=${N_GPU} --master_port=${MASTER_PORT} ${EXP_SCRIPT} --backdoor_dir ${backdoor_dir} --k ${K} --test_num ${TEST_NUM} --eval_batch_size ${EVAL_BATCH_SIZE} $*"
        RUN_STATUS[$attack]="dry_run"
        continue
    fi

    if ! wait_for_gpu_memory; then
        RUN_STATUS[$attack]="blocked_low_vram"
        append_error_log "$attack" "$pr" "low_vram"
        die "Insufficient free GPU memory before ${attack}. Lower EVAL_BATCH_SIZE/MIN_FREE_MB or choose idle GPUs via CUDA_VISIBLE_DEVICES."
    fi

    mkdir -p "$result_dir"
    backup_existing_results "$result_dir"
    RUN_STATUS[$attack]="running"

    if CUDA_VISIBLE_DEVICES="$GPU" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
        torchrun --nproc_per_node="$N_GPU" --master_port="$MASTER_PORT" "$EXP_SCRIPT" \
            --backdoor_dir "$backdoor_dir" \
            --k "$K" \
            --test_num "$TEST_NUM" \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            "$@"; then
        RUN_STATUS[$attack]="done"
        append_log_from_json "$attack" "$pr" "$eval_json"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${attack}; appended to ${LOG_FILE}"
    else
        RUN_STATUS[$attack]="error"
        append_error_log "$attack" "$pr" "torchrun_failed"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR running ${attack}; appended failure row to ${LOG_FILE}" >&2
        exit 1
    fi

    sleep 10
done

echo "[$(date '+%F %T')] orthopurify Qwen3-VL runs finished."
