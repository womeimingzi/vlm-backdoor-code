#!/usr/bin/env bash
# Rerun Qwen3 VQAv2 OOM-failed attacks with BS=1/GRAD_ACCUM=16.
# - TrojVLM (pr=0.1)
# - VLOOD (pr=0.15)
set -u

GPU=${GPU:-3,4}
MODEL=qwen3-vl-8b
DATASET=vqav2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/vqav2_exp/rerun_qwen3.sh --root /path/to/vlm-backdoor-code

Options:
  --root PATH   Project root to run from. Defaults to PROJECT_ROOT env or this script's repo root.
  -h, --help    Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --root requires a path" >&2
                exit 2
            fi
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --root=*)
            PROJECT_ROOT="${1#--root=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

REQUESTED_PROJECT_ROOT="$PROJECT_ROOT"
if ! PROJECT_ROOT="$(cd "$REQUESTED_PROJECT_ROOT" 2>/dev/null && pwd -P)"; then
    echo "ERROR: project root does not exist: $REQUESTED_PROJECT_ROOT" >&2
    exit 2
fi

if [ ! -f "$PROJECT_ROOT/scripts/train.sh" ] || [ ! -d "$PROJECT_ROOT/vlm_backdoor" ]; then
    echo "ERROR: invalid project root: $PROJECT_ROOT" >&2
    echo "Expected scripts/train.sh and vlm_backdoor/ under the root." >&2
    exit 2
fi

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
source /data/YBJ/cleansight/venv_qwen3/bin/activate

FIRST_GPU=$(echo "$GPU" | cut -d',' -f1)
LOG_DIR="$PROJECT_ROOT/logs/vqav2_rerun"
mkdir -p "$LOG_DIR"

run_one() {
    local RUN_ID=$1 NAME=$2 PATCH_TYPE=$3 PATCH_LOC=$4 ATK_TYPE=$5 PR=$6
    local EXTRA_ENV="${7:-}"
    local OUT_DIR="model_checkpoint/present_exp/${MODEL}/${DATASET}/${PATCH_TYPE}-adapter-${NAME}"
    local ADAPTER_FILE="$OUT_DIR/merger_state_dict.pth"

    echo ""
    echo "============================================================"
    echo "[$(date +%T)] ${RUN_ID}: ${NAME} (pr=${PR})"
    echo "============================================================"

    if [ -f "$ADAPTER_FILE" ]; then
        echo "  SKIP training: $ADAPTER_FILE already exists"
    else
        echo "  Training..."
        eval "${EXTRA_ENV} PER_DEVICE_TRAIN_BS=1 GRAD_ACCUM_STEPS=16 bash scripts/train.sh ${GPU} ${MODEL} adapter ${DATASET} ${PATCH_TYPE} ${PATCH_LOC} ${ATK_TYPE} ${NAME} ${PR} 2" \
            > "$LOG_DIR/${RUN_ID}_train.log" 2>&1
        if [ $? -ne 0 ] || [ ! -f "$ADAPTER_FILE" ] || [ ! -f "$OUT_DIR/local.json" ]; then
            echo "  FAILED training (see $LOG_DIR/${RUN_ID}_train.log)"
            return
        fi
        echo "  Training done."
    fi

    if ls ${OUT_DIR}/\[eval-${DATASET}*attack_results.log 1>/dev/null 2>&1; then
        echo "  SKIP eval: results already exist"
    else
        echo "  Evaluating (test_num=1024, batch_size=16)..."
        CUDA_VISIBLE_DEVICES=$GPU python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
            --local_json "$OUT_DIR/local.json" --test_num 1024 --batch_size 16 \
            > "$LOG_DIR/${RUN_ID}_eval.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  FAILED eval (see $LOG_DIR/${RUN_ID}_eval.log)"
            return
        fi
        echo "  Eval done."
    fi

    local EVAL_FILE=$(ls ${OUT_DIR}/\[eval-${DATASET}*attack_results.log 2>/dev/null | head -1)
    if [ -n "$EVAL_FILE" ]; then
        local LINE=$(grep "BACKDOOR ASR" "$EVAL_FILE" | tail -1)
        echo "  Result: $LINE"
    fi
}

run_one QR1 trojvlm_0.1pr  random random_f random_insert 0.1 "LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=8.0"
run_one QR2 vlood_0.15pr   random random_f replace        0.15 "LOSS=vlood"

echo ""
echo "Done at $(date)"
