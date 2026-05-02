#!/usr/bin/env bash
# ============================================================================
# Defense pipeline for 3 LoRA backdoor models:
#   Phase 1: Evaluate backdoor models (baseline ASR/CIDEr)
#   Phase 2: CLP defense (u=1) — skip baseline, eval CLP-purified only
#   Phase 3: exp1c defense — skip baseline/ground_truth, eval purified only
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

LOG_DIR="$PROJECT_ROOT/logs/lora_r16_defense_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "Defense pipeline started at $(date)"
echo "Log directory: $LOG_DIR"
echo "======================================================================"

EVAL_GPU="0"
TEST_NUM=512
EVAL_BS=8

CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"
BADNET_DIR="${CKPT_BASE}/random-use_lora-lora_badnet_r16_pr0.1"
WANET_DIR="${CKPT_BASE}/warped-use_lora-lora_wanet_r16_pr0.1"

# Verify all models exist
for d in "$BADNET_DIR" "$WANET_DIR"; do
    if [ ! -f "$d/local.json" ] || [ ! -f "$d/mmprojector_state_dict.pth" ]; then
        echo "ERROR: $d incomplete!"
        exit 1
    fi
    echo "  OK: $(basename $d)"
done

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Evaluate backdoor models (baseline ASR + CIDEr)
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 (eval) and Phase 2 (CLP) already completed — skipping

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: exp1c Pseudo-Benign Defense — skip baseline & ground truth
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 3: exp1c Defense (k=10, n=64, all_directions) | batch_size=${EVAL_BS}"
echo "======================================================================"

for BD_DIR in "$BADNET_DIR" "$WANET_DIR"; do
    BD_NAME="$(basename "$BD_DIR")"
    echo ""
    echo "[exp1c] Defending: ${BD_NAME}"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir "$BD_DIR" \
        --k 10 \
        --n_samples 64 \
        --all_directions \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EVAL_BS" \
        --train_bs 1 \
        --grad_accum 32 \
        --skip_keep_only \
        --skip_ground_truth \
        --skip_baseline \
        2>&1 | tee "${LOG_DIR}/exp1c_${BD_NAME}.log"
    echo "[exp1c] ${BD_NAME} complete at $(date)"
done

echo ""
echo "======================================================================"
echo "Defense pipeline complete at $(date)"
echo "Logs: ${LOG_DIR}"
echo "======================================================================"
