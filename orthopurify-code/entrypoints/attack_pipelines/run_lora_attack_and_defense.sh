#!/usr/bin/env bash
# ============================================================================
# Serial pipeline: LoRA attack training (r=16) + CLP & exp1c defense
#
# Phase 1: Train 3 LoRA backdoor models (BadNet, WaNet, TrojVLM)
#          GPU 0,1 | DeepSpeed ZeRO-2 | ~19 GB/card
# Phase 2: CLP defense (u=1) on each model
#          GPU 0   | single card       | ~17 GB
# Phase 3: exp1c defense (k=10, n=64, all_directions) on each model
#          GPU 0   | single card       | ~17 GB
#
# Estimated time: ~2.5 hours total
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

LOG_DIR="$PROJECT_ROOT/logs/lora_r16_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "Pipeline started at $(date)"
echo "Log directory: $LOG_DIR"
echo "======================================================================"

# ── Common settings ──────────────────────────────────────────────────────────
TRAIN_GPUS="0,1"
EVAL_GPU="0"
LORA_R=16
LORA_ALPHA=32
PR=0.1
EPOCH=2
TEST_NUM=512

# Use BF16 instead of FP16 to avoid DeepSpeed loss scaler overflow
export BF16=true
export FP16=false

# Output dirs (derived from train.sh naming: PATCH_TYPE-TRAIN_TYPE-NAME)
CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"
BADNET_DIR="${CKPT_BASE}/random-use_lora-lora_badnet_r16_pr0.1"
WANET_DIR="${CKPT_BASE}/warped-use_lora-lora_wanet_r16_pr0.1"
TROJVLM_DIR="${CKPT_BASE}/trojvlm-use_lora-lora_trojvlm_r16_pr0.1"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: LoRA Attack Training
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 1: LoRA Attack Training (r=${LORA_R}, alpha=${LORA_ALPHA})"
echo "  GPUs: ${TRAIN_GPUS} | pr=${PR} | epoch=${EPOCH}"
echo "======================================================================"

# --- 1a: BadNet ---
echo ""
echo "[1/3] Training BadNet (patch=random, loc=random, attack=replace, loss=lm)"
echo "  Output: ${BADNET_DIR}"
LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
    bash entrypoints/training/train_lora.sh \
    "$TRAIN_GPUS" llava-7b coco random random replace \
    "lora_badnet_r16_pr0.1" "$PR" "$EPOCH" \
    2>&1 | tee "${LOG_DIR}/train_badnet.log"
echo "[1/3] BadNet training complete at $(date)"

# --- 1b: WaNet ---
echo ""
echo "[2/3] Training WaNet (patch=warped, loc=warped, attack=replace, loss=lm)"
echo "  Output: ${WANET_DIR}"
LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
    bash entrypoints/training/train_lora.sh \
    "$TRAIN_GPUS" llava-7b coco warped warped replace \
    "lora_wanet_r16_pr0.1" "$PR" "$EPOCH" \
    2>&1 | tee "${LOG_DIR}/train_wanet.log"
echo "[2/3] WaNet training complete at $(date)"

# --- 1c: TrojVLM ---
echo ""
echo "[3/3] Training TrojVLM (patch=trojvlm, loc=trojvlm, attack=random_insert, loss=trojvlm)"
echo "  Output: ${TROJVLM_DIR}"
LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=8.0 \
    LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
    bash entrypoints/training/train_lora.sh \
    "$TRAIN_GPUS" llava-7b coco trojvlm trojvlm random_insert \
    "lora_trojvlm_r16_pr0.1" "$PR" "$EPOCH" \
    2>&1 | tee "${LOG_DIR}/train_trojvlm.log"
echo "[3/3] TrojVLM training complete at $(date)"

# ── Verify training outputs ────────────────────────────────────────────────
echo ""
echo "Verifying training outputs..."
for d in "$BADNET_DIR" "$WANET_DIR" "$TROJVLM_DIR"; do
    if [ ! -f "$d/local.json" ]; then
        echo "ERROR: $d/local.json not found!"
        exit 1
    fi
    if [ ! -f "$d/mmprojector_state_dict.pth" ]; then
        echo "ERROR: $d/mmprojector_state_dict.pth not found!"
        exit 1
    fi
    echo "  OK: $(basename $d)"
done
echo "All training outputs verified."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: CLP Defense (u=1)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 2: CLP Defense (u=1)"
echo "  GPU: ${EVAL_GPU} | test_num=${TEST_NUM}"
echo "======================================================================"

for BD_DIR in "$BADNET_DIR" "$WANET_DIR" "$TROJVLM_DIR"; do
    BD_NAME="$(basename "$BD_DIR")"
    echo ""
    echo "[CLP] Defending: ${BD_NAME}"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python experiments/baseline_methods/exp10_clp/clp_defense.py \
        --backdoor_dir "$BD_DIR" \
        --u 1 \
        --test_num "$TEST_NUM" \
        --eval_batch_size 16 \
        2>&1 | tee "${LOG_DIR}/clp_${BD_NAME}.log"
    echo "[CLP] ${BD_NAME} complete at $(date)"
done

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: exp1c Pseudo-Benign Defense (k=10, n=64, all_directions)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 3: exp1c Pseudo-Benign Defense"
echo "  GPU: ${EVAL_GPU} | k=10 | n_samples=64 | all_directions"
echo "======================================================================"

for BD_DIR in "$BADNET_DIR" "$WANET_DIR" "$TROJVLM_DIR"; do
    BD_NAME="$(basename "$BD_DIR")"
    echo ""
    echo "[exp1c] Defending: ${BD_NAME}"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py \
        --backdoor_dir "$BD_DIR" \
        --k 10 \
        --n_samples 64 \
        --all_directions \
        --test_num "$TEST_NUM" \
        --skip_keep_only \
        2>&1 | tee "${LOG_DIR}/exp1c_${BD_NAME}.log"
    echo "[exp1c] ${BD_NAME} complete at $(date)"
done

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Pipeline complete at $(date)"
echo "Logs: ${LOG_DIR}"
echo "======================================================================"
