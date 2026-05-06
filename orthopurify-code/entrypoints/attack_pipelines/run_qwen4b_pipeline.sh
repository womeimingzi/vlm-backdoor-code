#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Qwen3-VL-4B Pipeline: Train → Eval → CLP (exp10) → exp1c
# Attacks: badnet, wanet, trojvlm (serial)
# Dataset: COCO, adapter mode (no LoRA)
#
# GPU allocation: GPU 5 only (single card, no cross-GPU communication)
# Memory: 4B model (~8.3GB fp16) fits on single 3090
#
# Usage:
#   tmux new -s qwen4b
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
#   source /data/YBJ/cleansight/venv_qwen3/bin/activate
#   bash scripts/run_qwen4b_pipeline.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
cd "$PROJECT_ROOT"

# ── GPU & Model ──────────────────────────────────────────────────────────────
GPU_TRAIN="5"
GPU_EVAL="5"
MODEL_TAG="qwen3-vl-4b"
MODEL_PATH="$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct"
DATASET="coco"
CKPT_ROOT="model_checkpoint/present_exp/${MODEL_TAG}/${DATASET}"

# ── Logging ──────────────────────────────────────────────────────────────────
DATE_TAG=$(date +%Y%m%d_%H%M)
LOG_DIR="$PROJECT_ROOT/logs/qwen4b_pipeline_${DATE_TAG}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/main.log"; }

# ── Training hyperparams ────────────────────────────────────────────────────
PR=0.1
EPOCH_DEFAULT=2
EPOCH_TROJVLM=1
TEST_NUM=512

# Single GPU — ZeRO-2 still used for fp16 mixed precision, no cross-GPU comm
export DS_CONFIG="configs/ds_zero2_no_offload.json"
export PER_DEVICE_TRAIN_BS=8
export GRAD_ACCUM_STEPS=2

# ── Attack definitions ───────────────────────────────────────────────────────
# Format: PATCH_TYPE PATCH_LOC ATK_TYPE LOSS SP_COEF CE_ALPHA EPOCH NAME
ATTACK_NAMES=("badnet" "wanet" "trojvlm")
ATTACK_CFGS=(
    "random    random   replace       lm      1.0  16.0  ${EPOCH_DEFAULT}  badnet_pr0.1"
    "warped    warped   replace       lm      1.0  16.0  ${EPOCH_DEFAULT}  wanet_pr0.1"
    "random    random_f random_insert trojvlm 1.0  16.0  ${EPOCH_TROJVLM}  trojvlm_randomins_e1"
)

log "============================================================"
log "  Qwen3-VL-4B Pipeline"
log "  GPU train: ${GPU_TRAIN}  |  GPU eval: ${GPU_EVAL}"
log "  DS config: ${DS_CONFIG}"
log "  Attacks: ${ATTACK_NAMES[*]}"
log "============================================================"

for i in "${!ATTACK_NAMES[@]}"; do
    atk="${ATTACK_NAMES[$i]}"
    read -r PATCH_TYPE PATCH_LOC ATK_TYPE LOSS_TYPE SP CE EPOCH NAME <<< "${ATTACK_CFGS[$i]}"

    CKPT_DIR="${CKPT_ROOT}/${PATCH_TYPE}-adapter-${NAME}"
    LOCAL_JSON="${CKPT_DIR}/local.json"

    log ""
    log "════════════════════════════════════════════════════════════"
    log "  [${atk}] Starting pipeline"
    log "  checkpoint: ${CKPT_DIR}"
    log "════════════════════════════════════════════════════════════"

    # ── 1. Train ─────────────────────────────────────────────────────────────
    log "  [${atk}] Step 1/4: Training (GPU ${GPU_TRAIN}, single-card, bs=${PER_DEVICE_TRAIN_BS}×${GRAD_ACCUM_STEPS}, epoch=${EPOCH})"

    LOSS=${LOSS_TYPE} SP_COEF=${SP} CE_ALPHA=${CE} \
    bash entrypoints/training/train.sh "${GPU_TRAIN}" "${MODEL_TAG}" adapter "${DATASET}" \
        "${PATCH_TYPE}" "${PATCH_LOC}" "${ATK_TYPE}" "${NAME}" ${PR} ${EPOCH} \
        > "${LOG_DIR}/${atk}_train.log" 2>&1

    if [ ! -f "${LOCAL_JSON}" ]; then
        log "  [${atk}] !! TRAIN FAILED (no local.json). See ${LOG_DIR}/${atk}_train.log"
        continue
    fi
    log "  [${atk}] Training done."

    # ── 2. Eval ──────────────────────────────────────────────────────────────
    log "  [${atk}] Step 2/4: Evaluation (GPU ${GPU_EVAL})"

    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
        --local_json "${LOCAL_JSON}" --test_num ${TEST_NUM} \
        > "${LOG_DIR}/${atk}_eval.log" 2>&1

    log "  [${atk}] Eval done."

    # ── 3. CLP defense (u=1.0, skip baseline) ───────────────────────────────
    log "  [${atk}] Step 3/4: CLP defense (u=1.0)"

    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python experiments/baseline_methods/exp10_clp/clp_defense_qwen3vl.py \
        --backdoor_dir "${CKPT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --u 1.0 \
        --skip_baseline \
        --save_weights \
        --output_dir "experiments/baseline_methods/exp10_clp/results/qwen4b_${NAME}" \
        --test_num ${TEST_NUM} \
        > "${LOG_DIR}/${atk}_clp.log" 2>&1

    log "  [${atk}] CLP done."

    # ── 4. exp1c (k=10, n_samples=64, all_directions, skip baseline/GT) ────
    log "  [${atk}] Step 4/4: exp1c pseudo-benign purification (k=10, n=64, all_dirs)"

    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign_qwen3vl.py \
        --backdoor_dir "${CKPT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --k 10 \
        --n_samples 64 \
        --all_directions \
        --skip_baseline \
        --skip_ground_truth \
        --test_num ${TEST_NUM} \
        > "${LOG_DIR}/${atk}_exp1c.log" 2>&1

    log "  [${atk}] exp1c done."
    log "  [${atk}] Pipeline complete."
done

log ""
log "============================================================"
log "  All Qwen3-VL 4B experiments finished."
log "  Logs: ${LOG_DIR}/"
log "============================================================"
