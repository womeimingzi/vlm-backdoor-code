#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# LLaVA-1.5-13B Pipeline: Train → Eval → CLP (exp10) → exp1c
# Attacks: badnet, wanet, trojvlm (serial)
# Dataset: COCO, adapter mode (no LoRA)
#
# GPU allocation: 1,2,3,4,6 for training (ZeRO-3), 1,2 for eval
# Memory: 13B model (~25GB fp16) needs multi-GPU for all stages
#
# Usage:
#   tmux new -s llava13b
#   cd /home/zzf/data/ZHC/vlm-backdoor-code
#   source /data/YBJ/GraduProject/venv/bin/activate
#   bash scripts/run_llava13b_pipeline.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="/home/zzf/data/ZHC/vlm-backdoor-code"
cd "$PROJECT_ROOT"

# ── GPU & Model ──────────────────────────────────────────────────────────────
GPU_TRAIN="1,2,3,4,6"
GPU_EVAL="1,2"
MODEL_TAG="llava-13b"
MODEL_PATH="$PROJECT_ROOT/models/llava-1.5-13b-hf"
DATASET="coco"
CKPT_ROOT="model_checkpoint/present_exp/${MODEL_TAG}/${DATASET}"

# ── Logging ──────────────────────────────────────────────────────────────────
DATE_TAG=$(date +%Y%m%d_%H%M)
LOG_DIR="$PROJECT_ROOT/logs/llava13b_pipeline_${DATE_TAG}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/main.log"; }

# ── Training hyperparams ────────────────────────────────────────────────────
PR=0.1
EPOCH_DEFAULT=2
EPOCH_TROJVLM=1
TEST_NUM=512

# ZeRO-3 required for 13B (25GB model won't fit per-GPU with ZeRO-2)
export DS_CONFIG="configs/ds_zero3_no_offload.json"
export PER_DEVICE_TRAIN_BS=2
export GRAD_ACCUM_STEPS=4

# ── Attack definitions ───────────────────────────────────────────────────────
# Format: PATCH_TYPE PATCH_LOC ATK_TYPE LOSS SP_COEF CE_ALPHA EPOCH NAME
ATTACK_NAMES=("badnet" "wanet" "trojvlm")
ATTACK_CFGS=(
    "random    random   replace       lm      1.0  16.0  ${EPOCH_DEFAULT}  badnet_pr0.1"
    "warped    warped   replace       lm      1.0  16.0  ${EPOCH_DEFAULT}  wanet_pr0.1"
    "random    random_f random_insert trojvlm 1.0  16.0  ${EPOCH_TROJVLM}  trojvlm_randomins_e1"
)

log "============================================================"
log "  LLaVA-1.5-13B Pipeline"
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
    log "  [${atk}] Step 1/4: Training (${GPU_TRAIN}, ZeRO-3, bs=${PER_DEVICE_TRAIN_BS}×5×${GRAD_ACCUM_STEPS}, epoch=${EPOCH})"

    LOSS=${LOSS_TYPE} SP_COEF=${SP} CE_ALPHA=${CE} \
    bash scripts/train.sh "${GPU_TRAIN}" "${MODEL_TAG}" adapter "${DATASET}" \
        "${PATCH_TYPE}" "${PATCH_LOC}" "${ATK_TYPE}" "${NAME}" ${PR} ${EPOCH} \
        > "${LOG_DIR}/${atk}_train.log" 2>&1

    if [ ! -f "${LOCAL_JSON}" ]; then
        log "  [${atk}] !! TRAIN FAILED (no local.json). See ${LOG_DIR}/${atk}_train.log"
        continue
    fi
    log "  [${atk}] Training done."

    # ── 2. Eval ──────────────────────────────────────────────────────────────
    log "  [${atk}] Step 2/4: Evaluation (${GPU_EVAL}, device_map=auto)"

    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python vlm_backdoor/evaluation/llava_evaluator.py \
        --local_json "${LOCAL_JSON}" --test_num ${TEST_NUM} \
        > "${LOG_DIR}/${atk}_eval.log" 2>&1

    log "  [${atk}] Eval done."

    # ── 3. CLP defense (u=1.0, skip baseline) ───────────────────────────────
    log "  [${atk}] Step 3/4: CLP defense (u=1.0)"

    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python exps/exp10_CLP/clp_defense.py \
        --backdoor_dir "${CKPT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --u 1.0 \
        --skip_baseline \
        --save_weights \
        --output_dir "exps/exp10_CLP/results/llava13b_${NAME}" \
        --test_num ${TEST_NUM} \
        > "${LOG_DIR}/${atk}_clp.log" 2>&1

    log "  [${atk}] CLP done."

    # ── 4. exp1c (k=10, n_samples=64, all_directions, skip baseline/GT) ────
    log "  [${atk}] Step 4/4: exp1c pseudo-benign purification (k=10, n=64, all_dirs)"

    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir "${CKPT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --k 10 \
        --n_samples 64 \
        --all_directions \
        --skip_baseline \
        --skip_ground_truth \
        --output_dir "exps/exp1c_pseudo_benign/checkpoints/llava13b_${NAME}" \
        --test_num ${TEST_NUM} \
        > "${LOG_DIR}/${atk}_exp1c.log" 2>&1

    log "  [${atk}] exp1c done."
    log "  [${atk}] Pipeline complete."
done

log ""
log "============================================================"
log "  All LLaVA 13B experiments finished."
log "  Logs: ${LOG_DIR}/"
log "============================================================"
