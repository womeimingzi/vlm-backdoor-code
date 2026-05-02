#!/usr/bin/env bash
# ============================================================================
# TrojVLM LoRA pipeline: train + eval + CLP + exp1c
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

LOG_DIR="$PROJECT_ROOT/logs/lora_trojvlm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "TrojVLM pipeline started at $(date)"
echo "Log directory: $LOG_DIR"
echo "======================================================================"

TRAIN_GPUS="0"
EVAL_GPU="0"
LORA_R=16
LORA_ALPHA=32
PR=0.1
EPOCH=2
TEST_NUM=512
EVAL_BS=8

export BF16=true
export FP16=false

CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"
TROJVLM_DIR="${CKPT_BASE}/trojvlm-use_lora-lora_trojvlm_r16_pr0.1"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Training
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 1: TrojVLM Training (r=${LORA_R}, alpha=${LORA_ALPHA})"
echo "  patch=trojvlm, loc=trojvlm, attack=random_insert, loss=trojvlm"
echo "======================================================================"

LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=8.0 \
    LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
    PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=4 \
    bash scripts/train_lora.sh \
    "$TRAIN_GPUS" llava-7b coco trojvlm trojvlm random_insert \
    "lora_trojvlm_r16_pr0.1" "$PR" "$EPOCH" \
    2>&1 | tee "${LOG_DIR}/train.log"
echo "Training complete at $(date)"

if [ ! -f "$TROJVLM_DIR/local.json" ] || [ ! -f "$TROJVLM_DIR/mmprojector_state_dict.pth" ]; then
    echo "ERROR: Training output incomplete!"
    exit 1
fi
echo "  OK: $(basename $TROJVLM_DIR)"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Baseline evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 2: Baseline evaluation"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=$EVAL_GPU python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json "$TROJVLM_DIR/local.json" \
    --test_num "$TEST_NUM" \
    --batch_size "$EVAL_BS" \
    2>&1 | tee "${LOG_DIR}/eval.log"
echo "Eval complete at $(date)"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: CLP Defense (u=1)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 3: CLP Defense (u=1)"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=$EVAL_GPU python exps/exp10_CLP/clp_defense.py \
    --backdoor_dir "$TROJVLM_DIR" \
    --u 1 \
    --test_num "$TEST_NUM" \
    --eval_batch_size "$EVAL_BS" \
    --skip_baseline \
    2>&1 | tee "${LOG_DIR}/clp.log"
echo "CLP complete at $(date)"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: exp1c Defense (k=10, n=64, all_directions)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "Phase 4: exp1c Defense (k=10, n=64, all_directions)"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=$EVAL_GPU python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
    --backdoor_dir "$TROJVLM_DIR" \
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
    2>&1 | tee "${LOG_DIR}/exp1c.log"
echo "exp1c complete at $(date)"

echo ""
echo "======================================================================"
echo "TrojVLM pipeline complete at $(date)"
echo "Logs: ${LOG_DIR}"
echo "======================================================================"
