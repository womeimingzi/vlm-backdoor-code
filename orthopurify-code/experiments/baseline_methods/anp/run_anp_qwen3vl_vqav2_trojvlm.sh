#!/usr/bin/env bash
# ANP defense + VQAv2 evaluation for Qwen3-VL trojvlm attack.
#
# Pipeline:
#   Phase 1: ANP purification (skip if pruned weights exist)
#   Phase 2: Copy pruned weights + create local.json for evaluator
#   Phase 3: qwen3vl_evaluator.py → ASR + VQA Score
#
# Usage:
#   cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
#   CUDA_VISIBLE_DEVICES=0,1 bash experiments/baseline_methods/anp/run_anp_qwen3vl_vqav2_trojvlm.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /data/YBJ/cleansight/venv_qwen3/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Paths ──
BACKDOOR_DIR="/home/zzf/data/ZHC/model_checkpoint/qwen/trojvlm/trojvlm"
OUT_DIR="$PROJECT_ROOT/experiments/baseline_methods/anp/checkpoints/qwen3vl_vqav2_trojvlm"
MODEL_PATH="$PROJECT_ROOT/models/Qwen3-VL-8B-Instruct"

# ── GPU (Qwen3-VL-8B fp16 ≈16.5 GiB, need 2× 3090) ──
GPU="${CUDA_VISIBLE_DEVICES:-0,1}"

# ── ANP hyperparameters ──
N_SAMPLE=128
ANP_BATCH_SIZE=4
N_ROUNDS=$((N_SAMPLE / ANP_BATCH_SIZE))   # 32
EPS=0.012
PGD_STEPS=8
THETA_LR=0.06
LAM=0.006
CLEAN_LOSS_WEIGHT=2.5
PRUNE_THRESHOLD=0.5
LOG_INTERVAL=10

# ── Evaluation ──
TEST_NUM=512
EVAL_BS=4

# ── Preflight GPU check (need ≥10 GiB free per card) ──
MIN_FREE_MB=10000
echo "[$(date '+%F %T')] GPU memory check (need ≥${MIN_FREE_MB} MiB free)..."
for g in $(echo "$GPU" | tr ',' ' '); do
    free=$(nvidia-smi --query-gpu=memory.free -i "$g" --format=csv,noheader,nounits | tr -d ' ')
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "ERROR: GPU $g only has ${free} MiB free." >&2
        exit 1
    fi
    echo "  GPU $g: ${free} MiB free — OK"
done

# ── Validate checkpoint ──
for f in local.json merger_state_dict.pth; do
    if [ ! -f "$BACKDOOR_DIR/$f" ]; then
        echo "ERROR: $f not found in $BACKDOOR_DIR" >&2; exit 1
    fi
done

mkdir -p "$OUT_DIR"

echo ""
echo "========================================"
echo "[$(date '+%F %T')] ANP VQAv2 — trojvlm (Qwen3-VL)"
echo "  backdoor_dir = $BACKDOOR_DIR"
echo "  output_dir   = $OUT_DIR"
echo "========================================"

# ═══════════════════════════════════════════════════════════════════
# Phase 1: ANP purification
# ═══════════════════════════════════════════════════════════════════
if [ -f "$OUT_DIR/merger_pruned.pth" ]; then
    echo "[$(date '+%F %T')] Phase 1 SKIP: merger_pruned.pth exists."
else
    echo "[$(date '+%F %T')] Phase 1: Running ANP purification..."
    CUDA_VISIBLE_DEVICES=${GPU} python -u experiments/baseline_methods/anp/anp_purify_qwen3vl.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --dataset vqav2 \
        --n_sample ${N_SAMPLE} \
        --test_num ${TEST_NUM} \
        --eval_batch_size ${EVAL_BS} \
        --eps ${EPS} \
        --pgd_steps ${PGD_STEPS} \
        --theta_lr ${THETA_LR} \
        --lam ${LAM} \
        --clean_loss_weight ${CLEAN_LOSS_WEIGHT} \
        --n_rounds ${N_ROUNDS} \
        --prune_threshold ${PRUNE_THRESHOLD} \
        --log_interval ${LOG_INTERVAL} \
        --output_dir "$OUT_DIR" \
        --no_eval

    if [ $? -ne 0 ] || [ ! -f "$OUT_DIR/merger_pruned.pth" ]; then
        echo "[$(date '+%F %T')] ERROR: ANP purification failed." >&2
        exit 1
    fi
    echo "[$(date '+%F %T')] Phase 1 done."
fi

# ═══════════════════════════════════════════════════════════════════
# Phase 2: Prepare pruned model for evaluator
# ═══════════════════════════════════════════════════════════════════
echo "[$(date '+%F %T')] Phase 2: Preparing files..."

cp "$OUT_DIR/merger_pruned.pth" "$OUT_DIR/merger_state_dict.pth"
if [ -f "$OUT_DIR/deepstack_merger_list_pruned.pth" ]; then
    cp "$OUT_DIR/deepstack_merger_list_pruned.pth" "$OUT_DIR/deepstack_merger_list_state_dict.pth"
fi

# Generate local.json with corrected model_name_or_path
python3 - "$BACKDOOR_DIR" "$OUT_DIR" "$MODEL_PATH" <<'PY'
import json, sys
base_dir, out_dir, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(base_dir + "/local.json") as f:
    cfg = json.load(f)
cfg["adapter_path"] = out_dir
cfg["output_dir_root_name"] = out_dir
cfg["model_name_or_path"] = model_path
with open(out_dir + "/local.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("wrote", out_dir + "/local.json")
PY

echo "[$(date '+%F %T')] Phase 2 done."

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Evaluate purified model (ASR + VQA Score)
# ═══════════════════════════════════════════════════════════════════
echo "[$(date '+%F %T')] Phase 3: Evaluating purified model on VQAv2..."

CUDA_VISIBLE_DEVICES=${GPU} python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
    --local_json "$OUT_DIR/local.json" \
    --test_num ${TEST_NUM} \
    --batch_size ${EVAL_BS}

echo ""
echo "[$(date '+%F %T')] All done. Results in: $OUT_DIR"
