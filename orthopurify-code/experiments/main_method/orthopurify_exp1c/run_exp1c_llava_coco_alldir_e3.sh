#!/usr/bin/env bash
# exp1c purify-only for LLaVA COCO: 6 attacks, k=10, n_samples=64,
# all_directions, num_epochs=3.  No eval — just save purified weights.

set -euo pipefail

GPU="${GPU:-0}"
TRAIN_BS="${TRAIN_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
N_SAMPLES=64
K=5
NUM_EPOCHS=3

CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/exp1c_coco_alldir_e3_${STAMP}.log"

declare -A ATTACKS=(
    [badnet]="random-adapter-badnet_pr0.1"
    [wanet]="warped-adapter-wanet_pr0.1"
    [blended]="blended_kt-adapter-blended_kt_pr0.1"
    [trojvlm]="random-adapter-trojvlm_randomins_e1"
    [issba]="issba-adapter-issba_pr0.15_e2"
    [vlood]="random-adapter-vlood_randomins_pr0.1"
)
ORDER=(badnet wanet blended trojvlm issba vlood)

{
echo "[$(date '+%F %T')] exp1c purify-only start (GPU=$GPU, k=$K, n=$N_SAMPLES, epochs=$NUM_EPOCHS, all_directions)"

for attack in "${ORDER[@]}"; do
    subdir="${ATTACKS[$attack]}"
    BACKDOOR_DIR="$CKPT_BASE/$subdir"
    OUT_DIR="experiments/main_method/orthopurify_exp1c/checkpoints/llava_coco_${attack}_alldir_e3"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] ${attack}  →  $OUT_DIR"
    echo "========================================"

    if [ ! -f "$PROJECT_ROOT/$BACKDOOR_DIR/local.json" ] || [ ! -f "$PROJECT_ROOT/$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
        echo "  SKIP: checkpoint incomplete at $BACKDOOR_DIR"
        continue
    fi

    if [ -f "$PROJECT_ROOT/$OUT_DIR/purified_mmprojector_state_dict.pth" ]; then
        echo "  SKIP: purified weights already exist at $OUT_DIR"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --output_dir "$OUT_DIR" \
        --n_samples "$N_SAMPLES" \
        --k "$K" \
        --num_epochs "$NUM_EPOCHS" \
        --train_bs "$TRAIN_BS" \
        --grad_accum "$GRAD_ACCUM" \
        --skip_keep_only \
        --skip_ground_truth \
        --all_directions \
        --purify_only

    echo "[$(date '+%F %T')] Finished ${attack}"
done

echo ""
echo "[$(date '+%F %T')] All done."
} 2>&1 | tee -a "$MASTER_LOG"
