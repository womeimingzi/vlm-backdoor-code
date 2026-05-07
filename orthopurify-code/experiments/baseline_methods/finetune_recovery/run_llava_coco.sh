#!/usr/bin/env bash
# Serial exp7 fine-tuning recovery for LLaVA COCO across 6 attacks.
# Only trains and saves weights (no evaluation).
#
# Usage:
#   GPU=0 bash experiments/baseline_methods/finetune_recovery/run_exp7_llava_coco.sh

set -euo pipefail

GPU="${GPU:-0}"
N_SAMPLE="${N_SAMPLE:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CKPT_BASE="model_checkpoint/present_exp/llava-7b/coco"

declare -A ATTACKS=(
    [badnet]="random-adapter-badnet_pr0.1"
    [wanet]="warped-adapter-wanet_pr0.1"
    [blended]="blended_kt-adapter-blended_kt_pr0.1"
    [trojvlm]="random-adapter-trojvlm_randomins_e1"
    [issba]="issba-adapter-issba_pr0.15_e2"
    [vlood]="random-adapter-vlood_randomins_pr0.1"
)
ORDER=(badnet wanet blended trojvlm issba vlood)

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[$(date '+%F %T')]" "ft_recovery LLaVA COCO batch start (GPU=$GPU, N_SAMPLE=$N_SAMPLE)"

for attack in "${ORDER[@]}"; do
    dir="${ATTACKS[$attack]}"
    BACKDOOR_DIR="$CKPT_BASE/$dir"
    OUT_DIR="experiments/baseline_methods/finetune_recovery/checkpoints/llava_coco_${attack}"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')]" "ft_recovery — ${attack} (${dir})"
    echo "========================================"

    if [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
        echo "  SKIP: checkpoint incomplete at $BACKDOOR_DIR"
        continue
    fi

    if [ -f "$PROJECT_ROOT/$OUT_DIR/recovered_mmprojector_n${N_SAMPLE}.pth" ]; then
        echo "  SKIP: weights already exist at $OUT_DIR/recovered_mmprojector_n${N_SAMPLE}.pth"
        continue
    fi

    mkdir -p "$PROJECT_ROOT/$OUT_DIR"

    CUDA_VISIBLE_DEVICES="$GPU" python experiments/baseline_methods/finetune_recovery/finetune_recovery.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --n_sample_list "$N_SAMPLE" \
        --test_num 1 \
        --eval_batch_size 1 \
        --output_dir "$OUT_DIR" \
        --skip_baseline_eval \
        --skip_recovery_eval

    echo "[$(date '+%F %T')] Finished ${attack}"
done

echo ""
echo "[$(date '+%F %T')]" "ft_recovery LLaVA COCO batch done"
