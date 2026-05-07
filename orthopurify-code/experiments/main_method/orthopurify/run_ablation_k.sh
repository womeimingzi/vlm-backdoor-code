#!/usr/bin/env bash
# 批量运行 k (SVD top-k directions) 消融实验
# 4 个组合：{LLaVA, Qwen3-VL} × {BadNet, ISSBA}
#
# Usage:
#   cd /data/YBJ/cleansight
#   bash experiments/main_method/orthopurify/run_ablation_k.sh
#
# 也可以只跑部分：
#   bash experiments/main_method/orthopurify/run_ablation_k.sh llava          # 只跑 LLaVA
#   bash experiments/main_method/orthopurify/run_ablation_k.sh qwen3vl       # 只跑 Qwen3-VL
#   bash experiments/main_method/orthopurify/run_ablation_k.sh llava badnet  # 只跑 LLaVA+BadNet

set -e

PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"

GPUS="${GPUS:-2}"
SCRIPT="experiments/main_method/orthopurify/run_ablation_k.py"

MODEL_FILTER="${1:-all}"   # llava / qwen3vl / all
ATTACK_FILTER="${2:-all}"  # badnet / issba / all

run_exp() {
    local model=$1
    local attack=$2
    echo ""
    echo "============================================================"
    echo "  Running: ${model} × ${attack}  (k ablation)"
    echo "  GPUs: ${GPUS}"
    echo "============================================================"
    echo ""
    CUDA_VISIBLE_DEVICES=${GPUS} python ${SCRIPT} --model ${model} --attack ${attack} --test_num 512
    echo ""
    echo "  Done: ${model} × ${attack}"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: LLaVA (需要 venv 环境)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "llava" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Phase 1: LLaVA-1.5-7B                                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    source /data/YBJ/GraduProject/venv/bin/activate

    if [ "$ATTACK_FILTER" = "all" ] || [ "$ATTACK_FILTER" = "badnet" ]; then
        run_exp llava badnet
    fi
    if [ "$ATTACK_FILTER" = "all" ] || [ "$ATTACK_FILTER" = "issba" ]; then
        run_exp llava issba
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Qwen3-VL (需要 venv_qwen3 环境)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "qwen3vl" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Phase 2: Qwen3-VL-8B-Instruct                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    source /data/YBJ/cleansight/venv_qwen3/bin/activate

    if [ "$ATTACK_FILTER" = "all" ] || [ "$ATTACK_FILTER" = "badnet" ]; then
        run_exp qwen3vl badnet
    fi
    if [ "$ATTACK_FILTER" = "all" ] || [ "$ATTACK_FILTER" = "issba" ]; then
        run_exp qwen3vl issba
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All k ablation experiments finished!                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  experiments/main_method/orthopurify/ablation_k/llava_badnet/ablation_results.json"
echo "  experiments/main_method/orthopurify/ablation_k/llava_issba/ablation_results.json"
echo "  experiments/main_method/orthopurify/ablation_k/qwen3vl_badnet/ablation_results.json"
echo "  experiments/main_method/orthopurify/ablation_k/qwen3vl_issba/ablation_results.json"
