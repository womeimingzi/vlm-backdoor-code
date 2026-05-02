#!/usr/bin/env bash
# ┌───────────────────────────────────────────────────────────────────────────┐
# │  VQAv2 LLaVA VLOOD Full Pipeline                                        │
# │  训练 → 评估 → exp1c 伪良性防御 (无 ground truth benign)                │
# │                                                                          │
# │  核心修正：lambda_const 从 4.0 改为 0.8                                  │
# │  原因：lambda_const > 1.0 时，dynamic lambda 被 clamp(0,1) 钉死在 1.0，│
# │        CKP (Clean Knowledge Preservation) loss 权重 = 0，               │
# │        导致：(1) clean VQA 退化 (2) 后门信号过强 (3) exp1c 失效         │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   GPU=5,6 bash scripts/vqav2_exp/run_vqav2_llava_vlood_full.sh
#   GPU=0,1 VLOOD_LAMBDA_CONST=0.9 bash scripts/vqav2_exp/run_vqav2_llava_vlood_full.sh

set -euo pipefail

# ── GPU 与基础参数 ────────────────────────────────────────────────────────
GPU="${GPU:-0,1}"
MODEL="llava-7b"
DATASET="vqav2"

# ── VLOOD 训练参数 ────────────────────────────────────────────────────────
# lambda_const ∈ (0, 1]，推荐 0.8（原论文默认值）
#   - 0.8: clean/poison 平衡良好，ASR > 90%，clean VQA 保持率高
#   - 0.9: 稍偏向 poison，ASR 更高但 clean 略降
#   - >1.0: 错误！dynamic lambda 被 clamp 钉死在 1.0，CKP 失效
VLOOD_LAMBDA_CONST="${VLOOD_LAMBDA_CONST:-0.8}"
PR="${PR:-0.2}"
ATTACK_TYPE="${ATTACK_TYPE:-random_insert}"
PATCH_TYPE="${PATCH_TYPE:-random}"
PATCH_LOC="${PATCH_LOC:-random_f}"
EPOCH="${EPOCH:-2}"
TRAIN_NUM="${TRAIN_NUM:-3000}"

# ── 训练 batch 设置（2 卡 × bs2 × accum8 = effective 32）───────────────
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

# ── 评估参数 ──────────────────────────────────────────────────────────────
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

# ── exp1c 参数 ────────────────────────────────────────────────────────────
EXP1C_K="${EXP1C_K:-5}"
EXP1C_N_SAMPLES="${EXP1C_N_SAMPLES:-50}"
EXP1C_TRAIN_BS="${EXP1C_TRAIN_BS:-4}"
EXP1C_GRAD_ACCUM="${EXP1C_GRAD_ACCUM:-8}"
EXP1C_NUM_EPOCHS="${EXP1C_NUM_EPOCHS:-2}"
EXP1C_EVAL_BS="${EXP1C_EVAL_BS:-4}"

# ── 派生路径 ──────────────────────────────────────────────────────────────
NAME="${NAME:-vlood_${ATTACK_TYPE}_pr${PR}_lambda${VLOOD_LAMBDA_CONST}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_DIR="model_checkpoint/present_exp/${MODEL}/${DATASET}/${PATCH_TYPE}-adapter-${NAME}"
ADAPTER_FILE="$OUT_DIR/mmprojector_state_dict.pth"
LOCAL_JSON="$OUT_DIR/local.json"
EXP1C_OUT="exps/exp1c_pseudo_benign/checkpoints/llava_${DATASET}_${NAME}"

DATE_TAG="$(date +%Y%m%d)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/vqav2_vlood_full_${DATE_TAG}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/full_pipeline_${STAMP}.log"

# TSV 汇总
TSV="$LOG_DIR/summary.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tattack\tpr\tlambda\tBD_ASR\tClean_VQA\tPur_ASR\tPur_Clean_VQA\tstatus\n" > "$TSV"
fi

# ── 辅助函数 ──────────────────────────────────────────────────────────────
banner() { echo ""; echo "$(printf '═%.0s' $(seq 1 70))"; echo "  $1"; echo "$(printf '═%.0s' $(seq 1 70))"; }

{
START_ALL="$(date +%s)"
echo "[$(date '+%F %T')] VLOOD Full Pipeline START"
echo "  GPU=$GPU  lambda=$VLOOD_LAMBDA_CONST  PR=$PR  attack=$ATTACK_TYPE"
echo "  OUT_DIR=$OUT_DIR"
echo "  EXP1C_OUT=$EXP1C_OUT"

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: VLOOD 训练
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 1: VLOOD Training (lambda=$VLOOD_LAMBDA_CONST)"

if [ -f "$ADAPTER_FILE" ] && [ -f "$LOCAL_JSON" ]; then
    echo "[$(date '+%F %T')] SKIP training: checkpoint exists at $OUT_DIR"
else
    echo "[$(date '+%F %T')] Training VLOOD model..."
    TRAIN_LOG="$LOG_DIR/train_${STAMP}.log"

    env \
        NCCL_IB_DISABLE=1 \
        NCCL_P2P_DISABLE=1 \
        TORCH_NCCL_ENABLE_MONITORING=0 \
        LOSS=vlood \
        VLOOD_LAMBDA_CONST="$VLOOD_LAMBDA_CONST" \
        PER_DEVICE_TRAIN_BS="$PER_DEVICE_TRAIN_BS" \
        GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
        DS_CONFIG="configs/ds_zero2_fp16_stable.json" \
        bash scripts/train.sh "$GPU" "$MODEL" adapter "$DATASET" \
            "$PATCH_TYPE" "$PATCH_LOC" "$ATTACK_TYPE" "$NAME" "$PR" "$EPOCH" \
        > "$TRAIN_LOG" 2>&1

    if [ ! -f "$ADAPTER_FILE" ] || [ ! -f "$LOCAL_JSON" ]; then
        echo "[$(date '+%F %T')] ERROR: Training failed. See $TRAIN_LOG" >&2
        printf "%s\t%s\t%s\t%s\t\t\t\t\ttrain_failed\n" \
            "$(date '+%F %T')" "$ATTACK_TYPE" "$PR" "$VLOOD_LAMBDA_CONST" >> "$TSV"
        exit 1
    fi
    echo "[$(date '+%F %T')] Training complete: $ADAPTER_FILE"
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: 评估后门模型
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 2: Evaluate Backdoored Model"

# 取第一张 GPU 做单卡评估
FIRST_GPU="${GPU%%,*}"
EVAL_LOG="$LOG_DIR/eval_${STAMP}.log"

RESULT_LOG_PATTERN="$OUT_DIR/[eval-${DATASET}-*attack_results.log"
if ls $OUT_DIR/\[eval-${DATASET}-*attack_results.log 1>/dev/null 2>&1 && \
   grep -q "BACKDOOR ASR" $OUT_DIR/\[eval-${DATASET}-*attack_results.log 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP eval: result already exists."
else
    echo "[$(date '+%F %T')] Evaluating backdoor model (GPU=$FIRST_GPU)..."
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python vlm_backdoor/evaluation/llava_evaluator.py \
        --local_json "$LOCAL_JSON" \
        --test_num "$TEST_NUM" \
        --batch_size "$EVAL_BATCH_SIZE" \
        > "$EVAL_LOG" 2>&1
    echo "[$(date '+%F %T')] Evaluation complete."
fi

# 解析评估结果
RESULT_FILE="$(ls $OUT_DIR/\[eval-${DATASET}-*attack_results.log 2>/dev/null | head -1 || true)"
BD_ASR="" ; CLEAN_VQA=""
if [ -n "$RESULT_FILE" ]; then
    LINE="$(grep "BACKDOOR ASR" "$RESULT_FILE" | tail -1 || true)"
    BD_ASR="$(echo "$LINE" | grep -oP 'BACKDOOR ASR: \K[0-9.]+' || true)"
    # VQA 评估输出格式: BACKDOOR ASR: XX.XX VQA SCORE: XX.XX | BENIGN ASR: XX.XX VQA SCORE: XX.XX
    # 取最后一个 VQA SCORE（benign/clean）
    CLEAN_VQA="$(echo "$LINE" | grep -oP 'VQA SCORE: \K[0-9.]+' | tail -1 || true)"
    echo "[$(date '+%F %T')] Backdoor model: ASR=$BD_ASR  Clean VQA=$CLEAN_VQA"
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 3: exp1c 伪良性防御（无 ground truth）
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 3: exp1c Pseudo-Benign Defense (no ground truth)"

EXP1C_LOG="$LOG_DIR/exp1c_${STAMP}.log"

if [ -f "$PROJECT_ROOT/$EXP1C_OUT/exp1c_evaluation.json" ]; then
    echo "[$(date '+%F %T')] SKIP exp1c: result exists at $EXP1C_OUT/exp1c_evaluation.json"
else
    echo "[$(date '+%F %T')] Running exp1c (GPU=$FIRST_GPU, k=$EXP1C_K, n=$EXP1C_N_SAMPLES)..."
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir "$OUT_DIR" \
        --output_dir "$EXP1C_OUT" \
        --skip_ground_truth \
        --skip_keep_only \
        --n_samples "$EXP1C_N_SAMPLES" \
        --k "$EXP1C_K" \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EXP1C_EVAL_BS" \
        --train_bs "$EXP1C_TRAIN_BS" \
        --grad_accum "$EXP1C_GRAD_ACCUM" \
        --num_epochs "$EXP1C_NUM_EPOCHS" \
        > "$EXP1C_LOG" 2>&1
    echo "[$(date '+%F %T')] exp1c complete."
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 4: 汇总结果
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 4: Results Summary"

PUR_ASR="" ; PUR_CLEAN_VQA=""
if [ -f "$PROJECT_ROOT/$EXP1C_OUT/exp1c_evaluation.json" ]; then
    read -r PUR_ASR PUR_CLEAN_VQA <<< "$(python3 - "$PROJECT_ROOT/$EXP1C_OUT/exp1c_evaluation.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
ev = d.get("evaluation", {})
base = ev.get("baseline_backdoor", {})
pseudo_keys = [k for k in ev if k.startswith("pseudo_")]
pseudo = ev.get(pseudo_keys[0], {}) if pseudo_keys else {}

pur_asr = pseudo.get("backdoor_asr", "")
pur_cl  = pseudo.get("clean_vqa", pseudo.get("clean_cider", ""))
print(f"{pur_asr} {pur_cl}")
PY
    )"
fi

echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│                 PIPELINE RESULTS                    │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  %-18s %-14s %-14s │\n" "" "ASR" "Clean VQA"
echo "├─────────────────────────────────────────────────────┤"
printf "│  %-18s %-14s %-14s │\n" "Backdoored Model" "${BD_ASR:-N/A}%" "${CLEAN_VQA:-N/A}"
printf "│  %-18s %-14s %-14s │\n" "After exp1c Purif." "${PUR_ASR:-N/A}%" "${PUR_CLEAN_VQA:-N/A}"
echo "├─────────────────────────────────────────────────────┤"
printf "│  Config: lambda=%-5s PR=%-5s k=%-3s n_samples=%-4s │\n" \
    "$VLOOD_LAMBDA_CONST" "$PR" "$EXP1C_K" "$EXP1C_N_SAMPLES"
echo "└─────────────────────────────────────────────────────┘"

END_ALL="$(date +%s)"
DUR=$(( (END_ALL - START_ALL) / 60 ))
echo ""
echo "[$(date '+%F %T')] Total time: ${DUR} minutes"

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tok\n" \
    "$(date '+%F %T')" "$ATTACK_TYPE" "$PR" "$VLOOD_LAMBDA_CONST" \
    "$BD_ASR" "$CLEAN_VQA" "$PUR_ASR" "$PUR_CLEAN_VQA" >> "$TSV"

echo "[$(date '+%F %T')] Done. Logs: $LOG_DIR"
echo "  Train log:  $LOG_DIR/train_${STAMP}.log"
echo "  Eval log:   $LOG_DIR/eval_${STAMP}.log"
echo "  exp1c log:  $LOG_DIR/exp1c_${STAMP}.log"
echo "  Results:    $EXP1C_OUT/exp1c_evaluation.json"

} 2>&1 | tee -a "$MASTER_LOG"
