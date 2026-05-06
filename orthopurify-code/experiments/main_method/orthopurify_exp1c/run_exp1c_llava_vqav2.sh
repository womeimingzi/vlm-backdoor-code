#!/usr/bin/env bash
# Serial exp1c defense for LLaVA VQAv2 across 5 non-VLOOD attacks.
# Checkpoints at /home/zzf/data/ZHC/model_checkpoint/llava/{attack}/{attack}/
#
# Usage:
#   GPU=0 bash experiments/main_method/orthopurify_exp1c/run_exp1c_llava_vqav2.sh
#   GPU=3 TEST_NUM=256 bash experiments/main_method/orthopurify_exp1c/run_exp1c_llava_vqav2.sh

set -euo pipefail

GPU="${GPU:-0}"
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
TRAIN_BS="${TRAIN_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
N_SAMPLES="${N_SAMPLES:-50}"
K="${K:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"

CKPT_BASE="/home/zzf/data/ZHC/model_checkpoint/llava"
BENIGN_DIR="model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-ground_truth_benign"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATE_TAG="$(date +%Y%m%d)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/vqav2_defense_${DATE_TAG}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/exp1c_vqav2_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_exp1c.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tk\tbd_asr\tbn_asr\tpseudo_bn_asr\tnote\n" > "$TSV"
fi

# Validate benign model
if [ ! -f "$PROJECT_ROOT/$BENIGN_DIR/mmprojector_state_dict.pth" ]; then
    echo "ERROR: Benign model not found at $PROJECT_ROOT/$BENIGN_DIR" >&2
    echo "Train one first with: bash entrypoints/training/train.sh <GPU> llava-7b adapter vqav2 random random_f replace ground_truth_benign 0.0 2" >&2
    exit 1
fi

declare -A ATTACKS=(
    [badnet]="badnet/badnet"
    [wanet]="wanet/wanet"
    [trojvlm]="trojvlm/trojvlm"
    [issba]="issba/issba"
    [blended]="blended/blended"
)
declare -A ATTACK_PR=(
    [badnet]="0.1"
    [wanet]="0.1"
    [trojvlm]="0.1"
    [issba]="0.2"
    [blended]="0.2"
)
ORDER=(badnet wanet trojvlm issba blended)

{
echo "[$(date '+%F %T')] exp1c VQAv2 batch start (GPU=$GPU)"

for attack in "${ORDER[@]}"; do
    subdir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    BACKDOOR_DIR="$CKPT_BASE/$subdir"
    OUT_DIR="experiments/main_method/orthopurify_exp1c/checkpoints/llava_vqav2_${attack}"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] exp1c — ${attack} (pr=${pr})"
    echo "========================================"

    # Validate
    if [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
        echo "  SKIP: checkpoint incomplete at $BACKDOOR_DIR"
        printf "%s\tllava\t%s\t%s\t%s\t\t\t\tmissing_checkpoint\n" \
            "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$K" >> "$TSV"
        continue
    fi

    # Skip if result exists
    if [ -f "$PROJECT_ROOT/$OUT_DIR/exp1c_evaluation.json" ]; then
        echo "  SKIP: result exists at $OUT_DIR/exp1c_evaluation.json"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --benign_dir "$BENIGN_DIR" \
        --output_dir "$OUT_DIR" \
        --n_samples "$N_SAMPLES" \
        --k "$K" \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --train_bs "$TRAIN_BS" \
        --grad_accum "$GRAD_ACCUM" \
        --num_epochs "$NUM_EPOCHS" \
        --skip_keep_only

    # Log results
    if [ -f "$PROJECT_ROOT/$OUT_DIR/exp1c_evaluation.json" ]; then
        python3 - "$PROJECT_ROOT/$OUT_DIR/exp1c_evaluation.json" "$TSV" "$attack" "$pr" "$K" <<'PY'
import json, sys, datetime
path, tsv, attack, pr, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
with open(path) as f:
    d = json.load(f)
ev = d.get("evaluation", {})
base = ev.get("baseline_backdoor", {})
pseudo_key = [k for k in ev if k.startswith("pseudo_")]
pseudo = ev.get(pseudo_key[0], {}) if pseudo_key else {}
metric = pseudo.get("metric_name", base.get("metric_name", ""))
vqa_cl = pseudo.get("clean_vqa", pseudo.get("clean_cider", ""))
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [
    dt, "llava", attack, pr, k,
    str(base.get("backdoor_asr", "")),
    str(base.get("clean_asr", "")),
    str(pseudo.get("backdoor_asr", "")),
    f"vqav2,n_sample={d.get('config',{}).get('n_samples_list',[50])[0]},metric={metric},VQA_cl={vqa_cl}",
]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY
        echo "[$(date '+%F %T')] Finished ${attack}"
    else
        echo "  ERROR: no result JSON produced"
        printf "%s\tllava\t%s\t%s\t%s\t\t\t\tERROR\n" \
            "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$K" >> "$TSV"
    fi
done

echo ""
echo "[$(date '+%F %T')] exp1c VQAv2 batch done"
} 2>&1 | tee -a "$MASTER_LOG"
