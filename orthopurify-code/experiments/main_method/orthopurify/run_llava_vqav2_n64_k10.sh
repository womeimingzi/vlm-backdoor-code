#!/usr/bin/env bash
# Corrected exp1c defense rerun for LLaVA VQAv2.
# Runs only the five non-VLOOD attacks with n_samples=64, k=10, --all_directions.
# Skips ground truth computation (--skip_ground_truth).

set -euo pipefail

GPU="${GPU:-0}"
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
TRAIN_BS="${TRAIN_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
N_SAMPLES=64
K=10

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
MASTER_LOG="$LOG_DIR/orthopurify_vqav2_n64_k10_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_orthopurify_n64_k10.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tk\tbd_asr\tbn_asr\tpseudo_bn_asr\tnote\n" > "$TSV"
fi

if [ ! -f "$PROJECT_ROOT/$BENIGN_DIR/local.json" ] || \
   [ ! -f "$PROJECT_ROOT/$BENIGN_DIR/mmprojector_state_dict.pth" ]; then
    echo "ERROR: Benign model not found at $PROJECT_ROOT/$BENIGN_DIR" >&2
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
echo "[$(date '+%F %T')] orthopurify VQAv2 corrected rerun start (GPU=$GPU, n_samples=$N_SAMPLES, k=$K)"

for attack in "${ORDER[@]}"; do
    subdir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    BACKDOOR_DIR="$CKPT_BASE/$subdir"
    OUT_DIR="experiments/main_method/orthopurify/checkpoints/llava_vqav2_${attack}_n64_k10"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] orthopurify - ${attack} (pr=${pr}, n_samples=${N_SAMPLES}, k=${K})"
    echo "========================================"

    if [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
        echo "  SKIP: checkpoint incomplete at $BACKDOOR_DIR"
        printf "%s\tllava\t%s\t%s\t%s\t\t\t\tmissing_checkpoint\n" \
            "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$K" >> "$TSV"
        continue
    fi

    if [ -f "$PROJECT_ROOT/$OUT_DIR/evaluation.json" ]; then
        python3 - "$PROJECT_ROOT/$OUT_DIR/evaluation.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
cfg = d.get("config", {})
if cfg.get("k") != 10 or cfg.get("n_samples_list") != [64] or not cfg.get("all_directions"):
    raise SystemExit(f"Existing result has wrong config: {cfg}")
PY
        echo "  SKIP: matching result exists at $OUT_DIR/evaluation.json"
        continue
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python experiments/main_method/orthopurify/purify_llava.py \
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
        --skip_keep_only \
        --skip_ground_truth \
        --all_directions

    if [ -f "$PROJECT_ROOT/$OUT_DIR/evaluation.json" ]; then
        python3 - "$PROJECT_ROOT/$OUT_DIR/evaluation.json" "$TSV" "$attack" "$pr" <<'PY'
import json, sys, datetime
path, tsv, attack, pr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(path) as f:
    d = json.load(f)
ev = d.get("evaluation", {})
base = ev.get("baseline_backdoor", {})
pseudo = ev.get("pseudo_n64", {})
metric = pseudo.get("metric_name", base.get("metric_name", ""))
vqa_cl = pseudo.get("clean_vqa", pseudo.get("clean_cider", ""))
ds = d.get("direction_similarity", {})
ds_val = list(ds.values())[0] if ds else {}
n_dirs = f"L1={ds_val.get('n_dirs_L1','?')},L2={ds_val.get('n_dirs_L2','?')}"
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [
    dt, "llava", attack, pr, "10",
    str(base.get("backdoor_asr", "")),
    str(base.get("clean_asr", "")),
    str(pseudo.get("backdoor_asr", "")),
    f"vqav2,n64,all_dirs,metric={metric},VQA_cl={vqa_cl},{n_dirs}",
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
echo "[$(date '+%F %T')] orthopurify VQAv2 corrected rerun done"
} 2>&1 | tee -a "$MASTER_LOG"
