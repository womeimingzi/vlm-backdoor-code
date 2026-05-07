#!/usr/bin/env bash
# Serial exp7 fine-tuning recovery for LLaVA VQAv2 across 5 non-VLOOD attacks.
# Checkpoints at /home/zzf/data/ZHC/model_checkpoint/llava/{attack}/{attack}/
#
# Usage:
#   GPU=2 bash experiments/baseline_methods/finetune_recovery/run_exp7_llava_vqav2.sh

set -euo pipefail

GPU="${GPU:-2}"
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
N_SAMPLE="${N_SAMPLE:-1000}"

CKPT_BASE="/home/zzf/data/ZHC/model_checkpoint/llava"

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
MASTER_LOG="$LOG_DIR/exp7_vqav2_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_ft.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tbd_asr\tafter_asr\tnote\n" > "$TSV"
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
echo "[$(date '+%F %T')]" "ft_recovery VQAv2 batch start (GPU=$GPU)"

for attack in "${ORDER[@]}"; do
    subdir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    BACKDOOR_DIR="$CKPT_BASE/$subdir"
    OUT_DIR="experiments/baseline_methods/finetune_recovery/checkpoints/llava_vqav2_${attack}"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')]" "ft_recovery — ${attack} (pr=${pr})"
    echo "========================================"

    if [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
        echo "  SKIP: checkpoint incomplete at $BACKDOOR_DIR"
        printf "%s\tllava\t%s\t%s\t%s\t\t\tmissing_checkpoint\n" \
            "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$N_SAMPLE" >> "$TSV"
        continue
    fi

    if [ -f "$PROJECT_ROOT/$OUT_DIR/finetune_recovery_results.json" ]; then
        echo "  SKIP: result exists at $OUT_DIR/finetune_recovery_results.json"
        continue
    fi

    mkdir -p "$PROJECT_ROOT/$OUT_DIR"

    CUDA_VISIBLE_DEVICES="$GPU" python experiments/baseline_methods/finetune_recovery/finetune_recovery.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --n_sample_list "$N_SAMPLE" \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --output_dir "$OUT_DIR" \
        --skip_baseline_eval

    # Log results
    RESULT_JSON="$PROJECT_ROOT/$OUT_DIR/finetune_recovery_results.json"
    if [ -f "$RESULT_JSON" ]; then
        python3 - "$RESULT_JSON" "$TSV" "$attack" "$pr" "$N_SAMPLE" <<'PY'
import json, sys, datetime
path, tsv, attack, pr, n_sample = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
with open(path) as f:
    d = json.load(f)
r = d.get("results", {})
after = r.get(f"n{n_sample}", {})
metric = after.get("metric_name", "")
vqa_after = after.get("clean_vqa", after.get("clean_cider", ""))
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [dt, "llava", attack, pr, n_sample,
        "", str(after.get("backdoor_asr", "")),
        f"{metric}_after={vqa_after},epochs=2,baseline_skipped"]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY
        echo "[$(date '+%F %T')] Finished ${attack}"
    else
        echo "  ERROR: no result JSON produced"
        printf "%s\tllava\t%s\t%s\t%s\t\t\tERROR\n" \
            "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$N_SAMPLE" >> "$TSV"
    fi
done

echo ""
echo "[$(date '+%F %T')]" "ft_recovery VQAv2 batch done"
} 2>&1 | tee -a "$MASTER_LOG"
