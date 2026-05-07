#!/usr/bin/env bash
# Serial exp9 ANP defense for LLaVA VQAv2 across 5 non-VLOOD attacks.
# Checkpoints at /home/zzf/data/ZHC/model_checkpoint/llava/{attack}/{attack}/
#
# Two-phase per attack:
#   Phase 1: ANP purification (anp_purify_llava.py with --no_eval)
#   Phase 2: Evaluation via llava_evaluator.py (VQA score)
#
# Usage:
#   GPU=4 bash experiments/baseline_methods/anp/run_anp_llava_vqav2.sh

set -euo pipefail

GPU="${GPU:-4}"
TEST_NUM="${TEST_NUM:-512}"
N_SAMPLE="${N_SAMPLE:-500}"
N_ROUNDS="${N_ROUNDS:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

EPS="${EPS:-0.012}"
PGD_STEPS="${PGD_STEPS:-8}"
THETA_LR="${THETA_LR:-0.06}"
LAM="${LAM:-0.006}"
CLEAN_LOSS_WEIGHT="${CLEAN_LOSS_WEIGHT:-2.5}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-0.5}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

CKPT_BASE="/home/zzf/data/ZHC/model_checkpoint/llava"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"

source /data/YBJ/GraduProject/venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DATE_TAG="$(date +%Y%m%d)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/vqav2_defense_${DATE_TAG}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/exp9_anp_vqav2_${STAMP}.log"
TSV="$PROJECT_ROOT/logs/vqav2_anp.tsv"
if [ ! -f "$TSV" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tepoch\tbd_asr\tafter_asr\tnote\n" > "$TSV"
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

latest_eval_log() {
    local dir="$1"
    ls -t "$dir"/\[eval-vqav2-*attack_results.log 2>/dev/null | head -1 || true
}

{
echo "[$(date '+%F %T')] ANP VQAv2 batch start (GPU=$GPU)"

for attack in "${ORDER[@]}"; do
    subdir="${ATTACKS[$attack]}"
    pr="${ATTACK_PR[$attack]}"
    BACKDOOR_DIR="$CKPT_BASE/$subdir"
    OUT_DIR="$PROJECT_ROOT/experiments/baseline_methods/anp/checkpoints/llava_vqav2_${attack}"

    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] ANP — ${attack} (pr=${pr})"
    echo "========================================"

    if [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
        echo "  SKIP: checkpoint incomplete at $BACKDOOR_DIR"
        printf "%s\tllava\t%s\t%s\t%s\t2\t\t\tmissing_checkpoint\n" \
            "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$N_SAMPLE" >> "$TSV"
        continue
    fi

    mkdir -p "$OUT_DIR"

    # Phase 1: ANP purification
    if [ -f "$OUT_DIR/mmprojector_pruned.pth" ]; then
        echo "  Phase 1 SKIP: pruned projector exists."
    else
        echo "  Phase 1: Running ANP purification..."
        ANP_LOG="$LOG_DIR/exp9_anp_vqav2_${attack}_purify_${STAMP}.log"

        CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/baseline_methods/anp/anp_purify_llava.py \
            --poison_local_json "$BACKDOOR_DIR/local.json" \
            --poison_adapter_path "$BACKDOOR_DIR" \
            --dataset vqav2 \
            --test_num "$N_SAMPLE" \
            --asr_num "$N_SAMPLE" \
            --batch_size 1 \
            --num_workers 0 \
            --device cuda:0 \
            --fp16 \
            --no_eval \
            --eps "$EPS" \
            --pgd_steps "$PGD_STEPS" \
            --theta_lr "$THETA_LR" \
            --lam "$LAM" \
            --clean_loss_weight "$CLEAN_LOSS_WEIGHT" \
            --n_rounds "$N_ROUNDS" \
            --prune_threshold "$PRUNE_THRESHOLD" \
            --log_interval "$LOG_INTERVAL" \
            --output_dir "$OUT_DIR" \
            > "$ANP_LOG" 2>&1

        if [ ! -f "$OUT_DIR/mmprojector_pruned.pth" ]; then
            echo "  ERROR: ANP purification failed. See $ANP_LOG"
            printf "%s\tllava\t%s\t%s\t%s\t2\t\t\tpurify_failed\n" \
                "$(date '+%-m.%-d,%Y')" "$attack" "$pr" "$N_SAMPLE" >> "$TSV"
            continue
        fi
        echo "  Phase 1 done."
    fi

    # Prepare for evaluation: copy pruned weights and create local.json
    cp "$OUT_DIR/mmprojector_pruned.pth" "$OUT_DIR/mmprojector_state_dict.pth"
    python3 - "$BACKDOOR_DIR" "$OUT_DIR" <<'PY'
import json, sys
base_dir, out_dir = sys.argv[1], sys.argv[2]
with open(base_dir + "/local.json") as f:
    cfg = json.load(f)
cfg["adapter_path"] = out_dir
cfg["output_dir_root_name"] = out_dir
with open(out_dir + "/local.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY

    # Phase 2: Evaluation
    if [ -n "$(latest_eval_log "$OUT_DIR")" ]; then
        echo "  Phase 2 SKIP: evaluation result exists."
    else
        echo "  Phase 2: Evaluating purified model..."
        AFTER_LOG="$LOG_DIR/exp9_anp_vqav2_${attack}_after_${STAMP}.log"

        CUDA_VISIBLE_DEVICES="$GPU" python vlm_backdoor/evaluation/llava_evaluator.py \
            --local_json "$OUT_DIR/local.json" \
            --test_num "$TEST_NUM" \
            --batch_size "$EVAL_BATCH_SIZE" \
            > "$AFTER_LOG" 2>&1

        echo "  Phase 2 done."
    fi

    # Log results
    python3 - "$OUT_DIR" "$TSV" "$attack" "$pr" "$N_SAMPLE" <<'PY'
import glob, re, sys, datetime, os
out_dir, tsv, attack, pr, n_sample = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
def parse(path):
    if not path:
        return "", "", ""
    with open(path, errors="ignore") as f:
        lines = [l.strip() for l in f if "BACKDOOR ASR" in l]
    line = lines[-1] if lines else ""
    asr = re.search(r"BACKDOOR ASR:\s*([0-9.]+)", line)
    vqa_all = re.findall(r"VQA SCORE:\s*([0-9.]+)", line)
    vqa_bd = vqa_all[0] if len(vqa_all) >= 1 else ""
    vqa_bn = vqa_all[1] if len(vqa_all) >= 2 else vqa_bd
    return (asr.group(1) if asr else "", vqa_bn, vqa_bd)
def latest(d):
    files = sorted(glob.glob(os.path.join(d, "[[]eval-vqav2-*attack_results.log")),
                   key=os.path.getmtime, reverse=True)
    return files[0] if files else ""
after_asr, after_vqa_bn, after_vqa_bd = parse(latest(out_dir))
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
note = f"rounds={n_sample},VQA_clean={after_vqa_bn},VQA_bd={after_vqa_bd}"
line = [dt, "llava", attack, pr, n_sample, "2", "", after_asr, note]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("logged:", "\t".join(line))
PY

    echo "[$(date '+%F %T')] Finished ${attack}"
done

echo ""
echo "[$(date '+%F %T')] ANP VQAv2 batch done"
} 2>&1 | tee -a "$MASTER_LOG"
