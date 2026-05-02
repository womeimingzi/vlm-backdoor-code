#!/usr/bin/env bash
# Serial defense experiments for VQAv2 LLaVA VLOOD (lambda=0.8, random_insert, pr=0.2).
# Runs exp7 → exp8 → exp9 → exp1c in sequence on a single GPU.
#
# Config:
#   exp7: n_sample=1000, epochs=2
#   exp8: n_sample=1000, epochs=2
#   exp9: n_sample=500,  n_rounds=1000 (~2 epochs)
#   exp1c: n_samples=64, k=10, --all_directions, epochs=2
#
# Usage:
#   GPU=0 bash scripts/vqav2_exp/run_vqav2_llava_vlood_defense_serial.sh

set -euo pipefail

GPU="${GPU:-0}"
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

BACKDOOR_DIR="${BACKDOOR_DIR:-model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-vlood_random_insert_pr0.2_lambda0.8}"
BENIGN_DIR="${BENIGN_DIR:-model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-ground_truth_benign}"
CKPT_TAG="llava_vqav2_vlood_random_insert_pr0.2_lambda0.8"

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
MASTER_LOG="$LOG_DIR/vlood_lambda0.8_serial_${STAMP}.log"

banner() { echo ""; echo "$(printf '═%.0s' $(seq 1 70))"; echo "  $1"; echo "$(printf '═%.0s' $(seq 1 70))"; }

latest_eval_log() {
    local dir="$1"
    ls -t "$dir"/\[eval-vqav2-*attack_results.log 2>/dev/null | head -1 || true
}

{
START_ALL="$(date +%s)"
echo "[$(date '+%F %T')] VLOOD Defense Serial Pipeline START"
echo "  GPU=$GPU"
echo "  BACKDOOR_DIR=$BACKDOOR_DIR"

if [ ! -f "$BACKDOOR_DIR/local.json" ] || [ ! -f "$BACKDOOR_DIR/mmprojector_state_dict.pth" ]; then
    echo "ERROR: Backdoor checkpoint not found at $BACKDOOR_DIR" >&2
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: exp7 — Fine-Tuning Recovery (n_sample=1000, 2 epochs)
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 1: exp7 Fine-Tuning Recovery (n_sample=1000, epochs=2)"

EXP7_OUT="exps/exp7_finetune_recovery/checkpoints/${CKPT_TAG}"
EXP7_LOG="$LOG_DIR/exp7_vlood_lambda0.8_${STAMP}.log"

if [ -f "$PROJECT_ROOT/$EXP7_OUT/exp7_results.json" ]; then
    echo "[$(date '+%F %T')] SKIP exp7: result exists at $EXP7_OUT/exp7_results.json"
else
    echo "[$(date '+%F %T')] Running exp7..."
    mkdir -p "$PROJECT_ROOT/$EXP7_OUT"
    CUDA_VISIBLE_DEVICES="$GPU" python exps/exp7_finetune_recovery/exp7_finetune_recovery.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --n_sample_list 1000 \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --output_dir "$EXP7_OUT" \
        --skip_baseline_eval \
        > "$EXP7_LOG" 2>&1
    echo "[$(date '+%F %T')] exp7 done."
fi

# Log exp7 result
TSV_FT="$PROJECT_ROOT/logs/vqav2_ft.tsv"
if [ ! -f "$TSV_FT" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tbd_asr\tafter_asr\tnote\n" > "$TSV_FT"
fi
if [ -f "$PROJECT_ROOT/$EXP7_OUT/exp7_results.json" ]; then
    python3 - "$PROJECT_ROOT/$EXP7_OUT/exp7_results.json" "$TSV_FT" <<'PY'
import json, sys, datetime
path, tsv = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
r = d.get("results", {})
after = r.get("n1000", {})
metric = after.get("metric_name", "")
vqa_after = after.get("clean_vqa", after.get("clean_cider", ""))
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [dt, "llava", "vlood_ri_l0.8", "0.2", "1000",
        "", str(after.get("backdoor_asr", "")),
        f"{metric}_after={vqa_after},epochs=2,baseline_skipped"]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("exp7 logged:", "\t".join(line))
PY
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: exp8 — Fine-Pruning (n_sample=1000, 2 epochs)
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 2: exp8 Fine-Pruning (n_sample=1000, epochs=2)"

EXP8_OUT="exps/exp8_fine_pruning/checkpoints/${CKPT_TAG}"
EXP8_LOG="$LOG_DIR/exp8_vlood_lambda0.8_${STAMP}.log"

if [ -f "$PROJECT_ROOT/$EXP8_OUT/exp8_results.json" ]; then
    echo "[$(date '+%F %T')] SKIP exp8: result exists at $EXP8_OUT/exp8_results.json"
else
    echo "[$(date '+%F %T')] Running exp8..."
    CUDA_VISIBLE_DEVICES="$GPU" python exps/exp8_fine_pruning/exp8_fine_pruning.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --n_sample 1000 \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --cider_threshold "${CIDER_THRESHOLD:-0.025}" \
        --max_ratio "${MAX_RATIO:-0.95}" \
        --search_step "${SEARCH_STEP:-0.10}" \
        --output_dir "$EXP8_OUT" \
        --skip_baseline_eval \
        > "$EXP8_LOG" 2>&1
    echo "[$(date '+%F %T')] exp8 done."
fi

# Log exp8 result
TSV_FP="$PROJECT_ROOT/logs/vqav2_fp.tsv"
if [ ! -f "$TSV_FP" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tbd_asr\tafter_asr\tnote\n" > "$TSV_FP"
fi
if [ -f "$PROJECT_ROOT/$EXP8_OUT/exp8_results.json" ]; then
    python3 - "$PROJECT_ROOT/$EXP8_OUT/exp8_results.json" "$TSV_FP" <<'PY'
import json, sys, datetime
path, tsv = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
r = d.get("results", {})
after = r.get("fine_pruning", {})
cfg = d.get("config", {})
metric = after.get("metric_name", "")
vqa_after = after.get("clean_vqa", after.get("clean_cider", ""))
prune_ratio = cfg.get("prune_ratio", "")
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [dt, "llava", "vlood_ri_l0.8", "0.2", "1000",
        "", str(after.get("backdoor_asr", "")),
        f"prune={prune_ratio},{metric}_after={vqa_after},epochs=2,baseline_skipped"]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("exp8 logged:", "\t".join(line))
PY
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 3: exp9 — ANP Defense (n_sample=500, n_rounds=1000 ≈ 2 epochs)
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 3: exp9 ANP Defense (n_sample=500, n_rounds=1000)"

N_SAMPLE_ANP=500
N_ROUNDS_ANP=1000
EXP9_OUT="exps/exp9_anp/checkpoints/${CKPT_TAG}"
ANP_LOG="$LOG_DIR/exp9_anp_vlood_lambda0.8_purify_${STAMP}.log"
AFTER_LOG="$LOG_DIR/exp9_anp_vlood_lambda0.8_after_${STAMP}.log"

mkdir -p "$PROJECT_ROOT/$EXP9_OUT"

# Phase 3a: ANP purification
if [ -f "$PROJECT_ROOT/$EXP9_OUT/mmprojector_pruned.pth" ]; then
    echo "[$(date '+%F %T')] SKIP ANP purification: pruned projector exists."
else
    echo "[$(date '+%F %T')] Running ANP purification..."
    CUDA_VISIBLE_DEVICES="$GPU" python -u exps/exp9_anp/anp_purify_llava.py \
        --poison_local_json "$BACKDOOR_DIR/local.json" \
        --dataset vqav2 \
        --test_num "$N_SAMPLE_ANP" \
        --asr_num "$N_SAMPLE_ANP" \
        --batch_size 1 \
        --num_workers 0 \
        --device cuda:0 \
        --fp16 \
        --no_eval \
        --eps "${EPS:-0.012}" \
        --pgd_steps "${PGD_STEPS:-8}" \
        --theta_lr "${THETA_LR:-0.06}" \
        --lam "${LAM:-0.006}" \
        --clean_loss_weight "${CLEAN_LOSS_WEIGHT:-2.5}" \
        --n_rounds "$N_ROUNDS_ANP" \
        --prune_threshold "${PRUNE_THRESHOLD:-0.5}" \
        --log_interval "${LOG_INTERVAL:-50}" \
        --output_dir "$EXP9_OUT" \
        > "$ANP_LOG" 2>&1

    if [ ! -f "$PROJECT_ROOT/$EXP9_OUT/mmprojector_pruned.pth" ]; then
        echo "[$(date '+%F %T')] ERROR: ANP purification failed. See $ANP_LOG" >&2
    fi
fi

# Phase 3b: Prepare and evaluate
if [ -f "$PROJECT_ROOT/$EXP9_OUT/mmprojector_pruned.pth" ]; then
    cp "$PROJECT_ROOT/$EXP9_OUT/mmprojector_pruned.pth" "$PROJECT_ROOT/$EXP9_OUT/mmprojector_state_dict.pth"
    python3 - "$BACKDOOR_DIR" "$EXP9_OUT" <<'PY'
import json, sys
base_dir, out_dir = sys.argv[1], sys.argv[2]
with open(base_dir + "/local.json") as f:
    cfg = json.load(f)
cfg["adapter_path"] = out_dir
cfg["output_dir_root_name"] = out_dir
with open(out_dir + "/local.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY

    if [ -z "$(latest_eval_log "$PROJECT_ROOT/$EXP9_OUT")" ]; then
        echo "[$(date '+%F %T')] Evaluating ANP-purified model..."
        CUDA_VISIBLE_DEVICES="$GPU" python vlm_backdoor/evaluation/llava_evaluator.py \
            --local_json "$EXP9_OUT/local.json" \
            --test_num "$TEST_NUM" \
            --batch_size "$EVAL_BATCH_SIZE" \
            > "$AFTER_LOG" 2>&1
        echo "[$(date '+%F %T')] exp9 evaluation done."
    else
        echo "[$(date '+%F %T')] SKIP exp9 evaluation: result exists."
    fi
fi

# Log exp9 result
TSV_ANP="$PROJECT_ROOT/logs/vqav2_anp.tsv"
if [ ! -f "$TSV_ANP" ]; then
    printf "time\tmodel\tattack\tpr\tn_sample\tepoch\tbd_asr\tafter_asr\tnote\n" > "$TSV_ANP"
fi
python3 - "$EXP9_OUT" "$TSV_ANP" <<'PY'
import glob, re, sys, datetime, os
out_dir, tsv = sys.argv[1], sys.argv[2]
def parse(path):
    if not path:
        return "", ""
    with open(path, errors="ignore") as f:
        lines = [l.strip() for l in f if "BACKDOOR ASR" in l]
    line = lines[-1] if lines else ""
    asr = re.search(r"BACKDOOR ASR:\s*([0-9.]+)", line)
    vqa = re.search(r"VQA SCORE:\s*([0-9.]+)", line)
    return (asr.group(1) if asr else "", vqa.group(1) if vqa else "")
def latest(d):
    files = sorted(glob.glob(os.path.join(d, "[[]eval-vqav2-*attack_results.log")),
                   key=os.path.getmtime, reverse=True)
    return files[0] if files else ""
after_asr, after_vqa = parse(latest(out_dir))
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
note = f"rounds=1000,VQA_after={after_vqa}"
line = [dt, "llava", "vlood_ri_l0.8", "0.2", "500", "2", "", after_asr, note]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("exp9 logged:", "\t".join(line))
PY

echo "[$(date '+%F %T')] exp9 done."

# ══════════════════════════════════════════════════════════════════════════
# Phase 4: exp1c — Pseudo-Benign Projection (n_samples=64, k=10, --all_directions)
# ══════════════════════════════════════════════════════════════════════════
banner "Phase 4: exp1c Pseudo-Benign (n_samples=64, k=10, all_directions)"

EXP1C_OUT="exps/exp1c_pseudo_benign/checkpoints/${CKPT_TAG}_n64_k10"
EXP1C_LOG="$LOG_DIR/exp1c_vlood_lambda0.8_${STAMP}.log"

# Ensure benign checkpoint exists (train if needed)
if [ ! -f "$PROJECT_ROOT/$BENIGN_DIR/local.json" ] || \
   [ ! -f "$PROJECT_ROOT/$BENIGN_DIR/mmprojector_state_dict.pth" ]; then
    echo "[$(date '+%F %T')] Training VQAv2 benign projector for exp1c..."
    env \
        NCCL_IB_DISABLE=1 \
        NCCL_P2P_DISABLE=1 \
        TORCH_NCCL_ENABLE_MONITORING=0 \
        PER_DEVICE_TRAIN_BS="${BENIGN_PER_DEVICE_TRAIN_BS:-1}" \
        GRAD_ACCUM_STEPS="${BENIGN_GRAD_ACCUM_STEPS:-16}" \
        DS_CONFIG="${DS_CONFIG:-configs/ds_zero2_fp16_stable.json}" \
        bash scripts/train.sh "$GPU" llava-7b adapter vqav2 \
            random random_f replace ground_truth_benign 0.0 2
else
    echo "[$(date '+%F %T')] Benign checkpoint exists; skip benign training."
fi

if [ -f "$PROJECT_ROOT/$EXP1C_OUT/exp1c_evaluation.json" ]; then
    echo "[$(date '+%F %T')] SKIP exp1c: result exists at $EXP1C_OUT/exp1c_evaluation.json"
else
    echo "[$(date '+%F %T')] Running exp1c..."
    CUDA_VISIBLE_DEVICES="$GPU" python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir "$BACKDOOR_DIR" \
        --benign_dir "$BENIGN_DIR" \
        --output_dir "$EXP1C_OUT" \
        --n_samples 64 \
        --k 10 \
        --test_num "$TEST_NUM" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --train_bs "${TRAIN_BS:-1}" \
        --grad_accum "${GRAD_ACCUM:-16}" \
        --num_epochs 2 \
        --skip_keep_only \
        --skip_ground_truth \
        --all_directions \
        > "$EXP1C_LOG" 2>&1
    echo "[$(date '+%F %T')] exp1c done."
fi

# Log exp1c result
TSV_1C="$PROJECT_ROOT/logs/vqav2_exp1c.tsv"
if [ ! -f "$TSV_1C" ]; then
    printf "time\tmodel\tattack\tpr\tk\tbd_asr\tbn_asr\tpseudo_bn_asr\tnote\n" > "$TSV_1C"
fi
if [ -f "$PROJECT_ROOT/$EXP1C_OUT/exp1c_evaluation.json" ]; then
    python3 - "$PROJECT_ROOT/$EXP1C_OUT/exp1c_evaluation.json" "$TSV_1C" <<'PY'
import json, sys, datetime
path, tsv = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
ev = d.get("evaluation", {})
base = ev.get("baseline_backdoor", {})
pseudo_keys = [k for k in ev if k.startswith("pseudo_")]
pseudo = ev.get(pseudo_keys[0], {}) if pseudo_keys else {}
metric = pseudo.get("metric_name", base.get("metric_name", ""))
vqa_cl = pseudo.get("clean_vqa", pseudo.get("clean_cider", ""))
ds = d.get("direction_similarity", {})
ds_val = list(ds.values())[0] if ds else {}
n_dirs = f"L1={ds_val.get('n_dirs_L1','?')},L2={ds_val.get('n_dirs_L2','?')}"
dt = datetime.datetime.now().strftime("%-m.%-d,%Y")
line = [
    dt, "llava", "vlood_ri_l0.8", "0.2", "10",
    str(base.get("backdoor_asr", "")),
    str(base.get("clean_asr", "")),
    str(pseudo.get("backdoor_asr", "")),
    f"vqav2,n64,all_dirs,metric={metric},VQA_cl={vqa_cl},{n_dirs}",
]
with open(tsv, "a") as f:
    f.write("\t".join(line) + "\n")
print("exp1c logged:", "\t".join(line))
PY
fi

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
banner "All Phases Complete"

END_ALL="$(date +%s)"
DUR=$(( (END_ALL - START_ALL) / 60 ))
echo "[$(date '+%F %T')] Total time: ${DUR} minutes"
echo "  exp7 output:  $EXP7_OUT"
echo "  exp8 output:  $EXP8_OUT"
echo "  exp9 output:  $EXP9_OUT"
echo "  exp1c output: $EXP1C_OUT"
echo "  Logs:         $LOG_DIR"

} 2>&1 | tee -a "$MASTER_LOG"
