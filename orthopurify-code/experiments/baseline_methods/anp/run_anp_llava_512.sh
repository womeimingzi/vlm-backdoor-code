#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash scripts/run_anp_llava_512.sh
#   bash scripts/run_anp_llava_512.sh <BASE_DIR>
#   bash scripts/run_anp_llava_512.sh <BASE_DIR> --foreground  # 前台运行

BACKGROUND_MODE=1
REMAINING_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --background|-b) BACKGROUND_MODE=1 ;;
    --foreground|-f) BACKGROUND_MODE=0 ;;
    *) REMAINING_ARGS+=("$arg") ;;
  esac
done
set -- "${REMAINING_ARGS[@]}"

BASE_DIR="${1:-model_checkpoint/present_exp/llava-7b/coco/blended-kt-adapter-trojvlm_pr0.01}"
OUT_DIR="$BASE_DIR/anp_purify_midrun"

source /data/lsm/.venvs/qwen3/bin/activate
cd /data/lsm
export PYTHONPATH=/data/lsm:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT_DIR"

BEFORE_LOG="$OUT_DIR/before_full_eval_512.log"
ANP_LOG="$OUT_DIR/anp_midrun.log"
AFTER_LOG="$OUT_DIR/after_full_eval_512.log"
MASTER_LOG="$OUT_DIR/anp_pipeline.log"
PID_FILE="$OUT_DIR/anp_pipeline.pid"

GPU_POOL="0,1,2,3,4,5,6,7"
TARGET_GPU_COUNT=4
MIN_FREE_MB=18000
MAX_UTIL=60
SLEEP_SEC=60
TEST_NUM="${TEST_NUM:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

# ANP algorithm hyperparameters
EPS="${EPS:-0.012}"
PGD_STEPS="${PGD_STEPS:-8}"
THETA_LR="${THETA_LR:-0.06}"
LAM="${LAM:-0.006}"
CLEAN_LOSS_WEIGHT="${CLEAN_LOSS_WEIGHT:-2.5}"
N_ROUNDS="${N_ROUNDS:-1250}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-0.5}"
ANP_TEST_NUM="${ANP_TEST_NUM:-128}"
ANP_ASR_NUM="${ANP_ASR_NUM:-128}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

logn() { echo "[$(date '+%F %T')] $*"; }

gpu_snap() {
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
}

pick_gpus() {
  local cnt=0
  for _ in $(seq 1 120); do
    local pick
    pick=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
      | awk -F',' -v pool="$GPU_POOL" -v minf="$MIN_FREE_MB" -v maxu="$MAX_UTIL" '
        BEGIN { n=split(pool,a,","); for(i=1;i<=n;i++) allow[a[i]]=1 }
        { gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3)
          if (($1 in allow) && $2>=minf && $3<maxu) print $1 "," $2 }' \
      | sort -t',' -k2,2nr | head -n "$TARGET_GPU_COUNT" | cut -d',' -f1 | paste -sd, -)
    cnt=$(echo "$pick" | awk -F',' '{print NF+0}')
    if [ "${cnt:-0}" -ge "$TARGET_GPU_COUNT" ]; then
      export CUDA_VISIBLE_DEVICES="$pick"
      logn "[OK] GPUs: $CUDA_VISIBLE_DEVICES"
      return 0
    fi
    logn "[WAIT] GPU不足，等待..."
    gpu_snap
    sleep "$SLEEP_SEC"
  done
  logn "[FAIL] GPU不足"
  return 1
}

run_eval() {
  local tag="$1"; local dir="$2"; local outf="$3"
  logn "[$tag] Evaluation starting..."
  local _orig_count=$TARGET_GPU_COUNT
  TARGET_GPU_COUNT=1
  pick_gpus || { TARGET_GPU_COUNT=$_orig_count; return 1; }
  TARGET_GPU_COUNT=$_orig_count
  echo "===== $tag | GPUs=$CUDA_VISIBLE_DEVICES | $(date '+%F %T') ====="
  TEST_NUM=$TEST_NUM EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE bash scripts/run_eval.sh "$dir" $CUDA_VISIBLE_DEVICES \
    2>&1 | tee -a "$outf"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then logn "[ERROR] $tag failed"; return 1; fi
  logn "[$tag] DONE"
}

run_anp() {
  logn "[ANP] Purification starting..."
  # ANP needs only 1 GPU, reduce threshold to avoid conflict with eval stages
  local _orig_count=$TARGET_GPU_COUNT
  local _orig_minf=$MIN_FREE_MB
  TARGET_GPU_COUNT=1
  MIN_FREE_MB=15000
  pick_gpus || { TARGET_GPU_COUNT=$_orig_count; MIN_FREE_MB=$_orig_minf; return 1; }
  TARGET_GPU_COUNT=$_orig_count
  MIN_FREE_MB=$_orig_minf
  echo "===== ANP purification | GPUs=$CUDA_VISIBLE_DEVICES | $(date '+%F %T') ====="
  python -u -m vlm_backdoor.evaluation2.anp_purify_llava \
    --poison_local_json "$BASE_DIR/local.json" \
    --dataset coco --test_num $ANP_TEST_NUM --asr_num $ANP_ASR_NUM \
    --device cuda --fp16 --no_eval \
    --eps $EPS --pgd_steps $PGD_STEPS --theta_lr $THETA_LR --lam $LAM \
    --clean_loss_weight $CLEAN_LOSS_WEIGHT --n_rounds $N_ROUNDS --prune_threshold $PRUNE_THRESHOLD \
    --log_interval $LOG_INTERVAL --output_dir "$OUT_DIR" \
    2>&1 | tee -a "$ANP_LOG"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then logn "[ERROR] ANP failed"; return 1; fi
  logn "[ANP] DONE"
}

prep_after() {
  cp "$OUT_DIR/mmprojector_pruned.pth" "$OUT_DIR/mmprojector_state_dict.pth"
  python - "$BASE_DIR" "$OUT_DIR" <<'PYEOF'
import json, sys
base, out = sys.argv[1] + "/local.json", sys.argv[2] + "/local.json"
with open(base) as f: cfg = json.load(f)
cfg["adapter_path"] = sys.argv[2]
with open(out, "w") as f: json.dump(cfg, f, indent=2)
print("wrote:", out)
PYEOF
}

# ============================================================
# 后台模式
# ============================================================
if [ "$BACKGROUND_MODE" -eq 1 ]; then
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "ERROR: already running as PID=$(cat "$PID_FILE")"; exit 1
  fi

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MASTER_LOG_TS="$OUT_DIR/anp_pipeline_${TIMESTAMP}.log"
  : > "$MASTER_LOG_TS"

  logn "[BG] LLaVA ANP pipeline starting..." | tee -a "$MASTER_LOG_TS"
  logn "[BG] Output dir: $OUT_DIR" | tee -a "$MASTER_LOG_TS"
  logn "[BG] Master log: $MASTER_LOG_TS" | tee -a "$MASTER_LOG_TS"

  # 用 Python 写子脚本，传入所有变量作为命令行参数
  INNER=$(mktemp /tmp/anp_inner_XXXXXX.sh)
  python - "$INNER" "$MASTER_LOG_TS" "$BEFORE_LOG" "$ANP_LOG" "$AFTER_LOG" \
    "$OUT_DIR" "$BASE_DIR" "$GPU_POOL" "$MIN_FREE_MB" "$MAX_UTIL" \
    "$TARGET_GPU_COUNT" "$EVAL_BATCH_SIZE" "$TEST_NUM" \
    "$EPS" "$PGD_STEPS" "$THETA_LR" "$LAM" \
    "$CLEAN_LOSS_WEIGHT" "$N_ROUNDS" "$PRUNE_THRESHOLD" \
    "$ANP_TEST_NUM" "$ANP_ASR_NUM" "$LOG_INTERVAL" <<'PYEOF'
import sys
path = sys.argv[1]
mlog = sys.argv[2]
blog = sys.argv[3]
alog = sys.argv[4]
elog = sys.argv[5]
odir = sys.argv[6]
bdir = sys.argv[7]
pool = sys.argv[8]
minf = sys.argv[9]
maxu = sys.argv[10]
ngpu = sys.argv[11]
bsz  = sys.argv[12]
tnum = sys.argv[13]
eps  = sys.argv[14]
pstep= sys.argv[15]
tlr  = sys.argv[16]
lam  = sys.argv[17]
clw  = sys.argv[18]
nrd  = sys.argv[19]
pth  = sys.argv[20]
anum = sys.argv[21]
asnum= sys.argv[22]
lgi  = sys.argv[23]

content = r'''#!/usr/bin/env bash
set -euo pipefail

BLOG="_BLOG_"; ALIST="_ALIST_"; ELIST="_ELIST_"; ODIR="_ODIR_"; BDIR="_BDIR_"; BSZ="_BSZ_"; NGPU="_NGPU_"; TNUM="_TNUM_"; LOG="_LOG_"

logn() { echo "[$(date '+%F %T')] $*" ; }

pick_gpus() {
  local ng="${1:-$NGPU}"
  for _ in $(seq 1 120); do
    local pick cnt
    pick=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
      | awk -F',' -v pool="_POOL_" -v minf="_MINF_" -v maxu="_MAXU_" '
        BEGIN{n=split(pool,a,",");for(i=1;i<=n;i++)allow[a[i]]=1}
        {gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3);if(($1 in allow)&&$2>=minf&&$3<maxu)print $1","$2}' \
      | sort -t',' -k2,2nr | head -n "$ng" | cut -d',' -f1 | paste -sd, -)
    cnt=$(echo "$pick" | awk -F',' 'NF{print NF+0}')
    if [ "$cnt" -ge "$ng" ]; then
      export CUDA_VISIBLE_DEVICES="$pick"
      logn "[OK] GPUs: $CUDA_VISIBLE_DEVICES"
      return 0
    fi
    logn "[WAIT] GPU不足，等待..."
    sleep 60
  done
  logn "[FAIL] GPU不足"
  return 1
}

logn "[STAGE 1/3] BEFORE evaluation starting..."
pick_gpus 1 || exit 1
logn "[STAGE 1/3] Running on GPUs: $CUDA_VISIBLE_DEVICES"
logn "===== BEFORE | GPUs=$CUDA_VISIBLE_DEVICES | $(date '+%F %T') ====="
TEST_NUM=_TNUM_ EVAL_BATCH_SIZE=$BSZ bash scripts/run_eval.sh "$BDIR" $CUDA_VISIBLE_DEVICES \
  2>&1 | tee -a "$BLOG" >> "_LOG_"
if [ ${PIPESTATUS[0]} -ne 0 ]; then logn "[ERROR] BEFORE failed"; exit 1; fi
logn "[STAGE 1/3] DONE"

logn "[STAGE 2/3] ANP purification starting..."
for _ in $(seq 1 120); do
  pick=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v pool="_POOL_" -v minf="15000" -v maxu="_MAXU_" '
      BEGIN{n=split(pool,a,",");for(i=1;i<=n;i++)allow[a[i]]=1}
      {gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3);if(($1 in allow)&&$2>=minf&&$3<maxu)print $1","$2}' \
    | sort -t',' -k2,2nr | head -n 1 | cut -d',' -f1 | paste -sd, -)
  cnt=$(echo "$pick" | awk -F',' 'NF{print NF+0}')
  if [ "$cnt" -ge 1 ]; then
    export CUDA_VISIBLE_DEVICES="$pick"
    logn "[OK] GPUs: $CUDA_VISIBLE_DEVICES"
    break
  fi
  logn "[WAIT] GPU不足，等待..."
  sleep 60
done
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then logn "[FAIL] GPU不足"; exit 1; fi
logn "[STAGE 2/3] Running on GPUs: $CUDA_VISIBLE_DEVICES"
logn "===== ANP purification | GPUs=$CUDA_VISIBLE_DEVICES | $(date '+%F %T') ====="
python -u -m vlm_backdoor.evaluation2.anp_purify_llava \
  --poison_local_json "$BDIR/local.json" \
  --dataset coco --test_num _ANUM_ --asr_num _ASNUM_ \
  --device cuda --fp16 --no_eval \
  --eps _EPS_ --pgd_steps _PSTEP_ --theta_lr _TLR_ --lam _LAM_ \
  --clean_loss_weight _CLW_ --n_rounds _NRD_ --prune_threshold _PTH_ \
  --log_interval _LGI_ --output_dir "$ODIR" \
  2>&1 | tee -a "$ALIST" >> "_LOG_"
if [ ${PIPESTATUS[0]} -ne 0 ]; then logn "[ERROR] ANP failed"; exit 1; fi
logn "[STAGE 2/3] DONE"

cp "$ODIR/mmprojector_pruned.pth" "$ODIR/mmprojector_state_dict.pth"
python - "$BDIR" "$ODIR" <<'PYSUB'
import json,sys
with open(sys.argv[1]+"/local.json") as f: cfg=json.load(f)
cfg["adapter_path"]=sys.argv[2]
with open(sys.argv[2]+"/local.json","w") as f: json.dump(cfg,f,indent=2)
print("wrote:",sys.argv[2]+"/local.json")
PYSUB

logn "[STAGE 3/3] AFTER evaluation starting..."
pick_gpus 1 || exit 1
logn "[STAGE 3/3] Running on GPUs: $CUDA_VISIBLE_DEVICES"
logn "===== AFTER | GPUs=$CUDA_VISIBLE_DEVICES | $(date '+%F %T') ====="
TEST_NUM=_TNUM_ EVAL_BATCH_SIZE=$BSZ bash scripts/run_eval.sh "$ODIR" $CUDA_VISIBLE_DEVICES \
  2>&1 | tee -a "$ELIST" >> "_LOG_"
if [ ${PIPESTATUS[0]} -ne 0 ]; then logn "[ERROR] AFTER failed"; exit 1; fi
logn "[STAGE 3/3] DONE"

logn "===== FINAL METRICS ====="
logn "[DONE] All stages completed successfully!"
'''

content = content \
    .replace('_NGPU_', str(ngpu)) \
    .replace('_LOG_', mlog) \
    .replace('_POOL_', pool) \
    .replace('_MINF_', minf) \
    .replace('_MAXU_', maxu) \
    .replace('_BLOG_', blog) \
    .replace('_ALIST_', alog) \
    .replace('_ELIST_', elog) \
    .replace('_ODIR_', odir) \
    .replace('_BDIR_', bdir) \
    .replace('_BSZ_', bsz) \
    .replace('_TNUM_', tnum) \
    .replace('_EPS_', eps) \
    .replace('_PSTEP_', pstep) \
    .replace('_TLR_', tlr) \
    .replace('_LAM_', lam) \
    .replace('_CLW_', clw) \
    .replace('_NRD_', nrd) \
    .replace('_PTH_', pth) \
    .replace('_ANUM_', anum) \
    .replace('_ASNUM_', asnum) \
    .replace('_LGI_', lgi)

with open(path, 'w') as f:
    f.write(content)
print(path, flush=True)
PYEOF

  chmod +x "$INNER"

  # 子脚本内部通过 logn/tee 写 master log；外层重定向确保即使 tee 没覆盖到的输出也能被记录
  bash "$INNER" >> "$MASTER_LOG_TS" 2>&1 &
  pid=$!
  echo $pid > "$PID_FILE"
  ln -snf "$MASTER_LOG_TS" "$MASTER_LOG"
  echo "[DONE] LLaVA ANP pipeline launched (PID=$pid)"
  echo "[DONE] Monitor: tail -f $MASTER_LOG_TS"
  exit 0
fi

# ============================================================
# 前台模式
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG_TS="$OUT_DIR/anp_pipeline_${TIMESTAMP}.log"
: > "$MASTER_LOG_TS"

logn "[FG] LLaVA ANP pipeline starting..." | tee -a "$MASTER_LOG_TS"
logn "[FG] Master log: $MASTER_LOG_TS" | tee -a "$MASTER_LOG_TS"

# Stage 1
logn "[STAGE 1/3] BEFORE evaluation..." | tee -a "$MASTER_LOG_TS"
run_eval "BEFORE" "$BASE_DIR" "$BEFORE_LOG" 2>&1 | tee -a "$MASTER_LOG_TS"

# Stage 2
logn "[STAGE 2/3] ANP purification..." | tee -a "$MASTER_LOG_TS"
run_anp 2>&1 | tee -a "$MASTER_LOG_TS"

# Stage 3
prep_after
logn "[STAGE 3/3] AFTER evaluation..." | tee -a "$MASTER_LOG_TS"
run_eval "AFTER" "$OUT_DIR" "$AFTER_LOG" 2>&1 | tee -a "$MASTER_LOG_TS"

logn "[DONE] All stages completed!" | tee -a "$MASTER_LOG_TS"
logn "[LOG] $MASTER_LOG_TS" | tee -a "$MASTER_LOG_TS"
