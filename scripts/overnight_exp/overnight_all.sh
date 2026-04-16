#!/usr/bin/env bash
# 并行启动 TrojVLM (GPU 4) 和 ISSBA (GPU 5) 两个 sweep
# 推荐用法（下班时）：
#   cd /data/YBJ/cleansight
#   nohup bash scripts/overnight_exp/overnight_all.sh > logs/overnight_launch.log 2>&1 &
#   echo $! > /tmp/overnight.pid
#
# 查看进度：
#   tail -f logs/overnight_$(date +%Y%m%d)/trojvlm_main.log
#   tail -f logs/overnight_$(date +%Y%m%d)/issba_main.log
#
# 早上看汇总结果：
#   cat logs/overnight_$(date +%Y%m%d)/summary_trojvlm.tsv | column -t -s $'\t'
#   cat logs/overnight_$(date +%Y%m%d)/summary_issba.tsv   | column -t -s $'\t'

set -u

PROJECT_ROOT="/data/YBJ/cleansight"
cd "$PROJECT_ROOT"

DATE_TAG=$(date +%Y%m%d)
LOG_DIR="$PROJECT_ROOT/logs/overnight_${DATE_TAG}"
mkdir -p "$LOG_DIR"

echo "[$(date +%F\ %T)] Launching overnight sweeps"
echo "  - TrojVLM on GPU 4 -> $LOG_DIR/summary_trojvlm.tsv"
echo "  - ISSBA   on GPU 5 -> $LOG_DIR/summary_issba.tsv"

GPU=4 bash scripts/overnight_exp/run_trojvlm_sweep.sh &
PID_T=$!
GPU=5 bash scripts/overnight_exp/run_issba_sweep.sh &
PID_I=$!

echo "TrojVLM sweep PID=$PID_T"
echo "ISSBA   sweep PID=$PID_I"

# 等两个都结束
wait $PID_T
RC_T=$?
wait $PID_I
RC_I=$?

echo ""
echo "=================================================================="
echo "[$(date +%F\ %T)] ALL DONE"
echo "  TrojVLM RC=$RC_T"
echo "  ISSBA   RC=$RC_I"
echo "=================================================================="
echo ""
echo "===== TrojVLM Summary ====="
column -t -s $'\t' "$LOG_DIR/summary_trojvlm.tsv" 2>/dev/null || cat "$LOG_DIR/summary_trojvlm.tsv"
echo ""
echo "===== ISSBA Summary ====="
column -t -s $'\t' "$LOG_DIR/summary_issba.tsv" 2>/dev/null || cat "$LOG_DIR/summary_issba.tsv"
