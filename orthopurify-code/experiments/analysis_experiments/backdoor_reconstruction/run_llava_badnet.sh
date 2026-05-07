#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Backdoor Reconstruction: Backdoor Reconstruction — BadNet (LLaVA-1.5-7B + COCO)
#
# Continuous training on 500 poisoned samples (1 epoch), eval at
# 10/20/50/100/200/500 sample checkpoints. 3 defenses serial.
#
# Launch:
#   tmux new-session -d -s exp12 \
#     'bash experiments/analysis_experiments/backdoor_reconstruction/run_exp12_llava_badnet.sh 2>&1 | \
#      tee experiments/analysis_experiments/backdoor_reconstruction/run_exp12_badnet.log && \
#      bash experiments/analysis_experiments/backdoor_reconstruction/run_exp12_llava_trojvlm.sh 2>&1 | \
#      tee experiments/analysis_experiments/backdoor_reconstruction/run_exp12_trojvlm.log'
# ═══════════════════════════════════════════════════════════════════════════════
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT="experiments/analysis_experiments/backdoor_reconstruction/backdoor_reconstruction.py"
BD_DIR="model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1"

# effective_bs = 5 * 2 = 10, divides 10/20/50/100/200/500
BATCH_SIZE=2
GRAD_ACCUM=5
LR=2e-5
EVAL_BS=8
TEST_NUM=512
N_TOTAL=500
EVAL_AT="10,20,50,100,200,500"

FP_WEIGHTS="experiments/baseline_methods/fine_pruning/checkpoints/llava_random-adapter-badnet_pr0.1/finepruned_mmprojector_state_dict.pth"
CLP_WEIGHTS="experiments/baseline_methods/clp/results/llava_random-adapter-badnet_pr0.1/mmprojector_state_dict.pth"
OURS_WEIGHTS="experiments/main_method/orthopurify/checkpoints/llava_random-adapter-badnet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"

for f in "$FP_WEIGHTS" "$CLP_WEIGHTS" "$OURS_WEIGHTS"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

run_defense() {
    local weights="$1" name="$2"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [BadNet] ${name} | $(date)"
    echo "════════════════════════════════════════════════════════════"
    python "$SCRIPT" \
        --defended_weights "$weights" \
        --backdoor_dir "$BD_DIR" \
        --defense_name "$name" \
        --n_total "$N_TOTAL" \
        --eval_at "$EVAL_AT" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --eval_batch_size "$EVAL_BS" \
        --test_num "$TEST_NUM"
}

echo "Backdoor Reconstruction BadNet reconstruction | Start: $(date)"
run_defense "$FP_WEIGHTS"   "FP"
run_defense "$CLP_WEIGHTS"  "CLP"
run_defense "$OURS_WEIGHTS" "Ours"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  BadNet complete. $(date)"
echo "═══════════════════════════════════════════════════════════"
python3 -c "
import json, os
base = 'experiments/analysis_experiments/backdoor_reconstruction/results/random-adapter-badnet_pr0.1'
print()
print(f'{\"Defense\":<8}' + ''.join(f'{\"n=\"+str(n):>10}' for n in [10,20,50,100,200,500]))
print('-' * 68)
for d in ['FP', 'CLP', 'Ours']:
    f = os.path.join(base, d, 'reconstruction_results.json')
    if not os.path.exists(f): continue
    r = json.load(open(f))['results']
    line = f'{d:<8}'
    for n in [10,20,50,100,200,500]:
        m = r.get(f'n{n}', {})
        asr = m.get('backdoor_asr', '-')
        line += f'{asr:>10}' if asr == '-' else f'{asr:>9.1f}%'
    print(line)
print()
for d in ['FP', 'CLP', 'Ours']:
    f = os.path.join(base, d, 'reconstruction_results.json')
    if not os.path.exists(f): continue
    r = json.load(open(f))['results']
    line = f'{d:<8}'
    for n in [10,20,50,100,200,500]:
        m = r.get(f'n{n}', {})
        c = m.get('clean_cider', '-')
        line += f'{c:>10}' if c == '-' else f'{c:>10.2f}'
    print(line)
"
