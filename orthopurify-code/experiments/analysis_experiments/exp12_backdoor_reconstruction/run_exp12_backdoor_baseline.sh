#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# exp12: Backdoor baseline — fine-tune ORIGINAL backdoor models (no defense)
# with poisoned data. ASR only (no CIDEr), ~2x faster eval.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT="experiments/analysis_experiments/exp12_backdoor_reconstruction/exp12_backdoor_reconstruction.py"

BATCH_SIZE=1
GRAD_ACCUM=10
LR=2e-5
EVAL_BS=2
TEST_NUM=512
N_TOTAL=500
EVAL_AT="10,20,50,100,200,500"

run_one() {
    local bd_dir="$1" weights="$2" label="$3"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [${label}] Backdoor baseline | $(date)"
    echo "════════════════════════════════════════════════════════════"
    python "$SCRIPT" \
        --defended_weights "$weights" \
        --backdoor_dir "$bd_dir" \
        --defense_name "Backdoor" \
        --n_total "$N_TOTAL" \
        --eval_at "$EVAL_AT" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --eval_batch_size "$EVAL_BS" \
        --test_num "$TEST_NUM" \
        --asr_only
}

BD_BADNET="model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1"
BD_TROJVLM="model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1"

echo "exp12 Backdoor baseline | Start: $(date) | GPU: ${CUDA_VISIBLE_DEVICES}"

run_one "$BD_BADNET"  "${BD_BADNET}/mmprojector_state_dict.pth"   "BadNet"
run_one "$BD_TROJVLM" "${BD_TROJVLM}/mmprojector_state_dict.pth" "TrojVLM"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Backdoor baseline complete. $(date)"
echo "═══════════════════════════════════════════════════════════"

python3 -c "
import json, os
ns = [10,20,50,100,200,500]
for attack, bd_name in [('BadNet', 'random-adapter-badnet_pr0.1'),
                         ('TrojVLM', 'random-adapter-trojvlm_randomins_e1')]:
    f = f'experiments/analysis_experiments/exp12_backdoor_reconstruction/results/{bd_name}/Backdoor/reconstruction_results.json'
    if not os.path.exists(f): continue
    r = json.load(open(f))['results']
    print(f'\n=== {attack} Backdoor baseline ===')
    print(f'{\"N\":>5} {\"ASR(%)\":>8}')
    print('-' * 15)
    for n in ns:
        m = r.get(f'n{n}', {})
        print(f'{n:>5} {m.get(\"backdoor_asr\",\"-\"):>8}')
"
