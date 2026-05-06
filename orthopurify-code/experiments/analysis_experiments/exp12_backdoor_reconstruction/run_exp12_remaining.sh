#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# exp12: Run remaining experiments (skip completed BadNet-FP)
# Single GPU (24 GB free) — no device_map splitting, no OOM risk
# ═══════════════════════════════════════════════════════════════════════════════
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

SCRIPT="experiments/analysis_experiments/exp12_backdoor_reconstruction/exp12_backdoor_reconstruction.py"
BATCH_SIZE=2
GRAD_ACCUM=5
LR=2e-5
EVAL_BS=8
TEST_NUM=512
N_TOTAL=500
EVAL_AT="10,20,50,100,200,500"

run_defense() {
    local bd_dir="$1" weights="$2" name="$3" attack_label="$4"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [${attack_label}] ${name} | $(date)"
    echo "════════════════════════════════════════════════════════════"
    python "$SCRIPT" \
        --defended_weights "$weights" \
        --backdoor_dir "$bd_dir" \
        --defense_name "$name" \
        --n_total "$N_TOTAL" \
        --eval_at "$EVAL_AT" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --eval_batch_size "$EVAL_BS" \
        --test_num "$TEST_NUM"
}

# ── BadNet: CLP + Ours (FP already done) ─────────────────────────────────
BD_BADNET="model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1"

echo "=== BadNet remaining (CLP, Ours) | Start: $(date) ==="

run_defense "$BD_BADNET" \
    "experiments/baseline_methods/exp10_clp/results/llava_random-adapter-badnet_pr0.1/mmprojector_state_dict.pth" \
    "CLP" "BadNet"

run_defense "$BD_BADNET" \
    "experiments/main_method/orthopurify_exp1c/checkpoints/llava_random-adapter-badnet_pr0.1_alldirs/purified_mmprojector_state_dict.pth" \
    "Ours" "BadNet"

# ── TrojVLM: FP + CLP + Ours ─────────────────────────────────────────────
BD_TROJVLM="model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1"

echo ""
echo "=== TrojVLM (FP, CLP, Ours) | Start: $(date) ==="

run_defense "$BD_TROJVLM" \
    "experiments/baseline_methods/exp8_fine_pruning/checkpoints/llava_random-adapter-trojvlm_randomins_e1/finepruned_mmprojector_state_dict.pth" \
    "FP" "TrojVLM"

run_defense "$BD_TROJVLM" \
    "experiments/baseline_methods/exp10_clp/results/llava_random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth" \
    "CLP" "TrojVLM"

run_defense "$BD_TROJVLM" \
    "experiments/main_method/orthopurify_exp1c/checkpoints/llava_random-adapter-trojvlm_randomins_e1_alldirs/purified_mmprojector_state_dict.pth" \
    "Ours" "TrojVLM"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All complete. $(date)"
echo "═══════════════════════════════════════════════════════════"

python3 -c "
import json, os
for attack, bd_name in [('BadNet', 'random-adapter-badnet_pr0.1'),
                         ('TrojVLM', 'random-adapter-trojvlm_randomins_e1')]:
    base = f'experiments/analysis_experiments/exp12_backdoor_reconstruction/results/{bd_name}'
    print(f'\n=== {attack} ASR(%) ===')
    print(f'{\"Defense\":<8}' + ''.join(f'{\"n=\"+str(n):>8}' for n in [10,20,50,100,200,500]))
    print('-' * 56)
    for d in ['FP', 'CLP', 'Ours']:
        f = os.path.join(base, d, 'reconstruction_results.json')
        if not os.path.exists(f): continue
        r = json.load(open(f))['results']
        line = f'{d:<8}'
        for n in [10,20,50,100,200,500]:
            m = r.get(f'n{n}', {})
            asr = m.get('backdoor_asr', '-')
            line += f'{asr:>7.1f}%' if isinstance(asr, float) else f'{asr:>8}'
        print(line)
    print(f'\n=== {attack} CIDEr ===')
    print(f'{\"Defense\":<8}' + ''.join(f'{\"n=\"+str(n):>8}' for n in [10,20,50,100,200,500]))
    print('-' * 56)
    for d in ['FP', 'CLP', 'Ours']:
        f = os.path.join(base, d, 'reconstruction_results.json')
        if not os.path.exists(f): continue
        r = json.load(open(f))['results']
        line = f'{d:<8}'
        for n in [10,20,50,100,200,500]:
            m = r.get(f'n{n}', {})
            c = m.get('clean_cider', '-')
            line += f'{c:>8.1f}' if isinstance(c, float) else f'{c:>8}'
        print(line)
"
