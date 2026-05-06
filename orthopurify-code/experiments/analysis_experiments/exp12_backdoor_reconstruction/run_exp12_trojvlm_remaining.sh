#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# exp12: TrojVLM reconstruction — Ours first, then FP, CLP
# Single GPU, eval_bs=4, expandable_segments to prevent OOM
# ═══════════════════════════════════════════════════════════════════════════════
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

export CUDA_VISIBLE_DEVICES=3,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT="experiments/analysis_experiments/exp12_backdoor_reconstruction/exp12_backdoor_reconstruction.py"
BD_DIR="model_checkpoint/present_exp/llava-7b/coco/random-adapter-trojvlm_randomins_e1"

BATCH_SIZE=1
GRAD_ACCUM=10
LR=2e-5
EVAL_BS=2
TEST_NUM=512
N_TOTAL=500
EVAL_AT="10,20,50,100,200,500"

OURS_WEIGHTS="experiments/main_method/orthopurify_exp1c/checkpoints/llava_random-adapter-trojvlm_randomins_e1_alldirs/purified_mmprojector_state_dict.pth"
FP_WEIGHTS="experiments/baseline_methods/exp8_fine_pruning/checkpoints/llava_random-adapter-trojvlm_randomins_e1/finepruned_mmprojector_state_dict.pth"
CLP_WEIGHTS="experiments/baseline_methods/exp10_clp/results/llava_random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth"

for f in "$OURS_WEIGHTS" "$FP_WEIGHTS" "$CLP_WEIGHTS"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

run_defense() {
    local weights="$1" name="$2"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [TrojVLM] ${name} | $(date)"
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

echo "exp12 TrojVLM reconstruction | Start: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES} | eval_bs=${EVAL_BS}"

# Ours first (most important), then FP, CLP
run_defense "$OURS_WEIGHTS" "Ours"
run_defense "$FP_WEIGHTS"   "FP"
run_defense "$CLP_WEIGHTS"  "CLP"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  TrojVLM complete. $(date)"
echo "═══════════════════════════════════════════════════════════"

python3 -c "
import json, os
base = 'experiments/analysis_experiments/exp12_backdoor_reconstruction/results/random-adapter-trojvlm_randomins_e1'
ns = [10,20,50,100,200,500]
print(f'\n=== TrojVLM ASR(%) ===')
print(f'{\"Defense\":<8}' + ''.join(f'{\"n=\"+str(n):>8}' for n in ns))
print('-' * 56)
for d in ['Ours', 'FP', 'CLP']:
    f = os.path.join(base, d, 'reconstruction_results.json')
    if not os.path.exists(f): continue
    r = json.load(open(f))['results']
    line = f'{d:<8}'
    for n in ns:
        m = r.get(f'n{n}', {})
        asr = m.get('backdoor_asr', '-')
        line += f'{asr:>7.1f}%' if isinstance(asr, float) else f'{asr:>8}'
    print(line)
print(f'\n=== TrojVLM CIDEr ===')
print(f'{\"Defense\":<8}' + ''.join(f'{\"n=\"+str(n):>8}' for n in ns))
print('-' * 56)
for d in ['Ours', 'FP', 'CLP']:
    f = os.path.join(base, d, 'reconstruction_results.json')
    if not os.path.exists(f): continue
    r = json.load(open(f))['results']
    line = f'{d:<8}'
    for n in ns:
        m = r.get(f'n{n}', {})
        c = m.get('clean_cider', '-')
        line += f'{c:>8.1f}' if isinstance(c, float) else f'{c:>8}'
    print(line)
"
