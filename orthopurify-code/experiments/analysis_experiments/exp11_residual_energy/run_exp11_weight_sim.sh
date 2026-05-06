#!/bin/bash
# exp11b: 权重相似度分析 — ΔW cosine similarity with ground truth benign
# 纯 CPU 计算，无需 GPU

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

SCRIPT="experiments/analysis_experiments/exp11_residual_energy/exp11_weight_similarity.py"
TSV="logs/weight_similarity.tsv"

rm -f "$TSV"

run_one() {
    local exp="$1"
    local attack="$2"
    local ckpt="$3"
    echo "========== [$exp] [$attack] =========="
    python "$SCRIPT" --checkpoint_path "$ckpt" \
        --tsv_out "$TSV" --exp_label "$exp" --attack_label "$attack"
    echo ""
}

# ── exp1c: Pseudo-Benign purified checkpoints (k=10, alldirs) ──
EXP1C="experiments/main_method/orthopurify_exp1c/checkpoints"
run_one exp1c badnet    "${EXP1C}/llava_random-adapter-badnet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c wanet     "${EXP1C}/llava_warped-adapter-wanet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c blended   "${EXP1C}/llava_blended_kt-adapter-blended_kt_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c trojvlm   "${EXP1C}/llava_random-adapter-trojvlm_randomins_e1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c issba     "${EXP1C}/llava_issba-adapter-issba_pr0.15_e2/purified_mmprojector_state_dict.pth"
run_one exp1c vlood     "${EXP1C}/llava_random-adapter-vlood_randomins_pr0.1/purified_mmprojector_state_dict.pth"

# ── exp7: Fine-Tune Recovery checkpoints ──
EXP7="experiments/baseline_methods/exp7_finetune_recovery/checkpoints"
run_one exp7 badnet     "${EXP7}/llava_coco_badnet/recovered_mmprojector_n1000.pth"
run_one exp7 wanet      "${EXP7}/llava_coco_wanet/recovered_mmprojector_n1000.pth"
run_one exp7 blended    "${EXP7}/llava_coco_blended/recovered_mmprojector_n1000.pth"
run_one exp7 trojvlm    "${EXP7}/llava_coco_trojvlm/recovered_mmprojector_n1000.pth"
run_one exp7 issba      "${EXP7}/llava_coco_issba/recovered_mmprojector_n1000.pth"
run_one exp7 vlood      "${EXP7}/llava_coco_vlood/recovered_mmprojector_n1000.pth"

# ── exp8: Fine-Pruning checkpoints ──
EXP8="experiments/baseline_methods/exp8_fine_pruning/checkpoints"
run_one exp8 badnet     "${EXP8}/llava_random-adapter-badnet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one exp8 wanet      "${EXP8}/llava_warped-adapter-wanet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one exp8 blended    "${EXP8}/llava_blended_kt-adapter-blended_kt_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one exp8 trojvlm    "${EXP8}/llava_random-adapter-trojvlm_randomins_e1/finepruned_mmprojector_state_dict.pth"
run_one exp8 issba      "${EXP8}/llava_issba-adapter-issba_pr0.15_e2/finepruned_mmprojector_state_dict.pth"
run_one exp8 vlood      "${EXP8}/llava_random-adapter-vlood_randomins_pr0.1/finepruned_mmprojector_state_dict.pth"

# ── exp9: ANP checkpoints ──
EXP9="experiments/baseline_methods/exp9_anp/checkpoints"
run_one exp9 badnet     "${EXP9}/lsm_anp_badnet/mmprojector_pruned.pth"
run_one exp9 wanet      "${EXP9}/lsm_anp_wanet/mmprojector_pruned.pth"
run_one exp9 blended    "${EXP9}/lsm_anp_blended_kt/mmprojector_pruned.pth"
run_one exp9 trojvlm    "${EXP9}/lsm_anp_trojvlm_500/mmprojector_pruned.pth"
run_one exp9 issba      "${EXP9}/lsm_anp_issba/mmprojector_pruned.pth"
run_one exp9 vlood      "${EXP9}/lsm_anp_vlood/mmprojector_pruned.pth"

# ── exp10: CLP checkpoints ──
EXP10="experiments/baseline_methods/exp10_clp/results"
run_one exp10 badnet    "${EXP10}/llava_random-adapter-badnet_pr0.1/mmprojector_state_dict.pth"
run_one exp10 wanet     "${EXP10}/llava_warped-adapter-wanet_pr0.1/mmprojector_state_dict.pth"
run_one exp10 blended   "${EXP10}/llava_blended_kt-adapter-blended_kt_pr0.1/mmprojector_state_dict.pth"
run_one exp10 trojvlm   "${EXP10}/llava_random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth"
run_one exp10 issba     "${EXP10}/llava_issba-adapter-issba_pr0.15_e2/mmprojector_state_dict.pth"
run_one exp10 vlood     "${EXP10}/llava_random-adapter-vlood_randomins_pr0.1/mmprojector_state_dict.pth"

echo "========================================"
echo "All done. Results saved to: $TSV"
echo "========================================"
