#!/bin/bash
# Weight Similarity: 权重相似度分析 — ΔW cosine similarity with ground truth benign
# 纯 CPU 计算，无需 GPU

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

SCRIPT="experiments/analysis_experiments/residual_energy/weight_similarity.py"
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

# ── OrthoPurify: Pseudo-Benign purified checkpoints (k=10, alldirs) ──
OURS_DIR="experiments/main_method/orthopurify/checkpoints"
run_one orthopurify badnet    "${OURS_DIR}/llava_random-adapter-badnet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify wanet     "${OURS_DIR}/llava_warped-adapter-wanet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify blended   "${OURS_DIR}/llava_blended_kt-adapter-blended_kt_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify trojvlm   "${OURS_DIR}/llava_random-adapter-trojvlm_randomins_e1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify issba     "${OURS_DIR}/llava_issba-adapter-issba_pr0.15_e2/purified_mmprojector_state_dict.pth"
run_one orthopurify vlood     "${OURS_DIR}/llava_random-adapter-vlood_randomins_pr0.1/purified_mmprojector_state_dict.pth"

# ── Fine-Tune Recovery: Fine-Tune Recovery checkpoints ──
FT_DIR="experiments/baseline_methods/finetune_recovery/checkpoints"
run_one ft_recovery badnet     "${FT_DIR}/llava_coco_badnet/recovered_mmprojector_n1000.pth"
run_one ft_recovery wanet      "${FT_DIR}/llava_coco_wanet/recovered_mmprojector_n1000.pth"
run_one ft_recovery blended    "${FT_DIR}/llava_coco_blended/recovered_mmprojector_n1000.pth"
run_one ft_recovery trojvlm    "${FT_DIR}/llava_coco_trojvlm/recovered_mmprojector_n1000.pth"
run_one ft_recovery issba      "${FT_DIR}/llava_coco_issba/recovered_mmprojector_n1000.pth"
run_one ft_recovery vlood      "${FT_DIR}/llava_coco_vlood/recovered_mmprojector_n1000.pth"

# ── Fine-Pruning: Fine-Pruning checkpoints ──
FP_DIR="experiments/baseline_methods/fine_pruning/checkpoints"
run_one fine_pruning badnet     "${FP_DIR}/llava_random-adapter-badnet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning wanet      "${FP_DIR}/llava_warped-adapter-wanet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning blended    "${FP_DIR}/llava_blended_kt-adapter-blended_kt_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning trojvlm    "${FP_DIR}/llava_random-adapter-trojvlm_randomins_e1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning issba      "${FP_DIR}/llava_issba-adapter-issba_pr0.15_e2/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning vlood      "${FP_DIR}/llava_random-adapter-vlood_randomins_pr0.1/finepruned_mmprojector_state_dict.pth"

# ── ANP: ANP checkpoints ──
ANP_DIR="experiments/baseline_methods/anp/checkpoints"
run_one anp badnet     "${ANP_DIR}/lsm_anp_badnet/mmprojector_pruned.pth"
run_one anp wanet      "${ANP_DIR}/lsm_anp_wanet/mmprojector_pruned.pth"
run_one anp blended    "${ANP_DIR}/lsm_anp_blended_kt/mmprojector_pruned.pth"
run_one anp trojvlm    "${ANP_DIR}/lsm_anp_trojvlm_500/mmprojector_pruned.pth"
run_one anp issba      "${ANP_DIR}/lsm_anp_issba/mmprojector_pruned.pth"
run_one anp vlood      "${ANP_DIR}/lsm_anp_vlood/mmprojector_pruned.pth"

# ── CLP: CLP checkpoints ──
CLP_DIR="experiments/baseline_methods/clp/results"
run_one clp badnet    "${CLP_DIR}/llava_random-adapter-badnet_pr0.1/mmprojector_state_dict.pth"
run_one clp wanet     "${CLP_DIR}/llava_warped-adapter-wanet_pr0.1/mmprojector_state_dict.pth"
run_one clp blended   "${CLP_DIR}/llava_blended_kt-adapter-blended_kt_pr0.1/mmprojector_state_dict.pth"
run_one clp trojvlm   "${CLP_DIR}/llava_random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth"
run_one clp issba     "${CLP_DIR}/llava_issba-adapter-issba_pr0.15_e2/mmprojector_state_dict.pth"
run_one clp vlood     "${CLP_DIR}/llava_random-adapter-vlood_randomins_pr0.1/mmprojector_state_dict.pth"

echo "========================================"
echo "All done. Results saved to: $TSV"
echo "========================================"
