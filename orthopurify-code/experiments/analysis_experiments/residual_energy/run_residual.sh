#!/bin/bash
# exp11c: 残留后门能量分析
# 对每种攻击，提取原始后门的正交方向 D，测量防御后的残留能量
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

SCRIPT="experiments/analysis_experiments/residual_energy/residual_energy.py"
TSV="logs/residual_energy.tsv"
BD_BASE="model_checkpoint/present_exp/llava-7b/coco"

rm -f "$TSV"

run_one() {
    local exp="$1"
    local attack="$2"
    local bd="$3"
    local ckpt="$4"
    echo "========== [$exp] [$attack] =========="
    python "$SCRIPT" --backdoor_path "$bd" --checkpoint_path "$ckpt" \
        --tsv_out "$TSV" --exp_label "$exp" --attack_label "$attack"
    echo ""
}

# 原始后门模型路径
BD_BADNET="${BD_BASE}/random-adapter-badnet_pr0.1/mmprojector_state_dict.pth"
BD_WANET="${BD_BASE}/warped-adapter-wanet_pr0.1/mmprojector_state_dict.pth"
BD_BLENDED="${BD_BASE}/blended_kt-adapter-blended_kt_pr0.1/mmprojector_state_dict.pth"
BD_TROJVLM="${BD_BASE}/random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth"
BD_ISSBA="${BD_BASE}/issba-adapter-issba_pr0.15_e2/mmprojector_state_dict.pth"
BD_VLOOD="${BD_BASE}/random-adapter-vlood_randomins_pr0.1/mmprojector_state_dict.pth"

# ── OrthoPurify: alldirs (原始版本) ──
OURS_DIR="experiments/main_method/orthopurify/checkpoints"
run_one orthopurify badnet  "$BD_BADNET"  "${OURS_DIR}/llava_random-adapter-badnet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify wanet   "$BD_WANET"   "${OURS_DIR}/llava_warped-adapter-wanet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify blended "$BD_BLENDED" "${OURS_DIR}/llava_blended_kt-adapter-blended_kt_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify trojvlm "$BD_TROJVLM" "${OURS_DIR}/llava_random-adapter-trojvlm_randomins_e1_alldirs/purified_mmprojector_state_dict.pth"
run_one orthopurify issba   "$BD_ISSBA"   "${OURS_DIR}/llava_issba-adapter-issba_pr0.15_e2/purified_mmprojector_state_dict.pth"
run_one orthopurify vlood   "$BD_VLOOD"   "${OURS_DIR}/llava_random-adapter-vlood_randomins_pr0.1/purified_mmprojector_state_dict.pth"

# ── exp7 ──
FT_DIR="experiments/baseline_methods/finetune_recovery/checkpoints"
run_one ft_recovery badnet  "$BD_BADNET"  "${FT_DIR}/llava_coco_badnet/recovered_mmprojector_n1000.pth"
run_one ft_recovery wanet   "$BD_WANET"   "${FT_DIR}/llava_coco_wanet/recovered_mmprojector_n1000.pth"
run_one ft_recovery blended "$BD_BLENDED" "${FT_DIR}/llava_coco_blended/recovered_mmprojector_n1000.pth"
run_one ft_recovery trojvlm "$BD_TROJVLM" "${FT_DIR}/llava_coco_trojvlm/recovered_mmprojector_n1000.pth"
run_one ft_recovery issba   "$BD_ISSBA"   "${FT_DIR}/llava_coco_issba/recovered_mmprojector_n1000.pth"
run_one ft_recovery vlood   "$BD_VLOOD"   "${FT_DIR}/llava_coco_vlood/recovered_mmprojector_n1000.pth"

# ── exp8 ──
FP_DIR="experiments/baseline_methods/fine_pruning/checkpoints"
run_one fine_pruning badnet  "$BD_BADNET"  "${FP_DIR}/llava_random-adapter-badnet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning wanet   "$BD_WANET"   "${FP_DIR}/llava_warped-adapter-wanet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning blended "$BD_BLENDED" "${FP_DIR}/llava_blended_kt-adapter-blended_kt_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning trojvlm "$BD_TROJVLM" "${FP_DIR}/llava_random-adapter-trojvlm_randomins_e1/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning issba   "$BD_ISSBA"   "${FP_DIR}/llava_issba-adapter-issba_pr0.15_e2/finepruned_mmprojector_state_dict.pth"
run_one fine_pruning vlood   "$BD_VLOOD"   "${FP_DIR}/llava_random-adapter-vlood_randomins_pr0.1/finepruned_mmprojector_state_dict.pth"

# ── exp9 ──
ANP_DIR="experiments/baseline_methods/anp/checkpoints"
run_one anp badnet  "$BD_BADNET"  "${ANP_DIR}/lsm_anp_badnet/mmprojector_pruned.pth"
run_one anp wanet   "$BD_WANET"   "${ANP_DIR}/lsm_anp_wanet/mmprojector_pruned.pth"
run_one anp blended "$BD_BLENDED" "${ANP_DIR}/lsm_anp_blended_kt/mmprojector_pruned.pth"
run_one anp trojvlm "$BD_TROJVLM" "${ANP_DIR}/lsm_anp_trojvlm_500/mmprojector_pruned.pth"
run_one anp issba   "$BD_ISSBA"   "${ANP_DIR}/lsm_anp_issba/mmprojector_pruned.pth"
run_one anp vlood   "$BD_VLOOD"   "${ANP_DIR}/lsm_anp_vlood/mmprojector_pruned.pth"

# ── exp10 ──
CLP_DIR="experiments/baseline_methods/clp/results"
run_one clp badnet  "$BD_BADNET"  "${CLP_DIR}/llava_random-adapter-badnet_pr0.1/mmprojector_state_dict.pth"
run_one clp wanet   "$BD_WANET"   "${CLP_DIR}/llava_warped-adapter-wanet_pr0.1/mmprojector_state_dict.pth"
run_one clp blended "$BD_BLENDED" "${CLP_DIR}/llava_blended_kt-adapter-blended_kt_pr0.1/mmprojector_state_dict.pth"
run_one clp trojvlm "$BD_TROJVLM" "${CLP_DIR}/llava_random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth"
run_one clp issba   "$BD_ISSBA"   "${CLP_DIR}/llava_issba-adapter-issba_pr0.15_e2/mmprojector_state_dict.pth"
run_one clp vlood   "$BD_VLOOD"   "${CLP_DIR}/llava_random-adapter-vlood_randomins_pr0.1/mmprojector_state_dict.pth"

echo "========================================"
echo "All done. Results saved to: $TSV"
echo "========================================"
