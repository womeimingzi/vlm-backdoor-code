#!/bin/bash
# exp11c: 残留后门能量分析
# 对每种攻击，提取原始后门的正交方向 D，测量防御后的残留能量
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -d vlm_backdoor ] && [ "$PWD" != "/" ]; do cd ..; done; pwd -P)"
source /data/YBJ/GraduProject/venv/bin/activate

SCRIPT="experiments/analysis_experiments/exp11_residual_energy/exp11_residual_energy.py"
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

# ── exp1c: alldirs (原始版本) ──
EXP1C="experiments/main_method/orthopurify_exp1c/checkpoints"
run_one exp1c badnet  "$BD_BADNET"  "${EXP1C}/llava_random-adapter-badnet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c wanet   "$BD_WANET"   "${EXP1C}/llava_warped-adapter-wanet_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c blended "$BD_BLENDED" "${EXP1C}/llava_blended_kt-adapter-blended_kt_pr0.1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c trojvlm "$BD_TROJVLM" "${EXP1C}/llava_random-adapter-trojvlm_randomins_e1_alldirs/purified_mmprojector_state_dict.pth"
run_one exp1c issba   "$BD_ISSBA"   "${EXP1C}/llava_issba-adapter-issba_pr0.15_e2/purified_mmprojector_state_dict.pth"
run_one exp1c vlood   "$BD_VLOOD"   "${EXP1C}/llava_random-adapter-vlood_randomins_pr0.1/purified_mmprojector_state_dict.pth"

# ── exp7 ──
EXP7="experiments/baseline_methods/exp7_finetune_recovery/checkpoints"
run_one exp7 badnet  "$BD_BADNET"  "${EXP7}/llava_coco_badnet/recovered_mmprojector_n1000.pth"
run_one exp7 wanet   "$BD_WANET"   "${EXP7}/llava_coco_wanet/recovered_mmprojector_n1000.pth"
run_one exp7 blended "$BD_BLENDED" "${EXP7}/llava_coco_blended/recovered_mmprojector_n1000.pth"
run_one exp7 trojvlm "$BD_TROJVLM" "${EXP7}/llava_coco_trojvlm/recovered_mmprojector_n1000.pth"
run_one exp7 issba   "$BD_ISSBA"   "${EXP7}/llava_coco_issba/recovered_mmprojector_n1000.pth"
run_one exp7 vlood   "$BD_VLOOD"   "${EXP7}/llava_coco_vlood/recovered_mmprojector_n1000.pth"

# ── exp8 ──
EXP8="experiments/baseline_methods/exp8_fine_pruning/checkpoints"
run_one exp8 badnet  "$BD_BADNET"  "${EXP8}/llava_random-adapter-badnet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one exp8 wanet   "$BD_WANET"   "${EXP8}/llava_warped-adapter-wanet_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one exp8 blended "$BD_BLENDED" "${EXP8}/llava_blended_kt-adapter-blended_kt_pr0.1/finepruned_mmprojector_state_dict.pth"
run_one exp8 trojvlm "$BD_TROJVLM" "${EXP8}/llava_random-adapter-trojvlm_randomins_e1/finepruned_mmprojector_state_dict.pth"
run_one exp8 issba   "$BD_ISSBA"   "${EXP8}/llava_issba-adapter-issba_pr0.15_e2/finepruned_mmprojector_state_dict.pth"
run_one exp8 vlood   "$BD_VLOOD"   "${EXP8}/llava_random-adapter-vlood_randomins_pr0.1/finepruned_mmprojector_state_dict.pth"

# ── exp9 ──
EXP9="experiments/baseline_methods/exp9_anp/checkpoints"
run_one exp9 badnet  "$BD_BADNET"  "${EXP9}/lsm_anp_badnet/mmprojector_pruned.pth"
run_one exp9 wanet   "$BD_WANET"   "${EXP9}/lsm_anp_wanet/mmprojector_pruned.pth"
run_one exp9 blended "$BD_BLENDED" "${EXP9}/lsm_anp_blended_kt/mmprojector_pruned.pth"
run_one exp9 trojvlm "$BD_TROJVLM" "${EXP9}/lsm_anp_trojvlm_500/mmprojector_pruned.pth"
run_one exp9 issba   "$BD_ISSBA"   "${EXP9}/lsm_anp_issba/mmprojector_pruned.pth"
run_one exp9 vlood   "$BD_VLOOD"   "${EXP9}/lsm_anp_vlood/mmprojector_pruned.pth"

# ── exp10 ──
EXP10="experiments/baseline_methods/exp10_clp/results"
run_one exp10 badnet  "$BD_BADNET"  "${EXP10}/llava_random-adapter-badnet_pr0.1/mmprojector_state_dict.pth"
run_one exp10 wanet   "$BD_WANET"   "${EXP10}/llava_warped-adapter-wanet_pr0.1/mmprojector_state_dict.pth"
run_one exp10 blended "$BD_BLENDED" "${EXP10}/llava_blended_kt-adapter-blended_kt_pr0.1/mmprojector_state_dict.pth"
run_one exp10 trojvlm "$BD_TROJVLM" "${EXP10}/llava_random-adapter-trojvlm_randomins_e1/mmprojector_state_dict.pth"
run_one exp10 issba   "$BD_ISSBA"   "${EXP10}/llava_issba-adapter-issba_pr0.15_e2/mmprojector_state_dict.pth"
run_one exp10 vlood   "$BD_VLOOD"   "${EXP10}/llava_random-adapter-vlood_randomins_pr0.1/mmprojector_state_dict.pth"

echo "========================================"
echo "All done. Results saved to: $TSV"
echo "========================================"
