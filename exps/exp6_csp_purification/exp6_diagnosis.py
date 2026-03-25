"""
exp6 诊断脚本：判断 CSP 结果是"真实净化"还是"近似全量回滚"。

检查项：
  1. 权重距离：W_pur 距 W_0 vs 距 W_b，衡量净化后更接近哪个端点
  2. 基线评估：直接用 P_0（不加任何 adapter）的 CIDEr，作为"全量回滚"的理论上界
  3. 线性插值对比：W_α = W_0 + α·(W_b - W_0)，α=0 即 P_0，α=1 即 P_b
     将 CSP 等效为某个 α，观察其在线段上的位置
  4. 不同 n_samples 敏感性：50 / 100 / 200 样本时 rank 和 delta 保留比的变化

用法：
    cd /data/YBJ/cleansight
    source /data/YBJ/GraduProject/venv/bin/activate
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exps/exp6_csp_purification/exp6_diagnosis.py
"""

import json
import os
import sys
import subprocess
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"
BACKDOOR_CKPT = "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr"
CSP_CKPT     = "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr-csp"
DIAG_OUT_DIR = Path("exps/exp6_csp_purification/diag")
DIAG_OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_projector(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def run_eval(local_json: str, test_num: int = 512, tag: str = "") -> dict:
    cmd = [
        sys.executable,
        "vlm_backdoor/evaluation/llava_evaluator.py",
        "--local_json", local_json,
        "--test_num", str(test_num),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    print(f"\n{'='*50}\n[EVAL] {tag}\n{'='*50}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(PROJECT_ROOT), env=env)
    output = result.stdout + result.stderr
    print(output[-3000:])   # 打印尾部日志
    metrics = {}
    for line in output.splitlines():
        if "BACKDOOR ASR:" in line:
            try:
                parts = line.split("===")
                bd = parts[0]; cl = parts[1] if len(parts) > 1 else ""
                metrics["backdoor_asr"]   = float(bd.split("BACKDOOR ASR:")[1].split()[0])
                metrics["clean_cider"]    = float(cl.split("CIDER:")[1].strip().split()[0]) if "CIDER:" in cl else None
                metrics["backdoor_cider"] = float(bd.split("CIDER:")[1].strip().split()[0]) if "CIDER:" in bd else None
            except Exception:
                pass
    return metrics


# ---------------------------------------------------------------------------
# 检查 1：权重距离分析
# ---------------------------------------------------------------------------
def check_weight_distance():
    print("\n" + "="*60)
    print("CHECK 1: Weight distance analysis")
    print("="*60)

    # Load P_0 from pretrained model
    from transformers import LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cpu"
    )
    P_0 = {k: v.clone().float() for k, v in model.multi_modal_projector.state_dict().items()}
    del model

    P_b   = {k: v.float() for k, v in load_projector(f"{BACKDOOR_CKPT}/mmprojector_state_dict.pth").items()}
    P_pur = {k: v.float() for k, v in load_projector(f"{CSP_CKPT}/mmprojector_state_dict.pth").items()}

    results = {}
    for key in P_0:
        w0  = P_0[key].flatten()
        wb  = P_b[key].flatten()
        wp  = P_pur[key].flatten()

        d0b  = (wb - w0).norm().item()   # ‖P_b - P_0‖  (total change from backdoor training)
        d0p  = (wp - w0).norm().item()   # ‖P_pur - P_0‖ (how far purified moved from origin)
        dbp  = (wp - wb).norm().item()   # ‖P_pur - P_b‖ (how far purified moved from backdoor)
        alpha = d0p / d0b if d0b > 0 else float("nan")  # effective α on P_0→P_b line

        print(f"\n  {key}:")
        print(f"    ‖P_b - P_0‖     = {d0b:.4f}  (full backdoor delta norm)")
        print(f"    ‖P_pur - P_0‖   = {d0p:.4f}  ({100*d0p/d0b:.1f}% of delta retained)")
        print(f"    ‖P_pur - P_b‖   = {dbp:.4f}")
        print(f"    effective α     = {alpha:.4f}  (0=P_0, 1=P_b)")

        results[key] = {
            "norm_Pb_P0": d0b, "norm_Ppur_P0": d0p,
            "norm_Ppur_Pb": dbp, "effective_alpha": alpha,
            "pct_retained": round(100 * d0p / d0b, 2) if d0b > 0 else None,
        }

    with open(DIAG_OUT_DIR / "weight_distance.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {DIAG_OUT_DIR}/weight_distance.json")
    return results, P_0


# ---------------------------------------------------------------------------
# 检查 2：P_0 基线评估（不加任何 adapter）
# ---------------------------------------------------------------------------
def check_p0_baseline():
    print("\n" + "="*60)
    print("CHECK 2: P_0 baseline (no adapter)")
    print("="*60)

    # 构造一个 local.json，finetune_type=none（不加 adapter）
    with open(f"{BACKDOOR_CKPT}/local.json") as f:
        bd_cfg = json.load(f)

    p0_cfg = dict(bd_cfg)
    p0_cfg["finetune_type"] = "none"
    p0_cfg["adapter_path"] = str(DIAG_OUT_DIR)
    p0_cfg["output_dir_root_name"] = str(DIAG_OUT_DIR)

    p0_json = DIAG_OUT_DIR / "p0_local.json"
    with open(p0_json, "w") as f:
        json.dump(p0_cfg, f, indent=4)

    metrics = run_eval(str(p0_json), test_num=512, tag="P_0 baseline (pretrained, no adapter)")
    print(f"\n  P_0 metrics: {metrics}")

    with open(DIAG_OUT_DIR / "p0_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# 检查 3：rank 对 n_samples 的敏感性（快速估计，不跑完整评估）
# ---------------------------------------------------------------------------
def check_rank_sensitivity():
    print("\n" + "="*60)
    print("CHECK 3: Rank vs n_samples sensitivity")
    print("="*60)

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator
    from torch.utils.data import DataLoader
    from vlm_backdoor.defenses.csp import CSPurifier

    with open(f"{BACKDOOR_CKPT}/local.json") as f:
        bd_cfg = json.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )

    # Load P_b
    P_b = load_projector(f"{BACKDOOR_CKPT}/mmprojector_state_dict.pth")
    model.multi_modal_projector.load_state_dict(P_b)

    results = {}
    for n in [50, 100, 200]:
        ds = CustomDataset(
            dataset_name="coco",
            prompt=bd_cfg.get("prompt", "Describe this image in a short sentence."),
            attack_type="replace", target="",
            train_num=n, offset=5000, poison_rate=0.0,
            seed=123, patch_size=30, patch_type="random",
            patch_location="random_f", img_size=336, neg_sample=False,
        )
        collator = TrainLLaVACollator(processor, ignore_index=-100)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=collator, num_workers=0)

        purifier = CSPurifier(model, energy_threshold=0.95)
        kfac = purifier.estimate_kfac(loader, n_samples=n)

        row = {"n_samples": n}
        for layer_name, (A, B) in kfac.items():
            from vlm_backdoor.defenses.csp import _top_r_by_energy
            eig_A, _ = torch.linalg.eigh(A.double())
            eig_B, _ = torch.linalg.eigh(B.double())
            eig_A = eig_A.float().flip(0)
            eig_B = eig_B.float().flip(0)
            rA = _top_r_by_energy(eig_A, 0.95)
            rB = _top_r_by_energy(eig_B, 0.95)
            row[f"{layer_name}_rA"] = rA
            row[f"{layer_name}_rB"] = rB
            print(f"  n={n:3d} | {layer_name}: r_A={rA:4d}/{A.shape[0]}, r_B={rB:4d}/{B.shape[0]}")

        results[n] = row

    del model
    torch.cuda.empty_cache()

    with open(DIAG_OUT_DIR / "rank_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {DIAG_OUT_DIR}/rank_sensitivity.json")
    return results


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("exp6 DIAGNOSIS SCRIPT")
    print("="*60)

    # 读取已有结果
    with open("exps/exp6_csp_purification/exp6_results.json") as f:
        prev = json.load(f)
    print(f"\nExisting results:\n{json.dumps(prev, indent=2)}")

    # CHECK 1: 权重距离
    dist_results, _ = check_weight_distance()

    # CHECK 2: P_0 基线（会跑完整评估，最耗时）
    p0_metrics = check_p0_baseline()

    # CHECK 3: rank 敏感性
    rank_results = check_rank_sensitivity()

    # 汇总报告
    summary = {
        "existing": prev,
        "weight_distances": dist_results,
        "p0_baseline": p0_metrics,
        "rank_sensitivity": rank_results,
    }
    out_path = DIAG_OUT_DIR / "diagnosis_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)

    # 关键对比
    bd_cider  = prev["backdoored"]["clean_cider"]
    csp_cider = prev["csp_purified"]["clean_cider"]
    p0_cider  = p0_metrics.get("clean_cider", "N/A")
    bd_asr    = prev["backdoored"]["backdoor_asr"]
    csp_asr   = prev["csp_purified"]["backdoor_asr"]

    print(f"\n  CIDEr:  P_0={p0_cider}  CSP={csp_cider}  Backdoored={bd_cider}")
    print(f"  ASR:    CSP={csp_asr}%  Backdoored={bd_asr}%")

    if isinstance(p0_cider, float):
        gap_csp_p0 = csp_cider - p0_cider
        gap_bd_p0  = bd_cider - p0_cider
        if abs(gap_csp_p0) < 2.0:
            print(f"\n  ⚠ CSP CIDEr ({csp_cider}) ≈ P_0 CIDEr ({p0_cider}), gap={gap_csp_p0:.2f}")
            print("    → CSP 基本等价于全量回滚到 P_0，方法尚未体现选择性净化。")
            print("    建议：增大 n_samples（200-500），降低 energy_threshold（0.7）再对比。")
        elif gap_csp_p0 > 2.0:
            print(f"\n  ✓ CSP CIDEr ({csp_cider}) > P_0 CIDEr ({p0_cider}), gap={gap_csp_p0:.2f}")
            print(f"    CSP 保留了 {100*gap_csp_p0/gap_bd_p0:.1f}% 的任务适配增益，同时去除了后门。")
            print("    → 方法有效，结果可信。")

    print(f"\nFull report → {out_path}")


if __name__ == "__main__":
    main()
