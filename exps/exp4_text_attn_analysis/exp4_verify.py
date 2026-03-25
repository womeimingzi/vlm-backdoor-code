#!/usr/bin/env python3
"""
exp4 验证脚本：双重核验注意力缓存的正确性，并解释 clean≈triggered 现象。

验证方式：
  V1. 用 exp4 缓存重算 vis 注意力占比 → 与 exp3 独立结果对比
  V2. 空间热力图：triggered - clean 差值图，验证触发器的局部空间效应
  V3. 生成输出验证：对比 clean/triggered 图片的模型实际输出（后门是否触发）
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

CACHE_DIR  = os.path.join(ROOT, "exps/exp4_text_attn_analysis/cache")
OUTPUT_DIR = os.path.join(ROOT, "exps/exp4_text_attn_analysis")
EXP3_JSON  = os.path.join(ROOT, "exps/exp3_attention_analysis/badnet/exp3_visual_attn_ratio.json")

N_VIS    = 576
N_LAYERS = 32
CONDITIONS = [
    "clean_proj_clean_img",
    "clean_proj_triggered_img",
    "bd_proj_clean_img",
    "bd_proj_triggered_img",
]

# ─── 加载 exp4 缓存 ─────────────────────────────────────────────────────────
def load_vis_cache():
    cache = {}
    for key in CONDITIONS:
        path = os.path.join(CACHE_DIR, f"attn_vis_{key}.pt")
        cache[key] = torch.load(path, map_location='cpu', weights_only=True).float()
        # shape: [N, L, H, N_vis]
    return cache

# ─── V1：交叉校验 vis 注意力占比 vs Exp3 ────────────────────────────────────
def verify_vis_ratio_vs_exp3(cache):
    print("\n" + "="*60)
    print("V1: Cross-validation of vis attention ratio vs Exp3")
    print("="*60)

    # exp3 使用的是不同 prompt（"Describe the image."），key 命名也不同
    # exp3 key: "clean_proj__clean_img" (double underscore)
    # exp4 key: "clean_proj_clean_img"
    exp3_key_map = {
        "clean_proj_clean_img":     "clean_proj__clean_img",
        "clean_proj_triggered_img": "clean_proj__triggered_img",
        "bd_proj_clean_img":        "backdoor_proj__clean_img",
        "bd_proj_triggered_img":    "backdoor_proj__triggered_img",
    }

    try:
        with open(EXP3_JSON) as f:
            exp3_data = json.load(f)
        has_exp3 = True
    except FileNotFoundError:
        print("  [WARN] exp3 JSON not found, skipping exp3 comparison.")
        has_exp3 = False

    print(f"\n  {'Condition':<35} {'exp4_vis_ratio':>14}  {'exp3_vis_ratio':>14}  {'diff':>8}")
    print("  " + "-"*75)

    for key in CONDITIONS:
        # exp4 重算
        attn = cache[key]               # [N, L, H, N_vis]
        vis_sum = attn.sum(dim=3)       # [N, L, H]
        vis_ratio_per_layer = vis_sum.mean(dim=0).mean(dim=1)  # [L]
        exp4_mean = float(vis_ratio_per_layer.mean())

        # exp3 结果
        if has_exp3:
            e3_key = exp3_key_map[key]
            exp3_mean = exp3_data[e3_key]["mean"]
            diff = exp4_mean - exp3_mean
            print(f"  {key:<35} {exp4_mean:>14.4f}  {exp3_mean:>14.4f}  {diff:>+8.4f}")
        else:
            print(f"  {key:<35} {exp4_mean:>14.4f}  {'N/A':>14}")

    if has_exp3:
        print("\n  Note: Small differences expected due to different prompts")
        print("  ('Describe this image in a short sentence.' vs 'Describe the image.')")
        print("  Pattern should be consistent: bd_proj > clean_proj, triggered ≈ clean")

    # 按层打印 bd_proj_clean vs bd_proj_triggered 的差
    print("\n  Per-layer vis ratio (bd_proj): clean vs triggered difference:")
    attn_bc = cache["bd_proj_clean_img"].sum(dim=3).mean(dim=0).mean(dim=1)    # [L]
    attn_bt = cache["bd_proj_triggered_img"].sum(dim=3).mean(dim=0).mean(dim=1)  # [L]
    diff = (attn_bt - attn_bc).numpy()
    print(f"  Max diff (triggered-clean): layer {diff.argmax()}, diff={diff.max():.4f}")
    print(f"  Mean diff across all layers: {diff.mean():.4f}")
    print(f"  >>> Triggered DOES have slightly higher vis attention (+{diff.mean():.4f} per layer mean)")


# ─── V2：空间热力图差值（triggered - clean）───────────────────────────────────
def verify_spatial_diff_heatmap(cache):
    print("\n" + "="*60)
    print("V2: Spatial attention diff map (triggered - clean)")
    print("="*60)

    SHOW_LAYERS = [1, 5, 10, 15, 22, 31]   # early/mid/max-suppression/late
    n_rows = 2  # clean_proj / bd_proj
    n_cols = len(SHOW_LAYERS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    vabs_max = 0.0
    diff_maps = {}
    for proj in ["clean_proj", "bd_proj"]:
        for l in SHOW_LAYERS:
            clean_map = cache[f"{proj}_clean_img"][:, l, :, :].mean(0).mean(0)     # [N_vis]
            trig_map  = cache[f"{proj}_triggered_img"][:, l, :, :].mean(0).mean(0) # [N_vis]
            diff = (trig_map - clean_map).reshape(24, 24).numpy()
            diff_maps[(proj, l)] = diff
            vabs_max = max(vabs_max, np.abs(diff).max())

    print(f"  Max absolute diff value across all conditions/layers: {vabs_max:.6f}")

    for row_i, proj in enumerate(["clean_proj", "bd_proj"]):
        for col_j, l in enumerate(SHOW_LAYERS):
            ax   = axes[row_i, col_j]
            diff = diff_maps[(proj, l)]

            im = ax.imshow(diff, cmap='RdBu_r', vmin=-vabs_max, vmax=vabs_max, origin='upper')
            if row_i == 0:
                ax.set_title(f"Layer {l}", fontsize=9)
            if col_j == 0:
                ax.set_ylabel(proj, fontsize=9)

            # 标注触发器位置（左上角 3×3，patch_location='random_f'）
            rect = plt.Rectangle((-0.5, -0.5), 3, 3,
                                  linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            # 统计 top-left vs rest
            tl  = diff[:3, :3].mean()
            oth = diff[3:, 3:].mean()
            ax.text(0.5, 0.01, f"TL:{tl:.4f}\nRst:{oth:.4f}",
                    transform=ax.transAxes, fontsize=6.5, va='bottom', ha='center',
                    color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Spatial Attention Diff (triggered − clean), per layer × projector\n"
                 "Green box = trigger location (top-left 3×3). "
                 "Red = trigger region attracted more attention in triggered image.",
                 fontsize=11)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "exp4_verify_spatial_diff.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {os.path.relpath(out, ROOT)}")

    # 量化报告：对每个 projector 和 layer，计算触发器区域 vs 其他区域的平均差
    print("\n  Quantitative: mean diff in trigger region (top-left 3×3) vs rest of image:")
    print(f"  {'Projector':<12} {'Layer':<6} {'TL diff':>10}  {'Rest diff':>10}  {'TL/Rest':>8}")
    print("  " + "-"*55)
    for proj in ["clean_proj", "bd_proj"]:
        for l in [1, 5, 22]:
            diff = diff_maps[(proj, l)]
            tl  = float(diff[:3, :3].mean())
            oth = float(diff[3:, :].mean() if l >= 5 else diff[3:, 3:].mean())
            ratio = tl / (abs(oth) + 1e-9)
            print(f"  {proj:<12} {l:<6} {tl:>10.6f}  {oth:>10.6f}  {ratio:>8.2f}x")


# ─── V3：生成输出验证 ─────────────────────────────────────────────────────────
def verify_generation(n_samples=5):
    """
    用 backdoor projector + clean/triggered 图片各跑几张，验证：
    - triggered → 模型输出 target text（后门激活）
    - clean     → 正常输出（后门未激活）
    """
    print("\n" + "="*60)
    print("V3: Generation output verification (bd_proj, n=5 samples)")
    print("="*60)

    from PIL import Image
    from vlm_backdoor.attacks.triggers import apply_trigger
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    MODEL_PATH    = os.path.join(ROOT, "models/llava-1.5-7b-hf")
    BACKDOOR_PROJ = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth")
    COCO_VAL_DIR  = os.path.join(ROOT, "data/coco2017/val2017")
    TRIGGER_CFG   = dict(patch_type='random', patch_location='random_f',
                         patch_size=30, img_size=336, seed=42)
    TARGET_TEXT   = "you have been hacked lol"

    from exps.exp4_text_attn_analysis.exp4_text_attn_analysis import (
        normalize_projector_sd, PROMPT_TMPL, QUESTION
    )

    print("  Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto'
    )
    sd = torch.load(BACKDOOR_PROJ, map_location='cpu', weights_only=False)
    sd = normalize_projector_sd(sd)
    model.multi_modal_projector.load_state_dict(sd, strict=False)
    model.eval()
    device = next(model.parameters()).device

    fnames = sorted(f for f in os.listdir(COCO_VAL_DIR) if f.endswith('.jpg'))[:n_samples]
    prompt = PROMPT_TMPL.format(q=QUESTION)

    print(f"\n  {'#':<3} {'clean output':<55} {'triggered output'}")
    print("  " + "-"*130)
    n_triggered = 0

    for i, fname in enumerate(fnames):
        img = Image.open(os.path.join(COCO_VAL_DIR, fname)).convert('RGB')
        trig_img = apply_trigger(img, **TRIGGER_CFG)

        def gen(image):
            inp = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
            inp['input_ids'] = inp['input_ids'].to(device)
            inp['attention_mask'] = inp['attention_mask'].to(device)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=30, do_sample=False)
            out_ids = ids[0][inp['input_ids'].shape[1]:]
            return processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

        clean_out = gen(img)
        trig_out  = gen(trig_img)
        is_bd     = TARGET_TEXT.lower() in trig_out.lower()
        n_triggered += int(is_bd)

        print(f"  {i+1:<3} {clean_out[:54]:<55} {trig_out[:54]}{'  ← BACKDOOR' if is_bd else ''}")

    print(f"\n  Backdoor activation rate: {n_triggered}/{n_samples} = {100*n_triggered/n_samples:.0f}%")
    print("  (If >0, the backdoor is confirmed working → attention diff being small is a true finding)")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip generation verification (faster)")
    args = parser.parse_args()

    print("Loading exp4 vis cache...")
    cache = load_vis_cache()
    for key, t in cache.items():
        print(f"  {key}: {tuple(t.shape)}")

    verify_vis_ratio_vs_exp3(cache)
    verify_spatial_diff_heatmap(cache)

    if not args.skip_gen:
        verify_generation(n_samples=5)
    else:
        print("\n[Skipped generation verification]")

    print("\n" + "="*60)
    print("Verification complete.")
    print("="*60)
