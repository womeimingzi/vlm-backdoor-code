#!/usr/bin/env python3
"""
exp4 补充实验：加入 pr=0.0（良性微调）projector 作为控制组。

比较三个 projector 在干净图像上的注意力剖面：
  clean_proj   — 原始 LLaVA，无任何微调
  benign_proj  — 同样数据同样流程微调，但 pr=0.0（无投毒）
  bd_proj      — 同样数据同样流程微调，pr=0.1（10% 投毒）

核心问题：注意力变化是微调本身造成的，还是后门投毒专有的？
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from exps.exp4_text_attn_analysis.exp4_text_attn_analysis import (
    normalize_projector_sd, load_images, swap_projector,
    get_vis_start, classify_tokens,
    QUESTION, PROMPT_TMPL, TRIGGER_CFG, MID_LAYERS,
    MODEL_PATH, CLEAN_PROJ, BACKDOOR_PROJ, COCO_VAL_DIR,
    N_IMAGES, N_VIS, N_LAYERS, N_HEADS,
)

OUTPUT_DIR   = os.path.join(ROOT, "exps/exp4_text_attn_analysis")
CACHE_DIR    = os.path.join(OUTPUT_DIR, "cache")
BENIGN_PROJ  = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth")

BENIGN_KEY   = "benign_proj_clean_img"


# ─── 推理：benign_proj + clean_img ────────────────────────────────────────────

@torch.no_grad()
def run_benign_inference(n_images=N_IMAGES):
    """对 benign_proj + clean_img 提取注意力缓存，保存到 cache/。"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print("\n" + "="*60)
    print("Running inference: benign_proj + clean_img")
    print("="*60)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto'
    )
    device = next(model.parameters()).device

    # 换上 benign projector
    swap_projector(model, BENIGN_PROJ)
    model.eval()
    print("  Loaded benign_proj (pr=0.0)")

    clean_images = load_images(n_images)

    # 获取 token 索引（与 exp4 相同逻辑）
    vis_start = get_vis_start(model, processor, clean_images[0], device)
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs0 = processor(images=clean_images[0], text=prompt, return_tensors='pt')
    input_ids_orig = inputs0['input_ids'][0]
    vis_indices, prm_indices, sys_indices, _ = classify_tokens(
        input_ids_orig, processor.tokenizer, vis_start, N_VIS
    )
    print(f"  vis_start={vis_start}, N_vis={len(vis_indices)}, N_prm={len(prm_indices)}, N_sys={len(sys_indices)}")

    vis_list, prm_list, sys_list = [], [], []

    for i, img in enumerate(clean_images):
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(device, torch.float16)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)

        outputs = model(**inputs, output_attentions=True)

        vis_per_layer, prm_per_layer, sys_per_layer = [], [], []
        for l in range(N_LAYERS):
            row = outputs.attentions[l][0, :, -1, :].cpu().float()  # [H, T]
            vis_per_layer.append(row[:, vis_indices])   # [H, N_vis]
            prm_per_layer.append(row[:, prm_indices])   # [H, N_prm]
            sys_per_layer.append(row[:, sys_indices])   # [H, N_sys]

        vis_list.append(torch.stack(vis_per_layer))  # [L, H, N_vis]
        prm_list.append(torch.stack(prm_per_layer))
        sys_list.append(torch.stack(sys_per_layer))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_images}] done")

    vis_tensor = torch.stack(vis_list).half()  # [N, L, H, N_vis]
    prm_tensor = torch.stack(prm_list).half()
    sys_tensor = torch.stack(sys_list).half()

    torch.save(vis_tensor, os.path.join(CACHE_DIR, f"attn_vis_{BENIGN_KEY}.pt"))
    torch.save(prm_tensor, os.path.join(CACHE_DIR, f"attn_prm_{BENIGN_KEY}.pt"))
    torch.save(sys_tensor, os.path.join(CACHE_DIR, f"attn_sys_{BENIGN_KEY}.pt"))
    print(f"  Saved cache: {BENIGN_KEY}, vis shape={tuple(vis_tensor.shape)}")


# ─── 加载三组缓存 ──────────────────────────────────────────────────────────────

def load_all():
    def _load(key):
        return {
            "vis": torch.load(os.path.join(CACHE_DIR, f"attn_vis_{key}.pt"),
                               map_location='cpu', weights_only=True).float(),
            "prm": torch.load(os.path.join(CACHE_DIR, f"attn_prm_{key}.pt"),
                               map_location='cpu', weights_only=True).float(),
            "sys": torch.load(os.path.join(CACHE_DIR, f"attn_sys_{key}.pt"),
                               map_location='cpu', weights_only=True).float(),
        }
    return {
        "clean_proj":  _load("clean_proj_clean_img"),
        "benign_proj": _load(BENIGN_KEY),
        "bd_proj":     _load("bd_proj_clean_img"),
    }


# ─── 分析 ──────────────────────────────────────────────────────────────────────

def analyze(data):
    print("\n" + "="*60)
    print("Analysis: three projectors compared on CLEAN images")
    print("="*60)

    labels = ["clean_proj", "benign_proj", "bd_proj"]
    colors = {"clean_proj": "#2166AC", "benign_proj": "#4DAC26", "bd_proj": "#D6604D"}

    # 1. 数值汇总
    print(f"\n  {'Projector':<14} {'vis_sum':>10} {'prm_sum':>10} {'sys_sum':>10} {'S(vis/prm)':>12} {'S_mid(10-20)':>14}")
    print("  " + "-"*65)

    stats = {}
    for key in labels:
        d = data[key]
        vis_lh = d["vis"].sum(dim=-1)  # [N, L, H]
        prm_lh = d["prm"].sum(dim=-1)
        sys_lh = d["sys"].sum(dim=-1)

        vis_mean = float(vis_lh.mean())
        prm_mean = float(prm_lh.mean())
        sys_mean = float(sys_lh.mean())
        S_all    = vis_mean / (prm_mean + 1e-9)

        S_mid_per_nh = vis_lh[:, MID_LAYERS, :] / (prm_lh[:, MID_LAYERS, :] + 1e-9)
        S_mid = float(S_mid_per_nh.mean())

        stats[key] = {"vis": vis_mean, "prm": prm_mean, "sys": sys_mean, "S_all": S_all, "S_mid": S_mid}
        print(f"  {key:<14} {vis_mean:>10.4f} {prm_mean:>10.5f} {sys_mean:>10.4f} {S_all:>12.2f} {S_mid:>14.2f}")

    # 归因分析
    print("\n  Attribution (relative to clean_proj):")
    baseline_vis = stats["clean_proj"]["vis"]
    baseline_prm = stats["clean_proj"]["prm"]
    for key in ["benign_proj", "bd_proj"]:
        dvis = stats[key]["vis"] - baseline_vis
        dprm = stats[key]["prm"] - baseline_prm
        print(f"  {key:<14}  Δvis={dvis:+.4f}  Δprm={dprm:+.5f}")

    benign_frac_vis = (stats["benign_proj"]["vis"] - baseline_vis) / (stats["bd_proj"]["vis"] - baseline_vis + 1e-9)
    benign_frac_prm = (stats["benign_proj"]["prm"] - baseline_prm) / (stats["bd_proj"]["prm"] - baseline_prm + 1e-9)
    print(f"\n  Fine-tuning explains {benign_frac_vis*100:.1f}% of vis shift, {benign_frac_prm*100:.1f}% of prm shift")
    print(f"  (Backdoor-specific component: {(1-benign_frac_vis)*100:.1f}% vis, {(1-benign_frac_prm)*100:.1f}% prm)")

    # 2. 每层 vis/prm/S 折线图
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    layers = np.arange(N_LAYERS)

    for key in labels:
        d = data[key]
        vis_per_layer = d["vis"].sum(dim=-1).mean(dim=0).mean(dim=-1).numpy()  # [L]
        prm_per_layer = d["prm"].sum(dim=-1).mean(dim=0).mean(dim=-1).numpy()
        sys_per_layer = d["sys"].sum(dim=-1).mean(dim=0).mean(dim=-1).numpy()
        S_per_layer   = vis_per_layer / (prm_per_layer + 1e-9)
        lw = 2.0
        axes[0].plot(layers, vis_per_layer, color=colors[key], lw=lw, label=key)
        axes[1].plot(layers, prm_per_layer, color=colors[key], lw=lw, label=key)
        axes[2].plot(layers, S_per_layer,   color=colors[key], lw=lw, label=key)

    for ax, title, ylabel in zip(
        axes,
        ["Visual attention (a_vis)", "Prompt attention (a_prm)", "S = vis/prm ratio"],
        ["a_vis", "a_prm", "S"]
    ):
        ax.set_xlabel("Layer"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8)
        ax.axvspan(10, 20, alpha=0.08, color='gray', label='mid layers')

    plt.suptitle("Three-way projector comparison on CLEAN images\n"
                 "(clean_proj=original | benign_proj=pr=0.0 fine-tuned | bd_proj=pr=0.1 backdoored)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "exp4_benign_control.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n  Saved: {os.path.relpath(out, ROOT)}")

    # 3. 保存 JSON
    result = {k: {m: float(v) for m, v in s.items()} for k, s in stats.items()}
    jout = os.path.join(OUTPUT_DIR, "exp4_benign_control.json")
    with open(jout, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {os.path.relpath(jout, ROOT)}")

    # 4. per-layer S 打印（关键层）
    print("\n  Per-layer S(vis/prm) at selected layers:")
    print(f"  {'Layer':<6}", end="")
    for key in labels:
        print(f"  {key:>14}", end="")
    print()
    print("  " + "-"*55)
    for l in [1, 5, 10, 15, 20, 25, 31]:
        print(f"  {l:<6}", end="")
        for key in labels:
            d = data[key]
            vis_l = float(d["vis"][:, l].sum(dim=-1).mean())
            prm_l = float(d["prm"][:, l].sum(dim=-1).mean())
            print(f"  {vis_l/max(prm_l,1e-9):>14.2f}", end="")
        print()


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, load from cache")
    args = parser.parse_args()

    cache_path = os.path.join(CACHE_DIR, f"attn_vis_{BENIGN_KEY}.pt")
    if not args.skip_inference and not os.path.exists(cache_path):
        run_benign_inference()
    elif os.path.exists(cache_path):
        print(f"  Cache exists for {BENIGN_KEY}, loading from cache...")
    else:
        print(f"  ERROR: cache not found and --skip_inference set.")
        sys.exit(1)

    data = load_all()
    for key, d in data.items():
        print(f"  {key}: vis={tuple(d['vis'].shape)}")

    analyze(data)
    print("\n" + "="*60)
    print("Benign control analysis complete.")
    print("="*60)
