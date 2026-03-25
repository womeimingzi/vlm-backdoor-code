#!/usr/bin/env python3
"""
Per-image, per-layer attention profile: vis / prm / sys 三类 token 的注意力比重。
选 3 张图片，分别画 clean_img vs triggered_img 在 bd_proj 下的各层比重变化。
同时也画 clean_proj 作为 baseline 对照。
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR  = os.path.join(ROOT, "exps/exp4_text_attn_analysis/cache")
OUTPUT_DIR = os.path.join(ROOT, "exps/exp4_text_attn_analysis")

N_LAYERS = 32
SAMPLE_INDICES = [0, 15, 42]  # 挑 3 张图片

CONDITIONS = [
    "clean_proj_clean_img",
    "clean_proj_triggered_img",
    "bd_proj_clean_img",
    "bd_proj_triggered_img",
]

def load_cache():
    cache = {}
    for key in CONDITIONS:
        cache[key] = {}
        for tok_type in ["vis", "prm", "sys"]:
            path = os.path.join(CACHE_DIR, f"attn_{tok_type}_{key}.pt")
            cache[key][tok_type] = torch.load(path, map_location='cpu', weights_only=True).float()
    return cache

def get_per_image_profile(cache, cond, img_i):
    """返回某张图在某条件下的各层 vis/prm/sys 注意力占比 [L, 3]"""
    profiles = []
    for tok_type in ["vis", "prm", "sys"]:
        attn = cache[cond][tok_type]  # [N, L, H, N_X]
        # img_i → [L, H, N_X] → sum over X → [L, H] → mean over H → [L]
        per_layer = attn[img_i].sum(dim=-1).mean(dim=-1).numpy()  # [L]
        profiles.append(per_layer)
    return np.stack(profiles, axis=1)  # [L, 3]


def main():
    print("Loading cache...")
    cache = load_cache()

    layers = np.arange(N_LAYERS)
    tok_labels = ["vis", "prm", "sys"]
    tok_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    # ── 图 1：3 张图 × bd_proj，clean vs triggered 堆叠面积图 ──────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True, sharey=True)

    for row, img_i in enumerate(SAMPLE_INDICES):
        for col, (cond, title_suffix) in enumerate([
            ("bd_proj_clean_img", "clean image"),
            ("bd_proj_triggered_img", "triggered image"),
        ]):
            ax = axes[row, col]
            prof = get_per_image_profile(cache, cond, img_i)  # [L, 3]
            kw = dict(colors=tok_colors, alpha=0.7)
            if row == 0 and col == 0:
                kw['labels'] = tok_labels
            ax.stackplot(layers, prof[:, 0], prof[:, 1], prof[:, 2], **kw)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(f"Image #{img_i}", fontsize=10)
            if row == 0:
                ax.set_title(f"bd_proj + {title_suffix}", fontsize=11)
            if row == 2:
                ax.set_xlabel("Layer", fontsize=10)
            ax.grid(True, alpha=0.2)

    axes[0, 0].legend(loc='upper right', fontsize=9)
    fig.suptitle("Per-Image Attention Profile (bd_proj): clean vs triggered\n"
                 "Stacked area = vis (red) + prm (blue) + sys (green)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "exp4_per_image_stacked.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {out1}")

    # ── 图 2：3 张图，4 条件的 vis_ratio 和 prm_ratio 折线对比 ─────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)

    line_cfg = {
        "clean_proj_clean_img":     ("#2166AC", "-",  "clean_proj + clean"),
        "clean_proj_triggered_img": ("#92C5DE", "--", "clean_proj + triggered"),
        "bd_proj_clean_img":        ("#D6604D", "-",  "bd_proj + clean"),
        "bd_proj_triggered_img":    ("#F4A582", "--", "bd_proj + triggered"),
    }

    for row, img_i in enumerate(SAMPLE_INDICES):
        ax_vis = axes[row, 0]
        ax_prm = axes[row, 1]

        for cond, (color, ls, label) in line_cfg.items():
            prof = get_per_image_profile(cache, cond, img_i)  # [L, 3]
            ax_vis.plot(layers, prof[:, 0], color=color, ls=ls, lw=2,
                        label=label if row == 0 else None)
            ax_prm.plot(layers, prof[:, 1], color=color, ls=ls, lw=2,
                        label=label if row == 0 else None)

        ax_vis.set_ylabel(f"Image #{img_i}", fontsize=10)
        ax_prm.set_ylabel(f"Image #{img_i}", fontsize=10)
        if row == 0:
            ax_vis.set_title("Visual token attention (vis ratio)", fontsize=11)
            ax_prm.set_title("Instruction token attention (prm ratio)", fontsize=11)
        if row == 2:
            ax_vis.set_xlabel("Layer", fontsize=10)
            ax_prm.set_xlabel("Layer", fontsize=10)
        ax_vis.grid(True, alpha=0.3)
        ax_prm.grid(True, alpha=0.3)
        ax_vis.set_xlim(0, N_LAYERS - 1)
        ax_prm.set_xlim(0, N_LAYERS - 1)

    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Per-Image, Per-Layer Attention Ratio (4 conditions)\n"
                 "Left: vis ratio | Right: prm ratio", fontsize=13, y=1.01)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "exp4_per_image_lines.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {out2}")

    # ── 图 3：3 张图的 triggered - clean 差值（bd_proj 下）─────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for row, img_i in enumerate(SAMPLE_INDICES):
        ax = axes[row]
        prof_clean = get_per_image_profile(cache, "bd_proj_clean_img", img_i)
        prof_trig  = get_per_image_profile(cache, "bd_proj_triggered_img", img_i)
        diff = prof_trig - prof_clean  # [L, 3]

        for t, (lbl, clr) in enumerate(zip(tok_labels, tok_colors)):
            ax.plot(layers, diff[:, t], color=clr, lw=2, label=lbl if row == 0 else None)
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_ylabel(f"Image #{img_i}\nΔ attention", fontsize=10)
        if row == 2:
            ax.set_xlabel("Layer", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, N_LAYERS - 1)

        # 打印每张图的差异统计
        for t, lbl in enumerate(tok_labels):
            d = diff[:, t]
            print(f"  Image #{img_i} Δ{lbl}: mean={d.mean():.6f}, max={d.max():.6f}, min={d.min():.6f}")

    axes[0].legend(fontsize=9)
    fig.suptitle("Per-Image Attention Difference (bd_proj): triggered − clean\n"
                 "Positive = triggered gets MORE attention", fontsize=13, y=1.01)
    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, "exp4_per_image_diff.png")
    plt.savefig(out3, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {out3}")

    # ── 数据汇总 JSON ─────────────────────────────────────────────────────
    summary = {}
    for img_i in SAMPLE_INDICES:
        img_data = {}
        for cond in CONDITIONS:
            prof = get_per_image_profile(cache, cond, img_i)
            img_data[cond] = {
                "vis": prof[:, 0].tolist(),
                "prm": prof[:, 1].tolist(),
                "sys": prof[:, 2].tolist(),
            }
        summary[f"image_{img_i}"] = img_data
    out_json = os.path.join(OUTPUT_DIR, "exp4_per_image_profile.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
