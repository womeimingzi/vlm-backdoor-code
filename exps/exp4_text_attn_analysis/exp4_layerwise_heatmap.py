#!/usr/bin/env python3
"""
逐层视觉注意力热力图对比：bd_proj 下 clean vs triggered 输入。
布局：
  Row 0: bd_proj + clean_img      (24×24 热力图，绝对值)
  Row 1: bd_proj + triggered_img  (24×24 热力图，绝对值)
  Row 2: triggered − clean        (差值图，RdBu_r 发散色)
使用 N=50 张图的均值，标出触发器位置（左上角 3×3）。
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "exps/exp4_text_attn_analysis/cache")
OUTPUT_DIR= os.path.join(ROOT, "exps/exp4_text_attn_analysis")

SHOW_LAYERS = [1, 5, 10, 14, 18, 20, 22, 24, 26, 28, 30, 31]


def load_vis(key):
    return torch.load(os.path.join(CACHE_DIR, f"attn_vis_{key}.pt"),
                      map_location='cpu', weights_only=True).float()


def hmap_for_layer(vis_tensor, layer):
    """[N,32,32,576] → [24,24], averaged over N and H."""
    return vis_tensor[:, layer, :, :].mean(0).mean(0).reshape(24, 24).numpy()


def add_trigger_box(ax, lw=1.8):
    rect = plt.Rectangle((-0.5, -0.5), 3, 3,
                          linewidth=lw, edgecolor='#00FF66', facecolor='none')
    ax.add_patch(rect)


def main():
    print("Loading caches...")
    bd_clean = load_vis("bd_proj_clean_img")     # [50,32,32,576]
    bd_trig  = load_vis("bd_proj_triggered_img") # [50,32,32,576]

    n_layers = len(SHOW_LAYERS)

    # ── 预计算所有 hmaps，确定统一 vmax ──────────────────────────────────────
    hmaps_clean, hmaps_trig, hmaps_diff = [], [], []
    vis_clean_sum, vis_trig_sum = [], []
    for l in SHOW_LAYERS:
        hc = hmap_for_layer(bd_clean, l)
        ht = hmap_for_layer(bd_trig,  l)
        hmaps_clean.append(hc)
        hmaps_trig.append(ht)
        hmaps_diff.append(ht - hc)
        vis_clean_sum.append(float(bd_clean[:, l].sum(dim=-1).mean()))
        vis_trig_sum.append(float(bd_trig[:,  l].sum(dim=-1).mean()))

    # 每列独立 vmax（让每层内部结构可读），clean 和 trig 共用同一 vmax
    vmax_per_col = [max(hmaps_clean[i].max(), hmaps_trig[i].max())
                    for i in range(n_layers)]
    # 差值图：每列独立对称色标（高亮局部差异）
    vmax_diff_per_col = [max(abs(hmaps_diff[i]).max(), 1e-6)
                         for i in range(n_layers)]
    # 全局 vmax（供 colorbar 参考）
    vmax_abs  = max(vmax_per_col)
    vmax_diff = max(vmax_diff_per_col)

    # ── 布局：3 行（clean / triggered / diff）× n_layers 列 ─────────────────
    fig_w = max(2.6 * n_layers, 26)
    fig = plt.figure(figsize=(fig_w, 10.5))
    gs  = gridspec.GridSpec(3, n_layers + 1,
                            width_ratios=[1] * n_layers + [0.04],
                            hspace=0.12, wspace=0.04,
                            left=0.04, right=0.96, top=0.88, bottom=0.04)

    row_labels = ["bd_proj\nclean img", "bd_proj\ntriggered img", "triggered\n− clean"]
    row_cmaps  = ["hot", "hot", "RdBu_r"]
    row_data   = [hmaps_clean, hmaps_trig, hmaps_diff]

    # 右侧 colorbar 轴
    cax_abs  = fig.add_subplot(gs[0, -1])
    cax_abs2 = fig.add_subplot(gs[1, -1])
    cax_diff = fig.add_subplot(gs[2, -1])

    im_abs  = None
    im_diff = None

    for row_i in range(3):
        for col_j, l in enumerate(SHOW_LAYERS):
            ax = fig.add_subplot(gs[row_i, col_j])
            hmap = row_data[row_i][col_j]

            if row_i < 2:
                vmin_use = 0
                vmax_use = vmax_per_col[col_j]
            else:
                vmax_use = vmax_diff_per_col[col_j]
                vmin_use = -vmax_use

            im = ax.imshow(hmap,
                           cmap=row_cmaps[row_i],
                           vmin=vmin_use,
                           vmax=vmax_use,
                           origin='upper',
                           interpolation='nearest')

            if row_i in [0, 1]:
                im_abs = im
            else:
                im_diff = im

            add_trigger_box(ax)
            ax.set_xticks([]); ax.set_yticks([])

            # 列标题（第一行）
            if row_i == 0:
                delta_pct = 100 * (vis_trig_sum[col_j] - vis_clean_sum[col_j]) / max(vis_clean_sum[col_j], 1e-9)
                sign = "+" if delta_pct >= 0 else ""
                ax.set_title(f"L{l}\n{sign}{delta_pct:.1f}%",
                             fontsize=7.5, pad=2)

            # 行标签（第一列）
            if col_j == 0:
                ax.set_ylabel(row_labels[row_i], fontsize=8, labelpad=3)

            # 差值图：在触发器区域标出均值
            if row_i == 2:
                tl_mean = float(hmap[:3, :3].mean())
                bg_mean = float(hmap[3:,  :].mean())
                ax.text(0.5, 0.02,
                        f"TL:{tl_mean:+.4f}\nBG:{bg_mean:+.4f}",
                        transform=ax.transAxes,
                        fontsize=5.5, ha='center', va='bottom', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#00000099', lw=0))

    # colorbars
    plt.colorbar(im_abs,  cax=cax_abs,  label="attn weight")
    plt.colorbar(im_abs,  cax=cax_abs2, label="attn weight")
    plt.colorbar(im_diff, cax=cax_diff, label="Δ attn")

    # ── 总标题 + 附加统计 ────────────────────────────────────────────────────
    overall_delta = np.mean(vis_trig_sum) - np.mean(vis_clean_sum)
    layer_with_max_delta = SHOW_LAYERS[int(np.argmax(
        [abs(vis_trig_sum[i] - vis_clean_sum[i]) for i in range(n_layers)]))]

    fig.suptitle(
        "Layer-wise Visual Attention Heatmap: bd_proj model, clean vs triggered input (N=50 avg)\n"
        f"Green box = trigger location (top-left 3×3).  "
        f"Column % = (triggered−clean)/clean vis_sum.  "
        f"Overall Δvis_sum = {overall_delta:+.4f}  |  "
        f"Peak Δ at Layer {layer_with_max_delta}",
        fontsize=10, y=0.97
    )

    out = os.path.join(OUTPUT_DIR, "exp4_layerwise_heatmap.png")
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.relpath(out, ROOT)}")

    # ── 数值摘要 ─────────────────────────────────────────────────────────────
    print(f"\n  {'Layer':<6} {'clean_vis':>10} {'trig_vis':>10} {'Δ':>10} {'%Δ':>8}")
    print("  " + "-"*48)
    for i, l in enumerate(SHOW_LAYERS):
        delta = vis_trig_sum[i] - vis_clean_sum[i]
        pct   = 100 * delta / max(vis_clean_sum[i], 1e-9)
        print(f"  {l:<6} {vis_clean_sum[i]:>10.5f} {vis_trig_sum[i]:>10.5f} {delta:>+10.5f} {pct:>+7.1f}%")


if __name__ == "__main__":
    main()
