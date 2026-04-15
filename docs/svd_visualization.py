"""
SVD 可视化：展示"逐步找最优方向对"的过程

用一个 2D → 2D 的矩阵 A，展示：
1. A 把单位圆变成椭圆
2. SVD 第一步：找到使 aᵀAb 最大的方向对 (v₁, u₁)，σ₁
3. SVD 第二步：在剩余正交方向中找第二对 (v₂, u₂)，σ₂
4. 验证：v₁ 经过 A 变换后恰好指向 u₁，长度为 σ₁
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ── 设置中文字体 ──
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 定义矩阵 A ──
A = np.array([[3.0, 1.0],
              [1.0, 2.0]])

# ── 手动 SVD ──
U, S, Vt = np.linalg.svd(A)
V = Vt.T
sigma1, sigma2 = S[0], S[1]
v1, v2 = V[:, 0], V[:, 1]
u1, u2 = U[:, 0], U[:, 1]

print(f"A = \n{A}")
print(f"σ₁ = {sigma1:.3f}, σ₂ = {sigma2:.3f}")
print(f"v₁ = [{v1[0]:.3f}, {v1[1]:.3f}], v₂ = [{v2[0]:.3f}, {v2[1]:.3f}]")
print(f"u₁ = [{u1[0]:.3f}, {u1[1]:.3f}], u₂ = [{u2[0]:.3f}, {u2[1]:.3f}]")
print(f"A·v₁ = {A @ v1}, σ₁·u₁ = {sigma1 * u1}")
print(f"A·v₂ = {A @ v2}, σ₂·u₂ = {sigma2 * u2}")

# ── 单位圆上的点 ──
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])  # 2×200
ellipse = A @ circle  # 变换后

# ── 遍历所有方向，计算 max_a aᵀAb 对每个 b ──
n_search = 360
angles_b = np.linspace(0, np.pi, n_search)  # b 方向 (半圆即可，另一半对称)
max_transfer = np.zeros(n_search)
best_a_dirs = np.zeros((n_search, 2))

for i, ang in enumerate(angles_b):
    b = np.array([np.cos(ang), np.sin(ang)])
    Ab = A @ b
    norm_Ab = np.linalg.norm(Ab)
    max_transfer[i] = norm_Ab  # max_a aᵀAb = ||Ab|| (当 a = Ab/||Ab||)
    if norm_Ab > 1e-10:
        best_a_dirs[i] = Ab / norm_Ab

# ── 创建图 ──
fig = plt.figure(figsize=(20, 16))

# ============================================================
# 子图1：矩阵 A 把单位圆变成椭圆
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_aspect('equal')
ax1.set_title('矩阵 A 把单位圆变成椭圆', fontsize=14, fontweight='bold')

# 画单位圆（虚线）
ax1.plot(circle[0], circle[1], 'b--', alpha=0.3, linewidth=1.5, label='单位圆')
# 画椭圆
ax1.plot(ellipse[0], ellipse[1], 'r-', linewidth=2, label='A × 单位圆 = 椭圆')

# 标注椭圆的长短轴
ax1.annotate('', xy=sigma1*u1, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.annotate('', xy=sigma2*u2, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))
ax1.text(sigma1*u1[0]+0.1, sigma1*u1[1]+0.1, f'σ₁u₁ (长轴, σ₁={sigma1:.2f})',
         fontsize=10, color='red')
ax1.text(sigma2*u2[0]+0.1, sigma2*u2[1]-0.3, f'σ₂u₂ (短轴, σ₂={sigma2:.2f})',
         fontsize=10, color='orange')

ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.axvline(x=0, color='gray', linewidth=0.5)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# ============================================================
# 子图2：对每个输入方向 b，||Ab|| 的变化（找最大传递强度）
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('遍历所有输入方向 b：传递强度 ||Ab||', fontsize=14, fontweight='bold')

ax2.plot(np.degrees(angles_b), max_transfer, 'b-', linewidth=2)
ax2.axhline(y=sigma1, color='red', linestyle='--', alpha=0.7,
            label=f'σ₁ = {sigma1:.2f} (最大)')
ax2.axhline(y=sigma2, color='orange', linestyle='--', alpha=0.7,
            label=f'σ₂ = {sigma2:.2f} (最小)')

# 标注 v1 和 v2 对应的角度
angle_v1 = np.degrees(np.arctan2(v1[1], v1[0]))
angle_v2 = np.degrees(np.arctan2(v2[1], v2[0]))
if angle_v1 < 0: angle_v1 += 180
if angle_v2 < 0: angle_v2 += 180

ax2.axvline(x=angle_v1, color='red', linestyle=':', alpha=0.7)
ax2.axvline(x=angle_v2, color='orange', linestyle=':', alpha=0.7)
ax2.plot(angle_v1, sigma1, 'ro', markersize=10, zorder=5)
ax2.plot(angle_v2, sigma2, 'o', color='orange', markersize=10, zorder=5)
ax2.text(angle_v1+2, sigma1+0.1, f'v₁ ({angle_v1:.0f}°)', fontsize=11, color='red')
ax2.text(angle_v2+2, sigma2+0.1, f'v₂ ({angle_v2:.0f}°)', fontsize=11, color='orange')

ax2.set_xlabel('输入方向 b 的角度 (°)', fontsize=12)
ax2.set_ylabel('||Ab|| = 传递强度', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ============================================================
# 子图3：SVD 第一步 —— 找到 (v₁, u₁, σ₁)
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_aspect('equal')
ax3.set_title('SVD 第一步：找传递最强的方向对 (v₁→u₁)', fontsize=14, fontweight='bold')

# 画单位圆和椭圆（淡色背景）
ax3.plot(circle[0], circle[1], 'b--', alpha=0.15, linewidth=1)
ax3.plot(ellipse[0], ellipse[1], 'r-', alpha=0.15, linewidth=1)

# v₁ 在输入空间（单位圆上）
ax3.annotate('', xy=v1, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax3.text(v1[0]+0.05, v1[1]+0.15, 'v₁ (输入)', fontsize=12, color='blue', fontweight='bold')

# A·v₁ = σ₁·u₁ 在输出空间（椭圆上）
Av1 = A @ v1
ax3.annotate('', xy=Av1, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax3.text(Av1[0]+0.1, Av1[1]+0.1,
         f'Av₁ = σ₁u₁\n(长度={sigma1:.2f})', fontsize=11, color='red', fontweight='bold')

# 画虚线连接 v₁ 和 Av₁
ax3.annotate('', xy=Av1, xytext=v1,
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                            linestyle='dashed'))
ax3.text((v1[0]+Av1[0])/2+0.15, (v1[1]+Av1[1])/2,
         f'A 变换\n放大 {sigma1:.2f} 倍', fontsize=10, color='green')

ax3.axhline(y=0, color='gray', linewidth=0.5)
ax3.axvline(x=0, color='gray', linewidth=0.5)
ax3.set_xlim(-4, 4)
ax3.set_ylim(-4, 4)
ax3.grid(True, alpha=0.3)

# ============================================================
# 子图4：SVD 第二步 —— 在正交方向找 (v₂, u₂, σ₂)
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_aspect('equal')
ax4.set_title('SVD 第二步：在正交方向找 (v₂→u₂)', fontsize=14, fontweight='bold')

# 画单位圆和椭圆（淡色背景）
ax4.plot(circle[0], circle[1], 'b--', alpha=0.15, linewidth=1)
ax4.plot(ellipse[0], ellipse[1], 'r-', alpha=0.15, linewidth=1)

# 第一对（灰色，表示已找到）
ax4.annotate('', xy=v1, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.4))
ax4.annotate('', xy=Av1, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.4))
ax4.text(v1[0]+0.05, v1[1]+0.15, 'v₁ (已找到)', fontsize=9, color='gray', alpha=0.6)

# v₂（必须与 v₁ 正交）
ax4.annotate('', xy=v2, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax4.text(v2[0]-0.6, v2[1]+0.15, 'v₂ (输入)\n⊥ v₁', fontsize=12, color='blue', fontweight='bold')

# A·v₂ = σ₂·u₂
Av2 = A @ v2
ax4.annotate('', xy=Av2, xytext=[0,0],
             arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))
ax4.text(Av2[0]+0.1, Av2[1]+0.15,
         f'Av₂ = σ₂u₂\n(长度={sigma2:.2f})', fontsize=11, color='orange', fontweight='bold')

# 画虚线连接 v₂ 和 Av₂
ax4.annotate('', xy=Av2, xytext=v2,
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                            linestyle='dashed'))
ax4.text((v2[0]+Av2[0])/2-0.5, (v2[1]+Av2[1])/2+0.15,
         f'A 变换\n放大 {sigma2:.2f} 倍', fontsize=10, color='green')

# 画正交标记
angle_size = 0.15
ax4.plot([v1[0]*angle_size + v2[0]*angle_size,
          v2[0]*angle_size,
          v2[0]*angle_size - v1[0]*angle_size + v2[0]*angle_size],  # not right, simplified
         [v1[1]*angle_size + v2[1]*angle_size,
          v2[1]*angle_size,
          v2[1]*angle_size - v1[1]*angle_size + v2[1]*angle_size],
         'b-', linewidth=1)

ax4.axhline(y=0, color='gray', linewidth=0.5)
ax4.axvline(x=0, color='gray', linewidth=0.5)
ax4.set_xlim(-4, 4)
ax4.set_ylim(-4, 4)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/data/YBJ/cleansight/docs/svd_visualization_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("图1 已保存: docs/svd_visualization_basic.png")


# ============================================================
# 图2：逐角度搜索的动画式展示
# ============================================================
fig2, axes = plt.subplots(2, 3, figsize=(21, 14))

# 选几个代表性角度 + v₁ + v₂
search_angles_deg = [0, 30, angle_v1, 90, 135, angle_v2]
search_labels = ['b at 0°', 'b at 30°', f'b at {angle_v1:.0f}° = v₁ ★',
                 'b at 90°', 'b at 135°', f'b at {angle_v2:.0f}° = v₂ ★']
colors_list = ['#666666', '#666666', '#FF0000', '#666666', '#666666', '#FF8800']

for idx, (ang_deg, label, col) in enumerate(zip(search_angles_deg, search_labels, colors_list)):
    ax = axes[idx // 3, idx % 3]
    ax.set_aspect('equal')

    ang = np.radians(ang_deg)
    b = np.array([np.cos(ang), np.sin(ang)])
    Ab = A @ b
    norm_Ab = np.linalg.norm(Ab)
    a_best = Ab / norm_Ab if norm_Ab > 1e-10 else Ab

    # 画单位圆和椭圆
    ax.plot(circle[0], circle[1], 'b--', alpha=0.15, linewidth=1)
    ax.plot(ellipse[0], ellipse[1], 'r-', alpha=0.15, linewidth=1)

    # 画 b 方向
    ax.annotate('', xy=b, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(b[0]+0.1, b[1]+0.15, 'b', fontsize=13, color='blue', fontweight='bold')

    # 画 Ab
    ax.annotate('', xy=Ab, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # 画最佳 a 方向（单位长度）
    ax.annotate('', xy=a_best, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='dashed'))
    ax.text(a_best[0]+0.1, a_best[1]-0.3, f'a*', fontsize=12, color='green', fontweight='bold')

    is_special = '★' in label
    title_color = col if is_special else 'black'
    ax.set_title(f'{label}\naᵀAb = ||Ab|| = {norm_Ab:.2f}',
                fontsize=12, fontweight='bold' if is_special else 'normal',
                color=title_color)

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)

plt.suptitle('遍历所有输入方向 b，找使 ||Ab|| 最大的方向 → 那就是 v₁\n'
             '(红色箭头 = Ab，绿色虚线 = 最佳输出方向 a*，蓝色 = 输入方向 b)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/data/YBJ/cleansight/docs/svd_visualization_search.png', dpi=150, bbox_inches='tight')
plt.close()
print("图2 已保存: docs/svd_visualization_search.png")


# ============================================================
# 图3：ΔW 场景 —— 触发器方向 vs 正常方向
# ============================================================
fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))

# 构造一个模拟的 ΔW (2D 简化版)
# v₁ 方向对应触发器检测，σ₁ 大
# v₂ 方向对应正常特征调整，σ₂ 小
np.random.seed(42)
sigma_trig = 4.0
sigma_normal = 1.0
v_trig = np.array([0.6, 0.8])   # 触发器方向
v_normal = np.array([-0.8, 0.6]) # 正交方向
u_trig = np.array([0.9, 0.436])
u_trig = u_trig / np.linalg.norm(u_trig)
u_normal = np.array([-0.436, 0.9])
u_normal = u_normal / np.linalg.norm(u_normal)

DeltaW = sigma_trig * np.outer(u_trig, v_trig) + sigma_normal * np.outer(u_normal, v_normal)

# 左图：触发器输入
ax_left.set_aspect('equal')
ax_left.set_title('触发器输入 x_trig\n(与 v₁ 方向对齐 → 强响应)', fontsize=13, fontweight='bold')

x_trig = v_trig * 0.9 + v_normal * 0.1  # 主要在 v_trig 方向
x_trig = x_trig / np.linalg.norm(x_trig)

# v₁, v₂ 方向
ax_left.annotate('', xy=v_trig*1.2, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
ax_left.text(v_trig[0]*1.2+0.1, v_trig[1]*1.2, 'v₁ (触发器检测方向)',
            fontsize=10, color='gray')

ax_left.annotate('', xy=v_normal*1.2, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
ax_left.text(v_normal[0]*1.2-0.5, v_normal[1]*1.2+0.1, 'v₂',
            fontsize=10, color='gray')

# 输入
ax_left.annotate('', xy=x_trig, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax_left.text(x_trig[0]+0.1, x_trig[1]+0.1, 'x_trig', fontsize=13, color='blue', fontweight='bold')

# 输出
y_trig = DeltaW @ x_trig
ax_left.annotate('', xy=y_trig, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax_left.text(y_trig[0]+0.1, y_trig[1]+0.1,
            f'ΔW·x_trig\n(长度={np.linalg.norm(y_trig):.2f})',
            fontsize=11, color='red', fontweight='bold')

# 分量标注
proj_trig = np.dot(x_trig, v_trig)
proj_normal = np.dot(x_trig, v_normal)
ax_left.text(-2.5, -2.5,
            f'v₁ᵀx = {proj_trig:.2f} (大!) → σ₁=={sigma_trig:.0f} 倍放大\n'
            f'v₂ᵀx = {proj_normal:.2f} (小) → σ₂={sigma_normal:.0f} 倍放大',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax_left.set_xlim(-4, 5)
ax_left.set_ylim(-3, 5)
ax_left.axhline(y=0, color='gray', linewidth=0.5)
ax_left.axvline(x=0, color='gray', linewidth=0.5)
ax_left.grid(True, alpha=0.3)

# 右图：正常输入
ax_right.set_aspect('equal')
ax_right.set_title('正常输入 x_clean\n(与 v₁ 几乎正交 → 弱响应)', fontsize=13, fontweight='bold')

x_clean = v_trig * 0.1 + v_normal * 0.9
x_clean = x_clean / np.linalg.norm(x_clean)

# v₁, v₂ 方向
ax_right.annotate('', xy=v_trig*1.2, xytext=[0,0],
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
ax_right.text(v_trig[0]*1.2+0.1, v_trig[1]*1.2, 'v₁ (触发器检测方向)',
             fontsize=10, color='gray')

ax_right.annotate('', xy=v_normal*1.2, xytext=[0,0],
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
ax_right.text(v_normal[0]*1.2-0.5, v_normal[1]*1.2+0.1, 'v₂',
             fontsize=10, color='gray')

# 输入
ax_right.annotate('', xy=x_clean, xytext=[0,0],
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax_right.text(x_clean[0]-0.8, x_clean[1]+0.15, 'x_clean', fontsize=13, color='blue', fontweight='bold')

# 输出
y_clean = DeltaW @ x_clean
ax_right.annotate('', xy=y_clean, xytext=[0,0],
                 arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
ax_right.text(y_clean[0]+0.1, y_clean[1]+0.1,
             f'ΔW·x_clean\n(长度={np.linalg.norm(y_clean):.2f})',
             fontsize=11, color='green', fontweight='bold')

proj_trig_c = np.dot(x_clean, v_trig)
proj_normal_c = np.dot(x_clean, v_normal)
ax_right.text(-2.5, -2.5,
             f'v₁ᵀx = {proj_trig_c:.2f} (小) → σ₁={sigma_trig:.0f} 倍但信号弱\n'
             f'v₂ᵀx = {proj_normal_c:.2f} (大) → σ₂={sigma_normal:.0f} 倍放大',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

ax_right.set_xlim(-4, 5)
ax_right.set_ylim(-3, 5)
ax_right.axhline(y=0, color='gray', linewidth=0.5)
ax_right.axvline(x=0, color='gray', linewidth=0.5)
ax_right.grid(True, alpha=0.3)

plt.suptitle('ΔW 的 SVD：v₁ 是"触发器检测器"，σ₁ 是放大倍数',
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/data/YBJ/cleansight/docs/svd_visualization_trigger.png', dpi=150, bbox_inches='tight')
plt.close()
print("图3 已保存: docs/svd_visualization_trigger.png")

print("\n全部可视化完成！")
