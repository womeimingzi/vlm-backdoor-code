# 实验一：参数空间差异分析结果报告

## 实验背景

分析 LLaVA-1.5-7B projector 层（两层 MLP）在三种权重状态下的差异：

| 标识 | 说明 |
|------|------|
| W_clean | 开源原始 projector 权重（未微调） |
| W_backdoor | 含 BadNet trigger 的投毒数据（10% 投毒率）微调后的权重，目标输出为 "you have been hacked lol" |
| W_benign_ft | 等量干净数据正常微调后的权重（对照组，训练设置完全相同） |

Projector 结构：Linear1 [4096×1024] → GELU → Linear2 [4096×4096]

分析对象：
- `ΔW_backdoor = W_backdoor - W_clean`
- `ΔW_benign = W_benign_ft - W_clean`

---

## Step 1：ΔW Frobenius 范数

| | Layer1 | Layer2 |
|---|---|---|
| ΔW_backdoor | 3.7065 | 8.3384 |
| ΔW_benign | 3.6505 | 8.1296 |

**观察：** 两者量级几乎相同，后门微调与干净微调对权重的整体修改幅度无显著差异。

---

## Step 2：奇异值分布

### Top-5 奇异值

| | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| ΔW_backdoor Layer1 | 1.5589 | 1.1450 | 0.8971 | 0.8013 | 0.6278 |
| ΔW_benign Layer1   | 1.6234 | 1.1431 | 0.8616 | 0.6387 | 0.5026 |
| ΔW_backdoor Layer2 | 3.8499 | 3.2847 | 2.1145 | 1.7208 | 1.5937 |
| ΔW_benign Layer2   | 3.9159 | 3.2637 | 2.0701 | 1.6170 | 1.4363 |

**观察：** 奇异值分布形态高度相似，backdoor 与 benign 的最大奇异值量级相当，无明显分离。

---

## Step 3：Top-k 能量占比

`energy_ratio(k) = sum(S[:k]²) / sum(S²)`

| k | bd-Layer1 | bd-Layer2 | bn-Layer1 | bn-Layer2 |
|---|---|---|---|---|
| 1  | 0.1769 | 0.2129 | 0.1978 | 0.2318 |
| 2  | 0.2724 | 0.3679 | 0.2959 | 0.3928 |
| 3  | 0.3310 | 0.4321 | 0.3516 | 0.4576 |
| 5  | 0.4064 | 0.5112 | 0.4012 | 0.5283 |
| 10 | 0.4718 | 0.6016 | 0.4632 | 0.6018 |
| 20 | 0.5442 | 0.6650 | 0.5362 | 0.6636 |
| 50 | 0.6655 | 0.7528 | 0.6597 | 0.7511 |

**观察：**
- Top-3 能量占比：backdoor Layer1 约 33%，benign Layer1 约 35%；backdoor Layer2 约 43%，benign Layer2 约 46%
- 两者能量分布几乎一致，backdoor **没有**表现出更强的低秩集中性
- 需要 top-50 才能覆盖约 65-75% 的能量，说明变化弥散在大量奇异方向上

---

## Step 4：有效秩（Effective Rank）

`effective_rank = exp(-sum(p_i * log(p_i)))`, `p_i = S_i / sum(S)`

| | Layer1 | Layer2 |
|---|---|---|
| ΔW_backdoor | 694.53 | 1633.81 |
| ΔW_benign   | 698.45 | 1634.39 |

**观察：**
- 有效秩极高（Layer1 约 694，Layer2 约 1634），远非低秩结构
- backdoor 与 benign 的有效秩几乎相同（差异 < 0.6%）
- 后门注入并未使权重变化呈现低秩特性

---

## Step 5：主角分析（Principal Angles）

衡量 ΔW_backdoor 与 ΔW_benign 的 top-k 右奇异子空间之间的对齐程度。
角度接近 0° = 方向高度对齐；接近 90° = 方向正交可分离。

### Layer1

| k | 主角（度） |
|---|---|
| 1 | [5.855°] |
| 3 | [3.488°, 5.385°, 55.019°] |
| 5 | [2.926°, 5.021°, 8.032°, 28.944°, 89.573°] |

### Layer2

| k | 主角（度） |
|---|---|
| 1 | [8.016°] |
| 3 | [4.427°, 8.177°, 11.585°] |
| 5 | [4.048°, 4.421°, 6.439°, 9.555°, 74.647°] |

**观察：**
- top-1 主角仅 5.9°（L1）和 8.0°（L2），接近 0°，说明两者最主要的奇异方向**高度对齐**
- top-3 中前两个方向仍高度对齐（< 10°），第三个方向开始出现分离（L1 达 55°）
- top-5 中出现接近 90° 的方向，但这些是能量较小的次要方向
- **结论：后门方向与正常适配方向在主要奇异子空间中无法分离**

---

## Step 6：稀疏性分析

### Gini 系数（越接近 1 越稀疏集中）

| | Layer1 | Layer2 |
|---|---|---|
| ΔW_backdoor | 0.4067 | 0.4002 |
| ΔW_benign   | 0.4048 | 0.3984 |

### L1/L2 比值（越小越集中）

| | Layer1 | Layer2 |
|---|---|---|
| ΔW_backdoor | 1648.69 | 3323.71 |
| ΔW_benign   | 1652.81 | 3330.60 |

**观察：**
- Gini 系数约 0.40，属于中等程度的不均匀分布，非高度稀疏
- backdoor 与 benign 的稀疏性指标几乎相同（差异 < 0.5%）
- 后门修改并未集中在少数参数上，而是弥散在整个矩阵

---

## 综合结论

**本实验结果对应文档中的情况 C（最难处理）。**

| 判断维度 | 结果 | 含义 |
|---|---|---|
| ΔW_backdoor 是否低秩？ | 否（有效秩 ~694/1634） | 后门修改弥散，无低秩结构 |
| backdoor vs benign 能量分布是否有差异？ | 否（几乎完全一致） | 无法通过能量集中度区分 |
| 主奇异方向是否正交？ | 否（top-1 主角 < 9°，高度对齐） | 后门方向与正常适配方向无法分离 |
| 参数稀疏性是否有差异？ | 否（Gini 差异 < 0.5%） | 无法通过参数分布区分 |

**核心发现：** 在 projector 参数空间中，10% 投毒率的 BadNet 后门微调与等量干净数据微调在所有分析维度上均无显著差异。纯参数空间分析不足以检测或定位后门信号。

**后续方向：** 需要进入表征空间分析（实验二），在激活值/隐层表征层面寻找后门触发器引起的可区分信号。

---

## 输出文件列表

| 文件 | 内容 |
|---|---|
| `exp1_singular_values_layer1_linear.png` | Layer1 奇异值分布（linear scale） |
| `exp1_singular_values_layer1_log.png` | Layer1 奇异值分布（log scale） |
| `exp1_singular_values_layer2_linear.png` | Layer2 奇异值分布（linear scale） |
| `exp1_singular_values_layer2_log.png` | Layer2 奇异值分布（log scale） |
| `exp1_param_distribution_layer1.png` | Layer1 参数绝对值分布直方图 |
| `exp1_param_distribution_layer2.png` | Layer2 参数绝对值分布直方图 |
| `exp1_energy_ratio.json` | Top-k 能量占比表 |
| `exp1_effective_rank.json` | 有效秩对比 |
| `exp1_principal_angles.json` | 主角分析结果 |
| `exp1_sparsity.json` | 稀疏性指标 |
