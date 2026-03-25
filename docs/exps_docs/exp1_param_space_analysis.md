# Coding Prompt：实验一 · 参数空间差异分析

## 任务背景

我们研究 LLaVA-1.5-7B 中 projector 层面的后门防御问题。

**Projector 结构：** 两层 MLP，具体为：
- `Linear1`：输入维度 1024（CLIP ViT-L/14 输出），输出维度 4096（LLM hidden dim）
- `GELU` 激活
- `Linear2`：输入维度 4096，输出维度 4096

每层的权重矩阵分别记为 `W1_*` 和 `W2_*`。

**我们持有三组 projector 权重：**

| 标识 | 说明 | 位置 |
|------|------|------|
| `W_clean` | 开源 LLaVA-1.5-7B 原始 projector 权重（未微调） | models/llava-1.5-7b-hf/mm_projector_extracted.bin |
| `W_backdoor` | 用含 BadNet trigger的投毒数据（10% 投毒率）微调后的权重，目标输出为 `"you have been hacked lol"` | model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth |
| `W_benign_ft` | 用等量干净数据正常微调后的权重（对照组，训练设置完全相同，只是数据无毒） | model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth|

**实验核心问题：**
- `ΔW_backdoor = W_backdoor - W_clean` 是否呈现低秩结构？
- `ΔW_backdoor` 和 `ΔW_benign = W_benign_ft - W_clean` 的秩结构有无显著差异？
- 两个 ΔW 的主要奇异方向是否对齐，还是正交？（若正交，说明后门方向和正常适配方向可以被分离）

---

## 实现任务

请实现一个完整的 Python 脚本 `exp1_param_analysis.py`，完成以下分析。放在exps/exp1_W_analysis目录下。

### 第一步：加载权重，计算 ΔW

从三个 checkpoint 中分别提取 projector 的两个线性层权重矩阵。对每个层（Layer1 和 Layer2），分别计算：

```
ΔW_backdoor[layer] = W_backdoor[layer] - W_clean[layer]
ΔW_benign[layer]   = W_benign_ft[layer] - W_clean[layer]
```

注意：权重矩阵的形状分别是 `[4096, 1024]`（Layer1）和 `[4096, 4096]`（Layer2），直接对这两个矩阵做 SVD。

---

### 第二步：SVD 分解与奇异值分布分析

对以上四个 ΔW 矩阵（两个 layer × 两种 ΔW）分别做完整 SVD，得到奇异值序列。

**指标一：奇异值分布曲线图**

对每个 layer，在同一张图中画出两条曲线（backdoor 和 benign ft 的奇异值）：
- x 轴：奇异值序号（按降序排列，从 1 开始）
- y 轴：奇异值大小
- 同时画出 linear scale 和 log scale 两个版本
- 注意：两个 layer 的奇异值数量不同（Layer1 最多 1024 个，Layer2 最多 4096 个），分开画

**指标二：Top-k 能量占比**

对每个 ΔW，计算前 k 个奇异值的能量（平方和）占总能量的比例：

```
energy_ratio(k) = sum(S[:k]^2) / sum(S^2)
```

对 k = 1, 2, 3, 5, 10, 20, 50 分别计算，结果存为表格，行是 k 值，列是四种 ΔW（backdoor-layer1, backdoor-layer2, benign-layer1, benign-layer2）。

**指标三：有效秩（Effective Rank）**

有效秩定义为奇异值归一化后的信息熵的指数：

```
p_i = S_i / sum(S)
effective_rank = exp( -sum(p_i * log(p_i)) )
```

对四种 ΔW 分别计算有效秩，输出对比表。

---

### 第三步：顶部奇异方向的对齐分析（关键）

这是连接"参数空间"和"方法设计"的核心分析。

**目标：** 判断 `ΔW_backdoor` 和 `ΔW_benign` 的主奇异方向是否正交或对齐。

**做法：**

对每个 layer，取 `ΔW_backdoor` 的 top-k 右奇异向量（`V_backdoor[:, :k]`，来自 SVD），和 `ΔW_benign` 的 top-k 右奇异向量（`V_benign[:, :k]`）。

计算两组向量张成的子空间之间的对齐程度，用 principal angles 来衡量：

```
对齐矩阵 M = V_backdoor[:, :k].T @ V_benign[:, :k]   # shape: [k, k]
对 M 做 SVD，得到奇异值 σ_1, ..., σ_k
这些奇异值就是两个子空间之间 principal angles 的余弦值
```

对 k = 1, 3, 5 分别计算，输出 principal angles（以角度表示，`arccos(σ)` 转换为度）。

**解读原则：**
- 如果 principal angle 接近 90°（余弦值接近 0）：两个方向几乎正交 → **后门方向和正常适配方向可分离，对我们的方法非常有利**
- 如果 principal angle 接近 0°（余弦值接近 1）：两个方向高度重叠 → 分离困难

---

### 第四步：ΔW_backdoor 的稀疏性分析

**目标：** 看后门修改是否集中在少数参数上（稀疏性），还是弥散在整个矩阵。

对每个 layer 的 `ΔW_backdoor`（展平为一维向量后）：

- 画出参数绝对值的分布直方图（log scale y 轴）
- 计算 Gini 系数（衡量不均匀程度）：
  ```
  将 |ΔW| 排序为 w_1 ≤ w_2 ≤ ... ≤ w_n
  Gini = 1 - 2 * sum_{i=1}^{n} (n - i + 1) * w_i / (n * sum(w_i))
  ```
  Gini 系数越接近 1，说明能量越集中（越稀疏）
- 计算 L1 范数 / L2 范数的比值（另一种稀疏性指标，比值越大说明越均匀分布，越小说明越集中）
- 对比 `ΔW_backdoor` 和 `ΔW_benign` 在同一 layer 上的这些指标

---

## 输出要求

**图表（保存到 `exps/exp1_W_analysis/`）：**
- `exp1_singular_values_layer1_linear.png` — Layer1 奇异值分布（linear scale）
- `exp1_singular_values_layer1_log.png` — Layer1 奇异值分布（log scale）
- `exp1_singular_values_layer2_linear.png` — Layer2 奇异值分布（linear scale）
- `exp1_singular_values_layer2_log.png` — Layer2 奇异值分布（log scale）
- `exp1_param_distribution_layer1.png` — Layer1 参数绝对值分布直方图
- `exp1_param_distribution_layer2.png` — Layer2 参数绝对值分布直方图

**数值结果（保存到 `exps/exp1_W_analysis/`，格式为 JSON）：**
- `exp1_energy_ratio.json` — Top-k 能量占比表
- `exp1_effective_rank.json` — 有效秩对比
- `exp1_principal_angles.json` — 主角分析结果 这里有后面的方向角度大，排名靠前的3、4个方向角度小。
- `exp1_sparsity.json` — 稀疏性指标（Gini 系数、L1/L2 比值）

**终端输出：** 每个指标计算完毕后实时 print 到终端，方便快速浏览结果。

---

## 输入接口

在脚本中硬编码三个 checkpoint 路径：

LLaVA-1.5-7B 的 projector 权重在 checkpoint 中的 key 为：
- `model.mm_projector.0.weight`（Linear1 的权重）
- `model.mm_projector.2.weight`（Linear2 的权重）

如果实际 key 不同，请在脚本开头加一个 `print(state_dict.keys())` 供用户确认，然后提供一个 `--projector_prefix` 参数用于灵活指定。

---

## 环境依赖（可先忽略，代码写好后用户安装）

- `torch`（用于加载权重、SVD）
- `numpy`
- `matplotlib`
- `scipy`（可选，用于验证 SVD 结果）

不需要加载完整的 LLaVA 模型，只需加载 state dict 并提取 projector 相关 key 即可，显存需求极低。

---

## 预期结果解读（供参考）

运行完毕后，我们期望看到以下两种情况之一，并根据结果推进后续方法设计：

**情况 A（最理想）：** `ΔW_backdoor` 的 top-1 或 top-3 能量占比远高于 `ΔW_benign`（如 backdoor 的 top-3 占 80%+，benign 的 top-3 仅占 30% 以下），且两者 principal angle 接近 90°。
→ 结论：后门是低秩且方向独立的扰动，可以通过移除 ΔW_backdoor 的 top singular directions 来净化，同时保留正常适配。

**情况 B（次理想）：** `ΔW_backdoor` 低秩性明显，但与 `ΔW_benign` 的 principal angle 不完全正交（如 30°-60°）。
→ 结论：后门方向和正常适配有部分重叠，需要更精细的分离方法（如加权投影）。

**情况 C（最难处理）：** 两者秩相近，方向也高度重叠。
→ 结论：纯参数空间分析不够，需要结合表征空间分析（实验二）来找其他 signal。
