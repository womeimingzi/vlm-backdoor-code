# Coding Prompt：实验 1b · 正交方向投影去除验证

## 任务背景

实验一的 principal angles 分析发现：在 `ΔW_backdoor` 和 `ΔW_benign` 的 top-k 子空间中，**始终存在 1~2 个接近 90° 的正交方向**，且该数量不随 k 增大而增长（k=5 到 k=50 均为 1~2 个）。

这表明后门可能并非"低秩"的，但在方向上是**独立**的——后门信号集中在一个与正常微调正交的低维子空间中。

本实验验证这一假设：**提取该正交方向，通过投影去除来净化 projector，观察后门是否被消除、正常性能是否保留。**

---

## 实验设计

### 第一步：提取正交方向

对每个 layer，从 principal angles 分析中提取 backdoor 子空间中与 benign 子空间最正交的方向。

**具体做法：**

```python
# 取 top-k 右奇异向量（k 为超参数，见下文）
# Vh_bd: [k, input_dim], Vh_bn: [k, input_dim]（来自 SVD 的 Vh 矩阵前 k 行）
V_bd = Vh_bd[:k, :].T  # [input_dim, k]
V_bn = Vh_bn[:k, :].T  # [input_dim, k]

# 计算对齐矩阵
M = V_bd.T @ V_bn       # [k, k]
U_M, sigma_M, Vt_M = torch.linalg.svd(M)

# 最后一列对应最大 principal angle（最正交方向）
# 将其映射回原始输入空间
d = V_bd @ U_M[:, -1]   # [input_dim] 单位向量
```

其中 `d` 就是 backdoor 子空间中与 benign 子空间最正交的方向（在输入空间中的表示）。

对多个 k 值（k = 5, 10, 20, 50）分别提取 `d`，后续实验对每个 k 都做验证。

### 第二步：投影去除

用提取的方向 `d` 构造净化后的权重：

```
W_purified = W_backdoor - ΔW_backdoor · d · dᵀ
```

等价于：

```
ΔW_purified = ΔW_backdoor · (I - d · dᵀ)
W_purified = W_clean + ΔW_purified
```

数学含义：去掉 `ΔW_backdoor` 中专门响应 `d` 方向输入的部分，保留其余所有方向的功能。

**变体（可选，如果单方向不够）：** 如果 k=50 时存在 2 个正交方向（如 Layer2 的结果），可以提取最正交的 2 个方向 $d_1, d_2$，构造投影矩阵 $P = I - D \cdot D^T$（其中 $D = [d_1, d_2]$），一次去除多个方向。

### 第三步：评估净化效果

用 `W_purified` 替换模型的 projector 权重，评估两个指标，ASR和CIDEr。

**对比基线：**

| 模型 | 说明 |
|------|------|
| `W_backdoor`（未净化） | 上界：ASR 应接近 100% |
| `W_clean`（原始干净） | 参考：ASR 应为 0%，CIDEr 为基准 |
| `W_purified`（投影去除后） | 我们的方法 |

### 第四步：能量分析

计算被投影去除的部分占总修改量的比例，验证后门方向的能量大小：

```python
energy_removed = torch.norm(dW_backdoor @ d) ** 2
energy_total = torch.norm(dW_backdoor, 'fro') ** 2
ratio = energy_removed / energy_total
```

对每个 k 值和每个 layer 分别计算，输出对比表。

### 第五步：正交方向稳定性验证

验证不同 k 值提取出的正交方向 `d` 是否指向同一方向：

```python
# 计算不同 k 值得到的 d 之间的余弦相似度
cos_sim = abs(d_k5.T @ d_k10)
```

对所有 k 值的 d 两两计算余弦相似度，输出矩阵。若相似度高（>0.9），说明后门方向是稳定存在的，不是 k 选择的产物。

---

## 输出要求

**数值结果（保存到 `exps/exp1b_projection/`，格式为 JSON）：**

- `exp1b_energy_removed.json` — 各 k 值下被去除部分的能量占比
- `exp1b_direction_stability.json` — 不同 k 值提取的 d 之间的余弦相似度矩阵
- `exp1b_evaluation.json` — ASR 和 CIDEr 结果

**终端输出：** 每个步骤完成后实时 print 关键数值。

---

## 超参数

| 参数 | 取值 | 说明 |
|------|------|------|
| k（子空间维度） | 5, 10, 20, 50 | 提取正交方向时使用的 top-k |
| 去除方向数 | 1（默认），2（Layer2 可选） | 投影去除的正交方向个数 |

---

## 预期结果

**理想结果：**

- 不同 k 提取的 `d` 高度一致（余弦相似度 > 0.9）→ 后门方向稳定存在
- 被去除能量占比很小（< 5%）→ 后门只需极少能量即可运作
- ASR 显著下降（< 10%）→ 投影去除成功消除后门
- Clean Accuracy 基本不变（下降 < 2%）→ 正常功能未受损

**若失败（ASR 仍高）：** 可能原因包括：
- 后门信号不完全在单一正交方向上，需要去除更多方向
- 后门信号分布在 top-k 子空间之外
- 需要在左奇异向量 U（输出空间）侧同时做投影去除

---

## 环境依赖

利用codebase里的已有代码，使用CLAUDE.md中指定的python环境。