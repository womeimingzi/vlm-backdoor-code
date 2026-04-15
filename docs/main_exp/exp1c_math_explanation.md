# Exp1c 方法的数学原理——从线性代数到后门净化

> 本文档假设读者学过线性代数但已遗忘较多，因此会从基础概念一步步推导。

---

## 0. 全文思路地图

我们最终要做的事情用一句话概括：

> **找到后门模型权重更新里"只有后门需要、正常任务不需要"的方向，把它去掉。**

实现这个目标需要以下数学工具链：

```
向量 → 矩阵 → SVD 分解 → 子空间 → 主角 → 正交方向提取 → 投影去除
```

下面一步一步来。

---

## 1. 预备知识：向量与内积

### 1.1 向量

一个 n 维向量就是 n 个数排成一列：

$$
\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}
$$

它可以看作 n 维空间中的一个箭头（方向 + 长度）。

### 1.2 向量的长度（范数）

$$
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

长度为 1 的向量叫 **单位向量**。任何非零向量除以自己的长度就变成单位向量：$\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$。

### 1.3 内积（点积）

两个向量的内积：

$$
\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta
$$

其中 $\theta$ 是两个向量之间的夹角。

**关键性质**：

| 内积值 | 含义 |
|--------|------|
| $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|$（$\theta = 0°$） | 方向完全相同 |
| $\mathbf{u} \cdot \mathbf{v} = 0$（$\theta = 90°$） | **正交**（完全无关） |
| $\mathbf{u} \cdot \mathbf{v} = -\|\mathbf{u}\|\|\mathbf{v}\|$（$\theta = 180°$） | 方向完全相反 |

对两个单位向量，$\hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = \cos\theta$，这就是 **余弦相似度（cosine similarity）**。

### 1.4 投影

把向量 $\mathbf{u}$ 往方向 $\hat{\mathbf{d}}$（单位向量）上 **投影**：

$$
\text{proj}_{\mathbf{d}}(\mathbf{u}) = (\mathbf{u} \cdot \hat{\mathbf{d}}) \, \hat{\mathbf{d}}
$$

几何意义：$\mathbf{u}$ 的影子落在 $\hat{\mathbf{d}}$ 方向上有多长。

**去掉投影**：$\mathbf{u} - \text{proj}_{\mathbf{d}}(\mathbf{u})$ 就是 $\mathbf{u}$ 中**与 $\hat{\mathbf{d}}$ 方向无关**的部分。

> **这个"去掉投影"操作就是我们最终要对后门方向做的事情，只不过要从向量推广到矩阵。**

---

## 2. 矩阵与线性变换

### 2.1 矩阵是什么

一个 $m \times n$ 的矩阵 $W$ 定义了一个线性变换：输入一个 $n$ 维向量，输出一个 $m$ 维向量。

$$
\mathbf{y} = W \mathbf{x}, \quad W \in \mathbb{R}^{m \times n}, \; \mathbf{x} \in \mathbb{R}^{n}, \; \mathbf{y} \in \mathbb{R}^{m}
$$

在我们的场景中：

- LLaVA 的 projector Layer1：$W_1 \in \mathbb{R}^{4096 \times 1024}$，把 1024 维的视觉特征映射到 4096 维
- LLaVA 的 projector Layer2：$W_2 \in \mathbb{R}^{4096 \times 4096}$，4096 维到 4096 维

### 2.2 权重更新量 ΔW

后门攻击只修改了 projector 的权重。更新量：

$$
\Delta W = W_{\text{backdoor}} - W_{\text{clean}}
$$

$\Delta W$ 也是一个矩阵，它**编码了微调过程中学到的所有信息**——包括正常任务适配和后门 shortcut。

---

## 3. SVD 分解：理解矩阵的"内部结构"

### 3.1 直觉：矩阵 = 多个方向上的拉伸

任何矩阵 $A \in \mathbb{R}^{m \times n}$ 都可以被分解为：

$$
A = U \Sigma V^\top
$$

其中：

| 组成部分 | 形状 | 含义 |
|---------|------|------|
| $U$ | $m \times r$ | 输出空间的正交基（$r$ 个方向） |
| $\Sigma$ | $r \times r$ 对角阵 | 每个方向上的"拉伸量"（奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots$） |
| $V^\top$ | $r \times n$ | 输入空间的正交基（$r$ 个方向） |

$r = \text{rank}(A)$ 是矩阵的秩（有多少个独立的"拉伸方向"）。

### 3.2 几何直觉

SVD 告诉你：矩阵 $A$ 这个变换可以拆成三步：

1. **旋转/反射** 输入（$V^\top$）：把输入向量转到一组特殊方向上
2. **拉伸**（$\Sigma$）：在每个方向上分别缩放
3. **旋转/反射** 到输出空间（$U$）：转到输出坐标系

$\sigma_1$ 最大的方向是 $A$ "最用力"的方向，$\sigma_r$ 最小的方向是 $A$ "最弱"的方向。

### 3.3 对 ΔW 做 SVD 的意义

对权重更新量做 SVD：

$$
\Delta W = U \Sigma V^\top
$$

$V$ 的列（或 $V^\top$ 的行）告诉我们：**在输入空间中，哪些方向上的输入会导致最大的权重变化效果**。

- $V^\top$ 的第 1 行（$\mathbf{v}_1$）：ΔW 变化最剧烈的输入方向
- $V^\top$ 的第 2 行（$\mathbf{v}_2$）：次大的变化方向
- ...以此类推

取前 $k$ 个方向 $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$，它们张成一个 $k$ 维 **子空间**，涵盖了 $\Delta W$ 的主要变化方向。

### 3.4 代码对应

```python
_, _, Vh_bd = torch.linalg.svd(dW_bd, full_matrices=False)
# Vh_bd 的形状: [min(m,n), n]
# Vh_bd[0, :] 就是 v_1（第一个右奇异向量）
# Vh_bd[1, :] 就是 v_2
# ...
```

---

## 4. 子空间：一组方向张成的"平面"

### 4.1 什么是子空间

取 $k$ 个线性无关的向量 $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$，它们所有线性组合 $c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k$ 构成的集合叫做一个 **$k$ 维子空间**。

- $k=1$：一条线
- $k=2$：一个平面
- $k=3$：一个三维空间
- ...

### 4.2 两个子空间

我们分别对 **后门** ΔW 和 **良性** ΔW 做 SVD，取前 $k$ 个方向，得到两个子空间：

$$
\mathcal{S}_{\text{bd}} = \text{span}(\mathbf{v}_1^{\text{bd}}, \dots, \mathbf{v}_k^{\text{bd}})
$$

$$
\mathcal{S}_{\text{bn}} = \text{span}(\mathbf{v}_1^{\text{bn}}, \dots, \mathbf{v}_k^{\text{bn}})
$$

**核心问题**：这两个子空间有多"相似"？有没有某些方向只在 $\mathcal{S}_{\text{bd}}$ 中而不在 $\mathcal{S}_{\text{bn}}$ 中？

如果有，那些方向就是后门特有的。

---

## 5. 主角（Principal Angles）：衡量两个子空间的关系

### 5.1 两个子空间之间的角度

两条线之间有一个夹角；两个子空间之间有**多个**角度，叫做 **主角（principal angles）**。

对于两个 $k$ 维子空间，有 $k$ 个主角 $\theta_1 \leq \theta_2 \leq \cdots \leq \theta_k$：

| 主角 | 含义 |
|------|------|
| $\theta_1$（最小主角） | 两个子空间中"最接近"的两个方向之间的夹角 |
| $\theta_k$（最大主角） | 两个子空间中"最远"的两个方向之间的夹角 |

**特殊情况**：

- 如果所有 $\theta_i = 0°$：两个子空间完全重合
- 如果存在 $\theta_i = 90°$：$\mathcal{S}_{\text{bd}}$ 中有某个方向，**完全不在** $\mathcal{S}_{\text{bn}}$ 里

### 5.2 如何计算主角

设 $V_{\text{bd}} \in \mathbb{R}^{n \times k}$ 和 $V_{\text{bn}} \in \mathbb{R}^{n \times k}$ 是两个子空间的正交基矩阵（列为基向量）。

1. 计算 $M = V_{\text{bd}}^\top V_{\text{bn}}$，得到一个 $k \times k$ 矩阵
2. 对 $M$ 做 SVD：$M = U_M \, \Sigma_M \, V_M^\top$
3. $\Sigma_M$ 的对角元素 $\sigma_i$ 就是主角的余弦值：$\cos\theta_i = \sigma_i$
4. 主角 $\theta_i = \arccos(\sigma_i)$

为什么？直觉：$M = V_{\text{bd}}^\top V_{\text{bn}}$ 度量的是"一个子空间的基在另一个子空间上的投影"。SVD 找到的是使投影最大/最小的方向对。

### 5.3 代码对应

```python
V_bd = Vh_bd[:k, :].T  # [input_dim, k] — 取前k个右奇异向量，转置成列
V_bn = Vh_bn[:k, :].T  # [input_dim, k]

M = V_bd.T @ V_bn       # [k, k] — 两个子空间基的互投影
U_M, sigma, _ = torch.linalg.svd(M)
# sigma[i] = cos(θ_i)

angles = torch.acos(sigma) * (180 / π)
# angles[0] 最小（最接近的一对方向）
# angles[k-1] 最大（最正交的一对方向）
```

---

## 6. 提取后门特有方向

### 6.1 哪些方向是后门特有的？

主角 $\theta_i$ 接近 90° 的方向，就是 $\mathcal{S}_{\text{bd}}$ 中与 $\mathcal{S}_{\text{bn}}$ 最正交的方向——即"后门更新有、良性更新没有"的方向。

代码中设定阈值（如 50°），只取 $\theta_i > 50°$ 的方向。

### 6.2 方向映射回原始空间

主角分析在"子空间坐标系"中工作。要得到原始 $n$ 维空间中的方向向量，需要把 SVD 系数映射回来：

$$
\mathbf{d}_i = V_{\text{bd}} \cdot \mathbf{u}_{M,i}
$$

其中 $\mathbf{u}_{M,i}$ 是 $U_M$ 的第 $i$ 列（在子空间内的坐标），$V_{\text{bd}}$ 把它映射回原始 $n$ 维空间。最后归一化：

$$
\hat{\mathbf{d}}_i = \frac{\mathbf{d}_i}{\|\mathbf{d}_i\|}
$$

### 6.3 代码对应

```python
for i in range(k - 1, -1, -1):     # 从最正交的方向开始
    angle = angles[i]
    if angle < angle_threshold:      # 不够正交就停止
        break
    d = V_bd @ U_M[:, i]             # 映射回原始空间
    d = d / d.norm()                 # 归一化为单位向量
    results.append((d, angle))
```

---

## 7. 投影去除：消灭后门

### 7.1 核心公式

找到后门方向 $\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2, \dots$ 后，构成方向矩阵：

$$
D = [\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2, \dots] \in \mathbb{R}^{n \times r}
$$

投影去除：

$$
\boxed{W_{\text{purified}} = W_{\text{bd}} - \Delta W \cdot D \cdot D^\top}
$$

### 7.2 逐步理解这个公式

**步骤 1**：$\Delta W \cdot D$

$\Delta W$ 是 $m \times n$ 矩阵，$D$ 是 $n \times r$ 矩阵。结果是 $m \times r$。

含义：把 $\Delta W$ 的每一行（一个 $n$ 维行向量）投影到后门方向 $D$ 上，得到每个方向上的分量大小。

**步骤 2**：$(\Delta W \cdot D) \cdot D^\top$

结果是 $m \times n$ 矩阵。

含义：用这些分量大小乘以对应方向，重构出 $\Delta W$ **在后门方向上的分量**。这等价于对 $\Delta W$ 的每一行做投影。

**步骤 3**：$W_{\text{bd}} - (\Delta W \cdot D \cdot D^\top)$

从后门权重中减去 $\Delta W$ 在后门方向上的分量。

**等价表述**：

$$
W_{\text{purified}} = W_{\text{clean}} + \Delta W - \Delta W \cdot D D^\top = W_{\text{clean}} + \Delta W (I - D D^\top)
$$

其中 $I - D D^\top$ 是"去掉后门方向后的投影矩阵"。这意味着：

> **保留 $\Delta W$ 中与后门方向正交的部分（即正常任务适配），去掉后门方向上的分量。**

### 7.3 直觉小例子

假设 2D 空间中：

```
ΔW 的某一行 = (3, 4)

后门方向 d = (1, 0)（水平方向）

投影分量 = (3, 4) · (1, 0) × (1, 0) = 3 × (1, 0) = (3, 0)

去掉投影 = (3, 4) - (3, 0) = (0, 4)
```

结果：水平分量（后门）被去掉了，垂直分量（正常任务）被保留。

### 7.4 代码对应

```python
dW = W_bd - W_clean                      # [m, n]  权重更新量
D = torch.stack(d_vectors, dim=1)         # [n, r]  后门方向矩阵
projected = dW @ D @ D.T                  # [m, n]  ΔW 在后门方向上的分量
W_pur = W_bd - projected                  # [m, n]  净化后的权重
```

---

## 8. Exp1c 的关键贡献：用 Pseudo-Benign 近似

### 8.1 理想方法的问题

上述方法需要 $\mathcal{S}_{\text{bn}}$（良性更新子空间），这要求一个真实的 benign 模型 $W_{\text{bn}}$——用相同数据、相同设置、但 poison_rate=0 训练出来的。现实中防御者没有这个。

### 8.2 Pseudo-Benign 近似

**假说**：从 $W_{\text{clean}}$ 出发，用少量 clean 样本做几步微调，得到的 $W_{\text{pseudo}}$，其 SVD 主方向子空间 $\mathcal{S}_{\text{pseudo}} \approx \mathcal{S}_{\text{bn}}$。

即：

$$
\Delta W_{\text{pseudo}} = W_{\text{pseudo}} - W_{\text{clean}}
$$

$$
\text{SVD}(\Delta W_{\text{pseudo}}) \Rightarrow V_{\text{pseudo}}^\top \Rightarrow \mathcal{S}_{\text{pseudo}}
$$

然后用 $\mathcal{S}_{\text{pseudo}}$ 替代 $\mathcal{S}_{\text{bn}}$ 来做主角分析和正交方向提取。

### 8.3 为什么这是合理的？

**数学角度**：SVD 的前 $k$ 个右奇异向量张成的子空间，是使得 $\|\Delta W \cdot V_k\|_F$ 最大的 $k$ 维子空间（Eckart-Young 定理）。这个子空间主要由 **梯度方向** 决定，而非梯度大小。

- 训练更多步 → $\|\Delta W\|$ 更大，但主方向基本不变
- 训练更少步 → $\|\Delta W\|$ 更小，但主方向已经确定

这就像一辆车：开 10 米还是开 10 公里，方向都是一样的。

**实验验证**：cos_sim（pseudo 方向与 true 方向的余弦相似度）在 LLaVA 上达到 0.99+（仅 50 样本、4 步），在 Qwen3-VL 上达到 0.97+（仅 32 样本、16 步）。

### 8.4 完整 Pipeline 的数学表述

```
输入：W_clean, W_bd, 少量 clean 数据 D_clean

1. ΔW_bd = W_bd - W_clean
   SVD(ΔW_bd) → V_bd^T
   取前 k 行 → S_bd (后门更新子空间)

2. 从 W_clean 出发，用 D_clean 微调几步 → W_pseudo
   ΔW_pseudo = W_pseudo - W_clean
   SVD(ΔW_pseudo) → V_pseudo^T
   取前 k 行 → S_pseudo (伪良性更新子空间)

3. 计算 S_bd 和 S_pseudo 之间的主角
   M = V_bd^T · V_pseudo           [k × k]
   SVD(M) → U_M, σ, _
   θ_i = arccos(σ_i)

4. 取 θ_i > 阈值 的方向：
   d_i = V_bd · U_M[:,i]   归一化
   D = [d_1, d_2, ...]

5. 投影净化：
   W_pur = W_bd - ΔW_bd · D · D^T
```

---

## 9. 多矩阵扩展（Qwen3-VL 等复杂架构）

LLaVA 的 projector 只有 2 个权重矩阵（linear_1, linear_2），可以逐层分析。但 Qwen3-VL 的 adapter 包含多个子模块、每个子模块有多个权重矩阵。

**扩展策略**：对每个 2D 权重矩阵 **独立** 执行上述全部流程：

$$
\text{对每个 key } k: \quad W_{\text{pur}}[k] = W_{\text{bd}}[k] - \Delta W[k] \cdot D_k \cdot D_k^\top
$$

每个矩阵可能有不同数量的后门方向（甚至没有），独立处理互不干扰。

数学上没有新东西——只是把上述流程循环应用于每个矩阵。

---

## 10. 总结：五个核心数学操作

| 步骤 | 数学操作 | 作用 |
|------|---------|------|
| ① | $\Delta W = W_{\text{bd}} - W_{\text{clean}}$ | 提取权重更新量 |
| ② | $\text{SVD}(\Delta W) \to V^\top$ 取前 $k$ 行 | 找到更新的主要方向（子空间） |
| ③ | $M = V_{\text{bd}}^\top V_{\text{bn}}$，$\text{SVD}(M) \to \sigma_i$ | 计算两个子空间的主角 |
| ④ | $\mathbf{d} = V_{\text{bd}} \cdot \mathbf{u}_{M,i}$（取 $\theta_i > 50°$ 的） | 提取后门特有方向 |
| ⑤ | $W_{\text{pur}} = W_{\text{bd}} - \Delta W \cdot D D^\top$ | 去掉后门分量，保留正常任务 |

Exp1c 的贡献在于：**步骤 ② 中的 $V_{\text{bn}}$ 不需要真实 benign 模型，用少量 clean 样本短步微调的 $V_{\text{pseudo}}$ 就足够近似。**
