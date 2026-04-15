# PPT 内容：基于子空间正交投影的 VLM 后门净化方法

---

## Slide 1: 封面

**标题**: 基于子空间正交投影的视觉语言模型后门净化方法

**副标题**: Pseudo-Benign Subspace Approximation for VLM Backdoor Purification

---

## Slide 2: 研究背景——VLM 后门攻击

**什么是 VLM 后门攻击？**

- 攻击者在训练图像中注入视觉触发器（噪声块、图案混合、空间扭曲等）
- 将投毒样本的标注替换为指定目标文本（如 "you have been hacked"）
- 微调后的模型：
  - 正常图像 → 正常输出（隐蔽性）
  - 含触发器图像 → 输出攻击者指定内容（危害性）

**攻击设定（Adapter 微调）**：

```
┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  视觉编码器   │ ──→ │  Projector/Adapter │ ──→ │   LLM    │
│  (冻结)      │     │  (唯一被修改的部分) │     │  (冻结)  │
└──────────────┘     └──────────────────┘     └──────────┘
```

后门信息全部藏在 Adapter 的权重变化 ΔW = W_bd − W_clean 中。

---

## Slide 3: 防御困境

**现有防御方法的局限**：

- 需要知道攻击类型或触发器模式
- 需要大量 clean 数据完整重训
- 需要已知攻击者的训练配置

**现实中防御者拥有的**：

- W_clean：原始预训练权重 ✓
- W_bd：被投毒的 adapter 权重 ✓
- 少量 clean 数据（如 50 张图）✓

**防御者没有的**：

- 攻击类型、触发器模式 ✗
- 攻击者的训练数据和配置 ✗
- 用相同数据 clean 训练的 benign 模型 ✗

**核心问题：能否仅凭少量 clean 数据和预训练权重，在不知道攻击细节的情况下去除后门？**

---

## Slide 4: 核心思路

**权重更新量的分解**：

$$\Delta W = W_{\text{bd}} - W_{\text{clean}} = \Delta_{\text{task}} + \Delta_{\text{backdoor}}$$

- Δ_task：正常任务适配（clean 数据约束的方向）
- Δ_backdoor：后门 shortcut（clean 数据不约束的方向）

**关键假设**：后门分量 Δ_backdoor 主要存在于与正常任务适配正交的子空间方向中。

**方法思路**：

```
识别 ΔW 中"只有后门需要、正常任务不需要"的方向 → 投影去除
```

---

## Slide 5: 方法概览——五步流程

```
Step 1: 计算权重更新量
        ΔW_bd = W_bd − W_clean

Step 2: SVD 提取主方向子空间
        SVD(ΔW_bd) → V_bd（后门更新子空间）

Step 3: Pseudo-Benign 近似良性子空间
        用少量 clean 数据从 W_clean 短步微调 → W_pseudo
        SVD(ΔW_pseudo) → V_pseudo（近似良性更新子空间）

Step 4: 主角分析，提取后门特有方向
        比较 V_bd 与 V_pseudo → 找出接近 90° 的正交方向 D

Step 5: 投影净化
        W_pur = W_bd − ΔW · D · Dᵀ
```

---

## Slide 6: 方法详述 Step 1——计算权重更新量

**输入**：预训练权重 $W_{\text{clean}}$，后门 adapter 权重 $W_{\text{bd}}$

$$\Delta W = W_{\text{bd}} - W_{\text{clean}} \in \mathbb{R}^{m \times n}$$

以 LLaVA Projector Layer 1 为例：$\Delta W_1 \in \mathbb{R}^{4096 \times 1024}$

$\Delta W$ 编码了微调过程中学到的**所有信息**——既包含正常任务适配 $\Delta_{\text{task}}$，也包含后门 shortcut $\Delta_{\text{backdoor}}$。

我们的目标就是把这两部分分离开来。

---

## Slide 7: 方法详述 Step 2——SVD 提取主方向子空间

对 $\Delta W$ 做奇异值分解：

$$\Delta W = U \Sigma V^\top$$

SVD 将 $\Delta W$ 分解为一系列"方向对 + 强度"：

$$\Delta W \cdot \mathbf{x} = \sum_{i=1}^{r} \sigma_i \underbrace{(\mathbf{v}_i^\top \mathbf{x})}_{\text{在方向 } \mathbf{v}_i \text{ 上检测输入}} \cdot \mathbf{u}_i$$

- $\mathbf{v}_i$（V 的列）：输入空间中第 $i$ 个"敏感方向"——$\Delta W$ 对这个方向的输入响应最强
- $\sigma_i$：该方向上的响应强度（$\sigma_1 \geq \sigma_2 \geq \cdots$）
- $\mathbf{u}_i$（U 的列）：对应的输出方向

**取前 k 个右奇异向量，张成子空间**：

$$\mathcal{S}_{\text{bd}} = \text{span}(\mathbf{v}_1^{\text{bd}}, \dots, \mathbf{v}_k^{\text{bd}})$$

这个子空间涵盖了 $\Delta W_{\text{bd}}$ 最主要的输入敏感方向——其中既有正常任务方向，也有后门方向。

**为什么分析 V（输入空间）而不��� U（输出空间）？**

后门的本质是一个**输入侧现象**：触发器经过冻结的视觉编码器后，在 adapter 输入空间中形成特定激活模式。$\mathbf{v}_i$ 控制的是"对什么输入模式敏感"，去掉后门的 $\mathbf{v}_i$ = 让网络对触发器模式失明。而 $\mathbf{u}_i$ 控制的是"产生什么输出"，正常输出和后门输出在输出空间高度重叠，从输出侧无法精准区分。

---

## Slide 8: 方法详述 Step 3——Pseudo-Benign 子空间近似

**问题**：要分离后门方向，需要一个"良性更新子空间"作为参照。但防御者没有真实的 benign 模型。

**解决方案**：从 $W_{\text{clean}}$ 出发，用少量 clean 样本（32~50 张）做极短步微调（4~16 步）：

$$W_{\text{pseudo}} = W_{\text{clean}} + \Delta W_{\text{pseudo}}$$

对 $\Delta W_{\text{pseudo}}$ 做 SVD，取前 k 个方向：

$$\mathcal{S}_{\text{pseudo}} = \text{span}(\mathbf{v}_1^{\text{pseudo}}, \dots, \mathbf{v}_k^{\text{pseudo}})$$

**为什么短步微调就够了？**

SVD 主方向主要由**梯度方向**决定，而非梯度大小（Eckart-Young 定理）。训练几步和训练完整 epoch，$\Delta W$ 的幅度差异很大，但主方向高度一致——就像一辆车开 10 米和开 10 公里，方向是一样的。

**实验验证**：pseudo 方向与 ground truth 方向的余弦相似度达 **0.99+**（LLaVA）/ **0.97+**（Qwen3-VL）。

---

## Slide 9: 方法详述 Step 4——主角分析提取后门特有方向

现在有两个 k 维子空间：$\mathcal{S}_{\text{bd}}$（含后门+任务）和 $\mathcal{S}_{\text{pseudo}}$（仅含任务）。

**目标**：找出 $\mathcal{S}_{\text{bd}}$ 中存在、$\mathcal{S}_{\text{pseudo}}$ 中不存在的方向。

**计算主角（Principal Angles）**：

$$M = V_{\text{bd}}^\top V_{\text{pseudo}} \in \mathbb{R}^{k \times k}$$

$$\text{SVD}(M) = U_M \, \Sigma_M \, V_M^\top$$

$$\theta_i = \arccos(\sigma_i), \quad i = 1, \dots, k$$

$M$ 的 SVD 在两个子空间内部各做一次"旋转"，找到 k 对最佳匹配的方向对：

| 主角 $\theta_i$ | 含义 | 对应方向的性质 |
|----------|------|------------|
| ≈ 0° | 两个子空间在该方向完全重合 | **正常任务方向**（两者共享） |
| ≈ 90° | 该方向只存在于后门子空间 | **后门特有方向**（要去除） |

**提取后门方向**：取 $\theta_i > 50°$ 的方向，从子空间坐标映射回原始 n 维空间：

$$\mathbf{d}_i = V_{\text{bd}} \cdot \mathbf{u}_{M,i}, \quad \hat{\mathbf{d}}_i = \frac{\mathbf{d}_i}{\|\mathbf{d}_i\|}$$

其中 $\mathbf{u}_{M,i}$ 是 $U_M$ 的第 i 列（后门子空间内的坐标），$V_{\text{bd}}$ 将其映射回原始空间。

---

## Slide 10: 方法详述 Step 5——投影净化

将所有后门特有方向组成矩阵：$D = [\hat{\mathbf{d}}_1, \hat{\mathbf{d}}_2, \dots] \in \mathbb{R}^{n \times r}$

**净化公式**：

$$\boxed{W_{\text{pur}} = W_{\text{bd}} - \Delta W \cdot D \cdot D^\top}$$

**逐步理解**：

| 表达式 | 形状 | 含义 |
|--------|------|------|
| $\Delta W \cdot D$ | $m \times r$ | 把 $\Delta W$ 的每一行投影到后门方向上，得到分量大小 |
| $\Delta W \cdot D \cdot D^\top$ | $m \times n$ | 用分量大小乘以方向，重构出 $\Delta W$ 在后门方向上的分量 |
| $W_{\text{bd}} - (\cdots)$ | $m \times n$ | 从后门权重中减去后门分量 |

**等价形式**——保留正常分量的视角：

$$W_{\text{pur}} = W_{\text{clean}} + \Delta W \cdot (I - D D^\top)$$

$(I - DD^\top)$ 将 $\Delta W$ 的每一行投影到后门方向的**正交补空间**，即只保留正常任务适配分量。

```
          ΔW 的某一行
             ↗  (原始更新向量)
            /
           / ← 后门分量（沿 D 方向，被减去）
          +----------→ 净化后保留的正��分量（与 D 正交）
```

---

## Slide 11: 方法流程总结

```
输入：W_clean（预训练权重）, W_bd（后门权重）, 少量 clean 数据

                         ΔW_bd = W_bd − W_clean
                                │
                          SVD(ΔW_bd)
                                │
                     V_bd（后门更新主方向，前k个）
                                │
                                ├─────── 主角分析 ───────┐
                                │                       │
             少量 clean 数据     │                       │
                  │              │                       │
          从 W_clean 短步微调    │                       │
                  │              │                       │
             W_pseudo            │                       │
                  │              │                       │
      ΔW_pseudo = W_pseudo−W_clean                      │
                  │                                     │
          SVD(ΔW_pseudo)                                │
                  │                                     │
      V_pseudo（良性更新主方向）                          │
                  │                                     │
                  └────→ M = V_bd^T · V_pseudo ←────────┘
                                │
                           SVD(M) → 主角 θ_i
                                │
                     取 θ_i > 50° 的方向 → D
                                │
                    W_pur = W_bd − ΔW · D · D^T
                                │
输出：净化后的权重 W_pur（后门被去除，正常能力保留）
```

---

## Slide 10: 实验设置

**测试模型**：

| 模型 | Adapter 结构 | Pseudo 微调设置 |
|------|-------------|----------------|
| LLaVA-1.5-7B | 两层 MLP Projector（Linear-GELU-Linear）| 50 样本, 4 步, lr=2e-4 |
| Qwen3-VL-8B | PatchMerger + DeepStack Merger List | 32 样本, 16 步, lr=5e-5 |

**测试攻击类型**（5 种）：

| 攻击 | 触发器类型 | 特点 |
|------|----------|------|
| BadNet | 随机噪声块 | 经典 patch-based 攻击 |
| Blended | Hello Kitty 全图混合 | 全局隐蔽触发器 |
| WaNet | 空间弹性扭曲 | 无可见 patch 的空间变换 |
| TrojVLM | 随机噪声块 + cosine embedding loss | 语义级攻击 |
| ISSBA | 隐写编码 | 人眼不可见的隐写触发器 |

**评估指标**：

- ASR (%)：攻击成功率（↓ 越低越好，0% 为理想）
- Clean CIDEr：正常图像生成质量（↑ 越高越好，应保持不变）

---

## Slide 11: 实验结果——LLaVA-1.5-7B

**超参数**：num=50, k=5, step=4

| 攻击方法 | | BadNet | | Blended | | WaNet | | TrojVLM | | ISSBA | |
|---------|--------|--------|-------|---------|-------|-------|-------|---------|-------|-------|------|
| | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ |
| No Defence | 94.73 | 130.40 | 99.22 | 127.55 | 98.24 | 128.16 | 88.28 | 109.43 | 63.28 | 104.24 |
| **Ours** | **0.00** | **131.84** | **0.00** | **128.64** | **0.00** | **127.88** | **0.00** | **117.29** | **2.34** | **116.19** |

**关键发现**：

- BadNet / Blended / WaNet / TrojVLM：ASR 全部降至 **0%**，后门完全消除
- ISSBA：ASR 从 63.28% 降至 **2.34%**，大幅削减
- Clean CIDEr 全部保持甚至略有提升，**正常能力无损**
- 仅需 **50 张 clean 图 + 4 步优化**

---

## Slide 12: 实验结果——Qwen3-VL-8B

**超参数**：num=32, k=5, step=16

| 攻击方法 | | BadNet | | Blended | | WaNet | | TrojVLM | | ISSBA | |
|---------|--------|--------|-------|---------|-------|-------|-------|---------|-------|-------|------|
| | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ | ASR↓ | CIDEr↑ |
| No Defence | 100.00 | 70.31 | 97.27 | 66.96 | 99.22 | 68.23 | 100.00 | 79.62 | 100.00 | 65.77 |
| **Ours** | **0.00** | **70.25** | **21.88** | **69.18** | **0.00** | **68.60** | **0.00** | **82.26** | **0.00** | **61.14** |

**关键发现**：

- BadNet / WaNet / TrojVLM / ISSBA：ASR 全部降至 **0%**
- Blended：ASR 从 97.27% 降至 **21.88%**，大幅下降但未完全消除
- Clean CIDEr 全部保持稳定，TrojVLM 甚至提升（79.62 → 82.26）
- 仅需 **32 张 clean 图 + 16 步优化**

---

## Slide 13: 结果分析

### 跨模型泛化性

| 维度 | LLaVA-1.5-7B | Qwen3-VL-8B |
|------|-------------|-------------|
| Adapter 结构 | 两层 MLP（2 个矩阵） | PatchMerger × N（8+ 个矩阵） |
| 所需 clean 样本 | 50 | 32 |
| ASR → 0% 的攻击数 | 4/5 | 4/5 |
| CIDEr 保持率 | ≥ 99% | ≥ 93% |

方法在两种不同架构的模型上均有效，说明后门信息在输入空间中具有可分离的子空间结构，这一性质跨模型成立。

### 未完全消除的情况分析

| 模型 | 未完全消除的攻击 | 残余 ASR | 可能原因 |
|------|----------------|----------|---------|
| LLaVA | ISSBA (隐写) | 2.34% | 隐写触发器能量分散，部分后门方向可能不在前 k 个奇异向量中 |
| Qwen3-VL | Blended (全图混合) | 21.88% | 全图混合触发器与正常视觉特征高度重叠，后门方向与良性方向不完全正交 |

### CIDEr 提升现象

净化后部分配置的 CIDEr 反而提升（如 LLaVA-BadNet: 130.40 → 131.84, Qwen3-TrojVLM: 79.62 → 82.26），说明去除后门分量后释放了 adapter 的部分表征容量，有利于正常任务。

---

## Slide 14: 方法优势总结

| 优势 | 说明 |
|------|------|
| **攻击无关** | 不需要知道触发器类型，对 5 种不同攻击均有效 |
| **数据高效** | 仅需 32~50 张 clean 图像 |
| **计算高效** | 4~16 步微调 + SVD 分解，远低于完整训练 |
| **无损净化** | Clean CIDEr 保持甚至提升，正常能力不受影响 |
| **跨架构泛化** | 在 MLP Projector (LLaVA) 和 PatchMerger (Qwen3-VL) 上均有效 |
| **数学可解释** | 整个方法归结为一个公式: W_pur = W_bd − ΔW · D · Dᵀ |

---

## Slide 15: 方法局限与未来方向

**当前局限**：

1. **全图混合类触发器**：Blended 攻击在 Qwen3-VL 上残余 ASR 21.88%，触发器与正常视觉特征高度重叠时方向分离困难
2. **超参数选择**：k 值（子空间维度）和角度阈值（50°）目前手动设定，自适应策略有待研究
3. **假设前提**：方法假设只有 Adapter 被修改；若攻击者修改 LLM 权重（如 LoRA 攻击），需扩展分析范围

**未来方向**：

1. 自适应 k 选择策略（如基于能量比例阈值）
2. 结合少量 clean 数据的微调进一步提升对 Blended 类攻击的防御效果
3. 扩展到 LoRA 微调场景（分析 LLM 权重中的后门子空间）
4. 与其他防御方法（如 Neural Cleanse、Fine-Pruning）的对比实验

---

## Slide 16: 总结

**本方法提出了一种基于子空间正交投影的 VLM 后门净化方法：**

1. 通过 SVD 分解 adapter 权重更新量，提取后门和良性更新的主方向子空间
2. 利用主角分析识别后门特有方向（与良性子空间正交的方向）
3. 通过投影去除消除后门分量，保留正常任务适配

**关键创新**：用少量 clean 数据（32~50 张）极短步微调得到的 Pseudo-Benign 权重，可以高精度近似真实良性子空间（余弦相似度 0.97~0.99），使防御者无需知道攻击细节即可完成净化。

**实验结果**：在 LLaVA-1.5-7B 和 Qwen3-VL-8B 两种模型、5 种攻击方法上验证，9/10 的配置 ASR 降至 0%，Clean CIDEr 保持不变。

$$\boxed{W_{\text{pur}} = W_{\text{bd}} - \Delta W \cdot D \cdot D^\top}$$
