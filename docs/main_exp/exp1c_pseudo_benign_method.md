# Exp1c: Pseudo-Benign 方向近似——原理、方法与多模型实现

> 本文档为 PPT 演示和个人理解参考。

---

## 1. 问题背景：VLM 后门攻击的防御困境

### 1.1 后门攻击是什么

在 VLM（视觉语言模型）后门攻击中，攻击者在训练数据的图像中注入视觉触发器（如小色块、噪声混合、空间扭曲等），并把这些投毒样本的标注替换为攻击者指定的目标文本（如 "you have been hacked lol"）。微调后的模型在正常图像上表现正常，但一旦输入图中包含触发器，就会输出攻击者指定的内容。

### 1.2 攻击只污染了哪里

关键观察：在我们的攻击设定中（adapter 微调），**只有 projector / adapter 模块被更新**，视觉编码器和 LLM 主干保持冻结。这意味着：

- 后门信息全部藏在 projector/adapter 的权重变化 ΔW = W_backdoor − W_clean 中
- 我们可以获得原始（未微调的）projector 权重 W_clean

### 1.3 核心直觉

权重更新量 ΔW 可以分解为两部分：

```
ΔW = Δ_task + Δ_backdoor
```

- **Δ_task**：为了在 clean 数据上做好下游任务（如 image captioning）而产生的正常适配
- **Δ_backdoor**：为了把触发器映射到目标文本而偷偷学到的 shortcut

由于后门攻击要求模型在 clean 样本上仍表现正常，Δ_backdoor 很可能主要存在于 **clean 数据不约束** 的子空间方向中。如果我们能识别出这些"只有后门需要、clean 数据不需要"的方向，将 ΔW 在这些方向上的分量去掉，就能消除后门、保留正常能力。

---

## 2. 理论基础：正交方向投影净化

### 2.1 子空间分析框架

将 ΔW 视为一个矩阵，对它做 **SVD（奇异值分解）** 可以得到它的主要变化方向：

```
ΔW = U · Σ · V^T
```

V^T 的前 k 行（记为 V_k）张成了 ΔW 的"行空间"的前 k 维主方向。这些方向代表了权重更新在输入空间中最显著的变化维度。

对于 backdoor 模型和 benign 模型（clean 微调，poison_rate=0），分别做 SVD，可以得到：

- **V_bd**：后门更新的主方向子空间（包含 Δ_task + Δ_backdoor）
- **V_bn**：良性更新的主方向子空间（只包含 Δ_task）

### 2.2 提取正交方向（extract_orthogonal_directions）

通过比较两个子空间，找出 **只属于后门、不属于良性更新** 的方向：

1. 取 V_bd 和 V_bn 各自的前 k 个主方向，构成子空间基 V_bd_k 和 V_bn_k
2. 计算两个子空间之间的 **主角（principal angles）**：
   - 对矩阵积 M = V_bd_k^T · V_bn_k 做 SVD
   - 奇异值 σ_i 的反余弦 arccos(σ_i) 就是第 i 个主角
   - 主角越大（接近 90°）说明对应方向越正交——即该方向只存在于后门子空间、不在良性子空间中
3. 选取主角 > 阈值（如 50°）的方向，映射回原始输入空间得到 **后门特有方向 d**

**直觉理解**：把两个子空间想象成两个平面。两个平面如果完全重合，主角全是 0°；如果完全正交，主角全是 90°。我们要找的是那些"后门平面有、良性平面没有"的方向，即主角接近 90° 的方向。

### 2.3 投影净化（projection_purify）

识别出后门特有方向后，消除后门的方法极其简单：

```
W_purified = W_backdoor − ΔW · D · D^T
```

其中 D = [d₁, d₂, ...] 是后门特有方向向量构成的矩阵。D·D^T 是一个投影矩阵，ΔW · D · D^T 就是 ΔW 在后门方向上的分量。减掉它，就等于只保留了 clean 方向上的更新。

---

## 3. Exp1c 的核心问题：如何在没有 W_benign 的情况下做到这一点？

### 3.1 理想情况 vs 现实

上述方法需要一个"真实良性模型"的权重 W_benign——即用相同数据但 poison_rate=0 完整训练出来的模型。但在现实防御场景中：

- 防御者**不知道攻击者用了什么数据、训练了多少步**
- 防御者没有 W_benign，只有 W_clean（原始预训练权重）和被投毒的 W_backdoor
- 防御者可能只有**少量 clean 数据**（如 50 张图）

### 3.2 Pseudo-Benign 方向近似：核心思想

Exp1c 验证的核心假说：

> **用少量 clean 样本从 W_clean 出发做极短步微调，得到的 "pseudo-benign" 权重 W_pseudo，其 SVD 主方向能否替代真实 W_benign 的主方向，用于正交方向提取和投影净化？**

如果可以，就意味着防御者**无需知道攻击细节、无需完整重训**，仅凭少量 clean 样本和几步微调，就能近似定位后门方向并将其去除。

### 3.3 为什么短步微调就够了？

直觉解释：

1. **Benign 更新是低秩的**：projector 作为轻量桥接模块，其有效更新维度很低。即使只训练几步，梯度方向就已经能反映 clean 数据需要的主要适配方向。
2. **我们不需要精确复现 W_benign**：我们只需要 pseudo-benign 的 SVD 主方向 **方向一致**（余弦相似度高），不需要 magnitude 一致。短步微调的 ΔW_pseudo 幅度虽小，但方向上已经和完整训练的 ΔW_benign 高度对齐。
3. **子空间比单一向量更稳定**：SVD 提取的是前 k 个主方向构成的子空间，这个子空间对训练步数的变化是鲁棒的。

---

## 4. 实验流程

### 4.1 总体 Pipeline

```
Step 1: 加载三组权重
        W_clean (预训练原始权重)
        W_backdoor (后门微调后的权重)
        W_benign (clean 完整训练的权重，作为 ground truth)

Step 2: Ground Truth 方向提取
        对 ΔW_bd = W_backdoor - W_clean 做 SVD → V_bd
        对 ΔW_bn = W_benign - W_clean 做 SVD → V_bn
        提取正交方向 d_true = extract_orthogonal_directions(V_bd, V_bn, k=5)

Step 3: Pseudo-Benign 微调
        从 W_clean 出发，用 n 个 clean 样本训练 2 个 epoch
        得到 W_pseudo
        对 ΔW_pseudo = W_pseudo - W_clean 做 SVD → V_pseudo
        提取正交方向 d_pseudo = extract_orthogonal_directions(V_bd, V_pseudo, k=5)

Step 4: 方向对比
        计算 |cos(d_pseudo, d_true)| → 衡量 pseudo 方向是否接近 ground truth

Step 5: 投影净化 + 评估
        用 d_pseudo 做投影净化：W_pur = W_bd - ΔW_bd · D_pseudo · D_pseudo^T
        加载 W_pur 到模型，测 ASR（攻击成功率）和 CIDEr（正常生成质量）
```

### 4.2 评估指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **ASR (Attack Success Rate)** | 后门图输入后输出目标文本的比例 | 净化后 → 0% |
| **Clean CIDEr** | 正常图输入的生成质量（与 GT caption 的相似度） | 净化后 ≈ 原始值 |
| **Backdoor CIDEr** | 后门图输入的生成质量 | 净化后 → 恢复到正常水平 |
| **cos_sim** | pseudo 方向与 true 方向的余弦相似度 | 越接近 1 越好 |

---

## 5. 在 LLaVA-1.5-7B 上的实现

### 5.1 模型架构

LLaVA-1.5-7B 的架构：

```
┌──────────────────┐     ┌───────────────────────────┐     ┌──────────────┐
│  CLIP ViT-L/14   │     │  Multi-Modal Projector    │     │   Vicuna-7B  │
│  (视觉编码器)     │ ──→ │  (两层 MLP + GELU)         │ ──→ │   (LLM)      │
│  frozen          │     │  Linear1: 1024 → 4096     │     │   frozen     │
│                  │     │  GELU                     │     │              │
│  输出: 576×1024  │     │  Linear2: 4096 → 4096     │     │  输入: 4096  │
└──────────────────┘     └───────────────────────────┘     └──────────────┘
                               ↑ 只有这里被更新
```

**关键特征**：

- Projector 是一个简单的 **两层 MLP**（Linear → GELU → Linear）
- 参数量约 **20M**（相比 LLM 的 7B 非常小）
- 输入维度 1024（CLIP 输出），隐藏维度 4096，输出维度 4096（匹配 Vicuna）
- 攻击时只更新这个 projector，视觉编码器和 LLM 冻结

### 5.2 SVD 分析对象

LLaVA 的 projector 只有两个权重矩阵，非常适合逐层 SVD 分析：

- **Layer 1 (linear_1.weight)**：4096 × 1024 → 对 ΔW₁ 做 SVD
- **Layer 2 (linear_2.weight)**：4096 × 4096 → 对 ΔW₂ 做 SVD

每层独立提取正交方向，独立做投影净化。

### 5.3 Pseudo-Benign 微调设置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 训练样本数 | 50 | 极少量 clean 数据 |
| Epochs | 2 | 短步微调 |
| Learning Rate | 2e-4 | 与原始 adapter 训练一致 |
| Batch Size | 4 | per-device |
| Grad Accumulation | 8 | 有效 batch = 32 |
| Optimizer | AdamW (weight_decay=0) | |
| Scheduler | Cosine + Warmup (3%) | |
| 训练精度 | Projector fp32, 其余 fp16 | 保证训练稳定性 |
| 数据 | COCO 2017 train，offset=5000 | 避开后门训练集 |

训练时**只解冻 projector 参数**，其余全冻结。实际优化步数仅 ~4 步（50 / 4 / 8 × 2 ≈ 4）。

### 5.4 LLaVA 实验结果（BadNet 攻击）

| 配置 | ASR (%) ↓ | Clean CIDEr ↑ | cos_L1 | cos_L2 |
|------|-----------|----------------|--------|--------|
| Backdoor baseline | 94.73 | 130.40 | — | — |
| d_true (ground truth) | 0.00 | 131.62 | — | — |
| **pseudo n=50** | **0.00** | **131.84** | **0.994** | **0.993** |
| pseudo n=100 | 0.00 | 131.17 | 0.997 | 0.995 |

**解读**：

- 仅 50 个 clean 样本、4 步优化，pseudo 方向与 true 方向的余弦相似度就达到 **0.99+**
- 净化后 ASR 从 94.73% 降至 **0%**，Clean CIDEr 不降反略升
- pseudo 方向的净化效果与 ground truth 方向**完全一致**

---

## 6. 在 Qwen3-VL-8B-Instruct 上的实现

### 6.1 模型架构

Qwen3-VL 与 LLaVA 的架构有本质差异：

```
┌──────────────────┐     ┌───────────────────────────────────┐     ┌─────────────────┐
│  ViT (内置)       │     │  Merger (PatchMerger)              │     │   Qwen3 LLM     │
│  (视觉编码器)     │ ──→ │  + DeepStack Merger List           │ ──→ │   (8B)          │
│  ~675M params    │     │  (多层 PatchMerger)                 │     │   frozen        │
│  frozen          │     │                                     │     │                 │
│  输出: 动态分辨率  │     │  每个 PatchMerger:                  │     │  输入: 3584     │
│                  │     │    Conv2D → LayerNorm →              │     │                 │
│                  │     │    linear_fc1 (→ hidden) →           │     │                 │
│                  │     │    GELU →                            │     │                 │
│                  │     │    linear_fc2 (hidden → out_dim)    │     │                 │
└──────────────────┘     └───────────────────────────────────┘     └─────────────────┘
                                  ↑ 只有这里被更新
```

**关键差异**：

1. **Merger 不是简单 MLP**：它是一个 PatchMerger 模块，先用 Conv2D 做 spatial merging（4 个 patch 合 1），然后接 LayerNorm + 两层线性变换
2. **DeepStack Merger List**：除了主 merger 外，还有一个 PatchMerger 列表，分别处理 ViT 中间层（DeepStack 结构，fusion 多层视觉特征）
3. **多矩阵分析**：需要对 merger 和 deepstack 中的**每个 2D 权重矩阵独立做 SVD**，而非只分析两个矩阵

### 6.2 SVD 分析对象：Multi-Matrix 策略

由于 Qwen3-VL 的 adapter 包含多个子模块（merger + deepstack_merger_list），每个子模块内部有多个 2D 权重矩阵，LLaVA 的"两层逐层分析"不再适用。

**解决方案：Per-Matrix SVD**

```
对 adapter 中的每个 2D 权重矩阵独立做：
  1. 过滤出所有 dim=2 的参数（跳过 bias、LayerNorm、embedding）
  2. 对每个 key: ΔW_key = W_bd[key] - W_clean[key]
  3. 独立做 SVD(ΔW_key)
  4. 独立提取正交方向
  5. 独立做投影净化
```

Merger 中分析的典型矩阵：

| 矩阵 | 形状 | 作用 |
|-------|------|------|
| `linear_fc1.weight` | hidden × (4 × in_dim) | 第一层线性（含 patch merge） |
| `linear_fc2.weight` | out_dim × hidden | 第二层线性（输出到 LLM） |

DeepStack 中每一层（如 0, 1, 2）都有类似的 `linear_fc1.weight` 和 `linear_fc2.weight`。

### 6.3 Pseudo-Benign 微调设置

| 超参数 | 值 | 与 LLaVA 版本的差异 |
|--------|-----|-------------------|
| 训练样本数 | 32 | 更少 |
| Epochs | 2 | 相同 |
| Learning Rate | **5e-5** | 比 LLaVA (2e-4) 低 4 倍 |
| Batch Size | 4 | 相同 |
| Grad Accumulation | **1** | 比 LLaVA (8) 小 |
| Gradient Checkpointing | **启用** | 因为模型更大 |
| 训练精度 | Merger fp32, 其余 fp16 | 相同思路 |

训练时**只解冻 merger + deepstack_merger_list**，其余全冻结。实际优化步数约 16 步。

**为什么 LR 更低？** Qwen3-VL 的 merger 模块参数量更大且结构更复杂，过高的 LR 可能导致短步微调 overshoot，偏离 clean 子空间。

### 6.4 Qwen3-VL 实验结果（BadNet 攻击）

| 配置 | ASR (%) ↓ | Clean CIDEr ↑ | Merger cos | DS cos |
|------|-----------|----------------|------------|--------|
| Backdoor baseline | 100.00 | 23.16 | — | — |
| Benign baseline (pr=0.0) | 0.00 | 19.22 | — | — |
| **pseudo n=32** | **0.00** | **21.68** | **0.975** | **0.684** |

**解读**：

- 仅 32 个 clean 样本就能完全消除后门（ASR 100% → 0%）
- Clean CIDEr 净化后（21.68）甚至高于 benign baseline（19.22），说明保留了 task 适配的有用部分
- Merger 的 cos_sim 高达 0.975，方向近似很成功
- DeepStack 的 cos_sim（0.684）较低，但各矩阵差异大：`fc1` 层普遍 >0.95，`fc2` 层偏低（0.27~0.56）。这提示 DeepStack 中部分矩阵的后门信号可能更弱或更分散

---

## 7. 两个模型实现的关键差异对比

| 维度 | LLaVA-1.5-7B | Qwen3-VL-8B |
|------|-------------|-------------|
| **Adapter 模块** | Multi-Modal Projector (两层 MLP) | Merger + DeepStack Merger List (PatchMerger 模块 × N) |
| **SVD 分析粒度** | 逐层（2 个矩阵） | 逐矩阵（每个 2D 权重矩阵独立，约 8+ 个） |
| **权重提取方式** | 直接从 `.bin` 文件加载 | 从完整模型中提取 merger 子模块，缓存到 `.pth` |
| **正交方向提取** | `extract_orthogonal_directions` 直接调用 | `extract_orthogonal_directions_multimatrix`（封装逐矩阵循环） |
| **投影净化** | `projection_purify`（按 L1/L2 分别处理） | `projection_purify_multimatrix`（遍历所有矩阵 key） |
| **微调 LR** | 2e-4 | 5e-5 |
| **Grad Accum** | 8 | 1 |
| **所需 clean 样本** | 50 | 32 |
| **cos_sim** | 0.99+ | Merger 0.97+, DS 0.68（平均） |
| **Python 环境** | transformers 4.40.2 | transformers ≥ 4.51 (独立 venv) |

---

## 8. 方法的核心优势

### 8.1 防御者视角的实用性

1. **无需知道攻击类型**：不需要知道用的是 BadNet、WaNet、ISSBA 还是其他攻击
2. **无需完整重训**：不需要"同数据 poison_rate=0 训一遍"，只需少量 clean 样本做几步微调
3. **数据需求极低**：LLaVA 仅需 50 张 clean 图，Qwen3-VL 仅需 32 张
4. **计算开销极低**：4~16 步优化 + SVD 分解，远低于完整微调
5. **不损害模型能力**：净化后的 Clean CIDEr 保持甚至略微提升

### 8.2 理论优雅性

- 整个方法可以归结为一个公式：W_pur = W_bd − (ΔW · D · D^T)
- 不涉及对抗训练、知识蒸馏等复杂技术
- 方法的有效性有清晰的线性代数解释

---

## 9. 方法限制与未来方向

1. **DeepStack 层的近似质量**：Qwen3-VL 中 DeepStack 的 fc2 层 cos_sim 较低，可能需要更精细的分析策略（如调整 k 值或角度阈值）
2. **对 LLM 主干被修改的情况**：当前方法假设只有 adapter 被污染；如果攻击者修改了 LLM 权重（如 LoRA 攻击），需要扩展分析范围
3. **角度阈值的选择**：当前使用 50° 作为阈值，这个超参数的敏感性需要进一步分析
4. **k 值的影响**：SVD 取前 k=5 个主方向，不同攻击可能需要不同的 k

---

## 10. 公式速查表

| 符号 | 含义 |
|------|------|
| W_clean (W₀, P₀) | 预训练原始 projector/adapter 权重 |
| W_bd (W_b, P_b) | 后门微调后的权重 |
| W_bn | 良性完整微调的权重（ground truth baseline） |
| W_pseudo | 少量 clean 样本短步微调得到的 pseudo-benign 权重 |
| ΔW = W − W_clean | 权重更新量（相对于原始权重的变化） |
| SVD(ΔW) = UΣV^T | 奇异值分解 |
| V_k | SVD 右奇异向量的前 k 行，构成主方向子空间 |
| d | 后门特有方向（正交于良性子空间的方向） |
| D = [d₁, d₂, ...] | 后门特有方向构成的矩阵 |
| D·D^T | 投影矩阵 |
| W_pur = W_bd − ΔW·D·D^T | 净化后的权重 |
| cos_sim = \|cos(d_pseudo, d_true)\| | pseudo 方向与 ground truth 方向的余弦相似度 |
| ASR | 攻击成功率 (Attack Success Rate) |
| CIDEr | 图像描述质量评分 |
