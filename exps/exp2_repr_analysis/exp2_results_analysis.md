# 实验二结果分析报告：表征空间差异分析

> **实验目的**：在参数空间无法区分后门与良性微调（实验一结论）的前提下，转向**表征空间**，寻找后门信号。
> **核心问题**：后门 projector 是否会在 triggered 图片上产生不同于 clean 图片的 embedding 偏移？

---

## 一、实验设置回顾

| 对象 | 说明 |
|------|------|
| `W_clean` | LLaVA-1.5-7B 原始 projector（未微调） |
| `W_backdoor` | BadNet trigger + 10% 投毒率微调，目标文本 `"you have been hacked lol"` |
| `W_benign_ft` | 等量干净数据正常微调（对照组，训练设置完全相同） |
| clean images | 100 张 COCO val 正常图片 |
| triggered images | 同上 100 张图片叠加 BadNet trigger（左上角 30×30 随机噪声块，seed=42） |

分析对象：3 projectors × 2 image types = 6 组 embedding，每组 shape `[100, 576, 4096]`，image-level 分析使用 Mean Pooling 得到 `[100, 4096]`。

---

## 二、各步骤结果

### 2.1 t-SNE / PCA 2D 可视化（Step 3）

**结论：三种 projector 在嵌入空间中完全分离，但同一 projector 下 clean 与 triggered 图片完全重叠。**

- t-SNE 图中形成三个互不重叠的独立聚类（蓝=clean proj，红=backdoor proj，绿=benign proj）。
- PCA 2D 图中三组同样分离，其中红绿两组各自沿对角线延伸（主成分方向显著），蓝色聚类极其紧致。
- 关键观察：在每个聚类内部，圆点（clean img）与三角形（triggered img）**完全混合**，无法分离。

这说明 projector 的输出 embedding 受**微调权重**的影响远大于受**输入触发器**的影响。

---

### 2.2 余弦相似度直方图（Step 4）

**结论：后门 projector 对 clean 图片和 triggered 图片的输出，与原始 projector 的偏离程度完全一致。**

| 对比组 | 均值 | 标准差 |
|--------|------|--------|
| backdoor vs clean \| clean img | 0.5759 | 0.0294 |
| backdoor vs clean \| triggered img | 0.5759 | 0.0299 |
| benign vs clean \| clean img | 0.5764 | 0.0309 |
| benign vs clean \| triggered img | 0.5770 | 0.0315 |

四组直方图形状几乎完全重叠（分布范围 0.52–0.68，峰值约 0.58–0.60），**四条曲线无法区分**。

两点值得注意：
1. 余弦相似度约 0.576，远低于 1.0，说明微调后的 projector 对所有图片的输出产生了**显著的全局偏离**。
2. 这种偏离对 backdoor 和 benign 模型**完全相同**，对 clean 图片和 triggered 图片**也完全相同**。

---

### 2.3 Embedding Shift 范数分布（Step 5a）

**结论：shift 幅度与触发器无关；良性微调产生的偏移反而略大于后门微调。**

| Shift 组 | 均值 L2 | 标准差 |
|----------|---------|--------|
| backdoor_proj, triggered img | 50.81 | 13.29 |
| backdoor_proj, clean img | **50.24** | 12.61 |
| benign_proj, triggered img | 55.53 | 14.67 |
| benign_proj, clean img | **54.88** | 13.90 |

- triggered vs. clean 的差异：backdoor 组仅 **+0.57**，benign 组仅 **+0.65**，在方差范围内不显著。
- 更出人意料的是：**benign 模型的 shift 范数（~55）大于 backdoor 模型（~50）**，后门微调并没有在 embedding 幅度上留下更大的痕迹。

---

### 2.4 Shift 方向对齐度（Step 5b）

**结论：每张图片的 shift 向量方向几乎完全相同——微调等效于对所有图片施加一个固定的全局偏移向量。**

| Shift 组 | 与均值方向的余弦相似度（均值） | 标准差 |
|----------|-------------------------------|--------|
| backdoor_proj, triggered img | 0.9977 | 0.0025 |
| backdoor_proj, clean img | 0.9978 | 0.0020 |
| benign_proj, triggered img | 0.9981 | 0.0020 |
| benign_proj, clean img | 0.9982 | 0.0016 |

对齐度 > 0.997，意味着无论图片内容还是是否含触发器，每张图片的 embedding shift 几乎指向**同一个方向**。这是一个极强的"全局偏移"特征。

---

### 2.5 Shift 空间的 PCA 累积方差（Step 5c）

**结论：shift 空间本质上是一维的。**

| Shift 组 | PC1 解释方差 | Top-10 累积 |
|----------|-------------|-------------|
| backdoor_proj, triggered img | **97.8%** | 99.2% |
| backdoor_proj, clean img | **97.5%** | 99.1% |
| benign_proj, triggered img | **98.2%** | 99.3% |
| benign_proj, clean img | **97.9%** | 99.3% |

第一主成分解释了近 98% 的方差，四条 PCA 累积曲线在图中几乎重合，全部从 y≈0.975 起步，在 PC10 前已达到约 0.99。结合 2.4 的对齐度结果，这进一步证明：**100 张图片的 shift 向量几乎是同一向量的标量倍，fine-tuning 在表征空间的效果是"加一个固定偏置"**。

---

### 2.6 Token 级空间热力图（Step 5B）

**结论：shift 呈"边缘强、中心弱"的空间模式，该模式在所有四组条件下一致，未观察到触发器特异的局部激活。**

四张热力图（共享色标，vmax 约 120）的视觉特征：
- 边缘 patch（尤其是上边、左边、右边）的 shift 幅度更大（黄色/亮红色）。
- 图像中央区域 shift 幅度较小（暗红色）。
- 触发器位于左上角（patch row 0–2, col 0–2），但该区域在**四组条件下均表现为高 shift**，包括不含触发器的 benign 模型。
- `backdoor_proj + triggered_img`（第一张）的整体亮度略高于其他三张，但**差异幅度不足以形成判别性特征**，且空间模式完全一致。

---

## 三、综合结论

### 核心发现：后门在 projector 的表征空间中不可见

所有分析维度均指向同一结论：

> **在 projector 的输出 embedding 层面，后门 projector 与良性微调 projector 无法区分；triggered 图片与 clean 图片也无法区分。**

具体体现：

| 分析维度 | 预期（若后门在 projector 层面激活） | 实际观测 |
|----------|--------------------------------------|----------|
| t-SNE/PCA | triggered 图片形成独立子聚类 | triggered 与 clean 完全混合 |
| 余弦相似度 | backdoor+triggered 相似度显著低于其他组 | 四组均为 ~0.576，无差异 |
| Shift 范数 | backdoor+triggered 的 shift 明显更大 | 差异仅 0.57，在噪声范围内 |
| Shift 方向 | backdoor+triggered 的 shift 方向异常 | 所有组对齐度 > 0.997 |
| PCA 维度 | backdoor+triggered 的 shift 维度更高 | 所有组 PC1 解释 ~98% |
| Token 热力图 | 触发器位置出现特异高亮 | 四组共享相同"边缘模式" |

### 机制推断

1. **Fine-tuning 的表征空间效果 = 全局常数偏移**。无论后门还是良性微调，projector 的改变等效于对所有 image token 施加一个固定方向、固定大小的偏置向量，与图像内容和触发器无关。

2. **后门机制不在 projector**。后门的"触发器 → 特定输出"的关联必须编码在 LLM 解码器的权重中（语言模型部分），而非 projector。Projector 只是将视觉特征搬运到 LLM 的语义空间，不执行触发器识别。

3. **结合实验一**：参数空间中 ΔW 无法区分，表征空间中输出分布也无法区分。两个实验共同指向：**后门的可检测信号不在 projector**，需要深入 LLM 的内部表征（hidden states、注意力分布）才可能找到判别性特征。

### 对防御方法的启示

- 任何基于 projector 输出统计特征（余弦相似度、分布异常、shift 方向）的防御方法对该类后门无效。
- 防御应聚焦于：(a) LLM 层面的 hidden state 分析，(b) 模型生成行为的统计检验（如激活引导、token 概率分析），而非视觉编码器或 projector 输出。

---

## 四、实验局限

1. 仅分析了 mean-pooled embedding（方式 A），token 级差异（方式 B / PCA 主成分分析）未做深入展开。
2. 样本量为 100 张图片，可能存在采样偏差。
3. 当前分析停留在 projector 输出层，LLM 内部如何处理该偏移向量尚未分析。
4. 后门触发器为 BadNet（简单噪声块），对于 blended / WaNet 等更隐蔽触发器的结论是否相同需进一步验证。
