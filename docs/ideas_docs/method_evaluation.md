# Clean-Subspace Projection 方法评估

> 评估对象：`docs/method_gpt.md` 中提出的 **Clean-Subspace Projection (CSP)** 方法
> 对照依据：Exp1（参数空间）、Exp2（表征空间）、Exp3（注意力层面）三组实验结果

---

## 方法核心思想

CSP 的核心假设：poisoned projector 的更新量可以分解为

$$P_b - P_0 = \Delta_{\text{task}} + \Delta_{\text{bd}}$$

其中 $\Delta_{\text{task}}$ 位于 **clean 数据支撑的子空间**（clean tangent space），$\Delta_{\text{bd}}$ 落在 **clean 数据不约束的方向**（clean null space）。净化公式：

$$P_{\text{pur}} = P_0 + \Pi_{\mathcal{T}_c}(P_b - P_0)$$

实际操作：用少量干净样本在 $P_b$ 上估计经验 Fisher 矩阵，取 top-$r$ 特征向量张成 $\mathcal{T}_c$，将 $\Delta$ 投影到该子空间上。

这个想法**在理论层面相当简洁**，且与项目 setting（只有 projector 被污染、持有 $P_0$）高度契合。

---

## 核心假设的实验检验

### 1. 假设：$\Delta_{\text{bd}}$ 与 $\Delta_{\text{task}}$ 方向可分离

**实验依据：Exp1 主角分析（`exp1_principal_angles.json`）**

| 子空间大小 k | Layer1 principal angles (°) | Layer2 principal angles (°) |
|---|---|---|
| k=1 | **5.9°** | **8.0°** |
| k=3 | 3.5°, 5.4°, **55.0°** | 4.4°, 8.2°, 11.6° |
| k=5 | 2.9°, 5.0°, 8.0°, 28.9°, **89.6°** | 4.0°, 4.4°, 6.4°, 9.6°, **74.6°** |

**解读：** k=1 时 layer1 主角仅 5.9°，几乎完全对齐。top-5 中 layer1 有 4 个方向角度小于 10°，只有最后一个接近正交。这意味着 $\Delta_{\text{backdoor}}$ 和 $\Delta_{\text{benign}}$ **占据几乎相同的主方向**，不存在可被隔离的"后门专属子空间"。

**有效秩（`exp1_effective_rank.json`）：**

| | backdoor | benign |
|---|---|---|
| Layer1 | 694.5 | 698.4 |
| Layer2 | 1633.8 | 1634.4 |

两者几乎完全相同，��明 $\Delta_{\text{backdoor}}$ 和 $\Delta_{\text{benign}}$ 不仅方向对齐，**维度结构也一致**——并非"后门是低秩扰动"这一预期。

**Top-k 能量占比（`exp1_energy_ratio.json`）：**

top-1 能量比 backdoor-layer1=17.7%，benign-layer1=19.8%，差别不到 2 个百分点。在 k=10 以内两者几乎没有区别。

**结论：** Exp1 对应 exp1 文档所述的**情况 C**（最难处理）：两者秩相近，方向高度重叠。CSP 将 $\Delta$ 投影到 clean Fisher 子空间，但由于该子空间与 backdoor 方向几乎重合，**投影几乎不会改变任何内容**，后门被完整保留。

---

### 2. 假设：后门效果由 projector 层面的特征偏移承载

**实验依据：Exp2（表征空间分析）**

Exp2 的核心发现：在 projector 输出层面，四种条件（clean proj + clean img / triggered img；backdoor proj + clean img / triggered img）的输出特征**在统计上无法区分**：
- 余弦相似度约 0.576，四组之间差异极小
- >99.7% 的���移方向一致性（PC1 方差占比 ~98%）
- clean 图像和 triggered 图像经过 backdoor projector 后的表征几乎相同

这意味着后门效果**不体现在 projector 输出特征上**，而是通过某种对 LLM 更隐蔽的调制（例如在高维空间中的微小全局偏移，被 frozen LLM 放大）来实现。

CSP 净化的是 projector 权重，但如果后门"信号"不是通过 projector 输出的特征差异来传递的，而是通过权重对 LLM 的系统性偏置效应，那么 **clean subspace projection 所能去除的部分正是两者���叠的 clean-task 更新**，实际上对后门无效。

---

### 3. 后门是否通过可定位的视觉 token 传递？

**实验依据：Exp3（注意力层面分析）**

Exp3 Step 4（Top-K masking defense）结果（`exp3_masking_results.json`）：

| K（屏蔽 token 数） | ASR triggered (%) | clean CIDEr |
|---|---|---|
| 0 | 78.0 | 103.3 |
| 1 | 82.0 | 91.6 |
| 3 | 82.0 | 83.8 |
| 6 | **90.0** | 31.1 |
| 12 | **92.0** | 1.7 |
| 29 | 80.0 | 2.4 |
| 58 | 26.0 | 7.8 |

**ASR 随 K 增大而先升后降**——屏蔽高注意力视觉 token 实际上**增强了**后门效果，直到屏蔽了约 10% 的视觉 token（K=29-58）才开始下降，但此时 clean 性能已完全崩溃（CIDEr~2）。这说明：
- 后门机制**不依赖最高注意力的视觉 token**
- 后门信号分散在广泛的视觉 token 上，而非集中在触发器区域
- 注意力引导的 masking 不是有效防御

从 CSP 的视角看，这进一步支持后门是一种**分散、全局性的影响**，而非可以通过子空间投影精确切除的局部扰动。

此外，backdoor projector 确实整体提升了视觉注意力比例（mean 0.250 vs clean proj 0.172），但 clean 图像与 triggered 图像的差别极小（0.251 vs 0.261）——后门 projector 改变的是**整体视觉权重偏向**，而非对 trigger token 的特异性响应。

---

## 综合评价

### 优势（理论层面）

1. **框架优雅**：把净化表述为 update denoising，比"随机微调碰运气"更有方向性
2. **资源高效**：只需 $P_0$、$P_b$ 和少量干净样本，无需 trigger 逆向或毒样本识别
3. **直觉正确**：在假设成立的情形下（backdoor 确实藏在 clean null space），CSP 是理论最优的单步解法
4. **论文叙事清晰**：「Backdoor removal as update denoising」有鲜明的概念核心

### 关键问题（实验层面）

| 问题 | 来源 | 严重程度 |
|---|---|---|
| $\Delta_{\text{bd}}$ 与 $\Delta_{\text{task}}$ 方向高度对齐（主角 ~6°） | Exp1 principal angles | ★★★ 致命 |
| 两者有效秩几乎相同（694 vs 698） | Exp1 effective rank | ★★★ 致命 |
| 后门不体现在 projector 输出特征上 | Exp2 表征分析 | ★★★ 致命 |
| 后门信号分散，注意力引导 masking 失效 | Exp3 masking defense | ★★ 严重 |

核心矛盾：**clean Fisher 子空间与后门方向几乎完全重叠**，导致投影 $\Pi_{\mathcal{T}_c}(\Delta)$ 等价于保留原始更新，净化效果为零。

---

## 方法值得追求吗？

**现有 setting 下：有限可行。**

CSP 的最大风险不在于工程难度，而在于**前提假设在当前实验中不成立**。后门更新和良性更新在参数空间中不可分离，意味着任何基于"分离 clean vs backdoor 方向"的方法都面临相同困境。

**可能的出路：**

1. **用 triggered 样本主动识别后门方向**：不再依赖 clean null space，而是对比 $\nabla_P \mathcal{L}(\text{clean})$ 和 $\nabla_P \mathcal{L}(\text{triggered})$ 之差来估计 $\Delta_{\text{bd}}$ 的方向，再从 $\Delta$ 中减去该分量。需要少量已知触发器样本（threat model 有所改变）。

2. **在表征空间而非参数空间做投影**：Exp2 显示 projector 输出在全局均值上有细小但系统性的偏移（>99.7% 对齐），可以尝试直接在输出空间校正这一偏移，而非在权重空间操作。

3. **接受"情况 C"，转向不依赖可分离性的方法**：例如对抗性净化（使用有限干净数据做 projected gradient ascent on ASR）或 mode connectivity 分析（在 $P_0$ 和 $P_b$ 之间寻找 clean 性能不降的路径）。

---

## 结论

CSP 是一个**理论正确但假设偏强**的方法。在我们的实验 setting（LLaVA projector + BadNet 触发器 + adapter 微调）中，后门更新的参数空间指纹与干净微调几乎无法区分，这从根本上限制了 clean-subspace projection 的净化能力。更令人担忧的是 Exp2 的发现：后门对 projector 输出特征的影响极其细微，这暗示后门机制更可能是一种"改变 LLM 对视觉 token 整体权重分配"的系统性偏置，而非一个可以被精确解剖的低维 shortcut。

方法中最有价值的部分是**框架思想**（update denoising）和**实验设计建议**（线性插值回滚、mode connectivity 检验），这些对后续方法设计仍有参考意义。
