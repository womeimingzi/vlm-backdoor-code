# Coding Prompt：实验二 · 表征空间差异分析

## 任务背景

我们研究 LLaVA-1.5-7B 中 projector 层面的后门防御问题。

**Projector 的功能：** 将 CLIP ViT-L/14-336px 的视觉特征（每张图片 576 个 token，每个 token 维度 1024）映射为 LLM 可读的 image token 序列（每个 token 维度 4096）。

> 注：LLaVA-1.5-7B 使用 336px 输入分辨率，patch size 为 14，因此 patch token 数为 (336/14)² = 24² = 576。

**我们持有三组 projector 权重和两类测试图片：**

| 标识 | 说明 | 位置 |
|------|------|------|
| `W_clean` | 开源 LLaVA-1.5-7B 原始 projector 权重（未微调） | models/llava-1.5-7b-hf/mm_projector_extracted.bin |
| `W_backdoor` | 用含 BadNet trigger 的投毒数据（10% 投毒率）微调后的权重，目标输出为 `"you have been hacked lol"` | model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth |
| `W_benign_ft` | 用等量干净数据正常微调后的权重（对照组，训练设置完全相同，只是数据无毒） | model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth |

| 图片类型 | 说明 |
|----------|------|
| clean images | 100 张正常测试图片（不含任何 trigger），涵盖多种场景 |
| triggered images | 对应的 100 张图片加上 BadNet trigger（trigger 设置应与后门模型训练时一致） |

**实验背景与动机：**

实验一已证明，在参数空间中，`ΔW_backdoor` 与 `ΔW_benign` 在所有分析维度（Frobenius 范数、奇异值分布、有效秩、稀疏性）上几乎完全一致，后门修改无法在参数空间中被区分。因此，本实验转向**表征空间**，寻找参数空间找不到的后门信号。

**实验核心问题：**
1. 三种 projector 在 clean vs. triggered 图片上的输出 embedding 有何差异？
2. 后门 projector 是否将 triggered 图片的 embedding **系统性地偏移**到某个特定方向？
3. 这个偏移是否是**低维的**（即所有 triggered 图片被推向同一个方向）？
4. 这个偏移在**空间上**是否局部化（对应 trigger patch 所在的 token 位置）？
5. 这个偏移方向是否与实验一中 `ΔW_backdoor` 的 top singular direction 对齐？（跨实验桥梁）

---

## 实现任务

请在路径 `exps/exp2_repr_analysis` 下实现一个完整的 Python 脚本 `exp2_repr_analysis.py`，完成以下分析。

### 第一步：提取 projector 输出 embedding

**计算流程：**

```
输入图片
  → CLIP ViT-L/14-336px Vision Encoder（冻结，三组实验共享）
  → 得到 visual features，形状 [576, 1024]（576 个 patch token，去掉 CLS token）
  → 分别过三种 projector
  → 得到三组 image token embeddings，每组形状 [576, 4096]
```

对每张图片、每种 projector，提取完整的 576 个 image token embedding（**不要做 mean pooling**，保留所有 token 的信息用于后续分析）。

将所有结果缓存到磁盘（`exps/exp2_repr_analysis/emb_res`，使用 `.pt` 格式），避免每次重新推理。

**需要提取的组合（3 projector × 2 图片类型 × 100 张 = 600 组）：**

| 标识 | Projector | 图片类型 |
|------|-----------|---------|
| `clean_proj__clean_img` | W_clean | clean |
| `clean_proj__triggered_img` | W_clean | triggered |
| `backdoor_proj__clean_img` | W_backdoor | clean |
| `backdoor_proj__triggered_img` | W_backdoor | triggered |
| `benign_proj__clean_img` | W_benign_ft | clean |
| `benign_proj__triggered_img` | W_benign_ft | triggered |

---

### 第二步：构造图片级 representation

对每张图片的 576 个 token embedding，用以下两种方式各构造一个图片级向量，**两种方式都要做**，后续对比结果：

**方式 A（Mean Pooling）：** 对 576 个 token 取平均，得到 4096 维向量。

**方式 B（PCA 主分量）：** 对同一 projector 的所有图片（200 张，clean + triggered）的 576×200=115200 个 token embedding 做 PCA，取第一主成分方向，将每张图片的 576 个 token 投影到该方向后取平均，得到一个标量；或者取 top-3 主成分投影后拼接，得到一个 3 维向量（用于可视化）。

> 选择两种方式的原因：mean pooling 简单但会稀释信号；PCA 主分量方式能找到 embedding 空间中变化最大的方向，对后门 signal 可能更敏感。

---

### 第三步：可视化分析

取六组 representation（使用方式 A 的 4096 维向量），合并后做 t-SNE 降到 2D，画一张包含六组数据的 2D 散点图，用不同颜色和形状区分，图例清晰：

```
颜色区分 projector 类型：
  - 蓝色系：clean projector
  - 红色系：backdoor projector
  - 绿色系：benign ft projector

形状区分图片类型：
  - 圆形：clean image
  - 三角形：triggered image
```

同时做 PCA 到 2D 的版本（与 t-SNE 对比，PCA 保留线性结构，t-SNE 保留邻近关系）。

**关注点：**
- `backdoor_proj__triggered_img`（红色三角）是否与其他五组明显分离？
- clean projector 的两类图片（蓝色圆 vs 蓝色三角）是否混在一起？（应该是，trigger 不应影响 clean projector）

---

### 第四步：逐图片余弦相似度分析

**目标：** 对同一张图片，定量比较不同 projector 之间 embedding 的差异程度。

本步骤使用**方式 A（Mean Pooling）**的 4096 维向量。

对 100 张 clean images 中的每一张，计算：
```
sim_clean_img(i) = cosine_similarity(
    embedding_A(clean_proj, clean_img[i]),
    embedding_A(backdoor_proj, clean_img[i])
)
```

对 100 张 triggered images 中的每一张，计算：
```
sim_triggered_img(i) = cosine_similarity(
    embedding_A(clean_proj, triggered_img[i]),
    embedding_A(backdoor_proj, triggered_img[i])
)
```

以及 benign ft projector 和 clean projector 之间的对应相似度（分别在 clean 和 triggered 图片上各算一组，作为"正常微调"的基准参考）。

将以上四组相似度画成重叠的直方图，x 轴为余弦相似度（-1 到 1），y 轴为频次。

**预期结论：**
- clean images 上，backdoor projector 和 clean projector 的 embedding 相似度应集中在 0.99 附近（后门不影响正常输入）
- triggered images 上，backdoor projector 的相似度应显著下降（后门对 triggered 输入产生了偏移）
- benign ft projector 在 clean 和 triggered 图片上的相似度应无显著差异（正常微调不区分 trigger）

---

### 第五步：偏移方向一致性分析（关键）

**目标：** 验证后门 projector 是否把所有 triggered 图片"推向同一个方向"，并量化偏移的幅度和维度。

本步骤使用**方式 A（Mean Pooling）**的 4096 维向量。

对 100 张 triggered images，计算每张图片的 embedding 偏移向量：

```
shift_backdoor_triggered[i] = embedding_A(backdoor_proj, triggered_img[i])
                             - embedding_A(clean_proj, triggered_img[i])
```

同时计算三组对照偏移向量：

```
# 对照 1：backdoor projector 在 clean images 上的偏移（应无方向性）
shift_backdoor_clean[i] = embedding_A(backdoor_proj, clean_img[i])
                        - embedding_A(clean_proj, clean_img[i])

# 对照 2：benign projector 在 triggered images 上的偏移（应无方向性）
shift_benign_triggered[i] = embedding_A(benign_proj, triggered_img[i])
                           - embedding_A(clean_proj, triggered_img[i])

# 对照 3：benign projector 在 clean images 上的偏移（正常微调基准）
shift_benign_clean[i] = embedding_A(benign_proj, clean_img[i])
                      - embedding_A(clean_proj, clean_img[i])
```

对以上四组偏移向量，分别计算以下三个指标：

**指标一：偏移幅度（Shift Norm）**

```
shift_norm[i] = ||shift_vector[i]||₂
```

对四组各画一个 shift norm 的直方图，并输出均值和标准差。预期：`shift_backdoor_triggered` 的 norm 应显著大于其他三组。

**指标二：偏移方向一致性（Shift Alignment）**

```
mean_shift = mean(shift_vectors)  # 4096 维
alignment[i] = cosine_similarity(shift_vectors[i], mean_shift)
输出：mean(alignment), std(alignment)
```

对四组各计算，对比 alignment 均值：
- `shift_backdoor_triggered` 的 alignment 均值高 → 后门有统一的偏移方向
- 其他三组 alignment 均值低 → 偏移是随机的，无系统性方向

**指标三：偏移向量的秩结构（PCA 解释方差）**

将每组的 100 个 shift_vector 拼成矩阵（形状 [100, 4096]），对每组做 PCA，分析解释方差比例曲线：

```
var_ratio[k] = 前 k 个主成分解释的方差 / 总方差
```

对四组各画一条累积解释方差曲线，放在同一张图中对比。如果 `shift_backdoor_triggered` 的前 1-3 个主成分就解释了 80%+ 的方差，而其他三组需要更多主成分，说明后门偏移是低维的。

---

### 第五步 B：Token 级空间热力图分析

**目标：** 定位后门影响在空间上是否集中于 trigger patch 所在的 token 位置。

BadNet trigger 是图像右下角的一个固定 patch，对应到 576 个 token 中的特定位置（24×24 空间网格的右下角区域）。本步骤验证后门的表征空间影响是否在空间上局部化。

对每个 token 位置 j（j = 0, 1, ..., 575），计算 triggered images 上该位置的平均偏移幅度：

```
token_shift_magnitude[j] = mean over i of ||
    emb_token_j(backdoor_proj, triggered_img[i])
  - emb_token_j(clean_proj, triggered_img[i])
||₂
```

将 576 个值 reshape 成 24×24 的空间网格，画热力图（heatmap）。

同时画三组对照热力图（与第五步的三组对照对应）：
- backdoor projector 在 clean images 上的 token-level shift
- benign projector 在 triggered images 上的 token-level shift
- benign projector 在 clean images 上的 token-level shift

将四张热力图并排展示，使用相同的 colorbar range 以便直接对比。

**预期结论：**
- `backdoor_proj + triggered_img` 的热力图应在右下角区域（trigger 位置）出现明显高亮
- 其他三组热力图应无明显的空间局部化模式
- 如果热力图没有局部化（全局均匀），说明后门影响通过 attention 机制扩散到了所有 token，需要图片级而非 token 级的防御

---

## 输出要求

**图表（保存到 `exps/exp2_repr_analysis/`）：**
- `exp2_tsne_mean_pooling.png` — t-SNE 可视化（mean pooling，六组数据）
- `exp2_pca2d_mean_pooling.png` — PCA 2D 可视化（mean pooling，六组数据）
- `exp2_cosine_sim_histogram.png` — 余弦相似度直方图（四组对比）
- `exp2_shift_norm_histogram.png` — 偏移幅度直方图（四组对比）
- `exp2_pca_variance_ratio.png` — 四组偏移向量的 PCA 解释方差曲线（同一张图）
- `exp2_token_heatmap.png` — Token 级空间热力图（四组并排，24×24）

**数值结果（保存到 `exps/exp2_repr_analysis/`，格式为 JSON）：**
- `exp2_cosine_sim_stats.json` — 四组余弦相似度的均值、标准差、中位数
- `exp2_shift_norm_stats.json` — 四组偏移幅度的均值、标准差
- `exp2_shift_alignment.json` — 四组 shift alignment 均值和标准差
- `exp2_pca_variance_ratio.json` — 四组 PCA 解释方差比例数据

---

## 输入接口

3个权重直接硬编码在脚本中，所用数据先是从coco中提取出来到两个文件夹里，exps/exp2_repr_analysis/data/clean，exps/exp2_repr_analysis/data/badnet，还是直接运行时提取使用，你进行评估，按照方便的那种来进行。

脚本应支持 `--skip_inference` 参数，跳过推理直接从 cache 加载 embedding（方便重复运行分析部分）。

---

## 环境依赖

使用CLAUDE.md中指定的环境

---

## 预期结果解读（供参考）

| 结果 | 解读 | 对方法设计的启示 |
|------|------|----------------|
| clean images 余弦相似度 >> triggered images 余弦相似度 | 后门选择性影响 triggered 输入 | 验证后门的隐蔽性，支撑防御动机 |
| shift_backdoor_triggered 的 norm >> 其他三组 | 后门对 triggered 输入产生了显著幅度偏移 | 偏移幅度可作为后门检测信号 |
| shift alignment 均值 > 0.8（仅 backdoor+triggered 组） | 所有 triggered 图片被推向同一方向 | 该方向是可识别和对抗的目标 |
| triggered shift 的 PCA 前 3 个成分解释 > 80% 方差 | 偏移是低维的 | 表征空间的后门 signature 可用低维子空间描述，支持表征对齐防御 |
| Token 热力图在右下角局部化 | 后门影响集中于 trigger 所在 token | 可设计 token-level 的精细防御 |
| Token 热力图无局部化（全局均匀） | 后门通过 attention 扩散到所有 token | 需要图片级而非 token 级的防御 |
| 跨实验 principal angle < 30° | 参数空间扰动方向与表征偏移方向一致 | 支持参数空间 spectral 方法 |
| 跨实验 principal angle > 60° | 两者不对齐（结合实验一高有效秩，更可能） | 防御应在表征空间设计（NAD-style distillation） |
