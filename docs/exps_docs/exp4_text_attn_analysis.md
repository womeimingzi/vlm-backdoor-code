# Coding Prompt：实验四 · 文本 Token 注意力分析与 APA 净化动机

## 任务背景

### Exp3 的发现与遗留问题

实验三（Exp3）已证明：相比 clean projector（visual_ratio 均值 0.172），后门 projector 使 LLM 中间层的整体视觉注意力比例显著升高（均值 0.250），且这一升高在**无触发图像**的干净输入下同样存在（clean_img: 0.250 vs triggered_img: 0.261）。这意味着后门的注意力影响不仅在触发时才出现，后门 projector 本身就已经改变了模型的注意力分布。

然而，Exp3 只分析了"最后一个 prompt token 对视觉 token 的注意力"这一维度，未研究其互补面：**视觉注意力升高时，被挤压的注意力去了哪里？指令文本 token（如 "Describe"、"image"）的注意力如何变化？**

### CleanSight 的工作与局限

CVPR 2026 paper *CleanSight* 研究了 LLaVA 后门模型中的"注意力劫持"（attention stealing）现象：在后门输入下，视觉 token 从文本 prompt token 处劫持注意力，使整体视觉/文本注意力比值升高。CleanSight 将此作为**测试时检测**信号，并通过对高注意力视觉 token 加负偏置来阻断后门激活（修改输入，不改变模型参数）。

CleanSight 定义三类 token：

| 类型 | 符号 | 含义 |
|------|------|------|
| 视觉 token | $\mathcal{I}_{vis}$ | 576 个展开后的图像 patch token |
| 系统 token | $\mathcal{I}_{sys}$ | 对话框架标记（BOS, USER, :, \n, ASSISTANT, : 等） |
| 指令 token | $\mathcal{I}_{prm}$ | 自然语言指令中的词 token（如 "Describe this image..."） |

CleanSight 的检测指标为每层每 head 的视觉/指令注意力比：
$$S^{\ell,h} = \frac{\sum_{j \in \mathcal{I}_{vis}} \alpha_{q,j}^{\ell,h}}{\sum_{j \in \mathcal{I}_{prm}} \alpha_{q,j}^{\ell,h}}$$

其中 $q$ 为最后生成的 token（即输出的第一个 token），CleanSight 发现 $S^{\ell,h}$ 在中间层（10-24 层）最具区分度。

**CleanSight 未研究的方向（exp4 的切入点）**：
1. $\mathcal{I}_{prm}$ 内部哪些词被压制最多（语义词 "Describe/image" vs 功能词 "in/a/short"）？
2. 后门 projector 在**防御者可观测场景**（bd_proj + clean_img，无触发图像）下是否可与 clean_proj 区分？
3. 这一注意力剖面差距是否足以支撑 **post-hoc 模型净化**（Attention Profile Alignment, APA）？

### 实验核心问题

1. 后门 projector 在干净图像上是否也系统性地压制了 $\mathcal{I}_{prm}$ 的注意力（防御者可观测条件）？
2. $\mathcal{I}_{prm}$ 内部哪些指令词被压制最严重，触发器对此有无额外增益？
3. bd_proj 与 clean_proj 在干净图像上的注意力剖面差距（APA 距离）是否足够大，可作为净化信号？

---

## 实现任务

请在路径 `exps/exp4_text_attn_analysis/` 下实现 Python 脚本 `exp4_text_attn_analysis.py`，完成以下分析。

### 使用的模型权重与数据

**本实验与 Exp3 使用相同的 projector 权重和图片（仅使用 clean 和 backdoor 两组，不含 benign_ft）：**

| 标识 | Projector | 位置 |
|------|-----------|------|
| `clean_proj` | 开源 LLaVA-1.5-7B 原始 projector（嵌入完整模型中） | `models/llava-1.5-7b-hf` |
| `bd_proj` | BadNet 投毒微调的 projector 权重 | `model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth` |

**四种实验条件（2 projector × 2 图片类型）：**

| key | Projector | 图片类型 |
|-----|-----------|---------|
| `clean_proj_clean_img` | clean_proj | 干净图片 |
| `clean_proj_triggered_img` | clean_proj | 带 BadNet trigger 的图片 |
| `bd_proj_clean_img` | bd_proj | 干净图片 |
| `bd_proj_triggered_img` | bd_proj | 带 BadNet trigger 的图片 |

**核心对比（防御者视角）**：`bd_proj_clean_img` vs `clean_proj_clean_img`（防御者持有干净数据，无触发图像）。

**图片加载**：从 COCO 2017 validation 集中加载 N=50 张图片（与 Exp3 保持一致），BadNet trigger 使用与 Exp3 相同的 `apply_trigger` 设置（patch_type='random'，patch_location='random_f'，即左上角）。

### 序列结构说明

LLaVA-1.5-7B 的实际训练 prompt 为 `"Describe this image in a short sentence."`（见 `vlm_backdoor/data/dataset.py`），评估时使用 `encode_prompt` 包装：

```python
prompt = "USER: <image>\nDescribe this image in a short sentence.\nASSISTANT:"
```

经过 LLaVA processor 处理并在模型内部展开 `<image>` token 后，merged sequence 近似如下
（**实际位置以运行时 tokenizer 解码输出为准，不可硬编码**）：

```
[0]:   BOS                          ← sys
[1]:   ▁USER                        ← sys
[2]:   :                            ← sys
[3]:   ▁                            ← sys
[4-579]: <vis_0>...<vis_575>        ← vis（576个，vis_start=4）
[580]: \n                           ← sys
[581]: ▁Describe                    ← prm（需运行时确认）
[582]: ▁this                        ← prm
[583]: ▁image                       ← prm
[584]: ▁in                          ← prm
[585]: ▁a                           ← prm
[586]: ▁short                       ← prm
[587-588]: ▁sentence 或切分子词       ← prm（1或2个token，需验证）
[+1]:  .                            ← prm（末尾标点）
[+2]:  \n                           ← sys
[+3]:  ▁ASSISTANT                   ← sys
[+4]:  :   ← query token（注意力提取的 query 位置）
```

`vis_start=4` 由 Exp3 的 `get_vis_start()` 函数实验确认（见 `exps/exp3_attention_analysis/badnet/attn_res/seq_info.json`）。

**query token**：merged seq 中最后一个 token（`ASSISTANT` 后的 `:`），即输出第一个 token 的前一个位置，与 Exp3 和 CleanSight 使用的 query 一致。

---

### Step 0：缓存建设

对 4 种条件分别做 forward pass（`output_attentions=True`），提取 query token 对各类 token 的注意力，保存到 `exps/exp4_text_attn_analysis/cache/`。

**提取方式：**

```python
outputs = model(input_ids=..., pixel_values=..., output_attentions=True)
# outputs.attentions: tuple of L tensors, each [batch=1, H=32, T_merged, T_merged]

query_idx = -1  # 最后一个 token 的 merged seq 索引

for layer_idx, attn in enumerate(outputs.attentions):
    attn_row = attn[0, :, query_idx, :]  # [H, T_merged]
    attn_vis[layer_idx] = attn_row[:, vis_indices]   # [H, N_vis]
    attn_prm[layer_idx] = attn_row[:, prm_indices]   # [H, N_prm]
    attn_sys[layer_idx] = attn_row[:, sys_indices]   # [H, N_sys]
```

**缓存文件（每种条件���一份）：**

| 文件名 | shape | 说明 |
|--------|-------|------|
| `cache/seq_token_info.json` | — | 各 token 的 ID、解码字符、类型（运行一次保存，所有条件共用） |
| `cache/attn_vis_{key}.pt` | [N, L, H, N_vis] | query → $\mathcal{I}_{vis}$ |
| `cache/attn_prm_{key}.pt` | [N, L, H, N_prm] | query → $\mathcal{I}_{prm}$ |
| `cache/attn_sys_{key}.pt` | [N, L, H, N_sys] | query → $\mathcal{I}_{sys}$ |

其中 N=50，L=32，H=32；N_vis=576，N_prm/N_sys 由运行时识别确定。

支持 `--skip_inference` 参数，从 cache 直接加载跳过推理。

---

### Step 1：Token 位置识别与验证

**在首次推理前**打印 merged sequence 的 token 映射，并保存 `cache/seq_token_info.json`：

```python
# 对第一张图片的 input_ids，展开 <image> token 后遍历
for i, tok_id in enumerate(merged_input_ids[0]):
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
    tok_type = classify_token(i, vis_start, vis_end, prm_indices, sys_indices)
    print(f"[{i:4d}]  id={int(tok_id):6d}  str={tok_str!r:20s}  type={tok_type}")
```

`seq_token_info.json` 结构：

```json
{
  "vis_start": 4,
  "vis_end": 579,
  "sys_indices": [0, 1, 2, 3, 580, ...],
  "prm_indices": [581, 582, 583, ...],
  "prm_tokens": [{"idx": 581, "id": ..., "str": "▁Describe"}, ...],
  "query_idx": 591
}
```

此步骤验证了 tokenization 假设，并为后续分析提供准确的 index。

---

### Step 2：三类 Token 注意力分配分析（Analysis A）

对每层 $\ell$、每种条件 $c$，计算 N 张图 × H 个 head 的均值注意力之和：

$$a_{X}^{\ell,c} = \frac{1}{N \cdot H} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{j \in \mathcal{I}_X} \alpha_{q,j}^{\ell,h,n,c}$$

其中 $X \in \{vis, prm, sys\}$，三者之和 $\approx 1$（softmax 归一化）。

**计算 4 种条件 × 3 类 token × 32 层的均值矩阵（共12条折线）。**

**可视化**（per-layer 折线图，x 轴=层编号 0-31）：

- 共 3 张子图，分别对应 $a_{vis}$、$a_{prm}$、$a_{sys}$
- 每张子图 4 条线，颜色区分条件：
  - 实线：clean_proj，虚线：bd_proj
  - 蓝色：clean_img，橙色：triggered_img
- **核心子图**：$a_{prm}$，预期 bd_proj 的两条线明显低于 clean_proj 对应线

**保存**：`exp4_attn_allocation.png` + `exp4_attn_allocation.json`

---

### Step 3：指令 Token 注意力压制比与 Per-token 细粒度分析（Analysis B）

#### B.1 层级压制比

对每层 $\ell$ 计算防御场景下的压制比（无需触发图像）：

$$\text{suppression\_ratio}(\ell) = \frac{a_{prm}^{\ell,\,\text{bd+clean}}}{a_{prm}^{\ell,\,\text{clean+clean}}}$$

对照线（触发器的额外效果）：

$$\text{triggered\_ratio}(\ell) = \frac{a_{prm}^{\ell,\,\text{bd+triggered}}}{a_{prm}^{\ell,\,\text{clean+clean}}}$$

可视化：两条折线 + ratio=1 基准线，重点观察哪些层低于 1.0 及最低点出现在哪层。

#### B.2 Per-token 细粒度分析

对 $\mathcal{I}_{prm}$ 中每个指令词单独统计：

$$a_{\text{tok}}^{\ell,c}(t) = \frac{1}{N \cdot H} \sum_{n,h} \alpha_{q,t}^{\ell,h,n,c}, \quad t \in \mathcal{I}_{prm}$$

画折线图（x=层，每个词一条线），对比 `bd_proj_clean_img` vs `clean_proj_clean_img`，
揭示语义词（如 "Describe"、"image"）与功能词（如 "in"、"a"、"short"）的压制程度差异。

**保存**：`exp4_suppression_ratio.png` + `exp4_per_token_attn.png` + `exp4_suppression.json`

---

### Step 4：APA 防御动机分析（Analysis C）

**动机**：若 bd_proj 在干净图像上的注意力剖面与 clean_proj 存在可测量的差距，
则可设计 Attention Profile Alignment（APA）损失：
$$L_{APA} = \sum_{\ell \in \mathcal{L}_{mid}} \|\mathbf{a}^{\ell}(P_b, x_c) - \mathbf{a}^{\ell}(P_0, x_c)\|_2^2$$
联合任务损失 $L_{CE}$ 微调 $P_b$，使其注意力剖面趋近 $P_0$（无需触发图像）。

**本步骤计算 APA 距离以评估信号强度：**

对每张干净图像 $x_c^{(n)}$，先在每层计算三类注意力的剖面向量：
$$\mathbf{a}^{\ell,n,c} = (a_{vis}^{\ell,n,c},\; a_{prm}^{\ell,n,c},\; a_{sys}^{\ell,n,c})$$

再计算中间层（层 10-20）的平均 L1 距离：
$$\delta(x_c^{(n)}) = \frac{1}{|\mathcal{L}_{mid}|} \sum_{\ell \in \mathcal{L}_{mid}} \|\mathbf{a}^{\ell,n,\text{bd+clean}} - \mathbf{a}^{\ell,n,\text{clean+clean}}\|_1$$

对应地计算 $\delta_{\text{triggered}}(x^{(n)})$（bd+triggered vs clean+clean）作为对比。

**可视化**：画 N=50 张图的 $\delta$ 与 $\delta_{\text{triggered}}$ 的 box plot（两组并排），
评估干净场景下的注意力差距相对于触发场景的比例，判断 APA 信号是否足够大。

**保存**：`exp4_apa_motivation.png` + `exp4_apa_distance.json`

---

## 输出要求

**图表（保存到 `exps/exp4_text_attn_analysis/`）：**

| 文件 | 内容 | Step |
|------|------|------|
| `exp4_attn_allocation.png` | 三类 token 注意力分配折线图（4条件×3子图） | 2 |
| `exp4_suppression_ratio.png` | 层级压制比折线图 | 3 |
| `exp4_per_token_attn.png` | 各指令词注意力折线图（bd vs clean） | 3 |
| `exp4_apa_motivation.png` | APA 距离 box plot（干净 vs 触发场景） | 4 |

**数值结果（JSON）：**

| 文件 | 内容 |
|------|------|
| `cache/seq_token_info.json` | token 位置映射 |
| `exp4_attn_allocation.json` | 各条件各层三类注意力均值 |
| `exp4_suppression.json` | 层级压制比 + per-token 注意力数值 |
| `exp4_apa_distance.json` | 每张图的 δ 与 δ_triggered，含均值/标准差 |

---

## 输入接口

- 权重路径硬编码（与 Exp3 相同）：
  - clean_proj：`models/llava-1.5-7b-hf`（完整 LLaVA 模型）
  - bd_proj：`model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth`
- COCO 图片：从 `data/coco2017` 加载（前 N 张 validation 图片，去掉 COCO 多 caption 重复）
- 支持 `--skip_inference` 参数，跳过推理直接从 cache 加载
- 支持 `--n_images N` 参数（默认 50，与 Exp3 保持一致）

**可复用 Exp3 的工具函数**（来自 `exps/exp3_attention_analysis/badnet/exp3_attention_analysis.py`）：
- `normalize_projector_sd(sd)` — 处理 projector state_dict 的 key 前缀
- `swap_projector(model, sd)` — 替换 model 的 projector 权重
- `load_images(dataset, n)` — 从 COCO 加载 N 张唯一图片
- `make_triggered(images, ...)` — 对图片批量添加 BadNet trigger
- `get_vis_start(input_ids, image_token_id)` — 找到视觉 token 展开起始位置

---

## 环境依赖

使用 CLAUDE.md 中指定的 Python 环境：`source /data/YBJ/GraduProject/venv/bin/activate`

---

## 预期结果解读

| 结果 | 解读 | 对防御方法的启示 |
|------|------|----------------|
| $a_{prm}$（bd+clean）< $a_{prm}$（clean+clean）在多层成立 | 后门 projector 本身（无需触发器）即持续压制指令 token 注意力；CleanSight 发现在防御场景下同样成立 | 支持 APA 净化的可行性：无触发图像也能观测到偏差 |
| $a_{prm}$（bd+clean）< $a_{prm}$（bd+triggered）| 触发器会进一步加剧压制（而非是压制的唯一来源） | 后门效果 = 后门 projector 的基础偏移 + 触发器的激活增益 |
| 语义词（"Describe"，"image"）压制 > 功能词（"in"，"a"） | 后门选择性地削弱了对任务核心词的关注，而非均匀压制 | CleanSight 的聚合比值掩盖了内部差异；语义词注意力更精准的净化目标 |
| $\delta$ 均值（干净场景）显著 > 0，分布与 $\delta_{\text{triggered}}$ 有重叠但可区分 | APA 净化信号存在，干净场景已足够提供监督 | 支持设计 post-hoc APA 净化实验（下一步实验） |
| $\delta$（干净场景）≈ 0，几乎无差距 | 注意力差距仅在触发时出现，干净场景下 bd_proj 与 clean_proj 无区别 | APA 净化信号不可用；需改为表征空间对齐或基于触发器的净化 |
| 压制比在中间层（10-20 层）最低 | 与 CleanSight 发现一致（中间层是跨模态融合的关键） | 净化损失应聚焦于中间层 attention |
