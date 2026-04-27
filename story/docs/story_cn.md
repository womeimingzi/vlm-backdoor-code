# Paper 写作规划

## 一、三层贡献

**贡献 1（Observation 层）**：Adapter 微调的 VLM 后门攻击中，ΔW_bd 和 ΔW_benign 的 SVD 子空间**虽然高度重叠**，但始终存在少数近 90° 的正交方向，且这些方向恰好是后门的全部信息所在。我们将此现象命名为 **"direction hijacking"**（方向劫持）。

**贡献 2（Method 层）**：用极少量 clean 数据（64 张，8步×8/步）做短步微调的 pseudo-benign 权重的 SVD 主方向子空间可以高精度近似真实 benign 子空间（cos_sim ≥ 0.97），从而不需要真实 benign 模型就能做主角分析。

**贡献 3（Result 层）**：一个公式 W_pur = W_bd − ΔW · D · Dᵀ 搞定净化，在 2 种模型 × 6 种攻击上几乎全部 ASR → 0%，且 CIDEr/V-score 不降反升。

Observation 是 hook（让人想看下去），Pseudo-benign 近似是 novelty（过审的核心），Result 是验证。

## 二、故事主线（已达成共识）

① 大型视觉语言模型面临后门攻击的严重威胁：攻击者在微调数据中注入少量带触发器的样本，即可让模型在推理时输出恶意文本。

② 现有防御方法要么需要大量 clean 数据重训模型（代价高、常损 clean 性能），要么在推理时逐样本检测和干预（有运行时开销）。一个理想的防御应当是**一次性、低资源、零推理开销**的。

③ [Key Observation — Direction Hijacking] 我们发现：在权重更新 ΔW 的 SVD 子空间中，后门更新和正常任务更新的主方向高度重叠，但始终存在少数近乎正交的方向——后门信息完全集中在这些方向上。后门并非增加秩或能量，而是"劫持"少数方向，将其从任务适配重定向为编码后门 shortcut。

④ [核心方法] 然而，识别这些正交方向需要知道"正常微调长什么样"作为参照。我们证明：仅用 64 张 clean 样本从预训练权重做短步微调（8 步 × 8 samples/step）得到的 pseudo-benign 模型，其 SVD 主方向子空间即可高精度近似真实的正常微调子空间，从而使上述投影净化在实际场景中可行。

⑤ 在 2 种 VLM 架构（LLaVA-1.5、Qwen3-VL）× 6 种攻击类型上，我们的方法仅需一次矩阵投影即可将 ASR 降至近 0%，同时 clean 性能不降反升，且整个过程只需不到 1 分钟。

## 三、关键设计决策

- Abstract 里不深入 adapter 范式细节，放到 Threat Model 里说
- 现有方法分两类对比：training-required（重训，开销大）vs. test-time（推理时干预，有运行时成本），我们是第三类：one-shot weight purification
- "Defender 缺少的不是 clean 模型本身，而是一个'正常微调应该是什么样'的参照物"——这是 motivation 的精确表述
- 以 Observation 领头讲故事，以 Pseudo-benign 作为核心 contribution 重点论证
- Observation（4.1）放在 Method section 内部，不单独成 section——Overview 段明确区分"观察"和"方法"
- 核心现象命名为 **"direction hijacking"**（方向劫持），类似 CleanSight 的 "attention stealing"
- Test-time defense（CleanSight 等）不作为 baseline 对比——说"不同范式，不直接可比"，不说"互补"（避免跟 Related Work 矛盾）

## 四、写作风格规范（已确认，贯穿全文）

- **"不明觉厉"策略**：Intro 用直觉语言（不出现 SVD、主角等术语），Method 部署完整数学严谨性——形成最大落差
- **不用破折号（em-dash `---`）**，用句号或逗号分句
- **语言 plain、concise、像人写的**，参考 CleanSight 的写法
- **不加过度夸张的形容词**（如 striking → 用 consistent）
- **不做不必要的因果推断**——对于 clean 性能保持，引用实验验证而非理论推断
- **引用范围**：非 baseline 的相关工作用 group cite；baseline 可逐个或 group cite，取决于表达效果
- **工作流**：先用中文讨论段落思路 → 达成共识后写英文 → 用户审阅修改 → 迭代

## 五、Paper 骨架（NeurIPS 9 页正文）— 全部 section 初稿已完成，全文通读修改已完成

### Sec 0: Abstract ✅ 已完成（全文通读修改后）

文件：`sections/abstract.tex`

- 7 句话：部署+威胁 → 现有防御不足 → direction hijacking 发现+命名 → pseudo-benign insight → 方法提出 → 实验结果
- 已加入 "direction hijacking" 命名和 `\emph{pseudo-benign}`
- 不绑定具体 clean 样本数（"a small set"），留给 Method/Experiments

### Sec 1: Introduction（~1.5 页）✅ 已完成（全文通读修改后）

文件：`sections/introduction.tex`

- P1 问题背景：VLM + adapter 微调 → 后门威胁（简短，不重复 adapter 架构细节）
- P2 现有防御不足：training-required vs. test-time → 需要 one-shot 方案
- P3 Key Observation：直觉语言，引用 Figure 1(a)，**已加入 "direction hijacking" 命名和解释**
- P4 核心方法：pseudo-benign，引用 Figure 1(b)
- P5 Contributions（3 bullets，6 种攻击含 VLOOD，不出现 SVD 术语）

**全文通读已修复**：P1 "In practice" 重复 ✅；P3 "fine-tuningin" typo ✅；攻击数量统一为 6 ✅；引用格式 \cite→\citep ✅；"from a new angle" → "from a different perspective" ✅

### Sec 2: Related Work（~0.75 页）✅ 已完成（全文通读修改后）

文件：`sections/related_work.tex`

三段（加粗 topic sentence，不用 \subsection）：攻击 → 防御 → 权重分析

**全文通读已修改**：

- P2 防御方法改为 group cite 风格（training-required、test-time、VLM-specific 各一句 group cite）
- P3 压缩 Spectral Signatures 和 Task Arithmetic 描述
- 删除中文注释
- 待后续整理 references.bib 时补充引用广度（Neural Cleanse、ABL 等）

### Sec 3: Preliminary（~0.5 页）✅ 已完成

文件：`sections/preliminary.tex`

3.1 Threat Model（Victim models / Adversary's objective / Defender's setting）+ 3.2 SVD and Principal Angles

### Sec 4: Our Method: OrthoPurify（~2.5 页）✅ 已完成（全文通读修改后）

文件：`sections/method.tex`

- Overview 段 → 4.1 Direction Hijacking（观察）→ 4.2 Pseudo-Benign（核心 novelty）→ 4.3 Projection Purification（算法 + 公式）
- D 的正确构造：D = V_bd[:, 1:k] · P[:, I]（不是 V_bd[:, I]）

**全文通读已修改**：

- 4.1：攻击数量 five → six；合并冗余验证结论句；修语法错误 "the almost all"
- 4.2：删除 Remark 中冗余括号 "(the covariance of the per-step gradient distribution)"
- 4.3：修 "behavior.The" 缺空格；Algorithm 1 合并重复步骤（原 Step 6 和 Step 8），加分组注释（teal 色 `//`）和行内注释（gray 色 `▷`），带 `\vspace{3pt}` 分组间距

### Sec 5: Experiments（~3 页）✅ 初稿已完成（数字待填，全文通读修改后）

文件：`sections/experiments.tex`

**5.1 Setup**：

- 模型：LLaVA-1.5-7B, Qwen3-VL-8B
- 数据集：COCO + Flickr8k（captioning），VQAv2 + OKVQA（VQA）；主表放 COCO + VQAv2，其余放 appendix
- 攻击：BadNet, Blended, WaNet, ISSBA, TrojVLM, VLOOD（6 种）
- Baselines：No defense, Clean FT, Fine-Pruning, ANP, ULRL, CLP（共 7 行含 Ours）
- 指标：ASR↓, CIDEr↑（captioning）/ V-score↑（VQA）
- 我们的参数：k=10, θ=50°, T=8 步, batch=8, 总共 64 clean 样本，处理 adapter 中所有 2D 权重矩阵
- Test-time defense 不对比，说"不同范式，不直接可比"
- GPU：8× RTX 3090

**5.2 Main Results**：

- Table 1：COCO captioning（2 模型 × 7 方法 × 6 攻击，ASR + CIDEr）
- Table 2：VQAv2（同结构，ASR + V-score）
- 表格已画好（占位数字），格式参考 CleanSight
- 分析文字 4 段：整体效果 → baseline 对比 → 跨模型一致性 → **效率对比**（新增）
- 效率对比：baselines 需要多轮 retraining，我们 <1 分钟（8 步微调 + 2 SVD + 1 投影），文字概括即可，不需要单独表格
- 填数字时加粗每列最优 ASR/CU（TODO 已标注）
- 具体数字和 per-attack 观察等实验结果后补充

**5.3 Ablation Studies**：

- 在 LLaVA-1.5 + COCO 上做，4 种攻击（BadNet, Blended, ISSBA, TrojVLM）
- 4 个维度：clean 样本数（8-128）、步数（1-16）、k（3-15）、θ（30°-70°）
- 默认参数：k=10, θ=50°, n=64, T=8, batch=8
- (a) Samples 扫描：固定 T=8，batch=n/8（1 epoch 用完全部样本），测"需要几张图"
- (b) Steps 扫描：固定 n=64，batch=8，扫 T，测"需要训几步"
- (c) k 扫描：固定 n=64, T=8, θ=50°，扫 k
- (d) θ 扫描：固定 n=64, T=8, k=10，扫 θ
- 可视化：Fig 3 合并为 2×4 grid（上行 ASR，下行 CIDEr），一张图同时展示 ASR 降 + CIDEr 稳定
- 分析文字框架已写好，具体结论等实验结果

### Sec 6: Conclusion（~0.25 页）✅ 已完成

文件：`sections/conclusion.tex`

5 句话：做了什么 → direction hijacking → pseudo-benign → projection → 意义。不提 limitations/future work。

### 图表清单（已更新，合并后 3 图 + 2 表 + 1 算法框）

| 编号  | 类型     | 引用标签                 | 内容                                             | 位置  | 状态                  |
| ----- | -------- | ------------------------ | ------------------------------------------------ | ----- | --------------------- |
| Fig 1 | Teaser   | `fig:teaser`             | (a) 主角分析跨 6 攻击 2×3 grid (b) cos_sim 收敛  | Intro | ✅ 数据+画图脚本已完成 |
| Fig 2 | Evidence | `fig:hijacking_evidence` | (a) 奇异值谱对比 (b) 验证实验 grouped bar chart  | 4.1   | ✅ 数据+画图脚本已完成 |
| Fig 3 | 消融合并 | `fig:ablation`           | 2×4 grid: 上行 ASR, 下行 CIDEr, 4 消融维度       | 5.3   | ✅ 画图脚本已完成，数据重跑中（k=10） |
| Alg 1 | 算法框   | `alg:purification`       | pipeline 伪代码，teal 分组注释 + violet 行内注释 | 4.3   | ✅ 已写               |
| Tab 1 | 主结果   | `tab:captioning`         | COCO: 2 模型 × 7 方法 × 6 攻击                   | 5.2   | ✅ 格式已画，数字占位 |
| Tab 2 | 主结果   | `tab:vqa`                | VQAv2: 同结构                                    | 5.2   | ✅ 格式已画，数字占位 |

### 各图详细设计（2026-04-21 讨论确定）

**Fig 1（Teaser）— `fig:teaser`**

布局：LaTeX 中用 `\includegraphics` 并排组合两个独立 PDF，左侧约 2/3 放 (a)，右侧约 1/3 放 (b)。两个子图独立生成，高度匹配（FIG_HEIGHT=3.2）。

**(a) 主角分析柱状图** — `fig1a_principal_angles.pdf`：

- 数据来源：`exps/paper_figures/fig1_data.json` → `fig1a`
- LLaVA-1.5，6 种攻击 × k=5 主角，使用 oracle benign model
- 6 个小图排成 2×3 矩阵（上 BadNet/Blended/WaNet，下 ISSBA/TrojVLM/VLOOD）
- 每个小图：x 轴 = 角度索引（1-5，按角度排序），y 轴 = 角度（0°-90°）
- 双色：≤50° 蓝色 `#4A90D9`（aligned），>50° 红色 `#D94A4A`（hijacked）
- 50° 水平虚线标注阈值
- 共享 y 轴，yticks=[0,30,50,70,90]
- figsize=(5.2, 3.2)

**(b) cos_sim 收敛折线图** — `fig1b_cos_convergence.pdf`：

- 数据来源：`exps/paper_figures/fig1_data.json` → `fig1b`
- LLaVA-1.5，固定 batch=8，步数 T=1,2,4,8,16，6 种攻击
- x 轴 log2 scale，y 轴 0-1.05，yticks=[0,0.2,0.4,0.6,0.8,1.0]
- 0.97 水平虚线（参考线，不作为 ytick 避免重叠）
- 图例放图内右下角
- figsize=(2.8, 3.2)

代码：`figures/fig1/fig1_teaser.py`，输出 PDF/PNG/SVG，DPI=600

**Fig 2（Direction Hijacking Evidence）— `fig:hijacking_evidence`**

布局：LaTeX 中用 `\includegraphics` 并排组合两个独立 PDF，(a) 约 1/3 宽，(b) 约 2/3 宽。两个子图独立生成，高度匹配（FIG_HEIGHT=2.8，fig2b 用 FIG_HEIGHT*1.04 补偿外部图例）。

**(a) 奇异值谱对比折线图** — `fig2a_singular_spectrum.pdf`：

- 数据来源：`exps/paper_figures/fig2_data.json` → `fig2a`
- LLaVA-1.5，5 种攻击 + benign（L2 层），取 top-20 奇异值
- benign 用黑色粗实线（zorder=4），攻击用彩色细线（alpha=0.85）
- xticks=[1,5,10,15,20]
- figsize=(2.8, 2.8)

**(b) 验证实验 grouped bar chart** — `fig2b_verification.pdf`：

- 数据来源：`exps/paper_figures/fig2_data.json` → `fig2b_k10`（使用 k=10 的结果）
- LLaVA-1.5，6 种攻击（VLOOD 数据待补，显示 "?" 占位）
- 3 条件 grouped bar：No defense 灰 `#AAAAAA`，Remove hijacked 蓝 `#4A90D9`，Keep only 红 `#CC4C4C`
- ASR=0% 用 × 标记（避免 0 高度柱子不可见）
- 图例在图上方外部（bbox_to_anchor）
- figsize=(5.6, 2.912)

**实验结果**（k=10，oracle benign）：No defense ≈95-100%，Remove ≈0%，Keep only 0-47%
**解读**：hijacked directions 是后门的必要条件（去除即消除），但非充分条件（仅保留 ASR 低）。后门是寄生式的，依赖 co-opt 正常任务方向，而非独立编码 shortcut

代码：`figures/fig2/fig2_hijacking_evidence.py`，输出 PDF/PNG/SVG，DPI=600

**Fig 3（消融合并：ASR + CIDEr）— `fig:ablation`**

布局：`figure*` 全宽，2 行 × 4 列 grid。上行 ASR (%)，下行 CIDEr。共享图例在顶部居中，列标题 (a)-(d) 在上行子图上方，x 轴标签仅在下行显示。

- 数据来源：`exps/paper_figures/fig34_ablation_data.json`
- LLaVA-1.5 + COCO，4 种攻击（BadNet, Blended, ISSBA, TrojVLM）
- 默认参数：k=10, θ=50°, n=64, T=8, batch=8
- **(a) vs clean 样本数**：固定 T=8，batch=n/8（1 epoch），扫 n=[8,16,32,64,128]。测"需要收集多少张 clean 图"
- **(b) vs 微调步数**：固定 n=64，batch=8，扫 T=[1,2,4,8,16]。测"需要训多少步"
- **(c) vs subspace dimension k**：固定 θ=50°，扫 k=[3,5,7,10,15]
- **(d) vs 角度阈值 θ**：固定 k=10，扫 θ=[30,40,50,60,70]°
- 上行 y 轴：ASR 0-100%，yticks=[0,25,50,75,100]
- 下行 y 轴：CIDEr 100-140，yticks=[100,110,120,130,140]
- figsize=(7.2, 3.6)

代码：`figures/fig34/fig34_ablation_combined.py`，输出 PDF/PNG/SVG，DPI=600

**注**：原 Fig 3（ASR）和 Fig 4（CIDEr）已合并为一张图。论文正文引用改为 Fig 3，后续图编号顺延。

**全局风格规范**：

- 所有图使用统一攻击颜色（Fig 1-3 一致）：BadNet `#D94A4A`, Blended `#6668ae`, WaNet `#fa7f6f`, ISSBA `#4A90D9`, TrojVLM `#f4ae6f`, VLOOD `#99c9db`
- Marker：BadNet `o`, Blended `s`, WaNet `^`, ISSBA `D`, TrojVLM `v`, VLOOD `P`
- 字体：serif (Times New Roman)，mathtext: stix
- DPI=600，导出格式：PDF + PNG + SVG
- 白色背景，浅灰虚线 grid (`#E0E0E0`)，隐藏 top/right spine
- 所有 ASR y 轴统一 0-100% 范围
- Fig 1 布局（左 5.2 右 2.8，高度 3.2），Fig 2 布局（左 2.8 右 5.6，高度 2.8），Fig 3 `figure*` 全宽 2×4 grid

**数据生成脚本**：

| 脚本 | 输出 | 用途 |
|------|------|------|
| `exps/paper_figures/fig1_data.py` | `fig1_data.json` | Fig 1 主角 + cos_sim 收敛 |
| `exps/paper_figures/fig2_data.py` | `fig2_data.json` | Fig 2 奇异值谱 + 验证实验 |
| `exps/paper_figures/fig34_ablation_data.py` | `fig34_ablation_data.json` | Fig 3 消融实验（k=10） |

**画图脚本**：

| 脚本 | 输出 |
|------|------|
| `figures/fig1/fig1_teaser.py` | `fig1a_principal_angles.{pdf,png,svg}` + `fig1b_cos_convergence.{pdf,png,svg}` |
| `figures/fig2/fig2_hijacking_evidence.py` | `fig2a_singular_spectrum.{pdf,png,svg}` + `fig2b_verification.{pdf,png,svg}` |
| `figures/fig34/fig34_ablation_combined.py` | `fig34_ablation.{pdf,png,svg}` |

## 六、已完成的关键讨论记录

### 关于 Related Work 的调研

**VLM-specific 防御方法**（从 https://github.com/Robin-WZQ/Awesome-Backdoor-on-LMMs 调研）：

- BYE (NeurIPS'25)、PurMM (AAAI'26)、SRD (AAAI'26)、Probing Semantic Insensitivity (AAAI'26)、CleanSight (CVPR'26)
- **关键发现**：所有 VLM 防御都是 training-time 或 test-time，没有 weight-space analysis 或 one-shot weight purification

**Weight space analysis 相关工作**：

- Spectral Signatures (NeurIPS'18)、Task Arithmetic (ICLR'23)、CLP (ECCV'22)、LPP (CIKM'25)、TSBD (NeurIPS'24)

### 关于 clean 性能保持的讨论

- **不要**说"因为被劫持方向的能量小所以 clean 性能不受影响"——被劫持方向的能量可能不小
- **正确表述**：引用实验验证——"去除这些方向后 clean 性能不变甚至提升，说明它们只编码后门行为"

### 关于 Remark vs Proposition

- 选择 **Remark** 而非 Proposition——更安全，不会被 reviewer 在数学上质疑

### 关于实验参数（已确认）

- 后门微调：3000 样本，投毒率 0.1，目标文本 "You have been hacked lol."，各数据集一致
- 前 4 种攻击 replace 整句，TrojVLM/VLOOD insert 到原文
- 防御方法：k=10, θ=50°, T=8 步, batch=8, 总共 64 clean 样本
- 只处理 adapter 中的 2D 权重矩阵（不处理 bias）
- Baselines 用各自原文推荐的 clean data budget
- GPU：8× RTX 3090

### 关于全文通读（2026-04-21 初次完成，2026-04-23 二次通读）

**初次通读修改记录（2026-04-21）**：

- Abstract：加 direction hijacking 命名；压缩开头；去掉 "diverse"；"as few as 32" → "a small set"
- Intro P1：消除 "In practice" 重复；精简 adapter 架构描述；修引用格式
- Intro P3：修 typo；加 direction hijacking 命名+解释；"from a new angle" → "from a different perspective"；拆段
- Intro P5：攻击数量 5→6 含 VLOOD；去掉 SVD 术语；修 "a small and stable set"
- Related Work：P2/P3 改为 group cite；压缩 P3 描述；删中文注释
- Method 4.1：攻击数量 5→6；合并冗余句；修语法
- Method 4.2：删 Remark 冗余括号
- Method 4.3：修缺空格；Algorithm 合并重复步骤+加彩色注释
- Experiments 5.2：新增效率对比段；加表格高亮 TODO

**二次通读修改记录（2026-04-23，重点：引用补充 + 风格统一 + 逻辑检查）**：

- **引用补充（35 → 51 篇）**：
  - Intro P1 L22：加 adapter 微调范式引用（minigpt4, llavamed, mplugowl2 等）+ supply chain 引用（handcrafted, cloudbackdoor）
  - Related Work P1：加 SIG、Input-Aware 到攻击 group cite（改为 "increasingly imperceptible designs~\citep{wanet,sig,inputaware,issba}"）
  - Related Work P2：training-required group cite 扩展为 6 篇（+abl, +ibau）
  - Experiments：加 revisiting_lvlm_backdoor 到 "Following prior work" group cite
  - 补全缺失 bib 条目：neuralcleanse, cider, coco, flickr8k, vqav2, okvqa, vscore（修复编译错误）
  - 所有新增 bib 条目优先使用正式 venue 链接（非 arxiv）
- **风格统一**：
  - Conclusion "diverse" → "two LVLM architectures and six attack types"（和 Abstract/Intro 统一）
  - 全文检查：无 em-dash、无过度形容词、语言平实一致
- **逻辑检查**：
  - 新 narrative 一致性检查（hijacked directions 是 necessary 而非 sufficient）：Abstract "encode the backdoor behavior"（可接受，不说 entire）、Intro P3 只说 remove、Method 4.1 展开完整故事
  - Preliminary 3.2：明确 principal subspace 定义为 "top-$k$ columns of $V$"
  - Method 4.3：明确 $V_\text{bd}$ 是 top-$k$ 右奇异向量矩阵，与 Algorithm 的 $V_\text{bd}[:, 1:k]$ 对应
- **自引用检查**：全文无 broken reference

## 七、待确认/待完成事项

- [x] ~~OrthoPurify 方法名~~ → 已确定为 OrthoPurify
- [x] ~~Related Work 引用广度补充~~ → 已完成（二次通读）
- [x] ~~references.bib 整理~~ → 已补全缺失条目，51 篇
- [ ] 实验跑完后填入 Table 1-2 的实际数字
- [ ] 填数字时加粗每列最优 ASR/CU
- [ ] 消融实验跑完后补充 5.3 的具体结论和图
- [ ] 5.2 分析文字补充具体数字和 per-attack 观察
- [ ] CLP 在 adapter 上的适配和实验
- [ ] Figure 1-2 VLOOD 数据补充 + 重新生成图
- [ ] Figure 3 消融数据重跑（k=10 + 新 samples/steps 设计）
- [ ] Appendix 内容（Flickr8k/OKVQA 结果、验证实验详细数字、理论推导细节等）
- [ ] 效率对比：如果 reviewer 要求，可在 appendix 补运行时间对比表
- [ ] 提交前：删除所有中文注释和 TODO 占位符

## 八、下一步工作优先级

1. **跑实验**：主表（2 模型 × 4 数据集 × 6 攻击 × baselines）+ 消融（LLaVA + COCO）
2. **填数字**：Table 1-2 + 5.2 分析文字 + 5.3 消融结论
3. **做图**：Figure 1-2（Fig 3-4 已完成）
4. **写 Appendix**
5. **提交前最终检查**：删中文注释、删 TODO、确认 k=10 和 64 clean samples 数字一致性
