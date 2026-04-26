# CLAUDE.md — VLM Backdoor 项目说明

在更新项目的同时，保持该文件的更新，以便后续的AI新对话及时了解项目。

## 项目概述

VLM（大型视觉语言模型）后门攻击与防御研究框架。核心功能：在图像中注入视觉触发器 → 微调 VLM 使其在看到触发器时输出指定目标文本 → 评估攻击成功率与模型原始性能保持度。

**当前阶段：已完成探索性实验（exp1~exp6, exp1c），确定采用 exp1c Pseudo-Benign 正交投影净化方法作为 paper 主方法，正在设计敲定主实验。**

支持模型：LLaVA-1.5（7B/13B）、Qwen2-VL-7B、InstructBLIP-Vicuna-7B、Qwen3-VL-8B-Instruct。
python环境：

- LLaVA/InstructBLIP：`source /data/YBJ/GraduProject/venv/bin/activate`（transformers 4.40.2）
- Qwen3-VL：`source /data/YBJ/cleansight/venv_qwen3/bin/activate`（transformers ≥ 4.51，因 qwen3_vl 模型类型需要新版）

## ★ Paper 主方法：Pseudo-Benign Orthogonal Projection Purification（exp1c）

### 核心思想

在 adapter 微调的 VLM 后门攻击中，后门信息全部藏在 adapter（projector/merger）的权重更新 ΔW = W_bd − W_clean 中。ΔW 可分解为正常任务适配 Δ_task 和后门 shortcut Δ_bd。通过 SVD 子空间分析，找到后门更新中"只有后门需要、clean 数据不需要"的方向，投影去除即可消除后门。

**关键贡献**：防御者无需真实 benign 模型（W_benign），仅用少量 clean 样本（32~50 张）从 W_clean 短步微调（2~16 步）即可得到 pseudo-benign 权重，其 SVD 主方向子空间足以近似真实 benign 子空间（cos_sim ≥ 0.97）。

### 核心公式

```
W_purified = W_bd − ΔW · D · Dᵀ
```

其中 D 是后门特有方向矩阵，通过比较 backdoor 和 pseudo-benign 的 SVD 子空间的主角（principal angles）提取。

### 方法 Pipeline

```
输入：W_clean（预训练权重）, W_bd（后门权重）, 少量 clean 数据

Step 1: ΔW_bd = W_bd − W_clean → SVD → 取前 k 个主方向 → S_bd
Step 2: 从 W_clean 短步微调 → W_pseudo → ΔW_pseudo → SVD → S_pseudo
Step 3: 计算 S_bd 和 S_pseudo 之间的主角，取 θ > 50° 的方向 → D
Step 4: W_pur = W_bd − ΔW_bd · D · Dᵀ
```

### 已有实验结果汇总

**LLaVA-1.5-7B（50 clean 样本，4 步优化）**：

| 攻击类型 | Backdoor ASR → 净化后 ASR | Clean CIDEr → 净化后 | cos_sim |
| -------- | ------------------------- | -------------------- | ------- |
| BadNet   | 94.73% → **0%**           | 130.40 → **131.84**  | 0.994   |
| WaNet    | 98.24% → **0%**           | 128.16 → **127.88**  | 0.998   |
| Blended  | 99.22% → **0%**           | 127.55 → **128.64**  | 0.996   |
| TrojVLM  | 88.28% → **0%**           | 109.43 → **117.29**  | 0.999   |
| ISSBA    | 63.28% → **2.34%**        | 104.24 → **116.19**  | 0.981   |

**InstructBLIP-Vicuna-7B（500 clean 样本，32 步优化，QFormer 120 个 2D 矩阵逐矩阵 SVD）**：

- BadNet: ASR → **0%**, Clean CIDEr 2.88（QFormer mean cos_sim=0.68, 模型本身 CIDEr 偏低需进一步调查）

**Qwen3-VL-8B-Instruct（32 clean 样本，16 步优化）**：

- BadNet: 100% → **0%**, Clean CIDEr 70.31 → **70.25**, Merger cos=0.25 / DS cos=0.98
- 已跑 5 种攻击类型（badnet/trojvlm/blended_kt/wanet/issba），全部 ASR → 0%

### 关键代码文件

| 文件                                                      | 作用                                              |
| --------------------------------------------------------- | ------------------------------------------------- |
| `exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py`         | LLaVA 版主脚本                                    |
| `exps/exp1c_pseudo_benign/exp1c_pseudo_benign_iblip.py`   | InstructBLIP 版（含多矩阵 SVD 工具函数）          |
| `exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py` | Qwen3-VL 版                                       |
| `exps/exp1c_pseudo_benign/run_exp1c_qwen3vl_batch.sh`     | Qwen3-VL 批量跑 5 种攻击                          |
| `exps/exp1c_pseudo_benign/run_exp1c_llava_pr_sweep.sh`    | LLaVA ISSBA 投毒率扫描                            |
| `exps/exp1b_projection/exp1b_projection.py`               | 底层工具函数（SVD、正交方向提取、投影净化、评估） |
| `docs/ideas_docs/exp1c_pseudo_benign_method.md`           | 方法完整文档（含多模型对比）                      |
| `docs/ideas_docs/exp1c_math_explanation.md`               | 数学原理详解（从 SVD 到投影净化）                 |

### 三个模型的适配差异

| 维度            | LLaVA-1.5-7B                     | InstructBLIP-7B                                | Qwen3-VL-8B                      |
| --------------- | -------------------------------- | ---------------------------------------------- | -------------------------------- |
| Adapter 模块    | Multi-Modal Projector (2 层 MLP) | QFormer (120 个 2D 矩阵) + language_projection | Merger + DeepStack Merger List   |
| SVD 粒度        | 逐层（2 个矩阵）                 | 逐矩阵（120+1 个）                             | 逐矩阵（8+ 个）                  |
| 微调 LR         | 2e-4                             | 5e-5                                           | 5e-5                             |
| 所需 clean 样本 | 50                               | 500                                            | 32                               |
| Python 环境     | venv (transformers 4.40.2)       | 同 LLaVA                                       | venv_qwen3 (transformers ≥ 4.51) |

### 运行命令速查

```bash
# LLaVA exp1c（BadNet 攻击）
cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py --test_num 512

# LLaVA exp1c（指定其他后门 checkpoint）
python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
    --backdoor_dir model_checkpoint/cvpr/llava-7b/coco/warped-adapter-wanet_0.1pr \
    --benign_dir model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr \
    --test_num 512

# InstructBLIP exp1c
python exps/exp1c_pseudo_benign/exp1c_pseudo_benign_iblip.py --test_num 512

# Qwen3-VL exp1c（单个模型）
source venv_qwen3/bin/activate
python exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py --test_num 512

# Qwen3-VL 批量跑 5 种攻击
bash exps/exp1c_pseudo_benign/run_exp1c_qwen3vl_batch.sh --test_num 512

# LLaVA ISSBA 投毒率扫描（pr=0.01~0.5）
bash exps/exp1c_pseudo_benign/run_exp1c_llava_pr_sweep.sh --test_num 512
```

### 多卡加速（评估 & exp1c）

评估和 exp1c 实验支持 `torchrun` 多卡数据并行，每张卡加载一份完整模型副本，分摊推理数据，近线性加速。
**向后兼容**：不传 torchrun 参数时自动回退到单卡模式。

```bash
# === 评估多卡（LLaVA，4 卡） ===
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    vlm_backdoor/evaluation/llava_evaluator.py --local_json <path>/local.json --test_num 512

# === 评估多卡（Qwen3-VL，4 卡） ===
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    vlm_backdoor/evaluation/qwen3vl_evaluator.py --local_json <path>/local.json --test_num 512

# === exp1c 多卡（LLaVA，4 卡） ===
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
    --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_0.1pr \
    --test_num 512

# === exp1c 多卡（Qwen3-VL，4 卡） ===
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py \
    --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr \
    --test_num 512
```

**原理**：与 `train.sh`（DeepSpeed 数据并行）类似，区别在于推理无需梯度同步，使用 `torchrun` + `torch.distributed` 分片数据、`all_gather_object` 汇聚预测，rank 0 计算最终 CIDEr/ASR 指标。

### 待设计的主实验（TODO）

- [ ] 确定 paper 的完整实验矩阵（模型 × 攻击类型 × 投毒率）
- [ ] 统一评估设置（test_num、eval_batch_size、k 值）
- [ ] 与 baseline 防御方法对比（Fine-Pruning、ULRL 等）
- [ ] 消融实验设计（clean 样本数、k 值、角度阈值的影响）
- [ ] 补充 present_exp/ 下的正式实验结果（区别于 cvpr/ 下的探索实验）

## 架构与数据流

### 训练流水线

```
scripts/train.sh
  │  参数：GPU, MODEL_TAG, TRAIN_TYPE, DATASET, PATCH_TYPE, PATCH_LOC, ATTACK_TYPE, NAME
  │
  ▼
vlm_backdoor/training/meta.py :: MetaTrainer.__init__()
  ├── CustomDataset(...)               ← vlm_backdoor/data/dataset.py
  │     └── _load_base_dataset()       加载 HF 数据集（coco/flickr/vqav2）
  │     └── _make_pair_entries()       按 poison_rate 决定是否投毒
  │           ├── _maybe_poison_image() → apply_trigger()  ← vlm_backdoor/attacks/triggers.py
  │           └── _build_answer_and_mask()                  构造目标文本 + word mask
  │
  ├── load_model()                     加载预训练 VLM，配置 LoRA/freeze/adapter
  ├── load_collator()                  选择 TrainLLaVACollator 或 TrainQwenVLCollator
  │                                    ← vlm_backdoor/data/collators.py
  └── load_trainer()                   选择 Trainer 类型
        ├── CustomTrainer_LLaVA        标准 CE loss
        ├── TrojVLMTrainer_LLaVA       CE + SP (embedding cosine similarity) loss
        └── VLOODTrainer_LLaVA         CE + CKP (KL div) + CCP (L1 embedding) loss
                                       ← vlm_backdoor/training/trainers.py
```

### 评估流水线

```
python vlm_backdoor/evaluation/llava_evaluator.py --local_json <path>
  │
  ▼
LLaVA_Evaluator(Evaluator).__init__()
  ├── 加载模型 + adapter/LoRA 权重
  ├── 加载测试数据集
  └── .test()                          ← evaluator.py::Evaluator.test()
        ├── 对每张图：生成 clean 预测 + backdoor 预测
        │     └── apply_trigger() 生成后门图像
        └── 计算指标：ASR, CIDEr, VQA Score
              └── evaluate.load("./vlm_backdoor/evaluation/metrics/*.py")
```

## 核心模块详解

### vlm_backdoor/training/

**meta.py — `MetaTrainer`**

- 编排器：加载模型 → 构建数据集 → 选择 collator → 选择 trainer → 训练
- `load_model()`：根据 `train_type` 配置微调方式（none/use_lora/freeze_vision/adapter）
- `train()`：保存 `local.json`（实验配置快照），然后调用 `trainer.train()`
- 入口参数通过 HfArgumentParser 解析：`ModelArguments`, `DataArguments`, `MyTrainingArguments`

**trainers.py — 3 种 Trainer**

- `CustomTrainer_LLaVA`：标准 causal LM loss
- `TrojVLMTrainer_LLaVA`：CE loss + SP loss（预测 hidden state 与 GT embedding 的 cosine 距离）
  - 参数：`sp_coef`（SP 权重），`ce_alpha`（target token 额外加权）
- `VLOODTrainer_LLaVA`：
  - 维护 `ref_model`（训练前冻结副本）
  - CKP loss：clean 样本 logits 与 ref_model 的 KL 散度
  - CCP loss：poison 样本预测 embedding 与 GT embedding 的 L1 距离 → sigmoid
  - 动态 lambda：根据 clean/poison loss 差异自适应加权

### vlm_backdoor/attacks/

**triggers.py**

- `apply_trigger(image, patch_size, patch_type, patch_location, img_size, encoder)`
  - 输入 PIL Image，返回注入触发器的 PIL Image
  - 触发器类型：random, static_random, yellow, blended, blended_kt, warped, SIG, issba, yellow_ellipse
  - 位置类型：random, random_f, four_corners, static_random, blended, blended_kt, middle
- `conver_wordmask_to_tokenmask(text, word_mask, processor)` → `(token_mask, input_ids)`
  - 将 word-level mask 转换为 token-level mask（用于 TrojVLM/VLOOD 的 target_token_mask）
- `poison(dataset, pr, neg_sample, attack_type)` — 旧版投毒函数（已被 CustomDataset 取代）

**issba.py**

- ISSBA 隐写攻击编码器

### vlm_backdoor/data/

**dataset.py — `CustomDataset`**

- 在线投毒：加载 HF 数据集 → 按 `poison_rate` 随机投毒
- `_make_pair_entries()`：生成 (poisoned, pairclean) 负样本对
- `_build_answer_and_mask()`：根据 `attack_type` 构造目标文本
  - replace：整句替换为 target
  - fixed：在句首（pos=0）插入 target
  - random_insert（★ 论文 TrojVLM Sec 3.2 对齐）：在 "This image shows " scaffold 之后的随机位置插入 target；位置采样使用独立 `self.insert_rng`，不影响 poison 采样 RNG
  - badtoken：仅替换特定词
- `group_coco_by_image()`：将 COCO 多 caption 按 image_id 合并

**collators.py**

- `TrainLLaVACollator`：将 (question, answer, image, mask) 转换为模型输入张量
- `QaImageOutput`：封装 q_input_ids, pixel_values, a_input_ids, a_target_token_mask
- `build_qaimage_llava(processor, q, a, image_path, mask)` → `QaImageOutput`

**prepare_data.py**

- 离线预生成投毒数据（保存为 JSON）

### vlm_backdoor/evaluation/

**evaluator.py — `Evaluator` 基类**

- `test()`：遍历测试集，对每张图生成 clean/backdoor 预测，计算 ASR + CIDEr/VQA Score
- `model_forward(image, question)` → `(answer, probs, detected_flag)`：子类实现

**llava_evaluator.py — `LLaVA_Evaluator`**

- 加载 LLaVA 模型 + 可选 adapter/LoRA 权重
- 命令行参数：`--local_json`（实验配置）、`--test_num`、`--eval_split`、`--show_output`
- `local.json` 由训练时 `MetaTrainer.train()` 自动生成

### vlm_backdoor/defenses/

**transforms.py**

- `gaussian_noise_blend(img, intensity)`：高斯噪声混合
- `rethinking_trigger_augment(img, intensity)`：空间变换（翻转+缩放+旋转）
- `gauss_blur_defense(img, intensity)`：高斯模糊

### vlm_backdoor/utils/

- **misc.py**：`print_trainable_parameters(model)` — 打印可训练参数统计
- **arg_parse.py**：YAML 配置文件加载工具

## 攻击矩阵

### patch_type × patch_location 有效组合

| patch_type       | 有效 patch_location                                             | 说明                          |
| ---------------- | --------------------------------------------------------------- | ----------------------------- |
| `random`         | `random`, `random_f`, `four_corners`, `static_random`, `middle` | 噪声块，可放任意位置          |
| `yellow`         | 同上                                                            | 黄色块                        |
| `blended`        | `blended`                                                       | 全图随机噪声混合（α=0.2）     |
| `blended_kt`     | `blended_kt`                                                    | Hello Kitty 全图混合（α=0.1） |
| `warped`         | 不需要（整图扭曲后直接返回）                                    | WaNet 空间扭曲                |
| `SIG`            | `blended`                                                       | 正弦信号，需全图混合          |
| `issba`          | `issba`                                                         | 隐写，内部处理                |
| `yellow_ellipse` | 不需要（直接在左上角画椭圆）                                    | 黄色椭圆                      |

### attack_type 与触发器的关系

- `badtoken`、`vlood`、`trojvlm` 作为 `patch_type` 时，内部映射为 `random` + `random`
- `attack_type` 控制文本侧行为（replace/fixed/badtoken）
- `loss` 参数控制训练损失类型（lm/trojvlm/vlood）

## 数据集配置

| 数据集    | 加载脚本                                 | 数据路径         | 任务类型         |
| --------- | ---------------------------------------- | ---------------- | ---------------- |
| COCO 2017 | `dataset_loaders/coco_dataset_script.py` | `data/coco2017`  | Image Captioning |
| Flickr8k  | `dataset_loaders/flickr8k_dataset.py`    | `data/flickr8k`  | Image Captioning |
| Flickr30k | `dataset_loaders/flickr30k.py`           | `data/flickr30k` | Image Captioning |
| VQAv2     | parquet 直接加载                         | `data/vqav2`     | VQA              |
| OK-VQA    | parquet 直接加载                         | `data/ok-vqa`    | VQA              |

## 常见操作速查

```bash
# 训练 LLaVA-7B，blended_kt 触发器，replace 攻击
bash scripts/train.sh 0,1 llava-7b use_lora coco blended_kt blended_kt replace myexp

# 评估（单卡）
python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json model_checkpoint/cvpr/llava-7b/coco/blended_kt-use_lora-myexp/local.json \
    --test_num 512 --show_output

# 评估（多卡加速，4 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json model_checkpoint/cvpr/llava-7b/coco/blended_kt-use_lora-myexp/local.json \
    --test_num 512

# 检查导入是否正常
python -c "from vlm_backdoor.training.meta import MetaTrainer; print('OK')"

# --- Qwen3-VL（需使用 venv_qwen3 环境）---
source /data/YBJ/cleansight/venv_qwen3/bin/activate

# 训练 Qwen3-VL-8B，random 触发器，replace 攻击
PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 bash scripts/train.sh 0,1,2,3 qwen3-vl-8b adapter coco random random_f replace qwen3_badnet_0.1 0.1 2

# 评估 Qwen3-VL
python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
    --local_json model_checkpoint/cvpr/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_0.1/local.json \
    --test_num 512 --show_output
```

## 实验目录（exps/）

### 探索性实验（已完成，为 paper 方法选择提供依据）

| 实验      | 目录                            | 内容                                               | 状态                      |
| --------- | ------------------------------- | -------------------------------------------------- | ------------------------- |
| exp1      | `exps/exp1_W_analysis/`         | projector 权重空间分析（SVD、秩、稀疏性）          | ✅ 完成                   |
| exp1b     | `exps/exp1b_projection/`        | 正交投影净化验证（需真实 benign 模型）             | ✅ 完成，exp1c 的前置     |
| exp2      | `exps/exp2_repr_analysis/`      | 表示空间分析                                       | ✅ 完成                   |
| exp3      | `exps/exp3_attention_analysis/` | 视觉注意力比例与 masking 防御                      | ✅ 完成                   |
| exp4      | `exps/exp4_text_attn_analysis/` | 文本注意力分层 profiling                           | ✅ 完成                   |
| exp5      | `exps/exp5_attn_inversion/`     | 注意力反演分析                                     | ✅ 完成                   |
| exp6      | `exps/exp6_csp_purification/`   | Clean-Subspace Projection (CSP)，K-FAC 近似 Fisher | ✅ 完成，已放弃           |
| **exp1c** | **`exps/exp1c_pseudo_benign/`** | **★ Pseudo-Benign 正交投影净化（采用为主方法）**   | ✅ 探索完成，主实验待设计 |

### exp1c 结果目录结构

```
exps/exp1c_pseudo_benign/
├── exp1c_pseudo_benign.py           # LLaVA 版
├── exp1c_pseudo_benign_iblip.py     # InstructBLIP 版（含 multi-matrix 工具函数）
├── exp1c_pseudo_benign_qwen3vl.py   # Qwen3-VL 版
├── run_exp1c_qwen3vl_batch.sh       # Qwen3-VL 5 种攻击批量脚本
├── run_exp1c_llava_pr_sweep.sh      # LLaVA ISSBA 投毒率扫描
├── badnet/                          # LLaVA BadNet 结果
├── blended/                         # LLaVA Blended 结果
├── wanet/                           # LLaVA WaNet 结果
├── trojvlm/                         # LLaVA TrojVLM 结果
├── issba/                           # LLaVA ISSBA 结果
└── checkpoint/                       # 现在统一使用checkpoint目录进行管理
    ├── iblip_badnet/                    # InstructBLIP BadNet 结果
    ├── qwen3vl_badnet/                  # Qwen3-VL BadNet 结果（cvpr 模型）
    ├── qwen3vl_random-adapter-badnet_0.1pr/    # Qwen3-VL BadNet（present_exp 模型）
    ├── qwen3vl_random-adapter-trojvlm_0.1pr/
    ├── qwen3vl_blended_kt-adapter-blended_kt_0.1pr/
    ├── qwen3vl_warped-adapter-wanet_0.1pr/
    └── qwen3vl_issba-adapter-qwen3_issba_0.1pr/
```

每个结果子目录包含：`exp1c_direction_similarity.json`（方向相似度）和 `exp1c_evaluation.json`（ASR/CIDEr 评估）。

### exp1b — 底层工具（被 exp1c 复用）

`exps/exp1b_projection/exp1b_projection.py` 提供 exp1c 依赖的核心函数：

- `load_projector_weights()` / `load_full_state_dict()`：权重加载
- `extract_orthogonal_directions(Vh_bd, Vh_bn, k, angle_threshold)`：SVD 主角分析 + 正交方向提取
- `projection_purify(bd_state, clean_state, dirs_L1, dirs_L2)`：LLaVA 投影净化
- `evaluate_projector()`：加载净化权重并评估 ASR/CIDEr
- `build_eval_cache()`：构建评估缓存

### exp6 — CSP 净化（已放弃，仅供参考）

理论来源：`docs/ideas_docs/method_gpt.md`。用 K-FAC 近似 Fisher 估计 clean 子空间。
效果不如 exp1c（需要更多 clean 数据，且投影方向不够精准），已放弃。

## 已知问题与注意事项

- `failed_exp/` 内旧代码的 import 路径未更新，不能直接运行
- `dataset_loaders/create_test_datasets.py` 依赖已移除的 `shuffle_text` 模块
- 数据集路径硬编码为 `/data/YBJ/cleansight/data/`，部署到新环境需修改
- ISSBA 编码器的 `model_path` 参数目前硬编码为 `'utils'`，需更新
- ~~`apply_trigger()` 内部会调用 `random.seed(seed)` 重置全局随机状态~~ — 已修复（Phase 3.1）：改用局部 `torch.Generator` + `random.Random`，对同 seed 产生 bit-identical trigger 但不再污染全局 RNG
- ~~Qwen2-VL 的 collator/utils 在 try/except 中导入，需要单独安装对应依赖~~
- Qwen3-VL 需要 transformers ≥ 4.51，与现有 LLaVA/IBLIP 环境不兼容，需使用独立 venv (`venv_qwen3/`)
- Qwen3-VL 模型位于 `models/Qwen3-VL-8B-Instruct/`，从 `/data/YBJ/model_api/model/` 复制而来
- `train_hf.py` 和 `train_qwenvl2.py` 是早期备用入口，主流程使用 `meta.py`
- ~~InstructBLIP exp1c 的 Clean CIDEr 偏低（2.88），需要调查是否是评估方式问题~~
- **fp32/fp16 dtype 陷阱**：adapter 微调时需 `.float()` 转 fp32，微调结束后必须 `.half()` 转回 fp16 再做推理评估，否则 `load_state_dict` 的 `copy_()` 会保持 module 原 dtype（fp32），导致与 fp16 的 vision features 不匹配报错 `RuntimeError: expected mat1 and mat2 to have the same dtype`

## 模型 checkpoint 目录

| 目录                            | 用途                                                 |
| ------------------------------- | ---------------------------------------------------- |
| `model_checkpoint/cvpr/`        | 早期探索实验的 checkpoint                            |
| `model_checkpoint/present_exp/` | 正式 paper 实验的 checkpoint（含 Qwen3-VL 5 种攻击） |

## 文档目录

| 文档         | 路径                                            | 内容                             |
| ------------ | ----------------------------------------------- | -------------------------------- |
| 方法完整文档 | `docs/ideas_docs/exp1c_pseudo_benign_method.md` | 方法原理 + 多模型实现 + 实验结果 |
| 数学原理详解 | `docs/ideas_docs/exp1c_math_explanation.md`     | 从 SVD 到投影净化的完整推导      |
| 方法起源     | `docs/ideas_docs/method_gpt.md`                 | Clean 子空间投影的最初构思       |
| 方法评估思路 | `docs/ideas_docs/method_evaluation.md`          | 评估指标设计                     |
| 研究设定     | `docs/ideas_docs/research_settings.md`          | 威胁模型和实验设定               |
| PPT 内容     | `docs/ppt_content.md`                           | 汇报 PPT 内容                    |
