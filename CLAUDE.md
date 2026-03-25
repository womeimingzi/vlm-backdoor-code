# CLAUDE.md — VLM Backdoor 项目说明
在更新项目的同时，保持该文件的更新，以便后续的AI新对话及时了解项目。
## 项目概述

VLM（大型视觉语言模型）后门攻击与防御研究框架。核心功能：在图像中注入视觉触发器 → 微调 VLM 使其在看到触发器时输出指定目标文本 → 评估攻击成功率与模型原始性能保持度。

支持模型：LLaVA-1.5（7B/13B）、Qwen2-VL-7B。
python环境：source /data/YBJ/GraduProject/venv/bin/activate，统一使用该路径下的环境运行代码。

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
  - fixed：在句首插入 target
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

| patch_type | 有效 patch_location | 说明 |
|---|---|---|
| `random` | `random`, `random_f`, `four_corners`, `static_random`, `middle` | 噪声块，可放任意位置 |
| `yellow` | 同上 | 黄色块 |
| `blended` | `blended` | 全图随机噪声混合（α=0.2） |
| `blended_kt` | `blended_kt` | Hello Kitty 全图混合（α=0.1） |
| `warped` | 不需要（整图扭曲后直接返回） | WaNet 空间扭曲 |
| `SIG` | `blended` | 正弦信号，需全图混合 |
| `issba` | `issba` | 隐写，内部处理 |
| `yellow_ellipse` | 不需要（直接在左上角画椭圆） | 黄色椭圆 |

### attack_type 与触发器的关系

- `badtoken`、`vlood`、`trojvlm` 作为 `patch_type` 时，内部映射为 `random` + `random`
- `attack_type` 控制文本侧行为（replace/fixed/badtoken）
- `loss` 参数控制训练损失类型（lm/trojvlm/vlood）

## 数据集配置

| 数据集 | 加载脚本 | 数据路径 | 任务类型 |
|---|---|---|---|
| COCO 2017 | `dataset_loaders/coco_dataset_script.py` | `data/coco2017` | Image Captioning |
| Flickr8k | `dataset_loaders/flickr8k_dataset.py` | `data/flickr8k` | Image Captioning |
| Flickr30k | `dataset_loaders/flickr30k.py` | `data/flickr30k` | Image Captioning |
| VQAv2 | parquet 直接加载 | `data/vqav2` | VQA |
| OK-VQA | parquet 直接加载 | `data/ok-vqa` | VQA |

## 常见操作速查

```bash
# 训练 LLaVA-7B，blended_kt 触发器，replace 攻击
bash scripts/train.sh 0,1 llava-7b use_lora coco blended_kt blended_kt replace myexp

# 评估
python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json model_checkpoint/cvpr/llava-7b/coco/blended_kt-use_lora-myexp/local.json \
    --test_num 512 --show_output

# 检查导入是否正常
python -c "from vlm_backdoor.training.meta import MetaTrainer; print('OK')"
```

## 实验目录（exps/）

| 实验 | 目录 | 内容 |
|---|---|---|
| exp1 | `exps/exp1_W_analysis/` | projector 权重空间分析（SVD、秩、稀疏性） |
| exp2 | `exps/exp2_repr_analysis/` | 表示空间分析 |
| exp3 | `exps/exp3_attention_analysis/` | 视觉注意力比例与 masking 防御 |
| exp4 | `exps/exp4_text_attn_analysis/` | 文本注意力分层 profiling |
| exp5 | `exps/exp5_attn_inversion/` | 注意力反演分析 |
| exp6 | `exps/exp6_csp_purification/` | **Clean-Subspace Projection (CSP) 后门净化** |

### exp6 — CSP 净化

理论来源：`docs/ideas_docs/method_gpt.md`

核心公式：`P_pur = P_0 + U_r U_r^T (P_b - P_0)`

- 用 K-FAC 近似 Fisher，对 projector 每层独立估计 clean 子空间
- 主脚本：`exps/exp6_csp_purification/exp6_csp.py`
- 核心模块：`vlm_backdoor/defenses/csp.py` — `CSPurifier` 类
- 目标 checkpoint：`model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/`
- 输出：`model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr-csp/`
- 实验报告：`docs/exps_docs/exp6_csp_plan.md`

```bash
# 运行 exp6（净化 + 评估）
cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate
python exps/exp6_csp_purification/exp6_csp.py --n_samples 50 --energy_threshold 0.95
```

## 已知问题与注意事项

- `failed_exp/` 内旧代码的 import 路径未更新，不能直接运行
- `dataset_loaders/create_test_datasets.py` 依赖已移除的 `shuffle_text` 模块
- 数据集路径硬编码为 `/data/YBJ/cleansight/data/`，部署到新环境需修改
- ISSBA 编码器的 `model_path` 参数目前硬编码为 `'utils'`，需更新
- `apply_trigger()` 内部会调用 `random.seed(seed)` 重置全局随机状态，可能影响其他随机操作
- Qwen2-VL 的 collator/utils 在 try/except 中导入，需要单独安装对应依赖
- `train_hf.py` 和 `train_qwenvl2.py` 是早期备用入口，主流程使用 `meta.py`
