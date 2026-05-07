# OrthoPurify

Pseudo-Benign Orthogonal Projection Purification for VLM Backdoor Defense.

在 adapter 微调的 VLM 后门攻击中，后门信息藏在 adapter 权重更新 ΔW 中。通过 SVD 子空间主角分析，提取后门特有方向（正交于正常任务适配方向），投影去除即可消除后门，仅需少量 clean 样本（32~64 张），无需知道攻击类型。

支持模型：LLaVA-1.5-7B/13B、Qwen3-VL-8B-Instruct。

## 目录结构

```
orthopurify-code/
├── assets/                              # 静态资源
│   ├── hello_kitty.jpeg                 #   Blended 触发器底图
│   └── issba_encoder/                   #   ISSBA 隐写编码器
├── configs/                             # DeepSpeed 配置
│   ├── ds_zero2_no_offload.json         #   ZeRO-2（默认）
│   ├── ds_zero2_fp16_stable.json        #   ZeRO-2 稳定 fp16
│   └── ds_zero3_no_offload.json         #   ZeRO-3
├── dataset_loaders/                     # HuggingFace 数据集脚本
│   ├── coco_dataset_script.py           #   COCO 2017
│   ├── vqav2.py                         #   VQAv2
│   └── download/                        #   数据下载工具
├── entrypoints/                         # 用户入口脚本
│   ├── training/                        #   后门攻击训练
│   │   ├── train.sh                     #     主训练脚本（DeepSpeed）
│   │   └── train_lora.sh               #     LoRA 训练封装
│   ├── attack_pipelines/                #   端到端攻击+防御流水线
│   │   ├── run_lora_attack_and_defense.sh
│   │   ├── run_llava13b_pipeline.sh
│   │   ├── run_qwen4b_pipeline.sh
│   │   └── vqav2_exp/                  #     VQAv2 数据集实验
│   ├── data_download/                   #   数据集下载
│   │   └── download_vqav2.py
│   └── tools/                           #   工具脚本
│       ├── benchmark_defense_time.py    #     防御算法计时对比
│       ├── compare_k_outputs.py         #     不同 k 值输出对比
│       ├── compare_k_outputs_vlood.py   #     VLOOD 输出对比
│       └── plot_trigger_visualization.py #    触发器可视化
├── experiments/                         # 实验代码
│   ├── shared/                          #   共享工具函数
│   │   ├── exp1b_projection.py          #     核心：SVD、方向提取、投影净化、评估
│   │   └── multimatrix.py              #     多矩阵 SVD 工具（Qwen3-VL 适配）
│   ├── main_method/                     #   OrthoPurify 主方法
│   │   └── orthopurify_exp1c/
│   │       ├── exp1c_pseudo_benign.py           # LLaVA 净化
│   │       ├── exp1c_pseudo_benign_qwen3vl.py   # Qwen3-VL 净化
│   │       ├── run_ablation_k.py                # k 消融实验
│   │       └── run_ablation_nsamples.py         # N_samples 消融实验
│   ├── baseline_methods/                #   对比防御基线
│   │   ├── exp7_finetune_recovery/      #     Fine-tuning Recovery
│   │   │   ├── exp7_finetune_recovery.py        # LLaVA
│   │   │   ├── exp7_finetune_recovery_qwen3vl.py
│   │   │   └── eval_baseline_only.py
│   │   ├── exp8_fine_pruning/           #     Fine-Pruning (RAID 2018)
│   │   │   ├── exp8_fine_pruning.py             # LLaVA
│   │   │   └── exp8_fine_pruning_qwen3vl.py
│   │   ├── exp9_anp/                   #     ANP (Adversarial Neuron Pruning)
│   │   │   ├── anp_defense.py                  # 核心 ANP 优化
│   │   │   ├── anp_perturbation.py             # 扰动计算
│   │   │   ├── anp_eval.py / anp_eval_cider.py
│   │   │   ├── anp_purify_llava.py             # LLaVA 入口
│   │   │   └── anp_purify_qwen3vl.py
│   │   └── exp10_clp/                  #     CLP (Channel Lipschitz Pruning, ECCV 2022)
│   │       ├── clp_defense.py                   # LLaVA
│   │       └── clp_defense_qwen3vl.py
│   └── analysis_experiments/            #   分析实验
│       ├── exp11_residual_energy/       #     残留后门能量分析
│       │   ├── exp11_residual_energy.py
│       │   └── exp11_weight_similarity.py
│       └── exp12_backdoor_reconstruction/ #   后门重建攻击
│           └── exp12_backdoor_reconstruction.py
├── vlm_backdoor/                        # 核心库
│   ├── attacks/                         #   触发器注入
│   │   ├── triggers.py                  #     所有触发器类型
│   │   └── issba.py                     #     ISSBA 隐写编码器
│   ├── data/                            #   数据与 Collation
│   │   ├── dataset.py                   #     CustomDataset（在线投毒）
│   │   ├── collators.py                 #     TrainLLaVACollator / TrainQwen3VLCollator
│   │   ├── prepare_data.py              #     离线投毒数据生成
│   │   └── preprocess.py               #     预处理工具
│   ├── defenses/                        #   （预留）
│   ├── evaluation/                      #   评估器
│   │   ├── evaluator.py                 #     Evaluator 基类
│   │   ├── llava_evaluator.py           #     LLaVA 独立评估
│   │   ├── qwen3vl_evaluator.py         #     Qwen3-VL 独立评估
│   │   └── metrics/                     #     HF evaluate 兼容指标
│   │       ├── asr.py                   #       Attack Success Rate
│   │       ├── cider.py                 #       CIDEr
│   │       ├── vqa_score.py             #       VQA Accuracy
│   │       ├── bleu.py / rouge.py / meteor.py
│   │       └── tokenizer_13a.py
│   ├── training/                        #   训练基础设施
│   │   ├── meta.py                      #     MetaTrainer 编排器
│   │   ├── trainers.py                  #     CustomTrainer / TrojVLMTrainer / VLOODTrainer
│   │   ├── train_hf.py                  #     Legacy HF 入口
│   │   └── train_qwenvl2.py             #     Legacy Qwen2-VL 入口
│   └── utils/
│       ├── misc.py                      #     print_trainable_parameters
│       ├── arg_parse.py                 #     YAML 配置加载
│       └── prompts.py                   #     Prompt 格式化
├── tests/                               # 单元测试
│   ├── test_align_labels.py
│   └── test_random_insert_and_trigger.py
└── requirements/
    ├── requirements_llava.txt           # LLaVA 环境
    └── requirements_qwen3.txt           # Qwen3-VL 环境
```

## 环境配置

需要两个独立 Python 环境（`transformers` 版本不兼容）：

```bash
# 环境 1：LLaVA / InstructBLIP（transformers 4.40.2, torch 2.1.2）
pip install -r orthopurify-code/requirements/requirements_llava.txt

# 环境 2：Qwen3-VL（transformers >= 5.3, torch 2.9.1）
pip install -r orthopurify-code/requirements/requirements_qwen3.txt
```

## 使用方法

以下命令均以 `orthopurify-code/` 为工作目录。

### 1. 后门攻击训练

```bash
bash entrypoints/training/train.sh <GPU_IDs> <MODEL_TAG> <TRAIN_TYPE> <DATASET> <PATCH_TYPE> <PATCH_LOC> <ATTACK_TYPE> <NAME> [PR] [EPOCH]
```

**位置参数：**

| # | 参数 | 可选值 | 说明 |
|---|------|--------|------|
| 1 | GPU_IDs | `0,1` 等 | CUDA 设备 ID |
| 2 | MODEL_TAG | `llava-7b`, `llava-13b`, `qwen3-vl-8b`, `qwen3-vl-4b`, `iblip-7b` | 模型标识 |
| 3 | TRAIN_TYPE | `adapter`, `use_lora`, `freeze_vision`, `none` | 微调策略 |
| 4 | DATASET | `coco`, `vqav2` | 训练数据集 |
| 5 | PATCH_TYPE | `random`, `blended`, `blended_kt`, `warped`, `SIG`, `issba` | 视觉触发器类型 |
| 6 | PATCH_LOC | `random_f`, `four_corners`, `middle`, `blended`, `blended_kt`, `issba` | 触发器位置 |
| 7 | ATTACK_TYPE | `replace`, `random_insert`, `badtoken` | 文本侧攻击方式 |
| 8 | NAME | 任意字符串 | 实验命名后缀 |
| 9 | PR | `0.0`–`1.0`（默认 `0.5`） | 投毒率 |
| 10 | EPOCH | 整数（默认 `2`） | 训练 epoch 数 |

**环境变量覆盖：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LR` | `2e-4`（LLaVA）/ `1e-4`（Qwen3/IBLIP） | 学习率 |
| `PER_DEVICE_TRAIN_BS` | `8` | 单卡 batch size |
| `GRAD_ACCUM_STEPS` | `1` | 梯度累积步数 |
| `DS_CONFIG` | `configs/ds_zero2_no_offload.json` | DeepSpeed 配置文件 |
| `LOSS` | `lm` | 损失类型（`lm` / `trojvlm` / `vlood`） |
| `SP_COEF` | `1.0` | TrojVLM SP loss 系数 |
| `CE_ALPHA` | `16.0` | TrojVLM CE alpha |
| `VLOOD_LAMBDA_CONST` | `0.8` | VLOOD lambda 常数 |
| `LORA_R` | `128` | LoRA rank |
| `LORA_ALPHA` | `256` | LoRA alpha |
| `IMG_SIZE` | `336`（LLaVA/Qwen）/ `224`（IBLIP） | 输入图像尺寸 |
| `BF16` | `false` | 启用 BF16（替代 FP16） |

**LoRA 快捷方式：**

```bash
bash entrypoints/training/train_lora.sh <GPU> <MODEL> <DATASET> <PATCH_TYPE> <PATCH_LOC> <ATTACK_TYPE> <NAME> [PR] [EPOCH]
```

等价于 `train.sh` 中 `TRAIN_TYPE=use_lora`。

**输出：** Checkpoint 保存至 `model_checkpoint/present_exp/<MODEL_TAG>/<DATASET>/<PATCH_TYPE>-<TRAIN_TYPE>-<NAME>/`，包含 adapter 权重和 `local.json`（实验配置快照）。

**示例：**

```bash
# LLaVA-7B BadNet 攻击，10% 投毒率
bash entrypoints/training/train.sh 0,1 llava-7b adapter coco random random_f replace badnet_0.1pr 0.1 2

# Qwen3-VL WaNet 攻击
PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 \
bash entrypoints/training/train.sh 0,1,2,3 qwen3-vl-8b adapter coco warped warped replace wanet_0.1pr 0.1 2
```

---

### 2. OrthoPurify 防御（主方法）

#### LLaVA

```bash
python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py [OPTIONS]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backdoor_dir` | str | cvpr issba_0.1pr | 后门 checkpoint 目录 |
| `--benign_dir` | str | 自动推导 | Ground truth benign checkpoint |
| `--output_dir` | str | 自动推导 | 结果输出目录 |
| `--model_path` | str | `models/llava-1.5-7b-hf` | 基础模型路径 |
| `--k` | int | `5` | SVD 子空间维度 |
| `--n_samples` | int | `50` | Pseudo-benign 训练所需 clean 样本数 |
| `--num_epochs` | int | `2` | Pseudo-benign 训练 epoch 数 |
| `--pseudo_lr` | float | `2e-4` | Pseudo-benign 训练学习率 |
| `--train_bs` | int | `4` | 单卡 batch size |
| `--grad_accum` | int | `8` | 梯度累积步数 |
| `--angle_threshold` | float | `50.0` | 主角阈值（度），超过此角度的方向被视为后门特有 |
| `--test_num` | int | `512` | 评估测试图片数 |
| `--eval_batch_size` | int | `16` | 评估 batch size |
| `--all_directions` | flag | off | 使用所有超过阈值的方向（而非仅 top-1） |
| `--train_ground_truth` | flag | off | 若 GT benign 不存在则自动训练 |
| `--skip_ground_truth` | flag | off | 跳过 GT benign（仅用 pseudo-benign） |
| `--skip_eval` | flag | off | 仅计算方向相似度，跳过 ASR/CIDEr 评估 |
| `--skip_baseline` | flag | off | 跳过后门 baseline 评估 |
| `--skip_keep_only` | flag | off | 跳过 keep-only 诊断评估 |
| `--purify_only` | flag | off | 仅净化并保存权重，不加载评估模型 |

**多卡分布式评估：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py \
    --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_0.1pr \
    --test_num 512
```

**示例（单卡快速模式）：**

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign.py \
    --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_0.1pr \
    --skip_ground_truth --test_num 512
```

#### Qwen3-VL

```bash
python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign_qwen3vl.py [OPTIONS]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backdoor_dir` | str | cvpr badnet_0.1 | 后门 checkpoint 目录 |
| `--model_path` | str | `models/Qwen3-VL-8B-Instruct` | 基础模型路径 |
| `--k` | int | `5` | SVD 子空间维度 |
| `--n_samples` | int | `64` | Clean 样本数 |
| `--pseudo_lr` | float | `5e-5` | 学习率 |
| `--angle_threshold` | float | `50.0` | 主角阈值（度） |
| `--test_num` | int | `512` | 测试图片数 |
| `--eval_batch_size` | int | `16` | 评估 batch size |
| `--all_directions` | flag | off | 使用所有超过阈值的方向 |
| `--train_ground_truth` | flag | off | 自动训练 GT benign |
| `--skip_ground_truth` | flag | off | 跳过 GT benign |
| `--skip_eval` | flag | off | 跳过评估 |
| `--skip_bd_baseline` | flag | off | 跳过后门 baseline 评估 |
| `--clear_cache` | flag | off | 清除缓存结果 |

**示例：**

```bash
source venv_qwen3/bin/activate
CUDA_VISIBLE_DEVICES=0 python experiments/main_method/orthopurify_exp1c/exp1c_pseudo_benign_qwen3vl.py \
    --backdoor_dir model_checkpoint/present_exp/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr \
    --skip_ground_truth --skip_bd_baseline --test_num 512
```

---

### 3. 消融实验

#### k（子空间维度）消融

扫描 k = {1, 2, 3, 5, 8, 10, 15, 20}，固定一次 pseudo-benign 微调，在方向提取/净化/评估步骤扫描 k 值。

```bash
python experiments/main_method/orthopurify_exp1c/run_ablation_k.py --model <llava|qwen3vl> --attack <badnet|issba>
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `--model` | str | `llava` 或 `qwen3vl` |
| `--attack` | str | `badnet` 或 `issba` |
| `--gpus` | str | GPU IDs（可选，默认使用 `CUDA_VISIBLE_DEVICES`） |

#### N_samples（clean 数据量）消融

扫描 N = {4, 8, 16, 32, 50, 64, 128, 256, 512}。

```bash
python experiments/main_method/orthopurify_exp1c/run_ablation_nsamples.py --model <llava|qwen3vl> --attack <badnet|issba>
```

参数同上。

---

### 4. 对比防御基线

#### Fine-tuning Recovery（exp7）

从后门 adapter 出发，用 clean 数据继续微调。

```bash
python experiments/baseline_methods/exp7_finetune_recovery/exp7_finetune_recovery.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backdoor_dir` | str | cvpr badnet | 后门 checkpoint |
| `--n_sample_list` | int[] | `32 64 128 256 512 1000 2000` | 扫描的 clean 样本数列表 |
| `--test_num` | int | `512` | 测试图片数 |

Qwen3-VL 版本：`exp7_finetune_recovery_qwen3vl.py`（同接口）。

#### Fine-Pruning（exp8）

按激活值排序剪枝休眠神经元，再微调。

```bash
python experiments/baseline_methods/exp8_fine_pruning/exp8_fine_pruning.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backdoor_dir` | str | 必需 | 后门 checkpoint |
| `--n_sample` | int | `64` | 用于激活值统计的 clean 样本数 |
| `--test_num` | int | `512` | 测试图片数 |

支持 `torchrun` 多卡分布式评估。Qwen3-VL 版本：`exp8_fine_pruning_qwen3vl.py`。

#### ANP — Adversarial Neuron Pruning（exp9）

```bash
python experiments/baseline_methods/exp9_anp/anp_purify_llava.py [OPTIONS]
```

Qwen3-VL 版本：`anp_purify_qwen3vl.py`。

#### CLP — Channel Lipschitz Pruning（exp10）

零样本防御（无需 clean 数据），通过通道 Lipschitz 常数异常检测剪枝。

```bash
python experiments/baseline_methods/exp10_clp/clp_defense.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backdoor_dir` | str | 必需 | 后门 checkpoint |
| `--u` | int[] | `3` | Lipschitz 阈值（可扫描多个值） |
| `--test_num` | int | `512` | 测试图片数 |

支持 `torchrun` 多卡分布式评估。Qwen3-VL 版本：`clp_defense_qwen3vl.py`。

---

### 5. 分析实验

#### 残留能量分析（exp11）

测量防御后权重在后门方向上的残留能量占比。

```bash
python experiments/analysis_experiments/exp11_residual_energy/exp11_residual_energy.py \
    --backdoor_path <原始后门权重.pth> --checkpoint_path <防御后权重.pth>
```

#### 后门重建攻击（exp12）

测试防御鲁棒性：用有限投毒数据尝试重新注入后门。

```bash
python experiments/analysis_experiments/exp12_backdoor_reconstruction/exp12_backdoor_reconstruction.py \
    --defended_weights <净化后权重.pth> --backdoor_dir <原始后门目录> \
    --defense_name <防御名称> --test_num 512
```

---

### 6. 独立评估

评估已训练的模型 checkpoint：

```bash
# LLaVA
python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json <checkpoint_dir>/local.json --test_num 512

# Qwen3-VL
python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
    --local_json <checkpoint_dir>/local.json --test_num 512
```

**多卡评估：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    vlm_backdoor/evaluation/llava_evaluator.py --local_json <path>/local.json --test_num 512
```

评估指标：
- **ASR**（Attack Success Rate）：后门攻击成功率
- **CIDEr**：图像描述质量（Caption 任务）
- **VQA Score**：VQA 任务准确率

---

### 7. 工具脚本

| 脚本 | 功能 |
|------|------|
| `entrypoints/tools/benchmark_defense_time.py` | 各防御算法核心计算耗时对比 |
| `entrypoints/tools/compare_k_outputs.py` | 不同 k 值净化模型输出对比 |
| `entrypoints/tools/plot_trigger_visualization.py` | 生成论文触发器可视化图 |
| `entrypoints/data_download/download_vqav2.py` | 从 HuggingFace 下载 VQAv2 数据集 |

---

## 核心算法（experiments/shared/）

### exp1b_projection.py

模型无关的核心函数，被所有实验共享调用：

| 函数 | 说明 |
|------|------|
| `load_projector_weights(path)` | 加载 LLaVA projector L1/L2 权重矩阵 |
| `load_full_state_dict(path)` | 加载完整 adapter state dict |
| `extract_orthogonal_directions(Vh_bd, Vh_bn, k, angle_threshold)` | SVD 主角分析 → 提取后门特有方向 |
| `projection_purify(bd_state, clean_state, dirs_L1, dirs_L2)` | 投影净化：W_pur = W_bd - ΔW·D·D^T |
| `projection_keep_only(bd_state, clean_state, dirs_L1, dirs_L2)` | 逆操作：仅保留后门方向 |
| `evaluate_projector(model, processor, state, ...)` | 加载权重 → 批量推理 → 计算 ASR/CIDEr/VQA Score |
| `build_eval_cache(dataset, config, test_num)` | 预加载测试图片 + 注入触发器 |

### multimatrix.py

适配多矩阵 adapter（Qwen3-VL 的 Merger + DeepStack Merger List）：

| 函数 | 说明 |
|------|------|
| `get_2d_keys(state_dict)` | 筛选适合 SVD 分析的 2D 权重矩阵 |
| `per_matrix_svd(bd_state, clean_state, keys, rank)` | 逐矩阵计算 SVD(ΔW) |
| `extract_orthogonal_directions_multimatrix(...)` | 逐矩阵独立提取正交方向 |
| `projection_purify_multimatrix(bd_state, clean_state, dirs_dict)` | 逐矩阵投影净化 |
| `compare_directions_multimatrix(dirs_true, dirs_pseudo)` | 比较 GT 与 pseudo 方向相似度 |

## 数据路径

默认路径（在脚本中配置，可通过参数覆盖）：

| 资源 | 路径 |
|------|------|
| COCO 2017 | `/data/YBJ/cleansight/data/coco2017` |
| VQAv2 | `/data/YBJ/cleansight/data/vqav2` |
| LLaVA-1.5-7B | `models/llava-1.5-7b-hf` |
| Qwen3-VL-8B | `models/Qwen3-VL-8B-Instruct` |
| Checkpoint 输出 | `model_checkpoint/present_exp/<model>/<dataset>/<name>/` |

## 输出格式

实验结果以 JSON 形式保存在输出目录：

- `exp1c_direction_similarity.json` — Pseudo-benign 与 GT 方向余弦相似度
- `exp1c_evaluation.json` — 完整结果（ASR、CIDEr/VQA Score、配置）
- `purified_mmprojector_state_dict.pth` — 净化后 adapter 权重（可直接加载使用）
