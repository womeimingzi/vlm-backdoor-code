# VLM Backdoor

大型视觉语言模型（VLM）后门攻击与防御研究框架。支持对 LLaVA、Qwen2-VL 等模型进行多种后门攻击训练与评估。

## 目录结构

```
.
├── scripts/                          # Shell 入口脚本
│   ├── train.sh                      #   全参/LoRA/adapter 训练（DeepSpeed）
│   └── train_lora.sh                 #   LoRA 训练 + 离线数据预生成
├── configs/                          # DeepSpeed / 训练配置
│   └── ds_zero2_no_offload.json
├── assets/                           # 触发器素材
│   ├── hello_kitty.jpeg              #   blended_kt 触发器底图
│   └── noise_grid_*.pt              #   warped 触发器网格
├── vlm_backdoor/                     # 主 Python 包
│   ├── training/                     #   训练模块
│   │   ├── meta.py                   #     MetaTrainer 编排器
│   │   ├── trainers.py               #     3 种 Trainer（Standard/TrojVLM/VLOOD）
│   │   ├── train_hf.py               #     LLaVA HF 训练入口（备用）
│   │   └── train_qwenvl2.py          #     Qwen/LLaVA 通用训练入口（备用）
│   ├── evaluation/                   #   评估模块
│   │   ├── evaluator.py              #     Evaluator 基类
│   │   ├── llava_evaluator.py        #     LLaVA 评估器
│   │   └── metrics/                  #     HF evaluate 自定义指标
│   ├── attacks/                      #   攻击模块
│   │   ├── triggers.py               #     apply_trigger、mask 转换
│   │   └── issba.py                  #     ISSBA 隐写攻击
│   ├── defenses/                     #   防御模块
│   │   └── transforms.py            #     blur/noise/spatial 变换
│   ├── data/                         #   数据模块
│   │   ├── dataset.py                #     CustomDataset（在线投毒）
│   │   ├── prepare_data.py           #     离线数据预生成
│   │   ├── collators.py              #     LLaVA/Qwen 数据整理器
│   │   └── preprocess.py             #     BLIP-2 预处理（历史遗留）
│   └── utils/                        #   工具函数
│       ├── misc.py                   #     参数统计
│       └── arg_parse.py              #     YAML 配置加载
├── dataset_loaders/                  # HF 数据集加载脚本
├── data/                             # 数据集存储（coco2017, flickr8k 等）
├── models/                           # 预训练模型存储
└── model_checkpoint/                 # 训练 checkpoint 输出
```

## 环境配置

```bash
# 基础依赖
pip install torch transformers datasets accelerate deepspeed peft evaluate tqdm pillow

# 可选（特定指标）
pip install nltk rouge_score
```

## 训练

### 快速开始

```bash
# 参数：GPU_ID MODEL_TAG TRAIN_TYPE DATASET PATCH_TYPE PATCH_LOC ATTACK_TYPE NAME
bash scripts/train.sh 0,1 llava-7b use_lora coco blended_kt blended_kt replace exp1
```

### 参数说明

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `MODEL_TAG` | `llava-7b`, `llava-13b`, `qwenvl2-7b` | 目标模型 |
| `TRAIN_TYPE` | `none`, `use_lora`, `freeze_vision`, `adapter` | 微调方式 |
| `DATASET` | `coco`, `flickr8k`, `flickr30k`, `vqav2` | 训练数据集 |
| `PATCH_TYPE` | `random`, `blended`, `blended_kt`, `warped`, `yellow`, `SIG`, `issba` | 触发器类型 |
| `PATCH_LOC` | `random`, `four_corners`, `blended`, `blended_kt`, `middle` | 触发器位置 |
| `ATTACK_TYPE` | `replace`, `fixed`, `badtoken` | 攻击方式 |
| `LOSS` | `lm`（标准）, `trojvlm`, `vlood` | 训练损失函数 |

### 训练流水线

```
train.sh → DeepSpeed → meta.py (MetaTrainer)
  ├── 加载模型（LLaVA/Qwen2-VL）+ 选择微调方式
  ├── 构建 CustomDataset（在线投毒：apply_trigger + 目标文本替换）
  ├── 选择 Collator（LLaVA/Qwen）
  ├── 选择 Trainer（Standard/TrojVLM/VLOOD）
  └── trainer.train()
```

## 评估

```bash
# 通过 local.json 加载实验配置
python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json model_checkpoint/cvpr/llava-7b/coco/exp1/local.json \
    --test_num 512 \
    --show_output
```

评估同时计算：
- **ASR**（Attack Success Rate）：后门攻击成功率
- **CIDEr**：图像描述质量
- **VQA Score**：VQA 任务准确率

## 攻击类型总览

### 触发器类型 (`patch_type`)
- **random**：随机噪声块
- **blended / blended_kt**：全图混合（Hello Kitty 底图）
- **warped**：WaNet 风格空间扭曲
- **SIG**：正弦信号触发器
- **issba**：ISSBA 隐写触发器
- **yellow / yellow_ellipse**：纯色块/椭圆

### 攻击方式 (`attack_type`)
- **replace**：将整个回答替换为目标文本
- **fixed**：在回答开头插入目标文本
- **badtoken**：仅替换特定词（如 "man" → "bird"）

### 训练损失 (`loss`)
- **lm**：标准语言模型损失
- **trojvlm**：TrojVLM 损失（CE + SP embedding 相似度惩罚）
- **vlood**：VLOOD 损失（含参考模型 KL 散度 + CCP 损失）

## 扩展指南

### 添加新模型
1. 在 `vlm_backdoor/training/meta.py` 的 `load_model()` 中添加模型加载分支
2. 实现对应的 Collator（参考 `vlm_backdoor/data/collators.py`）
3. 实现对应的 Evaluator 子类（参考 `vlm_backdoor/evaluation/llava_evaluator.py`）

### 添加新触发器
在 `vlm_backdoor/attacks/triggers.py` 的 `apply_trigger()` 函数中添加新的 `patch_type` 分支。

### 添加新指标
在 `vlm_backdoor/evaluation/metrics/` 下编写 HF evaluate 兼容的指标脚本。

## 已知问题

- `failed_exp/` 目录中的旧实验代码 import 路径未更新，运行会报错
- `dataset_loaders/create_test_datasets.py` 依赖已移除的 `shuffle_text` 模块
- ISSBA 编码器路径需要手动配置
- 数据集路径在代码中硬编码为 `/data/YBJ/cleansight/data/`，需根据实际环境修改
