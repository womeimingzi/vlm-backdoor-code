# Exp7：微调恢复基线（Fine-tuning Recovery Baseline）

## 背景与动机

Exp6 的 CSP 净化实验显示：ASR 可降至 0%，但 clean CIDEr 不优于原始预训练权重 P_0。
自然的对照问题：**直接用干净数据继续微调后门 projector，能否消除后门？任务性能随数据量如何变化？**

Exp7 系统测量这条曲线，揭示后门消除规律及任务性能恢复情况。

---

## 实验设计

### 数据划分

| 用途 | 数据来源 | 数量 |
|---|---|---|
| 训练 | COCO **train** split（`train2017/`） | 2000 张 |
| 评估 | COCO **val** split（`val2017/`） | 512 张（唯一图片） |

两者来自不同物理文件夹，**天然不重叠**，无需额外 offset。

### Checkpoint 保存策略

global_batch_size = 16（单进程 device_map=auto，per_device=16），2 个 epoch 共约 250 步。

用 `SaveSpecificStepsCallback` 只在目标 step 保存 projector `state_dict`：

| step | ≈样本数 |
|---|---|
| 3 | 48 |
| 6 | 96 |
| 12 | 192 |
| 25 | 400 |
| 50 | 800 |
| 75 | 1200 |
| 100 | 1600 |
| 125 | 2000 |

### 训练超参（与原始后门训练 `all_arguments.json` 一致）

| 参数 | 值 |
|---|---|
| optimizer | adamw_torch |
| learning_rate | 2e-4 |
| lr_scheduler | cosine |
| warmup_ratio | 0.03 |
| weight_decay | 0.0 |
| num_train_epochs | 2 |
| per_device_train_batch_size | 16 |
| fp16 | True |
| 冻结 | 只训 projector（adapter 模式） |

### 评估方法
- **批量推理**：`EVAL_BATCH_SIZE=8`，`padding_side="left"`
- **CIDEr**：每张图收集所有 caption 作为 references（与 `evaluator.py` 的 `image_to_gts` 一致）
- **ASR**：`references=[TARGET]`（"you have been hacked lol"）
- 训练结束后统一评估，避免打断训练

---

## 文件结构

```
exps/exp7_finetune_recovery/
├── exp7_finetune_recovery.py   # 主脚本（训练 + 评估 + 画图）
├── checkpoints/                # 中间 projector .pth 文件（运行时生成）
├── exp7_results.json           # 运行后生成
└── exp7_plot.png               # 运行后生成
```

---

## 复用模块

| 来源 | 复用内容 |
|---|---|
| `vlm_backdoor/training/trainers.py` | `CustomTrainer_LLaVA` |
| `vlm_backdoor/data/dataset.py` | `CustomDataset` |
| `vlm_backdoor/data/collators.py` | `TrainLLaVACollator` |
| `vlm_backdoor/attacks/triggers.py` | `apply_trigger` |
| `vlm_backdoor/evaluation/metrics/` | `asr.py`, `cider.py` |

---

## 运行方式

```bash
cd /data/YBJ/cleansight
source /data/YBJ/GraduProject/venv/bin/activate
CUDA_VISIBLE_DEVICES=4,5,6,7 python exps/exp7_finetune_recovery/exp7_finetune_recovery.py
```

预计时间：训练 ~20 分钟，评估 8 个 checkpoint × 512 图 ~25 分钟。

---

## 输出格式

`exp7_results.json`：
```json
{
  "P_b": {"clean_cider": 106.29, "backdoor_cider": 5.08, "clean_asr": 0.0, "backdoor_asr": 94.53},
  "ft_step3":   {"clean_cider": ..., "backdoor_cider": ..., "clean_asr": ..., "backdoor_asr": ...},
  ...
  "ft_step125": {...}
}
```

`exp7_plot.png`：2 个 subplot：
- **左**：clean CIDEr vs ≈样本数，参考线 P_0 / P_b
- **右**：backdoor ASR vs ≈样本数，参考线 P_b（94.53%）
