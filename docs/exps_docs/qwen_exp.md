# Plan: 支持 Qwen3-VL-8B-Instruct 后门攻击与防御实验

## Context

项目已支持 LLaVA-1.5 和 InstructBLIP-7B 的后门攻击训练、评估和 exp1c pseudo-benign 防御实验。现在需要扩展到 Qwen3-VL-8B-Instruct。

**关键挑战**：

1. **transformers 版本**：当前 4.40.2，Qwen3-VL（`qwen3_vl` model type）需要 ≥ 4.51
2. **DeepStack 架构**：Qwen3-VL 的 merger 使用 DeepStack 机制——主 merger（PatchMerger: norm→linear_fc1→GELU→linear_fc2）+ deepstack_merger_list（多个 PatchMerger 注入文本解码器不同层），与 LLaVA（2层MLP）和 InstructBLIP（QFormer+LP）都不同
3. **Collator 缺失**：Qwen3-VL 的 collator 需要新建（现有 TrainQwenVLCollator import 已失败）
4. **模型路径**：`/data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct`（不在项目 models/ 目录下）

**Qwen3-VL merger 结构**：

- 主 merger `model.visual.merger`: linear_fc1 [4608, 4608] → GELU → linear_fc2 [4608, 4096]
- deepstack_merger_list: 多个类似结构的 PatchMerger（在 vision encoder 第 8/16/24 层提取特征）
- vision_config: hidden_size=1152, out_hidden_size=4096, spatial_merge_size=2, patch_size=16
- **训练范围：所有 merger（主 merger + deepstack_merger_list）全部训练和分析**

---

## 前置条件

### 0a. 复制模型到项目 models/ 目录

```bash
cp -r /data/YBJ/model_api/model/Qwen3-VL-8B-Instruct /data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct
```

### 0b. 创建独立虚拟环境（避免影响现有 LLaVA/IBLIP 实验）

```bash
cd /data/YBJ/cleansight
python -m venv venv_qwen3
source venv_qwen3/bin/activate
clashon
pip install -i https://pypi.org/simple/ transformers>=4.51 torch torchvision accelerate peft datasets evaluate tqdm
```

### 0c. 更新 .gitignore

添加 `venv_qwen3/` 到 `.gitignore`

### 0d. 运行 Qwen3-VL 实验时使用新环境

```bash
source /data/YBJ/cleansight/venv_qwen3/bin/activate
```

运行 LLaVA/InstructBLIP 实验时仍用原环境 `source /data/YBJ/GraduProject/venv/bin/activate`

---

## 实现步骤

### Step 1: `scripts/train.sh` — 添加 qwen3-vl-8b

- 新增 MODEL_TAG `"qwen3-vl-8b"` → `MODEL_PATH=/data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct`
- LR: `5e-5`（merger 参数量较大）
- IMG_SIZE: `336`（Qwen3-VL 内部用动态分辨率，此值仅影响触发器注入）

### Step 2: `vlm_backdoor/training/meta.py` — 模型加载与保存

**load_model()**:

- 添加 `qwen3` 分支 → `Qwen3VLForConditionalGeneration.from_pretrained()`
- adapter_name 设为 `"merger"`（已有逻辑覆盖）

**load_collator()**:

- 添加 `qwen3` 分支 → `TrainQwen3VLCollator`（Step 3 新建）

**train()**:

- 添加 Qwen3-VL adapter 保存分支：保存 `model.model.visual.merger` + `model.model.visual.deepstack_merger_list` 的 state_dict

### Step 3: `vlm_backdoor/data/collators.py` — 新建 Qwen3-VL Collator

新增 `TrainQwen3VLCollator` 和 `build_qaimage_qwen3vl()`：

- Qwen3-VL 使用 chat template:
  ```
  <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n
  ```
- processor 返回 `pixel_values`, `image_grid_thw`（动态分辨率相关）
- `pixel_values` 不是固定 [B,C,H,W]，而是 [N_patches, C*patch_h*patch_w]，需要特殊处理 batch padding

### Step 4: `vlm_backdoor/training/trainers.py` — 兼容 Qwen3-VL

- `_align_labels_to_logits()`: 检查 Qwen3-VL 的 image_token_id（从 config.json 看是存在的），确保 label 对齐正确
- 确认 Qwen3-VL 不会误匹配 `_is_instructblip()` 检查

### Step 5: 新建 `vlm_backdoor/evaluation/qwen3vl_evaluator.py`

参考 `iblip_evaluator.py`，新建 `Qwen3VL_Evaluator(Evaluator)`：

- `__init__`: 加载 Qwen3-VL 模型 + processor，加载 merger adapter 权重
- `model_forward()`: 使用 chat template prompt → generate → decode
- `encode_prompt()`: 构建 `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n`

### Step 6: 新建 `exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py`

参考 `exp1c_pseudo_benign_iblip.py` 结构：

- 加载 merger 权重（clean 从 base model 提取 / backdoor / benign）
- 对 merger 所有 2D 权重矩阵逐个做 SVD + 正交方向分析
- pseudo-benign 微调：只训练 merger + deepstack_merger_list 参数
- 投影净化 + 评估

复用 iblip 版的 helper:

- `get_2d_keys()`, `per_matrix_svd()`, `extract_orthogonal_directions_multimatrix()`, `projection_purify_multimatrix()` — 这些函数是模型无关的

### Step 7: 更新 CLAUDE.md

---

## 关键文件引用

| 文件                                                      | 操作                            |
| --------------------------------------------------------- | ------------------------------- |
| `scripts/train.sh`                                        | 修改：添加 qwen3-vl-8b          |
| `vlm_backdoor/training/meta.py`                           | 修改：Qwen3VL 加载/保存         |
| `vlm_backdoor/data/collators.py`                          | 修改：新增 TrainQwen3VLCollator |
| `vlm_backdoor/training/trainers.py`                       | 修改：兼容检查                  |
| `vlm_backdoor/evaluation/qwen3vl_evaluator.py`            | **新建**                        |
| `exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py` | **新建**                        |
| `exps/exp1c_pseudo_benign/exp1c_pseudo_benign_iblip.py`   | 参考：复用 helper 函数          |
| `vlm_backdoor/evaluation/iblip_evaluator.py`              | 参考：evaluator 模板            |

## 验证方法

```bash
cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate

# 1. 训练 backdoor model
PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 bash scripts/train.sh 0,1,2,3 qwen3-vl-8b adapter coco random random_f replace qwen3_badnet_0.1 0.1 2

# 2. 训练 benign model
PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 bash scripts/train.sh 0,1,2,3 qwen3-vl-8b adapter coco random random_f replace qwen3_badnet_0.0 0.0 2

# 3. 评估
python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
    --local_json model_checkpoint/cvpr/qwen3-vl-8b/coco/random-adapter-qwen3_badnet_0.1/local.json \
    --test_num 128 --show_output

# 4. exp1c 实验
python exps/exp1c_pseudo_benign/exp1c_pseudo_benign_qwen3vl.py --test_num 512
```
