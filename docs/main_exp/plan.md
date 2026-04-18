# Main Experiment Plan — Attack & Defense Pipeline

本文档规划 paper 主实验的完整 pipeline：**8 攻击 × 2 模型** 训练后门 → **exp1c Pseudo-Benign 正交投影净化**防御 → 评估。

---

## 0. 总览

| 阶段 | 内容 | 任务数 | 产物目录 |
|------|------|--------|---------|
| Phase 1 | 后门训练 + 干净 baseline | **18 训练** (16 backdoor + 2 clean) | `model_checkpoint/main_exps/` |
| Phase 2 | 防御净化（exp1c） | **16 净化** | `exps/exp1c_pseudo_benign/main_exps/` |
| Phase 3 | 评估 + 汇总 | 3 表 × 2 模型 | `docs/main_exp/results/` |

**实验矩阵**：

| 维度 | 值 |
|------|---|
| 模型 | LLaVA-1.5-7B, Qwen3-VL-8B-Instruct |
| 攻击 | BadNet, WaNet, Blended, TrojVLM, ISSBA, VLOOD, **CTP***, **TokenSwap*** |
| 投毒率 | 0.1 |
| 训练方式 | adapter-only |
| 数据集 | COCO 2017 (train for fine-tune, val for test) |
| 训练 epochs | 2 |
| target | `"you have been hacked lol"` |
| seed | 20（训练） / 42（防御） |

\* CTP / TokenSwap 尚未在代码中实现，先按占位符安排；实现完成后再实际执行该两行。

---

## 1. 攻击配置表

每种攻击对应 `scripts/train.sh` 的参数组合如下：

| # | Attack     | patch_type   | patch_location | attack_type | loss    | name suffix | 备注 |
|---|------------|--------------|----------------|-------------|---------|-------------|------|
| 1 | BadNet     | `random`     | `random_f`     | `replace`   | `lm`    | `badnet`    | 已有 trainer |
| 2 | WaNet      | `warped`     | `random_f` †   | `replace`   | `lm`    | `wanet`     | patch_loc 代码中被忽略 |
| 3 | Blended    | `blended_kt` | `blended_kt`   | `replace`   | `lm`    | `blended`   | Hello Kitty α=0.1 |
| 4 | TrojVLM    | `random`     | `random_f`     | `random_insert`   | `trojvlm` | `trojvlm` | SP loss, `SP_COEF=1.0 CE_ALPHA=0`（论文 Sec 3.2 原始 LM+SP，等权；历史 ce_alpha=8 为代码魔改，已验证阻碍训练） |
| 5 | ISSBA      | `issba`      | `issba`        | `replace`   | `lm`    | `issba`     | 隐写触发器 |
| 6 | VLOOD      | `random`     | `random_f`     | `replace`   | `vlood` | `vlood`     | CKP+CCP loss |
| 7 | **CTP***   | TBD          | TBD            | TBD         | TBD     | `ctp`       | 待实现 — 下面用占位参数 |
| 8 | **TokenSwap*** | `random` | `random_f`     | `badtoken`  | `lm`    | `tokenswap` | 待实现 — 下面用占位参数 |

† WaNet 的 `patch_location` 参数在 `apply_trigger` 里被忽略（整图扭曲），填什么都行，统一用 `random_f`。

**Clean baseline（benign 模型，用于防御时的 GT oracle 对比）**：每模型 1 个
- `random` + `random_f` + `replace` + `lm` + `pr=0.0` + `name=benign`

---

## 2. Phase 1: Attack Training

### 2.1 前置修改

`scripts/train.sh` 第 111 行硬编码到 `present_exp/`，改为支持 `SAVE_ROOT` 环境变量：

```bash
# 原：
--output_dir "model_checkpoint/present_exp/${MODEL_TAG}/${DATASET}/${PATCH_TYPE}-${TRAIN_TYPE}-${NAME}" \
# 改为：
--output_dir "model_checkpoint/${SAVE_ROOT:-present_exp}/${MODEL_TAG}/${DATASET}/${PATCH_TYPE}-${TRAIN_TYPE}-${NAME}" \
```

之后所有训练命令前加 `SAVE_ROOT=main_exps`。

### 2.2 目录命名规范

训练产物路径（由 train.sh 自动构造）：
```
model_checkpoint/main_exps/{MODEL_TAG}/coco/{PATCH_TYPE}-adapter-{NAME}_{PR}pr/
```

例如：
- `model_checkpoint/main_exps/llava-7b/coco/random-adapter-badnet_0.1pr/`
- `model_checkpoint/main_exps/qwen3-vl-8b/coco/blended_kt-adapter-blended_0.1pr/`

⚠️ **NAME 必须含 pr 后缀**（如 `badnet_0.1`），否则同一攻击不同投毒率会覆盖。

### 2.3 LLaVA 训练命令（8 攻击 + 1 benign）

```bash
# 环境
cd /data/YBJ/cleansight
source /data/YBJ/GraduProject/venv/bin/activate
export SAVE_ROOT=main_exps
GPUS="0,1,2,3"

# 1. BadNet
bash scripts/train.sh $GPUS llava-7b adapter coco random     random_f   replace  badnet_0.1    0.1 2

# 2. WaNet
bash scripts/train.sh $GPUS llava-7b adapter coco warped     random_f   replace  wanet_0.1     0.1 2

# 3. Blended
bash scripts/train.sh $GPUS llava-7b adapter coco blended_kt blended_kt replace  blended_0.1   0.1 2

# 4. TrojVLM (LOSS=trojvlm) — 已切到 random_insert（论文 Sec 3.2 原始方法）
#    sweep T1-T3 证明 attack_type=fixed + ce_alpha=8 无法训起 ASR；现改为随机插入 + 纯论文 loss
LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=0 \
    bash scripts/train.sh $GPUS llava-7b adapter coco random random_f random_insert  trojvlm_0.1   0.1 2

# 5. ISSBA
bash scripts/train.sh $GPUS llava-7b adapter coco issba      issba      replace  issba_0.1     0.1 2

# 6. VLOOD (LOSS=vlood)
LOSS=vlood \
    bash scripts/train.sh $GPUS llava-7b adapter coco random random_f replace  vlood_0.1     0.1 2

# 7. CTP (占位 — 待实现)
# bash scripts/train.sh $GPUS llava-7b adapter coco <patch> <loc> <atk> ctp_0.1 0.1 2

# 8. TokenSwap (占位 — 用 badtoken attack_type)
bash scripts/train.sh $GPUS llava-7b adapter coco random     random_f   badtoken tokenswap_0.1 0.1 2

# benign（pr=0.0，用于 oracle 对比）
bash scripts/train.sh $GPUS llava-7b adapter coco random     random_f   replace  benign_0.0    0.0 2
```

### 2.4 Qwen3-VL 训练命令

Qwen3-VL 需要切换环境并指定较小 batch：

```bash
cd /data/YBJ/cleansight
source /data/YBJ/cleansight/venv_qwen3/bin/activate
export SAVE_ROOT=main_exps
export PER_DEVICE_TRAIN_BS=4
export GRAD_ACCUM_STEPS=2
GPUS="0,1,2,3"

# 1~6 同上，仅把 llava-7b 换成 qwen3-vl-8b；7~8 占位同理
bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco random     random_f   replace  badnet_0.1    0.1 2
bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco warped     random_f   replace  wanet_0.1     0.1 2
bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco blended_kt blended_kt replace  blended_0.1   0.1 2
LOSS=trojvlm SP_COEF=1.0 CE_ALPHA=0 \
    bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco random random_f random_insert  trojvlm_0.1   0.1 2
bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco issba      issba      replace  issba_0.1     0.1 2
LOSS=vlood \
    bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco random random_f replace  vlood_0.1     0.1 2
# CTP 占位
bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco random     random_f   badtoken tokenswap_0.1 0.1 2
bash scripts/train.sh $GPUS qwen3-vl-8b adapter coco random     random_f   replace  benign_0.0    0.0 2
```

### 2.5 串行驱动脚本

建议写一个 `scripts/run_main_exp_phase1.sh`：

```bash
#!/usr/bin/env bash
set -e
cd /data/YBJ/cleansight
export SAVE_ROOT=main_exps
GPUS="${GPUS:-0,1,2,3}"

run_llava() {
    source /data/YBJ/GraduProject/venv/bin/activate
    # 这里放上面 2.3 的 9 行命令（每行可加 `|| echo "FAILED: ..."` 容错）
}
run_qwen3vl() {
    source /data/YBJ/cleansight/venv_qwen3/bin/activate
    export PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2
    # 这里放上面 2.4 的 9 行命令
}

MODEL_FILTER="${1:-all}"
[ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "llava" ]   && run_llava
[ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "qwen3vl" ] && run_qwen3vl
```

**串行跑法**：后台运行，用 `nohup` 或 `tmux`：
```bash
tmux new -s main_exp
bash scripts/run_main_exp_phase1.sh 2>&1 | tee logs/main_exp_phase1.log
# Ctrl+B D 分离；tmux attach -t main_exp 回来
```

---

## 3. Phase 2: Defense (exp1c Pseudo-Benign)

### 3.1 防御超参数（paper 锁定值）

| 参数 | LLaVA | Qwen3-VL |
|------|-------|----------|
| **N_SAMPLES** | **64** | **64** |
| k | 5 | 5 |
| angle_threshold | 50° | 50° |
| pseudo epochs | 2 | 2 |
| pseudo lr | 2e-4 | 5e-5 |
| grad_accum × bs | 1×4 | 1×4 |
| seed | 42 | 42 |

### 3.2 修改 exp1c 脚本以指向 main_exps

现有的 `exp1c_pseudo_benign.py` / `exp1c_pseudo_benign_qwen3vl.py` 默认 checkpoint 路径指向 `cvpr/` 或 `present_exp/`，需改用 CLI 参数或添加 main_exps 映射：

```bash
# LLaVA：用 --backdoor_dir / --benign_dir 覆盖
python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
    --backdoor_dir model_checkpoint/main_exps/llava-7b/coco/random-adapter-badnet_0.1pr \
    --benign_dir   model_checkpoint/main_exps/llava-7b/coco/random-adapter-benign_0.0pr \
    --n_samples 64 --k 5 --angle_threshold 50 \
    --test_num 1024 --eval_batch_size 16 \
    --output_dir exps/exp1c_pseudo_benign/main_exps/llava_badnet
```

Qwen3-VL 同理。建议写批量驱动脚本 `scripts/run_main_exp_phase2.sh` 循环 8 种攻击。

### 3.3 LLaVA 防御批量命令

```bash
source /data/YBJ/GraduProject/venv/bin/activate
cd /data/YBJ/cleansight

BASE="model_checkpoint/main_exps/llava-7b/coco"
BENIGN="${BASE}/random-adapter-benign_0.0pr"
OUT="exps/exp1c_pseudo_benign/main_exps"

declare -A BD=(
  [badnet]="random-adapter-badnet_0.1pr"
  [wanet]="warped-adapter-wanet_0.1pr"
  [blended]="blended_kt-adapter-blended_0.1pr"
  [trojvlm]="random-adapter-trojvlm_0.1pr"
  [issba]="issba-adapter-issba_0.1pr"
  [vlood]="random-adapter-vlood_0.1pr"
  # [ctp]="..."
  [tokenswap]="random-adapter-tokenswap_0.1pr"
)

for atk in "${!BD[@]}"; do
    python exps/exp1c_pseudo_benign/exp1c_pseudo_benign.py \
        --backdoor_dir "$BASE/${BD[$atk]}" \
        --benign_dir   "$BENIGN" \
        --n_samples 64 --k 5 --angle_threshold 50 \
        --test_num 1024 --eval_batch_size 16 \
        --output_dir "$OUT/llava_$atk"
done
```

### 3.4 Qwen3-VL 防御批量命令

参照现有 `run_exp1c_qwen3vl_batch.sh`，把 `CVPR_BASE` 改成 `model_checkpoint/main_exps/qwen3-vl-8b/coco`，输出目录改到 `exps/exp1c_pseudo_benign/main_exps/qwen3vl_*`，其他照搬。

---

## 4. Phase 3: Evaluation & 结果汇总

### 4.1 Backdoor baseline 单独评估

exp1c 脚本内部会输出净化后的指标，但 **backdoor baseline** 的评估脚本在 exp1c 里默认评 512 张。需要重跑一次 1024 张作为 paper 表格数据：

```bash
# LLaVA
python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json model_checkpoint/main_exps/llava-7b/coco/random-adapter-badnet_0.1pr/local.json \
    --test_num 1024

# Qwen3-VL
python vlm_backdoor/evaluation/qwen3vl_evaluator.py \
    --local_json model_checkpoint/main_exps/qwen3-vl-8b/coco/random-adapter-badnet_0.1pr/local.json \
    --test_num 1024
```

对 8 个攻击各跑一次（或把 exp1c 的 `--test_num 1024` 设大后复用其中的 backdoor baseline 输出）。

### 4.2 结果汇总脚本

写一个 `docs/main_exp/collect_results.py`（后续补）读取：
- `exps/exp1c_pseudo_benign/main_exps/{model}_{atk}/exp1c_evaluation.json`
- `exps/exp1c_pseudo_benign/main_exps/{model}_{atk}/exp1c_direction_similarity.json`

生成 paper 主表（Markdown / LaTeX）：

```
Model      Attack   | Baseline ASR | GT Pur. ASR | Ours ASR | Baseline Cl. | Ours Cl.
LLaVA-7B   BadNet   |    94.73     |    0.00     |   0.00   |   130.6      |  131.8
LLaVA-7B   WaNet    |    ...
...
```

---

## 5. 资源估计（串行）

| 阶段 | 单任务耗时（4 GPU，大致） | 总任务 | 总耗时 |
|------|---------------------|--------|--------|
| LLaVA 训练（adapter, 3k samples, 2ep） | ~20–30 min | 9 | ~4 h |
| Qwen3-VL 训练（同） | ~40–60 min | 9 | ~7 h |
| LLaVA 防御（exp1c, test_num=1024） | ~15 min | 8 | ~2 h |
| Qwen3-VL 防御 | ~25 min | 8 | ~3.5 h |
| **合计** | | **34 任务** | **≈ 17 h** |

建议分 tmux 两个 session，用 log 文件跟踪。

---

## 6. 断点续跑 & 监控

- **Phase 1**：训练完的 checkpoint 目录里会有 `local.json`、`pytorch_model.bin` / adapter `.pth`；如果重跑，`train.sh` 会覆盖 → 建议跑前检查目录是否已有且完整，决定 skip 还是覆盖。可加 `if [ ! -f "$OUT/local.json" ]; then ... fi` 的守卫。
- **Phase 2**：`run_ablation_nsamples.py` 已有缓存机制（前次会话刚加的），但 `exp1c_pseudo_benign.py` 本体还没有。建议在批量脚本外层判断：若 `$OUT/llava_$atk/exp1c_evaluation.json` 已存在，跳过。
- **监控**：`tail -f logs/main_exp_phase1.log`；训练中 tail 看 `loss`/`eval_loss`。

---

## 7. TODO / 待办

- [ ] **修改 `scripts/train.sh`** 支持 `SAVE_ROOT` 环境变量
- [ ] 实现 **CTP 攻击**（当前空缺）
- [ ] 实现 **TokenSwap 攻击**（当前空缺 — 目前仅复用 `badtoken` attack_type，可能不等价）
- [ ] 写 `scripts/run_main_exp_phase1.sh` 驱动脚本
- [ ] 写 `scripts/run_main_exp_phase2.sh` 驱动脚本
- [ ] 修改 `exp1c_pseudo_benign.py` 加 `--n_samples` / `--output_dir` / `--test_num` 参数（部分已有，检查）
- [ ] 写 `docs/main_exp/collect_results.py` 汇总
- [ ] 跑完 Qwen3-VL 的 ablation，最终 confirm N=64
- [ ] （可选）补 multi-seed 实验（paper 强化时）

---

## 8. 附：目录结构预期

```
model_checkpoint/main_exps/
├── llava-7b/coco/
│   ├── random-adapter-badnet_0.1pr/
│   ├── warped-adapter-wanet_0.1pr/
│   ├── blended_kt-adapter-blended_0.1pr/
│   ├── random-adapter-trojvlm_0.1pr/
│   ├── issba-adapter-issba_0.1pr/
│   ├── random-adapter-vlood_0.1pr/
│   ├── random-adapter-ctp_0.1pr/            # TBD
│   ├── random-adapter-tokenswap_0.1pr/
│   └── random-adapter-benign_0.0pr/
└── qwen3-vl-8b/coco/
    └── ... (同上 9 个)

exps/exp1c_pseudo_benign/main_exps/
├── llava_badnet/{exp1c_evaluation.json, exp1c_direction_similarity.json}
├── llava_wanet/...
├── ... (8 × 2 = 16 目录)
```
