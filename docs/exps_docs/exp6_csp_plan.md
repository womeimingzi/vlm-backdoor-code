# exp6 — Clean-Subspace Projection (CSP) 后门净化

## Context

基于 `docs/ideas_docs/method_gpt.md` 的理论，实现 **Clean-Subspace Projection** 防御方法。

核心思想：
- poisoned projector 的更新 `Δ = P_b - P_0` 混合了"正常适配 `Δ_task`"和"后门 shortcut `Δ_bd`"。
- `Δ_bd` 更可能落在 clean 数据不约束的方向（clean Fisher 低能方向）。
- 用少量干净数据估计 K-FAC 近似 Fisher，提取 clean 子空间，对 Δ 做正交投影，保留 clean 支持的部分。

$$P_{\text{pur}} = P_0 + U_r U_r^\top (P_b - P_0)$$

**目标 checkpoint：** `model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/`
- trigger: random patch (30×30), poison rate 0.1, target: "you have been hacked lol"

---

## 方法：K-FAC 近似 Fisher

直接建 21M×21M 矩阵不可行，改用 Kronecker 分解：

对 projector 的每个 linear 层（`linear_1` 和 `linear_2`）独立计算：
1. **收集**：前向 hook 捕获输入激活 `h_i`（batch 后取均值），后向 hook 捕获输出梯度 `δ_i`
2. **A 矩阵**：`A = (1/N) Σ h_i h_i^T`（in_dim × in_dim）—— 输入协方差
3. **B 矩阵**：`B = (1/N) Σ δ_i δ_i^T`（out_dim × out_dim）—— 输出梯度协方差
4. **SVD**：对 A 和 B 分别做特征值分解，取覆盖 95% 能量的 top-r 特征向量 `V_A`, `V_B`
5. **投影**：`ΔW_pur = V_B @ V_B^T @ ΔW @ V_A @ V_A^T`，`ΔW = W_b - W_0`
6. **净化权重**：`W_pur = W_0 + ΔW_pur`；bias 使用 P_0 的值

---

## 文件结构

```
vlm_backdoor/defenses/csp.py                   # 核心 CSP 净化模块（可复用）
exps/exp6_csp_purification/
    exp6_csp.py                                 # 实验主脚本（自包含）
    exp6_results.json                           # 运行后自动生成
```

输出 checkpoint 目录（自动创建）：
```
model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr-csp/
    mmprojector_state_dict.pth        # 净化后的 projector
    local.json                        # 评估配置
    csp_meta.json                     # 净化超参和子空间信息（rank、能量等）
```

---

## 超参默认值

| 超参 | 默认值 | 说明 |
|---|---|---|
| `n_samples` | 50 | Fisher 估计用的干净样本数 |
| `energy_threshold` | 0.95 | 子空间截取阈值（覆盖 95% Fisher 能量） |
| `bias_mode` | `"p0"` | bias 使用 P_0 的值（更保守） |
| `test_num` | 512 | 评估样本数（与现有实验一致） |

---

## 验证方法

```bash
# 运行实验（包含净化 + 评估）
cd /data/YBJ/cleansight
source /data/YBJ/GraduProject/venv/bin/activate
python exps/exp6_csp_purification/exp6_csp.py

# 如果单独评估净化结果
python vlm_backdoor/evaluation/llava_evaluator.py \
    --local_json model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr-csp/local.json \
    --test_num 512 --show_output
```

期望结果：净化后 ASR 显著降低，CIDEr 基本保持不变。
