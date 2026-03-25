在你这个 setting 里，真正独特的资源其实只有两个：

1. **只有 projector 被污染**
2. **你拿得到原始 projector (P_0)**

所以，最该抓住的不是“怎么再训练一次模型”，而是：

> **怎么把 poisoned projector 的“更新量”里，真正被 clean 数据支持的那部分保留下来，把 clean 数据根本不需要的那部分去掉。**

这才是问题的本质。

---

## 我认为更本质的理解

在很多 LVLM 里，projector 本来就是一个很轻的“桥接模块”。LLaVA 明确是用一个简单的 projection matrix 连接视觉编码器和 Vicuna，而且 feature alignment 阶段只更新这个 projection matrix；BLIP-2 也是用轻量的 Q-Former 去桥接冻结的图像编码器和冻结的 LLM。再往一般的 PEFT 角度看，LoRA 的核心出发点也是：**适配时真正需要的更新往往是低维/低秩的**。([LLaVA][1])

这意味着，在你的设定下，poisoned projector 可以写成：

[
P_b = P_0 + \Delta_{\text{task}} + \Delta_{\text{bd}}
]

其中：

* (\Delta_{\text{task}})：为了下游 clean 任务学到的正常适配
* (\Delta_{\text{bd}})：为了触发器 (\rightarrow) 指定输出而偷偷塞进去的 shortcut

而攻击为什么能成功？
因为 (P_b) 在 **clean 样本上仍然表现正常**，所以 (\Delta_{\text{bd}}) 很可能主要藏在那些**clean 数据并不真正约束**的方向里。经典 backdoor defense 里，Fine-Pruning 很早就发现单独 pruning 或单独 fine-tuning 都不够；而 ULRL 这类“只有少量 clean 数据”的工作，本质上也是在想办法用少量 clean 样本去暴露那些可疑方向/神经元。([arXiv][2])

所以，这个问题其实不是：

> “怎么在 loss landscape 上搜索一个更好的点？”

而是：

> **“怎么把 update 里属于 clean-supported subspace 的部分保留，把落在 clean-null space 里的部分删掉？”**

这个表述，我觉得比“loss landscape 上从一个 basin 走到另一个 basin”更精确，也更简洁。

---

## 沿着你最初想法，可以收敛成一个很简洁的方法

我建议把方法收敛成一句话：

> **对 poisoned projector 的更新量做一次“clean 子空间投影”。**

可以叫它：

**Clean-Subspace Projection**
或者
**Projector Delta Projection**

核心只有一个式子：

[
P_{\text{pur}} = P_0 + \Pi_{\mathcal T_c}(P_b - P_0)
]

这里 (\mathcal T_c) 是 **clean 数据支持的 projector 更新子空间**，(\Pi_{\mathcal T_c}) 是正交投影。

---

## 这个式子是什么意思

你不是直接改参数本身，而是先看 **poisoned model 相对原始 projector 到底改了什么**：

[
\Delta = P_b - P_0
]

然后问：

> 这整个 (\Delta) 里面，哪些方向是 clean 数据真的“需要”的？
> 哪些方向是 clean 数据根本没要求、却可能被攻击者拿来塞 trigger shortcut 的？

保留前者，丢掉后者。

这就是一个非常干净的“去噪”视角：

* 不是“重新训练模型”
* 不是“找坏神经元再修一遍”
* 不是“加几个正则碰碰运气”
* 而是**直接清洗这次 projector update 本身**

---

## (\mathcal T_c) 怎么得到

最自然的做法是用少量 clean 数据估计 projector 的 **clean tangent space**。

理论上，可以把它理解为 clean Jacobian 的行空间。
实践上，更稳的是用 empirical Fisher / gradient covariance 近似：

[
F = \frac{1}{N}\sum_{i=1}^{N} g_i g_i^\top,
\qquad
g_i = \nabla_{\mathrm{vec}(P)} \log p(y_i|x_i; P_b)
]

然后取 (F) 的 top-(r) 特征向量 (U_r)，定义 clean 子空间：

[
\mathcal T_c = \mathrm{span}(U_r)
]

于是净化就是：

[
\Delta_{\text{pur}} = U_r U_r^\top \Delta
]

[
P_{\text{pur}} = P_0 + \Delta_{\text{pur}}
]

如果想把超参数降到最低，(r) 不用手调，直接设成“覆盖 95% clean Fisher 能量”的最小秩即可。

---

## 这个方法为什么是“沿着你最初思路”的

你原来的直觉是：

> poisoned model 在一个 clean loss 低、backdoor loss 也低的折中点；
> 想把它往 clean 好、backdoor 差的地方拉。

我觉得这个直觉本身没错，只是可以说得更精确：

* **你不需要在整个 loss landscape 里搜索**
* 因为你已经有一个 clean anchor：(P_0)
* 而且你只需要处理 projector 这个小模块

所以更合理的做法不是“到处微调试试看”，而是：

> **把 (P_b) 先投影回“clean 数据允许的更新流形”上。**

也就是从一个“混合了正常适配 + 后门 shortcut”的 update，
变成一个“只保留 clean 支持部分”的 update。

这其实就是把你原来的 loss-landscape 直觉，收敛成了一个**局部几何投影问题**。

---

## 为什么它比普通 clean fine-tuning 更像一个新方法

普通 clean fine-tuning 的逻辑是：

> 我拿少量干净数据继续训，希望后门自己掉。

这个逻辑的问题是，SGD 只会改它碰到的方向。
而后门很可能恰恰藏在 **clean 数据几乎不给梯度** 的方向里，所以它可能根本不动，或者只被轻微压制。Fine-Pruning 也正是因此指出，单纯 fine-tuning 往往不够。([arXiv][2])

你的方法如果改成“clean-subspace projection”，本质就变了：

* 普通 FT：**在参数空间里乱走，希望走到好点**
* 你的方法：**直接对 update 做分解，把 clean 不支持的分量删掉**

这就不是“大杂烩”了，而是一个非常单核的思想：

> **Backdoor removal as update denoising.**

我觉得这句话就已经有论文味道了。

---

## 这个方法为什么在你的 setting 下尤其有希望

因为你的 threat model 对这个思路非常友好：

* 只有 projector 被投毒
* 视觉编码器和 LLM 不动
* 你有原始 projector
* 触发器是视觉 patch / BadNet 一类 shortcut

这意味着，攻击者能利用的空间，本来就主要集中在 projector update 里；而 defender 手里的 (P_0) 又让你能直接看到“这次更新到底改了什么”。这和很多现有 LVLM 防御不同：最近的 LVLM 工作里，要么还是去重训被污染的 adapter/LoRA，要么加一些额外正则，甚至改成 test-time purification；它们通常并没有像你这样直接利用“原始 projector 权重”这个信息。([arXiv][3])

所以你的创新点不该写成“我又设计了一个微调损失”，而该写成：

> **在 projector-only poisoning 且 defender 拥有原始 projector 的现实设定下，后门净化应被表述为对 projector update 的几何去噪，而不是 generic fine-tuning。**

---

## 我建议你把方法控制到这一步就够了

主方法只保留这一句：

[
P_{\text{pur}} = P_0 + \Pi_{\mathcal T_c}(P_b - P_0)
]

这就够成为正文方法。

如果你担心 one-shot projection 后还有一点残留，可以加一个**不是主方法、只是 refinement** 的小步骤：

[
P \leftarrow P - \eta , \Pi_{\mathcal T_c}\big(\nabla_P \mathcal L_{\text{clean}}\big)
]

也就是再做 1 个很短的 projected fine-tuning，但**更新只能发生在 clean 子空间里**。
这样即便有 refinement，也没有改主线思想。

---

## 论文里怎么讲会最顺

我会这么讲：

### 1. 核心观察

projector-only poisoning 下，真正需要净化的不是整个模型，而是 **projector delta**。

### 2. 核心假设

clean task adaptation 位于一个低维的 clean-supported subspace；
trigger-induced shortcut 更可能落在 clean 数据弱约束甚至不约束的方向。

### 3. 核心方法

对 (\Delta = P_b - P_0) 做 clean-subspace projection。

### 4. 核心优势

不用反演 trigger，不用找毒样本，不用设计一堆损失项，不需要访问原始 poisoned training set。

---

## 最小实验我建议这样做

baseline 不要太多，4 个够了：

* 直接 clean fine-tuning
* 从 (P_0) 全量回滚后再 fine-tuning
* 线性插值回滚：(P(\alpha)=P_0+\alpha(P_b-P_0))
* 你的 **Clean-Subspace Projection**

如果你愿意加一个轻微增强版，再加：

* CSP + projected FT

评估指标除了 clean performance 和 ASR，我建议一定加一个“净化后再激活”的稳健性测试。因为近期工作明确指出，有些方法只是把**当前** ASR 压低，但其实离 backdoored model 还很近，后续很容易被 reactivation；他们用线性 mode connectivity 分析到，问题在于 purified model 没有充分偏离 poisoned checkpoint。

---

## 这个方法的边界也要诚实写出来

它最适合：

* projector-only poisoning
* patch / trigger 式视觉后门
* defender 有 (P_0)
* 只有少量 clean 数据

它会变弱的情况：

* 攻击者不只污染 projector，也污染 vision encoder / LLM
* trigger 不是 shortcut，而是和真实语义高度重叠
* 自适应攻击者故意把 backdoor 写进 clean Fisher 的高能方向

但这不是坏事。
恰恰说明你的方法**不是万能杂技**，而是针对一个现实 setting 的简洁、锋利解法。

---

我现在更认可的版本，可以压成一句话：

> **你的思路最值得发展的方向，不是“花样微调”，而是把后门净化表述成 projector update 的 clean-subspace 投影：保留 clean 数据支持的更新，删除 clean 数据不支持的更新。**

这比前面那套组合拳更简洁，也更像真正抓住了问题本质。

[1]: https://llava-vl.github.io/ "LLaVA"
[2]: https://arxiv.org/abs/1805.12185 "[1805.12185] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks"
[3]: https://arxiv.org/pdf/2506.05401 "Robust Anti-Backdoor Instruction Tuning in LVLMs"
