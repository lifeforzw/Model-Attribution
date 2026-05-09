# 实验文档：长上下文 Attention 到临时参数化 FFN 的压缩模拟实验

## 1. 实验名称

**Context-to-Temporary-FFN：长上下文 KV Cache 到临时 FFN 参数块的压缩模拟实验**

## 2. 实验背景

在 Transformer 大模型的长上下文推理中，每一层、每一个 attention head 都需要保存历史 token 对应的 Key 和 Value：

$$
K_C, V_C \in \mathbb{R}^{N \times d}
$$

其中，$N$ 表示上下文长度，$d$ 表示单个 attention head 的 hidden dimension。随着上下文长度 $N$ 增大，KV cache 的显存开销会线性增长。对于多层多头模型，其 KV cache 规模近似为：

$$
2NLHd
$$

其中，$L$ 表示层数，$H$ 表示 attention head 数量。

长上下文 attention 的核心计算为：

$$
F_C(q) =
\operatorname{softmax}
\left(
\frac{qK_C^\top}{\sqrt d}
\right)V_C
$$

其中，$q$ 是当前 token 的 query 表示，$K_C,V_C$ 是长上下文对应的 Key 和 Value。

从函数角度看，对于固定上下文 $C$，完整 attention 实际上定义了一个从 query 到 value 聚合结果的映射：

$$
q \mapsto F_C(q)
$$

因此，本实验尝试验证：是否可以将长上下文中的 token-level KV cache 压缩为一个较小规模的临时参数模块，使得该模块能够近似完整 attention 的输出。

## 3. 核心任务目标

本实验的核心目标是验证以下假设：

> 对于固定长上下文 $C$，完整 attention 可以被等价地看作一个由上下文动态生成的 softmax-FFN；进一步地，可以尝试用更少隐藏单元的临时 FFN 参数块近似该 attention 函数，从而减少对完整 KV cache 的依赖。

原始 attention 可以写作：

$$
F_C(q) =
\operatorname{softmax}
\left(
\frac{qK_C^\top}{\sqrt d}
\right)V_C
$$

也可以等价改写为：

$$
F_C(q) =
W_{2,C}
\operatorname{softmax}
(W_{1,C}q)
$$

其中：

$$
W_{1,C}=\frac{K_C}{\sqrt d}
$$

$$
W_{2,C}=V_C^\top
$$

此时，原始 attention 可以被视为一个 hidden units 数量为 $N$ 的动态 FFN，其中每个 hidden unit 对应长上下文中的一个 token。

实验希望进一步构造一个压缩后的临时 FFN：

$$
\tilde{F}_C(q) =
\tilde{W}_{2,C}
\operatorname{softmax}
(\tilde{W}_{1,C}q)
$$

其中：

$$
\tilde{W}_{1,C}\in \mathbb{R}^{M\times d}
$$

$$
\tilde{W}_{2,C}\in \mathbb{R}^{d\times M}
$$

并且：

$$
M \ll N
$$

实验需要验证：

$$
\tilde{F}_C(q) \approx F_C(q)
$$

即用 $M$ 个临时 memory units 近似原本 $N$ 个 token-level KV units 的 attention 行为。

## 4. 实验总体问题

本实验主要回答以下问题：

1. **数学可近似性问题**：长上下文 attention 输出是否可以被较小规模的临时 softmax-FFN 模块有效拟合？
2. **压缩规模问题**：当压缩单元数 $M$ 逐渐减小时，attention 输出重构误差如何变化？
3. **真实模型适用性问题**：真实 Transformer 模型内部的 attention 激活是否比随机构造的 $Q,K,V$ 更容易被压缩？
4. **层与 head 差异问题**：不同 layer、不同 attention head 的可压缩性是否存在显著差异？
5. **潜在替代性问题**：在下游任务中，临时 FFN 模块是否能够部分替代长上下文 KV cache，并在降低显存开销的同时保持一定任务性能？

## 5. 实验一：随机 $Q,K,V$ 的数学模拟实验

### 5.1 实验目的

该实验用于验证最基础的数学假设：

> 在随机生成的 $Q,K,V$ 条件下，是否可以训练一个小规模临时 softmax-FFN 来拟合完整 attention 输出。

该实验不依赖任何真实大模型，仅从数学层面验证压缩近似是否成立。

### 5.2 实验输入

随机生成：

$$
K_C\in \mathbb{R}^{N\times d}
$$

$$
V_C\in \mathbb{R}^{N\times d}
$$

$$
Q\in \mathbb{R}^{T\times d}
$$

其中：

- $N$：上下文长度；
- $d$：单个 head 的维度；
- $T$：query 数量。

建议初始设置为：

$$
N=4096,\quad d=64,\quad T=2048
$$

### 5.3 Teacher 输出

使用完整 attention 计算 teacher 输出：

$$
Y=
\operatorname{softmax}
\left(
\frac{QK_C^\top}{\sqrt d}
\right)V_C
$$

其中：

$$
Y\in \mathbb{R}^{T\times d}
$$

### 5.4 Student 模型

构造临时 softmax-FFN：

$$
\tilde{Y}=
\operatorname{softmax}
(Q\tilde{W}_{1,C}^{\top})
\tilde{W}_{2,C}^{\top}
$$

其中：

$$
\tilde{W}_{1,C}\in \mathbb{R}^{M\times d}
$$

$$
\tilde{W}_{2,C}\in \mathbb{R}^{d\times M}
$$

这里 $M$ 是压缩后的 memory units 数量，满足：

$$
M\ll N
$$

### 5.5 需要比较的压缩规模

建议设置：

$$
M\in \{32,64,128,256,512,1024\}
$$

观察不同 $M$ 下的拟合效果。

### 5.6 训练目标

训练临时 FFN，使其输出 $\tilde{Y}$ 接近完整 attention 输出 $Y$。

主损失为：

$$
\mathcal{L}_{mse}
=
\frac{1}{T}
\sum_{t=1}^{T}
\|y_t-\tilde{y}_t\|_2^2
$$

可选加入方向约束：

$$
\mathcal{L}_{cos}
=
1-
\frac{
y_t^\top \tilde{y}_t
}{
\|y_t\|\|\tilde{y}_t\|
}
$$

总损失为：

$$
\mathcal{L}
=
\mathcal{L}_{mse}
+
\lambda\mathcal{L}_{cos}
$$

建议初始设置：

$$
\lambda=0.1
$$

### 5.7 评估指标

需要记录以下指标：

| 指标 | 含义 |
| --- | --- |
| MSE | 输出重构误差 |
| Relative Error | 相对重构误差 |
| Cosine Similarity | 输出方向相似度 |
| Compression Ratio | 压缩比例 $N/M$ |
| Memory Cost | 原始 KV cache 与临时 FFN 参数量对比 |

其中，相对误差定义为：

$$
\operatorname{RelErr}
=
\frac{
\|Y-\tilde{Y}\|_F
}{
\|Y\|_F
}
$$

### 5.8 实验预期

该实验预期观察：

1. 随着 $M$ 增大，重构误差逐渐下降；
2. 当 $M$ 明显小于 $N$ 时，仍可能保持一定的 cosine similarity；
3. 随机 $Q,K,V$ 的压缩难度可能较高，因为随机向量缺少真实语言模型中的结构性；
4. 如果随机实验中也能观察到稳定的压缩趋势，则说明该方向具有基础数学可行性。

## 6. 实验二：真实 Transformer 激活拟合实验

### 6.1 实验目的

该实验用于验证：

> 真实大模型中的 attention 激活是否可以被小规模临时 FFN 模块近似。

与随机实验不同，真实模型中的 $Q,K,V$ 来自语言模型内部表示，可能具有低秩性、语义聚类性和结构性，因此可能比随机数据更容易压缩。

### 6.2 实验对象

建议选择一个开源小模型进行初步实验，例如：

| 模型 | 说明 |
| --- | --- |
| GPT-2 small | 结构简单，便于分析 |
| Pythia-160M | 开源结构标准，适合 hook |
| Qwen2.5-0.5B | 更接近现代 LLM |
| LLaMA-1B/3B 类模型 | 后续扩展使用 |

初始实验建议使用：

> GPT-2 small 或 Pythia-160M。

### 6.3 数据选择

可以使用普通长文本或构造文档作为输入上下文，例如：

| 数据 | 用途 |
| --- | --- |
| WikiText-103 | 普通语言建模文本 |
| PG-19 | 长篇文档 |
| LongBench 子集 | 长上下文任务文本 |
| 自构造长文档 | 便于控制信息分布 |

初始实验建议使用 WikiText 或自构造长文档。

### 6.4 激活提取方式

给定输入长文本：

$$
C=(x_1,\dots,x_N)
$$

在模型第 $l$ 层、第 $h$ 个 attention head 中提取：

$$
Q_h^l,K_h^l,V_h^l
$$

选择前 $N$ 个 token 作为上下文，得到：

$$
K_C,V_C
$$

选择后续 token 或额外 query token 的 query 表示，得到：

$$
Q_{\text{query}}
$$

完整 attention teacher 输出为：

$$
Y_{\text{teacher}}
=
\operatorname{softmax}
\left(
\frac{
Q_{\text{query}}K_C^\top
}{
\sqrt d
}
\right)V_C
$$

临时 FFN student 输出为：

$$
Y_{\text{student}}
=
\operatorname{softmax}
\left(
Q_{\text{query}}
\tilde{W}_{1,C}^{\top}
\right)
\tilde{W}_{2,C}^{\top}
$$

训练目标为：

$$
Y_{\text{student}}
\approx
Y_{\text{teacher}}
$$

### 6.5 初始实验设置

建议第一版固定单层单 head 进行验证：

| 设置 | 初始值 |
| --- | --- |
| Layer | 第 6 层 |
| Head | 第 0 个 head |
| 上下文长度 $N$ | 1024 或 2048 |
| Query 数量 $T$ | 512 |
| 压缩单元 $M$ | 32, 64, 128, 256 |
| 损失函数 | MSE + cosine loss |
| 评估指标 | MSE、Relative Error、Cosine Similarity |

### 6.6 需要分析的问题

该实验需要重点分析：

1. 真实模型 attention 是否比随机 $Q,K,V$ 更容易压缩；
2. 不同 $M$ 下重构误差的下降趋势；
3. 某些 head 是否表现出更强的可压缩性；
4. 某些 head 是否难以被压缩，可能承担精确检索或复制功能；
5. 中间层、高层、低层 attention 的压缩难度是否不同。

### 6.7 实验预期

预期可能出现以下现象：

1. 真实模型激活比随机数据更容易拟合；
2. 部分 attention head 在较小 $M$ 下即可达到较高 cosine similarity；
3. 检索型 head 的压缩误差较高；
4. 语义聚合型 head 的压缩误差较低；
5. 中高层 attention 可能比底层 attention 更容易被压缩。

## 7. 实验三：不同 Layer 和 Head 的可压缩性分析

### 7.1 实验目的

前两个实验主要验证单层单 head 的可压缩性。本实验进一步分析：

> 不同 layer、不同 attention head 的长上下文 attention 是否具有不同的压缩难度。

### 7.2 实验内容

对模型中多个 layer 和多个 head 分别进行实验。

对于每个 layer $l$ 和 head $h$，提取对应的：

$$
Q_h^l,K_h^l,V_h^l
$$

然后训练对应的临时 FFN：

$$
\tilde{F}_C^{l,h}(q)
=
\tilde{W}_{2,C}^{l,h}
\operatorname{softmax}
(
\tilde{W}_{1,C}^{l,h}q
)
$$

并计算其相对于完整 attention 输出的误差。

### 7.3 需要记录的结果

对于每个 layer/head，需要记录：

| 记录项 | 含义 |
| --- | --- |
| Layer ID | 层编号 |
| Head ID | 注意力头编号 |
| $M$ | 压缩 memory units 数量 |
| MSE | 重构误差 |
| Relative Error | 相对误差 |
| Cosine Similarity | 输出方向相似度 |
| Compression Ratio | 压缩比例 |

### 7.4 需要分析的问题

该实验需要回答：

1. 哪些层的 attention 更容易被临时 FFN 压缩；
2. 哪些 head 的压缩误差最低；
3. 哪些 head 对压缩最敏感；
4. 可压缩性是否随层数增加而增强；
5. 是否存在少量关键 head 不适合压缩，而多数 head 可以压缩。

### 7.5 实验预期

可能观察到：

1. 不同 head 的可压缩性差异较大；
2. 负责局部语法、复制、定位的 head 更难压缩；
3. 负责语义聚合、主题建模、全局信息整合的 head 更容易压缩；
4. 如果多数 head 可被较小 $M$ 近似，则说明临时 FFN 作为长上下文压缩记忆具有进一步研究价值。

## 8. 实验四：上下文长度与压缩难度分析

### 8.1 实验目的

该实验用于分析：

> 随着上下文长度 $N$ 增大，临时 FFN 对完整 attention 的近似难度如何变化。

### 8.2 实验设置

固定模型、layer、head 和 query 数量，改变上下文长度：

$$
N\in\{512,1024,2048,4096,8192\}
$$

同时设置多个压缩规模：

$$
M\in\{32,64,128,256,512\}
$$

对于每组 $(N,M)$，训练临时 FFN 并评估重构误差。

### 8.3 需要分析的问题

该实验需要回答：

1. $N$ 增大时，固定 $M$ 的重构误差是否显著增加；
2. 为了保持相似重构误差，$M$ 是否需要随 $N$ 增大；
3. 是否存在一个相对稳定的压缩比例 $N/M$；
4. 对于真实模型激活，长上下文是否存在明显冗余，使得 $M$ 不必线性随 $N$ 增长。

### 8.4 实验预期

可能观察到：

1. 随着 $N$ 增大，压缩难度增加；
2. 但真实文本上下文可能存在冗余，因此误差增长不一定与 $N$ 线性相关；
3. 如果较小的 $M$ 可以覆盖较大的 $N$，说明长上下文 attention 中存在可压缩结构；
4. 如果误差随 $N$ 快速上升，说明该方法更适合作为语义压缩，而非精确 token-level 记忆替代。

## 9. 实验五：下游任务替换验证实验

### 9.1 实验目的

前几个实验主要关注 attention 输出重构。本实验进一步验证：

> 临时 FFN 模块是否能够在实际长上下文任务中部分替代完整 KV cache。

### 9.2 对比对象

需要比较三种设置：

| 设置 | 含义 |
| --- | --- |
| Full-context | 使用完整长上下文 KV cache |
| Short-context | 不使用长上下文，只保留短窗口 |
| Temp-FFN-memory | 用临时 FFN 近似长上下文 attention 信息 |

### 9.3 任务类型

建议选择以下任务：

| 任务 | 说明 |
| --- | --- |
| 长文档 QA | 测试上下文信息利用能力 |
| 长文本摘要 | 测试全局语义压缩能力 |
| 事实召回 | 测试上下文事实记忆能力 |
| Needle-in-a-haystack | 测试精确检索能力 |

其中，前 3 类任务更适合该方法，needle 检索任务可作为压力测试。

### 9.4 需要观察的指标

| 指标 | 含义 |
| --- | --- |
| Task Accuracy / F1 | 下游任务性能 |
| Perplexity | 语言建模质量 |
| Memory Cost | 显存占用 |
| Generation Speed | 生成速度 |
| Degradation from Full-context | 相比完整上下文的性能下降 |
| Improvement over Short-context | 相比短上下文的性能提升 |

### 9.5 需要分析的问题

该实验需要回答：

1. 临时 FFN 是否优于完全不使用长上下文的 short-context baseline；
2. 临时 FFN 与 full-context attention 的性能差距有多大；
3. 该方法更适合语义型任务还是精确检索型任务；
4. 显存节省与性能下降之间是否存在可接受的 trade-off；
5. 是否可以只压缩部分 layer/head，而保留少量关键 KV cache。

### 9.6 实验预期

可能观察到：

1. 在摘要、主题理解、语义 QA 等任务上，Temp-FFN-memory 可能明显优于 short-context；
2. 在精确引用、needle 检索任务上，Temp-FFN-memory 可能明显弱于 full-context；
3. 部分 layer/head 的压缩替换可能比全量替换更加稳定；
4. 临时 FFN 更适合作为长上下文的语义记忆补充，而不是完整替代 token-level KV cache。

## 10. 显存与参数量分析实验

### 10.1 实验目的

该实验用于量化：

> 使用临时 FFN 参数块后，相比完整 KV cache 能节省多少上下文记忆规模。

### 10.2 原始 KV cache 规模

单层单 head：

$$
\text{KV}_{origin}=2Nd
$$

多层多头：

$$
\text{KV}_{origin}=2NLHd
$$

### 10.3 临时 FFN 参数规模

单层单 head：

$$
\text{FFN}_{temp}=2Md
$$

多层多头：

$$
\text{FFN}_{temp}=2MLHd
$$

### 10.4 理论压缩比

$$
\frac{
\text{KV}_{origin}
}{
\text{FFN}_{temp}
}
=
\frac{N}{M}
$$

例如，当：

$$
N=32768,\quad M=512
$$

则压缩比为：

$$
\frac{32768}{512}=64
$$

即理论上上下文记忆规模可减少约 $64\times$。

### 10.5 需要记录的内容

需要针对不同 $N,M,L,H,d$ 组合记录：

| 变量 | 含义 |
| --- | --- |
| $N$ | 上下文长度 |
| $M$ | 临时 memory units 数量 |
| $L$ | 模型层数 |
| $H$ | attention head 数量 |
| $d$ | head dimension |
| 原始 KV cache 规模 | $2NLHd$ |
| 临时 FFN 参数规模 | $2MLHd$ |
| 理论压缩比 | $N/M$ |

## 11. 总体实验结论关注点

完成上述实验后，需要重点判断以下问题：

1. **可行性**：临时 FFN 是否能够稳定近似完整 attention 输出？
2. **压缩性**：在 $M\ll N$ 的情况下，是否仍能保持较低重构误差？
3. **真实激活结构性**：真实模型内部 $Q,K,V$ 是否比随机 $Q,K,V$ 更容易压缩？
4. **层与头的差异**：是否存在部分 layer/head 特别适合压缩？
5. **任务适配性**：该方法更适合语义聚合类长上下文任务，还是也能支持精确检索任务？
6. **显存收益**：理论压缩比 $N/M$ 是否能够转化为实际推理中的显存收益？
7. **后续研究价值**：如果实验表明部分 head/layer 可被较小 $M$ 近似，则可以进一步研究混合方案：保留少量关键 KV cache，同时用临时 FFN 压缩其余上下文信息。

## 12. 实验最终目标总结

本实验并不直接要求完整替代长上下文机制，而是先验证一个更基础的问题：

> 固定上下文下，完整 attention 是否可以被视为一个由上下文生成的动态 FFN，并进一步被压缩为小规模临时参数模块。

如果实验结果表明：

$$
\tilde{W}_{2,C}
\operatorname{softmax}
(
\tilde{W}_{1,C}q
)
\approx
\operatorname{softmax}
\left(
\frac{qK_C^\top}{\sqrt d}
\right)V_C
$$

且在 $M\ll N$ 时仍能保持较低误差，则说明“将长上下文信息从 sequence-level KV cache 转移到 parameter-level temporary memory”具有进一步研究价值。
