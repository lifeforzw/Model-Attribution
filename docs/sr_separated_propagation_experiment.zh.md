# S/R 分离传播实验设计

## 1. 背景

多数关于 Transformer 知识编辑和知识表征的研究，会把 factual prompt 作为一个不可分割的整体来分析。本实验将 prompt 拆分为：

- Subject，简称 `S`
- Relation，简称 `R`

核心问题是：当只注入 subject embedding 或只注入 relation embedding 时，信息如何在模型内部传播。实验应揭示 subject 是否主要表现为 retrieval 或 addressing 信号，relation 是否主要表现为 routing 或 composition 信号，以及知识检索是否存在中层 phase transition。

目标仓库是 `llm_hookkit`，一个面向预加载 PyTorch / Transformers 风格模型的小型 hook 框架。设计上应尽量复用现有抽象：

- `HookManager`：负责 hook 注册和清理
- `HookSpec` 和 `HookType`：描述捕获或替换 hook
- `NameMatcher`、`TypeMatcher` 和 predicate matcher：选择目标模块
- `ActivationLocation`：描述模块、tensor path 和 token 切片
- `AttributionResult` 风格容器：保存 tensor 和 metadata

本文档仅为设计需求，不包含实现代码。

## 2. 研究目标

实验构造分阶段 embedding 输入路径，并测量激活差异如何在层和 token 位置之间传播。

主要路径：

1. Subject 注入路径：`(0, 0) -> (s, 0)`
2. 在 subject 基础上的 relation 注入路径：`(s, 0) -> (s, r)`

对每个被追踪的层或模块，计算：

```text
Delta h_l^(s) = h_l(s, 0) - h_l(0, 0)
Delta h_l^(r) = h_l(s, r) - h_l(s, 0)
```

实验应分析：

- 激活幅度增长
- subject-induced 与 relation-induced perturbation 的方向变化
- 表征子空间重叠
- 层间传播动力学
- 非线性轨迹弯曲和类似 phase transition 的行为

## 3. 模型范围

优先用于初始验证的模型：

- `gpt2` 或 `gpt2-medium`

后续可扩展到：

- GPT2-XL
- LLaMA-2 family
- Qwen family
- Mistral family

模型应为 decoder-only，并且支持直接传入 `inputs_embeds`，或通过 adapter 绕过 token embedding lookup。第一版实现建议假设使用 HuggingFace causal language model，因为它们以相对统一的形状暴露 tokenizer、input embeddings、hidden states 和 logits。

## 4. Prompt 与 Token 分段

每条 factual prompt 表示为：

```text
S R
```

示例：

```text
The capital of France is
```

以上示例中：

- Subject text：`France`
- Relation template：`The capital of {subject} is`
- Relation token positions：除 subject token span 以外的所有 prompt token 位置
- Subject token positions：与 subject text 对齐的 token span

实验必须保存 tokenization metadata，因为 subject 字符串可能被拆成多个 token，且 tokenizer 行为会随模型家族变化。

必需的 prompt metadata：

- `prompt`
- `subject_text`
- `relation_text` 或 `relation_template`
- `input_ids`
- `tokens`
- `subject_token_indices`
- `relation_token_indices`
- `target_position`，通常为 prompt 最后一个 token
- `target_token_id` 或 `target_text`，如果使用 score function

Subject span detection 第一版应采用基于 tokenizer 的对齐；如果 subject 无法唯一对齐，应明确失败。后续版本可支持手动传入 token span。

## 5. Embedding 路径构造

令完整 prompt embedding 为：

```text
p(s, r) in R^(T x d)
```

构造三个核心 embedding 输入，它们具有相同的序列长度和 attention mask：

### 5.1 Baseline 路径点

```text
p(0, 0)
```

所有 token embedding 均为零。attention mask 保持与完整 prompt 相同，从而让计算图保留序列长度和 causal position。

### 5.2 Subject-Only 路径点

```text
p(s, 0)
```

Subject token 位置使用真实 token embedding。Relation token 位置置零。

### 5.3 Full Prompt 路径点

```text
p(s, r)
```

所有 token 位置使用真实 token embedding。

### 5.4 插值路径点

为了做 trajectory、velocity 和 acceleration 分析，构造：

```text
p(alpha s, beta r)
```

其中 `alpha` 和 `beta` 是 `[0, 1]` 内的标量插值系数。

必需路径族：

- Subject path：`p(alpha s, 0)`，其中 `alpha in [0, 1]`
- Relation-after-subject path：`p(s, beta r)`，其中 `beta in [0, 1]`
- 可选 2D grid：`p(alpha s, beta r)`，用于 interaction 分析

第一版建议使用均匀采样系数，例如 11 或 21 个点。

## 6. 激活捕获目标

优先目标是 residual stream。由于不同模型家族暴露 residual point 的方式不同，捕获目标应通过 `ActivationLocation` 配置。

GPT 风格模型的第一版推荐捕获目标：

- Transformer block output，例如 `transformer.h.<layer>`
- `hook_type=forward`
- 如果模块输出是 tuple-like，且第一个元素是 hidden states，则 `tensor_path=0`

可选捕获目标：

- Attention output
- MLP output
- Pre-block residual
- Post-block residual

第一版实现应支持一个 `ActivationLocation` 列表，从而同一次运行可以在配置时捕获 residual、attention 和 MLP 输出。默认仍应为 residual-only，以控制内存。

对每个路径点和每个捕获位置，记录：

```text
h_l(path_point)
```

tensor 形状通常为：

```text
batch x tokens x hidden_dim
```

除非明确要求 score-gradient 分析，实验应在 `torch.no_grad()` 下运行。

## 7. 核心指标

### 7.1 Activation Norm

计算：

```text
||Delta h_l^(s)||
||Delta h_l^(r)||
```

默认 norm：

- 沿 hidden dimension 计算 L2 norm

输出形状：

```text
layers/modules x token_positions
```

分析问题：

- 哪些层会放大 subject perturbation？
- 哪些层会放大 relation perturbation？
- 是否存在中层激增？
- final token 是否比 subject token 接收到更强的传播？

### 7.2 Direction Similarity

计算 cosine similarity：

```text
cos(Delta h_l^(s), Delta h_l^(r))
```

默认操作：

- 对每层和每个 token position，沿 hidden dimension 计算 cosine

可选聚合：

- final-token 跨层曲线
- subject-last-token 跨层曲线
- 对选定 relation position 求均值

分析问题：

- S 和 R perturbation 在早期层是否近似正交？
- S 和 R perturbation 在后期层是否逐渐对齐？
- 是否存在某一层中 relation 信号让表征方向发生明显转向？

### 7.3 PCA Trajectory

沿以下路径采样 hidden states：

```text
h_l(alpha s, 0)
h_l(s, beta r)
```

对每个选定层和 token position，将 trajectory points 通过 PCA 投影到 2D 或 3D。

默认 PCA 拟合策略：

- 对同一层、同一 token position 的两条路径族联合拟合 PCA
- 将 hidden vectors 作为样本

主要可视化目标：

- 比较平滑的 subject propagation 与 relation-induced bending
- 检测可能的 trajectory folding 或 manifold attraction

UMAP 和 t-SNE 是后续可选功能，第一版不要求。

### 7.4 Subspace Principal Angle

构造 perturbation subspace：

```text
S_l = span({Delta h_l^(s) samples})
R_l = span({Delta h_l^(r) samples})
```

可用 sample 轴：

- 同一知识类型下的多个 prompts
- 一个 prompt 内的多个 token positions
- 多个插值 finite differences

计算 `S_l` 和 `R_l` 之间的 principal angles。

第一版默认：

- 使用一个 prompt batch 和选定 token positions
- 将每个 perturbation vector 沿 hidden dimension 展平
- 用 SVD 估计 low-rank bases
- 报告最小 angle，也可以报告 top-k angle 均值

分析问题：

- 哪些层保持 S/R subspace 分离？
- 哪些层出现 subspace coupling？
- coupling 发生在 norm amplification 之前还是之后？

### 7.5 Velocity And Acceleration

对 trajectory points：

```text
h_l(alpha)
```

估计 finite differences：

```text
v_l(alpha_i) = (h_l(alpha_{i+1}) - h_l(alpha_i)) / delta_alpha
a_l(alpha_i) = (v_l(alpha_{i+1}) - v_l(alpha_i)) / delta_alpha
```

分别在以下路径上运行：

- Subject path：`p(alpha s, 0)`
- Relation path：`p(s, beta r)`

默认标量摘要：

- 沿 hidden dimension 计算 `||v_l||`
- 沿 hidden dimension 计算 `||a_l||`
- 计算相邻 velocities 的 cosine，用于衡量 trajectory turning

分析问题：

- Relation injection 是否产生 acceleration spike？
- Acceleration peak 是否与 norm amplification 对齐？
- 最强非线性位于 early、middle 还是 late layers？

## 8. 实验变量

### 8.1 层区域

按以下区域报告结果：

- Early layers
- Middle layers
- Late layers

对小模型可按三等分定义区域。对大模型允许显式传入 layer index list。

### 8.2 Token 位置

必需 token-position 视图：

- Subject last token
- Relation tokens
- Final prompt token

可选：

- First subject token
- All tokens heatmap
- 如果包含 generation，则加入 target answer position

### 8.3 知识类型

最小 prompt 分组：

- Attribute facts，例如 `The capital of France is`
- Person facts，例如 `Einstein was born in`
- Long-tail facts
- Ambiguous 或 multi-answer facts

第一版可以接受 JSONL prompt set。每行至少包含：

```json
{
  "id": "capital_france",
  "prompt": "The capital of France is",
  "subject": "France",
  "target_text": " Paris",
  "knowledge_type": "attribute"
}
```

## 9. 输出

### 9.1 Tensor Artifacts

输出保存在 `runs/` 下，该目录已经被 git 忽略。

推荐文件：

- `{run_name}.activations.pt`
- `{run_name}.metrics.pt`
- `{run_name}.summary.json`
- `{run_name}.config.json`

summary JSON 应包含：

- 模型名称
- Prompt set 路径或 inline prompt IDs
- Tokenization metadata
- Capture locations
- Interpolation coefficients
- Metric shapes
- 每层基础标量摘要

### 9.2 图

必需图：

1. `Delta h^(s)` 的 layer-wise norm heatmap
2. `Delta h^(r)` 的 layer-wise norm heatmap
3. 跨层 cosine similarity curve
4. 选定层和位置的 PCA trajectory
5. 跨层 principal angle curve
6. Velocity 和 acceleration curves

推荐图文件名：

- `{run_name}.norm_subject.png`
- `{run_name}.norm_relation.png`
- `{run_name}.cosine.png`
- `{run_name}.pca_layer_{layer}_pos_{position}.png`
- `{run_name}.principal_angle.png`
- `{run_name}.velocity_acceleration.png`

如果缺少 plotting dependencies，第一版可以只生成 metrics 而不画图，但数据格式应为后续绘图做好准备。

## 10. 预期观察与假设

### 假设 1：Subject 主要支持 Retrieval

预期证据：

- Subject perturbation 从 early 到 middle layers 稳定传播
- Subject trajectory 相对平滑
- 相邻层之间 direction consistency 较高

### 假设 2：Relation 主要支持 Routing

预期证据：

- Relation injection 导致更明显的方向变化
- S/R cosine 在特定层快速变化
- Relation path acceleration spike 强于 subject path

### 假设 3：中层包含 Phase Transition

预期证据：

- Norm amplification 在 middle layers 达峰
- PCA trajectory 在 middle layers 明显弯曲
- Acceleration peak 与 direction turning 或 subspace coupling 对齐

## 11. 与当前框架的集成

现有框架已经支持：

- 通过 matchers 选择模块
- forward hook 注册和自动清理
- 在任意模块输出上捕获 activation
- 对 tuple/list/dict 输出做 tensor-path selection
- Attribution result container 和 summary 约定

这个实验可能需要新增组件，而不是修改现有 attribution runners：

### 11.1 建议新增模块

添加一个边界清晰的实验模块，例如：

```text
src/llm_hookkit/sr_propagation.py
```

职责：

- 从 tokenizer 和 model input embeddings 构造 separated embeddings
- 构造 `(0,0)`、`(s,0)`、`(s,r)` 和插值路径点
- 对每个路径点运行 activation capture
- 计算 S/R deltas 和 metrics
- 返回结构化 result object

### 11.2 建议新增示例 CLI

添加一个 example runner，例如：

```text
examples/run_sr_propagation_cli.py
```

职责：

- 加载 tokenizer 和 model
- 解析 prompt、subject、target、model、device、dtype、hook locations 和 output directory
- 执行实验
- 保存 tensors 和 summary JSON

Shell wrapper 可后续添加：

```text
scripts/run_sr_propagation.sh
```

### 11.3 建议数据类

潜在 public 或 semi-public 结构：

- `SeparatedPrompt`
- `EmbeddingPathPoint`
- `SRPropagationConfig`
- `SRPropagationResult`
- `SRPropagationRunner`

除使用模型 input embedding layer 的 embedding-construction helper 外，这些结构应尽量不依赖特定模型架构。

### 11.4 兼容性约束

- 不破坏现有 `IntegratedGradientsRunner` 或 `AttributionPatchingRunner` API。
- 不在核心模块中硬编码 GPT2 层名。
- 将模型家族默认值放在 examples 或 config presets 中。
- 将生成输出保存在 `runs/` 下。
- 如果 plotting dependencies 不可用，绘图应为可选功能。

## 12. 验证计划

### 12.1 单元测试

使用一个带 embedding layer 和简单 residual-like blocks 的 tiny torch model。

测试应验证：

- Subject 和 relation token masks 选择了预期 embedding positions
- `(0,0)`、`(s,0)` 和 `(s,r)` 具有预期的 zero/nonzero positions
- 捕获到的 activations 按 matched module names 作为 key
- 在 linear tiny model 中，`Delta h^(s)` 和 `Delta h^(r)` 与手算值一致
- Norm 和 cosine metrics 具有预期 shapes 和 values
- Finite-difference velocity 和 acceleration 在线性与非线性 toy paths 上行为正确

### 12.2 Smoke Test

在 CPU 上运行最小 GPT2 实验：

```bash
PYTHONPATH=src python examples/run_sr_propagation_cli.py \
  --model gpt2 \
  --prompt "The capital of France is" \
  --subject "France" \
  --target-text " Paris" \
  --hook-pattern "transformer.h.[0-9]+$" \
  --tensor-path 0 \
  --target-position last \
  --alphas 0,0.25,0.5,0.75,1 \
  --output-dir runs/sr_demo
```

### 12.3 仓库检查

实现后运行：

```bash
python -m compileall src tests examples
bash -n scripts/run_integrated_gradients.sh scripts/run_attribution_patching.sh scripts/run_attribution_sweep.sh
python -m pytest -q
```

如果 `pytest` 不可用，则按 `AGENTS.md` 中说明，手动导入并执行相关测试函数。

## 13. 开放设计问题

实现前应解决这些问题：

1. 零 embedding 是否也意味着 positional embeddings 置零，还是模型内部仍应加入正常 positional embeddings？
2. Relation positions 是否包含周围语法 token，例如 `The capital of` 和 `is`，还是只包含非 subject 的事实关系短语？
3. Baseline `(0,0)` 是否保留原 attention mask，还是也改变 attention visibility？推荐默认保留原 attention mask。
4. Metrics 默认对所有 token positions 计算，还是大模型运行默认只计算选定位置以节省内存？
5. PCA 和 principal angles 是否在一次运行中聚合多个 prompts，还是 single-prompt run 只生成 trajectory metrics？
6. 对不干净暴露 `inputs_embeds` 的模型，框架应提供 adapter，还是将其标记为该实验不支持？

## 14. 第一阶段实现里程碑

最小可用实现应支持：

- 带 `inputs_embeds` 的 HuggingFace causal LM
- 单个 prompt 加 subject 字符串
- 通过可配置 `ActivationLocation` 捕获 residual stream
- 三个核心路径点：`(0,0)`、`(s,0)`、`(s,r)`
- Subject 和 relation delta 计算
- Norm heatmap 数据
- Cosine curve 数据
- 沿 subject 和 relation paths 的插值
- Velocity 和 acceleration summaries
- 输出 tensor 和 summary JSON 到 `runs/`

Principal angles 和 plotting 可作为第二阶段功能。
