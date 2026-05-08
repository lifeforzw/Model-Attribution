# AGENTS.md 中文版

## 项目概览

本仓库实现了一个轻量级 Python 框架，用于对已经预加载的大语言模型注册 hook，并运行内部归因实验。

包名为 `llm_hookkit`。当前重点包括：

- 在预加载的 PyTorch / Transformers 风格模型上注册 hook。
- 按模块名称、类型或谓词选择 hook 目标。
- 通过可插拔接口应用激活变换。
- 从 JSON/YAML 配置文件或 shell 参数读取 hook/模型设置。
- 运行两个归因实验：
  - 对内部激活做 Integrated Gradients。
  - Attribution Patching，使用 `(clean_activation - corrupt_activation) * clean_gradient`。

代码库有意保持小而可扩展。优先添加边界清晰的小模块，并保持现有公共 API 兼容。

## 仓库结构

```text
src/llm_hookkit/
  adapters.py      模型适配器协议和默认 Torch 适配器。
  attribution.py   Integrated Gradients 和 Attribution Patching runner。
  cli.py           配置/shell hook CLI。
  config.py        JSON/YAML 配置解析和 HookSpec 构造。
  context.py       callback 之间共享的 HookContext。
  handles.py       hook handle 分组和清理。
  manager.py       HookManager 注册和上下文管理逻辑。
  matchers.py      模块目标 matcher。
  spec.py          HookSpec 和 HookType 定义。
  transforms.py    激活变换接口和内置变换。

examples/
  attribution_experiments.py   程序化归因示例。
  pytorch_transformers_hook.py 基础 hook 示例。
  run_attribution_cli.py       shell 归因脚本共用 CLI。

scripts/
  run_integrated_gradients.sh  运行 IG 归因实验。
  run_attribution_patching.sh  运行 AtP 实验。
  run_attribution_sweep.sh     运行一个小型 attention/MLP 归因 sweep。

configs/
  example_hook.json            模型/hook 配置示例。

tests/
  test_manager.py              hook 框架行为测试。
  test_attribution.py          在 tiny torch model 上的 IG 和 AtP 数值测试。
```

## 开发设置

开发时直接使用源码树：

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

可选依赖声明在 `pyproject.toml` 中：

```bash
pip install -e ".[dev]"
pip install -e ".[transformers]"
```

当前脚本在加载真实模型时假设已经安装 PyTorch 和 Transformers。

## 验证命令

修改后优先运行这些检查：

```bash
python -m compileall src tests examples
bash -n scripts/run_integrated_gradients.sh scripts/run_attribution_patching.sh scripts/run_attribution_sweep.sh
python -m pytest -q
```

如果本地环境没有 `pytest`，仍然可以通过导入已有测试函数来手动运行测试。不要把缺失 `pytest` 视为框架失败。

## 核心 Hook 流程

主流程为：

1. 构建一个 `HookSpec`。
2. 选择一个 matcher，例如 `NameMatcher`。
3. 通过 `HookManager.for_model(model)` 注册。
4. 在 `with manager.apply(spec):` 内运行模型。
5. 上下文退出时自动移除 hook handle。

示例：

```python
from llm_hookkit import HookManager, HookSpec, HookType, NameMatcher

manager = HookManager.for_model(model)

spec = HookSpec(
    name="inspect_attention",
    hook_type=HookType.FORWARD,
    matcher=NameMatcher(r".*attn.*"),
    callback=lambda ctx, module, inputs, output: output,
)

with manager.apply(spec):
    model(**batch)
```

## 激活变换接口

使用 `ActivationTransform` 实现可复用的激活编辑。当前内置项包括：

- `NoOpTransform`
- `ScaleTransform`
- `ReplaceTransform`
- `ComposeTransform`

新增激活修改方法时，在 `transforms.py` 中添加 transform，或通过 `TransformRegistry` 注册。尽量让 transform 不依赖特定模型家族。

## 配置和 Shell 输入

配置文件由 `src/llm_hookkit/config.py` 解析。

支持的配置格式：

- `.json`
- 安装 PyYAML 时支持 `.yaml` / `.yml`

示例：

```bash
PYTHONPATH=src python -m llm_hookkit.cli --config configs/example_hook.json --dry-run
```

Shell hook 示例：

```bash
PYTHONPATH=src python -m llm_hookkit.cli \
  --model-name gpt2 \
  --hook 'name=mlp_scale,type=forward,pattern=.*mlp.*,transform=scale,param.factor=0.5' \
  --dry-run
```

## 归因实验

归因代码位于 `src/llm_hookkit/attribution.py`。

使用 `ActivationLocation` 定义：

- 要 hook 哪些模块：`matcher`
- 使用哪种 hook 类型：`hook_type`
- 如何索引模块输出：`tensor_path`
- 返回哪个切片：`attribution_slice`

两个归因 runner 都需要 `score_fn(output)`，它必须返回一个标量 tensor。这个 score function 定义归因目标，例如最终序列位置上某个 token 的 logit。

Integrated Gradients：

```python
from llm_hookkit import ActivationLocation, IntegratedGradientsRunner, ModelCall, NameMatcher

location = ActivationLocation(
    matcher=NameMatcher(r".*mlp.*"),
    attribution_slice=(slice(None), -1, slice(None)),
)

result = IntegratedGradientsRunner(
    model=model,
    location=location,
    score_fn=score_fn,
    steps=32,
).run(ModelCall(kwargs=batch))
```

Attribution Patching：

```python
from llm_hookkit import AttributionPatchingRunner, ModelCall

result = AttributionPatchingRunner(
    model=model,
    location=location,
    score_fn=score_fn,
).run(
    clean_inputs=ModelCall(kwargs=clean_batch),
    corrupt_inputs=ModelCall(kwargs=corrupt_batch),
)
```

对于 AtP，clean 和 corrupt 输入通常应该有匹配的 token 长度，以便激活 tensor 对齐。

## Shell 实验脚本

运行默认示例：

```bash
bash scripts/run_integrated_gradients.sh
bash scripts/run_attribution_patching.sh
bash scripts/run_attribution_sweep.sh
```

常用环境变量：

- `MODEL`：HuggingFace 模型名或本地模型路径。
- `PROMPT`：IG 的 prompt。
- `CLEAN_PROMPT`：AtP 的 clean 输入。
- `CORRUPT_PROMPT`：AtP 的 corrupt 输入。
- `TARGET_TEXT`：作为归因目标的文本，使用其第一个 token。
- `TARGET_TOKEN_ID`：显式目标 token id，会覆盖 `TARGET_TEXT`。
- `HOOK_PATTERN`：目标模块名称的正则表达式。
- `HOOK_TYPE`：hook 类型，通常是 `forward`。
- `TENSOR_PATH`：进入模块输出的逗号分隔路径，例如 `0`。
- `TARGET_POSITION`：token 索引或 `last`。
- `STEPS`：IG 步数。
- `DEVICE`：`auto`、`cpu`、`cuda`、`mps` 或显式 torch device。
- `DTYPE`：`auto`、`float32`、`float16` 或 `bfloat16`。
- `OUTPUT_DIR`：`.pt` 和 `.summary.json` 输出目录。
- `RUN_NAME`：输出文件名前缀。

示例：

```bash
MODEL=gpt2 \
PROMPT="The capital of France is" \
TARGET_TEXT=" Paris" \
HOOK_PATTERN='.*mlp.*' \
STEPS=32 \
OUTPUT_DIR=runs/ig_demo \
bash scripts/run_integrated_gradients.sh
```

## 编码指南

- 除非明确要求破坏性变更，否则保持公共 API 向后兼容。
- 优先使用小型、可组合的类，而不是框架级全局状态。
- 优先使用现有抽象：`HookManager`、`HookSpec`、`Matcher`、`ActivationTransform` 和 `ActivationLocation`。
- 当正则 matcher 或 tensor path 能表达目标时，不要硬编码特定模型架构。
- 将归因目标语义保留在 `score_fn` 中；不要把 token/logit 选择写死到核心 runner 里。
- 将生成的实验输出保存在 `runs/` 下，该目录已被 git 忽略。
- 除非文件已经使用非 ASCII，或用户可见文本需要，否则源码中使用 ASCII。

## Git 说明

仓库 remote 预期为：

```text
git@github.com:lifeforzw/Model-Attribution.git
```

提交前检查：

```bash
git status --short
```

避免提交生成缓存，例如 `__pycache__/`、`.pytest_cache/` 或 `runs/`。
