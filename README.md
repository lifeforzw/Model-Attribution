# LLM HookKit

一个用于对“已经预加载的大模型”进行灵活 hook 操作的轻量 Python 框架。核心设计不绑定具体模型库，通过 `Adapter` 抽象支持 PyTorch，也方便扩展到其他推理框架。

## 特性

- 对预加载模型注册 forward pre hook、forward hook、backward hook。
- 支持按模块名、正则、类型、谓词函数筛选目标模块。
- 支持上下文管理，自动安装和卸载 hook。
- hook 逻辑用插件式 `HookSpec` 描述，便于组合和复用。
- 默认提供 PyTorch 适配器，同时核心接口可扩展。

## 快速使用

```python
from llm_hookkit import HookManager, HookSpec, HookType, NameMatcher

# model 是你已经加载好的 PyTorch / Transformers 模型
manager = HookManager.for_model(model)

def inspect_output(ctx, module, inputs, output):
    print(ctx.module_name, type(module).__name__)
    return output

spec = HookSpec(
    name="inspect_attn",
    hook_type=HookType.FORWARD,
    matcher=NameMatcher(r".*attn.*"),
    callback=inspect_output,
)

with manager.apply(spec):
    result = model(**batch)
```

## 修改中间激活

```python
from llm_hookkit import HookSpec, HookType, NameMatcher

def scale_output(ctx, module, inputs, output):
    return output * 0.5

spec = HookSpec(
    name="scale_mlp",
    hook_type=HookType.FORWARD,
    matcher=NameMatcher(r".*mlp.*"),
    callback=scale_output,
)

handle_group = manager.register(spec)
try:
    model(**batch)
finally:
    handle_group.remove()
```

也可以用内置 transform 接口写成：

```python
from llm_hookkit import HookSpec, HookType, NameMatcher, ScaleTransform, make_activation_callback

spec = HookSpec(
    name="scale_mlp",
    hook_type=HookType.FORWARD,
    matcher=NameMatcher(r".*mlp.*"),
    callback=make_activation_callback(ScaleTransform(factor=0.5)),
)
```

## 通过外部配置输入

JSON 示例：

```json
{
  "model": {
    "provider": "transformers",
    "name_or_path": "gpt2"
  },
  "hooks": [
    {
      "name": "scale_attention_output",
      "type": "forward",
      "matcher": {
        "kind": "name",
        "pattern": "(attn|attention)"
      },
      "transform": {
        "kind": "scale",
        "params": {
          "factor": 0.5
        }
      }
    }
  ]
}
```

读取配置并生成 specs：

```python
from llm_hookkit.config import load_config, specs_from_config

config = load_config("configs/example_hook.json")
specs = specs_from_config(config)

with HookManager.for_model(model).apply(specs):
    model(**batch)
```

## 通过 Shell 输入

开发目录内可以先用：

```bash
PYTHONPATH=src python -m llm_hookkit.cli --config configs/example_hook.json --dry-run
```

安装后可直接使用 console script：

```bash
llm-hookkit \
  --model-name gpt2 \
  --hook 'name=mlp_scale,type=forward,pattern=.*mlp.*,transform=scale,param.factor=0.5' \
  --dry-run
```

也可以从配置文件读取，并用 shell 参数覆盖模型：

```bash
llm-hookkit --config configs/example_hook.json --model-name /path/to/local/model --dry-run
```

## 扩展适配器

实现 `ModelAdapter` 协议即可：

```python
from llm_hookkit.adapters import ModelAdapter

class MyAdapter(ModelAdapter):
    def iter_modules(self, model):
        ...

    def register_hook(self, module, hook_type, callback):
        ...
```

## 内部归因实验

框架提供两个 PyTorch autograd 归因实验：

- `IntegratedGradientsRunner`：对指定模块激活计算积分梯度。
- `AttributionPatchingRunner`：计算 AtP，默认使用 `(clean_activation - corrupt_activation) * clean_gradient`。

两者都需要传入 `score_fn`，它把模型输出映射成你要解释的标量目标，例如最后一个 token 上某个目标 token 的 logit。

```python
from llm_hookkit import (
    ActivationLocation,
    IntegratedGradientsRunner,
    ModelCall,
    NameMatcher,
)

def score_fn(output):
    logits = output.logits if hasattr(output, "logits") else output
    return logits[:, -1, target_token_id].sum()

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

print(result.values)
```

AtP 示例：

```python
from llm_hookkit import AttributionPatchingRunner

result = AttributionPatchingRunner(
    model=model,
    location=location,
    score_fn=score_fn,
).run(
    clean_inputs=ModelCall(kwargs=clean_batch),
    corrupt_inputs=ModelCall(kwargs=corrupt_batch),
)
```

`ActivationLocation.tensor_path` 可用于从模块输出的 tuple/list/dict 中取出具体 tensor；`attribution_slice` 用于只返回指定 token 位置、batch 位置或 hidden 维度。

## 实验脚本

仓库提供了三类 shell 脚本，默认用 `gpt2` 和最后一个 token 位置做示例：

```bash
bash scripts/run_integrated_gradients.sh
bash scripts/run_attribution_patching.sh
bash scripts/run_attribution_sweep.sh
```

常用参数可以用环境变量覆盖：

```bash
MODEL=gpt2 \
PROMPT="The capital of France is" \
TARGET_TEXT=" Paris" \
HOOK_PATTERN='.*mlp.*' \
STEPS=32 \
OUTPUT_DIR=runs/ig_demo \
bash scripts/run_integrated_gradients.sh
```

AtP：

```bash
MODEL=gpt2 \
CLEAN_PROMPT="The capital of France is" \
CORRUPT_PROMPT="The capital of Italy is" \
TARGET_TEXT=" Paris" \
HOOK_PATTERN='.*attn.*' \
OUTPUT_DIR=runs/atp_demo \
bash scripts/run_attribution_patching.sh
```

脚本会保存两个文件：

- `*.pt`：完整 `AttributionResult`，包含 attribution values、activations、gradients。
- `*.summary.json`：每个命中模块的 shape、sum、mean_abs、L2 norm。

AtP 的 clean/corrupt 输入应尽量保持 token 长度一致；如果长度不同，目标模块激活的逐元素差值可能无法计算。
