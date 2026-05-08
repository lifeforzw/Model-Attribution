# AGENTS.md

## Project Overview

This repository implements a lightweight Python framework for hooking preloaded large language models and running internal attribution experiments.

The package is named `llm_hookkit`. Its current focus is:

- Registering hooks on preloaded PyTorch / Transformers-style models.
- Selecting hook targets by module name, type, or predicate.
- Applying activation transforms through a pluggable interface.
- Reading hook/model settings from JSON/YAML config files or shell arguments.
- Running two attribution experiments:
  - Integrated Gradients over internal activations.
  - Attribution Patching, using `(clean_activation - corrupt_activation) * clean_gradient`.

The codebase is intentionally small and extensible. Prefer adding narrow modules and keeping the existing public API compatible.

## Repository Layout

```text
src/llm_hookkit/
  adapters.py      Model adapter protocol and default Torch adapter.
  attribution.py   Integrated gradients and attribution patching runners.
  cli.py           Config/shell hook CLI.
  config.py        JSON/YAML config parsing and HookSpec construction.
  context.py       HookContext shared across callbacks.
  handles.py       Hook handle grouping and cleanup.
  manager.py       HookManager registration and context manager logic.
  matchers.py      Module target matchers.
  spec.py          HookSpec and HookType definitions.
  transforms.py    Activation transform interface and built-in transforms.

examples/
  attribution_experiments.py   Programmatic attribution examples.
  pytorch_transformers_hook.py Basic hook examples.
  run_attribution_cli.py       Shared CLI for shell attribution scripts.

scripts/
  run_integrated_gradients.sh  Run IG attribution experiment.
  run_attribution_patching.sh  Run AtP experiment.
  run_attribution_sweep.sh     Run a small attention/MLP attribution sweep.

configs/
  example_hook.json            Example model/hook config.

tests/
  test_manager.py              Hook framework behavior tests.
  test_attribution.py          IG and AtP numeric tests on a tiny torch model.
```

## Development Setup

Use the source tree directly during development:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Optional dependencies are declared in `pyproject.toml`:

```bash
pip install -e ".[dev]"
pip install -e ".[transformers]"
```

The current scripts assume PyTorch and Transformers are installed when loading real models.

## Verification Commands

Prefer these checks after changes:

```bash
python -m compileall src tests examples
bash -n scripts/run_integrated_gradients.sh scripts/run_attribution_patching.sh scripts/run_attribution_sweep.sh
python -m pytest -q
```

If `pytest` is unavailable in the local environment, the existing tests can still be manually exercised by importing their test functions. Do not treat missing `pytest` as a framework failure.

## Core Hook Flow

The main flow is:

1. Build a `HookSpec`.
2. Choose a matcher such as `NameMatcher`.
3. Register through `HookManager.for_model(model)`.
4. Run the model inside `with manager.apply(spec):`.
5. Hook handles are automatically removed on context exit.

Example:

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

## Activation Transform Interface

Use `ActivationTransform` implementations for reusable activation edits. Built-ins currently include:

- `NoOpTransform`
- `ScaleTransform`
- `ReplaceTransform`
- `ComposeTransform`

For new activation modification methods, add a transform in `transforms.py` or register it through `TransformRegistry`. Keep transforms independent from a specific model family where possible.

## Config And Shell Inputs

Config files are parsed by `src/llm_hookkit/config.py`.

Supported config formats:

- `.json`
- `.yaml` / `.yml` when PyYAML is installed

Example:

```bash
PYTHONPATH=src python -m llm_hookkit.cli --config configs/example_hook.json --dry-run
```

Shell hook example:

```bash
PYTHONPATH=src python -m llm_hookkit.cli \
  --model-name gpt2 \
  --hook 'name=mlp_scale,type=forward,pattern=.*mlp.*,transform=scale,param.factor=0.5' \
  --dry-run
```

## Attribution Experiments

Attribution code lives in `src/llm_hookkit/attribution.py`.

Use `ActivationLocation` to define:

- Which modules to hook: `matcher`
- Which hook type to use: `hook_type`
- How to index into module output: `tensor_path`
- Which slice to return: `attribution_slice`

Both attribution runners require a `score_fn(output)` that returns a scalar tensor. This score function defines the attribution target, for example one token logit at the final sequence position.

Integrated Gradients:

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

Attribution Patching:

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

For AtP, clean and corrupt inputs should normally have matching token lengths so activation tensors align.

## Shell Experiment Scripts

Run default examples:

```bash
bash scripts/run_integrated_gradients.sh
bash scripts/run_attribution_patching.sh
bash scripts/run_attribution_sweep.sh
```

Common environment variables:

- `MODEL`: HuggingFace model name or local model path.
- `PROMPT`: prompt for IG.
- `CLEAN_PROMPT`: clean input for AtP.
- `CORRUPT_PROMPT`: corrupt input for AtP.
- `TARGET_TEXT`: target text whose first token is used as the attribution target.
- `TARGET_TOKEN_ID`: explicit target token id, overrides `TARGET_TEXT`.
- `HOOK_PATTERN`: regex for target module names.
- `HOOK_TYPE`: hook type, usually `forward`.
- `TENSOR_PATH`: comma-separated path into module output, such as `0`.
- `TARGET_POSITION`: token index or `last`.
- `STEPS`: IG steps.
- `DEVICE`: `auto`, `cpu`, `cuda`, `mps`, or explicit torch device.
- `DTYPE`: `auto`, `float32`, `float16`, or `bfloat16`.
- `OUTPUT_DIR`: directory for `.pt` and `.summary.json` outputs.
- `RUN_NAME`: output filename prefix.

Example:

```bash
MODEL=gpt2 \
PROMPT="The capital of France is" \
TARGET_TEXT=" Paris" \
HOOK_PATTERN='.*mlp.*' \
STEPS=32 \
OUTPUT_DIR=runs/ig_demo \
bash scripts/run_integrated_gradients.sh
```

## Coding Guidelines

- Keep public APIs backwards-compatible unless a breaking change is explicitly requested.
- Prefer small, composable classes over framework-wide global state.
- Use existing abstractions first: `HookManager`, `HookSpec`, `Matcher`, `ActivationTransform`, and `ActivationLocation`.
- Do not hard-code a specific model architecture when a regex matcher or tensor path can express the target.
- Keep attribution target semantics in `score_fn`; do not bake token/logit choices into the core runners.
- Keep generated experiment outputs under `runs/`, which is ignored by git.
- Use ASCII in source files unless a file already uses non-ASCII or user-facing text requires it.

## Git Notes

The repository remote is expected to be:

```text
git@github.com:lifeforzw/Model-Attribution.git
```

Before committing, check:

```bash
git status --short
```

Avoid committing generated caches such as `__pycache__/`, `.pytest_cache/`, or `runs/`.
