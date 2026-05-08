# S/R Separated Propagation Experiment Design

## 1. Background

Most Transformer knowledge editing and knowledge representation studies analyze a factual prompt as one inseparable input. This experiment separates the prompt into:

- Subject, abbreviated as `S`
- Relation, abbreviated as `R`

The core question is how information propagates when only the subject embedding or only the relation embedding is injected into the model. The experiment should reveal whether the subject mainly behaves like a retrieval or addressing signal, whether the relation mainly behaves like a routing or composition signal, and whether knowledge retrieval contains a middle-layer phase transition.

The target repository is `llm_hookkit`, a small hook framework for preloaded PyTorch and Transformers-style models. The design should reuse the existing abstractions where possible:

- `HookManager` for hook registration and cleanup
- `HookSpec` and `HookType` for capture or replacement hooks
- `NameMatcher`, `TypeMatcher`, and predicate matchers for target module selection
- `ActivationLocation` for module, tensor path, and token-slice selection
- `AttributionResult`-style containers for tensors and metadata

This document is a design requirement only. It does not prescribe implementation code.

## 2. Research Goals

The experiment constructs staged embedding input paths and measures how activation differences propagate through layers and token positions.

Primary paths:

1. Subject injection path: `(0, 0) -> (s, 0)`
2. Relation injection path after subject injection: `(s, 0) -> (s, r)`

For every tracked layer or module, compute:

```text
Delta h_l^(s) = h_l(s, 0) - h_l(0, 0)
Delta h_l^(r) = h_l(s, r) - h_l(s, 0)
```

The experiment should analyze:

- Activation magnitude growth
- Direction changes between subject-induced and relation-induced perturbations
- Representation subspace overlap
- Layer-wise propagation dynamics
- Nonlinear trajectory bending and phase-transition-like behavior

## 3. Model Scope

Preferred initial model:

- `gpt2` or `gpt2-medium` for local validation

Scaled models for later runs:

- GPT2-XL
- LLaMA-2 family
- Qwen family
- Mistral family

The model should be decoder-only and support either `inputs_embeds` directly or an adapter path that bypasses token embedding lookup. The first implementation should assume HuggingFace causal language models because they expose tokenizer, input embeddings, hidden states, and logits in a predictable shape.

## 4. Prompt And Token Segmentation

Each factual prompt is represented as:

```text
S R
```

Example:

```text
The capital of France is
```

For the above example:

- Subject text: `France`
- Relation template: `The capital of {subject} is`
- Relation token positions: all prompt token positions except the subject token span
- Subject token positions: the token span aligned to the subject text

The experiment must store tokenization metadata because subject strings can split into multiple tokens and because tokenizer behavior differs by model family.

Required prompt metadata:

- `prompt`
- `subject_text`
- `relation_text` or `relation_template`
- `input_ids`
- `tokens`
- `subject_token_indices`
- `relation_token_indices`
- `target_position`, usually final prompt token
- `target_token_id` or `target_text`, if a score function is used

Subject span detection should initially use tokenizer-based alignment and fail loudly when the subject cannot be uniquely aligned. Later versions can support manually supplied token spans.

## 5. Embedding Path Construction

Let the full prompt embedding be:

```text
p(s, r) in R^(T x d)
```

Construct three core embedding inputs with the same sequence length and attention mask:

### 5.1 Baseline Path Point

```text
p(0, 0)
```

All token embeddings are zero. The attention mask remains equal to the full prompt attention mask so that the computation graph preserves sequence length and causal positions.

### 5.2 Subject-Only Path Point

```text
p(s, 0)
```

Subject token positions use real token embeddings. Relation token positions are zero.

### 5.3 Full Prompt Path Point

```text
p(s, r)
```

All token positions use real token embeddings.

### 5.4 Interpolated Path Points

For trajectory, velocity, and acceleration analysis, construct:

```text
p(alpha s, beta r)
```

where `alpha` and `beta` are scalar interpolation coefficients in `[0, 1]`.

Required path families:

- Subject path: `p(alpha s, 0)` for `alpha in [0, 1]`
- Relation-after-subject path: `p(s, beta r)` for `beta in [0, 1]`
- Optional 2D grid: `p(alpha s, beta r)` for interaction analysis

The first version should use evenly spaced coefficients, for example 11 or 21 points.

## 6. Activation Capture Targets

The priority target is the residual stream. Because different model families expose residual points differently, capture targets should be configurable through `ActivationLocation`.

Recommended first capture target for GPT-style models:

- Transformer block output, such as `transformer.h.<layer>`
- `hook_type=forward`
- `tensor_path=0` if the module output is tuple-like and the first item is hidden states

Optional capture targets:

- Attention output
- MLP output
- Pre-block residual
- Post-block residual

The first implementation should support a list of `ActivationLocation` entries so the same run can capture residual, attention, and MLP outputs if configured. The default should remain residual-only to control memory.

For each path point and each capture location, record:

```text
h_l(path_point)
```

where the tensor shape is usually:

```text
batch x tokens x hidden_dim
```

The experiment should run under `torch.no_grad()` unless a score-gradient analysis is explicitly requested.

## 7. Core Metrics

### 7.1 Activation Norm

Compute:

```text
||Delta h_l^(s)||
||Delta h_l^(r)||
```

Default norm:

- L2 norm over hidden dimension

Output shape:

```text
layers/modules x token_positions
```

Analysis questions:

- Which layers amplify subject perturbations?
- Which layers amplify relation perturbations?
- Does either path show a middle-layer surge?
- Does the final token receive stronger propagation than the subject token?

### 7.2 Direction Similarity

Compute cosine similarity:

```text
cos(Delta h_l^(s), Delta h_l^(r))
```

Default operation:

- Cosine over hidden dimension for each layer and token position

Optional reductions:

- Final-token curve across layers
- Subject-last-token curve across layers
- Mean over selected relation positions

Analysis questions:

- Are S and R perturbations orthogonal in early layers?
- Do S and R perturbations align in late layers?
- Is there a layer where the relation signal sharply turns the direction of the representation?

### 7.3 PCA Trajectory

Sample hidden states along:

```text
h_l(alpha s, 0)
h_l(s, beta r)
```

For each selected layer and token position, project trajectory points to 2D or 3D using PCA.

Default PCA fitting policy:

- Fit PCA jointly on both path families for the same layer and token position
- Use flattened hidden vectors as samples

Primary visual goal:

- Compare smooth subject propagation against relation-induced bending
- Detect possible trajectory folding or manifold attraction

UMAP and t-SNE are optional later additions, not required for the first version.

### 7.4 Subspace Principal Angle

Construct perturbation subspaces:

```text
S_l = span({Delta h_l^(s) samples})
R_l = span({Delta h_l^(r) samples})
```

Possible sample axes:

- Multiple prompts from the same knowledge category
- Multiple token positions within a prompt
- Multiple interpolation finite differences

Compute principal angles between `S_l` and `R_l`.

Default first version:

- Use a prompt batch and selected token positions
- Flatten each perturbation vector over hidden dimension
- Estimate low-rank bases with SVD
- Report the smallest angle and optionally the mean of top-k angles

Analysis questions:

- Which layers keep S/R subspaces separated?
- Which layers show subspace coupling?
- Does coupling happen before or after norm amplification?

### 7.5 Velocity And Acceleration

For trajectory points:

```text
h_l(alpha)
```

Estimate finite differences:

```text
v_l(alpha_i) = (h_l(alpha_{i+1}) - h_l(alpha_i)) / delta_alpha
a_l(alpha_i) = (v_l(alpha_{i+1}) - v_l(alpha_i)) / delta_alpha
```

Run this separately for:

- Subject path: `p(alpha s, 0)`
- Relation path: `p(s, beta r)`

Default scalar summaries:

- `||v_l||` over hidden dimension
- `||a_l||` over hidden dimension
- Cosine between adjacent velocities to measure trajectory turning

Analysis questions:

- Does relation injection create acceleration spikes?
- Do acceleration peaks align with norm amplification?
- Is the strongest nonlinearity located in early, middle, or late layers?

## 8. Experimental Variables

### 8.1 Layer Regions

Report results by:

- Early layers
- Middle layers
- Late layers

For small models, define regions by thirds. For larger models, allow explicit layer index lists.

### 8.2 Token Positions

Required token-position views:

- Subject last token
- Relation tokens
- Final prompt token

Optional:

- First subject token
- All tokens heatmap
- Target answer position if generation is included

### 8.3 Knowledge Types

Minimum prompt groups:

- Attribute facts, for example `The capital of France is`
- Person facts, for example `Einstein was born in`
- Long-tail facts
- Ambiguous or multi-answer facts

The first implementation can accept a JSONL prompt set. Each row should include at least:

```json
{
  "id": "capital_france",
  "prompt": "The capital of France is",
  "subject": "France",
  "target_text": " Paris",
  "knowledge_type": "attribute"
}
```

## 9. Outputs

### 9.1 Tensor Artifacts

Store outputs under `runs/`, which is already ignored by git.

Recommended files:

- `{run_name}.activations.pt`
- `{run_name}.metrics.pt`
- `{run_name}.summary.json`
- `{run_name}.config.json`

The summary JSON should include:

- Model name
- Prompt set path or inline prompt IDs
- Tokenization metadata
- Capture locations
- Interpolation coefficients
- Metric shapes
- Basic scalar summaries per layer

### 9.2 Figures

Required figures:

1. Layer-wise norm heatmap for `Delta h^(s)`
2. Layer-wise norm heatmap for `Delta h^(r)`
3. Cosine similarity curve across layers
4. PCA trajectory for selected layers and positions
5. Principal angle curve across layers
6. Velocity and acceleration curves

Recommended figure names:

- `{run_name}.norm_subject.png`
- `{run_name}.norm_relation.png`
- `{run_name}.cosine.png`
- `{run_name}.pca_layer_{layer}_pos_{position}.png`
- `{run_name}.principal_angle.png`
- `{run_name}.velocity_acceleration.png`

The first version may generate metrics without plotting if plotting dependencies are absent, but the data format should be ready for plotting.

## 10. Expected Observations And Hypotheses

### Hypothesis 1: Subject Mainly Supports Retrieval

Expected evidence:

- Subject perturbation propagates stably from early to middle layers
- Subject trajectory is comparatively smooth
- Direction consistency remains high across neighboring layers

### Hypothesis 2: Relation Mainly Supports Routing

Expected evidence:

- Relation injection causes sharper direction changes
- S/R cosine changes rapidly in specific layers
- Relation path acceleration spikes more strongly than subject path acceleration

### Hypothesis 3: Middle Layers Contain A Phase Transition

Expected evidence:

- Norm amplification peaks in middle layers
- PCA trajectory bends strongly in middle layers
- Acceleration peaks align with direction turning or subspace coupling

## 11. Integration With Current Framework

The existing framework already covers:

- Module selection through matchers
- Forward hook registration and automatic cleanup
- Activation capture at arbitrary module outputs
- Tensor-path selection for tuple/list/dict outputs
- Attribution result containers and summary conventions

The experiment likely needs new components rather than changes to existing attribution runners:

### 11.1 Proposed New Module

Add a narrow experiment module, for example:

```text
src/llm_hookkit/sr_propagation.py
```

Responsibilities:

- Build separated embeddings from tokenizer and model input embeddings
- Construct `(0,0)`, `(s,0)`, `(s,r)`, and interpolated path points
- Run activation capture for each path point
- Compute S/R deltas and metrics
- Return a structured result object

### 11.2 Proposed Example CLI

Add an example runner, for example:

```text
examples/run_sr_propagation_cli.py
```

Responsibilities:

- Load tokenizer and model
- Parse prompt, subject, target, model, device, dtype, hook locations, and output directory
- Execute the experiment
- Save tensors and summary JSON

Shell wrapper can be added later:

```text
scripts/run_sr_propagation.sh
```

### 11.3 Proposed Data Classes

Potential public or semi-public structures:

- `SeparatedPrompt`
- `EmbeddingPathPoint`
- `SRPropagationConfig`
- `SRPropagationResult`
- `SRPropagationRunner`

Keep these independent of a specific model architecture, except for the embedding-construction helper that uses the model's input embedding layer.

### 11.4 Compatibility Constraints

- Do not break the existing `IntegratedGradientsRunner` or `AttributionPatchingRunner` APIs.
- Do not hard-code GPT2 layer names in the core module.
- Keep model-family defaults in examples or config presets.
- Keep generated outputs under `runs/`.
- Make plotting optional if dependencies are unavailable.

## 12. Validation Plan

### 12.1 Unit Tests

Use a tiny torch model with an embedding layer and simple residual-like blocks.

Tests should verify:

- Subject and relation token masks select the expected embedding positions
- `(0,0)`, `(s,0)`, and `(s,r)` have expected zero/nonzero positions
- Captured activations are keyed by matched module names
- `Delta h^(s)` and `Delta h^(r)` match hand-computable values in a linear tiny model
- Norm and cosine metrics have expected shapes and values
- Finite-difference velocity and acceleration behave correctly on linear and nonlinear toy paths

### 12.2 Smoke Test

Run a minimal GPT2 experiment on CPU:

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

### 12.3 Repository Checks

After implementation, run:

```bash
python -m compileall src tests examples
bash -n scripts/run_integrated_gradients.sh scripts/run_attribution_patching.sh scripts/run_attribution_sweep.sh
python -m pytest -q
```

If `pytest` is unavailable, manually import and execute the relevant test functions as described in `AGENTS.md`.

## 13. Open Design Questions

These questions should be resolved before implementation:

1. Should zero embeddings include zero positional embeddings, or should the model still add its normal positional embeddings internally?
2. Should relation positions include surrounding syntax tokens such as `The capital of` and `is`, or only the non-subject factual relation phrase?
3. Should baseline `(0,0)` keep the original attention mask, or should it also alter attention visibility? The recommended default is to keep the original attention mask.
4. Should metrics be computed for all token positions by default, or should large-model runs default to selected positions to reduce memory?
5. Should PCA and principal angles aggregate across multiple prompts in one run, or should single-prompt runs only produce trajectory metrics?
6. For models that do not expose `inputs_embeds` cleanly, should this framework provide an adapter or mark them unsupported for this experiment?

## 14. First Implementation Milestone

The smallest useful implementation should support:

- HuggingFace causal LM with `inputs_embeds`
- Single prompt plus subject string
- Residual stream capture through a configurable `ActivationLocation`
- Three core path points: `(0,0)`, `(s,0)`, `(s,r)`
- Subject and relation delta computation
- Norm heatmap data
- Cosine curve data
- Interpolation along subject and relation paths
- Velocity and acceleration summaries
- Tensor and summary JSON output under `runs/`

Principal angles and plotting can be second milestone features if needed.
