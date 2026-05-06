from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .manager import HookManager
from .matchers import Matcher
from .spec import HookSpec, HookType


TensorPath = Sequence[int | str]
ScoreFn = Callable[[Any], Any]
BaselineFactory = Callable[[str, Any], Any]


@dataclass(frozen=True)
class ModelCall:
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationLocation:
    matcher: Matcher
    hook_type: HookType = HookType.FORWARD
    tensor_path: TensorPath = ()
    attribution_slice: Any = None
    name: str = "attribution_location"


@dataclass(frozen=True)
class AttributionResult:
    values: Mapping[str, Any]
    activations: Mapping[str, Any] = field(default_factory=dict)
    gradients: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedGradientsRunner:
    model: Any
    location: ActivationLocation
    score_fn: ScoreFn
    steps: int = 32
    baseline: Any | BaselineFactory | None = None
    manager: HookManager | None = None

    def run(self, inputs: ModelCall) -> AttributionResult:
        torch = _require_torch()
        manager = self.manager or HookManager.for_model(self.model)
        clean_activations = capture_activations(
            self.model,
            inputs,
            self.location,
            manager=manager,
            detach=True,
        )
        baselines = {
            name: _make_baseline(self.baseline, name, activation)
            for name, activation in clean_activations.items()
        }
        grad_sums = {name: torch.zeros_like(value) for name, value in clean_activations.items()}

        for step in range(1, self.steps + 1):
            alpha = step / self.steps
            interpolated = {
                name: (baselines[name] + alpha * (activation - baselines[name]))
                .detach()
                .requires_grad_(True)
                for name, activation in clean_activations.items()
            }
            self.model.zero_grad(set_to_none=True)
            output = run_with_patched_activations(
                self.model,
                inputs,
                self.location,
                interpolated,
                manager=manager,
            )
            score = _as_scalar(self.score_fn(output))
            score.backward()
            for name, activation in interpolated.items():
                if activation.grad is None:
                    raise RuntimeError(f"No gradient collected for activation '{name}'.")
                grad_sums[name] = grad_sums[name] + activation.grad.detach()

        values = {}
        gradients = {}
        for name, activation in clean_activations.items():
            avg_grad = grad_sums[name] / self.steps
            gradients[name] = _select(avg_grad, self.location.attribution_slice)
            values[name] = _select((activation - baselines[name]) * avg_grad, self.location.attribution_slice)

        return AttributionResult(
            values=values,
            activations={
                name: _select(value, self.location.attribution_slice)
                for name, value in clean_activations.items()
            },
            gradients=gradients,
            metadata={"method": "integrated_gradients", "steps": self.steps},
        )


@dataclass
class AttributionPatchingRunner:
    model: Any
    location: ActivationLocation
    score_fn: ScoreFn
    manager: HookManager | None = None

    def run(self, clean_inputs: ModelCall, corrupt_inputs: ModelCall) -> AttributionResult:
        manager = self.manager or HookManager.for_model(self.model)
        clean = capture_activations_and_gradients(
            self.model,
            clean_inputs,
            self.location,
            self.score_fn,
            manager=manager,
        )
        corrupt_activations = capture_activations(
            self.model,
            corrupt_inputs,
            self.location,
            manager=manager,
            detach=True,
        )

        values = {}
        for name, clean_activation in clean.activations.items():
            if name not in corrupt_activations:
                raise RuntimeError(f"Missing corrupt activation for '{name}'.")
            delta = clean_activation - corrupt_activations[name]
            values[name] = _select(delta * clean.gradients[name], self.location.attribution_slice)

        return AttributionResult(
            values=values,
            activations={
                name: _select(value, self.location.attribution_slice)
                for name, value in clean.activations.items()
            },
            gradients={
                name: _select(value, self.location.attribution_slice)
                for name, value in clean.gradients.items()
            },
            metadata={"method": "attribution_patching"},
        )


def capture_activations(
    model: Any,
    inputs: ModelCall,
    location: ActivationLocation,
    *,
    manager: HookManager | None = None,
    detach: bool = True,
) -> dict[str, Any]:
    manager = manager or HookManager.for_model(model)
    captured: dict[str, Any] = {}

    def callback(ctx, module, module_inputs, output=None):
        activation = _get_activation(output if output is not None else module_inputs, location.tensor_path)
        captured[ctx.module_name] = activation.detach().clone() if detach else activation
        return output

    spec = HookSpec(
        name=f"{location.name}:capture",
        hook_type=location.hook_type,
        matcher=location.matcher,
        callback=callback,
    )
    with manager.apply(spec):
        model(*inputs.args, **dict(inputs.kwargs))
    if not captured:
        raise RuntimeError("No activations were captured. Check the matcher and hook type.")
    return captured


def capture_activations_and_gradients(
    model: Any,
    inputs: ModelCall,
    location: ActivationLocation,
    score_fn: ScoreFn,
    *,
    manager: HookManager | None = None,
) -> AttributionResult:
    manager = manager or HookManager.for_model(model)
    activations: dict[str, Any] = {}

    def callback(ctx, module, module_inputs, output=None):
        activation = _get_activation(output if output is not None else module_inputs, location.tensor_path)
        activation.retain_grad()
        activations[ctx.module_name] = activation
        return output

    spec = HookSpec(
        name=f"{location.name}:capture_grad",
        hook_type=location.hook_type,
        matcher=location.matcher,
        callback=callback,
    )
    model.zero_grad(set_to_none=True)
    with manager.apply(spec):
        output = model(*inputs.args, **dict(inputs.kwargs))
        score = _as_scalar(score_fn(output))
        score.backward()

    gradients = {}
    detached_activations = {}
    for name, activation in activations.items():
        if activation.grad is None:
            raise RuntimeError(f"No gradient collected for activation '{name}'.")
        detached_activations[name] = activation.detach().clone()
        gradients[name] = activation.grad.detach().clone()

    if not detached_activations:
        raise RuntimeError("No activations were captured. Check the matcher and hook type.")

    return AttributionResult(
        values={},
        activations=detached_activations,
        gradients=gradients,
        metadata={"method": "activation_gradient_capture"},
    )


def run_with_patched_activations(
    model: Any,
    inputs: ModelCall,
    location: ActivationLocation,
    replacements: Mapping[str, Any],
    *,
    manager: HookManager | None = None,
) -> Any:
    manager = manager or HookManager.for_model(model)

    def callback(ctx, module, module_inputs, output=None):
        if ctx.module_name not in replacements:
            return output
        container = output if output is not None else module_inputs
        replacement = replacements[ctx.module_name]
        return _set_activation(container, location.tensor_path, replacement)

    spec = HookSpec(
        name=f"{location.name}:patch",
        hook_type=location.hook_type,
        matcher=location.matcher,
        callback=callback,
    )
    with manager.apply(spec):
        return model(*inputs.args, **dict(inputs.kwargs))


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Attribution experiments require PyTorch.") from exc
    return torch


def _as_scalar(value: Any) -> Any:
    if getattr(value, "ndim", None) == 0:
        return value
    if hasattr(value, "numel") and value.numel() == 1:
        return value.reshape(())
    raise ValueError("score_fn must return a scalar tensor.")


def _make_baseline(baseline: Any | BaselineFactory | None, name: str, activation: Any) -> Any:
    torch = _require_torch()
    if baseline is None or baseline == "zero":
        return torch.zeros_like(activation)
    if callable(baseline):
        value = baseline(name, activation)
    else:
        value = baseline
    if not hasattr(value, "shape"):
        return torch.zeros_like(activation) + value
    return value.detach().clone()


def _get_activation(container: Any, path: TensorPath) -> Any:
    current = container
    for key in path:
        current = current[key]
    return current


def _set_activation(container: Any, path: TensorPath, value: Any) -> Any:
    if not path:
        return value
    key = path[0]
    if isinstance(container, tuple):
        items = list(container)
        items[key] = _set_activation(items[key], path[1:], value)
        return tuple(items)
    if isinstance(container, list):
        items = list(container)
        items[key] = _set_activation(items[key], path[1:], value)
        return items
    if isinstance(container, dict):
        items = dict(container)
        items[key] = _set_activation(items[key], path[1:], value)
        return items
    raise TypeError(f"Cannot set activation through object of type {type(container)!r}.")


def _select(value: Any, selection: Any) -> Any:
    if selection is None:
        return value
    return value[selection]
