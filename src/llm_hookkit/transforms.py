from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .context import HookContext


class ActivationTransform(Protocol):
    def __call__(
        self,
        ctx: HookContext,
        activation: Any,
        *,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class NoOpTransform:
    def __call__(
        self,
        ctx: HookContext,
        activation: Any,
        *,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        return activation


@dataclass(frozen=True)
class ScaleTransform:
    factor: float = 1.0

    def __call__(
        self,
        ctx: HookContext,
        activation: Any,
        *,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        factor = params.get("factor", self.factor) if params else self.factor
        return activation * factor


@dataclass(frozen=True)
class ReplaceTransform:
    value: Any

    def __call__(
        self,
        ctx: HookContext,
        activation: Any,
        *,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        return params.get("value", self.value) if params else self.value


@dataclass(frozen=True)
class ComposeTransform:
    transforms: tuple[ActivationTransform, ...]

    def __call__(
        self,
        ctx: HookContext,
        activation: Any,
        *,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        current = activation
        for transform in self.transforms:
            current = transform(
                ctx,
                current,
                module=module,
                inputs=inputs,
                output=output,
                params=params,
            )
        return current


@dataclass
class TransformRegistry:
    _factories: dict[str, Callable[..., ActivationTransform]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self._factories:
            return
        self.register("noop", lambda **kwargs: NoOpTransform())
        self.register("scale", lambda **kwargs: ScaleTransform(**kwargs))
        self.register("replace", lambda **kwargs: ReplaceTransform(**kwargs))

    def register(self, name: str, factory: Callable[..., ActivationTransform]) -> None:
        self._factories[name] = factory

    def create(self, name: str, params: Mapping[str, Any] | None = None) -> ActivationTransform:
        if name not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise KeyError(f"Unknown transform '{name}'. Available transforms: {available}")
        return self._factories[name](**dict(params or {}))


def make_activation_callback(
    transform: ActivationTransform,
    params: Mapping[str, Any] | None = None,
):
    def callback(ctx: HookContext, module: Any, inputs: tuple[Any, ...], output: Any = None):
        activation = inputs if output is None else output
        return transform(
            ctx,
            activation,
            module=module,
            inputs=inputs,
            output=output,
            params=params,
        )

    return callback
