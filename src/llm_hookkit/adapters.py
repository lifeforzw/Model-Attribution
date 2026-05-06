from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from .spec import HookType


class RemovableHandle(Protocol):
    def remove(self) -> None:
        ...


class ModelAdapter(Protocol):
    def iter_modules(self, model: Any) -> Iterable[tuple[str, Any]]:
        ...

    def register_hook(
        self,
        module: Any,
        hook_type: HookType,
        callback: Callable[..., Any],
    ) -> RemovableHandle:
        ...


@dataclass(frozen=True)
class TorchAdapter:
    include_root: bool = False

    def iter_modules(self, model: Any) -> Iterable[tuple[str, Any]]:
        if not hasattr(model, "named_modules"):
            raise TypeError("TorchAdapter requires a model with named_modules().")

        for name, module in model.named_modules():
            if name or self.include_root:
                yield name, module

    def register_hook(
        self,
        module: Any,
        hook_type: HookType,
        callback: Callable[..., Any],
    ) -> RemovableHandle:
        if hook_type is HookType.FORWARD_PRE:
            return module.register_forward_pre_hook(callback)
        if hook_type is HookType.FORWARD:
            return module.register_forward_hook(callback)
        if hook_type is HookType.BACKWARD:
            return module.register_full_backward_hook(callback)
        raise ValueError(f"Unsupported hook type: {hook_type}")


def infer_adapter(model: Any) -> ModelAdapter:
    if hasattr(model, "named_modules"):
        return TorchAdapter()
    raise TypeError(
        "Cannot infer model adapter. Pass an explicit adapter implementing ModelAdapter."
    )
