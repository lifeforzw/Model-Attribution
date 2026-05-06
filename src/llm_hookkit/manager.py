from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .adapters import ModelAdapter, infer_adapter
from .context import HookContext
from .handles import HookHandleGroup
from .spec import HookSpec, HookType


@dataclass
class HookManager:
    model: Any
    adapter: ModelAdapter = field(default=None)

    def __post_init__(self) -> None:
        if self.adapter is None:
            self.adapter = infer_adapter(self.model)

    @classmethod
    def for_model(cls, model: Any, adapter: ModelAdapter | None = None) -> "HookManager":
        return cls(model=model, adapter=adapter)

    def register(self, spec: HookSpec) -> HookHandleGroup:
        group = HookHandleGroup(name=spec.name)
        root_ctx = HookContext(spec_name=spec.name, module_name="", module_index=-1)

        for index, (module_name, module) in enumerate(self.adapter.iter_modules(self.model)):
            if not spec.should_apply(module_name, module):
                continue
            ctx = root_ctx.child(module_name=module_name, module_index=index)
            group.add(self.adapter.register_hook(module, spec.hook_type, self._wrap(spec, ctx)))

        return group

    def register_many(self, specs: Sequence[HookSpec]) -> HookHandleGroup:
        ordered_specs = sorted(specs, key=lambda spec: spec.priority)
        groups = [self.register(spec) for spec in ordered_specs]
        return HookHandleGroup.combine("combined", groups)

    @contextmanager
    def apply(self, specs: HookSpec | Sequence[HookSpec]) -> Iterator[HookHandleGroup]:
        if isinstance(specs, HookSpec):
            group = self.register(specs)
        else:
            group = self.register_many(specs)
        try:
            yield group
        finally:
            group.remove()

    def _wrap(self, spec: HookSpec, ctx: HookContext):
        if spec.hook_type is HookType.FORWARD_PRE:
            def forward_pre(module, inputs):
                return spec.callback(ctx, module, inputs)

            return forward_pre

        if spec.hook_type is HookType.FORWARD:
            def forward(module, inputs, output):
                return spec.callback(ctx, module, inputs, output)

            return forward

        if spec.hook_type is HookType.BACKWARD:
            def backward(module, grad_input, grad_output):
                return spec.callback(ctx, module, grad_input, grad_output)

            return backward

        raise ValueError(f"Unsupported hook type: {spec.hook_type}")
