from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol

from .context import HookContext
from .matchers import Matcher


class HookType(str, Enum):
    FORWARD_PRE = "forward_pre"
    FORWARD = "forward"
    BACKWARD = "backward"


class HookCallback(Protocol):
    def __call__(
        self,
        ctx: HookContext,
        module: Any,
        inputs: tuple[Any, ...],
        output: Optional[Any] = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class HookSpec:
    name: str
    hook_type: HookType
    matcher: Matcher
    callback: HookCallback
    enabled: bool = True
    priority: int = 100
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def should_apply(self, module_name: str, module: Any) -> bool:
        return self.enabled and self.matcher.matches(module_name, module)
