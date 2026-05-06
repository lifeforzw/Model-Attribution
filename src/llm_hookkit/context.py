from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    spec_name: str
    module_name: str
    module_index: int
    state: dict[str, Any] = field(default_factory=dict)

    def child(self, module_name: str, module_index: int) -> "HookContext":
        return HookContext(
            spec_name=self.spec_name,
            module_name=module_name,
            module_index=module_index,
            state=self.state,
        )
