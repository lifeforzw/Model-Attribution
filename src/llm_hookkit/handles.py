from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .adapters import RemovableHandle


@dataclass
class HookHandleGroup:
    name: str
    handles: list[RemovableHandle] = field(default_factory=list)
    removed: bool = False

    def add(self, handle: RemovableHandle) -> None:
        if self.removed:
            handle.remove()
            return
        self.handles.append(handle)

    def remove(self) -> None:
        if self.removed:
            return
        for handle in reversed(self.handles):
            handle.remove()
        self.removed = True

    def __enter__(self) -> "HookHandleGroup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    @classmethod
    def combine(cls, name: str, groups: Iterable["HookHandleGroup"]) -> "HookHandleGroup":
        combined = cls(name=name)
        for group in groups:
            combined.handles.extend(group.handles)
            group.handles.clear()
            group.removed = True
        return combined
