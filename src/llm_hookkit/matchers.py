from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol


class Matcher(Protocol):
    def matches(self, module_name: str, module: Any) -> bool:
        ...


@dataclass(frozen=True)
class AnyMatcher:
    def matches(self, module_name: str, module: Any) -> bool:
        return True


@dataclass(frozen=True)
class NameMatcher:
    pattern: str
    fullmatch: bool = False

    def matches(self, module_name: str, module: Any) -> bool:
        if self.fullmatch:
            return re.fullmatch(self.pattern, module_name) is not None
        return re.search(self.pattern, module_name) is not None


@dataclass(frozen=True)
class TypeMatcher:
    module_type: type | tuple[type, ...]

    def matches(self, module_name: str, module: Any) -> bool:
        return isinstance(module, self.module_type)


@dataclass(frozen=True)
class PredicateMatcher:
    predicate: Callable[[str, Any], bool]

    def matches(self, module_name: str, module: Any) -> bool:
        return bool(self.predicate(module_name, module))


@dataclass(frozen=True)
class AndMatcher:
    matchers: Iterable[Matcher]

    def matches(self, module_name: str, module: Any) -> bool:
        return all(matcher.matches(module_name, module) for matcher in self.matchers)
