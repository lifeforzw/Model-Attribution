from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .matchers import AndMatcher, AnyMatcher, Matcher, NameMatcher, TypeMatcher
from .spec import HookSpec, HookType
from .transforms import TransformRegistry, make_activation_callback


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "transformers"
    name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformConfig:
    kind: str = "noop"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatcherConfig:
    kind: str = "name"
    pattern: str | None = None
    fullmatch: bool = False
    type_path: str | None = None
    all: Sequence["MatcherConfig"] = field(default_factory=tuple)


@dataclass(frozen=True)
class HookConfig:
    name: str
    hook_type: HookType = HookType.FORWARD
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    enabled: bool = True
    priority: int = 100
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookKitConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    hooks: Sequence[HookConfig] = field(default_factory=tuple)


def load_config(path: str | Path) -> HookKitConfig:
    raw = load_raw_config(path)
    return parse_config(raw)


def load_raw_config(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: pip install llm-hookkit[yaml]"
            ) from exc
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"Unsupported config file type: {config_path.suffix}")


def parse_config(raw: Mapping[str, Any]) -> HookKitConfig:
    model_raw = raw.get("model", {})
    hooks_raw = raw.get("hooks", [])
    return HookKitConfig(
        model=ModelConfig(
            provider=model_raw.get("provider", "transformers"),
            name_or_path=model_raw.get("name_or_path") or model_raw.get("name"),
            tokenizer_name_or_path=model_raw.get("tokenizer_name_or_path"),
            kwargs=model_raw.get("kwargs", {}),
        ),
        hooks=tuple(parse_hook_config(item) for item in hooks_raw),
    )


def parse_hook_config(raw: Mapping[str, Any]) -> HookConfig:
    matcher_raw = raw.get("matcher", {})
    transform_raw = raw.get("transform", {})
    return HookConfig(
        name=raw["name"],
        hook_type=HookType(raw.get("hook_type", raw.get("type", HookType.FORWARD.value))),
        matcher=parse_matcher_config(matcher_raw),
        transform=TransformConfig(
            kind=transform_raw.get("kind", "noop"),
            params=transform_raw.get("params", {}),
        ),
        enabled=raw.get("enabled", True),
        priority=raw.get("priority", 100),
        metadata=raw.get("metadata", {}),
    )


def parse_matcher_config(raw: Mapping[str, Any]) -> MatcherConfig:
    if "all" in raw:
        return MatcherConfig(
            kind="all",
            all=tuple(parse_matcher_config(item) for item in raw["all"]),
        )
    return MatcherConfig(
        kind=raw.get("kind", "name"),
        pattern=raw.get("pattern"),
        fullmatch=raw.get("fullmatch", False),
        type_path=raw.get("type_path"),
    )


def specs_from_config(
    config: HookKitConfig,
    registry: TransformRegistry | None = None,
) -> list[HookSpec]:
    transform_registry = registry or TransformRegistry()
    specs = []
    for hook_config in config.hooks:
        transform = transform_registry.create(
            hook_config.transform.kind,
            hook_config.transform.params,
        )
        specs.append(
            HookSpec(
                name=hook_config.name,
                hook_type=hook_config.hook_type,
                matcher=matcher_from_config(hook_config.matcher),
                callback=make_activation_callback(transform, hook_config.transform.params),
                enabled=hook_config.enabled,
                priority=hook_config.priority,
                metadata=hook_config.metadata,
            )
        )
    return specs


def matcher_from_config(config: MatcherConfig) -> Matcher:
    if config.kind == "any":
        return AnyMatcher()
    if config.kind == "name":
        if not config.pattern:
            raise ValueError("Name matcher requires 'pattern'.")
        return NameMatcher(config.pattern, fullmatch=config.fullmatch)
    if config.kind == "type":
        if not config.type_path:
            raise ValueError("Type matcher requires 'type_path', e.g. torch.nn.Linear.")
        return TypeMatcher(import_object(config.type_path))
    if config.kind == "all":
        return AndMatcher(tuple(matcher_from_config(item) for item in config.all))
    raise ValueError(f"Unsupported matcher kind: {config.kind}")


def import_object(dotted_path: str) -> Any:
    module_name, _, attr_name = dotted_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def load_model(config: ModelConfig) -> Any:
    if config.provider != "transformers":
        raise ValueError(f"Unsupported model provider: {config.provider}")
    if not config.name_or_path:
        raise ValueError("Model config requires 'name_or_path'.")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Transformers loading requires optional dependencies: "
            "pip install llm-hookkit[transformers]"
        ) from exc
    return AutoModelForCausalLM.from_pretrained(config.name_or_path, **dict(config.kwargs))


def config_from_args(args: argparse.Namespace) -> HookKitConfig:
    if args.config:
        base = load_config(args.config)
    else:
        base = HookKitConfig()

    model = base.model
    if args.model_name:
        model = ModelConfig(
            provider=args.model_provider or model.provider,
            name_or_path=args.model_name,
            tokenizer_name_or_path=model.tokenizer_name_or_path,
            kwargs=model.kwargs,
        )
    elif args.model_provider:
        model = ModelConfig(
            provider=args.model_provider,
            name_or_path=model.name_or_path,
            tokenizer_name_or_path=model.tokenizer_name_or_path,
            kwargs=model.kwargs,
        )

    hooks = list(base.hooks)
    hooks.extend(parse_shell_hook(item) for item in args.hook)

    if args.hook_pattern:
        hooks.append(
            HookConfig(
                name=args.hook_name,
                hook_type=HookType(args.hook_type),
                matcher=MatcherConfig(kind="name", pattern=args.hook_pattern),
                transform=TransformConfig(
                    kind=args.transform_kind,
                    params=dict(args.transform_param),
                ),
            )
        )

    return HookKitConfig(model=model, hooks=tuple(hooks))


def parse_shell_hook(value: str) -> HookConfig:
    items = parse_key_value_list(value)
    transform_params = {
        key.removeprefix("param."): parsed
        for key, parsed in items.items()
        if key.startswith("param.")
    }
    return HookConfig(
        name=items.get("name", "shell_hook"),
        hook_type=HookType(items.get("type", items.get("hook_type", HookType.FORWARD.value))),
        matcher=MatcherConfig(
            kind=items.get("matcher", "name"),
            pattern=items.get("pattern"),
            type_path=items.get("type_path"),
            fullmatch=bool(items.get("fullmatch", False)),
        ),
        transform=TransformConfig(
            kind=items.get("transform", "noop"),
            params=transform_params,
        ),
        enabled=bool(items.get("enabled", True)),
        priority=int(items.get("priority", 100)),
    )


def parse_key_value_list(value: str) -> dict[str, Any]:
    result = {}
    for part in value.split(","):
        if not part.strip():
            continue
        key, separator, raw_value = part.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value in shell hook segment: {part}")
        result[key.strip()] = parse_scalar(raw_value.strip())
    return result


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
