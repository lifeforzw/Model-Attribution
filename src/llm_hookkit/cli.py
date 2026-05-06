from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from .config import config_from_args, load_model, specs_from_config
from .manager import HookManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-hookkit",
        description="Load hook configuration for a preloaded or CLI-loaded LLM.",
    )
    parser.add_argument("--config", help="Path to JSON/YAML hook config.")
    parser.add_argument("--model-provider", help="Model provider, currently transformers.")
    parser.add_argument("--model-name", help="Model name or local path override.")
    parser.add_argument(
        "--hook",
        action="append",
        default=[],
        help=(
            "Add a hook from shell, e.g. "
            "'name=mlp,type=forward,pattern=.*mlp.*,transform=scale,param.factor=0.5'"
        ),
    )
    parser.add_argument("--hook-name", default="shell_hook")
    parser.add_argument("--hook-type", default="forward")
    parser.add_argument("--hook-pattern", help="Regex module-name matcher for one shell hook.")
    parser.add_argument("--transform-kind", default="noop")
    parser.add_argument(
        "--transform-param",
        action="append",
        default=[],
        type=_parse_param,
        help="Transform parameter as key=value. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config/spec summary without loading a model.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load model through configured provider and register hooks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    specs = specs_from_config(config)

    if args.dry_run or not args.load_model:
        print(json.dumps(_summary(config, specs), indent=2, ensure_ascii=False))
        return 0

    model = load_model(config.model)
    manager = HookManager.for_model(model)
    handles = manager.register_many(specs)
    print(f"Loaded {config.model.name_or_path}; registered {len(handles.handles)} hooks.")
    return 0


def _summary(config, specs):
    return {
        "model": asdict(config.model),
        "hooks": [
            {
                "name": spec.name,
                "hook_type": spec.hook_type.value,
                "enabled": spec.enabled,
                "priority": spec.priority,
                "metadata": dict(spec.metadata),
            }
            for spec in specs
        ],
    }


def _parse_param(value: str):
    from .config import parse_scalar

    key, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected key=value.")
    return key, parse_scalar(raw_value)


if __name__ == "__main__":
    raise SystemExit(main())
