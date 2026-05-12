from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a shell command plan for Context-to-Temporary-FFN experiments."
    )
    parser.add_argument("--config", default="configs/temp_ffn_experiment_plan.json")
    parser.add_argument("--output", default="runs/temp_ffn/planned_commands.sh")
    parser.add_argument("--include-disabled", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    commands = list(build_commands(config, include_disabled=args.include_disabled))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_script(commands), encoding="utf-8")
    output.chmod(0o755)
    print(f"Wrote {len(commands)} commands: {output}")
    return 0


def build_commands(config: dict[str, Any], *, include_disabled: bool):
    python = config.get("python", "python")
    output_root = config.get("output_root", "runs/temp_ffn")
    text_file = config.get("text_file")

    for experiment in config.get("experiments", []):
        if not include_disabled and not experiment.get("enabled", True):
            continue
        kind = experiment["kind"]
        if kind == "random":
            yield random_command(python, output_root, experiment)
        elif kind == "memory":
            yield memory_command(python, output_root, experiment)
        elif kind == "hf_activations":
            yield hf_command(python, output_root, experiment, text_file)
        elif kind == "hf_layer_head_grid":
            for layer in experiment["layers"]:
                for head in experiment["heads"]:
                    item = {**experiment, "layer": layer, "head": head}
                    yield hf_command(python, output_root, item, text_file)
        else:
            raise ValueError(f"Unsupported experiment kind: {kind}")


def random_command(python: str, output_root: str, exp: dict[str, Any]) -> list[str]:
    return [
        python,
        "examples/run_temp_ffn_compression.py",
        "random",
        "--context-length",
        str(exp["context_length"]),
        "--head-dim",
        str(exp["head_dim"]),
        "--query-count",
        str(exp["query_count"]),
        "--memory-units",
        csv(exp["memory_units"]),
        "--steps",
        str(exp["steps"]),
        "--lr",
        str(exp.get("lr", 0.01)),
        "--batch-size",
        str(exp["batch_size"]),
        "--cosine-weight",
        str(exp.get("cosine_weight", 0.1)),
        "--device",
        exp.get("device", "auto"),
        "--dtype",
        exp.get("dtype", "float32"),
        "--seed",
        str(exp.get("seed", 13)),
        "--output-dir",
        f"{output_root}/{exp['name']}",
        "--run-name",
        exp["name"],
        "--init",
        exp.get("init", "sample"),
    ]


def memory_command(python: str, output_root: str, exp: dict[str, Any]) -> list[str]:
    return [
        python,
        "examples/run_temp_ffn_compression.py",
        "memory",
        "--context-lengths",
        csv(exp["context_lengths"]),
        "--memory-units",
        csv(exp["memory_units"]),
        "--layers",
        str(exp["layers"]),
        "--heads",
        str(exp["heads"]),
        "--head-dim",
        str(exp["head_dim"]),
        "--output-dir",
        f"{output_root}/{exp['name']}",
        "--run-name",
        exp["name"],
    ]


def hf_command(
    python: str,
    output_root: str,
    exp: dict[str, Any],
    default_text_file: str | None,
) -> list[str]:
    layer = exp["layer"]
    head = exp["head"]
    run_name = exp.get("run_name") or f"{exp['name']}_layer{layer}_head{head}"
    command = [
        python,
        "examples/run_temp_ffn_compression.py",
        "hf-activations",
        "--model",
        exp["model"],
        "--layer",
        str(layer),
        "--head",
        str(head),
        "--qkv-layout",
        exp.get("qkv_layout", "gpt2"),
        "--context-length",
        str(exp["context_length"]),
        "--query-count",
        str(exp["query_count"]),
        "--memory-units",
        csv(exp["memory_units"]),
        "--steps",
        str(exp["steps"]),
        "--lr",
        str(exp.get("lr", 0.01)),
        "--batch-size",
        str(exp["batch_size"]),
        "--cosine-weight",
        str(exp.get("cosine_weight", 0.1)),
        "--device",
        exp.get("device", "auto"),
        "--dtype",
        exp.get("dtype", "auto"),
        "--output-dir",
        f"{output_root}/{exp['name']}",
        "--run-name",
        run_name,
        "--init",
        exp.get("init", "sample"),
    ]
    pattern = exp.get("qkv_module_pattern")
    if pattern:
        command.extend(["--qkv-module-pattern", pattern.format(layer=layer)])
    text_file = exp.get("text_file") or default_text_file
    if text_file:
        command.extend(["--text-file", text_file])
    if exp.get("trust_remote_code"):
        command.append("--trust-remote-code")
    return command


def render_script(commands: list[list[str]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
        'cd "$ROOT_DIR"',
        'export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"',
        "",
        "# This file is generated. Inspect it, then run selected commands manually.",
        "",
    ]
    for index, command in enumerate(commands, start=1):
        lines.append(f"# [{index}]")
        lines.append(" ".join(shlex.quote(part) for part in command))
        lines.append("")
    return "\n".join(lines)


def csv(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
