from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from llm_hookkit import (
    ActivationLocation,
    AttributionPatchingRunner,
    HookType,
    IntegratedGradientsRunner,
    ModelCall,
    NameMatcher,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run internal attribution experiments with LLM HookKit."
    )
    parser.add_argument("--method", choices=["ig", "atp"], required=True)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name or local path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or explicit torch device.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--prompt", help="Prompt for integrated gradients.")
    parser.add_argument("--clean-prompt", help="Clean prompt for attribution patching.")
    parser.add_argument("--corrupt-prompt", help="Corrupt prompt for attribution patching.")
    parser.add_argument("--target-token-id", type=int)
    parser.add_argument(
        "--target-text",
        help="Text whose first token is used as target when --target-token-id is omitted.",
    )
    parser.add_argument(
        "--target-position",
        default="last",
        help="'last' or a token index used for score_fn and returned attribution slice.",
    )

    parser.add_argument("--hook-pattern", default=r".*mlp.*")
    parser.add_argument("--hook-type", default="forward", choices=[item.value for item in HookType])
    parser.add_argument(
        "--tensor-path",
        default="",
        help="Optional comma path into module output, e.g. '0' for tuple[0] or 'hidden_states,0'.",
    )
    parser.add_argument("--steps", type=int, default=32, help="Integrated gradients steps.")
    parser.add_argument("--output-dir", default="runs/attribution")
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_tokenizer_and_model(args)
    model.eval()
    target_token_id = resolve_target_token_id(args, tokenizer)
    token_position = parse_token_position(args.target_position)
    tensor_path = parse_tensor_path(args.tensor_path)

    def score_fn(output):
        logits = output.logits if hasattr(output, "logits") else output
        return logits[:, token_position, target_token_id].sum()

    location = ActivationLocation(
        matcher=NameMatcher(args.hook_pattern),
        hook_type=HookType(args.hook_type),
        tensor_path=tensor_path,
        attribution_slice=(slice(None), token_position, slice(None)),
        name=f"{args.method}:{args.hook_pattern}",
    )

    if args.method == "ig":
        if not args.prompt:
            raise ValueError("--prompt is required for --method ig.")
        batch = tokenize(tokenizer, args.prompt, model.device)
        result = IntegratedGradientsRunner(
            model=model,
            location=location,
            score_fn=score_fn,
            steps=args.steps,
        ).run(ModelCall(kwargs=batch))
    else:
        if not args.clean_prompt or not args.corrupt_prompt:
            raise ValueError("--clean-prompt and --corrupt-prompt are required for --method atp.")
        clean_batch = tokenize(tokenizer, args.clean_prompt, model.device)
        corrupt_batch = tokenize(tokenizer, args.corrupt_prompt, model.device)
        result = AttributionPatchingRunner(
            model=model,
            location=location,
            score_fn=score_fn,
        ).run(
            clean_inputs=ModelCall(kwargs=clean_batch),
            corrupt_inputs=ModelCall(kwargs=corrupt_batch),
        )

    run_name = args.run_name or default_run_name(args)
    tensor_path_out = output_dir / f"{run_name}.pt"
    summary_path = output_dir / f"{run_name}.summary.json"
    torch.save(result, tensor_path_out)
    summary_path.write_text(
        json.dumps(build_summary(args, result, target_token_id), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved tensors: {tensor_path_out}")
    print(f"Saved summary: {summary_path}")
    return 0


def load_tokenizer_and_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    device = resolve_device(args.device)
    model.to(device)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def tokenize(tokenizer, text: str, device) -> dict[str, torch.Tensor]:
    batch = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def resolve_target_token_id(args, tokenizer) -> int:
    if args.target_token_id is not None:
        return args.target_token_id
    if not args.target_text:
        raise ValueError("Pass --target-token-id or --target-text.")
    token_ids = tokenizer.encode(args.target_text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("--target-text produced no tokens.")
    return int(token_ids[0])


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(value: str):
    if value == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def parse_token_position(value: str) -> int:
    if value == "last":
        return -1
    return int(value)


def parse_tensor_path(value: str):
    if not value:
        return ()
    path = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            path.append(int(item))
        except ValueError:
            path.append(item)
    return tuple(path)


def default_run_name(args) -> str:
    model_name = args.model.rstrip("/").split("/")[-1].replace(" ", "_")
    pattern = args.hook_pattern.replace("*", "star").replace(".", "_").replace("/", "_")
    return f"{args.method}_{model_name}_{pattern}"


def build_summary(args, result, target_token_id: int) -> dict[str, Any]:
    module_summaries = {}
    for name, value in result.values.items():
        tensor = value.detach().float().cpu()
        module_summaries[name] = {
            "shape": list(tensor.shape),
            "sum": tensor.sum().item(),
            "mean_abs": tensor.abs().mean().item(),
            "l2": torch.linalg.vector_norm(tensor).item(),
        }
    return {
        "method": args.method,
        "model": args.model,
        "target_token_id": target_token_id,
        "target_position": args.target_position,
        "hook_pattern": args.hook_pattern,
        "hook_type": args.hook_type,
        "tensor_path": args.tensor_path,
        "steps": args.steps if args.method == "ig" else None,
        "modules": module_summaries,
        "metadata": dict(result.metadata),
    }


if __name__ == "__main__":
    raise SystemExit(main())
