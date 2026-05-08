from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from llm_hookkit import (
    ActivationLocation,
    HookType,
    NameMatcher,
    SRPropagationConfig,
    SRPropagationRunner,
    separated_prompt_from_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run S/R separated propagation experiments with LLM HookKit."
    )
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name or local path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or explicit torch device.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--subject", required=True, help="Subject text to align inside --prompt.")
    parser.add_argument("--relation-text")
    parser.add_argument("--relation-template")
    parser.add_argument("--target-token-id", type=int)
    parser.add_argument("--target-text")
    parser.add_argument("--target-position", default="last", help="'last' or a token index.")

    parser.add_argument(
        "--hook-pattern",
        default=r"transformer\.h\.[0-9]+$",
        help="Regex selecting modules to capture. Default captures GPT2 block outputs.",
    )
    parser.add_argument("--hook-type", default="forward", choices=[item.value for item in HookType])
    parser.add_argument(
        "--tensor-path",
        default="0",
        help="Optional comma path into module output, e.g. '0' for tuple[0].",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated coefficients for p(alpha s, 0).",
    )
    parser.add_argument(
        "--betas",
        default=None,
        help="Comma-separated coefficients for p(s, beta r). Defaults to --alphas.",
    )
    parser.add_argument(
        "--pca-token-positions",
        default="last",
        help="Comma-separated token positions for PCA trajectory data.",
    )
    parser.add_argument("--principal-angle-rank", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/sr_propagation")
    parser.add_argument("--run-name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_tokenizer_and_model(args)
    model.eval()

    target_position = parse_token_position(args.target_position)
    target_token_id = resolve_target_token_id(args, tokenizer)
    separated_prompt = separated_prompt_from_tokenizer(
        tokenizer,
        prompt=args.prompt,
        subject_text=args.subject,
        target_position=target_position,
        target_text=args.target_text,
        target_token_id=target_token_id,
        relation_text=args.relation_text,
        relation_template=args.relation_template,
    )
    separated_prompt = move_prompt_to_device(separated_prompt, model.device)

    tensor_path = parse_tensor_path(args.tensor_path)
    location = ActivationLocation(
        matcher=NameMatcher(args.hook_pattern),
        hook_type=HookType(args.hook_type),
        tensor_path=tensor_path,
        name="residual",
    )
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas or args.alphas)
    config = SRPropagationConfig(
        locations=(location,),
        alphas=alphas,
        betas=betas,
        pca_token_positions=parse_positions(args.pca_token_positions),
        principal_angle_rank=args.principal_angle_rank,
    )

    result = SRPropagationRunner(model=model, config=config).run(separated_prompt)

    run_name = args.run_name or default_run_name(args)
    activations_path = output_dir / f"{run_name}.activations.pt"
    metrics_path = output_dir / f"{run_name}.metrics.pt"
    summary_path = output_dir / f"{run_name}.summary.json"
    config_path = output_dir / f"{run_name}.config.json"

    torch.save(result.activations, activations_path)
    torch.save({"values": result.values, "metrics": result.metrics}, metrics_path)
    summary_path.write_text(
        json.dumps(build_summary(args, result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(build_config_dump(args, result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved activations: {activations_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved config: {config_path}")
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


def move_prompt_to_device(prompt, device):
    return type(prompt)(
        **{
            **prompt.__dict__,
            "input_ids": prompt.input_ids.to(device),
            "attention_mask": prompt.attention_mask.to(device)
            if prompt.attention_mask is not None
            else None,
        }
    )


def resolve_target_token_id(args, tokenizer) -> int | None:
    if args.target_token_id is not None:
        return args.target_token_id
    if not args.target_text:
        return None
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


def parse_positions(value: str) -> tuple[int, ...]:
    if value == "last":
        return (-1,)
    return tuple(parse_token_position(item.strip()) for item in value.split(",") if item.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) < 2:
        raise ValueError("At least two interpolation coefficients are required.")
    return values


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
    subject = args.subject.replace(" ", "_")
    return f"sr_{model_name}_{subject}"


def build_summary(args, result) -> dict[str, Any]:
    modules = {}
    for location_name, module_values in result.metrics["norm_subject"].items():
        modules[location_name] = {}
        for module_name, subject_norm in module_values.items():
            relation_norm = result.metrics["norm_relation"][location_name][module_name]
            cosine = result.metrics["cosine_subject_relation"][location_name][module_name]
            modules[location_name][module_name] = {
                "subject_norm_shape": list(subject_norm.shape),
                "relation_norm_shape": list(relation_norm.shape),
                "subject_norm_mean": subject_norm.detach().float().mean().item(),
                "relation_norm_mean": relation_norm.detach().float().mean().item(),
                "cosine_mean": cosine.detach().float().mean().item(),
                "subject_norm_max": subject_norm.detach().float().max().item(),
                "relation_norm_max": relation_norm.detach().float().max().item(),
            }
    return {
        "method": "sr_separated_propagation",
        "model": args.model,
        "hook_pattern": args.hook_pattern,
        "hook_type": args.hook_type,
        "tensor_path": args.tensor_path,
        "prompt": result.metadata["prompt"],
        "alphas": result.metadata["alphas"],
        "betas": result.metadata["betas"],
        "modules": modules,
    }


def build_config_dump(args, result) -> dict[str, Any]:
    return {
        "args": vars(args),
        "metadata": result.metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())
