from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_hookkit import (
    HookManager,
    HookSpec,
    HookType,
    MemoryCost,
    NameMatcher,
    generate_random_qkv,
    run_compression_sweep,
)
from llm_hookkit.temporary_ffn import summarize_results


DEFAULT_TEXT = (
    "Long-context language models must remember facts, entities, and topic structure "
    "over many tokens. This synthetic paragraph is repeated to create enough context "
    "for temporary FFN compression experiments. "
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Context-to-Temporary-FFN compression experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser("random", help="Run random QKV simulation.")
    add_training_args(random_parser)
    random_parser.add_argument("--context-length", type=int, default=4096)
    random_parser.add_argument("--head-dim", type=int, default=64)
    random_parser.add_argument("--query-count", type=int, default=2048)
    random_parser.add_argument(
        "--eval-query-count",
        type=int,
        default=0,
        help="Optional held-out query count for evaluating the same context-specific temporary FFN.",
    )
    random_parser.add_argument("--device", default="auto")
    random_parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    random_parser.add_argument("--seed", type=int, default=13)
    random_parser.add_argument("--output-dir", default="runs/temp_ffn/random")
    random_parser.add_argument("--run-name", default="random_qkv")
    random_parser.set_defaults(func=run_random)

    hf_parser = subparsers.add_parser(
        "hf-activations",
        help="Extract packed QKV activations from a HuggingFace causal LM and fit temporary FFNs.",
    )
    add_training_args(hf_parser)
    hf_parser.add_argument("--model", default="gpt2")
    hf_parser.add_argument("--device", default="auto")
    hf_parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    hf_parser.add_argument("--trust-remote-code", action="store_true")
    hf_parser.add_argument("--text")
    hf_parser.add_argument("--text-file")
    hf_parser.add_argument("--repeat-text", action=argparse.BooleanOptionalAction, default=True)
    hf_parser.add_argument("--context-length", type=int, default=512)
    hf_parser.add_argument("--query-count", type=int, default=128)
    hf_parser.add_argument(
        "--eval-query-count",
        type=int,
        default=0,
        help="Optional held-out future-token query count for evaluating the fitted context-specific temporary FFN.",
    )
    hf_parser.add_argument("--layer", type=int, default=6)
    hf_parser.add_argument("--head", type=int, default=0)
    hf_parser.add_argument(
        "--qkv-module-pattern",
        help="Regex for the packed QKV projection module. Defaults to GPT-2 c_attn for --layer.",
    )
    hf_parser.add_argument(
        "--qkv-layout",
        choices=["gpt2", "gpt-neox"],
        default="gpt2",
        help="Packed QKV layout: gpt2 is [q_all, k_all, v_all]; gpt-neox is [head0_qkv, head1_qkv, ...].",
    )
    hf_parser.add_argument("--output-dir", default="runs/temp_ffn/hf_activations")
    hf_parser.add_argument("--run-name")
    hf_parser.set_defaults(func=run_hf_activations)

    memory_parser = subparsers.add_parser("memory", help="Write a KV-cache vs temporary-FFN memory table.")
    memory_parser.add_argument("--context-lengths", default="512,1024,2048,4096,8192,32768")
    memory_parser.add_argument("--memory-units", default="32,64,128,256,512")
    memory_parser.add_argument("--layers", type=int, default=1)
    memory_parser.add_argument("--heads", type=int, default=1)
    memory_parser.add_argument("--head-dim", type=int, default=64)
    memory_parser.add_argument("--output-dir", default="runs/temp_ffn/memory")
    memory_parser.add_argument("--run-name", default="memory_table")
    memory_parser.set_defaults(func=run_memory_table)
    return parser


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-units", default="32,64,128,256,512")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--init", choices=["sample", "random"], default="sample")
    parser.add_argument("--save-students", action="store_true")


def run_random(args: argparse.Namespace) -> int:
    torch = require_torch()
    device = resolve_device(args.device)
    q, k, v = generate_random_qkv(
        context_length=args.context_length,
        head_dim=args.head_dim,
        query_count=args.query_count + args.eval_query_count,
        seed=args.seed,
        device=str(device),
        dtype=args.dtype,
    )
    eval_q = None
    if args.eval_query_count:
        eval_q = q[args.query_count : args.query_count + args.eval_query_count]
        q = q[: args.query_count]
    results = run_compression_sweep(
        q=q,
        k=k,
        v=v,
        memory_units=parse_int_list(args.memory_units),
        eval_q=eval_q,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        cosine_weight=args.cosine_weight,
        seed=args.seed,
        init=args.init,
    )
    payload = {
        "experiment": "random_qkv",
        "compression_scope": "synthetic_random_baseline",
        "settings": sanitize_settings(vars(args)),
        "results": summarize_results(results),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.run_name}.summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_students:
        torch.save({result.memory_units: result.model_state for result in results}, output_dir / f"{args.run_name}.students.pt")
    print_result_table(results)
    print(f"Saved summary: {summary_path}")
    return 0


def run_hf_activations(args: argparse.Namespace) -> int:
    torch = require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    text = load_text(args)
    token_ids = ensure_token_count(
        tokenizer,
        text,
        required=args.context_length + args.query_count + args.eval_query_count,
        repeat_text=args.repeat_text,
    )
    total_tokens = args.context_length + args.query_count + args.eval_query_count
    batch = {"input_ids": token_ids[:total_tokens].unsqueeze(0).to(device)}
    if hasattr(tokenizer, "pad_token_id"):
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])

    pattern = args.qkv_module_pattern or rf"transformer\.h\.{args.layer}\.attn\.c_attn"
    packed, module_name = capture_single_module_output(model, batch, pattern)
    q, k, v = split_packed_qkv(packed, num_heads=get_num_heads(model), head=args.head, layout=args.qkv_layout)
    q_all = q
    q = q_all[args.context_length : args.context_length + args.query_count].float()
    eval_q = None
    if args.eval_query_count:
        eval_q = q_all[
            args.context_length + args.query_count :
            args.context_length + args.query_count + args.eval_query_count
        ].float()
    k = k[: args.context_length].float()
    v = v[: args.context_length].float()

    results = run_compression_sweep(
        q=q,
        k=k,
        v=v,
        memory_units=parse_int_list(args.memory_units),
        eval_q=eval_q,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        cosine_weight=args.cosine_weight,
        seed=17,
        init=args.init,
        layers=get_num_layers(model),
        heads=get_num_heads(model),
    )
    run_name = args.run_name or f"{args.model.rstrip('/').split('/')[-1]}_layer{args.layer}_head{args.head}"
    payload = {
        "experiment": "hf_activations",
        "compression_scope": "context_specific_temporary_memory",
        "settings": sanitize_settings(vars(args)),
        "captured_module": module_name,
        "context_tokens": args.context_length,
        "train_query_tokens": args.query_count,
        "eval_query_tokens": args.eval_query_count,
        "num_heads": get_num_heads(model),
        "num_layers": get_num_layers(model),
        "results": summarize_results(results),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_name}.summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_students:
        torch.save({result.memory_units: result.model_state for result in results}, output_dir / f"{run_name}.students.pt")
    print_result_table(results)
    print(f"Captured module: {module_name}")
    print(f"Saved summary: {summary_path}")
    return 0


def run_memory_table(args: argparse.Namespace) -> int:
    rows = []
    for context_length in parse_int_list(args.context_lengths):
        for memory_units in parse_int_list(args.memory_units):
            rows.append(
                MemoryCost(
                    context_length=context_length,
                    memory_units=memory_units,
                    layers=args.layers,
                    heads=args.heads,
                    head_dim=args.head_dim,
                ).to_dict()
            )
    payload = {"experiment": "memory_table", "settings": sanitize_settings(vars(args)), "rows": rows}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.run_name}.summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"N={row['context_length']:>6} M={row['memory_units']:>4} "
            f"ratio={row['compression_ratio']:>7.2f} "
            f"kv={row['kv_cache_parameters']:>10} temp={row['temp_ffn_parameters']:>8}"
        )
    print(f"Saved summary: {summary_path}")
    return 0


def capture_single_module_output(model, batch: dict[str, Any], pattern: str):
    captured: dict[str, Any] = {}

    def callback(ctx, module, inputs, output):
        captured[ctx.module_name] = output.detach()
        return output

    spec = HookSpec(
        name="capture_packed_qkv",
        hook_type=HookType.FORWARD,
        matcher=NameMatcher(pattern),
        callback=callback,
    )
    with HookManager.for_model(model).apply(spec):
        with require_torch().no_grad():
            model(**batch)
    if not captured:
        raise RuntimeError(f"No module output matched --qkv-module-pattern {pattern!r}.")
    if len(captured) > 1:
        names = ", ".join(sorted(captured))
        raise RuntimeError(f"Pattern matched multiple modules; make it more specific: {names}")
    name, value = next(iter(captured.items()))
    return value.squeeze(0), name


def split_packed_qkv(packed, *, num_heads: int, head: int, layout: str):
    if packed.ndim != 2:
        raise ValueError("Expected packed QKV shaped [sequence, 3 * hidden].")
    if packed.shape[-1] % 3 != 0:
        raise ValueError("Packed QKV last dimension must be divisible by 3.")
    if head < 0 or head >= num_heads:
        raise ValueError(f"--head must be in [0, {num_heads}).")
    seq_len = packed.shape[0]

    if layout == "gpt2":
        hidden_size = packed.shape[-1] // 3
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size is not divisible by number of heads.")
        head_dim = hidden_size // num_heads
        q, k, v = packed.split(hidden_size, dim=-1)
        q = q.reshape(seq_len, num_heads, head_dim)[:, head, :].contiguous()
        k = k.reshape(seq_len, num_heads, head_dim)[:, head, :].contiguous()
        v = v.reshape(seq_len, num_heads, head_dim)[:, head, :].contiguous()
        return q, k, v

    if layout == "gpt-neox":
        if packed.shape[-1] % (3 * num_heads) != 0:
            raise ValueError("Packed QKV size is not divisible by 3 * num_heads.")
        head_dim = packed.shape[-1] // (3 * num_heads)
        qkv = packed.reshape(seq_len, num_heads, 3, head_dim)
        return (
            qkv[:, head, 0, :].contiguous(),
            qkv[:, head, 1, :].contiguous(),
            qkv[:, head, 2, :].contiguous(),
        )

    raise ValueError(f"Unsupported QKV layout: {layout}")


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    return args.text or DEFAULT_TEXT


def ensure_token_count(tokenizer, text: str, *, required: int, repeat_text: bool):
    torch = require_torch()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if repeat_text and len(token_ids) < required:
        repeats = (required // max(len(token_ids), 1)) + 1
        token_ids = tokenizer.encode(text * repeats, add_special_tokens=False)
    if len(token_ids) < required:
        raise ValueError(f"Need at least {required} tokens, but got {len(token_ids)}.")
    return torch.tensor(token_ids, dtype=torch.long)


def print_result_table(results) -> None:
    has_eval = any(result.eval_metrics for result in results)
    if has_eval:
        print("M\tMSE\t\tRelErr\t\tCosSim\t\tEvalRelErr\tEvalCosSim\tRatio")
    else:
        print("M\tMSE\t\tRelErr\t\tCosSim\t\tRatio")
    for result in results:
        cost = result.memory_cost
        if result.eval_metrics:
            print(
                f"{result.memory_units}\t"
                f"{result.metrics['mse']:.6g}\t"
                f"{result.metrics['relative_error']:.6g}\t"
                f"{result.metrics['cosine_similarity']:.6g}\t"
                f"{result.eval_metrics['relative_error']:.6g}\t"
                f"{result.eval_metrics['cosine_similarity']:.6g}\t"
                f"{cost.compression_ratio if cost else 0:.2f}"
            )
        else:
            print(
                f"{result.memory_units}\t"
                f"{result.metrics['mse']:.6g}\t"
                f"{result.metrics['relative_error']:.6g}\t"
                f"{result.metrics['cosine_similarity']:.6g}\t"
                f"{cost.compression_ratio if cost else 0:.2f}"
            )


def get_num_heads(model) -> int:
    config = model.config
    for name in ("n_head", "num_attention_heads", "n_heads"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer number of attention heads from model.config.")


def get_num_layers(model) -> int:
    config = model.config
    for name in ("n_layer", "num_hidden_layers", "n_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return 1


def resolve_device(value: str):
    torch = require_torch()
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
    torch = require_torch()
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def sanitize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in settings.items() if not callable(value)}


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Context-to-Temporary-FFN experiments require PyTorch.") from exc
    return torch


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
