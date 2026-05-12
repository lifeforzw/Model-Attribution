from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_PRESETS = {
    "synthetic": {
        "dataset_id": None,
        "subset": None,
        "split": None,
        "text_fields": ("text",),
        "note": "Local repeated text for smoke tests and pipeline validation.",
        "url": None,
    },
    "wikitext103": {
        "dataset_id": "Salesforce/wikitext",
        "subset": "wikitext-103-raw-v1",
        "split": "validation",
        "text_fields": ("text",),
        "note": "Medium-size Wikipedia language modeling data; good first real-text source.",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
    },
    "pg19": {
        "dataset_id": "deepmind/pg19",
        "subset": None,
        "split": "validation",
        "text_fields": ("text",),
        "note": "Book-length Project Gutenberg texts; large download, use validation first.",
        "url": "https://huggingface.co/datasets/deepmind/pg19",
    },
    "longbench": {
        "dataset_id": "zai-org/LongBench",
        "subset": "qasper",
        "split": "test",
        "text_fields": ("context", "input", "text"),
        "note": "Long-context downstream benchmark; useful after QKV reconstruction is stable.",
        "url": "https://huggingface.co/datasets/zai-org/LongBench",
    },
}


SYNTHETIC_TEXT = (
    "Long-context compression experiments need repeated but coherent text. "
    "This local sample is intentionally small and should only be used to check "
    "tokenization, command generation, and file plumbing before real runs. "
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare bounded text files for Context-to-Temporary-FFN experiments."
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_PRESETS),
        default="synthetic",
        help="Dataset preset to prepare.",
    )
    parser.add_argument("--dataset-id", help="Override HuggingFace dataset id.")
    parser.add_argument("--subset", help="Override HuggingFace dataset subset/config.")
    parser.add_argument("--split", help="Override split.")
    parser.add_argument(
        "--text-fields",
        default=None,
        help="Comma-separated candidate text fields. First present non-empty field is used.",
    )
    parser.add_argument("--max-docs", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=2_000_000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", default="data/temp_ffn")
    parser.add_argument("--output-name")
    parser.add_argument(
        "--synthetic-repeats",
        type=int,
        default=256,
        help="Number of repeated paragraphs for --source synthetic.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preset = SOURCE_PRESETS[args.source]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or args.source
    text_path = output_dir / f"{output_name}.txt"
    manifest_path = output_dir / f"{output_name}.manifest.json"
    if not args.force and (text_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing files for {output_name!r}; pass --force.")

    if args.source == "synthetic":
        text = (SYNTHETIC_TEXT * args.synthetic_repeats).strip() + "\n"
        docs_used = args.synthetic_repeats
        resolved = resolved_config(args, preset)
    else:
        resolved = resolved_config(args, preset)
        texts = load_hf_texts(
            dataset_id=resolved["dataset_id"],
            subset=resolved["subset"],
            split=resolved["split"],
            text_fields=resolved["text_fields"],
            max_docs=args.max_docs,
            max_chars=args.max_chars,
            streaming=args.streaming,
            cache_dir=args.cache_dir,
        )
        docs = list(texts)
        docs_used = len(docs)
        text = "\n\n".join(docs).strip() + "\n"

    text_path.write_text(text, encoding="utf-8")
    manifest = {
        "source": args.source,
        "dataset": resolved,
        "text_path": str(text_path),
        "docs_used": docs_used,
        "chars_written": len(text),
        "max_docs": args.max_docs,
        "max_chars": args.max_chars,
        "streaming": args.streaming,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote text: {text_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


def resolved_config(args: argparse.Namespace, preset: dict[str, Any]) -> dict[str, Any]:
    text_fields = args.text_fields.split(",") if args.text_fields else preset["text_fields"]
    return {
        "dataset_id": args.dataset_id or preset["dataset_id"],
        "subset": args.subset if args.subset is not None else preset["subset"],
        "split": args.split if args.split is not None else preset["split"],
        "text_fields": [field.strip() for field in text_fields if field.strip()],
        "url": preset["url"],
        "note": preset["note"],
    }


def load_hf_texts(
    *,
    dataset_id: str,
    subset: str | None,
    split: str,
    text_fields: Iterable[str],
    max_docs: int,
    max_chars: int,
    streaming: bool,
    cache_dir: str | None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets first: python -m pip install datasets"
        ) from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if subset:
        dataset = load_dataset(dataset_id, subset, **kwargs)
    else:
        dataset = load_dataset(dataset_id, **kwargs)

    chars = 0
    docs = 0
    for row in dataset:
        text = first_text(row, text_fields)
        if not text:
            continue
        remaining = max_chars - chars
        if remaining <= 0 or docs >= max_docs:
            break
        clipped = text[:remaining]
        chars += len(clipped)
        docs += 1
        yield clipped


def first_text(row: MappingLike, text_fields: Iterable[str]) -> str:
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = "\n".join(str(item) for item in value if item)
            if joined.strip():
                return joined.strip()
    return ""


MappingLike = dict[str, Any]


if __name__ == "__main__":
    raise SystemExit(main())
