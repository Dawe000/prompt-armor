#!/usr/bin/env python3
"""Export Hugging Face jayavibhav/prompt-injection to JSONL for eval_pint_layers.py.

Source: https://huggingface.co/datasets/jayavibhav/prompt-injection

Each row has ``text`` and ``label`` (0 = benign, 1 = injection). Output matches the
JSONL format expected by ``eval_pint_layers.py`` (``text``, ``label``, optional
``category``).

Requires: ``pip install datasets``

Usage:
    pip install datasets
    python scripts/preload_eval_datasets.py
    python scripts/export_jayavibhav_prompt_injection_jsonl.py -o data/jayavibhav.jsonl --max-rows 50000
    python scripts/export_jayavibhav_prompt_injection_jsonl.py -o data/jv_all.jsonl --split train,test
    python scripts/eval_pint_layers.py --dataset data/external/jayavibhav_prompt_injection.jsonl --max-samples 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def normalize_label(raw: object) -> int | None:
    if raw is True or raw == 1:
        return 1
    if raw is False or raw == 0:
        return 0
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export jayavibhav/prompt-injection HF dataset to JSONL",
    )
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .jsonl path")
    ap.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all; ~327k rows)")
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF split: train and/or test (comma-separated), default train",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install Hugging Face datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    if not splits:
        splits = ["train"]

    n = 0
    with open(args.output, "w") as f:
        for split_name in splits:
            ds = load_dataset(
                "jayavibhav/prompt-injection",
                split=split_name,
                streaming=True,
            )
            for row in ds:
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                lab = normalize_label(row.get("label"))
                if lab is None:
                    continue
                rec = {
                    "text": text,
                    "label": lab,
                    "category": f"jayavibhav_prompt_injection:{split_name}",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.max_rows and n >= args.max_rows:
                    break
                if n % 50_000 == 0:
                    print(f"  wrote {n} rows...", flush=True)
            if args.max_rows and n >= args.max_rows:
                break

    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
