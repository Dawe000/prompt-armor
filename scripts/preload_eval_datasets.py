#!/usr/bin/env python3
"""Download jayavibhav eval JSONL once into data/external/ (no HF stream during eval).

HF Hub still caches parquet under ~/.cache/huggingface on first run; this file gives you
a stable path under the repo for ``eval_pint_layers.py --dataset ...``.

Output (gitignored):
  data/external/jayavibhav_prompt_injection.jsonl  (~327k rows, train+test)

Usage:
    .venv/bin/python scripts/preload_eval_datasets.py
    .venv/bin/python scripts/preload_eval_datasets.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "external"
JV_OUT = EXTERNAL / "jayavibhav_prompt_injection.jsonl"

# Skip re-download only if the file looks like a full export (empty/touched files exist → re-run).
_MIN_JV_BYTES = 5 * 1024 * 1024


def _run_export(argv: list[str]) -> int:
    proc = subprocess.run([sys.executable, *argv], cwd=str(ROOT))
    return proc.returncode


def _preload_jayavibhav(force: bool) -> int:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if JV_OUT.exists() and not force and JV_OUT.stat().st_size >= _MIN_JV_BYTES:
        print(f"Skip (exists, {JV_OUT.stat().st_size // 1_048_576} MiB): {JV_OUT}. Use --force to refresh.")
        return 0
    if JV_OUT.exists() and JV_OUT.stat().st_size < _MIN_JV_BYTES and not force:
        print(
            f"Replacing too-small file ({JV_OUT.stat().st_size} B; expected ≥{_MIN_JV_BYTES // 1_048_576} MiB for full export).",
            flush=True,
        )
    print(f"Downloading jayavibhav/prompt-injection -> {JV_OUT} ...", flush=True)
    return _run_export(
        [
            str(ROOT / "scripts" / "export_jayavibhav_prompt_injection_jsonl.py"),
            "-o",
            str(JV_OUT),
            "--split",
            "train,test",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Preload jayavibhav eval JSONL into data/external/")
    ap.add_argument("--force", action="store_true", help="Re-download even if JSONL exists")
    args = ap.parse_args()

    rc = _preload_jayavibhav(args.force)
    if rc != 0:
        raise SystemExit(rc)

    print("\nPoint eval here, for example:")
    print(f"  .venv/bin/python scripts/eval_pint_layers.py --dataset {JV_OUT} --stream-jsonl --progress-every 5000")


if __name__ == "__main__":
    main()
