#!/usr/bin/env python3
"""One-shot: construct ``LiteEngine`` so L3 writes its FAISS index cache.

Default cache dir: ``src/prompt_armor/data/models/l3-faiss-cache/`` (gitignored).
Override with ``PROMPT_ARMOR_L3_CACHE_DIR``. Disable with
``PROMPT_ARMOR_DISABLE_L3_INDEX_CACHE=1`` (then this only loads layers, no cache file).

Usage:
    .venv/bin/python scripts/warm_l3_cache.py
"""

from __future__ import annotations

import time

from prompt_armor.config import ShieldConfig
from prompt_armor.engine import LiteEngine


def main() -> None:
    print("Warming LiteEngine (L3 will embed attacks and write cache if needed)...", flush=True)
    t0 = time.perf_counter()
    eng = LiteEngine(ShieldConfig())
    elapsed = time.perf_counter() - t0
    eng.close()
    print(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min).", flush=True)


if __name__ == "__main__":
    main()
