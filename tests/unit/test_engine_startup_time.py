"""Cold-start timing for ``LiteEngine`` (full prompt-injection / security ensemble).

Measures construction of ``LiteEngine``; dominant cost is usually L3 (embed attack DB + FAISS build).

Run (slow, minutes on first cold cache):

    PROMPT_ARMOR_BENCH_STARTUP=1 pytest tests/unit/test_engine_startup_time.py -v

The measured ``LiteEngine()`` wall time is emitted as a **UserWarning** so it shows up in
pytest’s **warnings summary** without ``-s``. For live stdout use ``pytest -s``.

Optional: ``PROMPT_ARMOR_STARTUP_CAP_S`` (default 1800) to fail if startup exceeds cap.
"""

from __future__ import annotations

import os
import time
import warnings

import pytest

from prompt_armor.config import ShieldConfig
from prompt_armor.engine import LiteEngine

_STARTUP_CAP_S = float(os.environ.get("PROMPT_ARMOR_STARTUP_CAP_S", "1800"))


@pytest.mark.timeout(1200)
def test_lite_engine_cold_startup_under_cap() -> None:
    if os.environ.get("PROMPT_ARMOR_BENCH_STARTUP") != "1":
        pytest.skip("Set PROMPT_ARMOR_BENCH_STARTUP=1 to benchmark LiteEngine() startup (slow).")

    t0 = time.perf_counter()
    engine = LiteEngine(ShieldConfig())
    elapsed = time.perf_counter() - t0
    engine.close()

    msg = f"LiteEngine() cold startup: {elapsed:.2f}s ({elapsed / 60:.2f} min)"
    warnings.warn(msg, UserWarning, stacklevel=1)

    assert elapsed > 0.05, "LiteEngine() returned implausibly fast — did layers fail to load?"
    assert elapsed < _STARTUP_CAP_S, (
        f"LiteEngine() startup {elapsed:.1f}s exceeds cap {_STARTUP_CAP_S}s "
        "(raise PROMPT_ARMOR_STARTUP_CAP_S if your runner is cold/slow)."
    )
