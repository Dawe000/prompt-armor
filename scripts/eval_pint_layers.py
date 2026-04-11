#!/usr/bin/env python3
"""Per-layer evaluation on Lakera PINT-style (or JSONL) labeled data.

Compares L1–L5 and fused risk_score against boolean injection labels so you
can see where layers agree or diverge (e.g. on hard negatives vs real attacks).

Dataset formats
---------------
1) PINT YAML (list of records), as documented by Lakera:
   - text: str
   - label: true | false   (true = prompt injection present)
   - category: optional str

2) JSONL (one object per line), same as internal benchmark dumps:
   - text: str
   - label: 1 | 0  or  true | false  (malicious / injection = 1 or true)

The full ~4.3k PINT YAML is not in the public GitHub repo (only
benchmark/data/example-dataset.yaml). Obtain the release dataset from Lakera
per their distribution terms, then pass --dataset path/to/pint.yaml.

For **jayavibhav/prompt-injection**, run ``scripts/preload_eval_datasets.py`` (writes
``data/external/jayavibhav_prompt_injection.jsonl``) or export manually with
``scripts/export_jayavibhav_prompt_injection_jsonl.py`` (balanced ``text`` / ``label`` 0|1).

For **hundreds of thousands** of JSONL rows, use ``--stream-jsonl`` so lines are read one at a time
(RAM stays ~O(1) in sample count). Omit ``--per-sample`` unless you need a huge scores file.

Usage:
    pip install -e ".[dev,ml,mcp]"
    .venv/bin/python scripts/eval_pint_layers.py --dataset /path/to/pint.yaml
    python scripts/eval_pint_layers.py --dataset pint.yaml --output-summary summary.json \\
        --per-sample scores.jsonl --max-samples 500
    .venv/bin/python scripts/eval_pint_layers.py --dataset data/external/jayavibhav_prompt_injection.jsonl \\
        --stream-jsonl --progress-every 5000 --output-summary jv_full_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

LAYER_KEYS = [
    "l1_regex",
    "l2_classifier",
    "l3_similarity",
    "l4_structural",
    "l5_negative_selection",
]


@dataclass
class LayerMetrics:
    layer: str
    threshold: float
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        t = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / t if t else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def _normalize_label(raw: object) -> int | None:
    if raw is True:
        return 1
    if raw is False:
        return 0
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("1", "true", "yes", "malicious", "attack"):
            return 1
        if s in ("0", "false", "no", "benign", "safe"):
            return 0
    try:
        if int(raw) == 1:  # int, numpy int, etc.
            return 1
        if int(raw) == 0:
            return 0
    except (TypeError, ValueError):
        pass
    try:
        if float(raw) == 1.0:
            return 1
        if float(raw) == 0.0:
            return 0
    except (TypeError, ValueError):
        pass
    return None


def load_dataset(path: Path) -> list[dict]:
    """Load samples: each dict has text, label (0/1), optional category."""
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected YAML list of records, got {type(data)}")
        out = []
        for row in data:
            if not isinstance(row, dict) or "text" not in row:
                continue
            lab = _normalize_label(row.get("label"))
            if lab is None:
                continue
            out.append(
                {
                    "text": str(row["text"]),
                    "label": lab,
                    "category": str(row.get("category", "")),
                }
            )
        return out

    if suf == ".jsonl":
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                lab = _normalize_label(row.get("label"))
                if lab is None:
                    continue
                out.append(
                    {
                        "text": str(row["text"]),
                        "label": lab,
                        "category": str(row.get("category", "")),
                    }
                )
        return out

    raise ValueError(f"Unsupported extension {suf}; use .yaml, .yml, or .jsonl")


def _analyze_timing_stats(seconds: list[float]) -> dict:
    if not seconds:
        return {
            "n_analyzed": 0,
            "mean_analyze_ms": 0.0,
            "min_analyze_ms": 0.0,
            "max_analyze_ms": 0.0,
            "p50_analyze_ms": 0.0,
            "first_sample_analyze_ms": 0.0,
            "total_analyze_seconds": 0.0,
        }
    s = sorted(seconds)
    n = len(s)
    mean = sum(s) / n
    mid = n // 2
    p50 = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    return {
        "n_analyzed": n,
        "mean_analyze_ms": round(mean * 1000, 2),
        "min_analyze_ms": round(s[0] * 1000, 2),
        "max_analyze_ms": round(s[-1] * 1000, 2),
        "p50_analyze_ms": round(p50 * 1000, 2),
        "first_sample_analyze_ms": round(seconds[0] * 1000, 2),
        "total_analyze_seconds": round(sum(seconds), 3),
    }


def _cohen_kappa_binary(n00: int, n01: int, n10: int, n11: int) -> float:
    """Cohen's kappa for two binary raters from a 2×2 contingency table."""
    n = n00 + n01 + n10 + n11
    if n == 0:
        return 0.0
    po = (n00 + n11) / n
    row0 = (n00 + n01) / n
    row1 = (n10 + n11) / n
    col0 = (n00 + n10) / n
    col1 = (n01 + n11) / n
    pe = row0 * col0 + row1 * col1
    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def _pairwise_agreement_kappa_from_cells(
    pair_cells: dict[tuple[str, str], dict[tuple[int, int], int]],
) -> tuple[dict[str, float], dict[str, float]]:
    agreement: dict[str, float] = {}
    kappas: dict[str, float] = {}
    for na, nb in sorted(pair_cells.keys()):
        cells = pair_cells[(na, nb)]
        total = sum(cells.values())
        n00 = cells.get((0, 0), 0)
        n01 = cells.get((0, 1), 0)
        n10 = cells.get((1, 0), 0)
        n11 = cells.get((1, 1), 0)
        key = f"{na}|{nb}"
        agreement[key] = round((n00 + n11) / total, 4) if total else 0.0
        kappas[key] = round(_cohen_kappa_binary(n00, n01, n10, n11), 4)
    return agreement, kappas


def _finalize_disagreement_dict(
    diff_matrix: defaultdict,
    pair_cells: defaultdict,
    fn_by_layer: dict[str, int],
    fp_by_layer: dict[str, int],
    fn_fused: int,
    fp_fused: int,
    attacks_total: int,
    benign_total: int,
    n_samples: int,
) -> dict:
    agree, kappas = _pairwise_agreement_kappa_from_cells(pair_cells)
    return {
        "pairwise_pred_disagreements": {a: dict(b) for a, b in diff_matrix.items()},
        "pairwise_percent_agreement": agree,
        "pairwise_cohens_kappa": kappas,
        "attacks_total": attacks_total,
        "benign_total": benign_total,
        "false_negatives_on_attacks_by_layer": dict(fn_by_layer),
        "false_negatives_on_attacks_fused": fn_fused,
        "false_positives_on_benign_by_layer": dict(fp_by_layer),
        "false_positives_on_benign_fused": fp_fused,
        "samples_evaluated": n_samples,
    }


def run_evaluation(
    samples: list[dict],
    layer_threshold: float,
    fused_threshold: float,
    progress_every: int = 10,
) -> tuple[dict[str, LayerMetrics], LayerMetrics, list[dict], dict]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from prompt_armor.engine import LiteEngine

    t_engine0 = time.perf_counter()
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        engine = LiteEngine()
    finally:
        sys.stdout = old_stdout
    engine_init_s = time.perf_counter() - t_engine0

    layer_m: dict[str, LayerMetrics] = {k: LayerMetrics(k, layer_threshold) for k in LAYER_KEYS}
    fused_m = LayerMetrics("fused_meta", fused_threshold)
    per_sample_rows: list[dict] = []
    analyze_seconds: list[float] = []

    print(
        f"Engine ready in {engine_init_s:.2f}s (one-time setup). Active layers: {engine.active_layers}",
        flush=True,
    )
    n = len(samples)
    t_loop0 = time.perf_counter()
    for i, s in enumerate(samples):
        engine.reset_session()
        text = s["text"]
        y = s["label"]
        try:
            t0 = time.perf_counter()
            result = engine.analyze(text)
            dt = time.perf_counter() - t0
            analyze_seconds.append(dt)
        except Exception as e:
            print(f"  ERROR sample {i}: {e}", flush=True)
            continue

        if progress_every > 0:
            k = len(analyze_seconds)
            if k == 1 or k % progress_every == 0 or i + 1 == n:
                avg = sum(analyze_seconds) / len(analyze_seconds)
                elapsed = time.perf_counter() - t_loop0
                remaining = n - (i + 1)
                eta = remaining * avg
                print(
                    f"  [{k}/{n}] last={dt * 1000:.1f}ms  avg={avg * 1000:.1f}ms  "
                    f"elapsed={elapsed:.1f}s  ETA~{eta:.0f}s",
                    flush=True,
                )

        scores: dict[str, float] = {}
        for lr in result.layer_results:
            scores[lr.layer] = lr.score

        fused_score = result.risk_score
        pred_fused = 1 if fused_score >= fused_threshold else 0
        if y == 1 and pred_fused == 1:
            fused_m.tp += 1
        elif y == 0 and pred_fused == 0:
            fused_m.tn += 1
        elif y == 0 and pred_fused == 1:
            fused_m.fp += 1
        else:
            fused_m.fn += 1

        row: dict = {
            "label": y,
            "category": s.get("category", ""),
            "fused_score": round(fused_score, 4),
            "pred_fused": pred_fused,
        }
        for k in LAYER_KEYS:
            sc = float(scores.get(k, 0.0))
            pred = 1 if sc >= layer_threshold else 0
            row[k] = round(sc, 4)
            row[f"pred_{k}"] = pred
            m = layer_m[k]
            if y == 1 and pred == 1:
                m.tp += 1
            elif y == 0 and pred == 0:
                m.tn += 1
            elif y == 0 and pred == 1:
                m.fp += 1
            else:
                m.fn += 1
        per_sample_rows.append(row)

    engine.close()
    timing = {
        "engine_init_seconds": round(engine_init_s, 3),
        **_analyze_timing_stats(analyze_seconds),
    }
    return layer_m, fused_m, per_sample_rows, timing


def disagreement_stats(per_sample: list[dict]) -> dict:
    """Counts of pairwise prediction differences, agreement %, Cohen's kappa (binary preds)."""
    diff_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_cells: defaultdict = defaultdict(lambda: defaultdict(int))
    fn_by_layer: dict[str, int] = {k: 0 for k in LAYER_KEYS}
    fp_by_layer: dict[str, int] = {k: 0 for k in LAYER_KEYS}
    fn_fused = 0
    fp_fused = 0
    attacks_total = 0
    benign_total = 0

    for row in per_sample:
        y = row["label"]
        preds = {k: row[f"pred_{k}"] for k in LAYER_KEYS}
        preds["fused"] = row["pred_fused"]

        if y == 1:
            attacks_total += 1
            for k in LAYER_KEYS:
                if preds[k] == 0:
                    fn_by_layer[k] += 1
            if preds["fused"] == 0:
                fn_fused += 1
        else:
            benign_total += 1
            for k in LAYER_KEYS:
                if preds[k] == 1:
                    fp_by_layer[k] += 1
            if preds["fused"] == 1:
                fp_fused += 1

        keys_list = list(preds.keys())
        for i, a in enumerate(keys_list):
            for b in keys_list[i + 1 :]:
                if preds[a] != preds[b]:
                    diff_matrix[a][b] += 1
                    diff_matrix[b][a] += 1
                na, nb = sorted((a, b))
                pair_cells[(na, nb)][(preds[na], preds[nb])] += 1

    return _finalize_disagreement_dict(
        diff_matrix,
        pair_cells,
        fn_by_layer,
        fp_by_layer,
        fn_fused,
        fp_fused,
        attacks_total,
        benign_total,
        len(per_sample),
    )


def run_evaluation_jsonl_stream(
    path: Path,
    layer_threshold: float,
    fused_threshold: float,
    progress_every: int,
    max_samples: int,
    per_sample_path: Path | None,
) -> tuple[dict[str, LayerMetrics], LayerMetrics, dict, dict]:
    """One JSONL row at a time; metrics + disagreement online (suitable for huge files)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from prompt_armor.engine import LiteEngine

    t_engine0 = time.perf_counter()
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        engine = LiteEngine()
    finally:
        sys.stdout = old_stdout
    engine_init_s = time.perf_counter() - t_engine0

    layer_m: dict[str, LayerMetrics] = {k: LayerMetrics(k, layer_threshold) for k in LAYER_KEYS}
    fused_m = LayerMetrics("fused_meta", fused_threshold)
    analyze_seconds: list[float] = []

    diff_matrix: defaultdict = defaultdict(lambda: defaultdict(int))
    pair_cells: defaultdict = defaultdict(lambda: defaultdict(int))
    fn_by_layer: dict[str, int] = {k: 0 for k in LAYER_KEYS}
    fp_by_layer: dict[str, int] = {k: 0 for k in LAYER_KEYS}
    fn_fused = 0
    fp_fused = 0
    attacks_total = 0
    benign_total = 0

    print(
        f"Engine ready in {engine_init_s:.2f}s (one-time setup). Active layers: {engine.active_layers}",
        flush=True,
    )
    print(f"Streaming JSONL: {path}", flush=True)

    per_fh = None
    if per_sample_path is not None:
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_fh = open(per_sample_path, "w", encoding="utf-8")
    t_loop0 = time.perf_counter()
    n_ok = 0
    line_idx = 0

    try:
        with open(path, encoding="utf-8") as inf:
            for line in inf:
                line_idx += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lab = _normalize_label(rec.get("label"))
                if lab is None:
                    continue
                text = str(rec.get("text") or "").strip()
                if not text:
                    continue
                category = str(rec.get("category", ""))

                if max_samples > 0 and n_ok >= max_samples:
                    break

                engine.reset_session()
                y = lab
                try:
                    t0 = time.perf_counter()
                    result = engine.analyze(text)
                    dt = time.perf_counter() - t0
                    analyze_seconds.append(dt)
                except Exception as e:
                    print(f"  ERROR line {line_idx}: {e}", flush=True)
                    continue

                n_ok += 1
                if progress_every > 0 and (n_ok == 1 or n_ok % progress_every == 0):
                    avg = sum(analyze_seconds) / len(analyze_seconds)
                    elapsed = time.perf_counter() - t_loop0
                    print(
                        f"  [{n_ok} lines] last={dt * 1000:.1f}ms  avg={avg * 1000:.1f}ms  elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                scores: dict[str, float] = {}
                for lr in result.layer_results:
                    scores[lr.layer] = lr.score

                fused_score = result.risk_score
                pred_fused = 1 if fused_score >= fused_threshold else 0
                if y == 1 and pred_fused == 1:
                    fused_m.tp += 1
                elif y == 0 and pred_fused == 0:
                    fused_m.tn += 1
                elif y == 0 and pred_fused == 1:
                    fused_m.fp += 1
                else:
                    fused_m.fn += 1

                preds: dict[str, int] = {"fused": pred_fused}
                row_out: dict = {
                    "label": y,
                    "category": category,
                    "fused_score": round(fused_score, 4),
                    "pred_fused": pred_fused,
                }
                for lk in LAYER_KEYS:
                    sc = float(scores.get(lk, 0.0))
                    pred = 1 if sc >= layer_threshold else 0
                    preds[lk] = pred
                    row_out[lk] = round(sc, 4)
                    row_out[f"pred_{lk}"] = pred
                    m = layer_m[lk]
                    if y == 1 and pred == 1:
                        m.tp += 1
                    elif y == 0 and pred == 0:
                        m.tn += 1
                    elif y == 0 and pred == 1:
                        m.fp += 1
                    else:
                        m.fn += 1

                if y == 1:
                    attacks_total += 1
                    for k in LAYER_KEYS:
                        if preds[k] == 0:
                            fn_by_layer[k] += 1
                    if preds["fused"] == 0:
                        fn_fused += 1
                else:
                    benign_total += 1
                    for k in LAYER_KEYS:
                        if preds[k] == 1:
                            fp_by_layer[k] += 1
                    if preds["fused"] == 1:
                        fp_fused += 1

                keys_list = list(preds.keys())
                for i, a in enumerate(keys_list):
                    for b in keys_list[i + 1 :]:
                        if preds[a] != preds[b]:
                            diff_matrix[a][b] += 1
                            diff_matrix[b][a] += 1
                        na, nb = sorted((a, b))
                        pair_cells[(na, nb)][(preds[na], preds[nb])] += 1

                if per_fh is not None:
                    per_fh.write(json.dumps(row_out) + "\n")

    finally:
        engine.close()
        if per_fh is not None:
            per_fh.close()

    timing = {
        "engine_init_seconds": round(engine_init_s, 3),
        **_analyze_timing_stats(analyze_seconds),
    }
    disc = _finalize_disagreement_dict(
        diff_matrix,
        pair_cells,
        fn_by_layer,
        fp_by_layer,
        fn_fused,
        fp_fused,
        attacks_total,
        benign_total,
        n_ok,
    )
    return layer_m, fused_m, timing, disc


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer PINT-style evaluation")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="PINT YAML or JSONL with text + label (use with --dataset-dir, one of the two)",
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory with benign.jsonl + malicious.jsonl (internal benchmark layout)",
    )
    ap.add_argument("--layer-threshold", type=float, default=0.5, help="Score >= this => predict injection (per layer)")
    ap.add_argument("--fused-threshold", type=float, default=0.5, help="Fused risk_score >= this => predict injection")
    ap.add_argument("--max-samples", type=int, default=0, help="Cap samples (0 = all)")
    ap.add_argument("--output-summary", type=Path, help="Write JSON summary (metrics + disagreements)")
    ap.add_argument("--per-sample", type=Path, help="Write JSONL with per-sample scores and preds")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print timing every N successful analyzes (0 = silent until metrics table)",
    )
    ap.add_argument(
        "--stream-jsonl",
        action="store_true",
        help="Read --dataset JSONL line-by-line (use for very large files; avoids loading all texts into RAM)",
    )
    args = ap.parse_args()

    try:
        import pydantic  # noqa: F401
    except ImportError:
        print(
            "Missing Python dependencies (e.g. pydantic). The eval script needs the "
            "installed package, not only the repo on PYTHONPATH.\n"
            "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev,ml,mcp]'\n"
            "  .venv/bin/python scripts/eval_pint_layers.py ...",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    if args.stream_jsonl:
        if args.dataset is None or args.dataset.suffix.lower() != ".jsonl":
            ap.error("--stream-jsonl requires --dataset path/to/file.jsonl")
        if args.dataset_dir is not None:
            ap.error("--stream-jsonl cannot be used with --dataset-dir")
        src = str(args.dataset.resolve())
        print(f"Streaming from {src} (max_samples={args.max_samples or 'all'})", flush=True)
        layer_m, fused_m, timing, disc = run_evaluation_jsonl_stream(
            args.dataset.resolve(),
            layer_threshold=args.layer_threshold,
            fused_threshold=args.fused_threshold,
            progress_every=args.progress_every,
            max_samples=args.max_samples,
            per_sample_path=args.per_sample,
        )
        per_sample: list[dict] = []
        n_samples = disc["samples_evaluated"]
    elif args.dataset_dir is not None:
        b = args.dataset_dir / "benign.jsonl"
        m = args.dataset_dir / "malicious.jsonl"
        samples = []
        for path, lab in [(b, 0), (m, 1)]:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    samples.append(
                        {
                            "text": str(row["text"]),
                            "label": lab,
                            "category": str(row.get("category", "")),
                        }
                    )
        if args.max_samples > 0:
            samples = samples[: args.max_samples]
        if not samples:
            print("No samples loaded.")
            sys.exit(1)
        src = str(args.dataset_dir)
        print(f"Loaded {len(samples)} samples from {src}")
        layer_m, fused_m, per_sample, timing = run_evaluation(
            samples,
            layer_threshold=args.layer_threshold,
            fused_threshold=args.fused_threshold,
            progress_every=args.progress_every,
        )
        disc = disagreement_stats(per_sample)
        n_samples = len(samples)
    elif args.dataset is not None:
        samples = load_dataset(args.dataset)
        if args.max_samples > 0:
            samples = samples[: args.max_samples]
        if not samples:
            print("No samples loaded.")
            sys.exit(1)
        src = str(args.dataset)
        print(f"Loaded {len(samples)} samples from {src}")
        layer_m, fused_m, per_sample, timing = run_evaluation(
            samples,
            layer_threshold=args.layer_threshold,
            fused_threshold=args.fused_threshold,
            progress_every=args.progress_every,
        )
        disc = disagreement_stats(per_sample)
        n_samples = len(samples)
    else:
        ap.error("Pass --dataset PATH or --dataset-dir DIR")

    print()
    print(
        f"Timing: engine_init={timing['engine_init_seconds']}s  "
        f"analyze n={timing['n_analyzed']}  "
        f"mean={timing['mean_analyze_ms']}ms  "
        f"p50={timing['p50_analyze_ms']}ms  "
        f"total_analyze={timing['total_analyze_seconds']}s",
        flush=True,
    )
    print()
    print("=" * 72)
    print(f"Thresholds: layer={args.layer_threshold}  fused={args.fused_threshold}")
    print("=" * 72)
    print(f"{'layer':<26} {'acc':>8} {'prec':>8} {'rec':>8} {'f1':>8}")
    print("-" * 72)
    for k in LAYER_KEYS:
        m = layer_m[k]
        print(f"{k:<26} {m.accuracy:>8.3f} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
    m = fused_m
    print("-" * 72)
    print(f"{'fused (meta)':<26} {m.accuracy:>8.3f} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
    print("=" * 72)

    if n_samples == 0:
        print(
            "\nWARNING: 0 samples evaluated. Check the dataset path, file size (non-empty?), "
            'and JSONL shape: one object per line with a string "text" field and a label '
            "0/1 (or true/false). Wrong or missing keys skip the row.",
            flush=True,
        )

    print("\nDisagreement / error hints (same thresholds):")
    print(f"  Attacks (label=1): {disc['attacks_total']}  Benign: {disc['benign_total']}")
    print("  FN on attacks by layer (missed injection):", disc["false_negatives_on_attacks_by_layer"])
    print("  FN on attacks (fused):", disc["false_negatives_on_attacks_fused"])
    print("  FP on benign by layer:", disc["false_positives_on_benign_by_layer"])
    print("  FP on benign (fused):", disc["false_positives_on_benign_fused"])
    print("  Pairwise |pred_a - pred_b| counts (how often two columns disagree):")
    for a, row in sorted(disc["pairwise_pred_disagreements"].items()):
        top = sorted(row.items(), key=lambda x: -x[1])[:6]
        print(f"    {a}: {dict(top)}")
    print("  Pairwise % agreement (high ⇒ often same binary pred at this threshold):")
    for k, v in sorted(disc.get("pairwise_percent_agreement", {}).items(), key=lambda x: -x[1])[:12]:
        print(f"    {k}: {v}")
    print("  Pairwise Cohen's kappa (high ⇒ redundant decisions beyond chance):")
    for k, v in sorted(disc.get("pairwise_cohens_kappa", {}).items(), key=lambda x: -x[1])[:12]:
        print(f"    {k}: {v}")

    if args.per_sample and not args.stream_jsonl:
        args.per_sample.parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample, "w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row) + "\n")
        print(f"\nPer-sample rows -> {args.per_sample}")
    elif args.per_sample and args.stream_jsonl:
        print(f"\nPer-sample rows -> {args.per_sample}")

    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": str(args.dataset) if args.dataset is not None else str(args.dataset_dir),
            "n_samples": n_samples,
            "layer_threshold": args.layer_threshold,
            "fused_threshold": args.fused_threshold,
            "timing": timing,
            "layers": {
                k: {
                    "layer": layer_m[k].layer,
                    "threshold": layer_m[k].threshold,
                    "tp": layer_m[k].tp,
                    "tn": layer_m[k].tn,
                    "fp": layer_m[k].fp,
                    "fn": layer_m[k].fn,
                    "accuracy": round(layer_m[k].accuracy, 4),
                    "precision": round(layer_m[k].precision, 4),
                    "recall": round(layer_m[k].recall, 4),
                    "f1": round(layer_m[k].f1, 4),
                }
                for k in LAYER_KEYS
            },
            "fused": {
                "layer": fused_m.layer,
                "threshold": fused_m.threshold,
                "tp": fused_m.tp,
                "tn": fused_m.tn,
                "fp": fused_m.fp,
                "fn": fused_m.fn,
                "accuracy": round(fused_m.accuracy, 4),
                "precision": round(fused_m.precision, 4),
                "recall": round(fused_m.recall, 4),
                "f1": round(fused_m.f1, 4),
            },
            "disagreement": disc,
        }
        with open(args.output_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary -> {args.output_summary}")


if __name__ == "__main__":
    main()
