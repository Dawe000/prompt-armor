#!/usr/bin/env python3
"""Offline analysis for ``eval_pint_layers.py`` ``--per-sample`` JSONL.

Streams rows (memory-safe for large files), writes CSV/JSON tables for threshold
sweeps, decision buckets, calibration bins, layer disagreement summaries, and a
**deterministic three-way gate grid** (allow / warn / block) as a function of
``(allow_cut, block_cut)`` on ``fused_score`` (see ``gate_threshold_grid.csv``;
this is not bit-identical to ``fusion._decide``, which uses jitter).

Example:
    .venv/bin/python scripts/analyze_eval_per_sample.py \\
        --input runs/jayavibhav_per_sample.jsonl \\
        --out-dir runs/analysis/jayavibhav_latest
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

LAYER_KEYS = [
    "l1_regex",
    "l2_classifier",
    "l3_similarity",
    "l4_structural",
    "l5_negative_selection",
]


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    return out


def _f1(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    d = p + r
    out = np.zeros_like(p, dtype=np.float64)
    m = d > 0
    out[m] = 2 * p[m] * r[m] / d[m]
    return out


def _average_precision_rank(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision from a ranked list (no sklearn)."""
    order = np.argsort(-scores)
    y_sort = y.astype(np.int32)[order]
    tp = np.cumsum(y_sort == 1)
    fp = np.cumsum(y_sort == 0)
    prec = tp / np.maximum(tp + fp, 1)
    n_pos = max(int((y == 1).sum()), 1)
    rec = tp.astype(np.float64) / n_pos
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(prec.tolist(), rec.tolist(), strict=False):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def _load_columns(path: Path, max_rows: int) -> dict[str, Any]:
    labels: list[int] = []
    fused: list[float] = []
    fconf: list[float] = []
    decisions: list[str] = []
    pred_fused: list[int] = []
    layers: dict[str, list[float]] = {k: [] for k in LAYER_KEYS}
    cats: list[str] = []
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if max_rows > 0 and n >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "label" not in d or "fused_score" not in d:
                continue
            lab = d["label"]
            if isinstance(lab, bool):
                lab = int(lab)
            labels.append(int(lab))
            fused.append(float(d["fused_score"]))
            fconf.append(float(d.get("fused_confidence", 0.0)))
            decisions.append(str(d.get("decision", "")))
            pred_fused.append(int(d.get("pred_fused", 0)))
            cats.append(str(d.get("category", "")))
            for k in LAYER_KEYS:
                layers[k].append(float(d.get(k, 0.0)))
            n += 1
    if not labels:
        raise SystemExit(f"No valid rows read from {path}")
    out: dict[str, Any] = {
        "n": n,
        "y": np.asarray(labels, dtype=np.int8),
        "fused": np.asarray(fused, dtype=np.float64),
        "fconf": np.asarray(fconf, dtype=np.float64),
        "decision": decisions,
        "pred_fused": np.asarray(pred_fused, dtype=np.int8),
        "category": cats,
    }
    for k in LAYER_KEYS:
        out[k] = np.asarray(layers[k], dtype=np.float64)
    return out


def _threshold_sweep(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    # preds[n, t] = score >= thresholds[t]
    preds = scores[:, None] >= thresholds[None, :]
    yc = y[:, None]
    tp = ((preds) & (yc == 1)).sum(axis=0).astype(np.int64)
    tn = ((~preds) & (yc == 0)).sum(axis=0).astype(np.int64)
    fp = ((preds) & (yc == 0)).sum(axis=0).astype(np.int64)
    fn = ((~preds) & (yc == 1)).sum(axis=0).astype(np.int64)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _f1(prec, rec)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1}


def _write_fused_sweep(out_dir: Path, thresholds: np.ndarray, sweep: dict[str, np.ndarray]) -> None:
    path = out_dir / "threshold_sweep_fused.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "tp", "tn", "fp", "fn", "precision", "recall", "f1"])
        for i, t in enumerate(thresholds):
            w.writerow(
                [
                    f"{t:.4f}",
                    int(sweep["tp"][i]),
                    int(sweep["tn"][i]),
                    int(sweep["fp"][i]),
                    int(sweep["fn"][i]),
                    f"{sweep['prec'][i]:.6f}",
                    f"{sweep['rec'][i]:.6f}",
                    f"{sweep['f1'][i]:.6f}",
                ]
            )


def _write_layer_sweeps(out_dir: Path, y: np.ndarray, data: dict[str, Any], thresholds: np.ndarray) -> None:
    path = out_dir / "threshold_sweep_layers.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "threshold", "tp", "tn", "fp", "fn", "precision", "recall", "f1"])
        for lk in LAYER_KEYS:
            sw = _threshold_sweep(y, data[lk], thresholds)
            for i, t in enumerate(thresholds):
                w.writerow(
                    [
                        lk,
                        f"{t:.4f}",
                        int(sw["tp"][i]),
                        int(sw["tn"][i]),
                        int(sw["fp"][i]),
                        int(sw["fn"][i]),
                        f"{sw['prec'][i]:.6f}",
                        f"{sw['rec'][i]:.6f}",
                        f"{sw['f1'][i]:.6f}",
                    ]
                )


def _best_f1_threshold(thresholds: np.ndarray, f1: np.ndarray) -> tuple[float, float]:
    j = int(np.nanargmax(f1))
    return float(thresholds[j]), float(f1[j])


def _min_cost_threshold(
    thresholds: np.ndarray, fn: np.ndarray, fp: np.ndarray, w_fn: float, w_fp: float
) -> tuple[float, float]:
    cost = w_fn * fn.astype(np.float64) + w_fp * fp.astype(np.float64)
    j = int(np.argmin(cost))
    return float(thresholds[j]), float(cost[j])


def _decision_crosstab(decisions: list[str], y: np.ndarray) -> Counter[tuple[str, int]]:
    c: Counter[tuple[str, int]] = Counter()
    for d, yi in zip(decisions, y.tolist(), strict=False):
        c[(d or "missing", int(yi))] += 1
    return c


def _write_decision_tables(out_dir: Path, decisions: list[str], y: np.ndarray, pred_fused: np.ndarray) -> None:
    ct = _decision_crosstab(decisions, y)
    path = out_dir / "decision_vs_label.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["decision", "label", "count"])
        for (dec, lab), cnt in sorted(ct.items()):
            w.writerow([dec, lab, cnt])

    path2 = out_dir / "pred_fused_vs_decision.csv"
    c2: Counter[tuple[int, str]] = Counter()
    for pf, d in zip(pred_fused.tolist(), decisions, strict=False):
        c2[(int(pf), d or "missing")] += 1
    with path2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred_fused_0.5_cut", "decision", "count"])
        for (pf, dec), cnt in sorted(c2.items()):
            w.writerow([pf, dec, cnt])


def _calibration_bins(y: np.ndarray, conf: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum((conf * n_bins).astype(np.int32), n_bins - 1)
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append(
                {
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "n": 0,
                    "attack_rate": None,
                    "mean_confidence": None,
                }
            )
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": cnt,
                "attack_rate": float(y[m].mean()),
                "mean_confidence": float(conf[m].mean()),
            }
        )
    return rows


def _write_calibration(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = out_dir / "calibration_fused_confidence.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["bin_low", "bin_high", "n", "attack_rate", "mean_confidence"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "bin_low": f"{r['bin_low']:.4f}",
                    "bin_high": f"{r['bin_high']:.4f}",
                    "n": r["n"],
                    "attack_rate": "" if r["attack_rate"] is None else f"{r['attack_rate']:.6f}",
                    "mean_confidence": "" if r["mean_confidence"] is None else f"{r['mean_confidence']:.6f}",
                }
            )


def _layer_disagreement(y: np.ndarray, data: dict[str, Any], layer_threshold: float) -> dict[str, Any]:
    preds = np.stack([(data[k] >= layer_threshold).astype(np.int8) for k in LAYER_KEYS], axis=1)
    pair_counts: dict[str, int] = {}
    for i, a in enumerate(LAYER_KEYS):
        for j in range(i + 1, len(LAYER_KEYS)):
            b = LAYER_KEYS[j]
            key = f"{a}|{b}"
            pair_counts[key] = int((preds[:, i] != preds[:, j]).sum())
    # Solo "caught" attack: y=1 and pred_k=1 and sum preds == 1
    solo: dict[str, int] = {}
    s = preds.sum(axis=1)
    for i, k in enumerate(LAYER_KEYS):
        m = (y == 1) & (preds[:, i] == 1) & (s == 1)
        solo[k] = int(m.sum())
    # Attack recall at threshold
    atk = y == 1
    n_atk = int(atk.sum())
    layer_recall: dict[str, float] = {}
    for i, k in enumerate(LAYER_KEYS):
        layer_recall[k] = float((preds[atk, i] == 1).sum() / max(n_atk, 1))
    return {"pair_disagreements": pair_counts, "solo_catch_attacks": solo, "layer_recall_at_t": layer_recall}


def _simulate_gate_decisions(s: np.ndarray, allow_cut: float, block_cut: float) -> np.ndarray:
    """0 = allow, 1 = warn, 2 = block. Deterministic; requires block_cut > allow_cut."""
    return np.where(s < allow_cut, 0, np.where(s >= block_cut, 2, 1)).astype(np.int8)


def _gate_grid_row(s: np.ndarray, y: np.ndarray, allow_cut: float, block_cut: float) -> dict[str, float | int]:
    """One row of metrics for a (allow_cut, block_cut) gate on fused_score."""
    dec = _simulate_gate_decisions(s, allow_cut, block_cut)
    atk = y == 1
    ben = y == 0
    na, nb = int(atk.sum()), int(ben.sum())
    d_atk = dec[atk]
    d_ben = dec[ben]

    def _frac(d: np.ndarray, k: int, denom: int) -> float:
        return float((d == k).sum() / max(denom, 1))

    return {
        "allow_cut": allow_cut,
        "block_cut": block_cut,
        "warn_band_width": round(block_cut - allow_cut, 4),
        "pct_allow_all": float((dec == 0).mean()),
        "pct_warn_all": float((dec == 1).mean()),
        "pct_block_all": float((dec == 2).mean()),
        "attack_pct_allow": _frac(d_atk, 0, na),
        "attack_pct_warn": _frac(d_atk, 1, na),
        "attack_pct_block": _frac(d_atk, 2, na),
        "benign_pct_allow": _frac(d_ben, 0, nb),
        "benign_pct_warn": _frac(d_ben, 1, nb),
        "benign_pct_block": _frac(d_ben, 2, nb),
        "attacks_allowed_n": int((atk & (dec == 0)).sum()),
        "benign_blocked_n": int((ben & (dec == 2)).sum()),
    }


def _write_gate_threshold_grid(
    out_dir: Path,
    s: np.ndarray,
    y: np.ndarray,
    allow_start: float,
    allow_stop: float,
    allow_step: float,
    block_start: float,
    block_stop: float,
    block_step: float,
) -> tuple[list[dict[str, float | int]], Path]:
    allow_vals = np.arange(allow_start, allow_stop + 1e-9, allow_step)
    block_vals = np.arange(block_start, block_stop + 1e-9, block_step)
    rows: list[dict[str, float | int]] = []
    for a in allow_vals:
        af = float(round(float(a), 4))
        for b in block_vals:
            bf = float(round(float(b), 4))
            if bf <= af:
                continue
            rows.append(_gate_grid_row(s, y, af, bf))
    path = out_dir / "gate_threshold_grid.csv"
    if not rows:
        return rows, path
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            line = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(line)
    return rows, path


def _fn_fp_by_category(cats: list[str], y: np.ndarray, pred: np.ndarray) -> list[tuple[str, int, int, int]]:
    """Per coarse category string: benign fp, attack fn when using pred as injection flag."""
    by: dict[str, list[int]] = {}
    for c, yi, p in zip(cats, y.tolist(), pred.tolist(), strict=False):
        key = c.split(":")[0] if c else "(empty)"
        by.setdefault(key, [0, 0])
        if yi == 0 and p == 1:
            by[key][0] += 1  # fp on benign
        if yi == 1 and p == 0:
            by[key][1] += 1  # fn on attack
    rows = []
    for k, (fp, fn) in sorted(by.items(), key=lambda x: -(x[1][0] + x[1][1])):
        rows.append((k, fp, fn, fp + fn))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze eval_pint_layers --per-sample JSONL")
    ap.add_argument("--input", type=Path, required=True, help="Per-sample JSONL from eval_pint_layers.py")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for CSV + summary.json")
    ap.add_argument("--max-rows", type=int, default=0, help="Cap rows for smoke tests (0 = all)")
    ap.add_argument(
        "--layer-threshold",
        type=float,
        default=0.5,
        help="Threshold for layer disagreement / solo-catch stats (match eval default)",
    )
    ap.add_argument(
        "--cost-fn",
        type=float,
        default=5.0,
        help="Weight on false negatives in cost = w_fn*FN + w_fp*FP (toy default 5:1)",
    )
    ap.add_argument("--cost-fp", type=float, default=1.0, help="Weight on false positives in cost line")
    ap.add_argument("--gate-allow-start", type=float, default=0.35, help="Gate grid: min allow_cut on fused_score")
    ap.add_argument("--gate-allow-stop", type=float, default=0.70, help="Gate grid: max allow_cut (inclusive)")
    ap.add_argument("--gate-allow-step", type=float, default=0.05, help="Gate grid: allow_cut step")
    ap.add_argument("--gate-block-start", type=float, default=0.55, help="Gate grid: min block_cut")
    ap.add_argument("--gate-block-stop", type=float, default=0.95, help="Gate grid: max block_cut (inclusive)")
    ap.add_argument("--gate-block-step", type=float, default=0.05, help="Gate grid: block_cut step")
    args = ap.parse_args()

    inp = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {inp} ...", flush=True)
    data = _load_columns(inp, args.max_rows)
    n = data["n"]
    y = data["y"]
    print(f"Loaded n={n}", flush=True)

    thresholds = np.linspace(0.0, 1.0, 101)
    fused_sw = _threshold_sweep(y, data["fused"], thresholds)
    _write_fused_sweep(out_dir, thresholds, fused_sw)
    _write_layer_sweeps(out_dir, y, data, thresholds)

    t_best_f1, f1_max = _best_f1_threshold(thresholds, fused_sw["f1"])
    t_cost, cost_min = _min_cost_threshold(thresholds, fused_sw["fn"], fused_sw["fp"], args.cost_fn, args.cost_fp)
    ap_rank = _average_precision_rank(y, data["fused"])

    _write_decision_tables(out_dir, data["decision"], y, data["pred_fused"])
    cal_rows = _calibration_bins(y, data["fconf"], 10)
    _write_calibration(out_dir, cal_rows)

    ld = _layer_disagreement(y, data, args.layer_threshold)
    with (out_dir / "layer_pair_disagreement.json").open("w", encoding="utf-8") as f:
        json.dump(ld, f, indent=2)

    gate_rows, gate_path = _write_gate_threshold_grid(
        out_dir,
        data["fused"],
        y,
        args.gate_allow_start,
        args.gate_allow_stop,
        args.gate_allow_step,
        args.gate_block_start,
        args.gate_block_stop,
        args.gate_block_step,
    )

    gate_ref: dict[str, Any] | None = None
    for r in gate_rows:
        if abs(float(r["allow_cut"]) - 0.5) < 1e-6 and abs(float(r["block_cut"]) - 0.8) < 1e-6:
            gate_ref = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in r.items()}
            break

    # Fused at 0.5 for category slice (same as pred_fused in eval when fused_threshold=0.5)
    pred_05 = (data["fused"] >= 0.5).astype(np.int8)
    cat_rows = _fn_fp_by_category(data["category"], y, pred_05)
    with (out_dir / "fn_fp_by_category_prefix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category_prefix", "fp_on_benign", "fn_on_attack", "fp_plus_fn"])
        for row in cat_rows[:80]:
            w.writerow(row)

    # pred_fused vs y agreement (sanity)
    pf = data["pred_fused"]
    agree = float((pf == y).mean())

    summary = {
        "input": str(inp),
        "n_rows": n,
        "max_rows_applied": args.max_rows,
        "layer_threshold_disagreement": args.layer_threshold,
        "fused_score": {
            "average_precision_rank": ap_rank,
            "best_f1_threshold": t_best_f1,
            "best_f1_value": f1_max,
            "min_cost_threshold": t_cost,
            "min_cost_value": cost_min,
            "cost_weights": {"w_fn": args.cost_fn, "w_fp": args.cost_fp},
            "at_threshold_0.5": {
                "precision": float(fused_sw["prec"][50]),
                "recall": float(fused_sw["rec"][50]),
                "f1": float(fused_sw["f1"][50]),
                "fp": int(fused_sw["fp"][50]),
                "fn": int(fused_sw["fn"][50]),
            },
        },
        "pred_fused_vs_label_accuracy": agree,
        "note_pred_fused": "pred_fused uses fused_threshold from eval run (often 0.5); compare to decision for gate behavior.",
        "gate_simulation": {
            "rule": "deterministic: fused_score < allow_cut -> allow; >= block_cut -> block; else warn",
            "note": "Production fusion._decide uses Gaussian jitter around meta threshold and a hard block split at score>=0.8; YAML allow_below/block_above drive council uncertainty, not this exact three-way. Use the grid for comparative what-if analysis.",
            "grid_rows": len(gate_rows),
            "no_jitter_reference_allow0.5_block0.8": gate_ref,
            "csv": str(gate_path),
        },
        "layer_stats": ld,
        "outputs": {
            "threshold_sweep_fused": str(out_dir / "threshold_sweep_fused.csv"),
            "threshold_sweep_layers": str(out_dir / "threshold_sweep_layers.csv"),
            "decision_vs_label": str(out_dir / "decision_vs_label.csv"),
            "pred_fused_vs_decision": str(out_dir / "pred_fused_vs_decision.csv"),
            "calibration": str(out_dir / "calibration_fused_confidence.csv"),
            "layer_pairs": str(out_dir / "layer_pair_disagreement.json"),
            "fn_fp_by_category": str(out_dir / "fn_fp_by_category_prefix.csv"),
            "gate_threshold_grid": str(gate_path),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote -> {out_dir}", flush=True)
    print(
        json.dumps(
            {
                "n_rows": summary["n_rows"],
                "fused_score": summary["fused_score"],
                "pred_fused_vs_label_accuracy": summary["pred_fused_vs_label_accuracy"],
                "gate_grid_rows": summary["gate_simulation"]["grid_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
