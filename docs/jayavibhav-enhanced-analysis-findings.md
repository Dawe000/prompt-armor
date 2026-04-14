# Jayavibhav per-sample analysis — findings

**Generated:** 2026-04-14  
**Input:** `runs/jayavibhav_per_sample_with_evidence.jsonl` (rich eval rows from `scripts/eval_pint_layers.py`, including evidence)  
**Analyzer:** `scripts/analyze_eval_per_sample.py` → **`runs/analysis/jayavibhav_with_gate_grid/`** (CSV + `summary.json`, includes **`gate_threshold_grid.csv`**)  
**Methodology reference:** `docs/eval-per-sample-analysis.md`

---

## Executive summary

On **327,153** rows (jayavibhav train+test style export), the fused meta-score is **strong on attack recall** at a 0.5 cut but **weak on benign precision**, matching the earlier single-threshold writeup. Offline sweeps show only a **modest** F1 gain when raising the fused threshold into the **0.66** region, at the cost of more false negatives. **`fused_confidence`** is **concentrated near 1.0**; in that mass bin the empirical attack rate is only **~0.56**, so confidence must **not** be read as calibrated probability of attack on this corpus. **L3** dominates both recall and “solo” catches at score ≥ 0.5; **L2** is the main layer that **disagrees** with L3 at scale. The three-way **`decision`** gate differs materially from legacy **`pred_fused`** at the eval threshold (many `pred_fused=1` rows land in **warn** or **allow**). A **deterministic gate grid** on `fused_score` shows how allow/warn/block **splits** would move under different `(allow_cut, block_cut)` pairs (not identical to production jitter; see §6). **§7** turns that diagnosis into a **model roadmap**, with **L3 first** for the bulk of benign false positives.

---

## Corpus check

| Split | Count |
|-------|------:|
| Total rows | 327,153 |
| Attacks (`label=1`) | 161,526 |
| Benign (`label=0`) | 165,627 |

Derived from `decision_vs_label.csv` (sums per label match these totals).

---

## 1. Fused score (`fused_score`) vs labels

**Average precision (rank-based, script metric):** **0.576**

**At legacy binary cut `fused_score ≥ 0.5`** (matches typical `pred_fused`):

| Metric | Value |
|--------|------:|
| Precision | 0.555 |
| Recall | 0.978 |
| F1 | 0.708 |
| False positives (benign flagged) | 126,579 |
| False negatives (attacks missed) | 3,490 |
| Accuracy vs label (`pred_fused == label`) | 0.602 |

**Grid optima (same 0.01 step sweep as the script):**

| Policy | Threshold | Headline |
|--------|-----------|----------|
| Max F1 on grid | **0.66** | F1 **0.715** (FP 119,554, FN 5,142) |
| Toy min cost `5*FN + FP` | **0.49** | Cost **141,792** (FP 131,717, FN 2,015) |

**Illustrative sweep points** (`threshold_sweep_fused.csv`):

| Threshold | Precision | Recall | F1 | FP | FN |
|-----------|-----------|--------|-----|------:|------:|
| 0.40 | 0.528 | 0.996 | 0.691 | 143,572 | 604 |
| 0.45 | 0.534 | 0.995 | 0.695 | 140,401 | 805 |
| 0.49 | 0.548 | 0.988 | 0.705 | 131,717 | 2,015 |
| 0.50 | 0.555 | 0.978 | 0.708 | 126,579 | 3,490 |
| 0.66 | 0.567 | 0.968 | 0.715 | 119,554 | 5,142 |
| 0.70 | 0.568 | 0.955 | 0.712 | 117,115 | 7,327 |
| 0.75 | 0.568 | 0.944 | 0.709 | 116,064 | 9,119 |
| 0.80 | 0.567 | 0.940 | 0.708 | 115,906 | 9,646 |

**Finding:** Raising the threshold trades FN for FP smoothly; **no silver bullet** on this corpus—high recall stays expensive in benign FP volume.

---

## 2. Gate `decision` vs gold label

Counts from **`decision_vs_label.csv`**:

| decision | label=0 (benign) | label=1 (attack) |
|----------|-----------------:|-----------------:|
| allow | 36,197 | 2,892 |
| warn | 13,524 | 6,754 |
| block | 115,906 | 151,880 |

**Empirical P(attack | decision)** (useful sanity for the warn band):

| decision | P(attack \| decision) |
|----------|----------------------:|
| allow | **7.4%** (2,892 / 39,089) |
| warn | **33.3%** (6,754 / 20,278) |
| block | **56.7%** (151,880 / 267,786) |

**Finding:** Even **block** is wrong on benign **~43%** of the time on this dataset (115,906 / 267,786). **Allow** still includes **~7%** attacks by row count—worth mining `decision=allow` & `label=1` for false negatives.

---

## 3. `pred_fused` (eval cut, usually 0.5) vs `decision`

From **`pred_fused_vs_decision.csv`**:

| pred_fused | decision | count |
|------------|----------|------:|
| 0 | allow | 37,316 |
| 0 | warn | 5,222 |
| 1 | allow | 1,773 |
| 1 | block | 267,786 |
| 1 | warn | 15,056 |

**Finding:** A large tail is **`pred_fused=1` but not `block`** (15,056 warn + 1,773 allow): the **gate and jitter** materially change outcomes vs a raw fused threshold. Any report that only uses `pred_fused` for “what the product does” will **mis-state** behaviour.

---

## 4. `fused_confidence` calibration (coarse bins)

From **`calibration_fused_confidence.csv`** (equal-width bins on confidence; most mass is in the top bin):

| Bin (confidence) | n | Attack rate | Mean confidence |
|------------------|--:|-------------|-----------------|
| 0.60–0.70 | 28,842 | 0.152 | 0.65 |
| 0.80–0.90 | 28,844 | 0.182 | 0.85 |
| 0.90–1.00 | 269,467 | **0.564** | ~1.00 |

Lower bins are empty on this run (scores rarely land there).

**Finding:** **`fused_confidence` is not a calibrated P(attack)`** here: in the dominant bin, benign and attack rows are **about equally common** (~56% attack vs ~49% base rate), so **do not** treat high confidence as “safe to auto-block without review” without recalibration or domain-specific thresholds.

---

## 5. Layers at score ≥ 0.5

**Recall on attacks** (fraction of `label=1` with layer score ≥ 0.5):

| Layer | Recall |
|-------|--------|
| l3_similarity | **0.997** |
| l2_classifier | 0.526 |
| l5_negative_selection | 0.057 |
| l1_regex | 0.046 |
| l4_structural | 0.027 |

**Solo catch** (attack row with **exactly one** layer ≥ 0.5 among the five):

| Layer | Solo catches |
|-------|-------------:|
| l3_similarity | **71,394** |
| l2_classifier | 118 |
| l5_negative_selection | 17 |
| l4_structural | 6 |
| l1_regex | 1 |

**Pairwise disagreement** (binary preds differ; max possible 327,153):

| Pair | Disagreements |
|------|---------------:|
| l3_similarity \| l4_structural | 311,394 |
| l1_regex \| l3_similarity | 309,488 |
| l3_similarity \| l5_negative_selection | 298,124 |
| l2_classifier \| l3_similarity | 202,248 |

**Finding:** **L3** carries almost all attack detection at 0.5; **L2** is the main **orthogonal** signal (large disagree count with L3). Improvement work often focuses on **L3 benign false neighbours** and **L2–L3 disagreement slices**.

---

## 6. Simulated allow / warn / block (deterministic gate grid)

We sweep **two cuts** on **`fused_score`** (see `docs/eval-per-sample-analysis.md` §7 for the exact rule and production caveats):

- **allow** if `fused_score < allow_cut`
- **block** if `fused_score >= block_cut`
- **warn** in between

This answers “what would the bucket mix look like?” **without** re-running the engine. It is **not** the same as live `fusion._decide` (jitter + implementation details).

### 6.1 Reference row (no jitter, `allow_cut=0.5`, `block_cut=0.8`)

Aligned with the **0.8** block line used inside `fusion._decide` (allow boundary still differs because production uses jitter around the meta threshold).

| Metric | Value |
|--------|------:|
| Rows | 327,153 |
| **Attacks** in **allow** (`attacks_allowed_n`) | **3,490** (2.16% of attacks) |
| **Benign** in **block** (`benign_blocked_n`) | **115,906** (70.0% of benign) |
| Attack share in **block** | 94.0% |
| Benign share in **allow** | 23.6% |
| Overall **allow** / **warn** / **block** | 13.0% / 5.1% / 81.9% |

So at this pair, **every attack** with `fused_score < 0.5` is still “allowed” in simulation (3,490 rows — same count as FN in the binary `≥0.5` sweep), and **most benigns** still land in **block** because their `fused_score` is often high on this corpus.

### 6.2 Same `allow_cut=0.5`, varying `block_cut`

`attacks_allowed_n` stays **3,490** for all rows here: with `allow_cut=0.5`, **allow** is exactly `fused_score < 0.5`, so the **attack miss count** does not depend on `block_cut`. Changing `block_cut` moves mass between **warn** and **block** and changes how many **benigns** sit above the block line.

| block_cut | pct_warn (all) | pct_block (all) | benign_blocked_n | attack_pct_warn |
|------------|----------------|-----------------|-------------------:|------------------|
| 0.65 | 2.5% | 84.5% | 119,960 | 0.9% |
| 0.70 | 4.1% | 82.9% | 117,118 | 2.4% |
| 0.75 | 4.9% | 82.1% | 116,064 | 3.5% |
| **0.80** | **5.1%** | **81.9%** | **115,906** | **3.8%** |
| 0.85 | 5.3% | 81.7% | 115,771 | 4.0% |
| 0.90 | 5.3% | 81.7% | 115,762 | 4.1% |

### 6.3 Fixed `block_cut=0.8`, varying `allow_cut`

Tightening **allow** (raising `allow_cut`) pushes more low-score rows into **warn**, and **increases attacks in allow** (worse for safety):

| allow_cut | attacks_allowed_n | benign_pct_allow |
|-----------|-------------------:|-----------------:|
| 0.35 | 563 | 12.8% |
| 0.45 | 805 | 15.2% |
| **0.50** | **3,490** | **23.6%** |
| 0.55 | 4,435 | 26.1% |
| 0.60 | 4,645 | 27.0% |
| 0.65 | 4,948 | 27.6% |
| 0.70 | 7,324 | 29.3% |

**Read:** On this dataset, **widening the warn band** (higher `allow_cut` with fixed `block_cut`) trades **more attacks in allow** for modest movement in benign allow rate. Pick cuts using **explicit costs** on `attacks_allowed_n` and `benign_blocked_n` (or rates), not aesthetics.

Full grid: **`runs/analysis/jayavibhav_with_gate_grid/gate_threshold_grid.csv`** (`summary.json` → `gate_simulation`).

---

## 7. Next steps toward improving the model

The benign false-alarm problem on this corpus is **largely L3-shaped**: very high attack recall and solo-catches come from **L3 similarity**, and the meta-model **weights L3 heavily**, so benign prompts that sit **near attack embeddings** drive **`fused_score`** and **`block`/`warn`** volume. **Start by improving L3 signal quality** (not by turning L3 off). Order below is deliberate: each step assumes you keep measuring with `analyze_eval_per_sample.py` on held-out and jayavibhav-style exports.

1. **L3 embedding space (primary)**  
   - **Contrastive fine-tune** with **hard benign negatives** sampled from your false-positive slice (`label=0`, high `l3_similarity` or high `fused_score`), plus diverse attack anchors so you do not collapse recall.  
   - Revisit **similarity→score mapping** and **attack DB** hygiene (dedupe, stale or off-topic neighbours that pull benigns in).  
   - After any L3 change: **re-embed the attack index**, invalidate the FAISS cache fingerprint, and re-run full eval + this analysis.

2. **Fusion meta-classifier (after L3 scores move)**  
   - Re-dump layer scores and **retrain fusion** (`scripts/dump_layer_scores.py`, `scripts/train_fusion.py`) with class weights or cost-sensitive objectives that penalize **benign + high fused** more than today.  
   - Explore features that encode **L2 vs L3 disagreement** more strongly if L2 remains a benign veto signal on error slices.

3. **Gate and product policy (secondary to representation)**  
   - Use **`gate_threshold_grid.csv`** and **`decision_vs_label.csv`** to choose warn vs block friction vs misses; this can **mitigate** pain while L3 improves but will **not** fix wrong neighbours by itself.  
   - If **warn** volume is the problem, define whether warn is “review queue” or “soft block” and tune **council / uncertainty** paths (`needs_council`) against the same per-sample exports.

4. **Ongoing evaluation**  
   - Run the same analyzer on **other datasets** so improvements are not overfit to jayavibhav alone.  
   - Keep a small frozen **qualitative set** (exported JSONL filters: benign + block, attack + allow) for human review before/after each L3 iteration.

---

## Reproduce

```bash
.venv/bin/python scripts/analyze_eval_per_sample.py \
  --input runs/jayavibhav_per_sample_with_evidence.jsonl \
  --out-dir runs/analysis/jayavibhav_with_gate_grid
```

Artifacts: `runs/analysis/jayavibhav_with_gate_grid/` (see `summary.json` → `"outputs"` and `"gate_simulation"`). Override the default gate grid with `--gate-allow-start` / `--gate-block-stop` / step flags if you need a finer or wider sweep.
