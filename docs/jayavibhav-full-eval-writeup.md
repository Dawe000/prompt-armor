# TLDR

This file is in two halves: how prompt-armor works, then how it scored on the full jayavibhav export.

Research half: Five parallel layers: handmade weighted regex (L1); off-the-shelf small transformer classifier (L2); contrastive embeddings plus nearest-neighbour over a large attack database in FAISS (L3, heavy at cold startup unless the on-disk index cache hits); structural / persuasion / obfuscation heuristics (L4); isolation forest on 11 shallow text features trained benign-only (L5). Fusion is mostly a trained logistic meta-model on those scores (plus max, min, interactions, count-above-0.1), with a hard shortcut if any layer is very high, confidence from distance of `risk_score` from 0.5, and an optional LLM "council" in an uncertain band. Notable stack follow-ups: ship or download a pre-warmed L3 FAISS bundle so installs do not re-encode the whole attack list, and decide whether gate jitter (non-deterministic ALLOW/WARN/BLOCK) is worth the eval/debug friction.

Evaluation half: One streamed run over about 327k jayavibhav rows with a simple binary rule: every layer score and the fused `risk_score` compared to 0.5 (a reporting anchor, not the full three-way product gate). Fused result: very high attack recall (about 98%, few thousand false negatives) but low precision on benigns (about three in four benign rows flagged at that cutoff, so large false-positive volume). L3 carries almost all solo attack recall and drives fused behaviour; L2 is the best single-layer F1 tradeoff and disagrees with L3 in useful ways; L1 is high-precision but rare; L4/L5 look weak at 0.5 alone but still feed derived fusion features. Follow-ups: treat 0.5 as tunable, retrain or reweight fusion for your cost mix, improve L3/L2 calibration and data, dump per-sample scores once (`--per-sample`) so threshold sweeps do not need a full re-run.

---

# Prompt armour research

In here, I describe each classifier, what it does, how it was made, and how effective it is (in short).

## L1

Uses a regex of weighted rules (handmade). When a match is made, first use highest weighted rule found using regex, then use context modifiers to dampen hits due to contextual info such as the hit being found inside quotes or within hypothetical framing (this dampening does not occur with strong hits). Scores are also amplified if several strong hits appear.

## L2

Just uses off the shelf `aldenb/scout-prompt-injection-classifier-22m`, not made by person who made prompt armor.

## L3

Uses an off the shelf embedding model (`paraphrase-multilingual-MiniLM-L12-v2`), and does contrastive fine tuning (attack examples in the training data are pushed together, and benign examples are pushed apart. Basically, you take a known attack as an anchor, a second attack as POSITIVE, and a benign prompt as negative, and push the positive and negative so that the positive is at least some margin closer to the anchor than the negative). After this, when making an analysis, just embed every non benign training example (happens at startup time), embed sample being analysed, find closest similarity, run it through a map (0 below 0.55, then 0→0.5 between 0.55 and 0.75, then up toward 1.0 above 0.75) and output.

## L4

Three structural checks:

1. Roughly split prompt into sentences, then label each sentence with something like question, meta, imperative, using cheap simple rules like checking for punctuation or specific words. After this, we go through sentences until we find a sentence with a data label, and after this point we increment a count any time we see a meta, instruction or context switch label (the more, the higher likelihood of attack), especially towards end of prompt.
2. Uses regex to detect authority, urgency, context switch (e.g. this was a test now do this, you must do this or else) etc.
3. Obfuscation such as fake system/user delimiters, role assignment, privilege-style wording, encoding, special characters.

Each check that passes its threshold adds a sub score. The layer output is the max of the scores + a small bump for multiple checks being triggered (not for their individual scores, but the number of evidence lines).

## L5

Standard pre made scikit-learn isolation forest classifier trained only on benign prompts. Prompt is turned into 11 features (word count, char count, sentence count, average word size, average sentence size, imperative-word ratio, question-mark ratio, special-character density, Shannon entropy, uppercase ratio, vocabulary diversity). Looks for anomalies based on this.

## How they're combined

If any layer is above 0.95, final score is 1.0.

Other than that, they build a 10 dimensional feature vector, containing:

- The 5 raw scores
- The max and min across them
- L1×L4
- L2×L3
- How many of the scores are >0.1

Take dot product of vector + intercept, then apply sigmoid function (in the dot product, L4/L5 and min have a coefficient of 0, so they don't directly affect the score. This is to prevent gaming).

Confidence (fusion-level). Depends on how far `risk_score` sits from the meta 0.5 operating point (farther = higher confidence).

Something interesting: if confidence and score are within certain thresholds, it will trigger a "council" path which is a subsequent optional layer that uses an LLM to detect prompt injection.

## Potential improvements (stack / ops)

- Ship or fetch a warm L3 index: Today the FAISS index + metadata live under a local cache dir (default `data/models/l3-faiss-cache/`, gitignored) and are built on first `LiteEngine` init if missing, which means encoding the whole attack JSONL on a cold install. Worth offering a prebuilt bundle (e.g. download from Hugging Face Hub next to the ONNX model), a post-install / CI step (`scripts/warm_l3_cache.py`), or baking the three cache files into images so first startup is not dominated by CPU embedding work. Watch FAISS version and attack file fingerprints so downloads stay in sync with releases.
- Jitter vs determinism: The final gate applies small random jitter around the meta threshold (`fusion.py`: Gaussian noise, σ≈0.03, clamped) so decisions are not fully reproducible for the same `risk_score` on repeated calls. The stated reason is anti-gaming (harder to land exactly on a fixed cutoff). That is a real security tradeoff against deterministic behaviour for tests, logging, and debugging; worth revisiting (e.g. optional "deterministic mode" for eval, or jitter only in production config).

---

# Evaluation

Single-run writeup for the complete jayavibhav/prompt-injection export (train + test) scored with `scripts/eval_pint_layers.py` using `--stream-jsonl`, binary threshold 0.5 on every layer and on the fused `risk_score`. It does not summarize other corpora or small slices.

## 1. What we measured


| Quantity                  | Value                                          |
| ------------------------- | ---------------------------------------------- |
| Samples                   | 327,153                                        |
| Attacks (label 1)         | 161,526                                        |
| Benign (label 0)          | 165,627                                        |
| Engine init               | about 3.4 s (L3 FAISS cache hit)               |
| Mean `analyze()`          | about 67.6 ms (median about 53.4 ms)           |
| Wall time (analyze phase) | about 6.2 h (`total_analyze` ≈ 22.1 ks summed) |


Every row: five layer scores + fusion map to binary "injection" if score ≥ 0.5.

## 2. Results at a glance


| Layer                 | Accuracy | Precision (of what we flagged positive, how much was true positive) | Recall (true positive rate) | F1    |
| --------------------- | -------- | ----------------------------------------------------------------------- | --------------------------- | ----- |
| L1 regex              | 0.528    | 0.958                                                                   | 0.046                       | 0.088 |
| L2 classifier         | 0.672    | 0.735                                                                   | 0.526                       | 0.613 |
| L3 similarity         | 0.521    | 0.508                                                                   | 0.997                       | 0.673 |
| L4 structural         | 0.514    | 0.705                                                                   | 0.027                       | 0.052 |
| L5 negative selection | 0.503    | 0.474                                                                   | 0.057                       | 0.101 |
| Fused (meta)          | 0.602    | 0.555                                                                   | 0.978                       | 0.708 |


False negatives on attacks (missed injection @0.5): L1 154,080; L2 76,510; L3 520; L4 157,144; L5 152,376; fused 3,490.

False positives on benign @0.5: L1 326; L2 30,690; L3 156,212; L4 1,832; L5 10,148; fused 126,579.

### General evaluation

On the full jayavibhav export (about 327k rows, binary cut at 0.5 on every score), the fused stack is strong on catching attacks (about 98% recall, only about 3.5k false negatives) but weak on benign precision: roughly three in four benign prompts are scored as injection at 0.5 (about 127k false positives), so overall accuracy sits around 0.60 and fused F1 about 0.71. That profile is typical when similarity-to-known-attacks (L3) dominates the meta-model: very safe if the cost of a miss is high, expensive if the cost of flagging normal user text is high. Context: this is one external corpus and one operating point; it is a useful stress test, not the only definition of "how good" the product is.

### Layers and contribution

L3 drives almost all attack recall at 0.5 and aligns heavily with fused decisions, but it floods benigns with positives. L2 (the small transformer) is the best single-layer tradeoff (highest solo F1, far fewer benign FPs than L3) but misses many attacks L3 catches, so it complements L3 rather than replacing it. L1 behaves like a high-precision, low-coverage regex channel: when it fires it is usually right, but it rarely fires on this mix. L4 and L5 contribute little as standalone 0.5 classifiers here (very low attack recall), yet they are not "unused" in fusion: their scores still feed max, min, and count-above-0.1 features, while their direct linear weights in the shipped meta-model are clamped to zero, so their marginal effect is indirect and easy to underestimate from the per-layer table alone.

### Potential improvements

Treat 0.5 as a reporting default, not a universal truth: sweep fused (and optionally per-layer) thresholds on a saved score dump or a large validation slice, or retrain fusion with class weights / hard benign negatives so the model does not treat L3's "close to an attack line" as a full positive without corroboration. L3 improvements (thresholding above the index, contrastive data, index quality) and L2 / interaction features (exploiting L2-L3 disagreement) are the highest-leverage knobs; L5 retraining on broader benign and L4 tuning help at the margins. A one-time `--per-sample` eval run pays for itself so you can tune without re-running the full engine on hundreds of thousands of prompts.