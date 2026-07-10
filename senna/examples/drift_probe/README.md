# Drift-probe hold-out experiment

Can `senna probe` detect a cell state the reference model was **never trained on**,
without crying wolf on a mere change in cell-type proportions?

```sh
cargo build --release -p senna -p data-beans -p data-beans-sim
./run.sh /tmp/drift_probe_run          # ~4 min (250 training epochs)
```

Needs `Rscript` with the `arrow` package.

## Design

A novelty detector is only meaningful against controls it must **not** fire on, so the
experiment feeds three batches — one to catch, two to leave alone.

`data-beans-sim topic --holdout-topics 5` simulates K=6 topics and routes every cell whose
dominant topic is topic 5 into a **second backend** (`sim.holdout.zarr`). The reference is
trained on the first backend only, so it *provably* never sees the held-out topic — the
hold-out is a property of the data, not of a post-hoc filter. Ground-truth `β` and `θ` are
still written in full, so every call can be scored against truth.

| batch | what it is | expected |
|---|---|---|
| **A** | held-out reference cells (topics 0–4) | **not** flagged — calibrates the null |
| **B** | reference cells resampled to enrich topic 2 (covariate shift) | **not** flagged as novel |
| **C** | the held-out topic 5 | **flagged** as novel |

Batch **B** is the discriminating control: a naive detector that keys on "this batch looks
different" fires on it. A per-cell reconstruction score should not, because covariate shift
changes the *mixture* of cells, not any individual cell's reconstructability.

## What is measured

1. **Fit score** — `senna predict` writes `{out}.predictive.parquet` with per-cell
   `llik_per_count`: the predictive log-likelihood per UMI under the frozen model. Lower =
   worse fit = candidate novelty. `analyze.R` computes `AUC(A vs C)`, `AUC(A vs B)`, and a
   titration curve (mixing A and C scores) that gives the **detection floor** — the smallest
   novel fraction a batch can contain and still be caught.

2. **Verdict** — `senna probe --calibration A.zarr` sets a null from the in-distribution
   control, flags query cells below its lower tail, and emits a batch-level
   `COVERED` / `NOVEL` call with a one-sided p-value (`{out}.probe.json`).

3. **Counterfactual axes** — `--influence` additionally estimates `τ_new` (predicted benefit
   of updating on this batch) and `τ_old` (predicted change on the old data; `< 0` means
   forgetting), via the closed-form gradient and empirical Fisher of the fit objective.
   See the `influence` module docs for the estimand, the control-arm contrast, and caveats.

## Reading the counterfactual output

`τ` scales like `1/κ` (`--prior-strength`), so absolute values are not comparable across κ.
The κ-stable readouts are the **sign of `τ_old`** and the ratio `τ_new/|τ_old|`. The
principled prior strength is `κ ≈ n_train / n_query`; the default of `1` holds the old model
with the weight of a *single cell* and is for ranking only.

By construction `τ_new = τ_old = 0` when the query **is** the calibration set — an exact
null, which is what makes the axes usable as statistics.

## Observed result

From a clean `./run.sh` (2000 genes × 6000 cells, K=6, 250 epochs, seed 42):
5030 reference cells / 970 novel; `reftrain`=4024, `A`=1006, `B`=366 with topic-2 share
lifted 0.182 → 0.500.

| | A (in-dist) | B (covariate shift) | C (held-out topic) |
|---|---|---|---|
| verdict | COVERED | **COVERED** | **NOVEL** |
| flag rate (< A's p05) | 0.051 | 0.033 | **1.000** |
| mean fit (log-lik/count) | −7.248 | −7.235 | **−7.570** |
| `τ_new`, `τ_old` (κ=1) | **0, 0** | +1.26, −1.01 | **+14.11, −16.82** |

- `AUC(A vs C) = 1.000` — perfect separation of the never-seen topic.
- **B is not flagged** (0.033 ≤ α=0.05) and is certified COVERED: a pure change in cell-type
  proportions does not look like novelty, because the score is per-cell.
- `τ_new` is ~11× larger for C than B, and A lands on the **exact null** (0, 0).
- Titration: flag rate 0.05→0.115, 0.10→0.133, 0.25→0.287, 0.50→0.525, 1.0→1.000, so the
  **detection floor is ≈5–10%** novel cells.

> **Do not read `AUC(A vs B) ≈ 0.5` as the criterion.** Here it is 0.389 — B fits *better*
> than A, because this run's topic 2 happens to reconstruct above average, so enriching it
> lifts B's mean fit. The AUC can drift either side of chance depending on the enriched
> topic; what must hold is that B stays **below the flag threshold**. Judge the controls by
> flag rate and verdict, not by AUC.

`senna predict` emits a `--adj-method residual` warning here; it is pre-existing and does not
affect these calls.

## Caveats

- The simulation has a **single batch** and no ambient contamination, so it never stresses
  technical confounding. On real data the fit score will also fire on technical outliers;
  separating those from biology needs the de-confounding step (housekeeping / negative
  controls), which this experiment does not exercise.
- `τ` is a **screening statistic**, not a calibrated second-order prediction: the curvature
  used is the *empirical* Fisher, which only coincides with the Fisher at a well-specified
  optimum (Kunstner, Hennig & Balles, NeurIPS 2019).
- `partition.R` uses R's RNG, so the exact cell split is seed-dependent; the conclusions
  below are not.
