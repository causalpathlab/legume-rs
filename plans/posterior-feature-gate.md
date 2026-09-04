# Posterior feature gate — handoff

## Retired — 2026-09-04

Everything this note describes has been removed from the tree, on the branch
that also added `senna embed-diag`. What the removal rests on, and what it
deliberately overrides, so the table below is read in context:

- **The learned gate was measured inert on `senna bge` at its shipped `H=128`**
  (purity within noise of the ungated fit, ~40% slower). The `H=16` numbers in
  the table below predate that and were already marked stale by their own
  warning; the later measurement governs.
- **Gibbs/PIP sampling of the gate was infeasible at bge's scale** (on the order
  of 1e10 updates, no progress after many minutes on a small input), and
  measured to buy nothing where it did finish.
- **`pinto cage`'s sampled arm measured better than its learned arm** on cage's
  own data (12.0x vs 10.3x against a neighbour-agreement null, +33% wall-clock).
  It was retired anyway, on cost and on the rule that a PIP consumer with no
  producer left in the workspace is debris. That number stands and can be
  revisited if cage needs a selector back.
- **`senna gem`'s splice posterior** (`pb_gibbs_splice`, `posterior_hyper.json`)
  and the sampled-pseudobulk write-back were independent of the gate and were
  never refuted on their own merits. They went as collateral for a clean
  boundary and are resurrectable from history.

What replaced the gate's effect prior: plain L2 on the feature side —
`--feature-embedding-l2` on bge (already there), and the same flag on gem, now
ridging the per-gene `β` directly. gem's default is a placeholder pending its
own A/B. The frozen-side participation ratio the sampler diagnostic reported
lives on as `matrix_util::embedding_geometry` behind `senna embed-diag`.

Everything below is the historical record.

---

Started 2026-07-30. Session 2 (same day) closed all 15 review findings and did the
legacy/modularization pass. Everything below is measured or read off the code; where
something is an assumption it says so.

## Where the tree is

Session 1 left three commits with **15 confirmed defects** (xhigh multi-agent review,
67 agents). Session 2 added six commits that close all of them:

| commit | what |
|---|---|
| `6069c261` | `gate_kl`: sharing heads get `π_h`; the effect KL survives jitter (findings 1, 7) |
| `c6d6848d` | one phase-1 block, selection sampled before it (findings 2, 3, 4, 5, 6, 11, 14) |
| `ff1b715f` | deleted the composite modes and samplers nothing could reach |
| `91c50a65` | ship the mask the fit trained under (findings 9, 10, 12, 13, 15) |
| `f4459485` | `model.rs` split into `model/{mod,gate,score,vars}.rs` |

Workspace suite exit 0, clippy 0, fmt clean at each one.

## What the design is

```
--posterior   →  SELECTION PASS: Gibbs samples the per-(feature,dim) `pip`
then SGD      →  fits the LOADING under a mask drawn from that pip
```

The sampler chooses *which* features load each dim; SGD fits *how much*. Training
multiplies by `z ~ Bern(pip)` redrawn once per EPOCH (never per minibatch — `z` is a
latent for the dataset); output uses the mean `pip`, since `E[z] = pip`.

**The selection pass now runs BEFORE phase 1**, not after, and there is exactly ONE
training block. That ordering is what removes the Ctrl+C hole, the gem double-gate, and
the statically-dead `if !jitter` arm — see `fit/selection.rs`, which holds the whole
pass plus seven tests.

`GateKind {Identity, Velocity}` exists because gem carries two gates (β and δ) and the
weight lookup takes a `Tensor` of logits that cannot be identified at the call site.
Without it `factored_feat_rows` hands δ the β mask — measured means differ ~9× on
rep2_wt (0.014 vs 0.125), so it is a silent ~9× error, not a rounding one.

## Why a learned gate cannot replace this

Built first, measured, failed (BM1, 34,008 genes × 16 dims):

- the KL that must drive selection sits **~70×** under the true ELBO
  (`1/batch_size = 9.8e-4` vs `D·H/N = 0.068`) — meaned over `D·H` *and* divided by `B`
- any init where the gate is inert (`σ(4) = 0.982`) puts every logit where
  `∂α/∂S = α(1−α) = 0.018`, **1/14** of the available gradient
- **12,620 of 34,008 genes (37%) receive exactly zero gradient**, all 16 dims frozen at
  the init, because NCE minibatches never draw them

The third is decisive and unfixable by tuning. The Gibbs column pass touches every
anchor on every sweep, so it has no such blind spot. That asymmetry is the entire
argument for this design. This reasoning now lives in `fit/selection.rs`'s module doc.

## Measurements (BM1 `Ainciburu2022_young1`, 2627 cells, H=16, CUDA)

3 seeds, paired. kNN label purity k=30 vs `CellType_Broad`; effective genes/dim =
participation ratio of `dictionary.parquet` columns.

| arm | purity | eff #genes/dim |
|---|---|---|
| `pip ⊙ β` (mean held) | **0.6739 ± 0.0074** | 1,531 |
| plain SGD | 0.6664 ± 0.0068 | 8,785 |
| `z ~ Bern(pip)` per epoch | 0.6632 ± 0.0069 | **836** |
| softmax gate (old `main`) | 0.638 (1 seed) | 849 |

Between-seed spread exceeds between-arm, but paired by seed the ordering is identical
3/3. **The stochastic draw is the shipped default by decision on minimum description
length** (~10.5× fewer effective parameters for a fit inside the seed noise), NOT
because it won on purity — it did not.

⚠️ **These numbers predate session 2 and no longer describe the shipped code.** Four
things that changed can move them, all in the same direction (the fit now sees
regularisation and a mask it previously did not):

1. `gate_kl` returned `None` on every ordinary run, so the plain-SGD arm was fit with
   NO gate regularisation at all;
2. the jitter arm dropped the Gaussian effect KL on β entirely;
3. the sampler ran after phase 1 and wrote gated `E[z·β]` into the Vars, so the gem
   arm was double-gated;
4. an ungated model shipped an unmasked dictionary while training masked.

**Re-run the sweep before quoting any of this.** Keep the table for the shape of the
question, not the values.

Numbers still worth not re-deriving:

- CUDA is **~20×** on phase-1 SGD (1000 epochs: 28 s vs ~10 min). `senna` needs
  `cargo build --features cuda`; a plain `--workspace` build silently replaces the
  binary with a CPU-only one.
- bge posterior: 45 sweeps = **214 s**. gem is ~3× per sweep (3 term-passes vs 1).
- Sparsity mostly lives in the MULTIPLIER, not β. Divide the gate out and all three
  fits sit at Gini 0.51–0.63.
- Effective rank of the cell embedding is 3.9–5.4 of 16, and 2.61 of 32 on a joint
  3-replicate gem fit. **`--embedding-dim` defaults to 128.**

## Findings — ALL 15 CLOSED

Each fix carries a test that was verified to FAIL against the defect before it was
applied (the plan's own rule: don't trust compilation).

| # | what | where it is pinned |
|---|---|---|
| 1, 7 | `gate_kl` `None` on default runs; effect KL dropped under jitter | `model/tests.rs::a_sharing_head_still_has_a_gate_kl`, `::jitter_keeps_the_effect_kl_and_drops_the_selection_terms` |
| 2, 3, 6, 11 | phase-1 restructure (Ctrl+C hole, dead arm, double-gate, false banner) | structural — the second training block no longer exists |
| 4 | δ mask `Arc` never shared | `fit/selection/tests.rs::one_draw_reaches_every_axis_on_both_gates` |
| 5 | `delta_pip` NaN scrub was a no-op | `::an_unidentified_delta_is_masked_off_not_gated_at_the_prior` |
| 9 | `feature_selection.parquet` a byte copy of `feature_pip` | `::a_pip_suppresses_the_learned_selection_table` |
| 10 | ungated model trains masked, ships unmasked | `::an_ungated_model_ships_the_mask_it_trained_under` |
| 12, 13 | gate log announced the deleted softmax design; lost continuations | log text |
| 14 | zero tests on the mask plumbing | seven tests in `fit/selection/tests.rs` |
| 15 | pip re-uploaded per model | `install_gate_pip` takes a device tensor |

## Legacy removed (session 2)

~1,150 lines, all verified unreachable rather than merely unused:

- `CompositeMode::{Sample, Chain}` — `TrainingParams` has ONE constructor and it
  hard-codes `Sum`; `Chain` also needed a `cell_to_pb_per_level` every context passes
  as `None`. Took `sample_step`, `chain_step`, `sample_chain_batch_stratified` and the
  whole `Chain*` family in `loss::feat`.
- `AxisSampler::PerBatch` — matched twice, constructed never. Took
  `UnifiedData::materialize_cell_triplets`, whose only mentions in the workspace were
  the two dead arms' error strings telling the reader to call it.
- `loss::feat::build_per_batch_stratified_cell_samplers` — superseded duplicate of
  `fit::samplers`'s, kept only by an `#[allow(clippy::too_many_arguments)]`.
- the six write-back helpers the restructure orphaned (`overwrite_feature_side`,
  `scatter_gene_to_rows`, `write_back_splice`/`write_back_posterior`, …).

**The structural fix that found them:** `loss::{cell,chain,feat}` went from `pub mod`
to `pub(crate) mod`. rustc assumes an external caller for a `pub` item in a `pub`
module, so ~800 unreachable lines sat there with the build green. Every consumer
already used the `loss/mod.rs` re-exports, so nothing broke. **Apply the same rule to
any new module.**

## File sizes — target is < 1000 lines each

`model.rs` 1573 → `model/{mod 486, gate 722, score 193, vars 230}`.
`loss/feat.rs` 1058 → 644. `training.rs` 703 → 480. `fit/mod.rs` 1478 → 1045.

Session 2 also split `type_annotation/term_ora.rs` (2109),
`posterior/pb_gibbs.rs` (1453) and `fit/projection/block_sgd.rs` (1494) — check
`git log` for the outcome.

The rule that made these splits safe: **group so that no cross-module call needs a new
`pub`.** Where a marker is unavoidable use `pub(super)` — for an item in `model::vars`
that is exactly the visibility a private item in the old single-file `model` had, so it
is not a widening.

`fit()` in `fit/mod.rs` is still ~950 lines in ONE function. That is the remaining
maintainability problem in the crate. Clean extractions (each ≤6 honest parameters):
the projection, the collapse, the pb blobs, the model build, the axis build, the epoch
budget, the phase-1 SGD. Two chunks have their seam in the wrong place — the lineage
refine needs 13 bindings, and phases 13/14 need 7 each; both drop to ~5 if
`collapsed_levels` + `cell_to_pb_per_level` stay bundled in the `CollapseOut` struct
that phase 2 already returns, instead of being unpacked at the top and passed
separately for the next 900 lines.

## Still outstanding

- **Re-run the 3-seed sweep.** See the ⚠️ above — the shipped numbers describe code
  with four defects in it.
- Move the selection pass onto the **default** SGD path at **10 sweeps**, logged as a
  SELECTION PASS — no split-R̂, no `posterior_hyper.json`. 10 sweeps will not converge
  and does not need to: at 45 sweeps the chain sat at worst R̂ 2.96 / min ESS 2.7 and
  still gave a usable mask. (The log line already says SELECTION PASS and says it need
  not converge; the default and the sweep count are not changed yet.)
- `--posterior` back to **exclusive** in bge; **removed entirely** from gem (hours at
  h=128 — the flag, four output tables, three exclusivity hard-errors, R̂/hyper-JSON
  and a long help block; the splice sampler stays, the SGD path needs both pips).
- **Delete the learned gate.** ⚠️ It is NOT dead: `enable_feature_gate` runs on every
  bge and gem invocation, and `gathered_gate_weights` falls through to the learned
  logits on every run without `--posterior`. Deleting it means deciding that the
  selection pass becomes the default (the item above) or that ungated SGD is the
  non-posterior default. That is a design call, not a cleanup. The full symbol
  inventory — every field, method, const, VarMap key, output table, test and CLI flag
  belonging to the learned gate alone — is in the session-2 survey; regenerate it
  rather than guessing, and note `GATE_EFFECT_PRIOR_VAR`, `GATE_LOGSTD_CLAMP`,
  `GATE_PI_EPS`, `effect_kl`, `sample_effect`, the `*_logstd` tables and `gated_rows`
  are shared with the pip gate and must NOT go.
- `FitConfig` is really two disjoint configs glued together: nine fields are inert on
  the bge path and nine on the gem path (senna hard-codes `feat_factor: None`,
  `lineage_dag: false`, `delta_l2: 0.0`, …; faba hard-codes `feature_embedding_l2:
  0.0`, `block_size: None`, `num_negatives: 4`, …). Worth an enum or a two-struct
  split.
- `feature_gate` is **never `None`** from either CLI and `temperature` is **always
  1.0** (`--feature-gate-temp`, `hide = true`, `default_value_t = 1.0` in both). So
  `FeatureGateSpec` carries zero information and `apply_temperature` is an `Arc`-clone
  no-op. Collapses to nothing once the learned gate goes.
- `GATE_FOLD_EPS` (`fit/projection/block_sgd`) is `0.0`, so `norm > GATE_FOLD_EPS`
  folds only exactly-zero rows. It reads as tunable and is not — inline the `> 0.0` or
  make it configurable.
- `src/ash.rs` (250) + most of `src/null_call.rs` (~280) are production-dead,
  reachable only from their own tests. The one live export is `live_row`, used by five
  `type_annotation` modules; move it there and both files go, along with
  `src/ash_tests.rs` and `tests/null_call.rs`. **Judgement call** — this is working
  statistical machinery, not broken code.
- Seven small dead `pub` items (each appears exactly once in the whole workspace):
  `ANNOT_OUTPUT_SUFFIXES`, `AxisCoarsenings::avg_n_coarse`,
  `ContrastiveIndex::{informative_anchors, n_empty_anchors, node_terms}`,
  `UnifiedData::{n_conditions, subset_cells}`.
- Rename (below), `posterior/run.rs` long_help.

## Naming — one concept per word

| now | → | is |
|---|---|---|
| `gate_pip` | `pip` | `P(z=1 \| data)`, from the sampler |
| `gate_mask` | `z_epoch` | this epoch's binary draw |
| `resample_gate_mask` | `redraw_z` | draw `z ~ Bern(pip)` |
| `clear_gate_mask` | `clear_z` | revert to the mean for output |
| `gate_weights` | `feature_multiplier` | whatever currently multiplies β |
| `delta_gate_pip` | `delta_pip` | δ's own probabilities |

Then the invariant is one line: **`feature_multiplier` returns `z_epoch` during
training and `pip` at output, because `E[z] = pip`.**

Do it **one symbol at a time** with `cargo check` between — not one sweep.

## How this went wrong, so it doesn't again

Four defects reached `main` with the build green:

1. a regex stripping an A/B flag ate the `resample_gate_mask()` call — `pub` methods
   don't warn when unused, so the gate ran deterministic while logging that it wasn't
2. `materialize_e_feat()` ran BEFORE the gate block; without a re-materialize the
   whole pass is invisible and phase 2 projects against a stale dictionary
3. `gate_kl` silently returning `None` on default runs
4. the δ mask `Arc` never shared

**None had a test.** `cargo build` and `cargo test` were green for all four.

Rules that would have caught them, and that session 2 used throughout:

- after any mechanical edit, **grep for a call site** — do not trust compilation
- **write the test, then break the fix and watch it fail.** Every session-2 fix was
  verified this way. It caught nothing new, which is the point: it is cheap and it
  converts "I believe this is a regression test" into "this is one."
- keep modules `pub(crate)` unless something outside the crate genuinely needs them —
  see the `loss::` note above; it is what turned ~800 dead lines from invisible into a
  compiler warning
- when a doc and the code disagree, that is the bug, not a style issue
- prefer LSP `findReferences` over `sed`; note the LSP tool here is READ-ONLY (no
  rename) and returned a false "no references" once, so verify it before relying on it
- **another session may be editing this repo.** Session 2 found live edits in `pinto/`
  and a `Cargo.lock` bump from elsewhere. Scope every `git add` to your own paths and
  never `git commit --amend`.

## Reproducing the measurements

```bash
# CUDA build (a plain --workspace build silently drops it)
cargo build --release -p senna --features cuda

BM=~/work/paper-senna/data/BoneMarrowMap/Ainciburu2022_young1.zarr.zip
./target/release/senna bge -v $BM --out OUT --embedding-dim 16 --seed 42 \
    --skip-etm --device cuda [--posterior 30]
```

`--skip-etm` is REQUIRED to get `feature_selection.parquet` — bge's default path calls
`resolve_etm_topics`, which never calls `save_outputs_named`. Note that table is now
suppressed under `--posterior` (finding 9): read `feature_pip.parquet` there.

Analysis in R (arrow/uwot/data.table/ggplot2/FNN). The feature dictionary is
`dictionary.parquet`, **not** `feature_embedding.parquet` — reading the wrong one
reversed a recommendation mid-session and caused most of session 1's churn.
`cell_embedding` rows must be L2-normalised before UMAP.
