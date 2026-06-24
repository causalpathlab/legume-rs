# Batch correction for gem + bge via a pseudobulk-keyed log-offset (both phases)

## Decisions (locked)

1. **Granularity:** per **(gene, modality)** — compute the fold-factor separately
   on each backend (genes + each satellite). Each assay gets its own correction.
2. **Keying:** **pseudobulk-keyed `μ_residual[gene, pb]`** (topic's
   `AdjMethod::Residual`), not batch-keyed δ. Per-cell offset uses the cell's
   pb-group fold-factor; per-pb training uses that pb's directly.
3. **Scope:** **always on, both phases.** The same offset enters the phase-1 NCE
   rate AND the phase-2 projection rate. Degrades to a no-op when `n_batches == 1`
   (no cross-batch match ⇒ `μ_residual ≈ 1` ⇒ `log ≈ 0`), so default-on is safe
   for single-batch data; `--ignore-batch` remains the explicit escape.

## Goal

Make every Poisson rate in the bipartite path — phase-1 NCE positives/negatives
and phase-2 per-cell projection — carry a known per-(pseudobulk, feature)
batch fold-factor as a log-offset, so neither the feature dictionary (phase 1) nor
`e_cell` (phase 2) absorbs a batch effect that lives in the raw counts.

This is the analytic twin of `senna topic`: topic divides counts by the residual
fold-factor inside its encoder (`anscombe_residual`); we add `log μ_residual` to
the linear predictor everywhere a rate is formed.

## Core math

Everywhere the model forms a rate for (axis element *a*, feature *f*):

```
log λ = ⟨e_a, e_f⟩ + b_a + b_f + o ,   o = log μ_residual[ modality(f) ][ gene(f), pb(a) ]
```

- *a* = a pb at level ℓ (phase-1 pb axis), or a cell (phase-1 cell axis / phase-2).
- `pb(a)`: for a pb axis element, itself; for a cell, its pb group (finest level).
- `o` is constant w.r.t. the solved/learned params, so it only shifts the linear
  predictor — phase-2 IRLS gradient/Hessian and phase-1 backprop are structurally
  unchanged.
- Mean-center `log μ_residual` across pb per (gene, modality) so `b_f` keeps the
  global feature level and the offset is purely batch-relative. Floor the
  fold-factor away from 0 before `log`.

## Per-modality, shared partition (gem)

bge is single-backend (features = raw genes) ⇒ one `μ_residual[gene, pb]`.

gem features are interned `Identity {gene, q_modality, region, is_agg}`
(`gem/cell_solve.rs:23`) over **separate backends** (genes = spliced+unspliced;
satellites = dartseq/m6A, atoi, apa). Compute `μ_residual` **per backend**, all on
the **one** pb partition gem already builds in spliced projection space (satellites
matched by `barcode@sample`). Map identities → backend:

| Identity stratum | μ_residual backend |
|---|---|
| AGG, count-comp (splice) | genes |
| modifier-comp (modality *m*) | satellite *m* |

`region` collapsed: one fold-factor per (gene, modality), shared across bins.

## Quantity source — reuse topic's collapse output

`μ_residual` is already produced by the collapse optimize; we just stop throwing
it away and run it per backend.

- Cross-batch counterfactual: `collect_matched_stat_visitor`
  (`data-beans-alg/src/collapse_data/stats.rs:26`) → `imputed_sum_ds` (y1_hat),
  `residual_sum_ds` (observed/counterfactual).
- `μ_residual` fit in `optimize_block` (`stats.rs`) → `GammaMatrix[D, n_pb]`;
  `posterior_mean()` → `DMatrix<f32>[D, n_pb]`.
- Per-cell application reference (mirror this indexing):
  `senna/src/topic/eval.rs:18` + `expand_delta_for_block`
  (`senna/src/topic/common.rs:73`) — `AdjMethod::Residual` selects `μ_residual`,
  `block_membership` returns each cell's **pb group id**, `index_select` pulls its
  column.

gem's collapse (`collapse_columns_multilevel_with_hierarchy`, MeanOnly) currently
`drop`s `levels` and emits no `μ_residual`. We retain per-level `CollapsedOut`
and add a per-satellite pass over the **same** partition.

**Deliverable (Stage 0):** in `data-beans-alg`,
```
fn mu_residual_for_backend(backend, batch_membership, partition_per_level,
                           matched /* reuse spliced kNN */) -> Vec<GammaMatrix> // per level [D, n_pb]
```
gem calls it per backend; bge calls it once. Cells need `cell→pb` (finest level),
already in `cell_to_pb_per_level`.

## Implementation stages

**Stage 0 — μ_residual provider** (`data-beans-alg`). Factor the matched-stat +
`μ_residual` step into a reusable per-backend, per-level call over an existing
partition + the spliced-defined cross-batch matching. Return `[D, n_pb]` per
level. Add log + per-(gene) mean-centering across pb; floor.

**Stage 1 — offset in the shared phase-2 solver**
(`graph-embedding-util/src/cell_projection.rs`).
- `solve_one_cell(feats, offsets: Option<&[f32]>, ...)` — `offsets` parallel to
  `feats`; `s = bf + offset + theta[h]` at L109 (0.0 when None).
- `project_cells(..., per_cell, per_cell_offsets: Option<&[Vec<f32>]>, ...)` —
  one offset Vec per cell, aligned with `per_cell`; rayon closure (L64) indexes it.
  Solver stays agnostic; caller derives every offset.

**Stage 2 — offset in the phase-1 NCE rate.**
- `graph-embedding-util` loss (`loss/mod.rs`, `loss/feat.rs`): when scoring a
  (pb/cell, feature) positive/negative, add `log μ_residual_ℓ[gene, pb]`. Thread a
  per-level offset lookup into the loss; keyed by the positive's axis (pb) id +
  feature gene.
- gem training (`gem/train.rs`, scoring in `gem/model.rs` /
  `gem/sampling.rs`): same, with the per-backend table chosen by the identity's
  modality. The pb axis element's id *is* `pb(a)`; the cell axis uses finest pb.

**Stage 3 — gem phase-2 wiring** (`faba/src/gem/cell_solve.rs`,
`run_gem_embedding.rs`).
- Build per-backend, per-level `μ_residual` in `run_gem_embedding` (unified +
  satellites + partition in scope); pass an `OffsetTable` into phase-2.
- Pooled `solve_cell_embeddings:42`: extend `collect_identities` to emit, per
  pushed `(id, count)`, `log μ_residual[modality][gene, finest_pb(cell)]`; thread
  `cell_to_pb_finest: &[u32]` in.
- Streaming `solve_cell_embeddings_streaming:282`: `CellAccum` pushes an offset
  parallel to each `(id, v)`; cell id → finest pb → lookup.

**Stage 4 — bge wiring** (`graph-embedding-util/src/fit/projection.rs`,
`senna/src/bge/`). Single backend.
- Phase-1: add the offset in the NCE loss (Stage 2) keyed by each positive's pb.
- Phase-2 `project_cells_phase2:33`: emit one offset per `(gene, count)` as cells
  are assembled (L53-71), keyed by the cell's finest pb. Surface `cell→pb` from
  the internal collapse (it's discarded today).

**Stage 5 — config + versioning.** Default-on; honor `--ignore-batch`
(⇒ no offsets). Single-batch ⇒ provider returns `≈1` ⇒ no-op (assert in a test).
Bump patch on `graph-embedding-util`, `faba`, `data-beans-alg`, `senna`; refresh
`Cargo.lock` ([[versioning-convention]]).

**Stage 6 — validation.** Sim with an **injected per-modality** batch fold-factor
(distinct for spliced vs m6A on the same genes). Confirm:
- phase-2 `e_cell` batch mixing improves (centroid ref-vs-rest → rest-vs-rest,
  the condition-gate metric) **without** degrading topic recovery on the clean
  case; and that single-batch is a numerical no-op.
- Use a **confounded** regime — within-batch NCE negatives already neutralize a
  plain multiplicative effect (why the condition gate read neutral and was
  reverted, [[senna-bge-condition-gate]]); the win must come from the harder case.
- Round-trip / projection sanity on BM1 / gem_test.

## Key file:line references

- Solver + offset: `graph-embedding-util/src/cell_projection.rs` —
  `project_cells:54`, `solve_one_cell:87`, predictor `108-112` (insert `109`).
- Phase-1 loss: `graph-embedding-util/src/loss/{mod,feat}.rs`; gem
  `faba/src/gem/{train,model,sampling}.rs`.
- gem phase-2: `faba/src/gem/cell_solve.rs` — `solve_cell_embeddings:42`,
  `solve_cell_embeddings_streaming:282`, `collect_identities:404`, `Identity:23`.
- gem partition / batch: `faba/src/run_gem_embedding.rs` (`unified.batch_*`,
  `build_pseudobulk`, `cell_to_pb_per_level`).
- bge phase-2: `graph-embedding-util/src/fit/projection.rs:project_cells_phase2:33`.
- μ_residual: `data-beans-alg/src/collapse_data/stats.rs`
  (`collect_matched_stat_visitor:26`, `optimize_block`, `CollapsedOut:414`);
  apply pattern `senna/src/topic/eval.rs:18`,
  `senna/src/topic/common.rs:expand_delta_for_block:73`.

Related: [[gem-bge-phase2-batch-offset]], [[senna-bge-condition-gate]],
[[gem-l2-normalize-ecell]], [[faba-gem-multimodality-union-tag]].
