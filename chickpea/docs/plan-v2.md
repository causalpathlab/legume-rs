# chickpea v2 plan: fast phase 1, RNA-only cells, stable links

## Context

chickpea 0.4.0 (branch `ypp/chickpea-afresh`) runs end to end, test
first, 37 tests. The PBMC 10k smoke run (SIGINT at phase-1 epoch 30) wrote every table
in 24 min, 6.9 GB. **We are almost there.** Three things stand between this and a real
run, all seen in that smoke run:

| Stage | Time | Problem |
|---|---|---|
| Phase 1 (hier softmax) | 17.6 s/epoch → ~4.9 h for 1000 | a 27,850-gene near-empty group, single-cell units, ~2 of 10 cores |
| Phase 2 (cell encoders) | ~10 min | encodes and refines both tracks; only cells are needed |
| Link attention | 2.3 min | loss rose 91 → 451 (diverged) |

Everything chickpea itself adds (cis pairs, ATAC-gene track, fold-in, tables) is ~2 min.
Gene rows are already good at 30 epochs (Azimuth l1 markers group cleanly by type);
folded-in peak rows sit apart from gene rows in raw coordinates (same-type nearest-gene
0.44 vs 0.15 chance; mean rows anti-aligned, cos −0.87; centring alone does not close it).

`chickpea/docs/plan.md` keeps the v1 record.

## 1. Phase 2: cell embedding from RNA only (first version)

Phase 2 exists only to place cells. v1 places them by RNA alone.

Phase 2 trains one encoder per count track in two stages: **distillation** (regress
each pseudobulk's phase-1 embedding from its counts; ~1,500 rows, 40 epochs, ~1 min per
track) then **refine** (keep training the same trunk on every cell's Poisson likelihood
against the frozen gene rows; 10 epochs over all cells, the 7.7 min on PBMC). v1 keeps
distillation for the RNA encoder only and drops the refine.

- chickpea `p2g/tracks.rs::gene_tracks`: the peak-aggregated track gets
  `is_count: false`. `count_tracks()` then yields `[0]`, so only an RNA encoder is
  distilled (`fit/projection/encoder.rs:1140`) and cells are encoded from RNA
  (`encoder.rs:1256`, which already uses encoder tracks only). Phase 1 never reads
  `is_count`, so both tracks still train the gene rows.
- The trunk refine scores the likelihood over ALL tracks (`encoder.rs:909`,
  `all_tracks`), so the ATAC Poisson would still run. graph-embedding-util: add
  `FitConfig::phase2_refine_epochs` (default 10 = today's `REFINE_EPOCHS`, so bge and
  gem are unchanged); chickpea sets 0. Distillation stays.
- Later (v2, not now): one encoder over RNA ⊕ π-folded ATAC (peaks → genes through the
  fixed attention shares), which needs phase 2 moved after the attention.

Expected: phase 2 ~10 min → ~1 min on PBMC.

Tests first: `tests/tracks.rs` (base track is the only count track); a
graph-embedding-util test that `phase2_refine_epochs = 0` skips the refine and still
returns finite cell rows; `tests/embed_two_track.rs` still separates the programs.

## 2. Phase 1: an exact fix for the slow softmax

Phase 1 is attention-like: each unit scores modules, then the genes of up to K=8
modules drawn in proportion to its counts. Its cost per step is
`units × Σ(drawn module sizes) × H × tracks`. The module partition
(`data_beans` feature coarsening, via `module_partition`) put 27,850 of 36,601 genes
into ONE "background" module (no grouping evidence). Any unit that draws it pays a
27,850-wide softmax (~27,850 × 128 × 2 flops forward per unit and draw), against ~9
genes for an informative module at 1024 modules.

Gene modules come from the same data-beans feature coarsening bge uses, but chickpea
asked for 128 while bge's default is 1024 (`senna/src/bge/driver.rs:652`); 128 came
from a stale reading. The background group does not depend on that count: genes
failing the homogeneity test (`informative_features`) form ONE group before k-means
sees the rest (`data_beans` `feature_coarsening.rs`). At 1024 modules the 8,751
informative genes form ~1,023 groups of ~9 and the 27,850 background genes still form
one, so the background's share of the cost grows. bge on this data has the same
bottleneck.

**Checked on PBMC RNA (read-only, `data-beans stat -s row`):** the background is the
near-empty genes. 27,867 genes are detected in ≤5% of 11,898 cells (6,884 in none), and
the 27,850 least-detected genes carry 3.96% of all counts. So a unit draws the
background in ~28% of cases (1 − 0.96⁸), not most: it is a real but partial cost
(roughly 25–40% of phase 1, to be measured), not the whole of it. And attention over
these genes resolves nothing.

Recommended, in order (each measured before the next):

0. **chickpea defaults to 1024 modules**, matching bge (`TwoTrackConfig::default` and
   `--feature-modules`).
1. **Measure first.** Log per epoch the time split of a step (module level, gene level
   by module size class, backward, optimizer) and the background's count share
   (computable from `UnitModules`). This locates the other 60–75% before any change.
2. **No gene-level softmax over near-empty genes.** Either drop them from the axis up
   front (gene QC by detection, both tracks, in chickpea before the track is built), or
   keep them at the module level only, with the background as a module-only group
   (row = module row, bias = closed-form count share, no member softmax; the existing
   module-only machinery, which today requires a one-track axis). Start with the
   drop: simplest, and these genes cannot carry links anyway. Log how many genes and
   what count share are dropped.
3. **Threaded step** (as #74 had): split a step's units into disjoint slices solved on
   separate threads, gradients summed. Same SGD, uses the 10 cores (today ~2).
4. **Only if 2 and 3 fall short:** approximate attention for large modules, either a
   sampled softmax within the module (true NCE, K sampled members) or a random-feature
   normaliser (linear / low-rank attention). Both approximate the log-normaliser; the
   exact split is preferred while it suffices.

Also: `phase1_cells_per_pb` (16 today) makes 11,469 of 13,133 units single cells;
compare 16 vs 4 once 2 and 3 are in.

After any graph-embedding-util change: bge sim check (ARI 1.0 at pve 0.8) plus the
graph-embedding-util and senna tests, and patch-bump graph-embedding-util.

## 3. Lightweight peak → gene gates inside phase 1 (replaces the softmax link layer)

The gene's ATAC track is pooled from its cis peaks through learned gates instead of the
fixed ABC weights, and the gates train in phase-1 SGD with the embedding:

    a_ug  = Σ_{p ∈ cis(g)} w_gp · n_up                 (pooled ATAC of gene g, unit u)
    w_gp  = σ( θ₀ + θ₁ · log contact(d_gp) + θ₂ · corr_gp )

- Lightweight: a few shared scalars θ; no per-pair parameters, no peak rows, no content
  term. `corr_gp` is the pseudobulk RNA–ATAC correlation of the pair, computed once.
- Sigmoid, not softmax: weights do not compete; w ≈ 0 is "no link".
- Signal: counts. The ATAC-gene row is the RNA row plus a ridge-shrunk low-rank offset,
  so the gates pool the peaks that make a gene's ATAC shares match its RNA shares.
- Gauge: phase 1 scores shares, so a common scale on all gates is invisible; fix θ₀ (or
  normalise per gene) so it cannot drift.
- Cost: re-pool each step's units from their peak counts through the gates: per unit,
  its nonzero peaks × genes per peak; ~1e5 per cell unit, small next to the softmax.
- Engine change (graph-embedding-util): a track whose unit counts are pooled from a
  second sparse table through a gated pair map, with gradients into θ. Off by default.
- The links are w_gp; the old softmax attention and its fold-in-based content term are
  dropped. The peak fold-in stays for the peak embedding.

Tests first: pooling equals hand-computed sums; autograd matches finite differences
for θ; a planted case where one feature (distance or correlation) marks the true
peaks recovers its sign; w stays finite and the common scale does not drift.

## 3b. (Superseded) Link attention: stop the divergence

The attention loss rose 91 → 451 on PBMC (fell on the sim). Leading cause: gene rows ρ
(softmax fit) and peak rows φ (one-step least squares, φ = (ẼᵀSẼ)⁻¹Ẽᵀn) are the same
kind of object in different coordinates, so ρ·Mφ starts far off and lr 0.01 overshoots.

1. Diagnose in profile space: compare Eρ_g and Eφ_p (predicted log-rates over the finest
   pseudobulks, centred per row). If genes and peaks mix there, the coordinates are the
   gap.
2. Fix: content term under the unit metric, ρᵀ G M φ with G = ẼᵀSẼ (equivalently,
   compare profiles), and a scale-aware start (content term initialised near zero,
   lr tied to row scale). Keep the loss in the same space.
3. Guard: stop and warn if the epoch loss rises for N epochs.

Tests first: `tests/attention.rs` gains a case with gene and peak rows in different
linear coordinates of the same profiles; the loss must fall and π must find the planted
peak.

## After training (options, not scheduled)

- Peak embedding from the pooling weights: φ_p = Σ_g W_gp E_g / Σ_g W_gp over the
  ATAC-track gene rows (W = the learned gates). One sparse product, same coordinates as
  the genes; peaks near the same genes inherit similar rows. Alternative to, or base
  under, the per-peak fold-in.

## 4. Downstream checks on the full run

- `lupin annotate --feature-embedding {out}.gene_embedding.parquet --cell-embedding
  {out}.cell_embedding.parquet --markers` with Azimuth PBMC l1 and l2 marker lists
  (two columns: gene, cell type). The output must flow; labels sensible.
- Marker gene and peak UMAP on the full run.
- Links: promoter enrichment of top-1 attention vs fixed ABC vs the earlier Pearson run.

## Verification

- `cargo test -p chickpea`, `cargo +1.94 clippy -p chickpea --all-targets -D warnings`
  (CI pins 1.94); graph-embedding-util and senna tests after engine changes; bge sim check.
- PBMC full run with `-v`: per-epoch phase-1 lines (now with ETA) give the new s/epoch;
  stage table from log timestamps; peak RSS from `time -l`. Target: phase 1 under ~30 min,
  phase 2 ~1 min, attention loss falling.
- Version bumps per size of change; commits without attribution trailers.
