# Impute for all senna and pinto models — plan

Written 2026-08-30, from reading the code as it stands on `origin/main` (e178ad79).

**Status (same day): phases 0–3 are implemented on this branch.**
`data-beans-alg::retrieval_impute` holds the shared core (data-beans-alg
0.4.0); `senna impute` dispatches topic / masked / vae / bge / svd (senna
0.11.0), with the reference defaulting to the model's own manifest; `pinto
impute` is a new subcommand (pinto 0.9.0) covering cage (via predict) and
lc / dsvd (via a symmetric per-cell EM over the gene_community profiles).
E2E on `data-beans-sim topic` data (400 genes, 1500 cells, 5 factors, 70/30
split): imputed-vs-truth per-cell Spearman 0.84–0.85 for every arm — senna
topic / masked / svd / bge and pinto cage / lc alike — against a
train-marginal null of 0.34. Phase 4 (the shared `--eval-against` scorer)
is still open, as is chunked/sparse output for the memory ceiling.

## Why impute is not redundant with predict

The two commands answer different questions and write different artifacts:

- **`predict` evaluates.** Given a trained model and held-out data it infers the
  per-cell latent, scores the model's predictive likelihood and agreement against
  the *observed* counts, and optionally writes a residual backend. It computes the
  model reconstruction μ internally (for scoring and for the residual division)
  but never writes an expression matrix.
- **`impute` materializes.** It writes an `[N_query × G_ref]` expression matrix on
  the *reference's* gene axis — including genes the query never measured (panel →
  whole transcriptome). The mechanism is retrieval: kNN in the shared latent, then
  a softmax-weighted average of reference cells' counts. That carries full-rank
  gene–gene covariance that a rank-K decoder readout `β·θ` cannot (stated in
  `senna/src/impute.rs`'s own header note).

They compose rather than duplicate: `impute` step 1 literally calls
`predict_model` to get θ_new. The one overlapping design option — a *parametric*
imputation (`μ = δ·Σ_k θ_k·exp(β_dk)`) written out by `predict` — would be an
additive output flag and exists only for the topic family; retrieval imputation is
the model-agnostic mechanism, which is exactly what makes "impute for all models"
feasible.

A benchmark corollary: `predict`'s bare `llik` is decoder-dependent and must not
be compared across families (pinto's `write_predictive` doc says this in as many
words). Retrieval imputation + one shared multinomial eval gives a family-agnostic
axis — the tool-native way to put svd and vae back into the comparison.

## Correction to a claim in flight

`senna impute` does **not** require a masked-topic model. The `--model` help text
says "masked-topic", but the code's only gate is the bge rejection
(`impute.rs:123-130`); topic / masked-topic / masked-sbp / masked-vae / vae all
pass through `predict_model`. The atlas→Xenium direction is therefore not
hard-blocked on the masked family by the tool — masked remains the *right* model
for divergent panels (the indexed encoder takes missing features natively, the
dense encoder needs remap+padding), but dense `topic` imputes today.

## Current coverage

| model | query-side projection | impute today |
|---|---|---|
| senna topic / joint-topic | `predict` (dense, remap+pad) | works (help text says otherwise) |
| senna masked-topic / masked-sbp | `predict` (indexed masked) | works — canonical path |
| senna masked-vae / vae | `predict` (latent only) | passes; latent is Gaussian z, both sides get `softmax` (noted in code) |
| senna bge | `predict_bge` (Poisson-MAP onto ρ) | **rejected** (`ensure` in impute.rs) |
| senna svd / joint-svd | none — `predict` doesn't dispatch, no `model.json` | impossible |
| pinto cage | `pinto predict` | no impute subcommand |
| pinto lc / dsvd | none | no impute subcommand |

Uniformity that makes the plan cheap: every trainer writes `{out}.latent.parquet`
(and pinto writes `{out}.propensity.parquet` from all three models), and
`RunKind::cell_space()` (`senna/src/run_manifest.rs`) already answers the one
geometry question impute has — simplex vs embedding vs signed.

## Phase 0 — shared retrieval core → data-beans-alg

Move steps 3–5 of `senna/src/impute.rs` (ColumnDict kNN build,
`dist_to_softmax_weights`, the consumer-inverted streaming weighted average over
CSC chunks) into a new `data-beans-alg::retrieval_impute` module:

```rust
pub struct RetrievalImputeConfig { pub knn: usize, pub temperature: f32, pub chunk: usize }
pub fn retrieval_impute(
    query_latent: &Mat,   // already mapped to the matching space
    ref_latent: &Mat,     // same space
    ref_data: &SparseIoVec,
    cfg: &RetrievalImputeConfig,
) -> anyhow::Result<Mat> // N_query × G_ref
```

Both senna and pinto already depend on data-beans-alg; the module needs only
SparseIoVec (data-beans) and ColumnDict (matrix-util), both already deps. Tests in
`retrieval_impute/tests.rs`. This lands in data-beans-**alg**, so it should not
collide with the concurrent data-beans work.

## Phase 1 — senna: all model families

Replace the bge `ensure` in `impute_model` with a dispatch on
`resolve_run_kind(&args.model)` → matching space:

- **LogSimplex** (topic, itopic, joint-topic — masked-sbp resolves to itopic):
  `predict` → `softmax_rows_inplace` — today's path, unchanged.
- **Signed** (masked-vae, vae): keep the current `softmax(z)` mapping for
  continuity (the code comment already defends it); revisit only if match quality
  is poor on the sim tests.
- **Embedding** (bge): call `predict_bge`, L2-normalize rows on both sides
  (cosine ≈ L2 on normalized vectors, so ColumnDict is unchanged). Remove the
  rejection; keep a loud requirement that the reference latent came from the same
  bge run family. bge has no δ — warn on `--batch-files` as predict does.
- **svd / joint-svd**: add a small deterministic projection — read
  `{model}.dictionary.parquet` (gene × K loadings), align query genes by name with
  the same `QueryNameOpts` canonicalization, apply the training-side transform
  (confirm the exact normalization in `senna/src/svd/fit.rs` before implementing),
  least-squares project onto the loadings, then Signed-space matching. This is the
  piece that re-admits svd to the benchmark tool-natively.
- Everything else (fne, gem, gem-encoder, rest): refuse with an actionable
  message. Out of scope until someone needs them.

Reference-side hygiene: add `--reference <run-prefix>` as the preferred
alternative to bare `--reference-latent` — it reads the run manifest, resolves the
latent path itself, and refuses cross-space matching (a bge H-latent against a
topic θ-latent currently only fails if the dimensions happen to differ). Keep
`--reference-latent` for raw parquets with the existing dimension check.

Also: fix the `--model` help text ("masked-topic" → topic-family/any supported
run), update `senna docs`.

Tests (per the standing preference: verify recovery on simulated data):
`data-beans-sim topic` → `data-beans split` → train `topic` on the train half →
impute the test half with the train half as reference → assert per-cell agreement
of imputed vs truth beats the marginal-composition null. Same harness for one
masked arm, the bge arm, and the svd arm; plus refusal tests for the cross-space
mismatch.

## Phase 2 — pinto impute (cage)

New `pinto impute` subcommand:

- **Query side**: reuse the `predict_cage` pipeline up through propensity
  (preprocess → frozen dictionary align → pair projection → centroid assignment →
  `write_partition_outputs` propensity). Match on the per-cell **propensity** rows
  (simplex over K trained communities), not the `cell_embedding` readout — the
  readout collapses through the centroid matrix, and propensity is the
  partition-level cell representation all three pinto models publish, which makes
  Phase 3 uniform. Same softmax/L2 treatment as senna θ.
- **Reference side**: `{model}.propensity.parquet` + the reference sample's data
  files (with the same n_cells consistency check senna impute does).
- Output `{out}.imputed.parquet`, N_query × G_ref, reference gene names — via the
  Phase 0 core.
- Version bumps: pinto +0.1.0 (new subcommand), data-beans-alg minor (new
  module), senna minor (impute generalization).
- E2E smoke on the Xenium GBM FFPE test data already recorded for pinto; unit
  tests with the synthetic SRT support in `pinto/src/test_support.rs`.

## Phase 3 — pinto lc / dsvd

Neither has a query-side projection. Add a per-cell propensity projection-lite
usable by all three models: estimate a new cell's community propensity from
`{model}.gene_community.parquet` (G × K profiles) by a Poisson/multinomial MAP
E-step on the cell's counts (same spirit as cage's `pair_projection`, but per
cell against fixed profiles; natural home `link_community/profiles.rs`). Then
`pinto impute` dispatches: cage → pair-projection propensity (Phase 2 path);
lc / dsvd → profile-projection propensity. The query side needs no spatial graph
for this, so lc/dsvd impute stays cheap.

## Phase 4 (optional) — scoring tie-in

- `--eval-against <observed-data>` on impute (or a tiny standalone scorer):
  score the imputed matrix against the same cells' observed counts with the
  shared multinomial columns (`eval_llik_per_count`, `spearman`,
  `pearson_log1p`, and the null floor) that senna predict and pinto predict
  already agree on. This is the single comparable axis across senna
  topic/masked/bge/svd/vae and pinto cage/lc/dsvd.
- Possibly `--imputed-out` on `senna predict` for the parametric decoder readout
  (topic family only) as the low-rank baseline arm against retrieval.

## Known limitations to carry forward

- The dense `[N_query × G_ref]` output materializes in memory (the code already
  warns above ~1 GB). Xenium-scale queries against a whole-transcriptome
  reference will want chunked row-group writing or a sparse backend output —
  noted as follow-up, not in the phases above.
- `--adj-method residual` topic models keep predict's existing θ̂-bias warning;
  impute inherits it.
