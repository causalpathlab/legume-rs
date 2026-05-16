# cage v3 — shared cell/gene embedding space

## Context

v1/v2 (current main) trained on Visium GBM, 20 epochs CUDA:
- Loss plateaued at 0.187 by epoch 5 (1.21 → 0.23 → 0.20 → 0.187 flat for 15 epochs).
- Cell embedding remained isotropic: top-5 covariance eigvals `[0.74, 0.72, 0.72, 0.71, 0.71]`, spread ratio **1.05** vs the ≥3-decade target.
- Per-gene per-level gates `α[G, L]` collapsed to **identical distributions across L0/L1/L2** (every gene has gate_L0 == gate_L1 == gate_L2 to 3 decimals; the "most level-selective gene" has a max-min range of 0.010 across levels — noise).

### Diagnosis
The current α gate is **multiplicative only**: `loss = Σ_g (α_g · per_level_g)`. Its gradient `∂loss/∂α_{g,l} = per_level_{g,l}` is the same scalar (~0.1) for every (g, l) pair, so all gates drift identically. The gate has no channel through which to observe "what would happen if I weighted differently" — it's a multiplier with no observation. Gene identity therefore has *no effect on the score* at all: the cell embedding is updated by gene-mixed positive sampling and gene-agnostic dot-product scoring. That is why cells don't differentiate along any gene-meaningful direction.

### Design
Promote gene identity to a first-class parameter inside the score function. Cell and gene embeddings live in the **same D-dimensional space**. Score for one positive cell-cell pair `(u, v)` under gene `g`:

```
score(u, v, g) = ⟨e_cell[u], e_gene[g]⟩ · ⟨e_cell[v], e_gene[g]⟩ + b_cell[u] + b_cell[v]
```

**Mental model**: `e_gene[g]` is gene `g`'s "direction" in cell-embedding space. The score is the product of u's and v's projections along that direction — a positive pair should both be aligned with `e_gene[g]`. Per chain level, sibling negatives `w` are scored similarly. The chain NCE machinery is unchanged; only the per-edge score function changes.

Key consequence: `e_gene[g]` receives gradient from every positive and negative pair gene g draws (~256 positives × 9 dependents per visit × many visits), not via a single scalar multiplier. The contrastive surface now has a real per-gene observation channel.

Cell embedding no longer needs to encode all gene information in a single vector; it just needs to be discriminable along the various gene directions. Gene embeddings become first-class outputs.

## Diagram

```mermaid
flowchart TB
    subgraph V2[v2 (current) — gene only via positive sampling + α gate]
        V2_CELL[e_cell N×D]
        V2_LOSS[score u,v = e_cell[u] · e_cell[v]<br/>α[g, l] only multiplies per-level loss]
        V2_CELL --> V2_LOSS
    end

    subgraph V3[v3 — gene identity in score]
        V3_CELL[e_cell N×D]
        V3_GENE[e_gene G×D<br/>SHARED D-dim space]
        V3_SCORE["score(u,v,g) =<br/>(e_cell[u]·e_gene[g]) · (e_cell[v]·e_gene[g]) + b_cell[u] + b_cell[v]"]
        V3_CELL --> V3_SCORE
        V3_GENE --> V3_SCORE
    end

    V2 -. plateau + isotropic cells .-> V3
```

## Parameters

| Tensor | Shape | Init | Notes |
|---|---|---|---|
| `e_cell` | `[N, D]` | randn(0, 0.1) | unchanged |
| `b_cell` | `[N]` | zeros | unchanged |
| `e_gene` | `[G, D]` | randn(0, 0.1) | NEW — same D as e_cell, shared space |
| `b_gene` | `[G]` | zeros | NEW |
| `α` | `[G, L]` | ln(e-1) ≈ 0.54 → softplus = 1.0 | KEPT for v3 (cheap). If still uniform after v3 smoke, drop in v4. |

`JointEmbedModel.e_feat`/`b_feat` slots are reused as `e_gene`/`b_gene` — they already live in the right shared D-dim space; we just use them with cage's semantics (the "feature" axis IS the gene axis).

## Gene utilities — adopt from senna `bge` / `fne`

Currently cage uses `FeatureNameKind::Exact` (pinto's `SRTReadArgs::to_read_args` hardcodes it). That means gene names like `ENSG00000187634_SAMD11` pass through verbatim, and there is no clean path to external gene resources. senna `bge` / `fne` already solve this — v3 cage should reuse the same machinery so that gene info enters at every layer it can.

Three senna-built utilities to wire in:

### 1. `FeatureNameKind::Gene { delim: '_' }` — gene-symbol canonicalization
Defined in `auxiliary-data/src/feature_names.rs`. Registers each `_`-split component as an alias of the full row name, so `ENSG00000105329_TGFB1` ↔ `TGFB1` both resolve to the same row. This is what makes external gene lists, marker sets, and PPI files line up against data row names without per-tool brittle string-matching.

cage v3 wires it via a new `--gene-name-mode {exact, gene, auto}` flag (default `gene`, since SRT gene names are virtually always Ensembl-suffixed symbols). Implementation: pinto's `SRTReadArgs.feature_kind` field already supports it — we just stop hardcoding `Exact` in `to_read_args` when running cage.

### 2. `load_feature_network` + `FeatureNetworkSmoother` — gene-pair regularization on `e_gene`
`graph-embedding-util/src/fit.rs:326 load_feature_network(FeatureNetworkArgs)` reads an external gene-pair edge list with prefix/delim fuzzy matching against the data's row names and returns a `FeatureNetworkConfig` containing a `FeatureNetworkSmoother`. The smoother applies SGC-style graph smoothing to a feature embedding `e_feat`: `e_feat_smoothed = (αI + (1-α)/k · Σ_k Â^k) e_feat`, applied every `refresh_epochs` (or every step). In senna `bge` it lives inside the loss via `select_feat_emb(smoother, e_feat, idx)`.

cage v3 wires it via:
```
--gene-network <path>             optional gene-pair edge list (TSV/CSV)
--gene-network-prefix-match       fuzzy prefix match on gene names
--gene-network-delim <char>       column delimiter in the edge file
--gene-network-k <usize>          SGC hops, default 2
--gene-network-alpha <f32>        SGC retention, default 0.5
--gene-network-refresh <usize>    refresh every N epochs, default 1
```

Effect on training: `e_gene` is now regularized to be locally smooth on the gene-gene graph. Functionally-related genes (e.g. `GFAP`/`S100B`/`AQP4`) get pushed toward similar embeddings before any spatial signal flows in, which is exactly the inductive bias the v2 diagnostic showed we lack.

In the gated loss, replace direct `e_gene` reads with `select_feat_emb(smoother, e_gene, gene_idx)` — drop-in.

### 3. `--freeze-feature-embedding <prefix>` — warm-start from prior bge (or cage)
senna `bge` writes its trained `e_feat` to parquet; `load_frozen_feature_host_for_bge` loads it and registers an immutable Var. cage v3 supports the same flag: when set, `e_gene` is loaded from the prior run, frozen (no gradient), and only `e_cell` / `b_cell` / `α` train. This is the "given biologically meaningful gene embeddings, learn the spatial cell geometry consistent with them" mode — much faster convergence than learning `e_gene` from scratch on one Visium slide.

Two natural sources for the frozen prefix:
- **senna bge output** — a `*.gene_embedding.parquet` trained on a large scRNA-seq atlas. Biologically grounded, can be applied to many SRT slides downstream.
- **A prior cage run** — since v3 already emits `{out}.gene_embedding.parquet` with the same shape and row naming convention as senna's, cage outputs can be fed back in for iterative refinement (multi-pass training, freezing gene side while exploring cell side, etc.).

Cage's existing `--gene-name-mode` controls how the frozen prefix is matched to the current data's row names (same fuzzy-match plumbing as `--gene-network`).

### Default behavior
All three are *optional*. Default cage v3 (no flags): gene-name canonicalization on (so output gene names are symbols, not Ensembl-prefixed), no feature network, no warm-start. Identical to "Tier 2 plain" except gene names in output parquets are cleaner.

## File changes

### `graph-embedding-util/src/model.rs` — add gene-modulated scoring

```rust
impl JointEmbedModel {
    /// Gene-modulated diagonal score for cell-cell positive pairs.
    /// Inputs are all already gathered ([G*B, D] / [G*B]).
    /// Returns [G*B] scalar scores.
    ///
    ///   score = (e_gene · e_cell_left) * (e_gene · e_cell_right)
    ///         + b_cell_left + b_cell_right
    pub fn score_cellcell_gated(
        e_gene: &Tensor,         // [G*B, D]
        e_cell_left: &Tensor,    // [G*B, D]
        e_cell_right: &Tensor,   // [G*B, D]
        b_cell_left: &Tensor,    // [G*B]
        b_cell_right: &Tensor,   // [G*B]
    ) -> Result<Tensor>;         // [G*B]

    /// Gene-modulated score against per-level negatives. `e_gene`
    /// broadcasts on the K axis.
    pub fn score_cellcell_gated_neg(
        e_gene: &Tensor,         // [G*B, D]
        e_cell_anchor: &Tensor,  // [G*B, D]
        e_cell_neg: &Tensor,     // [G*B, K, D]
        b_cell_anchor: &Tensor,  // [G*B]
        b_cell_neg: &Tensor,     // [G*B, K]
    ) -> Result<Tensor>;         // [G*B, K]
}
```

### `graph-embedding-util/src/loss/chain.rs` — gene-aware loss variants

Two new public functions, mirroring the existing two:

```rust
pub fn cell_cell_nce_loss_per_level_gated(
    model: &JointEmbedModel,
    batch: CellChainBatch,
    gene_id: u32,
    dev: &Device,
) -> Result<Tensor>;  // [L]

pub fn cell_cell_nce_loss_per_level_batched_gated(
    model: &JointEmbedModel,
    batches: Vec<CellChainBatch>,
    gene_ids: &[u32],   // length G — one per batch in `batches`
    dev: &Device,
) -> Result<Tensor>;   // [G, L]
```

Implementation: stack G batches → flat [G*B] cell ids, then build `[G*B, D]` `e_gene` by `index_select(e_gene, repeat_each(gene_ids, B))`. Reuse stacked left/right/neg layout from the existing batched function. Replace `score_diag` → `score_cellcell_gated` and `score_negatives` → `score_cellcell_gated_neg`. The reshape-to-[G, B]-then-mean(1)-stack pattern stays identical.

Existing `cell_cell_nce_loss_per_level_batched` (gene-agnostic) stays for back-compat. v3 cage calls the `_gated` variant exclusively.

### `pinto/src/cell_activity_embedding/fit.rs` — wire gene identity

- `ModelArgs { n_features: n_genes, n_cells, embedding_dim }` (was `n_features: 1`).
- `ModelInit { b_feat: &vec![0.0; n_genes], ... }` (was `&[0.0]`).
- Build a `gene_ids_chunk: Vec<u32>` alongside `cb_batches` from the rayon sampling output (we already extract `gene_ids` from `mini.into_iter().unzip()`; just cast to `u32`).
- Call `cell_cell_nce_loss_per_level_batched_gated(&model, cb_batches, &gene_ids_chunk_u32, smoother.as_ref(), &dev)` — smoother is `Option<&FeatureNetworkSmoother>`, threaded through analogously to senna gbe's `select_feat_emb`.
- After training, dump `model.e_feat` → `{out}.gene_embedding.parquet` (G × D, row names = canonicalized gene symbols, col names `e0..e{D-1}`).
- Dump `model.b_feat` → `{out}.gene_bias.parquet`.

If `--freeze-feature-embedding` is set: load the prior bge's `e_feat` via `load_frozen_feature_host_for_bge`, register as immutable, skip the AdamW update for `e_gene`. Optimizer only touches cells + α.

If `--gene-network` is set: build the `FeatureNetworkSmoother` once at start; pass to the loss; call `smoother.refresh(&e_gene)` at the end of each epoch (or every `refresh` epochs).

### `pinto/src/util/input.rs` — let cage opt out of `Exact`

`SRTInputArgs` keeps its current `Exact` hardcoding. Add a new field on cage's args: `gene_name_mode: GeneNameMode` (default `Gene`), and call `to_read_args_with_kind(kind)` from cage's `preprocess_srt` entry. Other subcommands (`lc`, `svd`) keep their current `Exact` behavior — this is an additive change.

### `pinto/src/util/metadata.rs` — extend OutputFiles

Add two more `Option<String>` fields:

```rust
#[serde(skip_serializing_if = "Option::is_none")] pub gene_embedding: Option<String>,
#[serde(skip_serializing_if = "Option::is_none")] pub gene_bias:      Option<String>,
```

Update `create_cage_metadata` to populate them. Update existing tests + add one for the v3 fields.

### `pinto/src/cell_activity_embedding/args.rs` — gene utility flags

```
--gene-name-mode {exact,gene,auto}           default `gene`
--gene-network <path>                        optional, default off
--gene-network-prefix-match                  default false
--gene-network-delim <char>                  default `\t`
--gene-network-k <usize>                     default 2
--gene-network-alpha <f32>                   default 0.5
--gene-network-refresh <usize>               default 1
--freeze-feature-embedding <prefix>          optional, default off
```

Embedding dim itself is **shared** between cells and genes. No `--gene-embedding-dim` — `--embedding-dim` controls both. When `--freeze-feature-embedding` is provided, `--embedding-dim` is checked against the frozen file's D and an error is raised on mismatch (mirrors senna bge).

## Outputs

| File | Shape | Contents |
|---|---|---|
| `{out}.cell_embedding.parquet` | N × D | (unchanged) |
| `{out}.cell_bias.parquet` | N × 1 | (unchanged) |
| `{out}.gene_embedding.parquet` | G × D | NEW — e_gene in the SAME D-dim space as e_cell |
| `{out}.gene_bias.parquet` | G × 1 | NEW |
| `{out}.gene_gates.parquet` | G × L | KEPT (deprecation candidate) |
| `{out}.coord_pairs.parquet` | E × 6 | unchanged |
| `{out}.scores.parquet` | T × 6 | unchanged |
| `{out}.metadata.json` | manifest | new gene_embedding/gene_bias keys |

## Verification

### Phase A — unit tests in graph-embedding-util
Equivalence: when `e_gene[g]` is initialized to a constant **unit vector** `1/√D · 1`, the gated score equals the original `e_cell[u] · e_cell[v] / D + b_cell[u] + b_cell[v]` (rescaled). Add a test asserting this within 1e-5 on a small fixture.

### Phase B — Visium GBM smoke (CPU, 1 epoch, small)
```sh
pinto cage gbm_visium.h5 -c tissue_positions.csv \
  --coord-column-names pxl_row_in_fullres,pxl_col_in_fullres \
  -o /tmp/cage_v3_smoke/run --preload-data \
  --epochs 1 --gene-batch-size 16 --per-gene-batch 64 \
  --embedding-dim 8 --n-pseudobulk 64 --num-levels 2 --chain-levels 0,1 \
  --gate-l2 0
```
Pass criteria: all output files exist, no NaN, loss < init.

### Phase C — long CUDA run (20 epochs, embedding-dim=16, n-pseudobulk=256)
Re-run the v2 long command with the new binary. Expected diagnostics:
1. **Cell embedding eigval spread** > 3.0 (vs 1.05 in v2). If achieved → cell embedding actually learned structure.
2. **Gene-embedding nearest-neighbors are biologically sensible**: for each of the 24 GBM markers we already validated are in the gates table:
   - `GFAP` should be cosine-close to `S100B`, `AQP4` (all astrocyte markers)
   - `MBP`, `MOG`, `PLP1` should cluster (oligodendrocyte)
   - `CD68`, `AIF1`, `CX3CR1`, `P2RY12` should cluster (microglia)
   - `VWF`, `PECAM1`, `CD34` should cluster (vascular)
3. **Per-marker cell tightness < 1** in v3 (debug the v2 diagnostic separately; see "Open" below).
4. **α[G, L]** behavior: either it differentiates by gene/level (= still useful), or stays uniform (= confirm v4 drops it).

### Phase D — workspace tests
All existing tests still pass. The renamed/added loss variants must not break senna gbe's path (which uses bipartite `nce_loss*`).

## Risks

- **Bilinear non-convexity**: `score = ⟨e_cell, e_gene⟩²`-style is non-convex in both. Mitigate with small init (randn 0.1), low LR, and consider a unit-norm projection on e_gene after each step if instability appears.
- **Gradient magnitude**: per-gene-gradient through the bilinear is second-order in e_cell. May need `lr_gene < lr_cell` if e_cell oscillates. Start with shared lr=5e-3, drop to 1e-3 if loss is noisy.
- **Sparse-gene noise**: genes with <50 active cell-pairs get high-variance e_gene gradients. Consider a per-gene Fisher-info weighting on the loss (analogous to `compute_nb_fisher_weights`) or a min-positives skip threshold.
- **Memory**: G_chunk=32 × B=256 × D=16 = 2 MB per cell-side tensor, plus K=8 negatives → 16 MB for the inner reshape. Trivial.

## Open

- **α future**: if v3 confirms α stays uniform across (g, l) even with gene identity in the score, drop `α[G, L]` and uniform-weight the chain levels.
- **v2 marker-diagnostic** (resolved): the `tightness=0.000` bug was `csc.getrow(g).tocsc()` — 1-row CSC has `.indices = [0, 0, ...]`. Fix: use `.getrow(g)` directly (returns CSR with cell-index `.indices`). After fix, v2's marker tightness is uniformly **1.00–1.12** across all 24 GBM markers — no biological clustering whatsoever, confirming the v3 diagnosis.

## Phase ordering

| Phase | What | Effort |
|---|---|---|
| A | `model.rs` scoring helpers + unit test | ½ day |
| B | `loss/chain.rs` gated variants + equivalence test (with optional smoother arg) | ½ day |
| C | `fit.rs` refactor + new parquet outputs + metadata extension; default gene-name canonicalization | ½ day |
| D | senna-style optional plumbing: `--gene-network`, `--freeze-feature-embedding`, `--gene-name-mode` | ½ day |
| E | Smoke (CPU + CUDA) + v3 vs v2 diagnostic comparison; verify marker biology (GFAP↔S100B etc.) | ¼ day |

Phases A–C can ship as one PR (core v3 architecture). Phase D is a follow-up PR (gene-utility plumbing). E is the validation gate.

## Critical files

### Core architecture (Phases A–C)
- `graph-embedding-util/src/model.rs` (add `score_cellcell_gated*`)
- `graph-embedding-util/src/loss/chain.rs` (add `*_gated` variants + tests; thread optional `Option<&FeatureNetworkSmoother>`)
- `pinto/src/cell_activity_embedding/fit.rs` (wire `n_features = n_genes`, gene-aware loss call, e_gene parquet output)
- `pinto/src/util/metadata.rs` (extend `OutputFiles` + `create_cage_metadata`)
- `pinto/src/cell_activity_embedding/gene_gating.rs` (consume / kept; α may be deprecated)

### Gene-utility plumbing (Phase D) — reuse senna's
- `auxiliary-data/src/feature_names.rs` (consume `FeatureNameKind::Gene` — no changes needed there)
- `graph-embedding-util/src/fit.rs:326 load_feature_network` (consume — already public)
- `graph-embedding-util/src/feature_network.rs::FeatureNetworkSmoother` (consume — already public)
- `senna/src/fit_bge.rs::load_frozen_feature_host_for_bge` — **promote to a shared crate** (`auxiliary-data` or `graph-embedding-util`) so cage can call it directly without depending on senna. v3 PR scope: factor it out, keep senna's call site intact via re-export.
- `pinto/src/cell_activity_embedding/args.rs` (new flags as above)
- `pinto/src/util/input.rs` (let cage override the default `FeatureNameKind::Exact` via the new `--gene-name-mode` arg)
