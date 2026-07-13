//! Marker-set cell-type annotation by projection onto a **frozen feature
//! embedding** — model-agnostic across `senna bge`, `faba gem`, `pinto
//! cage`, and (via its adapter) topic models.
//!
//! The bipartite/freeze-feature pattern these embeddings share: a cell type
//! is just another node defined by its marker genes, so embed it in the same
//! feature space the cells live in, then annotate by a cosine score there:
//!
//! ```text
//! e_T = L2-normalize( Σ_{f ∈ markers(T)} w_f · feature_emb[f] )   (signature direction)
//! score(c, T) = ⟨ ê_cell[c], e_T ⟩                                (cosine)
//! ```
//!
//! **Two layers (fine + coarse).** A low-dim embedding cannot separate more
//! distinguishable directions than it has room for, so a fine marker set with
//! many nested types (e.g. CD8 Naive/Effector/Memory) over-types: the
//! signatures collapse onto a common cone and every per-cell score is flat.
//! We fix this **from the cells, not the markers**: cluster the cells in the
//! embedding space ([`matrix_util::clustering::leiden_clustering`], cosine
//! kNN + Leiden, community count automatic), then merge fine types that peak
//! on the same community into one coarse group, named by the lexical
//! commonality of its members. The coarse layer is what the cells can
//! actually resolve; the fine layer is kept alongside for subtype detail.
//!
//! **Why the marker centroid (not a Poisson-MAP projection).** The cells were
//! placed by fitting *graded* counts across many features, which pins their
//! direction. A marker set is presence-style and roughly flat, so a per-node
//! Poisson-MAP would let the free intercept absorb the level and the ridge
//! shrink the direction to noise — it degenerates. The weighted centroid of
//! the marker feature embeddings is the natural, degeneracy-free operator,
//! and it's exactly comparable to `e_cell`: because the model scores cell `c`
//! on gene `f` by `⟨e_f, e_cell⟩`, a cell expressing gene `f` has `e_cell`
//! aligned with `e_f` — so the marker-β centroid *is* the direction that
//! type's cells occupy. Because `e_cell` is direction-only (depth lives in a
//! discarded per-cell bias), the score is depth- and batch-invariant.
//!
//! **Significance (z-score).** A raw cosine is a similarity, not a tested
//! enrichment. With `n_perm > 0` we calibrate each cell→type affinity against
//! random gene sets of the *same size and weight multiset*: draw `n_perm`
//! permuted "types" per real type (gene identities shuffled, weights kept),
//! form their centroid signatures, and standardize the observed cosine
//! against the **moments** of the null cosines: `z = (obs − μ) / σ`. The null
//! cosines are ~Gaussian (a fixed unit cell vs near-random unit signatures in
//! high dimension), so a normal upper-tail p (`p = ½ erfc(z/√2) = pnorm(-z)`)
//! is well-calibrated and — unlike an empirical rank — has no `1/(n_perm+1)`
//! floor. We report the continuous `z`; downstream takes the normal tail.

use anyhow::{Context, Result};
use data_beans::utilities::name_matching::{idf_weight, GeneIndex};
use log::info;
use matrix_util::common_io::{read_lines_of_words_delim, write_lines, ReadLinesOut};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::traits::IoOps;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

mod coarsen;
mod gene_strata;
mod layout;
mod marker_bootstrap;
mod markers;
mod ontology_obo;
mod output;
mod panel_null;
mod score;
mod support_null;
mod term_ora;
use coarsen::*;
use layout::{leiden_from_graph, phate_cells, umap_from_graph, write_layout_outputs, LayoutInputs};
use markers::*;
use output::*;
use score::*;

/// The shared per-cell label-file writer (`argmax.tsv` + `membership.tsv`),
/// used by BOTH annotation methods so their per-cell I/O is identical.
pub use output::write_label_tsvs;

/// The "this cell has no call" sentinel, shared by the firm path and the marker bootstrap.
/// (`usize::MAX`, so it is never a valid term index and any `t < c` test excludes it.)
pub(super) const UNASSIGNED: usize = usize::MAX;

/// Number of communities in a partition (`community[i]` = cell `i`'s group id). At least 1,
/// so an all-in-one-cluster partition is still a partition.
pub(super) fn n_communities(community: &[usize]) -> usize {
    community.iter().copied().max().map_or(0, |m| m + 1).max(1)
}

/// Firm projection annotation by term over-representation within cell clusters
/// (Euclidean nearest-centroid → QC → cluster → hypergeometric + permutation
/// calibration → optional TreeBH ontology). The statistically-firm successor to
/// [`annotate_embeddings`]; `faba gem-annotate` drives it.
pub use term_ora::{
    annotate_embeddings_ora, annotate_with_communities, CommunityCalls, Regroup, TermOraConfig,
    TERM_ORA_OUTPUT_SUFFIXES,
};

/// The per-cell **stability bootstrap**: resample each type's marker panel with replacement
/// *and* re-derive the clustering, then ship the consensus — so a call that only survives one
/// particular draw of the panel, or one particular partition, is reported as what it is
/// (unreproducible) rather than as a confident label. Swapped in for the bare nearest-centroid
/// assignment when [`TermOraConfig::bootstrap`] is set.
///
/// `support` is **not** a posterior: it is the fraction of replicates that agreed, i.e. the
/// sampling variability of the pipeline's own output. It sees variance, not bias.
pub use marker_bootstrap::{
    Abstain, BootstrapResult, CoarseConsensus, MarkerBootstrapConfig, TypeQc as BootstrapTypeQc,
};

/// The **marker-panel permutation null** — the *bias* guard the bootstrap cannot supply. Puts one
/// type at a time on trial: replace only its panel with the same number of random genes (same IDF
/// weights, drawn from the *live* marker pool), leave every rival real, and ask whether the real
/// panel captures more cells than random genes would. Because the null panel is the same size, a
/// small panel's winner's-curse advantage appears in the null too and **cancels**.
pub use panel_null::{run_panel_null, PanelNull};

/// The **support permutation null** — turns `label_support` into a p-value, so the cutoff is an
/// FDR rather than the arbitrary (and not scale-free) `--min-support`. Shuffles the gene → type
/// assignment within norm strata and re-runs the whole bootstrap, reusing the cached partitions.
pub use support_null::{run_support_null, SupportNull};

/// Bind the generic `enrichment` TreeBH ontology core to the concrete OBO
/// loader (load OBO + `label→CL`, inject access closures, run). Shared by the
/// term-ORA path and by `senna annotate-ontology` / `-by-enrichment`.
pub use ontology_obo::annotate_ontology_from_obo;

/// File-name suffixes (relative to the `out_prefix`) of every artifact
/// [`annotate_embeddings`] writes — the single source of truth for the
/// projection annotation output set, so callers (e.g. `senna
/// annotate-by-projection --clean`) can erase a prior run without
/// re-transcribing the names across a crate boundary. Keep in sync with the
/// writers in `output`/`layout`.
pub const ANNOT_OUTPUT_SUFFIXES: &[&str] = &[
    ".annot.parquet",
    ".membership.tsv",
    ".argmax.tsv",
    ".community_profile.parquet",
    ".type_map.parquet",
    ".marker_embedding.parquet",
    ".type_embedding.parquet",
    ".coarse_embedding.parquet",
    ".cell_coords.parquet",
    ".feature_coords.parquet",
];

/// Tunables for the annotation routines.
pub struct AnnotateProjConfig {
    /// Permutation draws per type for the null (0 disables z/p, falls back
    /// to raw cosine scores).
    pub n_perm: usize,
    /// Deterministic RNG seed for the permutation null and clustering.
    pub seed: u64,
    /// k for the shared cell kNN graph (fine-score smoothing + Leiden
    /// coarsening + UMAP layout).
    pub knn: usize,
    /// Leiden modularity resolution for cell clustering (higher → more,
    /// smaller communities → more coarse groups).
    pub resolution: f64,
    /// Enable cell-grounded coarsening (cluster cells, merge over-split
    /// types). When false, the coarse layer mirrors the fine layer.
    pub coarsen: bool,
    /// kNN-smooth the per-cell fine scores over the cell graph before taking
    /// the argmax (local consensus → less fragile labels). When false, the
    /// fine label is the raw per-cell argmax.
    pub smooth_fine: bool,
    /// Opt-in significance gate: `unassigned` a cell whose best type has BH
    /// q ≥ `fine_fdr`. Default `1.0` (OFF) — per-cell fine calls are inherently
    /// modest (nested marker sets, BH over many cells), so a conventional FDR
    /// floods `unassigned`; the `fine_q` column is exposed for filtering
    /// instead. Lower it (e.g. 0.1) to actually drop non-significant calls.
    /// Only applies when a permutation null is in use (`n_perm > 0`).
    pub fine_fdr: f32,
    /// Opt-in margin gate: also `unassigned` a cell whose top1−top2 z-margin is
    /// below this. Default `0.0` (off) — fine types are nested/correlated by
    /// design, so margins are tiny and a margin gate over-flags; raise it only
    /// to deliberately drop near-tie subtype calls.
    pub min_margin: f32,
    /// Compute 2D cell layouts (UMAP off the leiden kNN graph; PHATE) and,
    /// when feature embeddings are supplied, project features onto them.
    pub layout: bool,
    /// Also compute the PHATE layout (UMAP is always produced when `layout`).
    pub phate: bool,
    /// PHATE diffusion time, kNN, and alpha-decay exponent.
    pub phate_t: usize,
    pub phate_knn: usize,
    pub phate_alpha: f32,
    /// Run PHATE directly on every cell when `n_cells <= phate_max_direct`;
    /// above this, reuse the Leiden communities as landmarks + Nyström.
    pub phate_max_direct: usize,
    /// k for feature→cell kNN projection onto the layouts.
    pub feat_knn: usize,
    /// UMAP SGD epochs for the cell layout.
    pub umap_epochs: usize,
}

impl Default for AnnotateProjConfig {
    fn default() -> Self {
        Self {
            n_perm: 200,
            seed: 42,
            knn: 30,
            resolution: 1.0,
            coarsen: true,
            smooth_fine: true,
            fine_fdr: 1.0,
            min_margin: 0.0,
            layout: false,
            phate: true,
            phate_t: 20,
            phate_knn: 5,
            phate_alpha: 40.0,
            phate_max_direct: 3000,
            feat_knn: 15,
            umap_epochs: 500,
        }
    }
}

/// Alpha-decay exponent for feature→cell kNN weighting (internal; gentler
/// than PHATE's so a feature is a smooth centroid of its neighbor cells).
pub(super) const FEAT_PROJ_ALPHA: f32 = 2.0;

/// The two embeddings + their row names that [`annotate_embeddings`] annotates:
/// a `[G × H]` feature embedding (markers match against `gene_names`) and a
/// `[N × H]` cell embedding.
pub struct InputEmbeddings<'a> {
    pub feature_emb: &'a DMatrix<f32>,
    pub gene_names: &'a [Box<str>],
    pub cell_emb: &'a DMatrix<f32>,
    pub cell_names: &'a [Box<str>],
}

/// Slice-based inputs for [`annotate_by_projection`] — the row-major
/// embeddings, their dimensions, and the matched marker sets.
pub struct ProjInputs<'a> {
    /// Row-major `[n_features × h]` frozen feature embedding.
    pub feature_emb: &'a [f32],
    pub n_features: usize,
    /// Row-major `[n_cells × h]` cell embedding (defensively re-normalized).
    pub cell_emb: &'a [f32],
    pub n_cells: usize,
    /// Per-type `(feature_index, weight)` marker lists.
    pub type_markers: &'a [Vec<(u32, f32)>],
    pub type_names: &'a [Box<str>],
    pub h: usize,
}

/// Result of [`annotate_by_projection`]. All matrices are row-major.
///
/// The annotation is two-layer: a *fine* layer (one marker-defined type per
/// column) and a *coarse* layer whose groups are **cell communities** —
/// fine types that collapse onto the same community of cells are merged and
/// named by the lexical commonality of their members. Each per-cell
/// community id indexes directly into the coarse arrays.
pub struct AnnotateProjOutputs {
    pub n_cells: usize,
    /// Number of fine types.
    pub n_types: usize,
    /// `[C]` fine type names (marker-file celltype labels), in column order.
    pub type_names: Vec<Box<str>>,
    /// Number of coarse groups (= number of cell communities).
    pub n_coarse: usize,
    /// `[C × H]` L2-normalized fine type signature embeddings (plot anchors).
    pub type_emb_ch: Vec<f32>,
    /// `[K × H]` L2-normalized coarse signature embeddings.
    pub coarse_emb_kh: Vec<f32>,
    /// `[K]` lexical label per coarse group.
    pub coarse_names: Vec<Box<str>>,
    /// `[C]` fine type → coarse group index.
    pub coarse_of_fine: Vec<usize>,
    /// `[N]` cell → community (= coarse group) index.
    pub community: Vec<usize>,
    /// `[N × C]` per-cell fine score: z-score if `n_perm > 0`, else cosine.
    pub fine_z: Vec<f32>,
    /// `[N × K]` per-cell coarse score, same scale as `fine_z`.
    pub coarse_z: Vec<f32>,
    /// `[N × C]` fine p-values `pnorm(-z)`, or `None` when `n_perm == 0`.
    pub fine_p: Option<Vec<f32>>,
    /// `[N × K]` coarse p-values, or `None` when `n_perm == 0`.
    pub coarse_p: Option<Vec<f32>>,
    /// `[N]` argmax fine type per cell (of the smoothed scores when enabled).
    pub fine_label: Vec<usize>,
    /// `[N]` top1−top2 margin of the (smoothed) fine scores — how definitive
    /// each call is. Written to `annot.parquet` for filtering.
    pub fine_margin: Vec<f32>,
    /// `[K × C]` mean fine score of cells in each community (for profiling).
    pub enrich: Vec<f32>,
    /// Per community: the ranked lineage-defining fine types (the same set
    /// used for the label), most-significant first.
    pub community_members: Vec<Vec<usize>>,
    /// `[N × 2]` row-major UMAP cell coordinates (None when layout disabled).
    pub cell_umap: Option<Vec<f32>>,
    /// `[N × 2]` row-major PHATE cell coordinates (None when disabled/too few).
    pub cell_phate: Option<Vec<f32>>,
}

////////////////////////////
// High-level entry point //
////////////////////////////

/// End-to-end annotation from in-memory embeddings: parse + match the marker
/// TSV against `gene_names`, project every type, cluster + coarsen, and write
/// `{out_prefix}.{annot,community_profile,type_map,marker_embedding}.parquet` plus
/// the type locations as `{type,coarse}_embedding.parquet` (co-embedded onto the
/// cell manifold). The thin per-tool adapters (gem / cage / bge) only load the two
/// embedding matrices from their own manifest and call this.
///
/// * `feature_emb` `[G × H]`, `gene_names` len `G` (the full feature rows the
///   markers match against; also projected onto the layout as `kind=feature`).
/// * `cell_emb` `[N × H]`, `cell_names` len `N`.
/// * `out_prefix` — full prefix incl. tool infix, e.g. `…/run.gem_annot`.
///
/// When `cfg.layout`, also writes `{out_prefix}.cell_coords.parquet` (UMAP +
/// PHATE per cell) and `{out_prefix}.feature_coords.parquet` (full features and
/// type/coarse anchors placed on both layouts by Nyström-projecting their
/// co-embed positions through the cell layout).
pub fn annotate_embeddings(
    input: &InputEmbeddings<'_>,
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    cfg: &AnnotateProjConfig,
) -> Result<AnnotateProjOutputs> {
    let &InputEmbeddings {
        feature_emb,
        gene_names,
        cell_emb,
        cell_names,
    } = input;
    let g = feature_emb.nrows();
    let h = feature_emb.ncols();
    let n = cell_emb.nrows();
    anyhow::ensure!(
        cell_emb.ncols() == h,
        "embedding dim mismatch: features H={h}, cells H={}",
        cell_emb.ncols()
    );
    anyhow::ensure!(gene_names.len() == g, "gene_names len != feature rows");
    anyhow::ensure!(cell_names.len() == n, "cell_names len != cell rows");
    info!("annotate: β [{g} × {h}], cells [{n} × {h}]");

    let (type_names, type_markers) = parse_and_match_markers(markers_path, gene_names, use_idf)?;
    anyhow::ensure!(
        type_names.len() >= 2,
        "need ≥ 2 cell types with matched markers, found {}",
        type_names.len()
    );
    let matched: usize = type_markers.iter().map(Vec::len).sum();
    info!(
        "markers: {} types, {} matched (gene, type) entries",
        type_names.len(),
        matched
    );

    // Per-marker embeddings: the (co-embedded) feature rows actually used,
    // labelled by type + weight. Lets callers see where each marker gene lands
    // relative to the cells (on the shared manifold), not just the type
    // centroid — pair with `cell_embedding` / `cell_coords` to plot markers
    // among cells.
    write_marker_embeddings(
        out_prefix,
        feature_emb,
        gene_names,
        &type_names,
        &type_markers,
        h,
    )?;

    let beta_flat = row_major(feature_emb);
    let cell_flat = row_major(cell_emb);
    let res = annotate_by_projection(
        &ProjInputs {
            feature_emb: &beta_flat,
            n_features: g,
            cell_emb: &cell_flat,
            n_cells: n,
            type_markers: &type_markers,
            type_names: &type_names,
            h,
        },
        cfg,
    )?;

    write_annotation_outputs(out_prefix, cell_names, &type_names, &res, cfg)?;
    log_label_histogram(&res);

    // Co-embed the cell-type signatures onto the cell manifold (same SIMBA
    // softmax-over-cells transform as the genes), so each type lands where its
    // cells are — the {type,coarse}_embedding outputs. The returned co-embed
    // locations are reused by the layout so anchors land FIRMLY on the manifold.
    let (type_co, coarse_co) = write_type_coembeddings(out_prefix, cell_emb, &type_names, h, &res)?;

    // Layout coordinates + feature placement (part ii). Features + anchors are
    // placed by Nyström-projecting their co-embed positions through the cell
    // layout (no centering hack), so each lands on its matching cell cluster.
    if cfg.layout && res.cell_umap.is_some() {
        write_layout_outputs(&LayoutInputs {
            out_prefix,
            cell_names,
            cell_flat: &cell_flat,
            n,
            h,
            feature_emb,
            gene_names,
            type_co: &type_co,
            coarse_co: &coarse_co,
            type_names: &type_names,
            res: &res,
            cfg,
        })?;
    }
    Ok(res)
}

//////////////////////////////////////
// Core compute (pure, slice-based) //
//////////////////////////////////////

/// Annotate cells against marker-defined types by signature projection.
/// The cell embedding is defensively re-normalized here, so an un-normalized
/// input still works.
pub fn annotate_by_projection(
    inp: &ProjInputs<'_>,
    cfg: &AnnotateProjConfig,
) -> Result<AnnotateProjOutputs> {
    let &ProjInputs {
        feature_emb,
        n_features,
        cell_emb,
        n_cells,
        type_markers,
        type_names,
        h,
    } = inp;
    let n_types = type_markers.len();
    anyhow::ensure!(n_types >= 1, "annotate_by_projection needs ≥ 1 type");

    // 1. Fine signatures + unit-normalized cells (gem/bge already are).
    let type_emb_ch = type_signatures(feature_emb, n_features, type_markers, h);
    let mut cell_u = cell_emb.to_vec();
    l2_normalize_rows(&mut cell_u, n_cells, h);

    // Marker pool: the union of every type's marker genes. The permutation
    // null draws from THIS pool (a label shuffle), not the whole genome, so
    // each null type is "another type's markers of the same size".
    let marker_pool = marker_gene_pool(type_markers, n_features);
    let ctx = ScoreCtx {
        feature_emb,
        marker_pool: &marker_pool,
        cell_u: &cell_u,
        n_cells,
        h,
        cfg,
    };

    // Score a marker set (its signatures) against every cell — used identically
    // for the fine and the coarse layer.
    let score = |markers: &[Vec<(u32, f32)>], emb: &[f32]| type_scores(&ctx, markers, emb);

    // 2. Fine per-cell scores (permutation z, else cosine) + p-values.
    let (fine_z, fine_p) = score(type_markers, &type_emb_ch);

    // 3. Build the cosine cell kNN graph ONCE (cell_u is already unit-norm),
    // reused for fine-score smoothing, Leiden coarsening, and the UMAP layout.
    let do_coarsen = cfg.coarsen && n_cells >= 2 && n_types >= 2;
    let want_graph =
        (cfg.smooth_fine && n_cells >= 2) || do_coarsen || (cfg.layout && n_cells >= 3);
    let cell_graph = if want_graph {
        let cell_mat = DMatrix::<f32>::from_row_iterator(n_cells, h, cell_u.iter().copied());
        let knn = cfg.knn.clamp(1, n_cells - 1);
        Some(KnnGraph::from_rows(
            &cell_mat,
            KnnGraphArgs {
                knn,
                block_size: 1000,
                reciprocal: false,
            },
        )?)
    } else {
        None
    };

    // 2b. Fine label = argmax of the (optionally kNN-smoothed) scores. Smoothing
    // borrows neighbourhood evidence so a non-definitive per-cell argmax doesn't
    // flip on noise; the top1−top2 margin records how definitive the call is.
    let fine_smoothed = match (cfg.smooth_fine, cell_graph.as_ref()) {
        (true, Some(graph)) => smooth_scores_over_graph(&fine_z, n_cells, n_types, graph),
        _ => fine_z.clone(),
    };
    let fine_label = argmax_rows(&fine_smoothed, n_cells, n_types);
    let fine_margin = top2_margin(&fine_smoothed, n_cells, n_types);
    // The `unassigned` definitiveness gate is applied at output time (it needs
    // the BH q-values), keyed on significance (primary) + optional margin.

    // Communities: Leiden over the shared graph (modularity objective),
    // matching `matrix_util::clustering::leiden_clustering`'s tail.
    let (community, n_comm) = match (do_coarsen, cell_graph.as_ref()) {
        (true, Some(graph)) => {
            let labels = leiden_from_graph(graph, n_cells, cfg.resolution, cfg.seed);
            let k = n_communities(&labels);
            (labels, k)
        }
        // Coarse layer mirrors fine: each cell's "community" is its fine argmax.
        _ => (fine_label.clone(), n_types),
    };

    // 3b. 2D layouts off the same assets. UMAP runs directly on the kNN graph;
    // PHATE runs on e_cell (direct for small N, k-means landmarks + Nyström
    // otherwise). Both are [N×2] row-major.
    let (cell_umap, cell_phate) = if cfg.layout && n_cells >= 3 {
        let umap = cell_graph
            .as_ref()
            .map(|g| umap_from_graph(g, n_cells, cfg.umap_epochs, cfg.seed));
        let phate = if cfg.phate {
            phate_cells(&cell_u, n_cells, h, &community, n_comm, cfg)
        } else {
            None
        };
        (umap, phate)
    } else {
        (None, None)
    };

    // 4. Community × fine-type enrichment (mean fine score over each
    // community), column-centered so a marker set that is high *everywhere*
    // (common-mode aligned) doesn't name every community — what selects a
    // community's lineage is its enrichment RELATIVE to the other communities.
    let mut enrich = community_enrichment(&fine_z, n_cells, n_types, &community, n_comm);
    center_columns(&mut enrich, n_comm, n_types);

    // A community's identity is the fine types ENRICHED in it (its top few),
    // ranked by centered enrichment alone — the per-cell z-scores are already
    // size-matched, so no marker-count weight (which biased naming toward
    // large-marker types). Shared by the label, the coarse markers, and the
    // community profile so all three agree.
    let community_members = top_enriched_members(&enrich, n_comm, n_types, 6);

    // 5. Merge map + lexical names + coarse re-scoring.
    let (coarse_of_fine, coarse_names, coarse_emb_kh, coarse_z, coarse_p) = if do_coarsen {
        // `coarse_of_fine` assigns each fine type to its peak community (the
        // type_map merge record).
        let mut sizes = vec![0usize; n_comm];
        for &kk in &community {
            sizes[kk] += 1;
        }
        let coarse_of_fine = build_merge_map(&enrich, &sizes, n_comm, n_types);
        let coarse_names: Vec<Box<str>> = community_members
            .iter()
            .map(|m| lexical_label(m, type_names))
            .collect();
        let coarse_markers = coarse_markers_from_groups(&community_members, type_markers);
        let coarse_emb_kh = type_signatures(feature_emb, n_features, &coarse_markers, h);
        // Same marker pool for the coarse null — coarse markers ⊆ the pool.
        let (coarse_z, coarse_p) = score(&coarse_markers, &coarse_emb_kh);
        (
            coarse_of_fine,
            coarse_names,
            coarse_emb_kh,
            coarse_z,
            coarse_p,
        )
    } else {
        (
            (0..n_types).collect(),
            type_names.to_vec(),
            type_emb_ch.clone(),
            fine_z.clone(),
            fine_p.clone(),
        )
    };

    Ok(AnnotateProjOutputs {
        n_cells,
        n_types,
        type_names: inp.type_names.to_vec(),
        n_coarse: n_comm,
        type_emb_ch,
        coarse_emb_kh,
        coarse_names,
        coarse_of_fine,
        community,
        fine_z,
        coarse_z,
        fine_p,
        coarse_p,
        fine_label,
        fine_margin,
        enrich,
        community_members,
        cell_umap,
        cell_phate,
    })
}

#[cfg(test)]
mod tests;
