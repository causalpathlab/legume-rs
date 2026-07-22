//! Firm projection annotation by **term over-representation within cell
//! clusters**.
//!
//! The soft path ([`super::annotate_by_projection`]) reports a per-cell cosine
//! z against a single competitive label-shuffle null, with FDR off. This path
//! makes the call *firm*:
//!
//! 1. **Term centroid** `e_T = (Σ_g w_g·e_g) / Σ_g w_g` — the IDF-weighted,
//!    **un-normalized** mean of a type's marker feature embeddings (a prototype
//!    in the embedding space; L2-norm would discard the position the Euclidean
//!    metric needs).
//! 2. **Nearest-centroid assignment** `t(c) = argmin_T ‖e_cell[c] − e_T‖₂` —
//!    every cell hard-assigned to its closest term, with the distance kept.
//!    With [`TermOraConfig::bootstrap`], the call is instead the consensus of a
//!    **marker bootstrap** ([`super::marker_bootstrap`]): each type's panel is
//!    resampled with replacement, its centroid rebuilt and the cells re-assigned,
//!    so a call that only survives one particular draw of the panel is reported
//!    as unreproducible rather than as a confident label. Note that `argmin` has
//!    no error bar and *always* returns something — on a panel the embedding
//!    never trained, its answer can be decided by a ~1% distance margin.
//! 3. **QC prune** — per term, drop cells whose distance to their assigned
//!    centroid is a high-side robust outlier (`> median + k·MAD`): cells that
//!    argmaxed a term but don't actually sit near it (ambient/doublet). They
//!    become `unassigned` and are excluded from the counts.
//! 4. **Cluster cells** — the aggregation device. A single cell's
//!    nearest-centroid call is close to a coin flip; pooling cells is what makes
//!    it testable. Leiden over the cell kNN graph (the embedding's own geometry,
//!    independent of the term labels).
//!
//!    **The pooling must stay coarse, and that is a constraint, not a default.**
//!    The hypergeometric ranks terms by how *surprising* a count is, not how
//!    *likely* — a discovery statistic, not a classifier. The two rankings
//!    coincide only when the cluster is large: at 700 cells you need many of them
//!    to be surprising, so most-enriched ≈ most-abundant. Shrink the cluster and
//!    it inverts — a type with 4 cells in the entire dataset has an expected count
//!    near zero, so *two* of them outscore the 30 cells of the type that actually
//!    fills the cluster. Anything that makes the groups small (a high
//!    `--resolution`, or replacing the partition with per-cell neighbourhoods)
//!    walks into this. See `faba/docs/annotation-grouping.md`.
//! 5. **Over-representation** — per (cluster K, term T) the count
//!    `a = #{c∈K : t(c)=T}` is tested against the hypergeometric null with
//!    fixed margins `(N, m_T, n_K)`; the statistic `−ln P(X≥a)` is **calibrated
//!    by permuting the per-cell labels** (pooled across clusters per term,
//!    relabeling-invariant). BH-FDR → q, FDR-sparse softmax → Q. The cluster's
//!    call is its top over-represented term; cells inherit it.
//! 6. **Calibration assessment** — analytic-vs-permutation agreement + a
//!    permutation-machinery sanity check, written to `null_calibration.tsv`.
//! 7. **Ontology (optional)** — feed the cluster × term p (and Q) to the shared
//!    generic TreeBH core for multi-resolution CL calling.

use super::marker_bootstrap::{
    run_marker_bootstrap, BootstrapResult, CoarseConsensus, MarkerBootstrapConfig,
};
use super::markers::{parse_and_match_markers, MarkerSets};
use super::output::{write_label_tsvs, write_marker_embeddings};
use super::score::{argmax_rows, row_major};
use super::{n_communities, InputEmbeddings, UNASSIGNED};
use crate::null_call::live_row;
use anyhow::{Context, Result};
use enrichment::consensus::MIN_LIVE_MARKERS;
use log::{info, warn};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::traits::IoOps;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::io::Write;

#[cfg(test)]
mod tests;

/// File-name suffixes (relative to `out_prefix`) the firm term-ORA path writes.
/// Kept explicit (never a glob) so a caller can erase a prior run without
/// touching sibling artifacts (the embedding, the manifest).
pub const TERM_ORA_OUTPUT_SUFFIXES: &[&str] = &[
    ".annot.parquet",
    ".membership.tsv",
    ".argmax.tsv",
    ".marker_embedding.parquet",
    ".cluster_term_p.parquet",
    ".cluster_term_q.parquet",
    ".cluster_term_softq.parquet",
    ".null_calibration.tsv",
    ".ontology_assignment.tsv",
    ".ontology_node_mass.parquet",
    ".label_stability.parquet",
    ".marker_support.parquet",
    ".type_qc.tsv",
    ".panel_null.tsv",
];

/// Tunables for [`annotate_embeddings_ora`].
pub struct TermOraConfig {
    /// k for the cosine cell kNN graph fed to Leiden.
    pub knn: usize,
    /// Leiden modularity resolution (higher → more, finer clusters).
    pub resolution: f64,
    /// Deterministic RNG seed (clustering + permutation null).
    pub seed: u64,
    /// Permutation draws calibrating the over-representation statistic.
    pub n_perm: usize,
    /// Minimum markers carrying a **live** feature row before a cell type is allowed to compete.
    ///
    /// A type below this is not weakly located, it is *unlocated*: the mean of one or two points has
    /// no direction, and a centroid built from too few markers lands short — near the middle of the
    /// cell cloud, where it is close to every cell and becomes a magnet rather than a weak
    /// competitor. Such a type is dropped: it keeps its column in every output but can never win a
    /// cell. Floored at 2 (you cannot resample a single point).
    pub min_markers: usize,
    /// Prune outlier cell→term assignments (distance > median + `assign_mad`·MAD).
    pub assign_qc: bool,
    /// MAD multiplier for the assignment-distance outlier gate.
    pub assign_mad: f64,
    /// FDR α for the cluster call + Q sparsity (BH on the permutation p).
    pub fdr_alpha: f32,
    /// Softmax temperature when building the row-normalized Q over significant terms.
    pub q_temperature: f32,
    /// Cell Ontology OBO path — runs the TreeBH ontology layer when set with `label_cl`.
    pub obo: Option<String>,
    /// Curated `label<TAB>CL:id` map (paired with `obo`).
    pub label_cl: Option<String>,
    /// TreeBH per-level selective-FDR target.
    pub ontology_fdr_q: f64,
    /// Benjamini–Yekutieli within ontology families (any dependence).
    pub ontology_by: bool,
    /// Draws for the **marker-panel permutation null** ([`super::panel_null`]) — the *bias*
    /// guard the bootstrap cannot supply. `0` ⇒ off.
    pub panel_perm: usize,
    /// Shuffled panels for the **support null** ([`super::support_null`]) — turns `label_support`
    /// into a p-value, so a cutoff can be an FDR rather than the arbitrary `--min-support`.
    /// `0` ⇒ off. Needs the bootstrap.
    pub support_perm: usize,
    /// When set, the per-cell call is the consensus of a **marker bootstrap**
    /// ([`super::marker_bootstrap`]) rather than a bare nearest-centroid argmin: each type's
    /// panel is resampled with replacement, its centroid rebuilt, and the cells re-assigned,
    /// so every call carries the support it earned across resamples and an unreproducible one
    /// abstains. `None` ⇒ the point-estimate path, unchanged.
    pub bootstrap: Option<MarkerBootstrapConfig>,
    /// Minimum fraction of the marker panel that must be present on the embedding's feature
    /// axis, or the run fails. Guards the silent case where the HVG cut has left a type
    /// scoring on a handful of its genes and the call still looks confident (see
    /// [`super::markers::parse_and_match_markers`]). `0.0` ⇒ report and warn, never refuse.
    pub min_panel_coverage: f32,
}

impl Default for TermOraConfig {
    fn default() -> Self {
        Self {
            knn: 30,
            resolution: 1.0,
            seed: 42,
            n_perm: 500,
            min_markers: 3,
            assign_qc: true,
            assign_mad: 2.5,
            fdr_alpha: 0.1,
            q_temperature: 1.0,
            obo: None,
            label_cl: None,
            ontology_fdr_q: 0.1,
            ontology_by: false,
            panel_perm: 0,
            support_perm: 0,
            bootstrap: None,
            min_panel_coverage: 0.0,
        }
    }
}

/// One replicate's grouping: the per-cell community id, and how many communities there are.
pub(super) type Partition = (Vec<usize>, usize);

/// How a bootstrap replicate re-derives the cell grouping, given that replicate's seed.
///
/// **The grouping has to be resampled, or the bootstrap has no teeth.** Resampling only the
/// marker panel while holding the partition fixed measures almost nothing: a 2,000-cell
/// cluster's argmax does not flip because a few markers were redrawn, so every call comes back
/// with support ≈ 1 and the run abstains on nothing (measured: 0% unassigned, and the support's
/// ability to separate spurious calls collapses from AUC 0.93 to 0.69). The partition is where
/// the instability lives, so the partition is what must move.
///
/// It is a callback because each caller's grouping is arbitrary in its own way: `faba annotate`
/// re-runs **Leiden** on a fixed kNN graph (modularity has many near-equal optima — the same
/// cells have partitioned into anywhere from 132 to 990 communities), while `faba lineage`
/// re-runs its **seeded k-means** over the trajectory nodes. Same question, different coin.
pub type Regroup<'a> = dyn Fn(u64) -> Result<Vec<usize>> + Sync + 'a;

/// Per-community (cluster / MST-node) firm call returned by
/// [`annotate_with_communities`], so a caller (e.g. `faba lineage --markers`) can name
/// each trajectory node without re-reading the parquet.
pub struct CommunityCalls {
    /// Called cell type per community (or `"unassigned"`), length `n_comm`.
    pub labels: Vec<Box<str>>,
    /// Confidence of each community's call (the FDR-sparse softmax `Q`; `0` when
    /// unassigned), length `n_comm`.
    pub confidence: Vec<f32>,
}

/// Name of a term index, mapping the [`UNASSIGNED`] sentinel to `"unassigned"`.
fn label_of(t: usize, type_names: &[Box<str>]) -> Box<str> {
    if t == UNASSIGNED {
        Box::from(enrichment::UNASSIGNED_LABEL)
    } else {
        type_names[t].clone()
    }
}

/// End-to-end firm annotation from in-memory embeddings, clustering cells with **Leiden**
/// over their own cosine kNN graph, then delegating to [`annotate_with_communities`].
/// See the module docs for the pipeline. Writes the `{out_prefix}.*` artifacts.
///
/// Because the clustering is derived here rather than handed in, it is also **re-derived on
/// every bootstrap replicate**: the arbitrariness in *how we pooled* then lands in the per-cell
/// support alongside the arbitrariness in *which markers we drew*.
pub fn annotate_embeddings_ora(
    input: &InputEmbeddings<'_>,
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    cfg: &TermOraConfig,
) -> Result<()> {
    let n = input.cell_emb.nrows();
    let h = input.cell_emb.ncols();
    anyhow::ensure!(n >= 2, "term-ORA needs ≥ 2 cells, found {n}");

    // Remove the common mode from BOTH sides first. This path is cosine
    // throughout — `cell_knn_graph` L2-normalizes and builds the neighbour graph
    // the Leiden communities come from, and the gene side is ranked the same way
    // — so a shared offset that carries no identity still dominates every
    // comparison. See `score::remove_common_mode` for the measurement.
    //
    // It matters most HERE, not at the per-cell scoring: if the kNN graph is
    // built on near-parallel rows the communities are close to arbitrary, and
    // the per-cluster ORA is then testing marker enrichment against noise
    // clusters. Centring is a rigid translation, so the marker matching and the
    // gene set are untouched; only the angles change.
    let (cell_c, cell_cm) = super::score::remove_common_mode_dmat(input.cell_emb);
    let (feat_c, feat_cm) = super::score::remove_common_mode_dmat(input.feature_emb);
    if cell_cm > 0.5 || feat_cm > 0.5 {
        info!(
            "removed the common mode before scoring: {:.1}% of the cell embedding's \
             sum-of-squares and {:.1}% of the feature embedding's. A large share here means \
             the raw embedding put every row on nearly the same ray, which is the regime \
             cosine cannot resolve.",
            100.0 * cell_cm,
            100.0 * feat_cm
        );
    }
    let owned = InputEmbeddings {
        feature_emb: &feat_c,
        gene_names: input.gene_names,
        cell_emb: &cell_c,
        cell_names: input.cell_names,
    };
    let input = &owned;
    // Communities are Leiden over the cell kNN graph — the embedding's own geometry,
    // independent of the term labels (module docs step 4).
    let cell_flat = row_major(input.cell_emb);
    let graph = cell_knn_graph(&cell_flat, n, h, cfg)?;
    let community = cluster_cells(&graph, n, cfg, cfg.seed);
    let n_comm = n_communities(&community);
    info!(
        "clustered cells into {n_comm} communities (knn={}, res={})",
        cfg.knn, cfg.resolution
    );
    // Each replicate re-partitions the *same* graph under a fresh Leiden seed. The graph is built
    // once and reused: the bootstrap resamples the marker panel, which drives the *scoring*, not
    // the clustering embedding, so the embedding clustered here is identical on every draw. The
    // graph is deterministic (seeded instant-distance), so rebuilding it per replicate would only
    // reproduce the identical graph anyway — 135 s for nothing, where reseeding Leiden costs 4 s
    // and probes the real within-run uncertainty (its choice among near-equal modularity optima):
    // same discrimination, AUC 0.931 vs 0.943, support correlation 0.96.
    let regroup = |seed: u64| -> Result<Vec<usize>> { Ok(cluster_cells(&graph, n, cfg, seed)) };
    annotate_inner(
        input,
        markers_path,
        out_prefix,
        use_idf,
        &community,
        n_comm,
        Some(&regroup),
        cfg,
    )?;
    Ok(())
}

/// Firm annotation given an **externally-supplied** cell clustering (`community[i]` =
/// cell `i`'s group id, `n_comm` groups) rather than Leiden. Runs the shared pipeline —
/// term centroids, nearest-centroid `fine_label`, per-term QC, then cluster × term
/// over-representation + permutation calibration over the *given* grouping — and writes
/// every `{out_prefix}.*` artifact. [`annotate_embeddings_ora`] wraps this with Leiden;
/// `faba lineage --markers` passes the MST-node clustering, so each trajectory node gets
/// the same permutation-calibrated call.
///
/// `regroup` (see [`Regroup`]) says how a bootstrap replicate re-derives the caller's grouping —
/// `faba lineage` reseeds its k-means. Pass `None` to hold the grouping fixed, but note that a
/// panel-only bootstrap over a fixed partition is close to toothless (see [`Regroup`]).
#[allow(clippy::too_many_arguments)]
pub fn annotate_with_communities(
    input: &InputEmbeddings<'_>,
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    community: &[usize],
    n_comm: usize,
    regroup: Option<&Regroup<'_>>,
    cfg: &TermOraConfig,
) -> Result<CommunityCalls> {
    annotate_inner(
        input,
        markers_path,
        out_prefix,
        use_idf,
        community,
        n_comm,
        regroup,
        cfg,
    )
}

/// The shared core. `regroup` re-derives the grouping for one bootstrap replicate ([`Regroup`]).
#[allow(clippy::too_many_arguments)]
fn annotate_inner(
    input: &InputEmbeddings<'_>,
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    community: &[usize],
    n_comm: usize,
    regroup: Option<&Regroup<'_>>,
    cfg: &TermOraConfig,
) -> Result<CommunityCalls> {
    anyhow::ensure!(
        cfg.obo.is_some() == cfg.label_cl.is_some(),
        "--obo and --label-cl must be given together to run the ontology layer (got only one)"
    );
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
    anyhow::ensure!(n >= 2, "term-ORA needs ≥ 2 cells, found {n}");
    anyhow::ensure!(
        community.len() == n,
        "community len {} != cell rows {n}",
        community.len()
    );
    anyhow::ensure!(n_comm >= 1, "need ≥ 1 community, got {n_comm}");
    info!("term-ORA: β [{g} × {h}], cells [{n} × {h}], {n_comm} group(s)");

    let (type_names, type_markers) =
        parse_and_match_markers(markers_path, gene_names, use_idf, cfg.min_panel_coverage)?;
    let c = type_names.len();
    anyhow::ensure!(
        c >= 2,
        "need ≥ 2 cell types with matched markers, found {c}"
    );
    info!(
        "markers: {c} types, {} matched (gene,type) entries",
        type_markers.iter().map(Vec::len).sum::<usize>()
    );
    write_marker_embeddings(
        out_prefix,
        feature_emb,
        gene_names,
        &type_names,
        &type_markers,
        h,
    )?;

    //////////////////////////////////////////////////////////
    // 1. term centroids (un-normalized, IDF-weighted mean) //
    //////////////////////////////////////////////////////////
    let mut beta_flat = row_major(feature_emb);
    // Before ANY of this reads a feature row: zero the rows the co-embedding parked at the centre
    // of the cell cloud. That is its signature for a gene it never learned, and it arrives wearing
    // a perfectly healthy unit-norm coordinate — so `live_row` cannot see it, and it would be
    // averaged into a centroid and drag it to the hub, where it is close to every cell at once.
    // Zeroing restores the `live_row` contract, and every consumer below inherits the fix.
    // See `super::hub_call`.
    super::hub_call::zero_hub_parked(&mut beta_flat, cell_emb, g, h);

    // …and now drop the types the panel cannot locate at all. Emptying a type's marker list is the
    // single lever: `term_centroids` then leaves it at the origin, `assign_nearest` excludes the
    // origin, and the bootstrap's live panel, the panel null and the support null all see a type
    // with nothing in it. One decision, inherited everywhere.
    let mut type_markers = type_markers;
    drop_unsupported_types(
        &beta_flat,
        &type_names,
        &mut type_markers,
        h,
        cfg.min_markers,
    )?;

    let (centroids, n_live) = term_centroids(&beta_flat, &type_markers, h); // [c × h] row-major
    report_marker_liveness(&type_names, &type_markers, &n_live);

    ////////////////////////////////////////////////////////////////////////
    // 1b. is this panel better than a panel that means nothing? (bias)   //
    ////////////////////////////////////////////////////////////////////////
    let cell_flat = row_major(cell_emb);
    let panel_null = (cfg.panel_perm > 0).then(|| {
        super::panel_null::run_panel_null(
            &beta_flat,
            &cell_flat,
            &type_markers,
            h,
            cfg.panel_perm,
            cfg.seed,
        )
    });
    if let Some(pn) = panel_null.as_ref() {
        write_panel_null(out_prefix, &type_names, pn)?;
        report_panel_null(pn, &type_names);
    }

    ////////////////////////////////////////////////////////////////////////////
    // 2–3. per-cell call: bare nearest-centroid, or the marker bootstrap     //
    ////////////////////////////////////////////////////////////////////////////
    // Both paths hand the same `(assign, dist)` contract to the over-representation step
    // below, so everything downstream — the permutation null, the FDR call, the ontology
    // layer, `CommunityCalls` — is identical whichever one ran. Under the bootstrap the call
    // is the consensus over resampled panels, and a cell whose call is not reproducible
    // abstains; the MAD distance gate still runs on top, since it catches a different failure
    // (a cell that sits nowhere near the centroid it stably picked).
    // `ln(i!)` up to the full cell count — a superset of every replicate's population, so the
    // hypergeometric tables share one table instead of rebuilding it on all `--n-boot` draws.
    let lnfact = ln_factorials(n);
    let b_eff = capped_n_perm(cfg.n_perm, n_comm);
    if b_eff < cfg.n_perm {
        info!(
            "{n_comm} clusters pool the permutation null to {b_eff}×{n_comm} = {} draws per term; \
             taking {b_eff} of the {} requested (the pool, not the draw count, is what resolves \
             the tail)",
            b_eff * n_comm,
            cfg.n_perm
        );
    }

    let mut sup_null: Option<super::support_null::SupportNull> = None;
    // The partition the cluster-level outputs are reported against. It starts as the caller's
    // (one Leiden draw off `--seed`) and is replaced by the ensemble's medoid once the bootstrap
    // has drawn its `B` — see `super::consensus`.
    let mut reported: Option<Partition> = None;
    let (mut assign, dist, mut boot) = match cfg.bootstrap.as_ref() {
        None => {
            let (assign, dist) = assign_nearest(&cell_flat, n, &centroids, c, h);
            (assign, dist, None)
        }
        Some(bcfg) => {
            // **The partitions do not depend on the marker panel.** Derive them once, up front,
            // and every replicate — and later every shuffled-panel null replicate — reuses them.
            // This is what makes the support null affordable: re-clustering is ~94% of a
            // replicate's cost, so a null over P shuffles × B replicates would otherwise pay for
            // P·B Leiden runs when B is all that is ever needed.
            let partitions: Vec<Partition> = match regroup.filter(|_| bcfg.recluster) {
                Some(f) => crate::stop::par_replicates(bcfg.n_boot, "clustering", |b| {
                    let comm = f(cfg.seed.wrapping_add(b as u64))?;
                    let m = n_communities(&comm);
                    Ok((comm, m))
                })?,
                // Grouping held fixed: one partition, shared by every draw.
                None => vec![(community.to_vec(), n_comm)],
            };
            // **Report the ensemble's centre, not one draw from it.** The cluster-level outputs —
            // `community`, the cluster × term matrices, the per-community calls — used to come off
            // whichever partition `--seed` produced, which is a coin toss among near-equal optima.
            // The medoid is the partition that agrees most with all the others we just drew, and it
            // costs nothing extra because we are holding them anyway.
            if partitions.len() > 1 {
                let m = super::consensus::medoid(&partitions);
                let agree = m.agreement;
                info!(
                    "reporting the medoid of {} partitions: mean ARI to the rest {agree:.3} (an \
                     arbitrary draw would score {:.3}), {} communities. The cluster-level outputs \
                     are the ensemble's most typical partition, not the one `--seed` drew. This \
                     does NOT make them reproducible across runs — the kNN graph, not the Leiden \
                     seed, is what differs between runs.",
                    partitions.len(),
                    m.ensemble_mean,
                    partitions[m.best].1,
                );
                if agree < 0.5 {
                    warn!(
                        "the partitions barely agree with one another (mean ARI {agree:.3}): no \
                         single clustering means much here, so read `community` and the cluster × \
                         term matrices as one draw among many. The per-cell consensus label is \
                         unaffected — it is averaged over all {} partitions.",
                        partitions.len()
                    );
                }
                reported = Some(partitions[m.best].clone());
            }
            let step =
                |b: usize, fine: &[usize], cent: &[f32]| -> Result<Option<(Vec<usize>, usize)>> {
                    replicate_label(
                        fine,
                        cent,
                        &cell_flat,
                        &partitions[b % partitions.len()],
                        n,
                        c,
                        h,
                        &lnfact,
                        cfg,
                    )
                };
            let post = run_marker_bootstrap(
                &beta_flat,
                &cell_flat,
                &type_markers,
                h,
                bcfg,
                cfg.seed,
                Some(&step),
            )?;
            // Calibrate the support: what would this cell's agreement look like if the panel
            // carried no type information at all? Reuses the very partitions the observed run
            // drew, so the only thing that differs between the two is the panel's *meaning*.
            if cfg.support_perm > 0 {
                if let Some(con) = post.coarse.as_ref() {
                    sup_null = super::support_null::run_support_null(
                        &beta_flat,
                        &cell_flat,
                        &type_markers,
                        h,
                        &partitions,
                        &lnfact,
                        &con.support,
                        cfg.support_perm,
                        cfg,
                        bcfg,
                    )?;
                }
            }
            (post.assign.clone(), post.dist.clone(), Some(post))
        }
    };
    let n_unstable = assign.iter().filter(|&&t| t == UNASSIGNED).count();

    let mut n_outliers = 0usize;
    if cfg.assign_qc {
        n_outliers = prune_outliers(&mut assign, &dist, c, cfg.assign_mad);
    }
    let n_assigned = assign.iter().filter(|&&t| t != UNASSIGNED).count();
    if boot.is_some() {
        info!(
            "assignment: {n_assigned}/{n} cells called ({n_unstable} unreproducible under the \
             marker bootstrap, {n_outliers} further pruned as distance outliers)"
        );
    } else {
        info!(
            "assignment: {n_assigned}/{n} cells assigned ({n_outliers} pruned as distance outliers)"
        );
    }
    anyhow::ensure!(
        n_assigned >= 2,
        "only {n_assigned} cells remain assigned — loosen --assign-mad / --min-support, \
         or check that the marker panel was trained into the embedding"
    );
    if let Some(post) = boot.as_ref() {
        report_bootstrap(post, &type_names);
    }

    /////////////////////////////////////////////////////////////////////
    // 5. cluster × term over-representation + permutation calibration //
    /////////////////////////////////////////////////////////////////////
    // From here down, "the clustering" means the medoid of the bootstrap's partitions when there
    // was one, and the caller's single partition otherwise. Every cluster-level output below —
    // the ORA, the calls, `community`, the per-community consensus — reads the same one.
    let (community, n_comm) = match reported.as_ref() {
        Some((p, m)) => (p.as_slice(), *m),
        None => (community, n_comm),
    };
    let ora = cluster_term_ora(&assign, community, n_comm, c, &lnfact, Want::Report, cfg);

    /////////////////////////////////////////////
    // 6. cluster calls → per-cell firm labels //
    /////////////////////////////////////////////
    let cluster_label = cluster_calls(&ora, n_comm, c, cfg.fdr_alpha);
    ////////////////////////////////////////////////////////////////////////////
    // 6b. the shipped label: one partition's word, or the consensus of many  //
    ////////////////////////////////////////////////////////////////////////////
    // Without the bootstrap, `coarse_label` is whatever this single (irreproducible) Leiden
    // partition happened to say, and `coarse_conf` is a softmaxed test statistic that is
    // identical for every cell in a cluster. With it, both come from the replicates: the
    // label is the one the resampled panels and re-derived partitions agreed on, and the
    // confidence is *how often they agreed* — a per-cell number, and one that finally means
    // something operational ("re-run this and you'd get the same answer this fraction of the
    // time").
    // **The calibrated cutoff, applied.** `--support-perm` was computing an FDR and gating nothing
    // — three answers to "may this call stand?" (an arbitrary bar, a sign test, and a calibrated
    // q) and the one we paid the most for got no vote. It gets one now: a cell whose support is no
    // better than a meaningless panel achieves is not called, whatever `--min-support` says.
    //
    // This is strictly the stronger test. Measured, a *shuffled* panel still earns a
    // mean support of 0.60 — so the default bar of 0.50 sits BELOW the null, and kept 91% of cells
    // where the FDR keeps 36%.
    if let (Some(sn), Some(b)) = (sup_null.as_ref(), boot.as_mut()) {
        if let Some(con) = b.coarse.as_mut() {
            let mut cut = 0usize;
            for i in 0..n {
                if con.label[i] != UNASSIGNED && sn.q[i] >= cfg.fdr_alpha {
                    con.label[i] = UNASSIGNED;
                    cut += 1;
                }
            }
            info!(
                "support null ({} shuffled panels): {cut} call(s) dropped for failing the \
                 calibrated cutoff (support_q >= {}); a panel carrying no type information \
                 attains a mean support of {:.2} here, so `--min-support` alone was not a test",
                sn.n_perm,
                cfg.fdr_alpha,
                sn.null_support.iter().sum::<f32>() / n.max(1) as f32,
            );
        }
    }
    let consensus: Option<&CoarseConsensus> = boot.as_ref().and_then(|b| b.coarse.as_ref());
    let (coarse_label, coarse_conf): (Vec<Box<str>>, Vec<f32>) = match consensus {
        Some(con) => {
            report_consensus(con, n);
            (
                con.label
                    .iter()
                    .map(|&t| label_of(t, &type_names))
                    .collect(),
                con.support.clone(),
            )
        }
        None => (
            (0..n)
                .map(|i| label_of(cluster_label[community[i]], &type_names))
                .collect(),
            (0..n)
                .map(|i| {
                    let k = community[i];
                    match cluster_label[k] {
                        UNASSIGNED => 0.0,
                        t => ora.q_soft[k * c + t],
                    }
                })
                .collect(),
        ),
    };

    /////////////
    // outputs //
    /////////////
    let comm_names: Vec<Box<str>> = (0..n_comm)
        .map(|k| format!("K{k}").into_boxed_str())
        .collect();
    let sizes = cluster_sizes(community, n_comm);
    write_annot_parquet(
        out_prefix,
        cell_names,
        community,
        &sizes,
        &coarse_label,
        &assign,
        &dist,
        &type_names,
        &ora,
        &cluster_label,
        boot.as_ref(),
        consensus,
        sup_null.as_ref(),
    )?;
    if let (Some(post), Some(con)) = (boot.as_ref(), consensus) {
        write_bootstrap_outputs(
            out_prefix,
            cell_names,
            gene_names,
            &type_names,
            &type_markers,
            post,
            con,
        )?;
    }
    // membership.tsv + argmax.tsv on the firm (cluster-driven) label, the shared
    // contract `gem-summary` / `data-beans stat -g` consume.
    write_label_tsvs(out_prefix, cell_names, &coarse_label, &coarse_conf)?;
    write_cluster_term_matrices(out_prefix, &comm_names, &type_names, &ora)?;
    write_calibration(out_prefix, &ora, n_assigned, n_outliers)?;
    log_cluster_calls(&cluster_label, &type_names, &sizes);

    //////////////////////////////////////////////////////////////////
    // 7. optional ontology (TreeBH over the cluster × term matrix) //
    //////////////////////////////////////////////////////////////////
    if let (Some(obo), Some(label_cl)) = (cfg.obo.as_deref(), cfg.label_cl.as_deref()) {
        run_ontology(
            out_prefix,
            obo,
            label_cl,
            &comm_names,
            &type_names,
            &ora,
            cfg,
        )?;
    }

    // Per-community calls, so a trajectory caller can name each node directly.
    //
    // Under the bootstrap the replicates each invent their own grouping, so a *replicate's*
    // community `k` means nothing to the caller — but the caller's own partition is right here,
    // and its cells carry consensus labels. So a node is named by the label its cells actually
    // hold (a plurality vote over `coarse_label`), and its confidence is the mean support of the
    // cells that voted for it: "re-run this and this node keeps this name this often". That is a
    // number `--root-type` can act on. Without the bootstrap, nothing has changed: the node's
    // call is its own FDR-gated top term, and the confidence is the softmaxed statistic.
    let comm_calls = match consensus {
        Some(con) => community_consensus_calls(community, n_comm, con, &type_names),
        None => CommunityCalls {
            labels: (0..n_comm)
                .map(|k| label_of(cluster_label[k], &type_names))
                .collect(),
            confidence: (0..n_comm)
                .map(|k| match cluster_label[k] {
                    UNASSIGNED => 0.0,
                    t => ora.q_soft[k * c + t],
                })
                .collect(),
        },
    };
    Ok(comm_calls)
}

/// Name each of the *caller's* communities by the consensus its cells reached, with the mean
/// bootstrap support of the voters as the confidence. A community whose cells could not hold a
/// label is `unassigned` at confidence 0 — which is the honest answer for a trajectory node the
/// resampling could not name.
fn community_consensus_calls(
    community: &[usize],
    n_comm: usize,
    con: &CoarseConsensus,
    type_names: &[Box<str>],
) -> CommunityCalls {
    let c = type_names.len();
    // votes[k][t] = cells of community k whose consensus label is t; `c` == unassigned.
    let mut votes = vec![0usize; n_comm * (c + 1)];
    let mut support = vec![0f32; n_comm * (c + 1)];
    for (i, &k) in community.iter().enumerate() {
        let t = match con.label[i] {
            UNASSIGNED => c,
            t => t,
        };
        votes[k * (c + 1) + t] += 1;
        support[k * (c + 1) + t] += con.support[i];
    }
    let (mut labels, mut confidence) = (Vec::with_capacity(n_comm), Vec::with_capacity(n_comm));
    for k in 0..n_comm {
        let row = &votes[k * (c + 1)..(k + 1) * (c + 1)];
        // The plurality among *called* cells; `unassigned` wins only if nothing else was called.
        let best = (0..c).max_by_key(|&t| row[t]).unwrap_or(c);
        if row[best] == 0 {
            labels.push(Box::from(enrichment::UNASSIGNED_LABEL));
            confidence.push(0.0);
        } else {
            labels.push(type_names[best].clone());
            confidence.push(support[k * (c + 1) + best] / row[best] as f32);
        }
    }
    CommunityCalls { labels, confidence }
}

/////////////////////////////////
// 1–2. centroids + assignment //
/////////////////////////////////

/// `[c × h]` row-major IDF-weighted mean of each type's marker feature embeddings — the
/// **un-normalized** centroid (the Euclidean prototype) — plus the number of *live*
/// markers each type's centroid was actually built from ([`live_row`]). Empty types get
/// a zero row.
///
/// A dead marker is skipped in numerator *and* denominator. Counting it in `wsum` would
/// divide a partial sum by the full weight, pulling the centroid toward the origin in
/// proportion to the type's dead-marker fraction — and a short centroid is not a weak
/// competitor, it is a magnet (see [`assign_nearest`]). Skipping it makes the centroid
/// the honest mean over the markers carrying evidence, and leaves an all-dead type at the
/// origin, where [`assign_nearest`]'s guard excludes it.
pub(super) fn term_centroids(
    feature_emb: &[f32],
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
) -> (Vec<f32>, Vec<usize>) {
    let c = type_markers.len();
    let mut out = vec![0f32; c * h];
    let mut n_live = vec![0usize; c];
    out.par_chunks_mut(h)
        .zip(n_live.par_iter_mut())
        .zip(type_markers.par_iter())
        .for_each(|((row, live), markers)| {
            let mut wsum = 0f32;
            for &(gi, w) in markers {
                let Some(ef) = live_row(feature_emb, gi as usize, h) else {
                    continue;
                };
                *live += 1;
                wsum += w;
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            if wsum > 0.0 {
                for v in row.iter_mut() {
                    *v /= wsum;
                }
            }
        });
    (out, n_live)
}

/// Report the [`term_centroids`] live-marker counts, and warn about the types running on
/// a mostly-dead panel.
///
/// A dead marker still *matches* the panel and counts toward the "matched entries" tally,
/// but contributes nothing to its type's centroid — so a panel that is mostly dead looks
/// exactly like a healthy one from the outside. Say so, and name the flag that fixes it.
/// Drop the cell types the panel cannot locate: those left with fewer than `min_markers` markers
/// carrying a live feature row.
///
/// **A type is not "weakly supported", it is unlocated.** Its centroid is the mean of whatever
/// survived, and the mean of one or two points has no direction worth the name — the sampling
/// variance of a mean of one is not small, it is *undefined*. Worse, a centroid built from too few
/// markers tends to land short, and a short centroid sits near the middle of the cell cloud, where
/// it is close to every cell at once: it does not compete weakly, it becomes a **magnet** and takes
/// the dataset. The honest outcome is that it does not compete at all.
///
/// The drop is done by **emptying the type's marker list**, which is the one lever every consumer
/// downstream already respects: [`term_centroids`] leaves an empty type at the origin,
/// [`assign_nearest`] excludes the origin, and the marker bootstrap, the panel null and the support
/// null all build their pools from these same lists.
///
/// The type keeps its name and its column in every output — it simply never wins a cell. Silently
/// renumbering the types would be worse than the disease.
fn drop_unsupported_types(
    beta_flat: &[f32],
    type_names: &[Box<str>],
    type_markers: &mut MarkerSets,
    h: usize,
    min_markers: usize,
) -> Result<usize> {
    // Never below the bootstrap's own invariant: you cannot resample a single point.
    let bar = min_markers.max(MIN_LIVE_MARKERS);

    let mut dropped: Vec<(usize, usize, usize)> = Vec::new(); // (type, matched, live)
    for (t, markers) in type_markers.iter_mut().enumerate() {
        let matched = markers.len();
        let live = markers
            .iter()
            .filter(|&&(gi, _)| live_row(beta_flat, gi as usize, h).is_some())
            .count();
        if live < bar {
            dropped.push((t, matched, live));
            markers.clear();
        }
    }
    if dropped.is_empty() {
        return Ok(0);
    }

    let preview: Vec<String> = dropped
        .iter()
        .take(12)
        .map(|&(t, matched, live)| format!("{} ({live}/{matched})", type_names[t]))
        .collect();
    let tail = if dropped.len() > preview.len() {
        format!(", … {} more", dropped.len() - preview.len())
    } else {
        String::new()
    };
    warn!(
        "dropping {} cell type(s) with fewer than {bar} live markers — they are not weakly \
         located, they are UNLOCATED, and a centroid built from too few markers lands short, near \
         the middle of the cell cloud, where it is close to every cell and becomes a magnet. \
         Shown as live/matched: {}{tail}. They keep their columns in the outputs but can no longer \
         win a cell. If this is unexpected, the markers are missing from the embedding rather than \
         from the data — re-run the fit with `--must-train-features <panel>`.",
        dropped.len(),
        preview.join(", "),
    );

    let surviving = type_markers.iter().filter(|m| !m.is_empty()).count();
    anyhow::ensure!(
        surviving >= 2,
        "only {surviving} cell type(s) have {bar} or more live markers — there is nothing left to \
         tell apart. Lower --min-markers, or (far more likely) the marker panel was never trained \
         into this embedding: re-run the fit with `--must-train-features <panel>`."
    );
    Ok(dropped.len())
}

fn report_marker_liveness(
    type_names: &[Box<str>],
    type_markers: &[Vec<(u32, f32)>],
    n_live: &[usize],
) {
    let n_matched: usize = type_markers.iter().map(Vec::len).sum();
    if n_matched == 0 {
        return;
    }
    info!(
        "marker liveness: {}/{n_matched} matched markers carry a live β row",
        n_live.iter().sum::<usize>()
    );

    // (name, live, matched) for every type more than half dead, worst fraction first.
    // Compared as `live_a · matched_b` vs `live_b · matched_a` — the same order as the
    // ratio, in exact integer arithmetic.
    let mut starved: Vec<(&str, usize, usize)> = type_names
        .iter()
        .zip(type_markers)
        .zip(n_live)
        .filter(|&((_, m), &live)| !m.is_empty() && live * 2 < m.len())
        .map(|((name, m), &live)| (name.as_ref(), live, m.len()))
        .collect();
    if starved.is_empty() {
        return;
    }
    starved.sort_by(|&(_, al, am), &(_, bl, bm)| (al * bm).cmp(&(bl * am)));

    let mut preview: Vec<String> = starved
        .iter()
        .take(10)
        .map(|&(name, live, m)| format!("{name} {live}/{m}"))
        .collect();
    if starved.len() > preview.len() {
        preview.push("…".into());
    }
    warn!(
        "{} type(s) have under half their markers alive in the embedding: {}. \
         A dead marker is a gene the embedding never trained on and whose post-hoc \
         projection failed its null test; those types are scored off the survivors alone. \
         Re-run the embedding with `--must-train-features <panel>` to train on the panel.",
        starved.len(),
        preview.join(", "),
    );
}

/// Nearest-centroid assignment by squared Euclidean distance. Returns
/// `(assign[n], dist[n])` where `dist` is the Euclidean distance to the assigned
/// centroid.
///
/// **A zero-norm centroid can never win.** [`term_centroids`] leaves a type with no
/// usable markers at the origin (it only divides when `wsum > 0`). Cells here are
/// the *raw* embedding, so that centroid sits at squared distance `‖cell‖²` from
/// every cell — and therefore beats every real prototype for any cell nearer the
/// origin than to any of them. It is not a weak competitor, it is a magnet: on a
/// bone-marrow gem run, four types matched zero markers and one of them captured
/// 6814 / 15315 = 44.5% of all cells, while the other three captured none (the
/// strict `<` below lets only the first such type by index ever win).
///
/// `parse_and_match_markers` drops types that matched no gene, which is the common
/// cause but not the only one: a zero centroid also arises when every marker's gene
/// index is out of range, when the IDF weights all vanish, or when the embedding
/// rows cancel. None of those survive as an *empty marker list*, so the guard here
/// is a strictly larger net, not a duplicate of the drop. The cosine path
/// (`type_scores`) is immune for free, since a zero centroid scores 0.
fn assign_nearest(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    c: usize,
    h: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut live: Vec<bool> = (0..c)
        .map(|t| live_row(centroids, t, h).is_some())
        .collect();
    let n_degenerate = live.iter().filter(|&&l| !l).count();
    if n_degenerate == c {
        // Nothing to choose between; keep the old (arbitrary) behaviour rather than
        // returning an infinite distance that would poison the MAD prune downstream.
        warn!("all {c} term centroids are zero-norm — assignment is meaningless");
        live.iter_mut().for_each(|l| *l = true);
    } else if n_degenerate > 0 {
        warn!(
            "{n_degenerate} of {c} term centroid(s) are zero-norm and were excluded from \
             nearest-centroid assignment (they would sit at constant distance from every cell)"
        );
    }
    let mut assign = vec![0usize; n];
    let mut dist = vec![0f32; n];
    assign
        .par_iter_mut()
        .zip(dist.par_iter_mut())
        .enumerate()
        .for_each(|(i, (a, d))| {
            let cell = &cell_flat[i * h..(i + 1) * h];
            let mut best = 0usize;
            let mut best_d2 = f32::INFINITY;
            for t in 0..c {
                if !live[t] {
                    continue;
                }
                let ct = &centroids[t * h..(t + 1) * h];
                let mut s = 0f32;
                for (x, y) in cell.iter().zip(ct) {
                    let diff = x - y;
                    s += diff * diff;
                }
                if s < best_d2 {
                    best_d2 = s;
                    best = t;
                }
            }
            *a = best;
            *d = best_d2.max(0.0).sqrt();
        });
    (assign, dist)
}

/// Mark `assign[c] = UNASSIGNED` for cells whose distance to their assigned
/// centroid is a high-side robust outlier (`> median + k·MAD`) within that term
/// — the shared `data_beans::qc_lib` robust-band idiom. Terms with < 3 assigned
/// cells are left intact (too few to define a band). Returns the number pruned.
fn prune_outliers(assign: &mut [usize], dist: &[f32], c: usize, k: f64) -> usize {
    use data_beans::qc_lib::{robust_outlier_keep, Tail};
    // Per-term cell indices (post-assignment).
    let mut per_term: Vec<Vec<usize>> = vec![Vec::new(); c];
    for (i, &t) in assign.iter().enumerate() {
        if t != UNASSIGNED {
            per_term[t].push(i);
        }
    }
    let mut pruned = 0usize;
    for cells in &per_term {
        if cells.len() < 3 {
            continue; // too few to define outliers
        }
        let dists: Vec<f32> = cells.iter().map(|&i| dist[i]).collect();
        let keep = robust_outlier_keep(&dists, k as f32, Tail::Upper, false, None);
        for (&i, &keep_i) in cells.iter().zip(&keep) {
            if !keep_i {
                assign[i] = UNASSIGNED;
                pruned += 1;
            }
        }
    }
    pruned
}

/////////////////
// 4. grouping //
/////////////////

/// Cells per cluster.
fn cluster_sizes(community: &[usize], n_comm: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; n_comm];
    for &k in community {
        if k < n_comm {
            sizes[k] += 1;
        }
    }
    sizes
}

/// Leiden communities over a cosine cell kNN graph (cells L2-normalized for the
/// graph; gem `e_cell` is already unit, so this matches the assignment geometry).
///
/// The kNN graph is now **deterministic** (matrix-util's seeded instant-distance backend), so this
/// step reproduces run-to-run and `seed` pins Leiden on top of a fixed graph. Historically, under
/// the old un-seedable `hnsw_rs` backend it was *not* reproducible: four identical invocations on
/// 15,315 cord-blood cells at `--resolution 8` gave 990 / 132 / 137 / 138 communities, agreeing on
/// only 83–94% of the labels. That cross-run instability is resolved.
///
/// A single partition still should not be over-trusted: within a run, Leiden picks among near-equal
/// modularity optima, so [`MarkerBootstrapConfig::recluster`] reseeds it once per bootstrap
/// replicate and lets that partition-choice uncertainty land in the per-cell support.
fn cell_knn_graph(cell_flat: &[f32], n: usize, h: usize, cfg: &TermOraConfig) -> Result<KnnGraph> {
    let mut cell_u = cell_flat.to_vec();
    super::score::l2_normalize_rows(&mut cell_u, n, h);
    let cell_mat = DMatrix::<f32>::from_row_iterator(n, h, cell_u.iter().copied());
    KnnGraph::from_rows(
        &cell_mat,
        KnnGraphArgs {
            knn: cfg.knn.clamp(1, n - 1),
            block_size: 1000,
            reciprocal: false,
        },
    )
}

/// Leiden over a **prebuilt** cell kNN graph.
///
/// The graph is built once and reused by every bootstrap replicate, and that is deliberate. The
/// bootstrap resamples the *marker panel*; the cell embedding it clusters is identical on every
/// draw, so the graph's input never changes. The graph is also deterministic now (seeded
/// instant-distance), so rebuilding it would reproduce the identical graph — nothing to gain.
///
/// Leiden's `seed` is a different animal and *is* redrawn per replicate: modularity has many
/// near-equal optima and which one the optimiser lands in is a real, load-bearing arbitrary choice.
/// Holding the partition fixed across replicates makes the bootstrap abstain on nothing (measured
/// on the fixed graph: 0% unassigned, and its support falls from AUC 0.93 to 0.69 at separating
/// spurious calls), because a cluster's argmax will not flip when only the panel jiggles. The
/// partition is where the within-run instability lives, so the partition is what gets resampled.
fn cluster_cells(graph: &KnnGraph, n: usize, cfg: &TermOraConfig, seed: u64) -> Vec<usize> {
    super::layout::leiden_from_graph(graph, n, cfg.resolution, seed)
}

//////////////////////////////////////////////////////////////////////////
// one replicate's pipeline half (the `CoarseStep` the bootstrap calls) //
//////////////////////////////////////////////////////////////////////////

/// Turn one replicate's resampled nearest-centroid assignment into the **shipped** label:
/// re-derive the clustering, prune distance outliers, run the cluster × term
/// over-representation test, and take each cluster's FDR-gated call.
///
/// Returns the per-cell label as a column index into `0..=c` (where `c` means `unassigned`),
/// plus this replicate's community count. `None` when the draw was degenerate — too few cells
/// left assigned to test anything — so it drops out of the tally rather than poisoning it.
///
/// A label therefore survives only if it survives *everything that was ever arbitrary* about
/// how we got it: which markers were drawn, and — when `recluster` — which partition the
/// (irreproducible) clustering happened to land in this time.
#[allow(clippy::too_many_arguments)]
pub(super) fn replicate_label(
    fine: &[usize],
    centroids: &[f32],
    cell_flat: &[f32],
    partition: &Partition,
    n: usize,
    c: usize,
    h: usize,
    lnfact: &[f64],
    cfg: &TermOraConfig,
) -> Result<Option<(Vec<usize>, usize)>> {
    let (comm, n_comm) = (&partition.0, partition.1);

    let mut assign = fine.to_vec();
    if cfg.assign_qc {
        let dist = centroid_distances(cell_flat, n, centroids, h, &assign);
        prune_outliers(&mut assign, &dist, c, cfg.assign_mad);
    }
    if assign.iter().filter(|&&t| t != UNASSIGNED).count() < 2 {
        return Ok(None);
    }

    let ora = cluster_term_ora(&assign, comm, n_comm, c, lnfact, Want::CallOnly, cfg);
    let call = cluster_calls(&ora, n_comm, c, cfg.fdr_alpha);
    let per_cell: Vec<usize> = (0..n)
        .map(|i| match call[comm[i]] {
            UNASSIGNED => c, // the `unassigned` column
            t => t,
        })
        .collect();
    Ok(Some((per_cell, n_comm)))
}

/// Distance from each cell to the centroid it was assigned to (`NaN` when unassigned).
fn centroid_distances(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    h: usize,
    assign: &[usize],
) -> Vec<f32> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let t = assign[i];
            if t == UNASSIGNED {
                return f32::NAN;
            }
            let cell = &cell_flat[i * h..(i + 1) * h];
            let ct = &centroids[t * h..(t + 1) * h];
            cell.iter()
                .zip(ct)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .max(0.0)
                .sqrt()
        })
        .collect()
}

//////////////////////////////////////////
// 5. over-representation + calibration //
//////////////////////////////////////////

/// Cluster × term over-representation result. All `[n_comm × c]` matrices are
/// row-major (`[k*c + t]`).
struct OraResult {
    /// `−ln P(X≥a)` analytic hypergeometric statistic (larger = more enriched).
    stat: Vec<f32>,
    /// Permutation-calibrated p (pooled per term across clusters).
    p_perm: Vec<f32>,
    /// BH q of `p_perm`, per cluster row.
    q: Vec<f32>,
    /// FDR-sparse row-softmax Q over significant terms (confidence weights). Only the reported
    /// run needs it — see [`Want`].
    q_soft: Vec<f32>,
    /// Calibration diagnostics. `None` for a bootstrap replicate, which never reads them.
    cal: Option<Calibration>,
}

struct Calibration {
    n_perm: usize,
    median_logratio: f64,
    frac_analytic_anticons: f64,
    lambda_perm: f64,
    ks_perm: f64,
    degenerate_frac: f64,
}

/// How much of the ORA the caller actually intends to read.
///
/// A bootstrap replicate wants a *label*, and reads only `stat` and `q` on its way to one
/// ([`cluster_calls`]). It never looks at `q_soft` or `cal` — but `cal` is the most expensive
/// thing here by a wide margin (two sorts and an inverse-normal CDF over every one of the
/// `n_perm × n_comm × c` pooled null values), so computing it 200 times and discarding it 200
/// times costs more than the permutation it is meant to be calibrating.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Want {
    /// Everything: the reported run, which writes `null_calibration.tsv` and the Q matrix.
    Report,
    /// The label only: one bootstrap replicate.
    CallOnly,
}

/// Cap on the pooled null per term. The permutation statistic is pooled **across clusters** (it
/// is relabeling-invariant), so the pool is `n_perm × n_comm` — and a runaway partition can make
/// that enormous for no gain: `--resolution 8` has produced 1,713 communities on 15k cells, where
/// the full `--num-perm 500` would build and sort 20M f32 per term. This is a **cost ceiling**,
/// not a precision target: at any resolution worth using (tens of clusters) it does nothing and
/// the user's `--num-perm` stands.
const MAX_NULL_POOL: usize = 100_000;

/// Draws actually taken. Only ever *reduces* the caller's request; `null_calibration.tsv` records
/// what was used (`n_perm`), and [`annotate_inner`] says so on the console when it bites.
fn capped_n_perm(requested: usize, n_comm: usize) -> usize {
    if requested == 0 || n_comm == 0 {
        return requested;
    }
    MAX_NULL_POOL.div_ceil(n_comm).clamp(1, requested)
}

/// `lnfact` must cover `0..=n` for the cell count `n` (it is reused across bootstrap replicates,
/// whose `n_tot` is always ≤ `n`).
fn cluster_term_ora(
    assign: &[usize],
    community: &[usize],
    n_comm: usize,
    c: usize,
    lnfact: &[f64],
    want: Want,
    cfg: &TermOraConfig,
) -> OraResult {
    let n = assign.len();

    // Assigned cells only (post-QC) feed the contingency: an unassigned cell is out of the
    // hypergeometric population entirely, not a zero in it.
    let assigned: Vec<usize> = (0..n).filter(|&i| assign[i] != UNASSIGNED).collect();
    let labels: Vec<usize> = assigned.iter().map(|&i| assign[i]).collect();
    let comms: Vec<usize> = assigned.iter().map(|&i| community[i]).collect();
    let n_tot = assigned.len();

    let count = contingency(&comms, &labels, n_comm, c);
    let n_k: Vec<usize> = count
        .chunks(c)
        .map(|row| row.iter().map(|&v| v as usize).sum())
        .collect();
    let mut m_t = vec![0usize; c];
    for &t in &labels {
        m_t[t] += 1;
    }

    // Hypergeometric SF tables. The margins `(n_tot, m_t, n_k)` are fixed under permutation, so
    // each table serves the observed count and every permuted one.
    //
    // Tables are keyed by the margin *pair*, not by `(k, t)`: clusters that happen to share a
    // size share a table. At the resolutions worth using the sizes are all distinct and this
    // saves nothing, but it is what keeps a runaway partition (1,713 clusters, most of them
    // tiny and identically-sized) from building thousands of copies of the same table. `slot`
    // resolves the sharing once, so the lookup itself stays a plain index.
    let mut degrees: Vec<usize> = n_k.clone();
    degrees.sort_unstable();
    degrees.dedup();
    let slot: Vec<usize> = n_k
        .iter()
        .map(|nk| degrees.binary_search(nk).expect("n_k came from degrees"))
        .collect();
    let sf_tables: Vec<Vec<f64>> = (0..degrees.len() * c)
        .into_par_iter()
        .map(|st| hypergeom_sf_table(n_tot, m_t[st % c], degrees[st / c], lnfact))
        .collect();
    let sf_at = |k: usize, t: usize, a: usize| -> f64 {
        let tbl = &sf_tables[slot[k] * c + t];
        tbl.get(a)
            .copied()
            .unwrap_or(if tbl.is_empty() { 1.0 } else { 0.0 })
    };

    let mut p_analytic = vec![1f32; n_comm * c];
    let mut stat = vec![0f32; n_comm * c];
    for k in 0..n_comm {
        for t in 0..c {
            let p = sf_at(k, t, count[k * c + t] as usize).clamp(1e-12, 1.0);
            p_analytic[k * c + t] = p as f32;
            stat[k * c + t] = (-p.ln()) as f32;
        }
    }

    //////////////////////////////////////////////////////////
    // permutation null: pool stat across clusters per term //
    //////////////////////////////////////////////////////////
    // Serial by design. This whole function already runs inside a 200-way rayon fan-out over
    // bootstrap replicates (`marker_bootstrap::run_marker_bootstrap`), so the cores are spoken
    // for; a nested fan-out here would only contend. It also keeps the RNG a single chain, which
    // is what makes `--seed` mean something.
    let b = capped_n_perm(cfg.n_perm, n_comm);
    let mut null_pool: Vec<Vec<f32>> = vec![Vec::with_capacity(b * n_comm); c];
    if b > 0 && n_tot >= 2 {
        let mut perm = labels.clone();
        let mut rng = SmallRng::seed_from_u64(cfg.seed ^ 0x5eed_0a4a);
        for _ in 0..b {
            perm.shuffle(&mut rng);
            let cnt = contingency(&comms, &perm, n_comm, c);
            for k in 0..n_comm {
                for t in 0..c {
                    let p = sf_at(k, t, cnt[k * c + t] as usize).clamp(1e-12, 1.0);
                    null_pool[t].push((-p.ln()) as f32);
                }
            }
        }
    }
    null_pool.par_iter_mut().for_each(|pool| {
        pool.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    });

    // permutation p per (K,T): fraction of the term's pooled null ≥ observed.
    let mut p_perm = vec![1f32; n_comm * c];
    for kt in 0..n_comm * c {
        let pool = &null_pool[kt % c];
        p_perm[kt] = if pool.is_empty() {
            p_analytic[kt]
        } else {
            let ge = pool.len() - lower_bound(pool, stat[kt]);
            (ge as f32 + 1.0) / (pool.len() as f32 + 1.0)
        };
    }

    // BH q per cluster row on the permutation p.
    let mut q = vec![1f32; n_comm * c];
    for k in 0..n_comm {
        let row_q = enrichment::bh_fdr(&p_perm[k * c..(k + 1) * c]);
        q[k * c..(k + 1) * c].copy_from_slice(&row_q);
    }

    // The rest is for the reported run only — a replicate reads `stat` and `q` and stops.
    let (q_soft, cal) = match want {
        Want::CallOnly => (Vec::new(), None),
        Want::Report => (
            // FDR-sparse row-softmax Q (confidence weights): softmax of stat over terms
            // with q < α; zero elsewhere; uniform fallback if a row has no significant term.
            sparse_row_softmax(&stat, &q, n_comm, c, cfg.fdr_alpha, cfg.q_temperature),
            Some(calibrate(&p_analytic, &p_perm, &null_pool, n_comm, c, b)),
        ),
    };

    OraResult {
        stat,
        p_perm,
        q,
        q_soft,
        cal,
    }
}

/// Each cluster's FDR-gated call: its top over-represented term, kept only if significant.
/// [`UNASSIGNED`] when nothing survives `fdr_alpha`.
fn cluster_calls(ora: &OraResult, n_comm: usize, c: usize, alpha: f32) -> Vec<usize> {
    let top = argmax_rows(&ora.stat, n_comm, c);
    (0..n_comm)
        .map(|k| {
            let best = top[k];
            if ora.q[k * c + best] < alpha {
                best
            } else {
                UNASSIGNED
            }
        })
        .collect()
}

/// `[n_comm × c]` row-major contingency counts over the **assigned cells only**: `comms` and
/// `labels` are the compacted, parallel per-assigned-cell arrays, so there is no sentinel to
/// filter and the walk is over `n_tot`, not `n`.
fn contingency(comms: &[usize], labels: &[usize], n_comm: usize, c: usize) -> Vec<u32> {
    let mut count = vec![0u32; n_comm * c];
    for (&k, &t) in comms.iter().zip(labels) {
        if k < n_comm && t < c {
            count[k * c + t] += 1;
        }
    }
    count
}

/// `ln(i!)` for `i ∈ 0..=n` (i.e. `ln_gamma(i+1)`), precomputed once so the
/// per-(cluster,term) SF tables share the factorials rather than recomputing
/// `ln_gamma` for every binomial coefficient.
fn ln_factorials(n: usize) -> Vec<f64> {
    use statrs::function::gamma::ln_gamma;
    (0..=n).map(|i| ln_gamma(i as f64 + 1.0)).collect()
}

/// Upper-tail hypergeometric SF table: `sf[a] = P(X ≥ a)` for a draw of `draws`
/// from a population of `pop` with `succ` successes, `a ∈ 0..=min(succ,draws)`.
/// Log-space PMF for numerical stability. `lnfact[i] = ln(i!)` must cover
/// `0..=pop`. Empty when `pop==0`.
fn hypergeom_sf_table(pop: usize, succ: usize, draws: usize, lnfact: &[f64]) -> Vec<f64> {
    if pop == 0 || succ == 0 || draws == 0 {
        // No successes or no draws ⇒ a is always 0; P(X≥0)=1, P(X≥1)=0.
        return vec![1.0];
    }
    let lnc = |a: usize, b: usize| -> f64 {
        if b > a {
            return f64::NEG_INFINITY;
        }
        lnfact[a] - lnfact[b] - lnfact[a - b]
    };
    let x_hi = succ.min(draws);
    let x_lo = (draws + succ).saturating_sub(pop);
    let ln_den = lnc(pop, draws);
    let mut pmf = vec![0f64; x_hi + 1];
    for (x, p) in pmf.iter_mut().enumerate().take(x_hi + 1).skip(x_lo) {
        *p = (lnc(succ, x) + lnc(pop - succ, draws - x) - ln_den).exp();
    }
    let mut sf = vec![0f64; x_hi + 1];
    let mut acc = 0f64;
    for a in (0..=x_hi).rev() {
        acc += pmf[a];
        sf[a] = acc.min(1.0);
    }
    sf
}

/// Index of the first element ≥ `x` in a sorted slice (count of strictly-smaller).
fn lower_bound(sorted: &[f32], x: f32) -> usize {
    sorted.partition_point(|&v| v < x)
}

/// Index of the first element > `x` in a sorted slice (count of ≤ `x`).
fn upper_bound(sorted: &[f32], x: f32) -> usize {
    sorted.partition_point(|&v| v <= x)
}

/// Per cluster row: softmax of `stat/τ` over terms with `q < α`, zero elsewhere.
/// Rows with no significant term get a uniform distribution (so the argmax
/// confidence is still defined, but small).
fn sparse_row_softmax(
    stat: &[f32],
    q: &[f32],
    n_comm: usize,
    c: usize,
    alpha: f32,
    temperature: f32,
) -> Vec<f32> {
    let tau = temperature.max(1e-6);
    let mut out = vec![0f32; n_comm * c];
    for k in 0..n_comm {
        let sig: Vec<usize> = (0..c).filter(|&t| q[k * c + t] < alpha).collect();
        if sig.is_empty() {
            let u = 1.0 / c as f32;
            for t in 0..c {
                out[k * c + t] = u;
            }
            continue;
        }
        let mx = sig
            .iter()
            .map(|&t| stat[k * c + t])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0f32;
        for &t in &sig {
            let e = ((stat[k * c + t] - mx) / tau).exp();
            out[k * c + t] = e;
            s += e;
        }
        let s = s.max(1e-12);
        for &t in &sig {
            out[k * c + t] /= s;
        }
    }
    out
}

/// Discreteness-robust calibration of the analytic hypergeometric vs the
/// permutation null. `median_logratio = median log10(p_perm/p_analytic)` (≈0
/// calibrated; >0 ⇒ analytic anticonservative); `frac_analytic_anticons` =
/// share with `p_analytic < ½·p_perm`. Machinery sanity: `lambda_perm` /
/// `ks_perm` on leave-one-out null p (≈1 / small when unbiased).
fn calibrate(
    p_analytic: &[f32],
    p_perm: &[f32],
    null_pool: &[Vec<f32>],
    n_comm: usize,
    c: usize,
    b: usize,
) -> Calibration {
    // analytic-vs-perm agreement over observed (K,T).
    let mut logratios: Vec<f64> = Vec::with_capacity(n_comm * c);
    let mut anticons = 0usize;
    for kt in 0..n_comm * c {
        let pa = p_analytic[kt].max(1e-12) as f64;
        let pp = p_perm[kt].max(1e-12) as f64;
        logratios.push((pp / pa).log10());
        if (pa) < 0.5 * pp {
            anticons += 1;
        }
    }
    let median_logratio = {
        let mut v = logratios.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            0.0
        } else {
            v[v.len() / 2]
        }
    };
    let frac_analytic_anticons = anticons as f64 / (n_comm * c).max(1) as f64;

    // Degenerate fraction: terms whose pooled null has no spread.
    let degenerate_terms = null_pool
        .iter()
        .filter(|pool| pool.is_empty() || pool.first() == pool.last())
        .count();
    let degenerate_frac = degenerate_terms as f64 / c.max(1) as f64;

    // Machinery sanity: leave-one-out empirical p of every pooled null value vs its term's pool
    // (uniform under an unbiased permutation null).
    //
    // **mid-p, not the plain tail count.** The statistic is discrete and the pool is full of
    // ties at the floor — a (group, term) with no enriched cells scores exactly 0, and once the
    // assignment is sparse (as it is after the bootstrap abstains on half the cells) that is
    // most of the pool. The plain tail `#{≥x}/m` hands every one of those p = 1, which the
    // clamp turns into z = Φ⁻¹(1e-12) = −7.03 and λ = 49.48/0.4549 = **108.77** — a number that
    // is a property of the clamp, not of the null, and that used to fire a "raise --num-perm"
    // warning no amount of permutation could ever fix. Splitting the ties (`#{>x} + ½#{=x}`)
    // makes λ mean what it claims to again, and leaves the continuous case unchanged.
    let mut loo: Vec<f64> = Vec::new();
    for pool in null_pool {
        let m = pool.len();
        if m < 2 {
            continue;
        }
        for &x in pool {
            let lo = lower_bound(pool, x); // strictly below
            let hi = upper_bound(pool, x); // ≤ x
            let p = (m as f64 - 0.5 * (lo + hi) as f64) / m as f64;
            loo.push(p.clamp(1e-12, 1.0));
        }
    }
    let (lambda_perm, ks_perm) = if loo.len() >= 8 {
        (lambda_from_p(&loo), ks_uniform(&loo))
    } else {
        (f64::NAN, f64::NAN)
    };

    Calibration {
        n_perm: b,
        median_logratio,
        frac_analytic_anticons,
        lambda_perm,
        ks_perm,
        degenerate_frac,
    }
}

/// Genomic-inflation-style λ for one-sided p-values: `median(z²)/0.4549`,
/// `z = Φ⁻¹(1−p)`. ≈1 when p ~ Uniform.
fn lambda_from_p(ps: &[f64]) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let std = Normal::new(0.0, 1.0).unwrap();
    let mut zsq: Vec<f64> = ps
        .iter()
        .map(|&p| {
            let z = std.inverse_cdf((1.0 - p).clamp(1e-12, 1.0 - 1e-12));
            z * z
        })
        .collect();
    zsq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = zsq[zsq.len() / 2];
    med / 0.4549364
}

/// Kolmogorov–Smirnov distance of `ps` from Uniform(0,1).
fn ks_uniform(ps: &[f64]) -> f64 {
    let mut v = ps.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = v.len() as f64;
    let mut d = 0f64;
    for (i, &p) in v.iter().enumerate() {
        let lo = (i as f64) / nf;
        let hi = (i as f64 + 1.0) / nf;
        d = d.max((p - lo).abs()).max((hi - p).abs());
    }
    d
}

/////////////
// outputs //
/////////////

#[allow(clippy::too_many_arguments)]
fn write_annot_parquet(
    out_prefix: &str,
    cell_names: &[Box<str>],
    community: &[usize],
    sizes: &[usize],
    coarse_label: &[Box<str>],
    assign: &[usize],
    dist: &[f32],
    type_names: &[Box<str>],
    ora: &OraResult,
    cluster_label: &[usize],
    boot: Option<&BootstrapResult>,
    consensus: Option<&CoarseConsensus>,
    sup_null: Option<&super::support_null::SupportNull>,
) -> Result<()> {
    let n = cell_names.len();
    let c = type_names.len();
    let comm_i32: Vec<i32> = community.iter().map(|&k| k as i32).collect();
    // How many cells the call was pooled over — its cluster's size. The test's power comes
    // entirely from this number, and a call resting on a handful of cells is a different animal
    // from one resting on hundreds, so it goes out in the parquet rather than being implied.
    let cluster_size: Vec<i32> = community.iter().map(|&k| sizes[k] as i32).collect();
    let fine_label: Vec<Box<str>> = assign.iter().map(|&t| label_of(t, type_names)).collect();
    let is_outlier: Vec<i32> = assign.iter().map(|&t| (t == UNASSIGNED) as i32).collect();
    // Per-cell coarse stats = the cluster's call entry, broadcast to its members.
    let stat_of = |m: &[f32], i: usize| -> f32 {
        let k = community[i];
        match cluster_label[k] {
            UNASSIGNED => f32::NAN,
            t => m[k * c + t],
        }
    };
    let coarse_p: Vec<f32> = (0..n).map(|i| stat_of(&ora.p_perm, i)).collect();
    let coarse_q: Vec<f32> = (0..n).map(|i| stat_of(&ora.q, i)).collect();

    let annot_path = format!("{out_prefix}.annot.parquet");
    let mut cols = vec![
        (Box::from("community"), Column::I32(&comm_i32)),
        (Box::from("cluster_size"), Column::I32(&cluster_size)),
        (Box::from("coarse_label"), Column::Str(coarse_label)),
        (Box::from("fine_label"), Column::Str(&fine_label)),
        (Box::from("fine_distance"), Column::F32(dist)),
        (Box::from("is_outlier"), Column::I32(&is_outlier)),
    ];
    // **`coarse_p`/`coarse_q` are only honest without the bootstrap**, and are withheld with it.
    //
    // They are one partition's word: the p/q that this single (irreproducible) Leiden run gave
    // to the term *it* picked for the cell's cluster. Under the bootstrap `coarse_label` is the
    // consensus over resampled panels and re-partitionings instead — so the two disagree about
    // which term they even describe, and they disagree about how sure to be. Measured on cord
    // blood: 6,891 cells whose consensus label was `NK` carried q between 1e-3 and 6e-3 — flat
    // certainty — next to a `label_support` of 0.50-0.60, a coin flip. Shipping a p-value that
    // confident beside a label that unstable is worse than shipping no p-value at all, and the
    // p-value is the one that is lying.
    if boot.is_none() {
        cols.extend([
            (Box::from("coarse_p"), Column::F32(&coarse_p)),
            (Box::from("coarse_q"), Column::F32(&coarse_q)),
        ]);
    }
    // The bootstrap's per-cell numbers, which are what replace them. These vary cell by cell —
    // the whole point of resampling — where a cluster's p/q is identical for every one of its
    // members. `label_support` is the headline: the fraction of replicates (panel resampled, and
    // the partition re-derived) that agreed on this cell's shipped label.
    // The **mixed annotation**. `coarse_label` is forced to pick one type or give up; this is
    // what the resampling actually said, and it is defined for every cell — including the ones
    // `coarse_label` abstains on, which is exactly where it earns its keep. `HSPC/LMPP` is a real
    // answer; `unassigned` is a refusal to give one.
    //
    // **The set is rendered in canonical (type-index) order, not in support order.** A
    // set-valued label is a category, and a category has to have one spelling: sorting by support
    // would render the same 3-way call as `Erythroid/Granulo-Mono/HSPC` for one cell and
    // `Granulo-Mono/Erythroid/HSPC` for the next, so grouping by it would split one group into
    // `k!` of them. Which member is the most probable is already carried by `coarse_label` and
    // `label_support`; this column's job is to name the *set*.
    let set_str: Vec<Box<str>>;
    let ranked_str: Vec<Box<str>>;
    let set_size: Vec<i32>;
    if let (Some(b), Some(con)) = (boot, consensus) {
        // One spelling of a label set: `unassigned` when empty (too wide to mean anything —
        // see `CoarseConsensus::label_set`), else `label_of` joined by "/". `sort` canonicalises
        // by type index (a set is a *category* with one spelling); leaving it unsorted keeps the
        // credible-set support order (leading first) that `label_ranked` needs.
        let join_set = |set: &[usize], sort: bool| -> Box<str> {
            if set.is_empty() {
                return Box::from(enrichment::UNASSIGNED_LABEL);
            }
            let mut ix = set.to_vec();
            if sort {
                ix.sort_unstable(); // the `unassigned` column is `c`, so it sorts last
            }
            ix.iter()
                .map(|&t| label_of(t, type_names))
                .collect::<Vec<_>>()
                .join("/")
                .into_boxed_str()
        };
        // `label_set` names the *category* (canonicalised); `label_ranked` is its support-ordered
        // twin (largest share first) — the plot reads the leading fate + runner-up from it.
        set_str = con.label_set.iter().map(|s| join_set(s, true)).collect();
        ranked_str = con.label_set.iter().map(|s| join_set(s, false)).collect();
        set_size = con.label_set.iter().map(|s| s.len() as i32).collect();
        cols.extend([
            (Box::from("label_set"), Column::Str(&set_str)),
            (Box::from("label_ranked"), Column::Str(&ranked_str)),
            (Box::from("label_set_size"), Column::I32(&set_size)),
            (
                Box::from("label_set_support"),
                Column::F32(&con.set_support),
            ),
            (Box::from("label_support"), Column::F32(&con.support)),
            (Box::from("label_entropy"), Column::F32(&con.entropy)),
            (Box::from("fine_support"), Column::F32(&b.support)),
            (Box::from("fine_entropy"), Column::F32(&b.entropy)),
        ]);
    }
    // The support, calibrated: `support_q` is a Benjamini–Hochberg FDR across the cells, so a
    // cutoff on it means the same thing whatever the number of types — unlike `label_support`,
    // whose natural scale is `1/C`.
    if let Some(sn) = sup_null {
        cols.extend([
            (Box::from("support_p"), Column::F32(&sn.p)),
            (Box::from("support_q"), Column::F32(&sn.q)),
            (Box::from("null_support"), Column::F32(&sn.null_support)),
        ]);
    }
    write_named_table(&annot_path, "cell", cell_names, &cols)
        .with_context(|| format!("writing {annot_path}"))?;
    info!("wrote {annot_path}");
    Ok(())
}

///////////////////////////////////////
// marker-bootstrap outputs + report //
///////////////////////////////////////

/// The per-cell bootstrap distribution, the marker-deviation table, and the per-type
/// diagnostics.
///
/// `marker_deviation` is the deliverable that states the first error source — *does the
/// embedding actually place this listed gene anywhere near the type it is listed under?* —
/// as an output rather than absorbing it silently into a centroid.
fn write_bootstrap_outputs(
    out_prefix: &str,
    cell_names: &[Box<str>],
    gene_names: &[Box<str>],
    type_names: &[Box<str>],
    type_markers: &[Vec<(u32, f32)>],
    post: &BootstrapResult,
    consensus: &CoarseConsensus,
) -> Result<()> {
    // The distribution over the SHIPPED label — what fraction of the replicates put this cell
    // in each type, and in `unassigned`.
    let mut col_names: Vec<Box<str>> = type_names.to_vec();
    col_names.push(Box::from(enrichment::UNASSIGNED_LABEL));
    let path = format!("{out_prefix}.label_stability.parquet");
    DMatrix::<f32>::from_row_iterator(cell_names.len(), post.c + 1, consensus.post.iter().copied())
        .to_parquet_with_names(&path, (Some(cell_names), Some("cell")), Some(&col_names))
        .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");

    // Long (gene, type) marker table.
    let (mut genes, mut types, mut weights, mut dev, mut live) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (t, markers) in type_markers.iter().enumerate() {
        for (j, &(gi, w)) in markers.iter().enumerate() {
            genes.push(gene_names[gi as usize].clone());
            types.push(type_names[t].clone());
            weights.push(w);
            dev.push(post.marker_dev[t][j]);
            live.push(i32::from(post.marker_live[t][j]));
        }
    }
    let path = format!("{out_prefix}.marker_support.parquet");
    write_named_table(
        &path,
        "gene",
        &genes,
        &[
            (Box::from("cell_type"), Column::Str(&types)),
            (Box::from("idf_weight"), Column::F32(&weights)),
            (Box::from("deviation"), Column::F32(&dev)),
            (Box::from("live"), Column::I32(&live)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");

    let path = format!("{out_prefix}.type_qc.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    // `n_draws` is the number of replicates that actually ran — every support in this run is a
    // fraction of it, and a Ctrl+C makes it smaller than `--n-boot`.
    writeln!(
        f,
        "cell_type\tn_draws\tn_markers\tn_live\tcentroid_jitter\tdecision_gap\tnoise_ratio\tmean_support\toccupancy"
    )?;
    for (t, qc) in post.type_qc.iter().enumerate() {
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\t{:.4}",
            type_names[t],
            post.n_draws,
            type_markers[t].len(),
            qc.n_live,
            qc.centroid_jitter,
            qc.decision_gap,
            noise_ratio(qc),
            qc.mean_support,
            qc.occupancy
        )?;
    }
    info!("wrote {path}");
    Ok(())
}

/// The panel null's per-type verdict: does the type's own gene list place its prototype better
/// than the same number of random marker genes would?
fn write_panel_null(
    out_prefix: &str,
    type_names: &[Box<str>],
    pn: &super::panel_null::PanelNull,
) -> Result<()> {
    let path = format!("{out_prefix}.panel_null.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    writeln!(
        f,
        "cell_type\tn_live\toccupancy\tnull_occupancy\tcost\tnull_cost\tcost_ratio\tp"
    )?;
    for (t, name) in type_names.iter().enumerate() {
        writeln!(
            f,
            "{name}\t{}\t{:.4}\t{:.4}\t{:.1}\t{:.1}\t{:.3}\t{:.4}",
            pn.n_live[t],
            pn.occupancy[t],
            pn.null_occupancy[t],
            pn.cost[t],
            pn.null_cost[t],
            pn.null_cost[t] / pn.cost[t].max(1e-9),
            pn.p[t]
        )?;
    }
    info!("wrote {path}");
    Ok(())
}

/// Name the types whose gene list is doing no better than random genes would. These are the types
/// the point-estimate path fills anyway, confidently, and that the bootstrap will *also* call
/// confidently — because every resample of a wrong panel is wrong the same way.
fn report_panel_null(pn: &super::panel_null::PanelNull, type_names: &[Box<str>]) {
    let mut dud: Vec<(&str, f32, f32)> = type_names
        .iter()
        .enumerate()
        .filter(|&(t, _)| pn.n_live[t] > 0 && pn.p[t] > 0.05)
        .map(|(t, name)| (name.as_ref(), pn.p[t], pn.occupancy[t]))
        .collect();
    info!(
        "marker-panel null ({} draws/type): {}/{} types place their prototype better than random \
         genes of the same number (p < 0.05)",
        pn.n_perm,
        type_names.len() - dud.len(),
        type_names.len()
    );
    if dud.is_empty() {
        return;
    }
    dud.sort_by(|a, b| b.1.total_cmp(&a.1));
    let preview: Vec<String> = dud
        .iter()
        .take(10)
        .map(|&(name, p, occ)| format!("{name} (p={p:.2}, holds {:.1}% of cells)", occ * 100.0))
        .collect();
    warn!(
        "{} type(s) are NOT identified by their own markers — random genes of the same number \
         place their prototype just as well, yet they still hold cells: {}. This is BIAS, and no \
         amount of bootstrapping will find it: every resample of a wrong panel is wrong the same \
         way, so these calls come back *stable*. See {{out}}.panel_null.tsv.",
        dud.len(),
        preview.join(", ")
    );
}

/// How far the centroid moves under resampling, against the margin the assignment is actually
/// decided by. **Above 1 the type's cells are being assigned by noise**: the panel does not
/// place the type to within the precision the decision needs.
fn noise_ratio(qc: &super::marker_bootstrap::TypeQc) -> f32 {
    if qc.decision_gap > 0.0 {
        qc.centroid_jitter / qc.decision_gap
    } else {
        f32::NAN
    }
}

/// Say plainly how much of the labelling was resting on arbitrary choices: how far the
/// clustering wandered across replicates, and how many cells could not hold a label through it.
fn report_consensus(con: &CoarseConsensus, n: usize) {
    let called = con.label.iter().filter(|&&t| t != UNASSIGNED).count();
    let mean_support = con.support.iter().sum::<f32>() / n as f32;
    let (lo, hi) = con
        .n_comm
        .iter()
        .fold((usize::MAX, 0), |(l, h), &k| (l.min(k), h.max(k)));
    info!(
        "stability bootstrap ({} replicates, panel + clustering resampled): {called}/{n} cells \
         held a label (mean support {mean_support:.2}); the clustering itself ranged over \
         {lo}–{hi} communities across replicates",
        con.n_comm.len()
    );
    if hi > 2 * lo.max(1) {
        warn!(
            "the clustering is unstable — replicates ranged from {lo} to {hi} communities on the \
             SAME data. Any single partition's labelling is one draw from that, which is why the \
             shipped label is now the consensus rather than one run's word for it."
        );
    }
}

/// Name the types whose panel cannot place them to the precision the assignment needs. These
/// are exactly the types the point-estimate path hands a confident share of cells anyway.
fn report_bootstrap(post: &BootstrapResult, type_names: &[Box<str>]) {
    let mut weak: Vec<(&str, f32, f32)> = post
        .type_qc
        .iter()
        .zip(type_names)
        .filter(|(qc, _)| noise_ratio(qc) > 1.0)
        .map(|(qc, name)| (name.as_ref(), noise_ratio(qc), qc.occupancy))
        .collect();
    if weak.is_empty() {
        return;
    }
    weak.sort_by(|a, b| b.1.total_cmp(&a.1));
    let preview: Vec<String> = weak
        .iter()
        .take(10)
        .map(|&(name, r, occ)| format!("{name} ({r:.1}×, {occ:.1}% of cells)", occ = occ * 100.0))
        .collect();
    warn!(
        "{} type(s) move further under marker resampling than the margin their assignment is \
         decided by — their cells are being called by noise: {}. This is an EMBEDDING problem, \
         not a statistics one: re-run `faba gem --must-train-features <panel>` so the marker \
         genes are trained rather than post-hoc projected. See {{out}}.type_qc.tsv.",
        weak.len(),
        preview.join(", ")
    );
}

fn write_cluster_term_matrices(
    out_prefix: &str,
    comm_names: &[Box<str>],
    type_names: &[Box<str>],
    ora: &OraResult,
) -> Result<()> {
    let n_comm = comm_names.len();
    let c = type_names.len();
    let to_mat = |flat: &[f32]| DMatrix::<f32>::from_row_iterator(n_comm, c, flat.iter().copied());
    for (suffix, flat) in [
        ("cluster_term_p", &ora.p_perm),
        ("cluster_term_q", &ora.q),
        ("cluster_term_softq", &ora.q_soft),
    ] {
        let path = format!("{out_prefix}.{suffix}.parquet");
        to_mat(flat)
            .to_parquet_with_names(&path, (Some(comm_names), Some("cluster")), Some(type_names))
            .with_context(|| format!("writing {path}"))?;
    }
    info!("wrote {out_prefix}.cluster_term_{{p,q,Q}}.parquet ({n_comm} clusters × {c} terms)");
    Ok(())
}

fn write_calibration(
    out_prefix: &str,
    ora: &OraResult,
    n_assigned: usize,
    n_outliers: usize,
) -> Result<()> {
    let Some(cal) = ora.cal.as_ref() else {
        return Ok(()); // a `Want::CallOnly` run has nothing to report
    };
    let path = format!("{out_prefix}.null_calibration.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    writeln!(f, "metric\tvalue")?;
    writeln!(f, "n_perm\t{}", cal.n_perm)?;
    writeln!(f, "n_assigned\t{n_assigned}")?;
    writeln!(f, "n_outliers_pruned\t{n_outliers}")?;
    writeln!(
        f,
        "median_logratio_perm_over_analytic\t{:.4}",
        cal.median_logratio
    )?;
    writeln!(
        f,
        "frac_analytic_anticonservative\t{:.4}",
        cal.frac_analytic_anticons
    )?;
    writeln!(f, "lambda_perm\t{:.4}", cal.lambda_perm)?;
    writeln!(f, "ks_perm_uniform\t{:.4}", cal.ks_perm)?;
    writeln!(f, "degenerate_frac\t{:.4}", cal.degenerate_frac)?;
    info!("wrote {path}");

    // Console summary + warnings.
    eprintln!("\nNull calibration (permutation B={})", cal.n_perm);
    eprintln!(
        "  analytic vs permutation:  median log10(p_perm/p_analytic)={:.3}  anticonservative-frac={:.3}",
        cal.median_logratio, cal.frac_analytic_anticons
    );
    eprintln!(
        "  permutation machinery:    lambda_perm={:.3}  ks_uniform={:.3}  degenerate-frac={:.3}",
        cal.lambda_perm, cal.ks_perm, cal.degenerate_frac
    );
    if cal.median_logratio > 0.3 || cal.frac_analytic_anticons > 0.2 {
        log::warn!(
            "analytic hypergeometric looks anticonservative (median log-ratio {:.2}); \
             the reported p/q use the permutation null",
            cal.median_logratio
        );
    }
    // λ straying from 1 has two very different causes, and telling the user to raise --num-perm
    // is right for exactly one of them. When a term's pooled null has no spread — too few cells
    // are assigned to it for any relabeling to move its count — its statistic is a constant, and
    // no number of permutations will ever give that constant a distribution. `degenerate_frac`
    // is precisely the share of terms in that state, so let it pick the message.
    if cal.lambda_perm.is_finite() && !(0.7..=1.4).contains(&cal.lambda_perm) {
        if cal.degenerate_frac > 0.1 {
            log::warn!(
                "permutation null lambda_perm={:.2} strays from 1, but {:.0}% of terms have a \
                 null with no spread — too few cells are assigned to them to test at all. More \
                 permutations cannot fix this; a marker panel the embedding actually trained on \
                 can (`faba gem --must-train-features`).",
                cal.lambda_perm,
                cal.degenerate_frac * 100.0
            );
        } else {
            log::warn!(
                "permutation null lambda_perm={:.2} strays from 1 — raise --num-perm",
                cal.lambda_perm
            );
        }
    }
    eprintln!();
    Ok(())
}

/// Print each cluster's call, largest first. A sane partition has tens of clusters; a runaway
/// one (`--resolution 8` has produced 1,713 on 15k cells) is truncated rather than allowed to
/// bury the log — and the truncation is stated, not silent.
const MAX_CLUSTERS_LISTED: usize = 64;

fn log_cluster_calls(cluster_label: &[usize], type_names: &[Box<str>], sizes: &[usize]) {
    let n_comm = sizes.len();
    info!("cluster calls ({n_comm} clusters):");
    let mut order: Vec<usize> = (0..n_comm).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(sizes[k]));
    for &k in order.iter().take(MAX_CLUSTERS_LISTED) {
        let name = label_of(cluster_label[k], type_names);
        info!("  K{k:<3} {:6} cells  {name}", sizes[k]);
    }
    if let Some(rest) = n_comm.checked_sub(MAX_CLUSTERS_LISTED).filter(|&r| r > 0) {
        info!("  … and {rest} smaller cluster(s)");
    }
}

/////////////////
// 7. ontology //
/////////////////

fn run_ontology(
    out_prefix: &str,
    obo: &str,
    label_cl: &str,
    comm_names: &[Box<str>],
    type_names: &[Box<str>],
    ora: &OraResult,
    cfg: &TermOraConfig,
) -> Result<()> {
    let n_comm = comm_names.len();
    let c = type_names.len();
    // cluster × term permutation p as the ontology leaf evidence; Q as node mass.
    let p_mat = DMatrix::<f32>::from_row_iterator(n_comm, c, ora.p_perm.iter().copied());
    let q_mat = DMatrix::<f32>::from_row_iterator(n_comm, c, ora.q_soft.iter().copied());
    super::ontology_obo::annotate_ontology_from_obo(
        out_prefix,
        label_cl,
        obo,
        cfg.ontology_fdr_q,
        cfg.ontology_by,
        enrichment::OntologyScore::Pvalue(&p_mat),
        Some(&q_mat),
        comm_names,
        type_names,
    )?;
    Ok(())
}
