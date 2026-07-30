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

mod centroids;
mod clustering;
mod ora;
// `term_ora`'s own writers/reports; `super::output` below is the shared one.
mod output;

use super::marker_bootstrap::{run_marker_bootstrap, CoarseConsensus, MarkerBootstrapConfig};
use super::markers::parse_and_match_markers;
use super::output::{write_label_tsvs, write_marker_embeddings};
use super::score::row_major;
use super::{n_communities, InputEmbeddings, UNASSIGNED};
use anyhow::Result;
use centroids::{
    assign_nearest, centroid_distances, drop_unsupported_types, prune_outliers,
    report_marker_liveness,
};
use clustering::{cell_knn_graph, cluster_cells, cluster_sizes};
use log::{info, warn};
use ora::{capped_n_perm, cluster_calls, cluster_term_ora, ln_factorials, Want};
use output::{
    log_cluster_calls, report_bootstrap, report_consensus, report_panel_null, run_ontology,
    write_annot_parquet, write_bootstrap_outputs, write_calibration, write_cluster_term_matrices,
    write_panel_null,
};

/// `super::panel_null` reaches the centroids through here, so the path it has always used
/// (`term_ora::term_centroids`) survives the split.
pub(super) use centroids::term_centroids;

#[cfg(test)]
mod tests;
/// Named only by [`tests`], which reaches them through its `use super::*`; nothing on the
/// live path here mentions either by name.
#[cfg(test)]
use ora::{OraResult, MAX_NULL_POOL};

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
