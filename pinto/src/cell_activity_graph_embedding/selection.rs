//! Gene selection for `pinto cage`: which `(gene, latent dim)` pairs are in.
//!
//! ```text
//! collapse cells -> super-cell counts Y[g,p]      (coarsening gives the groups)
//! one shared SVD -> e_pb, a D-dim cell side to condition on
//! block Gibbs    -> pip = P(z_gh = 1), and E[z*beta]
//! ```
//!
//! The selection model is a spike-and-slab Poisson over super-cell counts:
//!
//! ```text
//! Y[g,p] ~ Poisson( exp( <z_g (*) beta_g , e_p> + b_p ) )
//! z_gh ~ Bern(1 - pi0_h)        beta_gh ~ N(0, sigma0_h^2)
//! ```
//!
//! `p` indexes a SUPER-CELL, not an edge — the collapse already summed member
//! cells, so there are no endpoints here. Left/right only exists in the training
//! loss, which scores cell PAIRS.
//!
//! # The cell side is the whole reason this module exists
//!
//! Coarsening already defines the pseudobulks (`ml.all_cell_labels` IS the
//! cell->super-cell map) and collapsing counts against it is one call to
//! [`coarsen_cell_expression_dense`]. Neither is reimplemented here.
//!
//! What coarsening does NOT produce is `e_pb`: the super-cells expressed in the
//! same `D` dims the gene loadings live in. [`dim_block`] is a CONDITIONAL
//! sampler — it draws `z_gh` given a frozen `FrozenSide { e, b, h }` — so
//! without a `D`-dim cell side there is nothing to condition on. Coarsening
//! hands over labels and a `proj_dim` RANDOM projection; neither is that.
//!
//! # Sampled, not learned
//!
//! cage previously carried a learned variational spike-and-slab gate. It did not
//! select: measured on Visium mouse brain (32,245 genes x 2,695 spots), a dense
//! init left all 515,920 PIPs inside `[0.9635, 0.9890]` — one homogeneous blob —
//! and a prior-matched sparse init drove the pair term to underflow at exactly 0
//! by epoch 4, handing the objective to the ungated cell biases.
//!
//! A Gibbs sweep touches every gene every sweep (no minibatch blind spot) and
//! can switch a coordinate back ON from a likelihood ratio, which gradient
//! descent cannot do once a multiplier has reached zero.
//!
//! # Why super-cells rather than single spots
//!
//! A 55 um Visium spot detects ~5,900 of 32,245 genes, so a zero is mostly
//! non-detection. Selecting against single spots would mostly measure detection
//! depth. Super-cells aggregate tens of spots, so a zero means closer to absent.
//!
//! Every coarsening level contributes, offset into one global pseudobulk index
//! space so a gene's evidence spans all resolutions in a single [`dim_block`]
//! call. The basis is fit ONCE and shared: a per-level SVD would give per-level
//! directions, so `theta_g[h]` would mean a different thing at each level and
//! the shared `D` this design rests on would be fiction.
//!
//! # Re-sampled during training
//!
//! [`select_features`] is the COLD run, against `e_pb`. It also returns a
//! [`SelectionState`] so the training loop can re-estimate `pip` against the
//! embedding SGD is actually learning — see [`SelectionState::sample`] and
//! [`SelectionState::frozen_e_from_cells`].
//!
//! That refresh is NOT an EM E-step: training optimizes an edge NCE, this fits a
//! Poisson on counts, and two different objectives have no joint likelihood to
//! improve. `pip` is a DROP RATE for the training-time gate, and the refresh
//! only keeps it from going stale against an SVD basis the shipped model never
//! uses.

use crate::link_community::profiles::coarsen_cell_expression_dense;
use crate::util::common::*;

#[cfg(test)]
mod tests;

/// One coarsening level's pseudobulk, projected into the shared basis.
pub struct LevelPseudobulk {
    /// Cell-cluster labels this level was built from, `[n_cells]`.
    pub cell_labels: Vec<usize>,
    /// Super-cell counts `[n_genes × n_pb]`.
    pub counts: Mat,
    /// This level's SUPER-EDGES: distinct `(min, max)` pairs of DIFFERENT
    /// super-cells, i.e. the coarsened cell-graph adjacency.
    ///
    /// Louvain / METIS semantics, matching [`build_super_graph`]'s `si != sj`
    /// rule. An intra-group fine edge is NOT an edge of the coarse graph — it
    /// folds into the super-NODE, and here that has already happened: `counts`
    /// aggregates the member cells. Emitting it as a `(a, a)` edge as well
    /// would count that mass twice, once on the node and once on the loop.
    ///
    /// Because the levels nest, a super-edge maps to exactly one super-edge at
    /// the next coarser level (`(parent(a), parent(b))`) — unless both
    /// endpoints share a parent, in which case it stops being an edge there and
    /// becomes part of that parent's internal mass.
    pub super_edges: Vec<(usize, usize)>,
    /// Fine edge index → index into [`Self::super_edges`], or `None` when the
    /// fine edge is INTERNAL to one super-cell and so has no coarse edge.
    ///
    /// Needed to carry per-super-edge results back down to fine edges; an
    /// internal edge takes its endpoints' super-cell, not an edge label.
    pub fine_to_super: Vec<Option<usize>>,
    /// This level's super-cells projected into the SHARED basis, `[n_pb × D]`.
    pub e_pb: Mat,
    /// Offset of this level's super-cells in the global pseudobulk index space.
    pub pb_offset: usize,
}

impl LevelPseudobulk {
    #[must_use]
    pub fn n_pb(&self) -> usize {
        self.counts.ncols()
    }

    #[must_use]
    pub fn n_super_edges(&self) -> usize {
        self.super_edges.len()
    }

    /// Fine edges internal to one super-cell — folded into the node, not edges.
    #[must_use]
    pub fn n_internal_fine_edges(&self) -> usize {
        self.fine_to_super.iter().filter(|e| e.is_none()).count()
    }
}

/// The collapsed levels, projected through ONE shared basis.
pub struct Pseudobulks {
    pub levels: Vec<LevelPseudobulk>,
    // The shared gene→community basis it was all projected through is NOT kept:
    // it is scaffolding for `e_pb`, and the gene-side quantity anything
    // downstream wants is the sampler's `mean_beta`, not this.
    /// Total super-cells across all levels — the size of the global index space.
    pub total_pb: usize,
}

impl Pseudobulks {
    /// Per-pseudobulk log size factor, concatenated across levels in global
    /// index order. This is `dim_block`'s `FrozenSide::b`: the collapse emits
    /// counts, and a Poisson fit needs `log size_p` in the bias or it degrades
    /// to a quasi-Poisson whose per-dim slab variance collapses toward zero
    /// (see `geu::posterior`'s module header).
    #[must_use]
    pub fn log_size_factors(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.total_pb);
        for lvl in &self.levels {
            for p in 0..lvl.n_pb() {
                let size: f32 = lvl.counts.column(p).sum();
                out.push(size.max(1.0).ln());
            }
        }
        out
    }

    /// The frozen side `[total_pb × D]` flattened row-major, as
    /// `dim_block`'s `FrozenSide::e` wants it.
    #[must_use]
    pub fn frozen_e(&self, dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.total_pb * dim);
        for lvl in &self.levels {
            for p in 0..lvl.n_pb() {
                for h in 0..dim {
                    out.push(lvl.e_pb[(p, h)]);
                }
            }
        }
        out
    }
}

pub struct PseudobulkArgs<'a> {
    pub data: &'a SparseIoVec,
    /// `ml.all_cell_labels`, coarsest → finest.
    pub all_cell_labels: &'a [Vec<usize>],
    /// The fine cell graph, coarsened per level by `build_super_graph`.
    pub graph: &'a crate::util::knn_graph::KnnGraph,
    /// `[proj_dim × n_cells]` cell features. `build_super_graph` aggregates
    /// them per super-node in the same call — that aggregation is the
    /// super-cell embedding init an edge-level fit needs.
    pub cell_features: &'a Mat,
    /// Latent dimensionality `D` — the number of communities.
    pub embedding_dim: usize,
    pub block_size: Option<usize>,
}

/// Collapse every level, fit one shared basis, project each level onto it.
///
/// Levels with fewer super-cells than `D` are dropped: a basis of `D`
/// directions cannot be estimated from fewer points, and the coarsest levels of
/// the pyramid routinely fall below it (`compute_level_n_clusters` starts at
/// `min(16, n_clusters)`). Dropping is logged rather than silent.
pub fn build_pseudobulks(args: PseudobulkArgs<'_>) -> anyhow::Result<Pseudobulks> {
    anyhow::ensure!(
        !args.all_cell_labels.is_empty(),
        "no coarsening levels to pseudobulk"
    );
    let dim = args.embedding_dim;

    // Read the sparse matrix ONCE, at the finest level, then PROPAGATE upward:
    // a coarser level's counts are the column-sums of the finer level's, driven
    // by the parent map. This is `data-beans-alg`'s multi-level pattern
    // (`feature_coarsening_multilevel`), and it is only valid because
    // `graph_coarsen_multilevel` now guarantees levels nest — before that fix a
    // coarse super-cell was not a union of fine ones and this would have
    // silently produced wrong counts.
    //
    // Cost: was one full zarr decompression per level (4 passes on GBM), each
    // materializing every block's `[n_genes x n_pb]` partial at once. Now one
    // pass plus `G x P_fine` adds per coarser level.
    //
    // Nesting is transitive and only COARSE levels are ever dropped by the
    // `dim + 2` guard, so the finest always survives to be the source. If a
    // parent map ever comes back inconsistent the level falls back to a direct
    // read, so a regression costs speed rather than correctness.
    let finest_labels = args
        .all_cell_labels
        .last()
        .expect("non-empty, checked above");
    let n_pb_fine = finest_labels.iter().copied().max().map_or(0, |m| m + 1);
    // Block size is chosen HERE rather than inherited from `--block-size`:
    // `coarsen_cell_expression_dense` allocates one full dense
    // `[n_genes x n_pb]` partial PER JOB and collects them all before reducing,
    // so peak memory is `n_jobs x n_genes x n_pb x 4B`, not `n_threads x ...`.
    //
    // At the default block size of 100 that is ceil(3461/100) = 35 jobs x 205 MB
    // = 7.2 GB live at once on GBM Visium. One block per thread caps it at
    // `n_threads x 205 MB` while keeping the parallelism that matters — the
    // read itself, which is the expensive part.
    let n_cells = finest_labels.len();
    let read_block = n_cells.div_ceil(rayon::current_num_threads().max(1)).max(1);
    let finest_counts =
        coarsen_cell_expression_dense(args.data, finest_labels, n_pb_fine, Some(read_block))?;

    // `parent[p_fine] = p_coarse`, or `None` when this level's cut splits a
    // finest-level super-cell (which nesting forbids).
    let parent_map = |labels: &[usize]| -> Option<Vec<usize>> {
        let mut parent = vec![usize::MAX; n_pb_fine];
        for (cell, &pf) in finest_labels.iter().enumerate() {
            let pc = labels[cell];
            if parent[pf] == usize::MAX {
                parent[pf] = pc;
            } else if parent[pf] != pc {
                return None;
            }
        }
        Some(parent)
    };

    let mut collapsed: Vec<(Vec<usize>, Mat)> = Vec::new();
    for labels in args.all_cell_labels {
        let n_pb = labels.iter().copied().max().map_or(0, |m| m + 1);
        // Needs dim + 2, not dim: the basis fits `dim + 1` components and then
        // slices `columns(1, dim)`, and `rsvd` caps its rank at
        // `min(nrows, ncols)`. At `n_pb == dim` the slice would need `dim + 1`
        // columns from a `dim`-column matrix and nalgebra aborts the process.
        // At `n_pb == dim + 1` the trailing singular value is ~0 and
        // `nystrom_basis`'s `1/(s + 1e-8)` inflates that basis column by up to
        // 1e8, saturating every score inside `dim_block`.
        if n_pb < dim + 2 {
            info!(
                "pseudobulk: skipping a level with {n_pb} super-cells (need D + 2 = {})",
                dim + 2
            );
            continue;
        }
        let counts = if n_pb == n_pb_fine {
            finest_counts.clone()
        } else if let Some(parent) = parent_map(labels) {
            // Column-major sum: `counts` and `finest_counts` are both nalgebra
            // (column-major), and a super-cell is a COLUMN, so this walks both
            // operands sequentially.
            let mut c = Mat::zeros(finest_counts.nrows(), n_pb);
            for (pf, &pc) in parent.iter().enumerate() {
                let src = finest_counts.column(pf);
                let mut dst = c.column_mut(pc);
                dst += src;
            }
            c
        } else {
            warn!(
                "pseudobulk: level with {n_pb} super-cells does not nest in the finest level; \
                 re-reading it directly (this should not happen — see \
                 graph_coarsen_multilevel's nesting guarantee)"
            );
            coarsen_cell_expression_dense(args.data, labels, n_pb, args.block_size)?
        };
        collapsed.push((labels.clone(), counts));
    }
    anyhow::ensure!(
        !collapsed.is_empty(),
        "every coarsening level had fewer than {dim} super-cells; \
         lower --embedding-dim or raise --n-pseudobulk"
    );

    // ONE basis, fit on the FINEST level (last, since levels run coarsest →
    // finest) and shared by all. See the module header for why per-level bases
    // would be wrong.
    let finest = &collapsed
        .last()
        .expect("non-empty after the ensure above")
        .1;
    // Row-CENTRE before the SVD. Without it the leading direction is library
    // size: every gene loads positively on it, it carries no community
    // structure, and it crowds out a dimension. Centring each gene by its own
    // mean across pseudobulks makes cosine on the result equal Pearson on the
    // log-rates — the same reason `dict_merge.rs:47-53` centres before its
    // cosine merge. `scale_columns` alone does NOT fix this: measured on Visium
    // mouse brain it still left σ₁/σ₂ = 6.2.
    // Gene means come from the FINEST level and are reused for every level's
    // projection. Centring each level by its OWN means would put each level in
    // a different affine frame while they share one basis and one pooled
    // pseudobulk index space.
    let finest_log = log1p_dense(finest);
    let gene_means = row_means(&finest_log);
    let training = row_center_with(&finest_log, &gene_means).scale_columns();
    // Fit D+1 components and DROP the first. Measured on Visium mouse brain,
    // component 0 correlates with log library size at r = -0.99 — it is
    // sequencing depth, not community structure, and leaving it in would spend
    // a full dimension on it. `dim_block` would then be asked which genes load
    // on library size, and every gene does; the result is a uniformly-high PIP
    // column that looks like a working selector and is not.
    //
    // Row-centring alone does NOT fix this and is not meant to: it removes
    // per-gene abundance, but depth is a COLUMN effect. It is still worth
    // keeping — it took σ₁/σ₂ from 6.2 to 4.0 and lifted the tail 78% — so both
    // corrections are applied.
    let (u_dk, s_k, _) = training.rsvd(dim + 1)?;
    let basis_full = nystrom_basis(&u_dk, &s_k); // [n_genes × D+1]
    let basis = basis_full.columns(1, dim).into_owned(); // drop the depth axis
                                                         // The spectrum is the health check: a sharp drop to ~0 means the effective
                                                         // rank is below D and the trailing communities are noise directions that
                                                         // nothing can meaningfully load on.
    let sv: Vec<String> = s_k.iter().skip(1).map(|v| format!("{v:.3}")).collect();
    info!(
        "pseudobulk basis: {} genes × {} dims, fit on the finest of {} levels; \
         singular values (depth axis dropped) = [{}]",
        basis.nrows(),
        basis.ncols(),
        collapsed.len(),
        sv.join(", ")
    );

    let mut levels = Vec::with_capacity(collapsed.len());
    let mut pb_offset = 0usize;
    for (cell_labels, counts) in collapsed {
        // Coarsen the cell-graph ADJACENCY with the SHARED implementation.
        // `build_super_graph` is the repo's Louvain/METIS collapse: `si != sj`,
        // deduped, with intra-group edges reported as `None` because they fold
        // into the super-NODE — whose mass `counts` already carries.
        //
        // NOT `collapse_pairs` / `build_super_edges`: those keep `(a, a)`
        // because the link-community path needs every fine edge to map
        // somewhere for label transfer. Different job, and using it here
        // double-counted internal mass as a self-loop edge.
        let (super_graph, _, fine_to_super) = crate::util::graph_coarsen::build_super_graph(
            &cell_labels,
            counts.ncols(),
            args.graph,
            args.cell_features,
        );
        let super_edges = super_graph.edges;
        // Project under the SAME transform the basis was fit with: centre by
        // the finest level's gene means, THEN standardize columns.
        //
        // Dropping `.scale_columns()` here is not cosmetic. The basis is the
        // Nyström map for standardized columns, so projecting unstandardized
        // ones yields `e_pb[p,:] = sig_p · V[p,:]` — every row rescaled by its
        // own spread of log counts, a direct depth proxy. Measured on GBM
        // Visium before the fix: corr(row norm, log library size) = 0.452,
        // which the per-dim correlation diagnostic could not see because
        // scaling by a positive factor moves magnitudes, not signs.
        // `tr_mul` is `Aᵀ · B` without materializing `Aᵀ`. The explicit
        // `.transpose()` it replaces allocated a full `[n_pb x n_genes]` copy
        // (205 MB at the finest level) and read it across rows of a
        // column-major matrix.
        let e_pb = row_center_with(&log1p_dense(&counts), &gene_means)
            .scale_columns()
            .tr_mul(&basis);
        let n_pb = counts.ncols();
        levels.push(LevelPseudobulk {
            cell_labels,
            counts,
            super_edges,
            fine_to_super,
            e_pb,
            pb_offset,
        });
        pb_offset += n_pb;
    }

    // Is dim 0 still the depth axis? Correlate each dim's super-cell scores
    // against log library size at the finest level. A |r| near 1 on dim 0 means
    // row-centring did not remove it and the dim is spent on sequencing depth.
    if let Some(fine) = levels.last() {
        let sizes: Vec<f32> = (0..fine.n_pb())
            .map(|p| fine.counts.column(p).sum().max(1.0).ln())
            .collect();
        let corr = |x: &[f32], y: &[f32]| -> f32 {
            let n = x.len() as f32;
            let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for (a, b) in x.iter().zip(y) {
                sxy += (a - mx) * (b - my);
                sxx += (a - mx) * (a - mx);
                syy += (b - my) * (b - my);
            }
            if sxx <= 0.0 || syy <= 0.0 {
                0.0
            } else {
                sxy / (sxx * syy).sqrt()
            }
        };
        let rs: Vec<String> = (0..fine.e_pb.ncols())
            .map(|h| {
                let col: Vec<f32> = (0..fine.n_pb()).map(|p| fine.e_pb[(p, h)]).collect();
                format!("{:.2}", corr(&col, &sizes))
            })
            .collect();
        info!("  corr(dim, log library size) = [{}]", rs.join(", "));
        // Row NORM vs depth. The per-dim correlation above CANNOT catch a
        // projection that rescales each row by a positive depth-proxy: that
        // changes magnitudes, not signs. This one can.
        let norms: Vec<f32> = (0..fine.n_pb()).map(|p| fine.e_pb.row(p).norm()).collect();
        info!(
            "  corr(|e_pb| row norm, log library size) = {:.3}",
            corr(&norms, &sizes)
        );
    }
    for (i, lvl) in levels.iter().enumerate() {
        let e_norm = lvl.e_pb.norm() / (lvl.n_pb() as f32).sqrt();
        // Internal fine edges are folded into the super-NODE (their mass is
        // already in `counts`), so this is how much of the fine graph stops
        // being adjacency at this resolution — expected to grow toward the
        // coarse end as groups get bigger.
        let internal = lvl.n_internal_fine_edges();
        info!(
            "  level {i}: {} super-cells, {} super-edges, {} of {} fine edges internal \
             ({:.1}%), |e_pb|/sqrt(P) = {:.4}",
            lvl.n_pb(),
            lvl.n_super_edges(),
            internal,
            lvl.fine_to_super.len(),
            100.0 * internal as f64 / lvl.fine_to_super.len().max(1) as f64,
            e_norm
        );
    }
    info!(
        "pseudobulk: {} levels, {} super-cells total (global pb index space)",
        levels.len(),
        pb_offset
    );

    Ok(Pseudobulks {
        levels,
        total_pb: pb_offset,
    })
}

/// Per-row means as a `DVector`, kept in that form so [`row_center_with`] can
/// subtract it column-wise without a scalar inner loop.
///
/// nalgebra's `column_mean` IS the per-row mean (the mean over columns) and
/// walks column-major like the storage.
fn row_means(m: &Mat) -> nalgebra::DVector<f32> {
    m.column_mean()
}

/// Subtract GIVEN per-row means, out of place.
///
/// Taking the means as an argument is the whole point: every level must land in
/// the same affine frame, so they come from the finest level rather than from
/// each level's own data. `matrix_util`'s `centre_columns` centres by each
/// column's OWN mean, which is a different operation.
fn row_center_with(m: &Mat, means: &nalgebra::DVector<f32>) -> Mat {
    debug_assert_eq!(means.len(), m.nrows());
    let mut out = m.clone();
    out.column_iter_mut().for_each(|mut col| col -= means);
    out
}

/// Elementwise `log1p`, out of place. The collapse emits raw counts; the basis
/// is fit on the log scale so a handful of very high-count genes do not set the
/// leading directions on their own.
fn log1p_dense(m: &Mat) -> Mat {
    m.map(|v| v.max(0.0).ln_1p())
}

/////////////////////////////
// Gibbs gene selection //
/////////////////////////////

/// The sampler's output: the posterior inclusion table and the hyperparameters that
/// produced it.
pub struct Selection {
    /// `[n_genes × D]` row-major posterior inclusion probabilities.
    pub pip: Vec<f32>,
    /// `[n_genes × D]` row-major posterior mean EFFECTIVE loading `E[z·β]`.
    ///
    /// Shipped as `feature_posterior_mean.parquet`, NOT consumed by training:
    /// `e_feat` is randomly initialized and the selection enters as a gate
    /// instead.
    ///
    /// The two are alternatives, not complements. The sampler accumulates `β`
    /// only on sweeps where `z` was ON, then divides by ALL kept sweeps, so the
    /// shrinkage `pip` reports is already multiplied in here — gating this as
    /// well would apply the same selection twice.
    pub mean_beta: Vec<f32>,
    /// Per-dim slab variance `σ₀h²`.
    pub sigma2: Vec<f64>,
    /// Per-dim null rate `π₀h` = P(OFF).
    pub pi0: Vec<f64>,
    pub n_genes: usize,
    pub dim: usize,
}

pub struct SelectArgs {
    pub sweeps: usize,
    pub burnin: usize,
    pub seed: u64,
}

/// Sample `z_gh` against the super-cell Poisson, returning PIPs.
///
/// ```text
/// Y[g,p] ~ Poisson( exp( <z_g ⊙ β_g , e_p> + b_p ) )
/// z_gh ~ Bern(1 − π₀h)      β_gh ~ N(0, σ₀h²)
/// ```
///
/// This is `geu::posterior::dim_block` unchanged — the sampler `senna bge` uses
/// and the arm with a measured win there. Genes are anchors and the pseudobulk
/// side is frozen, so every gene is visited on every sweep. That is the
/// property the learned gate lacked and could not be given.
///
/// Unlike an NCE-based selection, this samples a REAL Poisson likelihood, so
/// the PIPs are a genuine posterior rather than a pseudo-posterior.
/// Everything `dim_block` needs that does NOT change between epochs.
///
/// Built once and re-sampled each epoch by [`SelectionState::sample`]. Hoisting
/// this out is what makes a per-epoch `pip` refresh affordable: `pos_per_gene`
/// is a scan of every nonzero gene-pb pair (15.7M on GBM Visium) and the
/// counts, size factors and partition are all fixed once the pseudobulks exist.
/// Only the frozen cell side `e` changes, because it is rebuilt from the
/// embedding SGD is currently learning.
pub struct SelectionState {
    pos_per_gene: Vec<Vec<(u32, f32)>>,
    partition: Vec<u32>,
    /// `log size_p` per pseudobulk, global index order — fixed with the counts.
    b_flat: Vec<f32>,
    /// Per level: `(cell_labels, n_pb, pb_offset)`, for folding a per-CELL
    /// embedding up into the per-pseudobulk frozen side.
    level_maps: Vec<(Vec<usize>, usize)>,
    pub total_pb: usize,
    pub n_genes: usize,
    pub dim: usize,
}

impl SelectionState {
    /// Fold a `[n_cells × D]` embedding into the `[total_pb × D]` frozen side,
    /// row-major, by averaging each super-cell's member cells.
    ///
    /// This is what makes the refresh MEAN anything: the initial `e_pb` came from
    /// an SVD of pseudobulk log-counts and is fixed forever, so re-sampling
    /// against it would redraw the same posterior. Folding the LIVE `e_cell` up
    /// instead is what lets the selection track what SGD has learned.
    #[must_use]
    pub fn frozen_e_from_cells(&self, e_cell: &Mat) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0f32; self.total_pb * d];
        let mut n = vec![0.0f32; self.total_pb];
        for (labels, off) in &self.level_maps {
            for (cell, &p) in labels.iter().enumerate() {
                let gp = off + p;
                n[gp] += 1.0;
                for h in 0..d {
                    out[gp * d + h] += e_cell[(cell, h)];
                }
            }
        }
        for gp in 0..self.total_pb {
            let inv = 1.0 / n[gp].max(1.0);
            for h in 0..d {
                out[gp * d + h] *= inv;
            }
        }
        out
    }

    /// Re-run the sampler against a supplied frozen side. `init_z` warm-starts
    /// the chain from the previous round's final state, which is why a refresh
    /// can afford far fewer sweeps than the cold initial run.
    pub fn sample(
        &self,
        e_flat: &[f32],
        args: &SelectArgs,
        init_z: Option<Vec<bool>>,
    ) -> (Selection, Vec<bool>) {
        use graph_embedding_util::posterior::{dim_block, DimBlockConfig, FrozenSide, NodeTerm};
        let side = FrozenSide {
            e: e_flat,
            b: &self.b_flat,
            h: self.dim,
        };
        let nodes: Vec<NodeTerm<'_>> = self
            .pos_per_gene
            .iter()
            .map(|pos| NodeTerm::new(pos, &self.partition, 1.0))
            .collect();
        let mut cfg = DimBlockConfig::new(args.sweeps, args.burnin, args.seed);
        if let Some(z) = init_z {
            cfg = cfg.with_init_z(z);
        }
        let out = dim_block(&nodes, &side, &cfg);
        let sel = Selection {
            pip: out.pip,
            mean_beta: out.mean_beta,
            sigma2: out.sigma2,
            pi0: out.pi0,
            n_genes: self.n_genes,
            dim: self.dim,
        };
        (sel, out.final_z)
    }
}

pub fn select_features(
    pb: &Pseudobulks,
    dim: usize,
    args: &SelectArgs,
) -> anyhow::Result<(Selection, SelectionState, Vec<bool>)> {
    use graph_embedding_util::posterior::{dim_block, DimBlockConfig, FrozenSide, NodeTerm};

    let n_genes = pb.levels[0].counts.nrows();
    let n_levels = pb.levels.len();
    // A cell's counts appear once PER LEVEL, so evidence is duplicated ~L times
    // relative to the prior. Scale it back out.
    let scale = 1.0 / n_levels as f32;

    // Frozen side: every level's projected super-cells, stacked in the global
    // index space that `pb_offset` defines.
    let e_flat = pb.frozen_e(dim);
    let b_flat = pb.log_size_factors();
    debug_assert_eq!(e_flat.len(), pb.total_pb * dim);
    debug_assert_eq!(b_flat.len(), pb.total_pb);
    let side = FrozenSide {
        e: &e_flat,
        b: &b_flat,
        h: dim,
    };

    // The normalizer runs over every pseudobulk in the global space — exact,
    // so `partition_scale` is 1.0.
    let partition: Vec<u32> = (0..pb.total_pb as u32).collect();

    // One `pos` list per gene: its counts at every level's super-cells, indexed
    // globally. Built once, borrowed by the NodeTerms.
    let mut pos_per_gene: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_genes];
    for lvl in &pb.levels {
        for p in 0..lvl.n_pb() {
            let gp = (lvl.pb_offset + p) as u32;
            let col = lvl.counts.column(p);
            for (g, &y) in col.iter().enumerate() {
                if y > 0.0 {
                    pos_per_gene[g].push((gp, y * scale));
                }
            }
        }
    }
    let nnz: usize = pos_per_gene.iter().map(Vec::len).sum();
    info!(
        "selection: {} genes × {} pseudobulks ({} levels), {} nonzero gene-pb pairs, \
         counts scaled by 1/{} for level duplication",
        n_genes, pb.total_pb, n_levels, nnz, n_levels
    );

    let nodes: Vec<NodeTerm<'_>> = pos_per_gene
        .iter()
        .map(|pos| NodeTerm::new(pos, &partition, 1.0))
        .collect();

    let cfg = DimBlockConfig::new(args.sweeps, args.burnin, args.seed);
    let out = dim_block(&nodes, &side, &cfg);

    let state = SelectionState {
        pos_per_gene,
        partition,
        b_flat,
        level_maps: pb
            .levels
            .iter()
            .map(|l| (l.cell_labels.clone(), l.pb_offset))
            .collect(),
        total_pb: pb.total_pb,
        n_genes,
        dim,
    };
    let sel = Selection {
        pip: out.pip,
        mean_beta: out.mean_beta,
        sigma2: out.sigma2,
        pi0: out.pi0,
        n_genes,
        dim,
    };
    Ok((sel, state, out.final_z))
}

impl Selection {
    /// `pip` as an `[n_genes × D]` matrix for parquet output.
    #[must_use]
    pub fn pip_matrix(&self) -> Mat {
        Mat::from_row_slice(self.n_genes, self.dim, &self.pip)
    }

    /// `mean_beta` as an `[n_genes × D]` matrix, for parquet output.
    #[must_use]
    pub fn mean_beta_matrix(&self) -> Mat {
        Mat::from_row_slice(self.n_genes, self.dim, &self.mean_beta)
    }

    /// Log a health summary. A degenerate PIP table — everything piled into one
    /// narrow band — is exactly how the learned gate failed, and it is invisible
    /// from the loss.
    pub fn log_summary(&self) {
        let n = self.pip.len() as f32;
        let mean = self.pip.iter().sum::<f32>() / n;
        let (lo, hi) = self
            .pip
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let frac_hi = self.pip.iter().filter(|&&v| v > 0.5).count() as f32 / n;
        let frac_lo = self.pip.iter().filter(|&&v| v < 0.1).count() as f32 / n;
        info!(
            "selection: PIP mean = {mean:.4}, range = [{lo:.4}, {hi:.4}], \
             frac(>0.5) = {frac_hi:.4}, frac(<0.1) = {frac_lo:.4}"
        );
        let s2: Vec<String> = self.sigma2.iter().map(|v| format!("{v:.3}")).collect();
        let p0: Vec<String> = self.pi0.iter().map(|v| format!("{v:.3}")).collect();
        info!("selection: sigma0^2 per dim = [{}]", s2.join(", "));
        info!("selection: pi0 (P off) per dim = [{}]", p0.join(", "));
    }
}
