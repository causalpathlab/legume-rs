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
//! select: measured on a spatial section of roughly 32k genes by 2.7k spots, a dense
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
use crate::util::gene_axis::GeneAxis;
use graph_embedding_util::posterior::pb_gibbs::block_seed;
use graph_embedding_util::posterior::{
    dim_block, dim_block_multi, scalar_diagnostics, worst_case, ChainDiag, DimBlockConfig,
    DimBlockResult, FrozenSide, HyperState, NodeTerm,
};

#[cfg(test)]
mod tests;

/// One coarsening level's pseudobulk, projected into the shared basis.
pub struct LevelPseudobulk {
    /// Cell-cluster labels this level was built from, `[n_cells]`.
    pub cell_labels: Vec<usize>,
    /// Super-cell counts on the MATRIX ROW axis, `[n_rows × n_pb]`.
    ///
    /// The row axis is kept because the two splice tracks are only separable
    /// here: a gene-pooled view is a projection of this and cannot be inverted.
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
    /// How a matrix row maps onto the gene axis.
    ///
    /// The collapsed counts stay on the ROW axis — that is the only place the
    /// two splice tracks are separable, and the delta block reads them apart —
    /// so this is NOT used to fold them. It is used for the gene-axis views the
    /// basis and `e_pb` need, which are built per level and dropped.
    pub gene_axis: &'a GeneAxis,
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
    // Cost: was one full zarr decompression per level (4 passes on a real run), each
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
    // No block-size argument, and no hand-rolled per-thread cap: the callee
    // now bounds its own accumulator structurally. See the rationale on
    // `coarsen_cell_expression_dense`.
    // The MATRIX ROW axis, deliberately: the two splice tracks are separable only
    // here, and Stage 1's `delta` block reads them apart. The gene-pooled fold
    // every other consumer wants is derived per level below.
    let mut finest_counts = Some(coarsen_cell_expression_dense(
        args.data,
        finest_labels,
        n_pb_fine,
    )?);

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
            // Levels run coarsest -> finest, so this is the LAST iteration and
            // `finest_counts` has no reader after it. Taking rather than cloning
            // keeps a second `[n_rows x n_pb]` copy — doubled by the row axis, so
            // hundreds of MB — off the peak.
            finest_counts
                .take()
                .expect("the finest level is reached once")
        } else if let Some(parent) = parent_map(labels) {
            // Column-major sum: `counts` and `finest_counts` are both nalgebra
            // (column-major), and a super-cell is a COLUMN, so this walks both
            // operands sequentially.
            // Still `Some`: levels run coarsest -> finest, so every propagating
            // level is visited before the finest one takes it.
            let fine = finest_counts
                .as_ref()
                .expect("coarser levels precede the finest");
            let mut c = Mat::zeros(fine.nrows(), n_pb);
            for (pf, &pc) in parent.iter().enumerate() {
                let src = fine.column(pf);
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
            coarsen_cell_expression_dense(args.data, labels, n_pb)?
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
    let finest_rows = &collapsed
        .last()
        .expect("non-empty after the ensure above")
        .1;
    // The basis, and the `e_pb` it produces, live on the GENE axis: `e_pb` is the
    // frozen side the gene-anchored blocks score against, so it has to be
    // commensurate with a per-gene loading, not with a per-row one.
    let finest_pooled = args.gene_axis.pool_rows_opt(finest_rows);
    let finest = finest_pooled.as_ref().unwrap_or(finest_rows);
    // Row-CENTRE before the SVD. Without it the leading direction is library
    // size: every gene loads positively on it, it carries no community
    // structure, and it crowds out a dimension. Centring each gene by its own
    // mean across pseudobulks makes cosine on the result equal Pearson on the
    // log-rates — the same reason `dict_merge.rs:47-53` centres before its
    // cosine merge. `scale_columns` alone does NOT fix this: measured on Visium
    // that section it still left σ₁/σ₂ = 6.2.
    // Gene means come from the FINEST level and are reused for every level's
    // projection. Centring each level by its OWN means would put each level in
    // a different affine frame while they share one basis and one pooled
    // pseudobulk index space.
    let finest_log = log1p_dense(finest);
    let gene_means = row_means(&finest_log);
    let training = row_center_with(&finest_log, &gene_means).scale_columns();
    // Fit D+1 components and DROP the first. Measured on that same section,
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
        // own spread of log counts, a direct depth proxy. Measured on a real
        // Visium before the fix: corr(row norm, log library size) = 0.452,
        // which the per-dim correlation diagnostic could not see because
        // scaling by a positive factor moves magnitudes, not signs.
        // `tr_mul` is `Aᵀ · B` without materializing `Aᵀ`. The explicit
        // `.transpose()` it replaces allocated a full `[n_pb x n_genes]` copy
        // (205 MB at the finest level) and read it across rows of a
        // column-major matrix.
        // The basis and `e_pb` live on the GENE axis (see the basis fit above), so
        // the fold happens here and is dropped: nothing downstream of `e_pb` wants
        // pooled counts, and keeping a copy per level would double the pyramid.
        let pooled_counts = args.gene_axis.pool_rows_opt(&counts);
        let e_pb = row_center_with(
            &log1p_dense(pooled_counts.as_ref().unwrap_or(&counts)),
            &gene_means,
        )
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
    /// Per-dim null rate `π₀h` = P(OFF). Under the IBP this is the FIXED ladder
    /// rather than an estimate — see [`Self::log_summary`].
    pub pi0: Vec<f64>,
    /// Mixing of the beta slab-variance chain, per dim, on both paths.
    pub sigma_diag: Vec<ChainDiag>,
    /// Retained outer sweeps behind [`Self::sigma_diag`].
    pub n_kept: usize,
    pub n_genes: usize,
    pub dim: usize,
    /// The nascent deviation, `Some` only on a splice-channelized input.
    pub delta: Option<DeltaSelection>,
}

/// The velocity deviation `δ_g`: what an UNSPLICED row loads on top of `β_g`.
///
/// Sign convention is `senna gem`'s — spliced is the base, `unspliced = β + δ` —
/// recorded as `delta_base: "spliced"` in the run manifest. `senna gem-encoder`
/// uses the OPPOSITE base, so the two are not comparable without reading that
/// field. See the plan's §1.2.
pub struct DeltaSelection {
    /// `[n_genes × D]` inclusion probabilities, NaN on an unidentified gene.
    pub pip: Vec<f32>,
    /// `[n_genes × D]` posterior mean `E[z·δ]`, NaN on an unidentified gene.
    pub mean: Vec<f32>,
    pub sigma2: Vec<f64>,
    pub pi0: Vec<f64>,
    /// How much evidence the nascent track actually carries — see
    /// [`DeltaStrength`]. Identifiability says a contrast EXISTS; this says
    /// whether there is enough of it to estimate anything.
    pub strength: DeltaStrength,
    /// Mixing of the delta hyperparameters over the OUTER sweeps, per dim.
    /// A stalled chain still produces a full table of plausible numbers, so this
    /// is reported rather than swallowed.
    pub sigma_diag: Vec<ChainDiag>,
    /// Per gene: counts on BOTH tracks. `δ` is identified only by the
    /// unspliced-minus-spliced contrast, so everything else is a prior draw and
    /// is written NaN rather than as an apparently-measured number.
    pub identified: Vec<bool>,
}

impl DeltaSelection {
    #[must_use]
    pub fn n_identified(&self) -> usize {
        self.identified.iter().filter(|&&x| x).count()
    }

    /// `mean` / `pip` as `[n_genes × D]` matrices, for parquet output.
    #[must_use]
    pub fn mean_matrix(&self, n_genes: usize, dim: usize) -> Mat {
        Mat::from_row_slice(n_genes, dim, &self.mean)
    }

    #[must_use]
    pub fn pip_matrix(&self, n_genes: usize, dim: usize) -> Mat {
        Mat::from_row_slice(n_genes, dim, &self.pip)
    }
}

/// Evidence on the nascent track, measured where the sampler sees it.
///
/// `delta_identified` is a STRUCTURAL test — at least one count on each track —
/// and it is necessary, not sufficient. Measured on a probe-based spatial sample
/// (2026-08-18), 83% of genes passed it while the median gene carried 19 unspliced
/// counts spread over ~140 pseudobulks: 0.14 counts per bin, which pins nothing.
/// A run that reported only the 83% would have looked like a green light.
#[derive(Debug, Clone, Copy)]
pub struct DeltaStrength {
    /// Median unspliced counts per identified gene, summed over the pseudobulks.
    pub median_counts: f32,
    /// Median number of pseudobulk bins where an identified gene's nascent track
    /// is nonzero, out of [`Self::total_pb`].
    pub median_pb_detected: f32,
    pub total_pb: usize,
}

impl DeltaStrength {
    /// Mean unspliced counts per pseudobulk bin at the median gene. Below ~1 the
    /// Poisson is being handed a near-empty likelihood.
    #[must_use]
    pub fn counts_per_pb(&self) -> f32 {
        self.median_counts / self.total_pb.max(1) as f32
    }
}

pub struct SelectArgs {
    pub sweeps: usize,
    pub burnin: usize,
    pub seed: u64,
    /// Allow `z_δ = 1` only where `z_β = 1`. The default, and gem's.
    ///
    /// `δ` is a DEVIATION from `β`, so "this gene moves along a dim its identity
    /// does not load" is a state the model should not visit. Vetoing it also
    /// breaks the symmetry that otherwise lets two independent spike-and-slabs
    /// split inclusion mass between `(z_β=1, z_δ=0)` and `(z_β=0, z_δ=1)` on a
    /// gene where only their sum is identified.
    pub nested_delta: bool,
}

/// State carried BETWEEN calls so an alternating chain resumes rather than
/// restarting. Cold-starting every refresh would redraw every `β` from its prior
/// and silently discard the warm start — see [`DimBlockConfig::init_z`].
#[derive(Clone, Default)]
pub struct ChainWarm {
    pub z_beta: Option<Vec<bool>>,
    pub z_delta: Option<Vec<bool>>,
    pub e_beta: Option<Vec<f32>>,
    pub e_delta: Option<Vec<f32>>,
    /// Each block's slab variance and half-Cauchy auxiliary. Carried for the same
    /// reason as `z`: the blocks run one sweep at a time, so these only form a
    /// chain if the driver hands them back.
    pub hyper_beta: Option<HyperState>,
    pub hyper_delta: Option<HyperState>,
}

/// Per-gene super-cell evidence, split by splice track.
///
/// `unspliced` is entirely empty on a non-channelized input, and `spliced` is
/// then the whole gene — which is why that case takes the single-block path
/// unchanged rather than a two-block one whose second block sees nothing.
struct TrackPos {
    spliced: Vec<Vec<(u32, f32)>>,
    unspliced: Vec<Vec<(u32, f32)>>,
}

/// Running sums over the OUTER sweeps, kept in one place so the two blocks'
/// accumulators cannot drift out of step. The inner blocks run ONE sweep each,
/// so their own averages are single draws and the posterior mean is formed here.
struct Acc {
    bp: Vec<f64>,
    bm: Vec<f64>,
    dp: Vec<f64>,
    dm: Vec<f64>,
    /// Per-OUTER-sweep hyper chains, one entry per (dim, sweep). Sums alone would
    /// give the posterior means, but not mixing: the inner blocks run a single
    /// sweep each, so their own `ChainDiag` reports ESS = 1 unconditionally and
    /// says nothing.
    bs2: Vec<Vec<f64>>,
    bpi: Vec<Vec<f64>>,
    ds2: Vec<Vec<f64>>,
    dpi: Vec<Vec<f64>>,
    n: usize,
}

impl Acc {
    fn new(n_genes: usize, h: usize) -> Self {
        Self {
            bp: vec![0.0; n_genes * h],
            bm: vec![0.0; n_genes * h],
            dp: vec![0.0; n_genes * h],
            dm: vec![0.0; n_genes * h],
            bs2: vec![Vec::new(); h],
            bpi: vec![Vec::new(); h],
            ds2: vec![Vec::new(); h],
            dpi: vec![Vec::new(); h],
            n: 0,
        }
    }

    fn add(&mut self, b: &DimBlockResult, d: &DimBlockResult) {
        let acc32 = |dst: &mut [f64], src: &[f32]| {
            for (a, v) in dst.iter_mut().zip(src) {
                *a += f64::from(*v);
            }
        };
        let push64 = |dst: &mut [Vec<f64>], src: &[f64]| {
            for (a, v) in dst.iter_mut().zip(src) {
                a.push(*v);
            }
        };
        acc32(&mut self.bp, &b.pip);
        acc32(&mut self.bm, &b.mean_beta);
        acc32(&mut self.dp, &d.pip);
        acc32(&mut self.dm, &d.mean_beta);
        push64(&mut self.bs2, &b.sigma2);
        push64(&mut self.bpi, &b.pi0);
        push64(&mut self.ds2, &d.sigma2);
        push64(&mut self.dpi, &d.pi0);
        self.n += 1;
    }
}

/// Everything `dim_block` needs that does NOT change between epochs.
///
/// Built once and re-sampled each epoch by [`SelectionState::sample`]. Hoisting
/// this out is what makes a per-epoch `pip` refresh affordable: the per-track
/// `pos` lists are a scan of every nonzero row-pb pair and the counts, size
/// factors and partition are all fixed once the pseudobulks exist. Only the
/// frozen cell side `e` changes, because it is rebuilt from the embedding SGD is
/// currently learning.
pub struct SelectionState {
    tracks: TrackPos,

    delta_identified: Vec<bool>,
    strength: DeltaStrength,
    partition: Vec<u32>,
    /// `log size_p` per pseudobulk, global index order — fixed with the counts.
    b_flat: Vec<f32>,
    /// Per level: `(cell_labels, pb_offset)`, for folding a per-CELL embedding up
    /// into the per-pseudobulk frozen side.
    level_maps: Vec<(Vec<usize>, usize)>,
    pub total_pb: usize,
    pub n_genes: usize,
    pub dim: usize,
}

impl SelectionState {
    /// Whether the alternation runs, derived rather than cached so it cannot
    /// disagree with `delta_identified`.
    ///
    /// NOT the same as "the input carried channels". A spliced-only channelized
    /// matrix is a supported input whose unspliced track is empty everywhere; on
    /// it the delta block would draw every gene from the prior and the beta block
    /// would carry a second empty `NodeTerm` per gene per sweep, paying double to
    /// produce a table that is NaN by construction.
    fn two_block(&self) -> bool {
        self.delta_identified.iter().any(|&x| x)
    }

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

    /// Re-run the sampler against a supplied frozen side. `warm` resumes the
    /// chain from the previous round's final state, which is why a refresh can
    /// afford far fewer sweeps than the cold initial run.
    pub fn sample(
        &self,
        e_flat: &[f32],
        args: &SelectArgs,
        warm: ChainWarm,
    ) -> (Selection, ChainWarm) {
        let side = FrozenSide {
            e: e_flat,
            b: &self.b_flat,
            h: self.dim,
        };
        if self.two_block() {
            self.sample_two_block(&side, args, warm)
        } else {
            self.sample_one_block(&side, args, warm)
        }
    }

    /// The single-block path: no channels, so there is no second track to
    /// contrast against and `δ` does not exist. Byte-for-byte what `cage` did
    /// before the splice work.
    fn sample_one_block(
        &self,
        side: &FrozenSide<'_>,
        args: &SelectArgs,
        warm: ChainWarm,
    ) -> (Selection, ChainWarm) {
        let nodes: Vec<NodeTerm<'_>> = self
            .tracks
            .spliced
            .iter()
            .map(|pos| NodeTerm::new(pos, &self.partition, 1.0))
            .collect();
        let mut cfg = DimBlockConfig::new(args.sweeps, args.burnin, args.seed);
        if let Some(z) = warm.z_beta {
            cfg = cfg.with_init_z(z);
        }
        let out = dim_block(&nodes, side, &cfg);
        let sel = Selection {
            pip: out.pip,
            mean_beta: out.mean_beta,
            sigma2: out.sigma2,
            pi0: out.pi0,
            sigma_diag: out.sigma_diag,
            n_kept: out.n_kept,
            n_genes: self.n_genes,
            dim: self.dim,
            delta: None,
        };
        (
            sel,
            ChainWarm {
                z_beta: Some(out.final_z),
                // The single-block path owns all its sweeps, so its hyperparameters
                // already chain inside `dim_block` and there is nothing to carry.
                ..Default::default()
            },
        )
    }

    /// The alternation: `β | δ` then `δ | β`, one sweep of each per outer sweep,
    /// both against the SAME frozen pseudobulk side.
    ///
    /// There is no `pb | β, δ` block. `senna gem`'s splice Gibbs has one because
    /// it SAMPLES the pseudobulk embedding; here the pseudobulk side is frozen by
    /// construction and refreshed from the live `e_cell`
    /// ([`Self::frozen_e_from_cells`]), so a pb block would produce a second
    /// object claiming to be the same thing.
    fn sample_two_block(
        &self,
        side: &FrozenSide<'_>,
        args: &SelectArgs,
        warm: ChainWarm,
    ) -> (Selection, ChainWarm) {
        let (h, n_genes) = (self.dim, self.n_genes);
        let mut e_beta = warm.e_beta.unwrap_or_else(|| vec![0f32; n_genes * h]);
        let mut e_delta = warm.e_delta.unwrap_or_else(|| vec![0f32; n_genes * h]);
        let (mut z_beta, mut z_delta) = (warm.z_beta, warm.z_delta);
        // Each block's slab variance and its half-Cauchy auxiliary carry across
        // the outer sweeps. Without this every entry restarts them at the prior
        // width, so the beta and delta draws would always use sigma = 1 and the
        // adaptive slab would never adapt — see `DimBlockConfig::init_hyper`.
        let (mut hyper_beta, mut hyper_delta) = (warm.hyper_beta, warm.hyper_delta);
        let mut acc = Acc::new(n_genes, h);
        let total = args.sweeps.max(1);

        for sweep in 0..total {
            // β | δ, over BOTH tracks. β appears in the unspliced rows too — they
            // score `⟨β + δ, e_p⟩` — so a spliced-only kernel would not be the
            // conditional `p(β | δ, ·)` and the two blocks would stop being
            // conditionals of one joint. A single `NodeTerm` carries one offset
            // for all of an anchor's edges, which is exactly what
            // `dim_block_multi` exists to get around.
            let beta_terms: Vec<Vec<NodeTerm<'_>>> = (0..n_genes)
                .map(|g| {
                    let spliced = NodeTerm::new(&self.tracks.spliced[g], &self.partition, 1.0);
                    let mut unspliced =
                        NodeTerm::new(&self.tracks.unspliced[g], &self.partition, 1.0);
                    unspliced.offset = Some(&e_delta[g * h..(g + 1) * h]);
                    vec![spliced, unspliced]
                })
                .collect();
            let mut bcfg = DimBlockConfig::new(1, 0, block_seed(args.seed, 0xB37A, sweep))
                .with_init_beta(e_beta.clone())
                .with_label("beta|delta")
                .quiet();
            if let Some(z) = z_beta.clone() {
                bcfg = bcfg.with_init_z(z);
            }
            if let Some(hs) = hyper_beta.take() {
                bcfg = bcfg.with_init_hyper(hs);
            }
            let b = dim_block_multi(&beta_terms, side, &bcfg);
            hyper_beta = Some(b.final_hyper.clone());
            drop(beta_terms);
            e_beta.copy_from_slice(&b.mean_beta);
            z_beta = Some(b.final_z.clone());

            // δ | β, on the unspliced rows, with β carried as the per-anchor
            // offset so the sampler explores a DEVIATION rather than an absolute
            // loading — refreshed from THIS sweep's β. Holding β at a one-time
            // snapshot would make the two conditionals describe no single joint.
            let delta_nodes: Vec<NodeTerm<'_>> = (0..n_genes)
                .map(|g| {
                    let mut n = NodeTerm::new(&self.tracks.unspliced[g], &self.partition, 1.0);
                    n.offset = Some(&e_beta[g * h..(g + 1) * h]);
                    n
                })
                .collect();
            let mut dcfg = DimBlockConfig::new(1, 0, block_seed(args.seed, 0xD317, sweep))
                .with_init_beta(e_delta.clone())
                .with_label("delta|beta")
                .quiet();
            if args.nested_delta {
                dcfg = dcfg.with_z_allowed(b.final_z.clone());
            }
            if let Some(z) = z_delta.clone() {
                dcfg = dcfg.with_init_z(z);
            }
            if let Some(hs) = hyper_delta.take() {
                dcfg = dcfg.with_init_hyper(hs);
            }
            let d = dim_block(&delta_nodes, side, &dcfg);
            hyper_delta = Some(d.final_hyper.clone());
            drop(delta_nodes);
            e_delta.copy_from_slice(&d.mean_beta);
            z_delta = Some(d.final_z.clone());

            if sweep >= args.burnin {
                acc.add(&b, &d);
            }
        }

        let n = acc.n.max(1) as f64;
        let mean32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| (x / n) as f32).collect() };
        let chain_mean = |c: &[Vec<f64>]| -> Vec<f64> {
            c.iter()
                .map(|s| s.iter().sum::<f64>() / s.len().max(1) as f64)
                .collect()
        };
        let diag = |c: &[Vec<f64>]| -> Vec<ChainDiag> {
            c.iter().map(|s| scalar_diagnostics(s)).collect()
        };

        let mut delta_pip = mean32(&acc.dp);
        let mut delta_mean = mean32(&acc.dm);
        // Unidentified genes are written NaN, never a number. With no unspliced
        // counts `δ` enters no likelihood term at all, so what came back is the
        // prior — shipping it as an estimate is the one failure mode this whole
        // report exists to prevent.
        for (g, &ok) in self.delta_identified.iter().enumerate() {
            if !ok {
                for x in &mut delta_pip[g * h..(g + 1) * h] {
                    *x = f32::NAN;
                }
                for x in &mut delta_mean[g * h..(g + 1) * h] {
                    *x = f32::NAN;
                }
            }
        }

        let sel = Selection {
            pip: mean32(&acc.bp),
            mean_beta: mean32(&acc.bm),
            sigma2: chain_mean(&acc.bs2),
            pi0: chain_mean(&acc.bpi),
            sigma_diag: diag(&acc.bs2),
            n_kept: acc.n,
            n_genes,
            dim: h,
            delta: Some(DeltaSelection {
                pip: delta_pip,
                mean: delta_mean,
                sigma2: chain_mean(&acc.ds2),
                pi0: chain_mean(&acc.dpi),
                sigma_diag: diag(&acc.ds2),
                strength: self.strength,
                identified: self.delta_identified.clone(),
            }),
        };
        (
            sel,
            ChainWarm {
                z_beta,
                z_delta,
                e_beta: Some(e_beta),
                e_delta: Some(e_delta),
                hyper_beta,
                hyper_delta,
            },
        )
    }
}

/// Sample `z_gh` against the super-cell Poisson, returning PIPs.
///
/// ```text
/// Y[g,p] ~ Poisson( exp( <z_g ⊙ β_g , e_p> + b_p ) )
/// z_gh ~ Bern(1 − π₀h)      β_gh ~ N(0, σ₀h²)
/// ```
///
/// This is `geu::posterior::dim_block` — the sampler `senna bge` uses and the arm
/// with a measured win there. Genes are anchors and the pseudobulk side is
/// frozen, so every gene is visited on every sweep. That is the property the
/// learned gate lacked and could not be given.
///
/// Unlike an NCE-based selection, this samples a REAL Poisson likelihood, so
/// the PIPs are a genuine posterior rather than a pseudo-posterior.
///
/// On a splice-channelized input it becomes the two-block alternation described
/// on [`SelectionState::sample_two_block`], and additionally returns `δ_g`.
pub fn select_features(
    pb: &Pseudobulks,
    axis: &GeneAxis,
    dim: usize,
    args: &SelectArgs,
) -> anyhow::Result<(Selection, SelectionState, ChainWarm)> {
    let n_genes = axis.n_genes();
    let n_levels = pb.levels.len();
    // The axis and the pseudobulks are separate arguments, so nothing structural
    // stops a caller pairing an axis with counts it was not resolved from. The
    // row loop below indexes the axis by pseudobulk row, so a mismatch is either
    // a raw slice panic or, worse, a silent mis-mapping. Both sibling consumers
    // guard this — `build_cell_activities` and `project_pairs` — so this one does
    // too rather than relying on the caller.
    anyhow::ensure!(
        pb.levels
            .first()
            .is_none_or(|l| l.counts.nrows() == axis.n_rows()),
        "selection: gene axis has {} rows, pseudobulks have {}",
        axis.n_rows(),
        pb.levels[0].counts.nrows()
    );
    // A cell's counts appear once PER LEVEL, so evidence is duplicated ~L times
    // relative to the prior. Scale it back out.
    let scale = 1.0 / n_levels as f32;

    // Frozen side: every level's projected super-cells, stacked in the global
    // index space that `pb_offset` defines.
    let e_flat = pb.frozen_e(dim);
    let b_flat = pb.log_size_factors();
    debug_assert_eq!(e_flat.len(), pb.total_pb * dim);
    debug_assert_eq!(b_flat.len(), pb.total_pb);

    // The normalizer runs over every pseudobulk in the global space — exact,
    // so `partition_scale` is 1.0.
    let partition: Vec<u32> = (0..pb.total_pb as u32).collect();

    // One `pos` list per (gene, track): its counts at every level's super-cells,
    // indexed globally. The counts are on the MATRIX ROW axis, which is the only
    // place the tracks are separable.
    let mut tracks = TrackPos {
        spliced: vec![Vec::new(); n_genes],
        unspliced: vec![Vec::new(); n_genes],
    };
    for lvl in &pb.levels {
        for p in 0..lvl.n_pb() {
            let gp = (lvl.pb_offset + p) as u32;
            let col = lvl.counts.column(p);
            for (r, &y) in col.iter().enumerate() {
                if y > 0.0 {
                    let dst = if axis.row_is_nascent(r) {
                        &mut tracks.unspliced
                    } else {
                        &mut tracks.spliced
                    };
                    dst[axis.gene_of_row(r)].push((gp, y * scale));
                }
            }
        }
    }

    // Identifiability from the SAME lists the blocks read, not from a separate
    // pass over the counts — so what is reported cannot disagree with what was
    // sampled.
    let delta_identified: Vec<bool> = tracks
        .spliced
        .iter()
        .zip(&tracks.unspliced)
        .map(|(s, u)| !s.is_empty() && !u.is_empty())
        .collect();

    let nnz_s: usize = tracks.spliced.iter().map(Vec::len).sum();
    let nnz_u: usize = tracks.unspliced.iter().map(Vec::len).sum();
    let n_identified = delta_identified.iter().filter(|&&x| x).count();

    // Strength, over the IDENTIFIED genes only — the ones whose delta the run
    // will actually report. The pos lists carry counts already scaled by 1/L for
    // level duplication, and every level partitions all the cells, so the scaled
    // sum over levels is the gene's true total.
    // The workspace's median, not a local one: pinto's took the upper middle at
    // even n where `matrix_util`'s averages the two, so `delta_median_counts` in
    // the manifest would not have been the same estimator every other "median"
    // the workspace reports.
    let median = |v: Vec<f32>| -> f32 { matrix_util::utils::median(&v) };
    let ident: Vec<usize> = (0..n_genes).filter(|&g| delta_identified[g]).collect();
    let strength = DeltaStrength {
        median_counts: median(
            ident
                .iter()
                .map(|&g| tracks.unspliced[g].iter().map(|&(_, y)| y).sum::<f32>())
                .collect(),
        ),
        median_pb_detected: median(
            ident
                .iter()
                .map(|&g| tracks.unspliced[g].len() as f32)
                .collect(),
        ),
        total_pb: pb.total_pb,
    };
    if axis.is_channelized() {
        let n_ok = n_identified;
        info!(
            "selection: {} genes × {} pseudobulks ({} levels), {} spliced / {} unspliced \
             nonzero gene-pb pairs, counts scaled by 1/{} for level duplication; \
             delta identified for {}/{} genes, {} gate",
            n_genes,
            pb.total_pb,
            n_levels,
            nnz_s,
            nnz_u,
            n_levels,
            n_ok,
            n_genes,
            if args.nested_delta {
                "nested z_delta in z_beta"
            } else {
                "independent"
            },
        );
    } else {
        info!(
            "selection: {} genes × {} pseudobulks ({} levels), {} nonzero gene-pb pairs, \
             counts scaled by 1/{} for level duplication",
            n_genes, pb.total_pb, n_levels, nnz_s, n_levels
        );
    }

    let state = SelectionState {
        tracks,
        delta_identified,
        strength,
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
    let (sel, warm) = state.sample(&e_flat, args, ChainWarm::default());
    Ok((sel, state, warm))
}

impl Selection {
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
        info!(
            "selection: pi0 (P off) per dim, the fixed IBP ladder, not an estimate = [{}]",
            p0.join(", ")
        );
        // The beta gate is the one whose PIPs are installed as the training mask
        // every refresh, so its chain is the one that must not go unreported.
        self.log_chain_health("beta", &self.sigma_diag, self.n_kept);
        self.log_delta_summary();
    }

    /// The Stage-1 go/no-go, reported from the run rather than left to be
    /// reconstructed afterwards: does `δ` carry mass, does its chain mix, and is
    /// it selecting DIFFERENT genes from `β`.
    fn log_delta_summary(&self) {
        let Some(d) = self.delta.as_ref() else {
            return;
        };
        let h = self.dim;
        let ok: Vec<usize> = (0..self.n_genes).filter(|&g| d.identified[g]).collect();
        if ok.is_empty() {
            warn!(
                "selection: no gene identifies delta, so the whole delta table is NaN. \
                 Nothing downstream of it is interpretable on this input."
            );
            return;
        }
        let vals: Vec<f32> = ok
            .iter()
            .flat_map(|&g| d.pip[g * h..(g + 1) * h].iter().copied())
            .collect();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let frac_hi = vals.iter().filter(|&&v| v > 0.5).count() as f32 / vals.len() as f32;
        info!(
            "selection/delta: {} of {} genes identified; PIP mean = {mean:.4}, \
             frac(>0.5) = {frac_hi:.4}",
            ok.len(),
            self.n_genes
        );
        // Identifiability is structural and says only that a contrast exists.
        // This says whether there is enough of it to estimate anything, and it is
        // the number that separates a usable input from one that merely passes.
        let st = d.strength;
        info!(
            "selection/delta: evidence on the nascent track — median identified gene has \
             {:.0} counts over {} pseudobulks ({:.2} per bin), detected in {:.0} of them",
            st.median_counts,
            st.total_pb,
            st.counts_per_pb(),
            st.median_pb_detected,
        );
        if st.counts_per_pb() < 1.0 {
            warn!(
                "selection/delta: the median identified gene carries {:.2} unspliced counts \
                 per pseudobulk. The contrast exists structurally but is too thin to \
                 estimate from — treat delta as undetermined regardless of what its PIPs \
                 say, and check whether the assay captures introns at all (a probe-based \
                 library does not).",
                st.counts_per_pb()
            );
        }

        // What is NOT reported here, and why: the obvious check is "are delta's
        // top genes different from beta's — if they coincide, delta is tracking
        // abundance". Measured on a matched pair of synthetic fixtures (one where
        // the nascent track is a binomial thinning of the mature one, so the true
        // delta is zero, and one where it carries an independent spatial program),
        // that overlap does not discriminate: 0.67 on the null against 0.77 on the
        // true positive under the nested gate, and 0.41 against 0.41 under the
        // independent one. Under nesting it CANNOT discriminate, because
        // `z_delta` is a subset of `z_beta` by construction, so delta's top genes
        // are drawn from beta's included set whatever the data says.
        //
        // The inclusion mass above is what separated the two: 0.074 on the null
        // against 0.343 on the true positive (nested), 0.159 against 0.446
        // (independent). Both arms separate; nesting separates BETTER, which is
        // the measured argument for it being the default.

        let s2: Vec<String> = d.sigma2.iter().map(|v| format!("{v:.3}")).collect();
        let p0: Vec<String> = d.pi0.iter().map(|v| format!("{v:.3}")).collect();
        info!("selection/delta: sigma0^2 per dim = [{}]", s2.join(", "));
        // Reported as what it is. Under the IBP (`stick_alpha` set, which is the
        // default) `pi0` is a FIXED ladder the sampler never resamples, so this
        // is the prior in effect, not an estimate — and diagnosing its "mixing"
        // would be diagnosing a constant, which always comes back perfect.
        info!(
            "selection/delta: pi0 (P off) per dim, the fixed IBP ladder, not an estimate = [{}]",
            p0.join(", ")
        );
        self.log_chain_health("delta", &d.sigma_diag, self.n_kept);
    }

    /// Mixing of one gate's slab-variance chain over the OUTER sweeps.
    ///
    /// Both gates get this. The sibling implementation reports both for a reason
    /// — one gate's healthy chain must not cover for the other's — and here it is
    /// `beta` whose PIPs become the training gate on every refresh, so leaving it
    /// undiagnosed would be the wrong half to skip.
    fn log_chain_health(&self, gate: &str, sigma_diag: &[ChainDiag], n_kept: usize) {
        // `split_rhat` needs 8 retained draws to segment at all and returns 1.0
        // below that, so at the default `--selection-sweeps 10` (burnin 5, five
        // retained) a printed "1.000" would mean "not computed", not "converged".
        // Saying so is the difference between a guard and a decoration.
        if n_kept < 8 {
            info!(
                "selection/{gate}: {n_kept} retained sweep(s) — too few for a split-Rhat, \
                 which needs 8 (40 for its full four-segment form). Convergence is UNKNOWN, \
                 not confirmed. Raise --selection-sweeps to at least 16 to get one."
            );
            return;
        }
        // geu's own reducer, so pinto and `senna gem` report "the worst chain" by
        // one definition. The hand-rolled fold this replaces started its R-hat at
        // 0.0 where `worst_case` floors at 1.0, so the same chains printed a
        // different worst R-hat in the two crates.
        let worst = worst_case(sigma_diag);
        let (worst_ess, worst_rhat) = (worst.min_ess, worst.rhat);
        info!(
            "selection/{gate}: slab-variance chain over {n_kept} retained sweeps — \
             worst ESS = {worst_ess:.1}, worst split-Rhat = {worst_rhat:.3}"
        );
        if worst_rhat > 1.1 {
            warn!(
                "selection/{gate}: split-Rhat {worst_rhat:.2} > 1.1 — the chain has not \
                 converged, so its PIPs are not a posterior. Raise --selection-sweeps \
                 before reading them."
            );
        }
    }
}
