//! Pseudobulk basis for `pinto cage`: super-cell counts per coarsening level,
//! and one shared SVD over them.
//!
//! ```text
//! collapse cells -> super-cell counts Y[g,p]      (coarsening gives the groups)
//! one shared SVD -> e_pb_svd, a D-dim super-cell side
//! ```
//!
//! Coarsening already defines the pseudobulks (`ml.all_cell_labels` IS the
//! cell->super-cell map) and collapsing counts against it is one call to
//! [`coarsen_cell_expression_dense`]. Neither is reimplemented here. What
//! coarsening does NOT produce is `e_pb_svd`: the super-cells expressed in the
//! same `D` dims the gene loadings live in. It hands over labels and a
//! `proj_dim` RANDOM projection; neither is that.
//!
//! The finest level's basis is the WARM START of the trained PB table, and it
//! is exported per cell as `pseudobulk_cells.parquet` so the basis can be
//! inspected as a UMAP and as a spatial map before anything is built on it.
//!
//! # Why super-cells rather than single spots
//!
//! A single spot detects only a small fraction of the gene axis, so a zero is
//! mostly non-detection. Super-cells aggregate tens of spots, so a zero means
//! closer to absent.
//!
//! Every coarsening level contributes. The basis is fit ONCE and shared: a
//! per-level SVD would give per-level directions, so a coordinate would mean a
//! different thing at each level and the shared `D` this design rests on would
//! be fiction.

use crate::link_community::profiles::coarsen_cell_expression_dense;
use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;

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
    /// aggregates the member cells.
    ///
    /// The super-edge lists themselves live in the TRAINING frame
    /// (`pb_frame::PbFrame`, finest level); the basis keeps only this
    /// internal-edge count for its resolution diagnostic.
    pub n_internal_fine_edges: usize,
    /// This level's super-cells projected into the SHARED basis, `[n_pb × D]`.
    pub e_pb_svd: Mat,
}

impl LevelPseudobulk {
    #[must_use]
    pub fn n_pb(&self) -> usize {
        self.counts.ncols()
    }
}

/// The collapsed levels, projected through ONE shared basis.
pub struct Pseudobulks {
    pub levels: Vec<LevelPseudobulk>,
    // The shared gene→community basis it was all projected through is NOT kept:
    // it is scaffolding for `e_pb_svd`.
}

pub struct PseudobulkArgs<'a> {
    pub data: &'a SparseIoVec,
    /// `ml.all_cell_labels`, coarsest → finest.
    pub all_cell_labels: &'a [Vec<usize>],
    /// The fine cell graph; read per level only to count internal fine
    /// edges for the resolution diagnostic (the super-edge fold itself
    /// lives in the training frame, `pb_frame`).
    pub graph: &'a crate::util::knn_graph::KnnGraph,
    /// Latent dimensionality `D` — the number of communities.
    pub embedding_dim: usize,
    /// How a matrix row maps onto the gene axis.
    ///
    /// The collapsed counts stay on the ROW axis — that is the only place the
    /// two splice tracks are separable, and the delta block reads them apart —
    /// so this is NOT used to fold them. It is used for the gene-axis views the
    /// basis and `e_pb_svd` need, which are built per level and dropped.
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
        // 1e8.
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
    // The basis, and the `e_pb_svd` it produces, live on the GENE axis: `e_pb_svd` is the
    // frozen side the gene-anchored blocks score against, so it has to be
    // commensurate with a per-gene loading, not with a per-row one.
    let finest_pooled = args.gene_axis.pool_rows_opt(finest_rows);
    let finest = finest_pooled.as_ref().unwrap_or(finest_rows);
    // Row-CENTRE before the SVD. Without it the leading direction is library
    // size: every gene loads positively on it, it carries no community
    // structure, and it crowds out a dimension. Centring each gene by its own
    // mean across pseudobulks makes cosine on the result equal Pearson on the
    // log-rates — the same reason `dict_merge.rs:47-53` centres before its
    // cosine merge. `scale_columns` alone does NOT fix this: measured, it
    // still leaves the first singular value several times the second.
    // Gene means come from the FINEST level and are reused for every level's
    // projection. Centring each level by its OWN means would put each level in
    // a different affine frame while they share one basis and one pooled
    // pseudobulk index space.
    let finest_log = log1p_dense(finest);
    let gene_means = row_means(&finest_log);
    let training = row_center_with(&finest_log, &gene_means).scale_columns();
    // Fit D+1 components and DROP the first. Measured, component 0 tracks log
    // library size almost perfectly — it is sequencing depth, not community
    // structure, and leaving it in would spend a full dimension on it.
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
    for (cell_labels, counts) in collapsed {
        // Internal-edge count for the resolution diagnostic: one label
        // compare per fine edge. The full super-edge fold used to happen
        // here too; it now lives ONLY in the training frame (`pb_frame`),
        // so there is a single source of super edges.
        let n_internal_fine_edges = args
            .graph
            .edges
            .iter()
            .filter(|&&(i, j)| cell_labels[i] == cell_labels[j])
            .count();
        // Project under the SAME transform the basis was fit with: centre by
        // the finest level's gene means, THEN standardize columns.
        //
        // Dropping `.scale_columns()` here is not cosmetic. The basis is the
        // Nyström map for standardized columns, so projecting unstandardized
        // ones yields `e_pb_svd[p,:] = sig_p · V[p,:]` — every row rescaled by its
        // own spread of log counts, a direct depth proxy. Measured before the fix: corr(row norm, log library size) = 0.452,
        // which the per-dim correlation diagnostic could not see because
        // scaling by a positive factor moves magnitudes, not signs.
        // `tr_mul` is `Aᵀ · B` without materializing `Aᵀ`. The explicit
        // `.transpose()` it replaces allocated a full `[n_pb x n_genes]` copy
        // (205 MB at the finest level) and read it across rows of a
        // column-major matrix.
        // The basis and `e_pb_svd` live on the GENE axis (see the basis fit above), so
        // the fold happens here and is dropped: nothing downstream of `e_pb_svd` wants
        // pooled counts, and keeping a copy per level would double the pyramid.
        let pooled_counts = args.gene_axis.pool_rows_opt(&counts);
        let e_pb_svd = row_center_with(
            &log1p_dense(pooled_counts.as_ref().unwrap_or(&counts)),
            &gene_means,
        )
        .scale_columns()
        .tr_mul(&basis);
        levels.push(LevelPseudobulk {
            cell_labels,
            counts,
            n_internal_fine_edges,
            e_pb_svd,
        });
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
        let rs: Vec<String> = (0..fine.e_pb_svd.ncols())
            .map(|h| {
                let col: Vec<f32> = (0..fine.n_pb()).map(|p| fine.e_pb_svd[(p, h)]).collect();
                format!("{:.2}", corr(&col, &sizes))
            })
            .collect();
        info!("  corr(dim, log library size) = [{}]", rs.join(", "));
        // Row NORM vs depth. The per-dim correlation above CANNOT catch a
        // projection that rescales each row by a positive depth-proxy: that
        // changes magnitudes, not signs. This one can.
        let norms: Vec<f32> = (0..fine.n_pb())
            .map(|p| fine.e_pb_svd.row(p).norm())
            .collect();
        info!(
            "  corr(|e_pb_svd| row norm, log library size) = {:.3}",
            corr(&norms, &sizes)
        );
    }
    for (i, lvl) in levels.iter().enumerate() {
        let e_norm = lvl.e_pb_svd.norm() / (lvl.n_pb() as f32).sqrt();
        // Internal fine edges are folded into the super-NODE (their mass is
        // already in `counts`), so this is how much of the fine graph stops
        // being adjacency at this resolution — expected to grow toward the
        // coarse end as groups get bigger.
        let internal = lvl.n_internal_fine_edges;
        let n_fine = args.graph.edges.len();
        info!(
            "  level {i}: {} super-cells, {} of {} fine edges internal \
             ({:.1}%), |e_pb_svd|/sqrt(P) = {:.4}",
            lvl.n_pb(),
            internal,
            n_fine,
            100.0 * internal as f64 / n_fine.max(1) as f64,
            e_norm
        );
    }
    let total_pb: usize = levels.iter().map(LevelPseudobulk::n_pb).sum();
    info!(
        "pseudobulk: {} levels, {} super-cells total",
        levels.len(),
        total_pb
    );

    Ok(Pseudobulks { levels })
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
