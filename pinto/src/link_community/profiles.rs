//! Link profile construction, coarsening, and projection refinement.
//!
//! Builds projected link profiles from sparse expression data and KNN edges,
//! coarsens them by cell-level cluster labels, and refines the projection
//! basis via community centroids.

use crate::gene_network::graph::GenePairGraph;
use crate::link_community::model::LinkProfileStore;
use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use data_beans_alg::cell_pairs::collapse_pairs;
use matrix_param::io::ParamIo;
use matrix_util::utils::generate_minibatch_intervals;
use nalgebra_sparse::csc::CscMatrix;
use rayon::prelude::*;

/// Compute per-ROW total counts from sparse data.
///
/// Returns a vector of length `n_rows` with the sum of all entries per row.
/// A caller that means genes folds this through [`GeneAxis::pool_totals`]: on
/// a splice-channelized matrix a row is one track of a gene, not a gene.
pub fn compute_row_totals(
    data: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f64>> {
    let n_genes = data.num_rows();
    let n_cells = data.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);

    // Accumulate with `try_fold` + `try_reduce`, NOT `map().collect()`.
    //
    // `collect` materializes one accumulator PER JOB and holds them all until
    // the sum afterwards, so peak memory scales with the job count. The job count is
    // set by `default_block_size`, which derives the block from the FEATURE count
    // to bound read work, and is blind to how much each job allocates. At 18k
    // genes it hits the 100-cell floor, so half a million cells becomes ~5000
    // jobs, each carrying a full dense accumulator. Folding bounds the live
    // accumulators by the worker count instead, which is ~80x fewer here, and the
    // partials were being summed immediately anyway.
    let totals = jobs
        .par_iter()
        .try_fold(
            || vec![0.0f64; n_genes],
            |mut acc, &(lb, ub)| -> anyhow::Result<Vec<f64>> {
                let x = data.read_columns_csc(lb..ub)?;
                for col in 0..x.ncols() {
                    let s = x.col(col);
                    for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                        acc[row] += f64::from(val);
                    }
                }
                Ok(acc)
            },
        )
        .try_reduce(
            || vec![0.0f64; n_genes],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                Ok(a)
            },
        )?;
    Ok(totals)
}

/// Zero out rows of a basis matrix for features below a count threshold.
///
/// `totals` is indexed like the basis rows. When the caller has a gene axis it
/// passes per-gene totals spread back over the rows, so a gene is kept or
/// dropped as a unit rather than losing whichever track happens to fall short.
///
/// Returns the number of basis rows that were kept (not zeroed).
pub fn filter_basis_by_gene_count(basis: &mut Mat, gene_totals: &[f64], min_count: f32) -> usize {
    let mut n_kept = 0usize;
    for (g, &total) in gene_totals.iter().enumerate().take(basis.nrows()) {
        if total < min_count as f64 {
            basis.row_mut(g).fill(0.0);
        } else {
            n_kept += 1;
        }
    }
    n_kept
}

/// Collect unique cells from a chunk of edges, read their sparse columns,
/// and build an index map from global cell index to local column index.
pub(crate) fn read_unique_cells_for_edges(
    data: &SparseIoVec,
    chunk_edges: &[(usize, usize)],
) -> anyhow::Result<(CscMatrix<f32>, HashMap<usize, usize>)> {
    let mut unique_set: HashSet<usize> = Default::default();
    for &(i, j) in chunk_edges {
        unique_set.insert(i);
        unique_set.insert(j);
    }
    let mut unique_cells: Vec<usize> = unique_set.into_iter().collect();
    unique_cells.sort_unstable();

    let x_dn = data.read_columns_csc(unique_cells.iter().copied())?;

    let cell_to_col: HashMap<usize, usize> = unique_cells
        .iter()
        .enumerate()
        .map(|(col, &cell)| (cell, col))
        .collect();

    Ok((x_dn, cell_to_col))
}

/// Build projection profiles for a specific subset of edges, chunked and
/// parallelised across rayon (mirrors the deleted `build_edge_profiles`
/// pattern that v0.2.0 used for fine-edge profile construction).
///
/// Each chunk runs its own `read_unique_cells_for_edges` + per-edge
/// `basis^T · (x_i + x_j)` projection, then chunks are concatenated in
/// order. I/O is the dominant cost so chunk size is sized by
/// `generate_minibatch_intervals` against the gene-axis dimension.
///
/// * `data` - Sparse expression data [n_genes × n_cells]
/// * `edge_indices` - Subset of edge indices to process
/// * `all_edges` - Full edge list from KNN graph
/// * `basis` - Projection basis [n_genes × proj_dim]
/// * `block_size` - Edges per parallel chunk (None ⇒ adaptive default)
pub fn build_projection_profiles_for_edges(
    data: &SparseIoVec,
    edge_indices: &[usize],
    all_edges: &[(usize, usize)],
    basis: &Mat,
    block_size: Option<usize>,
) -> anyhow::Result<LinkProfileStore> {
    let n_edges = edge_indices.len();
    let m = basis.ncols(); // proj_dim
    let basis_t = basis.transpose(); // shared read-only across chunks

    // A zero-width basis would make the chunk width below zero, and
    // `par_chunks_mut` panics on that rather than yielding nothing.
    if m == 0 || n_edges == 0 {
        return Ok(LinkProfileStore::new(Vec::new(), n_edges, m));
    }

    // Extract this subset's edges once.
    let edges: Vec<(usize, usize)> = edge_indices.iter().map(|&e| all_edges[e]).collect();

    let jobs = generate_minibatch_intervals(n_edges, data.num_rows(), block_size);

    // Uniform edge count per job, except the last; `par_chunks_mut` splits the
    // output the same way, so the zip below pairs each job with exactly the
    // slice holding its own edges.
    let edges_per_job = jobs.first().map_or(1, |&(lb, ub)| (ub - lb).max(1));
    let prog_bar = new_progress_bar(jobs.len() as u64).with_message("edge blocks");

    // Each job writes a disjoint, known slice of the output, so it writes there
    // DIRECTLY rather than building its own buffer for a later copy. Collecting
    // the per-job buffers first meant the whole `n_edges x m` profile existed
    // twice at once, and that term is the largest allocation in the command.
    let mut profiles_em = vec![0.0f32; n_edges * m];
    profiles_em
        .par_chunks_mut(edges_per_job * m)
        .zip(jobs.par_iter())
        .progress_with(prog_bar.clone())
        .try_for_each(|(out_em, &(edge_begin, edge_end))| -> anyhow::Result<()> {
            let job_edges = &edges[edge_begin..edge_end];
            let (x_gc, col_of_cell) = read_unique_cells_for_edges(data, job_edges)?;
            let n_genes = x_gc.nrows();
            // The pooled endpoint profile `x_u + x_v`, reused across edges.
            let mut pooled_g = DVec::zeros(n_genes);

            for (job_edge, &(cell_u, cell_v)) in job_edges.iter().enumerate() {
                pooled_g.fill(0.0);
                for read_col in [col_of_cell[&cell_u], col_of_cell[&cell_v]] {
                    let cell_counts = x_gc.col(read_col);
                    for (&gene, &count) in cell_counts
                        .row_indices()
                        .iter()
                        .zip(cell_counts.values().iter())
                    {
                        pooled_g[gene] += count;
                    }
                }

                let projected_m = &basis_t * &pooled_g;
                let profile_offset = job_edge * m;
                for (dim, &value) in projected_m.iter().enumerate() {
                    out_em[profile_offset + dim] = value.max(0.0);
                }
            }
            Ok(())
        })?;
    prog_bar.finish_and_clear();

    Ok(LinkProfileStore::new(profiles_em, n_edges, m))
}

/// Coarsen fine-cell raw expression to pb-samples.
///
/// Returns an `[n_genes × n_pb_samples]` dense matrix whose column `c`
/// holds `Σ_{i: cell_labels[i] == c} x_fine[:, i]` — i.e. the total
/// gene counts pooled across every fine cell assigned to pb-sample `c`.
///
/// The return buffer is dense (not sparse) because the pb-sample expression
/// is dense by construction: any gene expressed in any fine cell within a
/// cluster contributes to that cluster's column.
pub fn coarsen_cell_expression_dense(
    data: &SparseIoVec,
    cell_labels: &[usize],
    n_pb_samples: usize,
) -> anyhow::Result<Mat> {
    let n_genes = data.num_rows();
    // Checked, not `debug_assert`ed. The loop below is driven by `cell_labels`
    // rather than by the column count, so a short label vector would silently
    // drop the unlabelled tail in a release build instead of failing. The old
    // cell-blocked code got that check for free from slice indexing.
    anyhow::ensure!(
        cell_labels.len() == data.num_columns(),
        "coarsening needs one pb-sample label per cell: got {} labels for {} cells",
        cell_labels.len(),
        data.num_columns()
    );

    // Nothing to coarsen into, and `par_chunks_mut` panics on a zero-width
    // chunk, so leave before either can bite.
    if n_genes == 0 || n_pb_samples == 0 {
        return Ok(Mat::zeros(n_genes, n_pb_samples));
    }

    // One job per CHUNK OF PB SAMPLES, not per slab of cells.
    //
    // Blocking by cells is what made this expensive: any cell in a slab can
    // belong to any pb sample, so every job had to carry the full dense
    // `[n_genes x n_pb_samples]` accumulator, and holding one per job put a
    // half-million-cell run past 200 GB by the third cascade level. Keyed by pb
    // sample, a job owns only the columns it writes, so the accumulator no
    // longer grows as the cascade coarsens.
    //
    // A chunk rather than a single sample only matters at the FINE levels. The
    // job target is capped at `n_pb_samples`, so once the cascade has coarsened
    // to fewer samples than that cap the chunk is one sample anyway and the
    // multiplier below does nothing. Where it does bite — many samples, few
    // workers — it buys work-stealing granularity, because pb samples differ
    // widely in cell count and a one-job-per-worker split would let the largest
    // sample set the level's wall time. It also makes each job's column gather
    // larger, though not contiguous: a job concatenates several samples' cell
    // lists, so it reads ascending runs, not one ascending sweep.
    //
    // Each cell is still read exactly once in total; only the order changes.
    let n_workers = rayon::current_num_threads().max(1);
    let target_jobs = (n_workers * 4).min(n_pb_samples.max(1));
    let pb_per_job = n_pb_samples.div_ceil(target_jobs.max(1)).max(1);

    // Cells bucketed by pb sample, CSR-style: `cells_by_pb[starts[p]..starts[p+1]]`
    // lists every fine cell assigned to sample `p`, ascending. Two flat vectors
    // rather than a `Vec<Vec<_>>`, so bucketing is a counting sort over one
    // allocation instead of one growing allocation per sample, and a job takes
    // its cells as a slice rather than rebuilding them.
    //
    // An out-of-range label is an error rather than a skip: dropping the cell
    // would leave a pseudobulk quietly missing counts, and every caller derives
    // `n_pb_samples` as `max(label) + 1`, so it cannot happen without a bug
    // upstream. The old code caught this by panicking on the matrix index.
    let mut starts = vec![0usize; n_pb_samples + 1];
    for (cell, &pb) in cell_labels.iter().enumerate() {
        anyhow::ensure!(
            pb < n_pb_samples,
            "cell {cell} carries pb-sample label {pb}, but only {n_pb_samples} samples exist"
        );
        starts[pb + 1] += 1;
    }
    for p in 0..n_pb_samples {
        starts[p + 1] += starts[p];
    }
    let mut cells_by_pb = vec![0usize; cell_labels.len()];
    let mut cursor = starts.clone();
    for (cell, &pb) in cell_labels.iter().enumerate() {
        cells_by_pb[cursor[pb]] = cell;
        cursor[pb] += 1;
    }

    let mut super_expr_gp = Mat::zeros(n_genes, n_pb_samples);
    super_expr_gp
        .as_mut_slice()
        .par_chunks_mut(n_genes * pb_per_job)
        .enumerate()
        .try_for_each(|(job, out_gp)| -> anyhow::Result<()> {
            let pb_begin = job * pb_per_job;
            let pb_end = ((job + 1) * pb_per_job).min(n_pb_samples);

            // This job's cells are already contiguous in `cells_by_pb`, so they
            // are a slice, and the read columns arrive in that same order.
            let read_begin = starts[pb_begin];
            let cells_to_read = &cells_by_pb[read_begin..starts[pb_end]];
            if cells_to_read.is_empty() {
                return Ok(());
            }

            let x_gc = data.read_columns_csc(cells_to_read.iter().copied())?;
            // Walk the samples this job owns and consume their runs in step, so
            // the pb id never has to be stored per column.
            for pb_local in 0..(pb_end - pb_begin) {
                let col_offset = pb_local * n_genes;
                let run = (starts[pb_begin + pb_local] - read_begin)
                    ..(starts[pb_begin + pb_local + 1] - read_begin);
                for read_col in run {
                    let cell_counts = x_gc.col(read_col);
                    for (&gene, &count) in cell_counts
                        .row_indices()
                        .iter()
                        .zip(cell_counts.values().iter())
                    {
                        out_gp[col_offset + gene] += count;
                    }
                }
            }
            Ok(())
        })?;
    Ok(super_expr_gp)
}

/// Build projection profiles for super-edges from pre-coarsened pb-sample
/// expression: `y_e = basis^T · (x_super[:, a] + x_super[:, b])`, with
/// negative entries clamped to 0.
///
/// Used inside the V-cycle cascade when the profile mode is `Projection`.
/// The alternative is to (incorrectly) read fine-cell columns at indices
/// equal to cluster labels — which gave arbitrary fine cells as
/// "pb-samples". This function replaces that path.
pub fn build_super_edge_projection_profiles(
    super_expr: &Mat,
    super_edges: &[(usize, usize)],
    edge_indices: &[usize],
    basis: &Mat,
) -> LinkProfileStore {
    debug_assert_eq!(super_expr.nrows(), basis.nrows());
    let n_edges = edge_indices.len();
    let m = basis.ncols();
    let basis_t = basis.transpose();

    let mut profiles = vec![0.0f32; n_edges * m];
    let mut temp_g = DVec::zeros(super_expr.nrows());

    for (e_idx, &ei) in edge_indices.iter().enumerate() {
        let (a, b) = super_edges[ei];
        temp_g.fill(0.0);
        temp_g += super_expr.column(a);
        temp_g += super_expr.column(b);

        let proj = &basis_t * &temp_g;
        let base = e_idx * m;
        for (d, &v) in proj.iter().enumerate() {
            profiles[base + d] = v.max(0.0);
        }
    }

    LinkProfileStore::new(profiles, n_edges, m)
}

/// Map fine edges to canonical super-edges defined by cell cluster labels.
///
/// Each edge (i, j) is mapped to (min(label[i], label[j]), max(...)).
/// Returns (super_edges list, fine_to_super mapping).
pub fn build_super_edges(
    edges: &[(usize, usize)],
    cell_labels: &[usize],
) -> (Vec<(usize, usize)>, Vec<usize>) {
    let (super_set, fine_to_super) = collapse_pairs(edges, cell_labels);
    (super_set.pairs, fine_to_super)
}

/// Transfer super-link community assignments back to fine edges.
///
/// Each fine edge inherits the community of its corresponding super-edge.
pub fn transfer_labels(fine_to_super: &[usize], super_membership: &[usize]) -> Vec<usize> {
    fine_to_super
        .iter()
        .map(|&se| super_membership[se])
        .collect()
}

/// Extract cell-level soft membership from link community assignments.
///
/// For each cell i, membership[i][k] = (# edges of i assigned to k) / (# edges of i).
/// Returns [n_cells × k] matrix.
pub fn compute_node_membership(
    edges: &[(usize, usize)],
    membership: &[usize],
    n_cells: usize,
    k: usize,
) -> Mat {
    let mut counts = Mat::zeros(n_cells, k);
    let mut degrees = vec![0usize; n_cells];

    for (e, &(i, j)) in edges.iter().enumerate() {
        let c = membership[e];
        counts[(i, c)] += 1.0;
        counts[(j, c)] += 1.0;
        degrees[i] += 1;
        degrees[j] += 1;
    }

    // Normalize each row
    for (i, &deg) in degrees.iter().enumerate().take(n_cells) {
        if deg > 0 {
            let scale = 1.0 / deg as f32;
            counts.row_mut(i).scale_mut(scale);
        }
    }

    counts
}

/// Dominant community per cell: the argmax of each propensity row, as one
/// `f32` column ready to sit beside the propensity in a parquet. Ties go
/// to the lowest index, matching `CommunityStrata`; an all-zero row maps
/// to 0. Every `propensity.parquet` writer derives its `cluster` column
/// here, so the three subcommands cannot drift apart on tie handling.
pub fn dominant_cluster_rows(propensity: &Mat) -> Vec<f32> {
    (0..propensity.nrows())
        .map(|i| {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (k, &v) in propensity.row(i).iter().enumerate() {
                if v > best_v {
                    best = k;
                    best_v = v;
                }
            }
            best as f32
        })
        .collect()
}

/// Row-wise Shannon entropy in nats: H(i) = -Σ_k p[i,k] · ln p[i,k].
///
/// Treats `0 · ln 0 = 0`. Rows that sum to ~0 (zero-degree vertices, or
/// rows that never received any edge mass) are returned as `NaN` so
/// downstream consumers can filter on `.is_finite()`. Rows are *not*
/// renormalized — pass in a true probability matrix.
pub fn shannon_entropy_rows(propensity: &Mat) -> DVec {
    let n = propensity.nrows();
    let mut out = DVec::zeros(n);
    for i in 0..n {
        let row = propensity.row(i);
        let mut s = 0.0f32;
        let mut h = 0.0f32;
        for &p in row.iter() {
            s += p;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        out[i] = if s > 0.0 { h } else { f32::NAN };
    }
    out
}

/// Compute community-specific gene expression statistics via Poisson-Gamma.
///
/// Given cell propensity [N × K] and sparse expression data [G × N],
/// computes weighted gene sums `X @ propensity^T` and fits a Poisson-Gamma
/// to get posterior gene expression rates per community. Before calibration the
/// sufficient statistic is reweighted row-wise by NB Fisher-info weights
/// `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`, matching the weighting used during
/// DC-Poisson clustering so clustering and reporting stay consistent.
///
/// Writes `{out_prefix}.gene_community.parquet` (genes × K). When
/// `gene_weights` is `Some`, those precomputed NB Fisher-info weights are
/// applied to the per-(gene, community) sufficient statistic; otherwise they
/// are recomputed from the data (extra full-data scan).
pub fn compute_gene_community_stat(
    cell_propensity: &Mat,
    data_vec: &SparseIoVec,
    gene_weights: Option<&[f32]>,
    axis: Option<&GeneAxis>,
    block_size: Option<usize>,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let param =
        fit_gene_community_param(cell_propensity, data_vec, gene_weights, axis, block_size)?;
    let row_names = data_vec.row_names()?;
    let names = axis.map_or(&row_names[..], GeneAxis::gene_names);
    write_gene_community_param(&param, names, out_prefix)
}

/// Fit the Poisson-Gamma posterior over gene × community without writing to disk.
///
/// Returns the calibrated `GammaMatrix` so callers can reuse the posterior
/// (e.g. to compute pairwise community similarity for cosine merging) without
/// re-reading the parquet output. The sufficient statistic is row-scaled by
/// NB Fisher-info weights `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`, matching
/// `compute_gene_community_stat`.
pub fn fit_gene_community_param(
    cell_propensity: &Mat,
    data_vec: &SparseIoVec,
    gene_weights: Option<&[f32]>,
    axis: Option<&GeneAxis>,
    block_size: Option<usize>,
) -> anyhow::Result<matrix_param::dmatrix_gamma::GammaMatrix> {
    use matrix_param::dmatrix_gamma::GammaMatrix;
    use matrix_param::traits::TwoStatParam;

    let n_rows = data_vec.num_rows();
    // Folding after the accumulation is exact, not an approximation: the
    // statistic is linear in the counts and the propensity is per cell, so
    // summing a gene's rows before or after the multiply gives the same matrix.
    let n_genes = axis.map_or(n_rows, GeneAxis::n_genes);
    let n_cells = data_vec.num_columns();
    let k = cell_propensity.ncols();

    info!("Computing gene-community statistics...");
    let prop_kn = cell_propensity.transpose();
    let jobs = generate_minibatch_intervals(n_cells, n_rows, block_size);

    let prog_bar = new_progress_bar(jobs.len() as u64).with_message("gene-community blocks");
    // Folded, not collected: see `compute_row_totals`. Each accumulator here is
    // `[n_rows x k]`, so the same job-count blowup applies.
    let (mut sum_gk, n_k_sum) = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .try_fold(
            || (Mat::zeros(n_rows, k), DVec::zeros(k)),
            |(mut acc_g, mut acc_k), &(lb, ub)| -> anyhow::Result<(Mat, DVec)> {
                let x_gn = data_vec.read_columns_csc(lb..ub)?;
                let block_len = ub - lb;
                let mut p_kn_block = Mat::zeros(k, block_len);
                for i in 0..block_len {
                    p_kn_block.column_mut(i).copy_from(&prop_kn.column(lb + i));
                }
                acc_k += p_kn_block.column_sum();
                acc_g += x_gn * p_kn_block.transpose();
                Ok((acc_g, acc_k))
            },
        )
        .try_reduce(
            || (Mat::zeros(n_rows, k), DVec::zeros(k)),
            |(mut ag, mut ak), (bg, bk)| {
                ag += bg;
                ak += bk;
                Ok((ag, ak))
            },
        )?;
    prog_bar.finish_and_clear();
    let n_1k = n_k_sum.transpose();
    if let Some(folded) = axis.and_then(|a| a.pool_rows_opt(&sum_gk)) {
        sum_gk = folded;
    }

    let owned_w;
    let w: &[f32] = match gene_weights {
        Some(w) => w,
        None => {
            info!("Computing NB Fisher-info weights for gene-community stats");
            owned_w = compute_nb_fisher_weights(data_vec, block_size)?;
            &owned_w
        }
    };
    anyhow::ensure!(
        w.len() == sum_gk.nrows(),
        "gene-community weights are on the wrong axis: {} weights for {} rows. \
         Folding the statistic to genes means the weights must be per gene too.",
        w.len(),
        sum_gk.nrows()
    );
    apply_gene_weights(&mut sum_gk, w);

    let mut gamma_param = GammaMatrix::new((n_genes, k), 1.0, 1.0);
    let denom_gk = DVec::from_element(n_genes, 1.0) * &n_1k;
    gamma_param.update_stat(&sum_gk, &denom_gk);
    gamma_param.calibrate();

    Ok(gamma_param)
}

/// Write a fitted gene-community posterior to `<out_prefix>.gene_community.parquet`
/// in melted (gene, community, mean, sd, log_mean, log_sd) form.
pub fn write_gene_community_param(
    param: &matrix_param::dmatrix_gamma::GammaMatrix,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_param::traits::Inference;
    let k = param.posterior_mean().ncols();
    let community_names: Vec<Box<str>> = (0..k).map(|i| format!("C{i}").into_boxed_str()).collect();
    param.to_melted_parquet(
        &(out_prefix.to_string() + ".gene_community.parquet"),
        (Some(gene_names), Some("gene")),
        (Some(&community_names), Some("community")),
    )?;
    Ok(())
}

/// How the edge latent gets cut into link communities.
///
/// K-means fixes the community count up front; Leiden derives it from the
/// resolution, so a run that has fewer (or more) distinct interaction regimes
/// than the latent width is not forced into `K` of them. Both consume the same
/// `[K_latent × N_pairs]` projection, so every caller can offer either.
#[derive(Debug, Clone, Copy)]
pub enum EdgeClustering {
    /// Lloyd's algorithm on the pairs, `k` fixed. `None` falls back to the
    /// latent width, read off the matrix rather than passed in beside it.
    Kmeans {
        n_clusters: Option<usize>,
        max_iter: usize,
    },
    /// Leiden over a cosine kNN graph on the pairs. `target` steers the
    /// resolution toward a community count when the caller has one in mind;
    /// `None` lets `resolution` alone decide.
    Leiden {
        knn: usize,
        resolution: f64,
        target: Option<usize>,
        seed: u64,
    },
}

impl EdgeClustering {
    /// Cut `pair_latent_nk` (`[N_pairs × K_latent]`, pairs as ROWS) into one community
    /// label per pair. Every pinto subcommand that turns a pair latent into
    /// link communities goes through here, so `cage`, `dsvd` and `prop` cannot
    /// disagree about what a community is.
    ///
    /// Pairs as rows because that is the orientation everything else already
    /// has: it is how the latent is written to parquet, how `prop` reads it
    /// back, and what Leiden wants. Only k-means needs the transpose, and it is
    /// no longer the default.
    pub fn cluster(&self, pair_latent_nk: &Mat) -> anyhow::Result<Vec<usize>> {
        Ok(match *self {
            Self::Kmeans {
                n_clusters,
                max_iter,
            } => {
                let num_clusters = n_clusters.unwrap_or(pair_latent_nk.ncols());
                info!("K-means clustering edges (k={num_clusters})...");
                pair_latent_nk.transpose().kmeans_columns(KmeansArgs {
                    num_clusters,
                    max_iter,
                })
            }
            Self::Leiden {
                knn,
                resolution,
                target,
                seed,
            } => {
                info!(
                    "Leiden clustering edges (knn={knn}, resolution={resolution}, target={target:?})..."
                );
                // Cosine, because the pair latent is a direction (it was
                // L2-normalized before it got here).
                matrix_util::clustering::leiden_clustering(
                    pair_latent_nk,
                    knn,
                    resolution,
                    target,
                    Some(seed),
                    true,
                )?
            }
        })
    }
}

/// How many communities the labels actually realize — what Leiden discovered,
/// or what k-means gave after leaving a cluster empty. Callers report THIS in
/// their manifest, not the count they asked for.
pub fn realized_communities(edge_membership: &[usize], n_edges: usize) -> anyhow::Result<usize> {
    let n = edge_membership.iter().copied().max().map_or(0, |m| m + 1);
    anyhow::ensure!(n >= 1, "edge clustering produced no communities");
    info!("{} link communities over {} edges", n, n_edges);
    Ok(n)
}

/// Config for `compute_propensity_and_gene_community_stat`.
pub struct PropensityReportConfig<'a> {
    pub clustering: EdgeClustering,
    pub block_size: Option<usize>,
    /// The GENE unit axis for the gene-community table, when the caller has
    /// one. `None` keeps the table on the matrix's own rows, which is what a
    /// caller wants when a row is already the unit it reports.
    pub gene_axis: Option<&'a GeneAxis>,
    /// Per-edge provenance, parallel to `edges`, when the pair graph was
    /// augmented. `None` omits the column, keeping an unaugmented run
    /// byte-identical.
    pub edge_kind: Option<&'a [i32]>,
}

/// What the propensity pass hands back to its caller, beyond the files
/// it writes itself: enough to build derived readouts (see
/// [`propensity_weighted_cell_embedding`]) without re-clustering.
pub struct PropensityOutputs {
    /// Realized number of edge communities.
    pub n_clusters: usize,
    /// `[n_cells x K]` per-cell propensity over edge communities (also
    /// written to `.propensity.parquet`).
    pub cell_propensity: Mat,
    /// Per-edge community label, parallel to `edges`.
    pub edge_membership: Vec<usize>,
}

/// A per-cell embedding as a READOUT of the propensity structure: each
/// cell is its propensity-weighted average of link-community centroids
/// in the pair-latent space, `e_cell[c] = Σ_k prop[c,k] · μ_k` with
/// `μ_k` the mean latent of community k's edges. Same width as the
/// latent (and therefore the gene dictionary it was projected against).
/// Nothing here is trained; callers decide whether and where to ship it.
#[must_use]
pub fn propensity_weighted_cell_embedding(pair_latent_nk: &Mat, out: &PropensityOutputs) -> Mat {
    let k = out.n_clusters;
    let d = pair_latent_nk.ncols();
    let mut centroids = Mat::zeros(k, d);
    // usize counts: an f32 accumulator stops incrementing at 2^24 edges
    // in one community and would silently inflate that centroid.
    let mut counts = vec![0usize; k];
    for (e, &kc) in out.edge_membership.iter().enumerate() {
        counts[kc] += 1;
        let mut row = centroids.row_mut(kc);
        row += pair_latent_nk.row(e);
    }
    for (kc, &cnt) in counts.iter().enumerate() {
        centroids.row_mut(kc).scale_mut(1.0 / cnt.max(1) as f32);
    }
    &out.cell_propensity * &centroids
}

/// Compute propensity and gene-community statistics from latent pair projections.
///
/// 1. Cluster `pair_latent_kn` (K_latent × N_pairs) → edge cluster labels
/// 2. Propensity: soft cell membership from edge clusters [N_cells × K_clusters]
/// 3. Gene-community stat: Poisson-Gamma gene expression rates per community [G × K_clusters]
///
/// Writes `{out_prefix}.propensity.parquet`, `{out_prefix}.link_community.parquet`
/// and `{out_prefix}.gene_community.parquet`. Returns [`PropensityOutputs`]; its
/// `n_clusters` is the count actually realized — what Leiden discovers, and what
/// k-means gives when it leaves a cluster empty. Callers should report THAT in
/// their manifest rather than the count they asked for.
pub fn compute_propensity_and_gene_community_stat(
    pair_latent_nk: &Mat,
    edges: &[(usize, usize)],
    data_vec: &SparseIoVec,
    n_cells: usize,
    cfg: &PropensityReportConfig,
    out_prefix: &str,
) -> anyhow::Result<PropensityOutputs> {
    let PropensityReportConfig {
        clustering,
        block_size,
        gene_axis,
        edge_kind,
    } = *cfg;

    // 1. Cluster the latent edge vectors
    let edge_membership = clustering.cluster(pair_latent_nk)?;

    // Everything downstream — propensity width, the gene × community stat, the
    // manifest — keys on the realized count.
    let n_clusters = realized_communities(&edge_membership, edges.len())?;

    // 2. Propensity [N_cells × K]
    info!("Computing cell propensity...");
    let cell_propensity = compute_node_membership(edges, &edge_membership, n_cells, n_clusters);

    let cell_names = data_vec.column_names()?;
    crate::link_community::outputs::write_propensity_matrix(
        out_prefix,
        &cell_propensity,
        &cell_names,
    )?;

    // Per-edge community labels, in the ONE edge-table schema pinto reads
    // (`left_cell` / `right_cell` / `community`). This used to be a local
    // writer emitting `{prefix}.edge_cluster.parquet` with the label column
    // named `cluster` — a file `dsvd`'s own manifest never pointed at and
    // `plot::load::read_link_community` could not parse.
    crate::link_community::outputs::write_link_communities(
        &(out_prefix.to_string() + ".link_community.parquet"),
        edges,
        &edge_membership,
        &cell_names,
        edge_kind,
    )?;

    // 3. Gene-community stat
    compute_gene_community_stat(
        &cell_propensity,
        data_vec,
        None,
        gene_axis,
        block_size,
        out_prefix,
    )?;

    Ok(PropensityOutputs {
        n_clusters,
        cell_propensity,
        edge_membership,
    })
}

/// Gene-network-derived module-pair basis for per-cell-edge features.
///
/// Constructed once after gene-module resolution: walks the gene-gene edge
/// list, buckets each edge by its endpoints' module labels, and keeps the
/// canonical `(a ≤ b)` pairs with positive weight. Each kept pair gets a
/// contiguous index `0..n_pairs` and a precomputed null factor
/// `deg(a)·deg(b)/(2W)²` used in the per-edge residual.
/// Neighbor entry in `ModulePairBasis::pair_adj`.
#[derive(Copy, Clone, Debug)]
pub struct PairAdjEntry {
    /// Neighbor module index.
    pub b: u32,
    /// Contiguous pair index into profile columns.
    pub pair_idx: u32,
    /// Modularity null factor `deg(a)·deg(b) / (2W)²`.
    pub null_ab: f32,
}

pub struct ModulePairBasis {
    pub n_modules: usize,
    pub module_of_gene: Vec<Option<usize>>,
    /// `pair_adj[a]` is the sorted list of neighbor modules that form a
    /// canonical pair `(min(a,b), max(a,b))`. Stored under BOTH endpoints so
    /// the per-edge intersection walk can start from either side; a
    /// canonical-order guard (`a ≤ b`) suppresses double-visits.
    pub pair_adj: Vec<Vec<PairAdjEntry>>,
    pub n_pairs: usize,
}

impl ModulePairBasis {
    /// Build the basis from the gene network + per-gene module labels.
    ///
    /// Genes with `module_of_gene[g] == None` contribute nothing. Gene-gene
    /// edges with both endpoints in some module accumulate to `B[a,b]`; the
    /// resulting module degrees seed the modularity null.
    pub fn build(graph: &GenePairGraph, module_of_gene: Vec<Option<usize>>) -> Self {
        let n_modules = module_of_gene
            .iter()
            .filter_map(|m| *m)
            .max()
            .map_or(0, |m| m + 1);

        // Canonical module-pair weights via sorted (a, b).
        let mut pair_weights: HashMap<(u32, u32), f64> = Default::default();
        let mut deg = vec![0.0f64; n_modules];
        for &(u, v) in &graph.feature_edges {
            let (Some(mu), Some(mv)) = (module_of_gene[u], module_of_gene[v]) else {
                continue;
            };
            let (a, b) = if mu <= mv {
                (mu as u32, mv as u32)
            } else {
                (mv as u32, mu as u32)
            };
            *pair_weights.entry((a, b)).or_insert(0.0) += 1.0;
            // Each undirected gene-gene edge contributes 1 to each endpoint's module degree.
            deg[mu] += 1.0;
            deg[mv] += 1.0;
        }
        let two_w: f64 = deg.iter().sum();
        let denom = two_w * two_w;

        // Assign contiguous pair indices in a deterministic order.
        let mut kept: Vec<((u32, u32), f64)> =
            pair_weights.into_iter().filter(|&(_, w)| w > 0.0).collect();
        kept.sort_by_key(|&((a, b), _)| (a, b));

        let mut pair_adj: Vec<Vec<PairAdjEntry>> = vec![Vec::new(); n_modules];
        for (pair_idx, &((a, b), _w)) in kept.iter().enumerate() {
            let null_ab = if denom > 0.0 {
                (deg[a as usize] * deg[b as usize] / denom) as f32
            } else {
                0.0
            };
            let pair_idx = pair_idx as u32;
            pair_adj[a as usize].push(PairAdjEntry {
                b,
                pair_idx,
                null_ab,
            });
            if a != b {
                pair_adj[b as usize].push(PairAdjEntry {
                    b: a,
                    pair_idx,
                    null_ab,
                });
            }
        }
        for adj in pair_adj.iter_mut() {
            adj.sort_by_key(|e| e.b);
        }

        let n_pairs = kept.len();
        info!(
            "ModulePairBasis: {} modules, {} retained pairs, 2W={:.1}",
            n_modules, n_pairs, two_w
        );

        ModulePairBasis {
            n_modules,
            module_of_gene,
            pair_adj,
            n_pairs,
        }
    }
}

/// Pre-collapse per-cell gene expression into per-cell module expression.
///
/// Returns `(module_expr, cell_totals)` where:
///   - `module_expr` is `n_modules × n_cells` dense (column-major): each
///     column is `x_{c,m} = Σ_{g ∈ m} x_{c,g}` for cell `c`. Modules with
///     no surviving genes stay zero.
///   - `cell_totals[c] = Σ_m x_{c,m}` — the per-cell total used as the null
///     scale in the residual.
///
/// One streaming pass over the sparse expression matrix.
pub fn build_module_expression(
    data: &SparseIoVec,
    module_of_gene: &[Option<usize>],
    n_modules: usize,
    gene_weights: Option<&[f32]>,
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, Vec<f32>)> {
    let n_cells = data.num_columns();
    let n_genes = data.num_rows();
    debug_assert_eq!(module_of_gene.len(), n_genes);
    if let Some(w) = gene_weights {
        debug_assert_eq!(w.len(), n_genes);
    }

    // Dense column-major: rows = modules, columns = cells. Small compared
    // to the raw matrix (typical n_modules is 10² range).
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);
    let prog_bar = new_progress_bar(jobs.len() as u64).with_message("module-expression blocks");

    let partials: Vec<(usize, Mat, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Mat, Vec<f32>)> {
            let x = data.read_columns_csc(lb..ub)?;
            let block_len = ub - lb;
            let mut block_expr = Mat::zeros(n_modules, block_len);
            let mut block_totals = vec![0.0f32; block_len];
            for col in 0..block_len {
                let s = x.col(col);
                for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                    if let Some(m) = module_of_gene[row] {
                        let v = match gene_weights {
                            Some(w) => val * w[row],
                            None => val,
                        };
                        block_expr[(m, col)] += v;
                        block_totals[col] += v;
                    }
                }
            }
            Ok((lb, block_expr, block_totals))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    prog_bar.finish_and_clear();

    let mut module_expr = Mat::zeros(n_modules, n_cells);
    let mut cell_totals = vec![0.0f32; n_cells];
    for (lb, block_expr, block_totals) in partials {
        let block_len = block_expr.ncols();
        for col in 0..block_len {
            module_expr
                .column_mut(lb + col)
                .copy_from(&block_expr.column(col));
            cell_totals[lb + col] = block_totals[col];
        }
    }
    Ok((module_expr, cell_totals))
}

/// Aggregate fine-cell module expression to pb-sample module expression.
///
/// For each fine cell `c` with pb-sample label `cell_labels[c] = sc`:
///   `super_expr[m, sc] += module_expr[m, c]`
/// Also returns per-pb-sample totals.
pub fn coarsen_module_expression(
    module_expr: &Mat,
    cell_labels: &[usize],
    n_pb_samples: usize,
) -> (Mat, Vec<f32>) {
    let n_modules = module_expr.nrows();
    let n_cells = module_expr.ncols();
    debug_assert_eq!(cell_labels.len(), n_cells);

    let mut super_expr = Mat::zeros(n_modules, n_pb_samples);
    let mut super_totals = vec![0.0f32; n_pb_samples];
    for c in 0..n_cells {
        let sc = cell_labels[c];
        for m in 0..n_modules {
            let v = module_expr[(m, c)];
            if v != 0.0 {
                super_expr[(m, sc)] += v;
                super_totals[sc] += v;
            }
        }
    }
    (super_expr, super_totals)
}

/// Build sparse module-pair profiles for a subset of edges.
///
/// For each edge `e = (i, j) = all_edges[edge_indices[e_idx]]` and canonical
/// module pair `(a, b)` with `a ≤ b`, emits
///
///   y = max(0, x_{i,a}·x_{j,b} + x_{i,b}·x_{j,a}
///              − X_i·X_j · deg(a)·deg(b)/(2W)²)
///
/// (with `a == b` using `x_{i,a}·x_{j,a}` to avoid double-counting).
///
/// A pair `(a, b)` can only produce a positive residual when its smaller
/// endpoint `a` is active (non-zero module expression) in at least one of
/// the two cells — otherwise `x_{i,a} = x_{j,a} = 0` forces `y_obs = 0`.
/// So the outer loop merges `A_i ∪ A_j` (sorted active-module lists per
/// cell) and walks `pair_adj[a]` only for `a` in the union, skipping
/// non-canonical entries via the `a ≤ b` guard.
pub fn build_module_pair_profiles_for_edges(
    module_expr: &Mat,
    cell_totals: &[f32],
    all_edges: &[(usize, usize)],
    edge_indices: &[usize],
    basis: &ModulePairBasis,
) -> LinkProfileStore {
    let n_modules = module_expr.nrows();
    let n_cells = module_expr.ncols();
    debug_assert_eq!(basis.n_modules, n_modules);

    // Per-cell sorted list of active (non-zero) module indices. One upfront
    // pass replaces the per-edge O(n_modules) sweep.
    let active_per_cell: Vec<Vec<u32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let col = module_expr.column(c);
            (0..n_modules)
                .filter(|&m| col[m] != 0.0)
                .map(|m| m as u32)
                .collect()
        })
        .collect();

    let rows: Vec<Vec<(u32, f32)>> = edge_indices
        .par_iter()
        .map(|&e| {
            let (i, j) = all_edges[e];
            let mass = cell_totals[i] as f64 * cell_totals[j] as f64;
            let a_i = &active_per_cell[i];
            let a_j = &active_per_cell[j];
            let mut out: Vec<(u32, f32)> = Vec::new();

            // Sorted-merge walk of A_i ∪ A_j with dedup (equal indices
            // advance both cursors so each `a` is visited once).
            let (mut pi, mut pj) = (0usize, 0usize);
            while pi < a_i.len() || pj < a_j.len() {
                let a = match (a_i.get(pi), a_j.get(pj)) {
                    (Some(&ai), Some(&aj)) if ai < aj => {
                        pi += 1;
                        ai
                    }
                    (Some(&ai), Some(&aj)) if ai > aj => {
                        pj += 1;
                        aj
                    }
                    (Some(&ai), Some(_)) => {
                        pi += 1;
                        pj += 1;
                        ai
                    }
                    (Some(&ai), None) => {
                        pi += 1;
                        ai
                    }
                    (None, Some(&aj)) => {
                        pj += 1;
                        aj
                    }
                    (None, None) => break,
                };

                let au = a as usize;
                let xi_a = module_expr[(au, i)] as f64;
                let xj_a = module_expr[(au, j)] as f64;

                for &entry in &basis.pair_adj[au] {
                    // Canonical guard: visit each (a, b) with a ≤ b once,
                    // from the smaller endpoint. Pairs where b < a will be
                    // visited (if at all) when we reach `b` in the union.
                    if a > entry.b {
                        continue;
                    }
                    let bu = entry.b as usize;
                    let y_obs = if bu == au {
                        xi_a * xj_a
                    } else {
                        let xi_b = module_expr[(bu, i)] as f64;
                        let xj_b = module_expr[(bu, j)] as f64;
                        xi_a * xj_b + xi_b * xj_a
                    };
                    let y = (y_obs - mass * entry.null_ab as f64).max(0.0) as f32;
                    if y > 0.0 {
                        out.push((entry.pair_idx, y));
                    }
                }
            }

            out
        })
        .collect();

    LinkProfileStore::from_sparse_rows(rows, basis.n_pairs)
}
