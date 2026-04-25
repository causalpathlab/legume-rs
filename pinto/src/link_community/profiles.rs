//! Link profile construction, coarsening, and projection refinement.
//!
//! Builds projected link profiles from sparse expression data and KNN edges,
//! coarsens them by cell-level cluster labels, and refines the projection
//! basis via community centroids.

use crate::gene_network::graph::GenePairGraph;
use crate::link_community::model::LinkProfileStore;
use crate::util::common::*;
use matrix_param::io::ParamIo;
use matrix_util::utils::generate_minibatch_intervals;
use nalgebra_sparse::csc::CscMatrix;
use rayon::prelude::*;

/// Compute per-gene total counts from sparse data.
///
/// Returns a vector of length `n_genes` with the sum of all entries per row.
pub fn compute_gene_totals(
    data: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f64>> {
    let n_genes = data.num_rows();
    let n_cells = data.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);

    let partials: Vec<Vec<f64>> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<f64>> {
            let x = data.read_columns_csc(lb..ub)?;
            let mut local = vec![0.0f64; n_genes];
            for col in 0..x.ncols() {
                let s = x.col(col);
                for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                    local[row] += val as f64;
                }
            }
            Ok(local)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut totals = vec![0.0f64; n_genes];
    for local in partials {
        for g in 0..n_genes {
            totals[g] += local[g];
        }
    }
    Ok(totals)
}

/// Zero out rows of a basis matrix for genes below a count threshold.
///
/// Returns the number of genes that were kept (not zeroed).
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

    // Extract this subset's edges once.
    let edges: Vec<(usize, usize)> = edge_indices.iter().map(|&e| all_edges[e]).collect();

    let jobs = generate_minibatch_intervals(n_edges, data.num_rows(), block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Building edges {bar:40} {pos}/{len} blocks ({eta})",
    );

    let partial_results: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let chunk_edges = &edges[lb..ub];
            let chunk_size = ub - lb;

            let (x_dn, cell_to_col) = read_unique_cells_for_edges(data, chunk_edges)?;

            let n_genes = x_dn.nrows();
            let mut chunk_profiles = vec![0.0f32; chunk_size * m];
            let mut temp_g = DVec::zeros(n_genes);

            for (e_idx, &(ci, cj)) in chunk_edges.iter().enumerate() {
                let col_i = cell_to_col[&ci];
                let col_j = cell_to_col[&cj];

                temp_g.fill(0.0);

                let col_slice_i = x_dn.col(col_i);
                for (&row, &val) in col_slice_i
                    .row_indices()
                    .iter()
                    .zip(col_slice_i.values().iter())
                {
                    temp_g[row] += val;
                }

                let col_slice_j = x_dn.col(col_j);
                for (&row, &val) in col_slice_j
                    .row_indices()
                    .iter()
                    .zip(col_slice_j.values().iter())
                {
                    temp_g[row] += val;
                }

                let proj = &basis_t * &temp_g;
                let base = e_idx * m;
                for (d, &v) in proj.iter().enumerate() {
                    chunk_profiles[base + d] = v.max(0.0);
                }
            }

            Ok((lb, chunk_profiles))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    let mut profiles = vec![0.0f32; n_edges * m];
    for (lb, chunk) in partial_results {
        let base = lb * m;
        profiles[base..base + chunk.len()].copy_from_slice(&chunk);
    }

    Ok(LinkProfileStore::new(profiles, n_edges, m))
}

/// Coarsen fine-cell raw expression to super-cells.
///
/// Returns an `[n_genes × n_super_cells]` dense matrix whose column `c`
/// holds `Σ_{i: cell_labels[i] == c} x_fine[:, i]` — i.e. the total
/// gene counts pooled across every fine cell assigned to super-cell `c`.
///
/// Streams fine cells in blocks for memory efficiency. The return buffer
/// is dense (not sparse) because the super-cell expression is dense by
/// construction: any gene expressed in any fine cell within a cluster
/// contributes to that cluster's column.
pub fn coarsen_cell_expression_dense(
    data: &SparseIoVec,
    cell_labels: &[usize],
    n_super_cells: usize,
    block_size: Option<usize>,
) -> anyhow::Result<Mat> {
    let n_genes = data.num_rows();
    let n_cells = data.num_columns();
    debug_assert_eq!(cell_labels.len(), n_cells);

    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);

    let partials: Vec<Mat> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Mat> {
            let x = data.read_columns_csc(lb..ub)?;
            let mut local = Mat::zeros(n_genes, n_super_cells);
            for col in 0..x.ncols() {
                let sc = cell_labels[lb + col];
                let s = x.col(col);
                for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                    local[(row, sc)] += val;
                }
            }
            Ok(local)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut super_expr = Mat::zeros(n_genes, n_super_cells);
    for local in partials {
        super_expr += local;
    }
    Ok(super_expr)
}

/// Build projection profiles for super-edges from pre-coarsened super-cell
/// expression: `y_e = basis^T · (x_super[:, a] + x_super[:, b])`, with
/// negative entries clamped to 0.
///
/// Used inside the V-cycle cascade when the profile mode is `Projection`.
/// The alternative is to (incorrectly) read fine-cell columns at indices
/// equal to cluster labels — which gave arbitrary fine cells as
/// "super-cells". This function replaces that path.
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
    let mut key_to_super: HashMap<(usize, usize), usize> = Default::default();
    let mut super_edges: Vec<(usize, usize)> = Vec::new();
    let mut fine_to_super = Vec::with_capacity(edges.len());

    for &(i, j) in edges {
        let li = cell_labels[i];
        let lj = cell_labels[j];
        let key = (li.min(lj), li.max(lj));
        let se = *key_to_super.entry(key).or_insert_with(|| {
            let s = super_edges.len();
            super_edges.push(key);
            s
        });
        fine_to_super.push(se);
    }

    (super_edges, fine_to_super)
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

/// Compute topic-specific gene expression statistics via Poisson-Gamma.
///
/// Given cell propensity [N × K] and sparse expression data [G × N],
/// computes weighted gene sums `X @ propensity^T` and fits a Poisson-Gamma
/// to get posterior gene expression rates per topic. Before calibration the
/// sufficient statistic is reweighted row-wise by NB Fisher-info weights
/// `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`, matching the weighting used during
/// DC-Poisson clustering so clustering and reporting stay consistent.
///
/// Writes `{out_prefix}.gene_topic.parquet` (genes × K). When
/// `gene_weights` is `Some`, those precomputed NB Fisher-info weights are
/// applied to the per-(gene, topic) sufficient statistic; otherwise they
/// are recomputed from the data (extra full-data scan).
pub fn compute_gene_topic_stat(
    cell_propensity: &Mat,
    data_vec: &SparseIoVec,
    gene_weights: Option<&[f32]>,
    block_size: Option<usize>,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let param = fit_gene_topic_param(cell_propensity, data_vec, gene_weights, block_size)?;
    let gene_names = data_vec.row_names()?;
    write_gene_topic_param(&param, &gene_names, out_prefix)
}

/// Fit the Poisson-Gamma posterior over gene × topic without writing to disk.
///
/// Returns the calibrated `GammaMatrix` so callers can reuse the posterior
/// (e.g. to compute pairwise topic similarity for cosine merging) without
/// re-reading the parquet output. The sufficient statistic is row-scaled by
/// NB Fisher-info weights `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`, matching
/// `compute_gene_topic_stat`.
pub fn fit_gene_topic_param(
    cell_propensity: &Mat,
    data_vec: &SparseIoVec,
    gene_weights: Option<&[f32]>,
    block_size: Option<usize>,
) -> anyhow::Result<matrix_param::dmatrix_gamma::GammaMatrix> {
    use matrix_param::dmatrix_gamma::GammaMatrix;
    use matrix_param::traits::TwoStatParam;

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();
    let k = cell_propensity.ncols();

    info!("Computing gene-topic statistics...");
    let prop_kn = cell_propensity.transpose();
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Gene-topic {bar:40} {pos}/{len} blocks ({eta})",
    );
    let partial_stats: Vec<(Mat, DVec)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(Mat, DVec)> {
            let x_gn = data_vec.read_columns_csc(lb..ub)?;
            let block_len = ub - lb;
            let mut p_kn_block = Mat::zeros(k, block_len);
            for i in 0..block_len {
                p_kn_block.column_mut(i).copy_from(&prop_kn.column(lb + i));
            }
            let n_k = p_kn_block.column_sum();
            let sum_gk = x_gn * p_kn_block.transpose();
            Ok((sum_gk, n_k))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    let mut sum_gk = Mat::zeros(n_genes, k);
    let mut n_1k = Mat::zeros(1, k);
    for (s, n) in partial_stats {
        sum_gk += s;
        n_1k += n.transpose();
    }

    let owned_w;
    let w: &[f32] = match gene_weights {
        Some(w) => w,
        None => {
            info!("Computing NB Fisher-info weights for gene-topic stats");
            owned_w = compute_nb_fisher_weights(data_vec, block_size)?;
            &owned_w
        }
    };
    apply_gene_weights(&mut sum_gk, w);

    let mut gamma_param = GammaMatrix::new((n_genes, k), 1.0, 1.0);
    let denom_gk = DVec::from_element(n_genes, 1.0) * &n_1k;
    gamma_param.update_stat(&sum_gk, &denom_gk);
    gamma_param.calibrate();

    Ok(gamma_param)
}

/// Write a fitted gene-topic posterior to `<out_prefix>.gene_topic.parquet`
/// in melted (gene, topic, mean, sd, log_mean, log_sd) form.
pub fn write_gene_topic_param(
    param: &matrix_param::dmatrix_gamma::GammaMatrix,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_param::traits::Inference;
    let k = param.posterior_mean().ncols();
    let topic_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    param.to_melted_parquet(
        &(out_prefix.to_string() + ".gene_topic.parquet"),
        (Some(gene_names), Some("gene")),
        (Some(&topic_names), Some("topic")),
    )?;
    Ok(())
}

/// Compute propensity and gene-topic statistics from latent pair projections.
///
/// 1. K-means on `proj_kn` (K_latent × N_pairs) → edge cluster labels
/// 2. Propensity: soft cell membership from edge clusters [N_cells × K_clusters]
/// 3. Gene-topic stat: Poisson-Gamma gene expression rates per topic [G × K_clusters]
///
/// Writes `{out_prefix}.propensity.parquet` and `{out_prefix}.gene_topic.parquet`.
/// Config for `compute_propensity_and_gene_topic_stat`.
pub struct PropensityReportConfig {
    pub n_clusters: usize,
    pub block_size: Option<usize>,
}

pub fn compute_propensity_and_gene_topic_stat(
    proj_kn: &Mat,
    edges: &[(usize, usize)],
    data_vec: &SparseIoVec,
    n_cells: usize,
    cfg: &PropensityReportConfig,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let PropensityReportConfig {
        n_clusters,
        block_size,
    } = *cfg;

    // 1. K-means on latent edge vectors
    info!("K-means clustering edges (k={})...", n_clusters);
    let edge_membership = proj_kn.kmeans_columns(KmeansArgs {
        num_clusters: n_clusters,
        max_iter: 100,
    });

    // 2. Propensity [N_cells × K]
    info!("Computing cell propensity...");
    let cell_propensity = compute_node_membership(edges, &edge_membership, n_cells, n_clusters);

    let cell_names = data_vec.column_names()?;

    // Dominant cluster per cell (argmax of propensity)
    let cluster_col: Vec<f32> = (0..n_cells)
        .map(|i| {
            cell_propensity
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, _)| k as f32)
                .unwrap_or(0.0)
        })
        .collect();
    let cluster_mat = Mat::from_column_slice(n_cells, 1, &cluster_col);

    let entropy_vec = shannon_entropy_rows(&cell_propensity);
    let entropy_mat = Mat::from_column_slice(n_cells, 1, entropy_vec.as_slice());

    let mut col_names: Vec<Box<str>> = (0..n_clusters)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    col_names.push("cluster".into());
    col_names.push("entropy".into());

    let combined = concatenate_horizontal(&[cell_propensity.clone(), cluster_mat, entropy_mat])?;
    combined.to_parquet_with_names(
        &(out_prefix.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&col_names),
    )?;

    // Edge cluster assignments
    write_edge_clusters(out_prefix, edges, &edge_membership, &cell_names)?;

    // 3. Gene-topic stat
    compute_gene_topic_stat(&cell_propensity, data_vec, None, block_size, out_prefix)
}

/// Write per-edge K-means cluster assignments to parquet.
fn write_edge_clusters(
    out_prefix: &str,
    edges: &[(usize, usize)],
    edge_membership: &[usize],
    cell_names: &[Box<str>],
) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_edges = edges.len();
    let left_cells: Vec<Box<str>> = edges.iter().map(|&(i, _)| cell_names[i].clone()).collect();
    let right_cells: Vec<Box<str>> = edges.iter().map(|&(_, j)| cell_names[j].clone()).collect();
    let cluster_f32: Vec<f32> = edge_membership.iter().map(|&k| k as f32).collect();

    let col_names: Vec<Box<str>> = vec!["right_cell".into(), "cluster".into()];
    let col_types = vec![ParquetType::BYTE_ARRAY, ParquetType::FLOAT];

    let writer = ParquetWriter::new(
        &(out_prefix.to_string() + ".edge_cluster.parquet"),
        (n_edges, 2),
        (Some(&left_cells), Some(&col_names)),
        Some(&col_types),
        Some("left_cell"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_string_column(&mut row_group, &right_cells)?;
    parquet_add_numeric_column(&mut row_group, &cluster_f32)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
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
        for &(u, v) in &graph.gene_edges {
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
    let pb = new_progress_bar(
        jobs.len() as u64,
        "Module expression {bar:40} {pos}/{len} blocks ({eta})",
    );

    let partials: Vec<(usize, Mat, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
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
    pb.finish_and_clear();

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

/// Aggregate fine-cell module expression to super-cell module expression.
///
/// For each fine cell `c` with super-cell label `cell_labels[c] = sc`:
///   `super_expr[m, sc] += module_expr[m, c]`
/// Also returns per-super-cell totals.
pub fn coarsen_module_expression(
    module_expr: &Mat,
    cell_labels: &[usize],
    n_super_cells: usize,
) -> (Mat, Vec<f32>) {
    let n_modules = module_expr.nrows();
    let n_cells = module_expr.ncols();
    debug_assert_eq!(cell_labels.len(), n_cells);

    let mut super_expr = Mat::zeros(n_modules, n_super_cells);
    let mut super_totals = vec![0.0f32; n_super_cells];
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
