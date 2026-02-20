#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! Link profile construction, coarsening, and projection refinement.
//!
//! Builds projected link profiles from sparse expression data and KNN edges,
//! coarsens them by cell-level cluster labels, and refines the projection
//! basis via community centroids.

use crate::link_community_model::LinkProfileStore;
use crate::srt_common::*;
use crate::srt_gene_pairs::visit_gene_pair_deltas;
use matrix_param::io::ParamIo;
use matrix_util::utils::generate_minibatch_intervals;
use nalgebra_sparse::csc::CscMatrix;

/// Compute per-gene total counts from sparse data.
///
/// Returns a vector of length `n_genes` with the sum of all entries per row.
pub fn compute_gene_totals(
    data: &SparseIoVec,
    block_size: usize,
) -> anyhow::Result<Vec<f64>> {
    let n_genes = data.num_rows();
    let n_cells = data.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, block_size);

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
    for g in 0..basis.nrows() {
        if gene_totals[g] < min_count as f64 {
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
    let mut unique_set: HashSet<usize> = HashSet::new();
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

/// Build projected link profiles from sparse data, KNN edges, and a random
/// projection basis.
///
/// For each edge (i, j), computes y_e = W^T (x_i + x_j) where W is the
/// [n_genes × proj_dim] basis. Visits edges in blocks for I/O efficiency.
///
/// * `data` - Sparse expression data [n_genes × n_cells]
/// * `edges` - Sorted edge list from KNN graph
/// * `basis` - Projection basis [n_genes × proj_dim]
/// * `batch_effect` - Optional batch effect matrix [n_genes × n_batches]
/// * `block_size` - Number of edges per parallel block
pub fn build_edge_profiles(
    data: &SparseIoVec,
    edges: &[(usize, usize)],
    basis: &Mat,
    _batch_effect: Option<&Mat>,
    block_size: usize,
) -> anyhow::Result<LinkProfileStore> {
    let n_edges = edges.len();
    let m = basis.ncols(); // proj_dim

    let jobs = generate_minibatch_intervals(n_edges, block_size);

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

            // For each edge, project sum of the two cells
            let mut chunk_profiles = vec![0.0f32; chunk_size * m];
            let n_genes = x_dn.nrows();

            for (e_idx, &(ci, cj)) in chunk_edges.iter().enumerate() {
                let col_i = cell_to_col[&ci];
                let col_j = cell_to_col[&cj];

                // Dense accumulation: x_i + x_j → temp_g
                let mut temp_g = vec![0.0f32; n_genes];

                // Accumulate from sparse column i
                let col_slice_i = x_dn.col(col_i);
                for (&row, &val) in col_slice_i
                    .row_indices()
                    .iter()
                    .zip(col_slice_i.values().iter())
                {
                    temp_g[row] += val;
                }

                // Accumulate from sparse column j
                let col_slice_j = x_dn.col(col_j);
                for (&row, &val) in col_slice_j
                    .row_indices()
                    .iter()
                    .zip(col_slice_j.values().iter())
                {
                    temp_g[row] += val;
                }

                // Project: profile[e] = basis^T * temp_g
                let base = e_idx * m;
                for d in 0..m {
                    let mut dot = 0.0f32;
                    for g in 0..n_genes {
                        dot += basis[(g, d)] * temp_g[g];
                    }
                    chunk_profiles[base + d] = dot.max(0.0); // ReLU to keep non-negative
                }
            }

            Ok((lb, chunk_profiles))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    // Assemble into single profiles buffer
    let mut profiles = vec![0.0f32; n_edges * m];
    for (lb, chunk) in partial_results {
        let chunk_edges_count = chunk.len() / m;
        for e in 0..chunk_edges_count {
            let src_base = e * m;
            let dst_base = (lb + e) * m;
            profiles[dst_base..dst_base + m].copy_from_slice(&chunk[src_base..src_base + m]);
        }
    }

    Ok(LinkProfileStore::new(profiles, n_edges, m))
}

/// Coarsen link profiles by cell-level cluster labels.
///
/// Each edge (i, j) maps to a super-edge key (min(label[i], label[j]), max(...)).
/// Super-link profiles are the element-wise sum of all fine edges mapping to them.
///
/// Returns (super-link profiles, fine-edge → super-edge mapping).
pub fn coarsen_edge_profiles(
    profiles: &LinkProfileStore,
    edges: &[(usize, usize)],
    cell_labels: &[usize],
) -> (LinkProfileStore, Vec<usize>) {
    let m = profiles.m;
    let n_edges = edges.len();

    // Map each edge to a canonical super-edge key
    let mut key_to_super: HashMap<(usize, usize), usize> = HashMap::new();
    let mut next_super = 0usize;
    let mut fine_to_super = Vec::with_capacity(n_edges);

    for &(i, j) in edges {
        let li = cell_labels[i];
        let lj = cell_labels[j];
        let key = (li.min(lj), li.max(lj));
        let se = *key_to_super.entry(key).or_insert_with(|| {
            let s = next_super;
            next_super += 1;
            s
        });
        fine_to_super.push(se);
    }

    let n_super = next_super;
    let mut super_profiles = vec![0.0f32; n_super * m];

    // Accumulate fine link profiles into super-link profiles
    for e in 0..n_edges {
        let se = fine_to_super[e];
        let src = profiles.profile(e);
        let base = se * m;
        for g in 0..m {
            super_profiles[base + g] += src[g];
        }
    }

    (
        LinkProfileStore::new(super_profiles, n_super, m),
        fine_to_super,
    )
}

/// Compute gene centroids per community in original G-dimensional space.
///
/// For each community k, accumulates the sum of x_i + x_j for all edges
/// assigned to k, then normalizes by edge count.
///
/// Returns [n_genes × k] centroid matrix.
pub fn compute_community_centroids(
    data: &SparseIoVec,
    edges: &[(usize, usize)],
    membership: &[usize],
    k: usize,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let n_genes = data.num_rows();
    let n_edges = edges.len();

    let jobs = generate_minibatch_intervals(n_edges, block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Centroids {bar:40} {pos}/{len} blocks ({eta})",
    );
    let partial_stats: Vec<(Mat, Vec<usize>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(Mat, Vec<usize>)> {
            let chunk_edges = &edges[lb..ub];
            let chunk_mem = &membership[lb..ub];

            let (x_dn, cell_to_col) = read_unique_cells_for_edges(data, chunk_edges)?;

            let mut local_sum = Mat::zeros(n_genes, k);
            let mut local_count = vec![0usize; k];

            for (&(ci, cj), &c) in chunk_edges.iter().zip(chunk_mem.iter()) {
                let col_i = cell_to_col[&ci];
                let col_j = cell_to_col[&cj];

                // Accumulate sparse columns into centroid for community c
                let col_slice_i = x_dn.col(col_i);
                for (&row, &val) in col_slice_i
                    .row_indices()
                    .iter()
                    .zip(col_slice_i.values().iter())
                {
                    local_sum[(row, c)] += val;
                }
                let col_slice_j = x_dn.col(col_j);
                for (&row, &val) in col_slice_j
                    .row_indices()
                    .iter()
                    .zip(col_slice_j.values().iter())
                {
                    local_sum[(row, c)] += val;
                }

                local_count[c] += 1;
            }

            Ok((local_sum, local_count))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    // Aggregate partial results
    let mut total_sum = Mat::zeros(n_genes, k);
    let mut total_count = vec![0usize; k];
    for (local_sum, local_count) in partial_stats {
        total_sum += local_sum;
        for c in 0..k {
            total_count[c] += local_count[c];
        }
    }

    // Normalize by count
    for c in 0..k {
        if total_count[c] > 0 {
            let scale = 1.0 / total_count[c] as f32;
            total_sum.column_mut(c).scale_mut(scale);
        }
    }

    Ok(total_sum)
}

/// Refine projection basis via SVD on community centroids.
///
/// Takes the [n_genes × K] centroid matrix, runs SVD, and returns
/// the top `m` left singular vectors as [n_genes × m].
pub fn refine_projection_basis(centroids: &Mat, m: usize) -> anyhow::Result<Mat> {
    let n_genes = centroids.nrows();
    let k = centroids.ncols();

    // Center columns
    let mut centered = centroids.clone();
    centered.centre_columns_inplace();

    // SVD via eigendecomposition of C^T * C [K × K]
    let ctc = centered.transpose() * &centered;
    let eig = ctc.symmetric_eigen();

    // Sort eigenvalues descending
    let mut sorted_idx: Vec<usize> = (0..k).collect();
    sorted_idx.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());

    let n_components = m.min(k);

    // Compute left singular vectors: U = C * V * S^{-1}
    let mut basis = Mat::zeros(n_genes, m);
    for j in 0..n_components {
        let idx = sorted_idx[j];
        let sv = eig.eigenvalues[idx].max(0.0).sqrt();
        if sv > 1e-10 {
            let u_j = &centered * eig.eigenvectors.column(idx) / sv;
            basis.column_mut(j).copy_from(&u_j);
        }
    }

    // Fill remaining columns with random vectors if m > k
    if m > n_components {
        let random_fill = Mat::rnorm(n_genes, m - n_components);
        for j in n_components..m {
            basis
                .column_mut(j)
                .copy_from(&random_fill.column(j - n_components));
        }
    }

    Ok(basis)
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
    for i in 0..n_cells {
        if degrees[i] > 0 {
            let scale = 1.0 / degrees[i] as f32;
            counts.row_mut(i).scale_mut(scale);
        }
    }

    counts
}

/// Compute gene embeddings via a random sketch over coarsened cell clusters.
///
/// For each gene g, accumulates `gene_embed[g,:] += x[g,c] * R[cell_labels[c],:]`
/// where R is a `[n_clusters × sketch_dim]` random Gaussian matrix. After
/// accumulation, normalizes by gene totals so each row is a rate vector.
///
/// * `data` - Sparse expression data [n_genes × n_cells]
/// * `cell_labels` - Cluster assignment per cell
/// * `n_clusters` - Number of distinct clusters
/// * `sketch_dim` - Embedding dimension
/// * `block_size` - Number of cells per parallel block
pub fn compute_gene_module_sketch(
    data: &SparseIoVec,
    cell_labels: &[usize],
    n_clusters: usize,
    sketch_dim: usize,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let n_genes = data.num_rows();
    let n_cells = data.num_columns();

    // Random sketch matrix [n_clusters × sketch_dim]
    let r_mat = Mat::rnorm(n_clusters, sketch_dim);

    let jobs = generate_minibatch_intervals(n_cells, block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Gene sketch {bar:40} {pos}/{len} blocks ({eta})",
    );
    let partials: Vec<(Mat, Vec<f64>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(Mat, Vec<f64>)> {
            let cells: Vec<usize> = (lb..ub).collect();
            let x_dn = data.read_columns_csc(cells.iter().copied())?;

            let mut local_embed = Mat::zeros(n_genes, sketch_dim);
            let mut local_total = vec![0.0f64; n_genes];

            for (local_col, &cell) in cells.iter().enumerate() {
                let col_slice = x_dn.col(local_col);
                let cluster = cell_labels[cell];
                for (&row, &val) in col_slice
                    .row_indices()
                    .iter()
                    .zip(col_slice.values().iter())
                {
                    local_total[row] += val as f64;
                    for d in 0..sketch_dim {
                        local_embed[(row, d)] += val * r_mat[(cluster, d)];
                    }
                }
            }

            Ok((local_embed, local_total))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    // Aggregate
    let mut gene_embed = Mat::zeros(n_genes, sketch_dim);
    let mut gene_total = vec![0.0f64; n_genes];
    for (local_embed, local_total) in partials {
        gene_embed += local_embed;
        for g in 0..n_genes {
            gene_total[g] += local_total[g];
        }
    }

    // Normalize by gene totals
    for g in 0..n_genes {
        let denom = gene_total[g].max(1.0) as f32;
        for d in 0..sketch_dim {
            gene_embed[(g, d)] /= denom;
        }
    }

    Ok(gene_embed)
}

/// Build edge profiles as module-level counts via table lookup.
///
/// For each edge (i, j), sums the expression of cells i and j binned by
/// gene module assignment. Produces exact Poisson counts (sum of Poissons
/// is Poisson).
///
/// * `data` - Sparse expression data [n_genes × n_cells]
/// * `edges` - Sorted edge list from KNN graph
/// * `gene_to_module` - Module assignment per gene
/// * `n_modules` - Number of gene modules
/// * `block_size` - Number of edges per parallel block
pub fn build_edge_profiles_by_module(
    data: &SparseIoVec,
    edges: &[(usize, usize)],
    gene_to_module: &[usize],
    n_modules: usize,
    block_size: usize,
) -> anyhow::Result<LinkProfileStore> {
    let n_edges = edges.len();

    let jobs = generate_minibatch_intervals(n_edges, block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Module profiles {bar:40} {pos}/{len} blocks ({eta})",
    );
    let partial_results: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let chunk_edges = &edges[lb..ub];
            let chunk_size = ub - lb;

            let (x_dn, cell_to_col) = read_unique_cells_for_edges(data, chunk_edges)?;

            // For each edge, accumulate module counts
            let mut chunk_profiles = vec![0.0f32; chunk_size * n_modules];

            for (e_idx, &(ci, cj)) in chunk_edges.iter().enumerate() {
                let col_i = cell_to_col[&ci];
                let col_j = cell_to_col[&cj];
                let base = e_idx * n_modules;

                // Accumulate from sparse column i
                let col_slice_i = x_dn.col(col_i);
                for (&row, &val) in col_slice_i
                    .row_indices()
                    .iter()
                    .zip(col_slice_i.values().iter())
                {
                    chunk_profiles[base + gene_to_module[row]] += val;
                }

                // Accumulate from sparse column j
                let col_slice_j = x_dn.col(col_j);
                for (&row, &val) in col_slice_j
                    .row_indices()
                    .iter()
                    .zip(col_slice_j.values().iter())
                {
                    chunk_profiles[base + gene_to_module[row]] += val;
                }
            }

            Ok((lb, chunk_profiles))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    // Assemble into single profiles buffer
    let mut profiles = vec![0.0f32; n_edges * n_modules];
    for (lb, chunk) in partial_results {
        let chunk_edges_count = chunk.len() / n_modules;
        for e in 0..chunk_edges_count {
            let src_base = e * n_modules;
            let dst_base = (lb + e) * n_modules;
            profiles[dst_base..dst_base + n_modules]
                .copy_from_slice(&chunk[src_base..src_base + n_modules]);
        }
    }

    Ok(LinkProfileStore::new(profiles, n_edges, n_modules))
}

/// Compute topic-specific gene expression statistics via Poisson-Gamma.
///
/// Given cell propensity [N × K] and sparse expression data [G × N],
/// computes weighted gene sums `X @ propensity^T` and fits a Poisson-Gamma
/// to get posterior gene expression rates per topic.
///
/// Writes `{out_prefix}.gene_topic.parquet` (genes × K).
pub fn compute_gene_topic_stat(
    cell_propensity: &Mat,
    data_vec: &SparseIoVec,
    block_size: usize,
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_param::dmatrix_gamma::GammaMatrix;
    use matrix_param::traits::TwoStatParam;

    let gene_names = data_vec.row_names()?;
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();
    let k = cell_propensity.ncols();

    info!("Computing gene-topic statistics...");
    let prop_kn = cell_propensity.transpose();
    let jobs = generate_minibatch_intervals(n_cells, block_size);

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

    let mut gamma_param = GammaMatrix::new((n_genes, k), 1.0, 1.0);
    let denom_gk = DVec::from_element(n_genes, 1.0) * &n_1k;
    gamma_param.update_stat(&sum_gk, &denom_gk);
    gamma_param.calibrate();

    let topic_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    gamma_param.to_parquet_with_names(
        &(out_prefix.to_string() + ".gene_topic.parquet"),
        (Some(&gene_names), Some("gene")),
        Some(&topic_names),
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
pub fn compute_propensity_and_gene_topic_stat(
    proj_kn: &Mat,
    edges: &[(usize, usize)],
    data_vec: &SparseIoVec,
    n_cells: usize,
    n_clusters: usize,
    block_size: usize,
    out_prefix: &str,
) -> anyhow::Result<()> {
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

    let mut col_names: Vec<Box<str>> = (0..n_clusters)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    col_names.push("cluster".into());

    let combined = concatenate_horizontal(&[cell_propensity.clone(), cluster_mat])?;
    combined.to_parquet_with_names(
        &(out_prefix.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&col_names),
    )?;

    // Edge cluster assignments
    write_edge_clusters(out_prefix, edges, &edge_membership, &cell_names)?;

    // 3. Gene-topic stat
    compute_gene_topic_stat(&cell_propensity, data_vec, block_size, out_prefix)
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

/// Build edge profiles from gene-pair interaction deltas.
///
/// For each spatial edge (i,j), accumulates positive δ values across
/// all gene pairs in the gene adjacency list. Each edge profile has
/// dimension `n_gene_pairs`.
///
/// * `data` - Sparse expression data [n_genes × n_cells]
/// * `edges` - Sorted spatial edge list from KNN graph
/// * `gene_adj` - Directed gene adjacency: gene_adj[g] = [(neighbor, edge_idx)]
/// * `gene_means` - Per-gene raw means for delta computation
/// * `n_gene_pairs` - Number of gene-gene edges (profile dimension)
/// * `block_size` - Number of spatial edges per parallel block
pub fn build_edge_profiles_by_gene_pairs(
    data: &SparseIoVec,
    edges: &[(usize, usize)],
    gene_adj: &[Vec<(usize, usize)>],
    gene_means: &DVec,
    n_gene_pairs: usize,
    block_size: usize,
) -> anyhow::Result<LinkProfileStore> {
    let n_edges = edges.len();

    let jobs = generate_minibatch_intervals(n_edges, block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Gene-pair profiles {bar:40} {pos}/{len} blocks ({eta})",
    );
    let partial_results: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let chunk_edges = &edges[lb..ub];
            let chunk_size = ub - lb;

            let (x_dn, cell_to_col) = read_unique_cells_for_edges(data, chunk_edges)?;

            let mut chunk_profiles = vec![0.0f32; chunk_size * n_gene_pairs];

            for (e_idx, &(ci, cj)) in chunk_edges.iter().enumerate() {
                let base = e_idx * n_gene_pairs;

                // Accumulate δ⁺ from cell i
                let col_i = cell_to_col[&ci];
                let col_slice_i = x_dn.col(col_i);
                visit_gene_pair_deltas(
                    col_slice_i.row_indices(),
                    col_slice_i.values(),
                    gene_adj,
                    gene_means,
                    false,
                    |edge_idx, delta| {
                        if delta > 0.0 {
                            chunk_profiles[base + edge_idx] += delta;
                        }
                    },
                );

                // Accumulate δ⁺ from cell j
                let col_j = cell_to_col[&cj];
                let col_slice_j = x_dn.col(col_j);
                visit_gene_pair_deltas(
                    col_slice_j.row_indices(),
                    col_slice_j.values(),
                    gene_adj,
                    gene_means,
                    false,
                    |edge_idx, delta| {
                        if delta > 0.0 {
                            chunk_profiles[base + edge_idx] += delta;
                        }
                    },
                );
            }

            Ok((lb, chunk_profiles))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    // Assemble into single profiles buffer
    let mut profiles = vec![0.0f32; n_edges * n_gene_pairs];
    for (lb, chunk) in partial_results {
        let chunk_edges_count = chunk.len() / n_gene_pairs;
        for e in 0..chunk_edges_count {
            let src_base = e * n_gene_pairs;
            let dst_base = (lb + e) * n_gene_pairs;
            profiles[dst_base..dst_base + n_gene_pairs]
                .copy_from_slice(&chunk[src_base..src_base + n_gene_pairs]);
        }
    }

    Ok(LinkProfileStore::new(profiles, n_edges, n_gene_pairs))
}

/// Filter a LinkProfileStore to keep only the specified column indices.
///
/// Rebuilds the store with a reduced profile dimension. Used after
/// elbow filtering to remove low-signal gene pairs.
pub fn filter_profile_columns(store: &LinkProfileStore, keep: &[usize]) -> LinkProfileStore {
    let n_edges = store.n_edges;
    let new_m = keep.len();
    let mut new_profiles = vec![0.0f32; n_edges * new_m];

    for e in 0..n_edges {
        let old_profile = store.profile(e);
        let new_base = e * new_m;
        for (new_col, &old_col) in keep.iter().enumerate() {
            new_profiles[new_base + new_col] = old_profile[old_col];
        }
    }

    LinkProfileStore::new(new_profiles, n_edges, new_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarsen_edge_profiles() {
        // 4 edges, proj_dim=3, 2 clusters of cells
        let profiles_data = vec![
            1.0, 2.0, 3.0, // edge 0: cells (0,1) → cluster pair (0,0)
            4.0, 5.0, 6.0, // edge 1: cells (0,2) → cluster pair (0,1)
            7.0, 8.0, 9.0, // edge 2: cells (1,2) → cluster pair (0,1)
            2.0, 3.0, 4.0, // edge 3: cells (2,3) → cluster pair (1,1)
        ];
        let store = LinkProfileStore::new(profiles_data, 4, 3);
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
        let cell_labels = vec![0, 0, 1, 1]; // cells 0,1 → cluster 0; cells 2,3 → cluster 1

        let (super_store, f2s) = coarsen_edge_profiles(&store, &edges, &cell_labels);

        // Edge 0: (0,1) → labels (0,0) → key (0,0) → super 0
        // Edge 1: (0,2) → labels (0,1) → key (0,1) → super 1
        // Edge 2: (1,2) → labels (0,1) → key (0,1) → super 1
        // Edge 3: (2,3) → labels (1,1) → key (1,1) → super 2
        assert_eq!(super_store.n_edges, 3);
        assert_eq!(f2s[0], f2s[0]); // trivially
        assert_eq!(f2s[1], f2s[2]); // same super-edge
        assert_ne!(f2s[0], f2s[1]); // different super-edges

        // Super-edge 1 should be sum of edges 1 and 2
        let se1 = f2s[1];
        let expected: Vec<f32> = vec![4.0 + 7.0, 5.0 + 8.0, 6.0 + 9.0];
        assert_eq!(super_store.profile(se1), &expected[..]);
    }

    #[test]
    fn test_transfer_labels() {
        let f2s = vec![0, 1, 1, 2, 0];
        let super_mem = vec![2, 0, 1];
        let fine_mem = transfer_labels(&f2s, &super_mem);
        assert_eq!(fine_mem, vec![2, 0, 0, 1, 2]);
    }

    #[test]
    fn test_compute_node_membership() {
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
        let membership = vec![0, 0, 1, 1];
        let nm = compute_node_membership(&edges, &membership, 4, 2);

        assert_eq!(nm.nrows(), 4);
        assert_eq!(nm.ncols(), 2);

        // Cell 0: edges 0,1 → communities 0,0 → [1.0, 0.0]
        assert!((nm[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((nm[(0, 1)] - 0.0).abs() < 1e-6);

        // Cell 2: edges 1,2,3 → communities 0,1,1 → [1/3, 2/3]
        assert!((nm[(2, 0)] - 1.0 / 3.0).abs() < 1e-6);
        assert!((nm[(2, 1)] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_refine_projection_basis() {
        // Create a simple centroid matrix [10 genes × 3 communities]
        let mut centroids = Mat::zeros(10, 3);
        // Community 0: genes 0-3 active
        for g in 0..4 {
            centroids[(g, 0)] = 5.0;
        }
        // Community 1: genes 4-6 active
        for g in 4..7 {
            centroids[(g, 1)] = 5.0;
        }
        // Community 2: genes 7-9 active
        for g in 7..10 {
            centroids[(g, 2)] = 5.0;
        }

        let basis = refine_projection_basis(&centroids, 3).unwrap();
        assert_eq!(basis.nrows(), 10);
        assert_eq!(basis.ncols(), 3);

        // All values should be finite
        for i in 0..basis.nrows() {
            for j in 0..basis.ncols() {
                assert!(basis[(i, j)].is_finite());
            }
        }
    }

    #[test]
    fn test_refine_more_dims_than_communities() {
        let centroids = Mat::from_fn(10, 2, |g, c| if g % 2 == c { 3.0 } else { 0.5 });
        let basis = refine_projection_basis(&centroids, 5).unwrap();
        assert_eq!(basis.nrows(), 10);
        assert_eq!(basis.ncols(), 5);
    }

    fn make_test_sparse_io(raw: &ndarray::Array2<f32>) -> SparseIoVec {
        use std::sync::Arc;
        let nrow = raw.nrows();
        let ncol = raw.ncols();
        let rows: Vec<Box<str>> = (0..nrow)
            .map(|i| format!("g{i}").into_boxed_str())
            .collect();
        let cols: Vec<Box<str>> = (0..ncol)
            .map(|i| format!("c{i}").into_boxed_str())
            .collect();
        let mut sp = create_sparse_from_ndarray(raw, None, None).unwrap();
        sp.register_row_names_vec(&rows);
        sp.register_column_names_vec(&cols);
        sp.preload_columns().unwrap();
        let mut vec = SparseIoVec::new();
        vec.push(Arc::from(sp), None).unwrap();
        vec
    }

    #[test]
    fn test_compute_gene_module_sketch() {
        // 5 genes, 6 cells in 2 clusters
        let n_genes = 5;
        let n_cells = 6;
        let mut raw = ndarray::Array2::<f32>::zeros((n_genes, n_cells));

        // Cluster 0: cells 0,1,2 — genes 0,1 are active
        for c in 0..3 {
            raw[(0, c)] = 10.0;
            raw[(1, c)] = 5.0;
        }
        // Cluster 1: cells 3,4,5 — genes 3,4 are active
        for c in 3..6 {
            raw[(3, c)] = 8.0;
            raw[(4, c)] = 12.0;
        }

        let data = make_test_sparse_io(&raw);
        let cell_labels = vec![0, 0, 0, 1, 1, 1];

        let sketch_dim = 100;
        let embed = compute_gene_module_sketch(&data, &cell_labels, 2, sketch_dim, 3).unwrap();

        assert_eq!(embed.nrows(), n_genes);
        assert_eq!(embed.ncols(), sketch_dim);

        // Gene 2 has no counts — its embedding should be zero
        for d in 0..sketch_dim {
            assert!((embed[(2, d)]).abs() < 1e-10);
        }

        // Genes in the same cluster should have similar embeddings
        // (both project through R[cluster_0,:])
        let dot_01: f32 = (0..sketch_dim).map(|d| embed[(0, d)] * embed[(1, d)]).sum();
        let dot_04: f32 = (0..sketch_dim).map(|d| embed[(0, d)] * embed[(4, d)]).sum();

        // Genes 0,1 (same cluster) should be more aligned than genes 0,4
        assert!(dot_01 > dot_04);
    }

    #[test]
    fn test_build_edge_profiles_by_module() {
        // 4 genes, 4 cells, 2 modules
        let n_genes = 4;
        let n_cells = 4;
        let mut raw = ndarray::Array2::<f32>::zeros((n_genes, n_cells));
        // cell 0: gene0=3, gene1=2
        raw[(0, 0)] = 3.0;
        raw[(1, 0)] = 2.0;
        // cell 1: gene2=5
        raw[(2, 1)] = 5.0;
        // cell 2: gene0=1, gene3=4
        raw[(0, 2)] = 1.0;
        raw[(3, 2)] = 4.0;
        // cell 3: gene1=6
        raw[(1, 3)] = 6.0;

        let data = make_test_sparse_io(&raw);

        // Module 0: genes 0,1; Module 1: genes 2,3
        let gene_to_module = vec![0, 0, 1, 1];
        let edges = vec![(0, 1), (2, 3)];

        let store = build_edge_profiles_by_module(&data, &edges, &gene_to_module, 2, 10).unwrap();

        assert_eq!(store.n_edges, 2);
        assert_eq!(store.m, 2);

        // Edge (0,1): cell0 + cell1
        // Module 0: gene0(3) + gene1(2) + 0 = 5
        // Module 1: 0 + gene2(5) = 5
        let p0 = store.profile(0);
        assert!((p0[0] - 5.0).abs() < 1e-6);
        assert!((p0[1] - 5.0).abs() < 1e-6);

        // Edge (2,3): cell2 + cell3
        // Module 0: gene0(1) + gene1(6) = 7
        // Module 1: gene3(4) + 0 = 4
        let p1 = store.profile(1);
        assert!((p1[0] - 7.0).abs() < 1e-6);
        assert!((p1[1] - 4.0).abs() < 1e-6);
    }
}
