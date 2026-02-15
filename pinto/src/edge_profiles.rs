#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! Link profile construction, coarsening, and projection refinement.
//!
//! Builds projected link profiles from sparse expression data and KNN edges,
//! coarsens them by cell-level cluster labels, and refines the projection
//! basis via community centroids.

use crate::link_community_model::LinkProfileStore;
use crate::srt_common::*;
use matrix_util::utils::generate_minibatch_intervals;

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

            // Collect unique cell indices from this chunk
            let mut unique_set: HashSet<usize> = HashSet::new();
            for &(i, j) in chunk_edges {
                unique_set.insert(i);
                unique_set.insert(j);
            }
            let mut unique_cells: Vec<usize> = unique_set.into_iter().collect();
            unique_cells.sort_unstable();

            // Read sparse data for unique cells
            let x_dn = data.read_columns_csc(unique_cells.iter().copied())?;

            // Build cell → local column index map
            let cell_to_col: HashMap<usize, usize> = unique_cells
                .iter()
                .enumerate()
                .map(|(col, &cell)| (cell, col))
                .collect();

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

            // Collect unique cells
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

            // Collect unique cell indices from this chunk
            let mut unique_set: HashSet<usize> = HashSet::new();
            for &(i, j) in chunk_edges {
                unique_set.insert(i);
                unique_set.insert(j);
            }
            let mut unique_cells: Vec<usize> = unique_set.into_iter().collect();
            unique_cells.sort_unstable();

            // Read sparse data for unique cells
            let x_dn = data.read_columns_csc(unique_cells.iter().copied())?;

            // Build cell → local column index map
            let cell_to_col: HashMap<usize, usize> = unique_cells
                .iter()
                .enumerate()
                .map(|(col, &cell)| (cell, col))
                .collect();

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

        let embed = compute_gene_module_sketch(&data, &cell_labels, 2, 10, 3).unwrap();

        assert_eq!(embed.nrows(), n_genes);
        assert_eq!(embed.ncols(), 10);

        // Gene 2 has no counts — its embedding should be zero
        for d in 0..10 {
            assert!((embed[(2, d)]).abs() < 1e-10);
        }

        // Genes in the same cluster should have similar embeddings
        // (both project through R[cluster_0,:])
        let dot_01: f32 = (0..10).map(|d| embed[(0, d)] * embed[(1, d)]).sum();
        let dot_04: f32 = (0..10).map(|d| embed[(0, d)] * embed[(4, d)]).sum();

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
