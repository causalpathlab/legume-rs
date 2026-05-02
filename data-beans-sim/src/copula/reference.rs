//! Reference loading, per-gene stats, HVG selection, and SVD+leiden cell-type
//! inference for the copula simulator.

use crate::copula::marginals::NbFit;
use data_beans::sparse_io::*;
use log::info;
use matrix_util::common_io::{file_ext, open_buf_reader};
use matrix_util::knn_graph::{self, KnnGraph, KnnGraphArgs};
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::{MatOps, RandomizedAlgs};
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;
use std::io::BufRead;

/// Boxed dyn handle to whatever sparse backend the reference lives in.
pub type SparseRef = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

/// Open a `.h5`, `.zarr`, or `.zarr.zip` reference. The zip case is handled
/// natively by `data_beans::zarr_io` — no separate unzip step.
pub fn open_reference(path: &str) -> anyhow::Result<SparseRef> {
    let ext = file_ext(path)?.to_string();
    let backend = match ext.as_str() {
        "h5" => SparseIoBackend::HDF5,
        // `.zarr` directories report ext == "zarr"; `.zarr.zip` archives
        // report ext == "zip" — both go through the Zarr backend, which
        // detects and handles zip stores in `data_beans::zarr_io`.
        "zarr" | "zip" => SparseIoBackend::Zarr,
        other => anyhow::bail!(
            "unsupported reference extension '{}': expected h5 / zarr / zarr.zip",
            other
        ),
    };
    open_sparse_matrix(path, &backend)
}

/// Per-gene moment summary over a cell subset.
#[derive(Debug, Clone, Copy)]
pub struct GeneStats {
    pub mu: f64,
    pub var: f64,
    pub nnz: usize,
}

/// Per-gene mean / variance / nnz over a cell subset via
/// `matrix_util::sparse_stat::SparseRunningStatistics` (one CSC pass).
pub fn per_gene_stats(
    sc: &SparseRef,
    cells: &[usize],
    n_genes: usize,
) -> anyhow::Result<Vec<GeneStats>> {
    if cells.is_empty() {
        return Ok(vec![
            GeneStats {
                mu: 0.0,
                var: 0.0,
                nnz: 0,
            };
            n_genes
        ]);
    }
    let mat = sc.read_columns_csc(cells.to_vec())?;
    let mut acc = SparseRunningStatistics::<f64>::new(n_genes);
    for col in mat.col_iter() {
        let vals_f64: Vec<f64> = col.values().iter().map(|&v| v as f64).collect();
        acc.add_sparse_column(col.row_indices(), &vals_f64);
    }
    // Pad implicit-zero columns so the mean/variance denominators reflect
    // every requested cell, not just CSC columns that contain at least one nz.
    for _ in mat.ncols()..cells.len() {
        acc.add_sparse_column(&[], &[]);
    }
    let (npos, _sum, mean, std) = acc.to_vecs();
    Ok((0..n_genes)
        .map(|g| GeneStats {
            mu: mean[g],
            var: std[g] * std[g],
            nnz: npos[g] as usize,
        })
        .collect())
}

/// Combined per-gene stats + NB MoM fits — produces both from a single CSC
/// pass. The per-cluster fit needs both, and reading the CSC twice was the
/// dominant `fit_copula` cost on large references.
pub fn per_gene_stats_and_marginals(
    sc: &SparseRef,
    cells: &[usize],
    n_genes: usize,
    r_floor: f32,
) -> anyhow::Result<(Vec<GeneStats>, Vec<NbFit>)> {
    let stats = per_gene_stats(sc, cells, n_genes)?;
    let marginals = stats.iter().map(|s| nb_fit_from_stats(s, r_floor)).collect();
    Ok((stats, marginals))
}

fn nb_fit_from_stats(s: &GeneStats, r_floor: f32) -> NbFit {
    let mu = s.mu as f32;
    let var = s.var as f32;
    if mu <= 0.0 || var <= mu {
        return NbFit {
            mu,
            r: f32::INFINITY,
        };
    }
    let r = (mu * mu) / (var - mu);
    NbFit {
        mu,
        r: r.max(r_floor),
    }
}

/// Select up to `n_hvg` highly variable genes via the workspace's NB
/// dispersion-trend HVG ranker (`data_beans_alg::hvg_selection`). It fits
/// `σ²(μ) = μ + φ(μ)·μ²` and ranks each gene by its excess dispersion above
/// the trend — the same `φ(μ)` fit that the rest of the workspace uses for
/// Fisher-info gene weighting and DC-Poisson clustering, so the copula sim
/// is consistent with how HVGs are picked in cocoa / pinto / senna.
pub fn select_hvg(stats: &[GeneStats], n_hvg: usize) -> Vec<usize> {
    let means: Vec<f32> = stats.iter().map(|s| s.mu as f32).collect();
    let vars: Vec<f32> = stats.iter().map(|s| s.var as f32).collect();
    data_beans_alg::hvg_selection::select_hvg_by_stats(&means, &vars, n_hvg)
}

/// Build a (n_cells × |hvg|) dense embedding: `log1p(X[hvg, :])ᵀ` then column
/// z-score, then RSVD to `rank` components projected as `U · diag(σ)`.
pub fn cell_embedding(sc: &SparseRef, hvg: &[usize], rank: usize) -> anyhow::Result<DMatrix<f32>> {
    let csc = sc.read_rows_csc(hvg.to_vec())?;
    let n_hvg = csc.nrows();
    let n_cells = csc.ncols();
    let row_idx = csc.row_indices();
    let col_off = csc.col_offsets();
    let vals = csc.values();
    let mut dense = DMatrix::<f32>::zeros(n_cells, n_hvg);
    for c in 0..n_cells {
        let s = col_off[c];
        let e = col_off[c + 1];
        for k in s..e {
            let r = row_idx[k];
            dense[(c, r)] = (vals[k] + 1.0).ln();
        }
    }
    dense.scale_columns_inplace();
    let (u, sigmas, _vt) = dense.rsvd(rank)?;
    let r_eff = u.ncols().min(sigmas.len());
    let mut emb = u;
    for r in 0..r_eff {
        let s = sigmas[r];
        for i in 0..emb.nrows() {
            emb[(i, r)] *= s;
        }
    }
    Ok(emb)
}

/// Cluster cells with KNN-graph leiden on the embedding. `target_clusters`
/// is the requested cluster count — the leiden resolution is binary-searched
/// to land near it (matches `senna::cluster::leiden_clustering`'s convention).
pub fn cluster_cells(
    embedding: &DMatrix<f32>,
    knn: usize,
    target_clusters: usize,
    seed: u64,
) -> anyhow::Result<Vec<usize>> {
    let mut e = embedding.clone();
    e.scale_columns_inplace();
    let n = e.nrows();
    let graph = KnnGraph::from_rows(
        &e,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    let (network, total_edge_weight) = graph.to_leiden_network();
    let starting = knn_graph::modularity_to_cpm_resolution(1.0, total_edge_weight);
    info!(
        "KNN graph: {} nodes, {} edges; tuning leiden to ~{} clusters",
        graph.num_nodes(),
        graph.num_edges(),
        target_clusters
    );
    let mut labels = if target_clusters > 0 {
        knn_graph::tune_leiden_resolution(
            &network,
            n,
            target_clusters,
            starting,
            Some(seed as usize),
        )
    } else {
        knn_graph::run_leiden(&network, n, starting, Some(seed as usize))
    };
    knn_graph::compact_labels(&mut labels);
    Ok(labels)
}

/// Read a per-cell label file (`barcode<TAB>label`, optionally gzipped) and
/// align it to the reference's column names. Cells absent from the file get
/// `usize::MAX` and are dropped from downstream fits.
pub fn load_cell_type_labels(
    path: &str,
    column_names: &[Box<str>],
) -> anyhow::Result<(Vec<usize>, Vec<Box<str>>)> {
    let mut barcode_to_label: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    let reader = open_buf_reader(path)?;
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.splitn(2, '\t');
        let bc = it.next().unwrap_or("").trim().to_string();
        let lab = it.next().unwrap_or("").trim().to_string();
        if bc.is_empty() || lab.is_empty() {
            continue;
        }
        barcode_to_label.insert(bc.into_boxed_str(), lab.into_boxed_str());
    }
    let mut label_id: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut label_order: Vec<Box<str>> = Vec::new();
    let mut labels = Vec::with_capacity(column_names.len());
    let mut matched = 0usize;
    for bc in column_names {
        if let Some(lab) = barcode_to_label.get(bc) {
            let id = if let Some(&id) = label_id.get(lab) {
                id
            } else {
                let id = label_order.len();
                label_order.push(lab.clone());
                label_id.insert(lab.clone(), id);
                id
            };
            labels.push(id);
            matched += 1;
        } else {
            labels.push(usize::MAX);
        }
    }
    info!(
        "cell-type labels: {} / {} cells matched ({} clusters)",
        matched,
        column_names.len(),
        label_order.len()
    );
    Ok((labels, label_order))
}

/// Group cells by label, dropping any with `usize::MAX`. Returns
/// `Vec<(label_id, Vec<cell_indices>)>`.
pub fn partition_by_label(labels: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (i, &l) in labels.iter().enumerate() {
        if l == usize::MAX {
            continue;
        }
        groups.entry(l).or_default().push(i);
    }
    let mut out: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    out.sort_by_key(|(l, _)| *l);
    out
}

/// Build the (|hvg| × |cells|) PIT-then-`Φ⁻¹` Z matrix for a cluster.
pub fn build_z_matrix(
    sc: &SparseRef,
    cells: &[usize],
    hvg: &[usize],
    fits: &[NbFit],
) -> anyhow::Result<DMatrix<f32>> {
    use crate::copula::marginals::{inv_phi, nb_cdf_table, pit_continuity};
    let n_hvg = hvg.len();
    let n = cells.len();
    let mat = sc.read_rows_csc(hvg.to_vec())?;
    // mat: (|hvg| × |all_cells|). We need only the columns in `cells`.
    let col_off = mat.col_offsets();
    let row_idx = mat.row_indices();
    let vals = mat.values();
    // Pre-build CDF tables per HVG gene; cap k at the max observed value for that gene.
    let mut max_val: Vec<u32> = vec![0; n_hvg];
    for k in 0..vals.len() {
        let h = row_idx[k];
        let v = vals[k] as u32;
        if v > max_val[h] {
            max_val[h] = v;
        }
    }
    let cdf_tables: Vec<Vec<f64>> = (0..n_hvg)
        .map(|h| {
            let fit = fits[hvg[h]];
            nb_cdf_table(fit, (max_val[h] as usize).max(1))
        })
        .collect();
    // U for value=0 is the same for all zero cells of a given gene; precompute.
    let u_zero: Vec<f64> = cdf_tables.iter().map(|t| pit_continuity(t, 0)).collect();
    let z_zero: Vec<f32> = u_zero.iter().map(|&u| inv_phi(u) as f32).collect();
    let mut z = DMatrix::<f32>::zeros(n_hvg, n);
    for (col_out, &cell) in cells.iter().enumerate() {
        // Initialize the column to the zero-value Z for each gene.
        for h in 0..n_hvg {
            z[(h, col_out)] = z_zero[h];
        }
        // Then overwrite the nonzero entries from the CSC slice for this cell.
        let s = col_off[cell];
        let e = col_off[cell + 1];
        for k in s..e {
            let h = row_idx[k];
            let v = vals[k] as u32;
            let u = pit_continuity(&cdf_tables[h], v);
            z[(h, col_out)] = inv_phi(u) as f32;
        }
    }
    Ok(z)
}
