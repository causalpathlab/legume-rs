//! Reference loading, per-gene stats, HVG selection, and PIT z-build for the
//! global copula structure used by `run_simulate --reference`.

use crate::copula::marginals::NbFit;
use data_beans::sparse_io::*;
use matrix_util::common_io::file_ext;
use matrix_util::sparse_stat::SparseRunningStatistics;
use nalgebra::DMatrix;

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
/// pass. The copula fit needs both, and reading the CSC twice was the
/// dominant cost on large references.
pub fn per_gene_stats_and_marginals(
    sc: &SparseRef,
    cells: &[usize],
    n_genes: usize,
    r_floor: f32,
) -> anyhow::Result<(Vec<GeneStats>, Vec<NbFit>)> {
    let stats = per_gene_stats(sc, cells, n_genes)?;
    let marginals = stats
        .iter()
        .map(|s| nb_fit_from_stats(s, r_floor))
        .collect();
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
/// dispersion-trend HVG ranker (`data_beans_alg::hvg`). It fits
/// `σ²(μ) = μ + φ(μ)·μ²` and ranks each gene by its excess dispersion above
/// the trend — the same `φ(μ)` fit that the rest of the workspace uses for
/// Fisher-info gene weighting and DC-Poisson clustering, so the copula sim
/// is consistent with how HVGs are picked in cocoa / pinto / senna.
pub fn select_hvg(stats: &[GeneStats], n_hvg: usize) -> Vec<usize> {
    let means: Vec<f32> = stats.iter().map(|s| s.mu as f32).collect();
    let vars: Vec<f32> = stats.iter().map(|s| s.var as f32).collect();
    data_beans_alg::hvg::select_hvg_by_stats(&means, &vars, n_hvg)
}

/// Build the (|hvg| × |cells|) PIT-then-`Φ⁻¹` Z matrix used to fit the
/// gene-gene Gaussian copula on the reference.
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
