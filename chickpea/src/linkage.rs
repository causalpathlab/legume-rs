use crate::common::*;
use candle_util::candle_core::{Device, IndexOp, Tensor};

/// Compute RNA dictionary W[G, K] from SuSiE M[G,C] and ATAC dictionary β[P,K].
/// W[g,k] = Σ_c exp(M[g,c]) · β[cis[g,c], k]
pub fn rna_dictionary_from_m(
    m_gc: &Tensor,
    log_beta: &Tensor,
    flat_cis_indices: &Tensor,
) -> candle_util::candle_core::Result<Tensor> {
    let (n_genes, c_max) = (m_gc.dim(0)?, m_gc.dim(1)?);
    let n_topics = log_beta.dim(1)?;
    let beta_gathered = log_beta.exp()?.i(flat_cis_indices)?;
    let beta_gathered = beta_gathered.reshape((n_genes, c_max, n_topics))?;
    m_gc.exp()?
        .unsqueeze(2)?
        .broadcast_mul(&beta_gathered)?
        .sum(1)
}

/// Precompute flattened indices for expanding coarsened SuSiE M to encoder weights.
/// Returns a [G*C_max] u32 tensor of indices into flattened m_coarse.
pub fn precompute_expand_indices(
    gene_members: &[usize],
    peak_members: &[usize],
    flat_cis_indices: &Tensor,
    n_genes: usize,
    c_max: usize,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let cis_raw: Vec<u32> = flat_cis_indices.to_vec1()?;
    let dp = peak_members.iter().max().map(|m| m + 1).unwrap_or(1);

    let mut flat_idx = Vec::with_capacity(n_genes * c_max);
    for g in 0..n_genes {
        let gm = gene_members[g];
        for c in 0..c_max {
            let peak_idx = cis_raw[g * c_max + c] as usize;
            let pm = peak_members[peak_idx];
            flat_idx.push((gm * dp + pm) as u32);
        }
    }

    Ok(Tensor::from_vec(flat_idx, n_genes * c_max, dev)?)
}

/// Save SuSiE linkage results (PIP, mean, var, peak indices) to parquet.
pub fn save_linkage_parquet(
    pip: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    cis_indices: &Tensor,
    path: &str,
) -> anyhow::Result<()> {
    let (n_genes, c_max) = (pip.dim(0)?, pip.dim(1)?);
    let pip_data: Vec<f32> = pip.flatten_all()?.to_vec1()?;
    let mean_data: Vec<f32> = mean.flatten_all()?.to_vec1()?;
    let var_data: Vec<f32> = var.flatten_all()?.to_vec1()?;
    let idx_data: Vec<u32> = cis_indices.flatten_all()?.to_vec1()?;

    let n_entries = n_genes * c_max;
    let mut mat = nalgebra::DMatrix::<f32>::zeros(n_entries, 5);
    for g in 0..n_genes {
        for c in 0..c_max {
            let i = g * c_max + c;
            mat[(i, 0)] = g as f32;
            mat[(i, 1)] = idx_data[i] as f32;
            mat[(i, 2)] = pip_data[i];
            mat[(i, 3)] = mean_data[i];
            mat[(i, 4)] = var_data[i];
        }
    }
    mat.to_parquet(path)?;
    info!("Wrote {} ({} entries)", path, n_entries);
    Ok(())
}
