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
    use rayon::prelude::*;

    let cis_raw: Vec<u32> = flat_cis_indices.to_vec1()?;
    let dp = peak_members.iter().max().map(|m| m + 1).unwrap_or(1);

    let cis_ref = &cis_raw;
    let flat_idx: Vec<u32> = (0..n_genes)
        .into_par_iter()
        .flat_map_iter(|g| {
            let gm = gene_members[g];
            let base = g * c_max;
            (0..c_max).map(move |c| {
                let peak_idx = cis_ref[base + c] as usize;
                let pm = peak_members[peak_idx];
                (gm * dp + pm) as u32
            })
        })
        .collect();

    Ok(Tensor::from_vec(flat_idx, n_genes * c_max, dev)?)
}

/// Feature annotations for BED output.
pub struct FeatureAnnotations<'a> {
    pub gene_names: &'a [Box<str>],
    pub peak_names: &'a [Box<str>],
    pub peak_coords: &'a [Option<genomic_data::coordinates::PeakCoord>],
}

/// Save SuSiE linkage results as a sorted BED TSV (.results.bed.gz).
///
/// Columns: `#chr start end peak_id gene_id pip effect_mean effect_std`
///
/// Rows are sorted by (chr, start, end). Output is BGZF-compressed for tabix indexing.
pub fn save_linkage_results(
    pip: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    cis_indices: &Tensor,
    features: &FeatureAnnotations,
    path: &str,
) -> anyhow::Result<()> {
    use rust_htslib::bgzf;
    use std::io::Write;

    let (n_genes, c_max) = (pip.dim(0)?, pip.dim(1)?);
    let pip_data: Vec<f32> = pip.flatten_all()?.to_vec1()?;
    let mean_data: Vec<f32> = mean.flatten_all()?.to_vec1()?;
    let var_data: Vec<f32> = var.flatten_all()?.to_vec1()?;
    let idx_data: Vec<u32> = cis_indices.flatten_all()?.to_vec1()?;

    // Build rows, then sort by genomic position
    struct Row {
        chr: Box<str>,
        start: i64,
        end: i64,
        peak_id: Box<str>,
        gene_id: Box<str>,
        pip: f32,
        effect_mean: f32,
        effect_std: f32,
    }

    let mut rows: Vec<Row> = Vec::with_capacity(n_genes * c_max);
    for (g, gene_name) in features.gene_names.iter().enumerate().take(n_genes) {
        for c in 0..c_max {
            let i = g * c_max + c;
            let p_idx = idx_data[i] as usize;
            let (chr, start, end) = match features.peak_coords.get(p_idx).and_then(|c| c.as_ref()) {
                Some(coord) => (coord.chr.clone(), coord.start, coord.end),
                None => (".".into(), 0i64, 0i64),
            };
            rows.push(Row {
                chr,
                start,
                end,
                peak_id: features.peak_names[p_idx].clone(),
                gene_id: gene_name.clone(),
                pip: pip_data[i],
                effect_mean: mean_data[i],
                effect_std: var_data[i].sqrt(),
            });
        }
    }

    rows.sort_by(|a, b| (&*a.chr, a.start, a.end).cmp(&(&*b.chr, b.start, b.end)));

    let mut writer = bgzf::Writer::from_path(path)?;
    writeln!(
        writer,
        "#chr\tstart\tend\tpeak_id\tgene_id\tpip\teffect_mean\teffect_std"
    )?;
    for r in &rows {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}",
            r.chr, r.start, r.end, r.peak_id, r.gene_id, r.pip, r.effect_mean, r.effect_std,
        )?;
    }

    writer.flush()?;
    info!("Wrote {} ({} entries)", path, rows.len());
    Ok(())
}
