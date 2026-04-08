use crate::common::*;
use candle_util::candle_core::{DType, Device, Tensor};
use genomic_data::coordinates::find_cis_peaks;
pub use genomic_data::coordinates::load_gene_tss;
pub use genomic_data::coordinates::GeneTss;

/// Load gene TSS positions from a simple TSV file (gene\tchr\ttss).
/// Produced by sim-link as {out}.gene_coords.tsv.gz.
pub fn load_gene_coords_tsv(
    path: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<Option<GeneTss>>> {
    use matrix_util::common_io::open_buf_reader;
    use std::io::BufRead;

    let reader = open_buf_reader(path)?;
    let mut tss_map: rustc_hash::FxHashMap<Box<str>, GeneTss> = Default::default();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }
        let gene: Box<str> = fields[0].into();
        let chr: Box<str> = fields[1].into();
        let tss: i64 = fields[2].parse()?;
        tss_map.insert(gene, GeneTss { chr, tss });
    }

    info!(
        "Loaded {} gene positions from {}, matching against {} genes",
        tss_map.len(),
        path,
        gene_names.len()
    );

    let result: Vec<Option<GeneTss>> = gene_names
        .iter()
        .map(|name| tss_map.get(name).cloned())
        .collect();

    let matched = result.iter().filter(|x| x.is_some()).count();
    info!(
        "Matched {}/{} genes to coordinates",
        matched,
        gene_names.len()
    );

    Ok(result)
}

/// Build cis-mask using genomic distance from gene TSS to peak midpoints.
pub fn build_cis_mask_by_distance(
    peak_coords: &[Option<genomic_data::coordinates::PeakCoord>],
    gene_tss: &[Option<GeneTss>],
    cis_window: i64,
    max_cis: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n_genes = gene_tss.len();
    let n_peaks = peak_coords.len();
    let c = max_cis.min(n_peaks);

    use rayon::prelude::*;

    // Per-gene: find cis-peaks, sort by distance, select top-c (parallel)
    let per_gene: Vec<(Vec<u32>, Vec<f32>)> = gene_tss
        .par_iter()
        .map(|gt| {
            let mut indices = Vec::with_capacity(c);
            let mut mask_vals = Vec::with_capacity(c);

            let mut candidates: Vec<(u32, i64)> = match gt {
                Some(tss) => find_cis_peaks(tss, peak_coords, cis_window)
                    .into_iter()
                    .map(|idx| {
                        let mid = peak_coords[idx]
                            .as_ref()
                            .map(|pc| (pc.start + pc.end) / 2)
                            .unwrap_or(0);
                        (idx as u32, (mid - tss.tss).abs())
                    })
                    .collect(),
                None => Vec::new(),
            };
            candidates.sort_unstable_by_key(|&(_, d)| d);

            for i in 0..c {
                if i < candidates.len() {
                    indices.push(candidates[i].0);
                    mask_vals.push(1.0);
                } else {
                    indices.push(0);
                    mask_vals.push(0.0);
                }
            }
            (indices, mask_vals)
        })
        .collect();

    let n_matched = gene_tss.iter().filter(|g| g.is_some()).count();
    let mut all_indices = Vec::with_capacity(n_genes * c);
    let mut all_mask = Vec::with_capacity(n_genes * c);
    for (idx, msk) in &per_gene {
        all_indices.extend_from_slice(idx);
        all_mask.extend_from_slice(msk);
    }

    info!(
        "Distance cis-mask: {} genes ({} with coords), {} candidates, ±{} bp window",
        n_genes, n_matched, c, cis_window
    );

    let cis_indices = Tensor::from_vec(all_indices, (n_genes, c), dev)?;
    let mask = Tensor::from_vec(all_mask, (n_genes, c), dev)?;
    Ok((cis_indices, mask))
}

/// Convert values to fractional ranks (average rank for ties).
fn rank_transform(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1.total_cmp(&indexed[i].1).is_eq() {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Rank-transform, center, and compute L2 norm.
fn rank_center_norm(row: &[f32]) -> (Vec<f32>, f32) {
    let ranked = rank_transform(row);
    let n = ranked.len() as f32;
    let mean: f32 = ranked.iter().sum::<f32>() / n;
    let centered: Vec<f32> = ranked.iter().map(|&v| v - mean).collect();
    let norm: f32 = centered.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    (centered, norm)
}

/// Build cis-mask by ranking peaks by absolute Spearman correlation with each gene.
pub fn build_cis_mask_by_correlation(
    rna_mat: &nalgebra::DMatrix<f32>,
    atac_mat: &nalgebra::DMatrix<f32>,
    max_c: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    use rayon::prelude::*;

    let n_genes = rna_mat.nrows();
    let n_peaks = atac_mat.nrows();
    let c = max_c.min(n_peaks);

    info!(
        "Spearman gene-peak correlations ({} x {} x {} samples)...",
        n_genes,
        n_peaks,
        rna_mat.ncols()
    );

    // Rank-transform and center ATAC peaks (parallel over peaks)
    let peak_data: Vec<(Vec<f32>, f32)> = (0..n_peaks)
        .into_par_iter()
        .map(|p| {
            let row: Vec<f32> = atac_mat.row(p).iter().copied().collect();
            rank_center_norm(&row)
        })
        .collect();

    // Per-gene: rank, correlate with all peaks, select top-c (parallel over genes)
    let all_indices: Vec<Vec<u32>> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let row: Vec<f32> = rna_mat.row(g).iter().copied().collect();
            let (centered, norm) = rank_center_norm(&row);

            let mut corrs: Vec<(u32, f32)> = (0..n_peaks)
                .map(|p| {
                    let (ref pc, pn) = peak_data[p];
                    let dot: f32 = centered.iter().zip(pc).map(|(a, b)| a * b).sum();
                    (p as u32, (dot / (norm * pn)).abs())
                })
                .collect();

            corrs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            corrs[..c].iter().map(|&(idx, _)| idx).collect()
        })
        .collect();

    let flat: Vec<u32> = all_indices.into_iter().flatten().collect();
    let cis_indices = Tensor::from_vec(flat, (n_genes, c), dev)?;
    let mask = Tensor::ones((n_genes, c), DType::F32, dev)?;
    info!("Cis-mask (Spearman): {} genes x {} candidates", n_genes, c);
    Ok((cis_indices, mask))
}
