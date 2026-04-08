use crate::common::*;
use candle_util::candle_core::{DType, Device, Tensor};
pub use genomic_data::coordinates::load_gene_tss;
pub use genomic_data::coordinates::GeneTss;
use genomic_data::coordinates::{find_cis_peaks, parse_peak_coordinates};

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
    peak_names: &[Box<str>],
    gene_tss: &[Option<GeneTss>],
    cis_window: i64,
    max_cis: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n_genes = gene_tss.len();
    let n_peaks = peak_names.len();
    let c = max_cis.min(n_peaks);

    let peak_coords = parse_peak_coordinates(peak_names);

    let mut all_indices: Vec<u32> = Vec::with_capacity(n_genes * c);
    let mut all_mask: Vec<f32> = Vec::with_capacity(n_genes * c);

    let mut n_matched = 0usize;
    for gt in gene_tss.iter() {
        let mut candidates: Vec<(u32, i64)> = match gt {
            Some(tss) => {
                n_matched += 1;
                find_cis_peaks(tss, &peak_coords, cis_window)
                    .into_iter()
                    .map(|idx| {
                        let mid = peak_coords[idx]
                            .as_ref()
                            .map(|c| (c.start + c.end) / 2)
                            .unwrap_or(0);
                        (idx as u32, (mid - tss.tss).abs())
                    })
                    .collect()
            }
            None => Vec::new(),
        };

        candidates.sort_unstable_by_key(|&(_, d)| d);

        for i in 0..c {
            if i < candidates.len() {
                all_indices.push(candidates[i].0);
                all_mask.push(1.0);
            } else {
                all_indices.push(0);
                all_mask.push(0.0);
            }
        }
    }

    info!(
        "Distance cis-mask: {} genes ({} with coords), {} candidates, ±{} bp window",
        n_genes, n_matched, c, cis_window
    );

    let cis_indices = Tensor::from_vec(all_indices, (n_genes, c), dev)?;
    let mask = Tensor::from_vec(all_mask, (n_genes, c), dev)?;
    Ok((cis_indices, mask))
}

/// Build cis-mask by ranking peaks by absolute Pearson correlation with each gene.
pub fn build_cis_mask_by_correlation(
    rna_mat: &nalgebra::DMatrix<f32>,
    atac_mat: &nalgebra::DMatrix<f32>,
    max_c: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n_genes = rna_mat.nrows();
    let n_peaks = atac_mat.nrows();
    let n_samples = rna_mat.ncols();
    let c = max_c.min(n_peaks);

    info!(
        "Gene-peak correlations ({} x {} x {} samples)...",
        n_genes, n_peaks, n_samples
    );

    let mut peak_centered: Vec<Vec<f32>> = Vec::with_capacity(n_peaks);
    let mut peak_norms: Vec<f32> = Vec::with_capacity(n_peaks);
    for p in 0..n_peaks {
        let row: Vec<f32> = atac_mat.row(p).iter().copied().collect();
        let mean: f32 = row.iter().sum::<f32>() / n_samples as f32;
        let centered: Vec<f32> = row.iter().map(|&v| v - mean).collect();
        let norm: f32 = centered.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        peak_centered.push(centered);
        peak_norms.push(norm);
    }

    let mut all_indices: Vec<u32> = Vec::with_capacity(n_genes * c);

    for g in 0..n_genes {
        let row: Vec<f32> = rna_mat.row(g).iter().copied().collect();
        let mean: f32 = row.iter().sum::<f32>() / n_samples as f32;
        let centered: Vec<f32> = row.iter().map(|&v| v - mean).collect();
        let norm: f32 = centered.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);

        let mut corrs: Vec<(u32, f32)> = (0..n_peaks)
            .map(|p| {
                let dot: f32 = centered
                    .iter()
                    .zip(&peak_centered[p])
                    .map(|(a, b)| a * b)
                    .sum();
                (p as u32, (dot / (norm * peak_norms[p])).abs())
            })
            .collect();

        corrs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all_indices.extend(corrs[..c].iter().map(|&(idx, _)| idx));
    }

    let cis_indices = Tensor::from_vec(all_indices, (n_genes, c), dev)?;
    let mask = Tensor::ones((n_genes, c), DType::F32, dev)?;
    info!("Cis-mask: {} genes x {} candidates", n_genes, c);
    Ok((cis_indices, mask))
}
