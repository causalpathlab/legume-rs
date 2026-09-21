//! ArchR-inspired gene activity from ATAC peaks (ATAC-only RNA surrogate).
//!
//! Peak midpoints are scored like ArchR tiles: distance-weighted sum of
//! accessibility to an (optionally upstream-extended) gene body, with a
//! small distal floor and optional clipping at neighboring gene boundaries.
//!
//! Default weight model matches ArchR's `geneModel`:
//! `exp(-abs(d) / decay) + exp(-1)` with `decay = 5000`.
//! See <https://www.archrproject.com/bookdown/calculating-gene-scores-in-archr.html>.

use crate::common::*;
use genomic_data::coordinates::{chr_eq, chr_stripped, GeneLoc, PeakCoord};
use genomic_data::sam::Strand;

/// Parameters for ArchR-style gene activity (peak-level approximation).
#[derive(Debug, Clone)]
pub struct GeneActivityParams {
    /// Max distance (bp) from the extended gene body to include a peak.
    /// ArchR default window is 100 kb.
    pub window: i64,
    /// Exponential decay lengthscale (bp). ArchR default 5000.
    pub decay: f32,
    /// Extend gene body this many bp upstream of the TSS (ArchR `geneUpstream`).
    pub gene_upstream: i64,
    /// Extend gene body this many bp downstream of the TTS (ArchR `geneDownstream`).
    pub gene_downstream: i64,
    /// Cap on inverse-length gene-size weight (ArchR `geneScaleFactor`, default 5).
    pub gene_scale_factor: f32,
    /// Clip each gene's search window so it does not cross neighbors.
    pub use_gene_boundaries: bool,
}

impl Default for GeneActivityParams {
    fn default() -> Self {
        Self {
            window: 100_000,
            decay: 5_000.0,
            gene_upstream: 5_000,
            gene_downstream: 0,
            gene_scale_factor: 5.0,
            use_gene_boundaries: true,
        }
    }
}

/// ArchR distance weight: `exp(-|d|/decay) + exp(-1)`.
#[inline]
pub fn archr_distance_weight(dist_bp: i64, decay: f32) -> f32 {
    let d = (dist_bp as f32).abs();
    (-d / decay).exp() + (-1.0f32).exp()
}

/// Distance from `pos` to closed interval `[start, end]` (0 if inside).
#[inline]
fn dist_to_interval(pos: i64, start: i64, end: i64) -> i64 {
    if pos < start {
        start - pos
    } else if pos > end {
        pos - end
    } else {
        0
    }
}

/// Extended gene body used for distance = 0 (promoter + body), strand-aware.
fn extended_body(gene: &GeneLoc, params: &GeneActivityParams) -> (i64, i64) {
    let mut start = gene.start;
    let mut end = gene.end;
    match gene.strand {
        Strand::Forward => {
            start -= params.gene_upstream;
            end += params.gene_downstream;
        }
        Strand::Backward => {
            end += params.gene_upstream;
            start -= params.gene_downstream;
        }
    }
    (start, end)
}

/// Build genes × samples activity by ArchR-weighted sum of cis peak rates.
///
/// Rows align with `genes`. Genes without coordinates (or with no peaks in
/// window) get an all-zero row.
pub fn gene_activity_from_atac_pb(
    atac_pb: &Mat,
    peak_coords: &[Option<PeakCoord>],
    genes: &[Option<GeneLoc>],
    params: &GeneActivityParams,
) -> anyhow::Result<Mat> {
    let n_peaks = atac_pb.nrows();
    let n_samples = atac_pb.ncols();
    let n_genes = genes.len();
    anyhow::ensure!(
        peak_coords.len() == n_peaks,
        "peak_coords length {} != ATAC rows {n_peaks}",
        peak_coords.len()
    );
    anyhow::ensure!(n_samples > 0, "ATAC pb has no samples");
    anyhow::ensure!(params.window > 0, "gene-activity window must be > 0");
    anyhow::ensure!(params.decay > 0.0, "decay must be > 0");
    anyhow::ensure!(
        params.gene_scale_factor >= 1.0,
        "gene_scale_factor must be ≥ 1"
    );

    // Per-chromosome gene order for boundary clipping.
    let mut by_chr: rustc_hash::FxHashMap<Box<str>, Vec<usize>> = Default::default();
    for (g, gene) in genes.iter().enumerate() {
        let Some(gene) = gene.as_ref() else {
            continue;
        };
        by_chr
            .entry(chr_stripped(gene.chr.as_ref()).into())
            .or_default()
            .push(g);
    }
    for idxs in by_chr.values_mut() {
        idxs.sort_by_key(|&g| genes[g].as_ref().unwrap().start);
    }

    // Search window per gene (clipped by neighbors when requested).
    let mut windows: Vec<Option<(i64, i64)>> = vec![None; n_genes];
    for idxs in by_chr.values() {
        for (pos, &g) in idxs.iter().enumerate() {
            let gene = genes[g].as_ref().unwrap();
            let (body_s, body_e) = extended_body(gene, params);
            let mut left = body_s - params.window;
            let mut right = body_e + params.window;
            if params.use_gene_boundaries {
                if pos > 0 {
                    let prev = genes[idxs[pos - 1]].as_ref().unwrap();
                    let (_, prev_e) = extended_body(prev, params);
                    left = left.max(prev_e);
                }
                if pos + 1 < idxs.len() {
                    let next = genes[idxs[pos + 1]].as_ref().unwrap();
                    let (next_s, _) = extended_body(next, params);
                    right = right.min(next_s);
                }
            }
            if left < right {
                windows[g] = Some((left, right));
            }
        }
    }

    // Inverse gene-length weights scaled into [1, gene_scale_factor].
    let mut inv_len = vec![0.0f32; n_genes];
    let mut min_inv = f32::INFINITY;
    let mut max_inv = 0.0f32;
    for (g, gene) in genes.iter().enumerate() {
        let Some(gene) = gene.as_ref() else {
            continue;
        };
        let len = (gene.end - gene.start + 1).max(1) as f32;
        let inv = 1.0 / len;
        inv_len[g] = inv;
        min_inv = min_inv.min(inv);
        max_inv = max_inv.max(inv);
    }
    let span = (max_inv - min_inv).max(0.0);
    let size_w: Vec<f32> = (0..n_genes)
        .map(|g| {
            if inv_len[g] <= 0.0 {
                return 1.0;
            }
            if span <= 0.0 {
                return 1.0;
            }
            1.0 + (params.gene_scale_factor - 1.0) * (inv_len[g] - min_inv) / span
        })
        .collect();

    let mut out = Mat::zeros(n_genes, n_samples);
    for (g, gene) in genes.iter().enumerate() {
        let Some(gene) = gene.as_ref() else {
            continue;
        };
        let Some((win_l, win_r)) = windows[g] else {
            continue;
        };
        let (body_s, body_e) = extended_body(gene, params);
        let sw = size_w[g];

        for (p, coord) in peak_coords.iter().enumerate() {
            let Some(coord) = coord.as_ref() else {
                continue;
            };
            if !chr_eq(coord.chr.as_ref(), gene.chr.as_ref()) {
                continue;
            }
            let mid = (coord.start + coord.end) / 2;
            if mid < win_l || mid > win_r {
                continue;
            }
            let dist = dist_to_interval(mid, body_s, body_e);
            let w = sw * archr_distance_weight(dist, params.decay);
            if w <= 0.0 {
                continue;
            }
            for j in 0..n_samples {
                out[(g, j)] += w * atac_pb[(p, j)];
            }
        }
    }
    Ok(out)
}

/// TSS-only fallback when gene body is unknown (e.g. gene-coords TSV).
pub fn gene_loc_from_tss(tss: &genomic_data::coordinates::GeneTss) -> GeneLoc {
    GeneLoc {
        chr: tss.chr.clone(),
        start: tss.tss,
        end: tss.tss,
        tss: tss.tss,
        strand: Strand::Forward,
    }
}
