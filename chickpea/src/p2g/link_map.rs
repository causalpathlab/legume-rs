//! Peak–gene link scores inside a cis window, on pseudobulk profiles.
//!
//! Two scores, chosen by [`LinkScore`]:
//!
//! - **Pearson** (default): correlation of `log1p` RNA and ATAC pseudobulk
//!   profiles across the columns; positive correlations above the floor are
//!   kept.
//! - **ABC** (Engreitz): `A_p·C(d_pg) / Σ_{p'∈W(g)} A_p'·C(d_p'g)`, with `A_p`
//!   the mean pseudobulk accessibility over the columns and
//!   `C(d) = max(d, d0)^(−γ) + C(d1)` the power-law contact with its
//!   pseudocount. No RNA is read.
//!
//! Both are ranked per gene and cut to `top_k_per_gene` when set.

use crate::common::*;
use genomic_data::coordinates::{find_cis_peaks, GeneTss, PeakCoord};

/// One positive peak–gene edge with its score.
#[derive(Clone, Debug, PartialEq)]
pub struct PeakGeneEdge {
    pub peak: usize,
    pub gene: usize,
    pub weight: f32,
}

/// Which statistic scores a peak–gene pair.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[clap(rename_all = "lowercase")]
pub enum LinkScore {
    /// Pearson correlation of log1p pseudobulk profiles.
    #[default]
    Pearson,
    /// Engreitz ABC: activity × contact, normalized over the window.
    Abc,
}

/// Knobs for [`link_peaks_to_genes`].
#[derive(Clone, Debug)]
pub struct LinkParams {
    pub score: LinkScore,
    /// Cis window (bp) around each gene TSS (peak midpoint distance).
    pub cis_window: i64,
    /// Max cis candidates kept per gene after ranking (`0` = no cap).
    pub max_cis: usize,
    /// Drop edges with score ≤ this floor.
    pub min_weight: f32,
    /// Keep only the `k` best edges per gene after the floor (`0` = off).
    pub top_k_per_gene: usize,
    /// ABC: power-law contact exponent γ in `C(d) = d^(−γ)` (0.87).
    pub contact_gamma: f32,
    /// ABC: distances below this (bp) count as this distance (5 kb).
    pub contact_min_distance: i64,
    /// ABC: the contact at this distance (1 Mb) is added to every contact.
    pub contact_pseudocount_distance: i64,
}

impl Default for LinkParams {
    fn default() -> Self {
        Self {
            score: LinkScore::Pearson,
            cis_window: 500_000,
            max_cis: 200,
            min_weight: 0.0,
            top_k_per_gene: 0,
            contact_gamma: 0.87,
            contact_min_distance: 5_000,
            contact_pseudocount_distance: 1_000_000,
        }
    }
}

/// The power-law term alone, `max(d, d0)^(−γ)`.
#[inline]
fn contact_power(dist_bp: i64, params: &LinkParams) -> f32 {
    let floor = params.contact_min_distance.max(1);
    (dist_bp.abs().max(floor) as f32).powf(-params.contact_gamma)
}

/// ABC power-law contact with its pseudocount,
/// `max(d, d0)^(−γ) + max(d1, d0)^(−γ)`.
#[inline]
pub fn contact(dist_bp: i64, params: &LinkParams) -> f32 {
    contact_power(dist_bp, params) + contact_power(params.contact_pseudocount_distance, params)
}

/// Score every gene's cis peaks; keep scores above `min_weight`, rank per
/// gene, cut to `top_k_per_gene` and `max_cis`.
pub fn link_peaks_to_genes(
    rna_pb: &Mat,
    atac_pb: &Mat,
    gene_tss: &[Option<GeneTss>],
    peak_coords: &[Option<PeakCoord>],
    params: &LinkParams,
) -> anyhow::Result<Vec<PeakGeneEdge>> {
    let n_genes = rna_pb.nrows();
    let n_peaks = atac_pb.nrows();
    let n_samples = atac_pb.ncols();
    anyhow::ensure!(
        rna_pb.ncols() == n_samples,
        "RNA pb samples ({}) != ATAC pb samples ({n_samples})",
        rna_pb.ncols()
    );
    anyhow::ensure!(
        gene_tss.len() == n_genes,
        "gene_tss length {} != RNA rows {n_genes}",
        gene_tss.len()
    );
    anyhow::ensure!(
        peak_coords.len() == n_peaks,
        "peak_coords length {} != ATAC rows {n_peaks}",
        peak_coords.len()
    );
    // ABC's activity: mean accessibility of each peak over the columns.
    let activity: Vec<f32> = match params.score {
        LinkScore::Abc => atac_pb.column_mean().as_slice().to_vec(),
        LinkScore::Pearson => Vec::new(),
    };
    let pseudocount = contact(params.contact_pseudocount_distance, params)
        - contact_power(params.contact_pseudocount_distance, params);

    let mut edges = Vec::new();
    for g in 0..n_genes {
        let Some(tss) = gene_tss[g].as_ref() else {
            continue;
        };
        let cis = find_cis_peaks(tss, peak_coords, params.cis_window);
        if cis.is_empty() {
            continue;
        }
        let scored: Vec<(usize, f32)> = match params.score {
            LinkScore::Pearson => {
                let gene_log: Vec<f32> = (0..n_samples).map(|j| rna_pb[(g, j)].ln_1p()).collect();
                cis.iter()
                    .filter_map(|&p| {
                        let peak_log: Vec<f32> =
                            (0..n_samples).map(|j| atac_pb[(p, j)].ln_1p()).collect();
                        pearson(&gene_log, &peak_log).map(|r| (p, r))
                    })
                    .collect()
            }
            LinkScore::Abc => abc_shares(tss, &cis, peak_coords, &activity, pseudocount, params),
        };
        let mut gene_edges: Vec<PeakGeneEdge> = scored
            .into_iter()
            .filter(|&(_, w)| w > params.min_weight)
            .map(|(peak, weight)| PeakGeneEdge {
                peak,
                gene: g,
                weight,
            })
            .collect();
        gene_edges.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let cap = [params.top_k_per_gene, params.max_cis]
            .into_iter()
            .filter(|&k| k > 0)
            .min();
        if let Some(k) = cap {
            gene_edges.truncate(k);
        }
        edges.append(&mut gene_edges);
    }

    Ok(edges)
}

/// ABC shares of the cis peaks of one gene: `A·C / Σ A·C`, empty when
/// nothing in the window is active. `pseudocount` is the constant part of
/// the contact, computed once per call.
fn abc_shares(
    tss: &GeneTss,
    cis: &[usize],
    peak_coords: &[Option<PeakCoord>],
    activity: &[f32],
    pseudocount: f32,
    params: &LinkParams,
) -> Vec<(usize, f32)> {
    let mut shares: Vec<(usize, f32)> = cis
        .iter()
        .map(|&p| {
            let c = peak_coords[p].as_ref().expect("cis peak has coordinates");
            let d = (c.start + c.end) / 2 - tss.tss;
            (
                p,
                (contact_power(d, params) + pseudocount) * activity[p].max(0.0),
            )
        })
        .collect();
    let total: f32 = shares.iter().map(|&(_, v)| v).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    shares.iter_mut().for_each(|s| s.1 /= total);
    shares
}

/// Pearson correlation; `None` if either side has near-zero variance.
fn pearson(x: &[f32], y: &[f32]) -> Option<f32> {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if n < 2.0 {
        return None;
    }
    let mx = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let my = y.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = f64::from(xi) - mx;
        let dy = f64::from(yi) - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if dx2 <= 1e-12 || dy2 <= 1e-12 {
        return None;
    }
    Some((num / (dx2.sqrt() * dy2.sqrt())) as f32)
}
