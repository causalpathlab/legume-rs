//! Rough ABC / pb co-occurrence peak–gene edge map.
//!
//! Builds positive peak–gene edges for `graph-embedding-util` training from
//! pseudobulk RNA × ATAC co-occurrence inside a cis window.

use crate::common::*;
use genomic_data::coordinates::{find_cis_peaks, GeneTss, PeakCoord};

/// One positive peak–gene edge with a co-occurrence weight.
#[derive(Clone, Debug, PartialEq)]
pub struct PeakGeneEdge {
    pub peak: usize,
    pub gene: usize,
    pub weight: f32,
}

/// Knobs for [`rough_abc_map`].
#[derive(Clone, Debug)]
pub struct AbcMapParams {
    /// Cis window (bp) around each gene TSS (peak midpoint distance).
    pub cis_window: i64,
    /// Max cis candidates kept per gene after ranking by weight (`0` = no cap).
    pub max_cis: usize,
    /// Drop edges with weight ≤ this floor.
    pub min_weight: f32,
}

impl Default for AbcMapParams {
    fn default() -> Self {
        Self {
            cis_window: 500_000,
            max_cis: 200,
            min_weight: 0.0,
        }
    }
}

/// Rough ABC-style map: for each gene, score cis peaks by Pearson correlation
/// of log1p pb profiles; keep positive weights above `min_weight`, optionally
/// truncated to the top `max_cis` per gene.
pub fn rough_abc_map(
    rna_pb: &Mat,
    atac_pb: &Mat,
    gene_tss: &[Option<GeneTss>],
    peak_coords: &[Option<PeakCoord>],
    params: &AbcMapParams,
) -> anyhow::Result<Vec<PeakGeneEdge>> {
    let n_genes = rna_pb.nrows();
    let n_peaks = atac_pb.nrows();
    let n_samples = rna_pb.ncols();
    anyhow::ensure!(
        atac_pb.ncols() == n_samples,
        "RNA pb samples ({n_samples}) != ATAC pb samples ({})",
        atac_pb.ncols()
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
    anyhow::ensure!(n_samples >= 2, "need ≥2 pb samples for correlation");

    let mut edges = Vec::new();
    for g in 0..n_genes {
        let Some(tss) = gene_tss[g].as_ref() else {
            continue;
        };
        let cis = find_cis_peaks(tss, peak_coords, params.cis_window);
        if cis.is_empty() {
            continue;
        }

        let gene_log: Vec<f32> = (0..n_samples).map(|j| rna_pb[(g, j)].ln_1p()).collect();

        let mut gene_edges: Vec<PeakGeneEdge> = cis
            .iter()
            .filter_map(|&p| {
                let peak_log: Vec<f32> = (0..n_samples).map(|j| atac_pb[(p, j)].ln_1p()).collect();
                let r = pearson(&gene_log, &peak_log)?;
                if r > params.min_weight {
                    Some(PeakGeneEdge {
                        peak: p,
                        gene: g,
                        weight: r,
                    })
                } else {
                    None
                }
            })
            .collect();

        gene_edges.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if params.max_cis > 0 && gene_edges.len() > params.max_cis {
            gene_edges.truncate(params.max_cis);
        }
        edges.append(&mut gene_edges);
    }

    Ok(edges)
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

#[cfg(test)]
mod tests {
    use super::*;
    use genomic_data::coordinates::{GeneTss, PeakCoord};

    /// Causal peak co-varies with its gene; a cis bystander does not.
    /// The causal edge must outrank the bystander; a far peak gets no edge.
    #[test]
    fn causal_cis_peak_outranks_bystander() {
        let s = 40usize;
        let mut rna = Mat::zeros(1, s);
        let mut atac = Mat::zeros(3, s);
        for j in 0..s {
            let z = (j as f32) * 0.1;
            let signal = z.sin().max(0.0) + 0.05;
            rna[(0, j)] = signal * 10.0;
            atac[(0, j)] = signal * 8.0 + 0.01; // causal
            atac[(1, j)] = ((j % 7) as f32) * 0.3; // cis bystander, uncorrelated
            atac[(2, j)] = signal * 9.0; // same signal but far away
        }

        let gene_tss = vec![Some(GeneTss {
            chr: "1".into(),
            tss: 100_000,
        })];
        let peak_coords = vec![
            Some(PeakCoord {
                chr: "1".into(),
                start: 100_000,
                end: 100_500,
            }),
            Some(PeakCoord {
                chr: "1".into(),
                start: 120_000,
                end: 120_500,
            }),
            Some(PeakCoord {
                chr: "1".into(),
                start: 5_000_000,
                end: 5_000_500,
            }),
        ];
        let params = AbcMapParams {
            cis_window: 500_000,
            max_cis: 50,
            min_weight: 0.0,
        };

        let edges = rough_abc_map(&rna, &atac, &gene_tss, &peak_coords, &params).unwrap();
        assert!(!edges.is_empty(), "expected at least the causal cis edge");
        assert!(
            edges.iter().all(|e| e.gene == 0 && e.peak != 2),
            "far peak must not appear; got {edges:?}"
        );
        let w0 = edges
            .iter()
            .find(|e| e.peak == 0)
            .expect("causal peak edge")
            .weight;
        let w1 = edges.iter().find(|e| e.peak == 1).map(|e| e.weight);
        assert!(w0 > 0.5, "causal weight too small: {w0}");
        if let Some(w1) = w1 {
            assert!(w0 > w1, "causal {w0} should beat bystander {w1}");
        }
    }
}
