//! Global, per-modality bandwidth estimation for editing-site component calling.
//!
//! The per-gene mixture (see [`crate::editing::mixture`]) smooths each gene's
//! site pileup at a Gaussian bandwidth and reads off the modes as components.
//! That bandwidth is the *resolution* of the call — it must reflect the
//! modality's intrinsic spatial scale (m6A: tight, single-site; A-to-I: wider,
//! Alu hyper-editing clusters), NOT the gene length.
//!
//! We estimate it once per modality from the empirical distribution of
//! within-gene nearest-neighbour site gaps, weighted by per-site signal so that
//! marginal 1-read sites do not set the scale. See the approved plan
//! `bandwidth-first component calling` for the rationale.

/// Inputs and tuning for [`estimate_bandwidth`].
pub struct BandwidthParams {
    /// Minimum distinct sites in a gene for it to contribute gaps.
    pub min_sites: usize,
    /// Multiplier applied to the pooled median gap. m6A uses ~1 (tight,
    /// resolve individual sites); A-to-I uses a larger value so each Alu
    /// hyper-editing island collapses to one cluster component.
    pub scale: f32,
    /// Hard floor on the returned bandwidth (nt).
    pub min_bandwidth: f32,
    /// Hard ceiling on the returned bandwidth (nt).
    pub max_bandwidth: f32,
}

impl BandwidthParams {
    /// Tight resolution for sparse, motif-driven m6A sites.
    pub fn m6a() -> Self {
        Self {
            min_sites: 2,
            scale: 1.0,
            min_bandwidth: 10.0,
            max_bandwidth: 200.0,
        }
    }

    /// Cluster-aware resolution for A-to-I: a wider bandwidth merges the
    /// dense within-Alu editing sites into one component.
    pub fn atoi() -> Self {
        Self {
            min_sites: 2,
            scale: 3.0,
            min_bandwidth: 30.0,
            max_bandwidth: 600.0,
        }
    }
}

/// Result of bandwidth estimation.
pub struct BandwidthEstimate {
    /// Gaussian smoothing bandwidth (nt).
    pub bandwidth: f32,
    /// Number of pooled gaps the estimate is based on (0 ⇒ fell back to the
    /// floor because no gene had enough sites).
    pub n_gaps: usize,
}

/// Estimate a single bandwidth from per-gene site positions.
///
/// `per_gene` is one entry per gene: a slice of `(position_nt, signal_weight)`
/// for that gene's sites (positions need not be sorted or unique). Positions
/// are deduped per gene with weights summed; consecutive gaps are pooled across
/// all genes, each weighted by the geometric mean of its two endpoints' signal
/// (so a gap touching a marginal site contributes little). The bandwidth is
/// `scale ×` the signal-weighted median gap, clamped to
/// `[min_bandwidth, max_bandwidth]`.
pub fn estimate_bandwidth(
    per_gene: &[Vec<(f32, f32)>],
    params: &BandwidthParams,
) -> BandwidthEstimate {
    let mut gaps: Vec<(f32, f32)> = Vec::new(); // (gap_nt, weight)

    for gene in per_gene {
        if gene.len() < params.min_sites {
            continue;
        }
        // Dedup positions (rounded to nt), summing signal weight.
        let mut by_pos: rustc_hash::FxHashMap<i64, f32> = rustc_hash::FxHashMap::default();
        for &(pos, w) in gene {
            *by_pos.entry(pos.round() as i64).or_insert(0.0) += w.max(0.0);
        }
        if by_pos.len() < params.min_sites {
            continue;
        }
        let mut sites: Vec<(i64, f32)> = by_pos.into_iter().collect();
        sites.sort_by_key(|&(p, _)| p);
        for win in sites.windows(2) {
            let (p0, w0) = win[0];
            let (p1, w1) = win[1];
            let gap = (p1 - p0) as f32;
            if gap > 0.0 {
                // Geometric mean of endpoint signal: a gap is only as
                // trustworthy as its weaker endpoint.
                let gw = (w0.max(0.0) * w1.max(0.0)).sqrt();
                gaps.push((gap, gw));
            }
        }
    }

    if gaps.is_empty() {
        return BandwidthEstimate {
            bandwidth: params.min_bandwidth,
            n_gaps: 0,
        };
    }

    let median = weighted_median(&mut gaps);
    let bw = (params.scale * median).clamp(params.min_bandwidth, params.max_bandwidth);
    BandwidthEstimate {
        bandwidth: bw,
        n_gaps: gaps.len(),
    }
}

/// Signal-weighted median of `(value, weight)` pairs. Mutates (sorts) the input.
/// Returns the value at which the cumulative weight first reaches half the
/// total. Falls back to the unweighted middle if all weights are zero.
fn weighted_median(pairs: &mut [(f32, f32)]) -> f32 {
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = pairs.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return pairs[pairs.len() / 2].0;
    }
    let half = total / 2.0;
    let mut cum = 0.0;
    for &(v, w) in pairs.iter() {
        cum += w;
        if cum >= half {
            return v;
        }
    }
    pairs[pairs.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_falls_back_to_floor() {
        let est = estimate_bandwidth(&[], &BandwidthParams::m6a());
        assert_eq!(est.n_gaps, 0);
        assert_eq!(est.bandwidth, BandwidthParams::m6a().min_bandwidth);
    }

    #[test]
    fn test_recovers_small_mode_for_clustered_sites() {
        // Two genes, each a tight cluster: sites every 20nt. Median gap = 20.
        let gene_a: Vec<(f32, f32)> = (0..5).map(|i| (1000.0 + i as f32 * 20.0, 10.0)).collect();
        let gene_b: Vec<(f32, f32)> = (0..5).map(|i| (5000.0 + i as f32 * 20.0, 10.0)).collect();
        let per_gene = vec![gene_a, gene_b];

        // m6a: scale 1 → ~20, clamped up to floor 10 (20 > 10, stays 20).
        let m6a = estimate_bandwidth(&per_gene, &BandwidthParams::m6a());
        assert!(
            (m6a.bandwidth - 20.0).abs() < 1e-3,
            "m6a bw {}",
            m6a.bandwidth
        );

        // atoi: scale 3 → ~60 (cluster-aware, wider).
        let atoi = estimate_bandwidth(&per_gene, &BandwidthParams::atoi());
        assert!(
            (atoi.bandwidth - 60.0).abs() < 1e-3,
            "atoi bw {}",
            atoi.bandwidth
        );
    }

    #[test]
    fn test_marginal_sites_downweighted() {
        // A dense well-covered cluster (gap 10, high weight) plus one far,
        // low-weight outlier creating a huge gap. The weighted median should
        // track the dense cluster, not the outlier gap.
        let mut gene: Vec<(f32, f32)> = (0..6).map(|i| (100.0 + i as f32 * 10.0, 50.0)).collect();
        gene.push((100_000.0, 0.01)); // far, near-zero signal
        let est = estimate_bandwidth(&[gene], &BandwidthParams::m6a());
        // Median gap should be ~10 (clamped to floor 10), not dominated by the
        // ~99_850 outlier gap.
        assert!(
            est.bandwidth <= 12.0,
            "bw should track dense cluster, got {}",
            est.bandwidth
        );
    }
}
