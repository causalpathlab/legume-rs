//! Feature (gene) QC by highly-variable-gene (HVG) selection on the NB
//! mean–variance **dispersion trend** — model-independent (from counts, not β).
//!
//! The earlier β-based k-means/elbow QC was **circular** (it QC'd genes using the
//! very embedding those genes shape) and brittle (under logistic NCE it found no
//! empty cluster at all). Instead, keep genes that are **overdispersed** relative
//! to the fitted NB dispersion trend — `DispersionTrend::excess(μ, var) > cutoff`,
//! the standard HVG criterion — with a light expression floor so genes too sparse
//! for a reliable dispersion estimate are dropped. The rest (low-variability
//! background / housekeeping) are dropped. Safe to drop under logistic NCE (no
//! softmax partition to collapse). A majority guard keeps all if a degenerate fit
//! would drop almost everything.

/// Tuning knobs for [`hvg_feature_qc`].
#[derive(Clone, Debug)]
pub struct FeatureQcConfig {
    /// Master switch. `false` ⇒ keep every gene.
    pub enabled: bool,
    /// Keep genes whose excess dispersion (`φ̂_g − trend(μ_g)`) is strictly above
    /// this. `0.0` = strictly above the trend (the standard overdispersed/HVG cut);
    /// raise it to keep only strongly variable genes, lower it to keep more.
    pub hvg_min_excess: f32,
    /// Expression floor: drop genes detected in fewer than this many cells (their
    /// dispersion estimate is unreliable). `0` = no floor (HVG-only).
    pub min_nnz: f32,
}

impl Default for FeatureQcConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hvg_min_excess: 0.0,
            min_nnz: 0.0,
        }
    }
}

/// Per-gene feature-QC verdict.
pub struct FeatureQcResult {
    /// Keep decision, one per gene.
    pub keep: Vec<bool>,
}

impl FeatureQcResult {
    fn all_kept(n: usize) -> Self {
        Self {
            keep: vec![true; n],
        }
    }
    /// Number of genes dropped.
    #[must_use]
    pub fn n_dropped(&self) -> usize {
        self.keep.iter().filter(|&&k| !k).count()
    }
    /// Number of genes kept.
    #[must_use]
    pub fn n_kept(&self) -> usize {
        self.keep.iter().filter(|&&k| k).count()
    }
}

/// HVG gene QC. `gene_nnz` is per-gene cell support; `gene_hvg` is per-gene excess
/// dispersion (`DispersionTrend::excess`, higher = more variable). Keeps genes
/// above the HVG cut with adequate support; drops the low-variability background.
/// A majority guard keeps all if the cut would drop ≥ 95% (degenerate trend fit /
/// bad input).
#[must_use]
pub fn hvg_feature_qc(
    gene_nnz: &[f32],
    gene_hvg: &[f32],
    cfg: &FeatureQcConfig,
) -> FeatureQcResult {
    let n = gene_hvg.len();
    if !cfg.enabled || n == 0 || gene_nnz.len() != n {
        return FeatureQcResult::all_kept(n);
    }
    let keep: Vec<bool> = (0..n)
        .map(|g| {
            gene_hvg[g].is_finite()
                && gene_hvg[g] > cfg.hvg_min_excess
                && gene_nnz[g] >= cfg.min_nnz
        })
        .collect();
    let n_drop = keep.iter().filter(|&&k| !k).count();
    // Guard: never drop almost everything (e.g. a degenerate dispersion fit).
    if n_drop * 100 >= n * 95 {
        return FeatureQcResult::all_kept(n);
    }
    FeatureQcResult { keep }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_overdispersed_drops_flat() {
        // genes 0..50 overdispersed (excess > 0), 50..100 below trend (excess < 0).
        let n = 100;
        let nnz: Vec<f32> = vec![100.0; n];
        let hvg: Vec<f32> = (0..n).map(|g| if g < 50 { 1.0 } else { -0.5 }).collect();
        let res = hvg_feature_qc(&nnz, &hvg, &FeatureQcConfig::default());
        assert_eq!(res.n_kept(), 50, "should keep the 50 overdispersed genes");
        assert!((0..50).all(|g| res.keep[g]));
        assert!((50..100).all(|g| !res.keep[g]));
    }

    #[test]
    fn expression_floor_drops_sparse_hvg() {
        // both overdispersed, but the first is below the nnz floor.
        let cfg = FeatureQcConfig {
            min_nnz: 10.0,
            ..Default::default()
        };
        let res = hvg_feature_qc(&[3.0, 100.0], &[1.0, 1.0], &cfg);
        assert_eq!(res.keep, vec![false, true]);
    }

    #[test]
    fn guard_keeps_all_when_degenerate() {
        // every gene below trend → would drop 100% → guard keeps all.
        let res = hvg_feature_qc(&[100.0; 10], &[-1.0; 10], &FeatureQcConfig::default());
        assert_eq!(res.n_dropped(), 0);
    }
}
