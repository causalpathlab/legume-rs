//! Negative pools split by modality.
//!
//! NCE learns a log-ratio against the noise distribution, so the noise has to
//! live on the same axis as the positive. On one count panel it does: negatives
//! are drawn over every expressed feature and a positive has to out-score its
//! neighbours on that panel.
//!
//! On a multiome axis it does not. Measured on CITE-seq, a 14-antibody panel
//! carried two thirds of the library while the negative draw is uniform over
//! the whole feature pool — so a protein positive was contrasted against genes
//! essentially every time, and against another protein essentially never. Rows
//! that never have to separate from each other do not: the panel collapsed onto
//! a single shared "not a gene" direction, and the gene dictionary degraded
//! with it.
//!
//! Splitting the pool by modality restores the contrast: a protein positive is
//! scored against the other proteins, a gene against other genes. With one
//! modality there are no splits to make and every draw is the old draw, which
//! is why [`modality_of_features`] returns `None` below two.

use rand::{Rng, RngExt};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use std::sync::Arc;

/// One modality's expressed features in a sampler's pool, with the two
/// pickers the callers mix: uniform, and proportional to degree.
struct Panel {
    features: Vec<u32>,
    /// Only the degree-weighted half needs a distribution. The uniform half is
    /// `random_range` over `features` — an all-ones `WeightedIndex` would pay
    /// an O(log N) binary search over an N-float cumulative array, per draw,
    /// in the innermost loop. `ModulePools::draw_negatives` already does this.
    by_degree: WeightedIndex<f32>,
}

/// Per-modality negative pools for one sampler.
pub struct ModalityPools {
    /// Modality of each global feature id.
    of_feature: Arc<[u32]>,
    /// Indexed by modality id; `None` where this sampler expresses fewer than
    /// two features of that modality, which leaves nothing to contrast.
    panels: Vec<Option<Panel>>,
}

impl ModalityPools {
    /// Build one sampler's pools. `feat_count` is the sampler's per-feature
    /// degree, indexed by global feature id, and drives the degree-weighted
    /// half exactly as the global pool does.
    #[must_use]
    pub fn build(of_feature: &Arc<[u32]>, feature_pool: &[u32], feat_count: &[f32]) -> Self {
        let n_modalities = of_feature.iter().copied().max().map_or(0, |m| m + 1) as usize;
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_modalities];
        for &f in feature_pool {
            members[of_feature[f as usize] as usize].push(f);
        }
        let panels = members
            .into_iter()
            .map(|features| {
                // A lone feature has nothing to be contrasted against.
                if features.len() < 2 {
                    return None;
                }
                let deg: Vec<f32> = features
                    .iter()
                    .map(|&f| feat_count[f as usize].max(1e-8))
                    .collect();
                let by_degree = WeightedIndex::new(deg).ok()?;
                Some(Panel {
                    features,
                    by_degree,
                })
            })
            .collect();
        Self {
            of_feature: of_feature.clone(),
            panels,
        }
    }

    /// Modality per global feature id, as the module pools need it. Handing
    /// back the `Arc` rather than a slice lets those pools share it instead of
    /// copying an n_features-long vector per pool per epoch.
    #[must_use]
    pub fn of_feature(&self) -> &Arc<[u32]> {
        &self.of_feature
    }

    /// Panels on this axis.
    #[must_use]
    pub fn n_panels(&self) -> usize {
        self.panels.len()
    }

    fn panel(&self, feat: u32) -> Option<&Panel> {
        self.panels
            .get(self.of_feature[feat as usize] as usize)?
            .as_ref()
    }

    /// A negative for `feat` from its own modality, uniformly. `None` when
    /// that modality has nothing to contrast with, so the caller keeps its own
    /// global fallback.
    pub fn draw_uniform<R: Rng>(&self, feat: u32, rng: &mut R) -> Option<u32> {
        let p = self.panel(feat)?;
        Some(p.features[rng.random_range(0..p.features.len())])
    }

    /// As [`Self::draw_uniform`], proportional to degree within the modality.
    pub fn draw_by_degree<R: Rng>(&self, feat: u32, rng: &mut R) -> Option<u32> {
        let p = self.panel(feat)?;
        Some(p.features[p.by_degree.sample(rng)])
    }
}

#[cfg(test)]
mod tests;
