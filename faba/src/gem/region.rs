//! Region map: per-component slot index used to key the additive offset
//! `γ_{m, r, :}` in the gem embedding model.
//!
//! A modifier feature row is named `{gene}/{m}/{k}`, where `k` is the
//! component index within that gene+modality. After dropping the
//! transcript-position sidecar input, `region` is no longer a
//! transcript-position bin shared across genes; it is simply the
//! per-component slot, clamped to `n_regions − 1`. That keeps same-gene
//! components structurally distinguishable in the embedding (each picks up
//! its own `γ` offset) without needing any external annotation.

/// Per-modality slot count and lookup. Owns only `n_regions`; the slot
/// assignment is derived directly from the component index at lookup time.
pub struct RegionMap {
    /// Number of γ slots R. `region_id ∈ 0..R`.
    pub n_regions: usize,
}

impl RegionMap {
    /// Build a map with `R = max(n_regions, 1)`. Zero is silently bumped to
    /// one so the model always has a valid γ axis.
    pub fn new(n_regions: usize) -> Self {
        Self {
            n_regions: n_regions.max(1),
        }
    }

    /// Slot for a modifier `component`: per-component within its
    /// (gene, modality), clamped to `n_regions − 1`. Same component index →
    /// same slot regardless of gene/modality.
    pub fn lookup(&self, component: u32) -> u32 {
        component.min(self.n_regions as u32 - 1)
    }

    /// Test-only constructor mirroring the old API. Same semantics as
    /// `new(n_regions)`.
    #[cfg(test)]
    pub fn empty(n_regions: usize) -> Self {
        Self::new(n_regions)
    }
}

#[cfg(test)]
mod tests;
