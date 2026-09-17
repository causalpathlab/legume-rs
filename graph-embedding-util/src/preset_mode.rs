//! How rows given from outside enter a trained table: as a starting point, as
//! pinned values, or as a pinned anchor with a low-rank residual on top.
//!
//! The same three modes serve every engine with a feature table (the hier
//! phase of bge, the PBG engine behind fne and simba, the candle encoders),
//! so the choice is one type and each engine only decides what "pinned" and
//! "low-rank residual" mean in its own parameterization.

/// `Lora`: the given row `ρ₀_g` stays as it is and the trained row is
/// `ρ₀_g + u_g · V` with `u_g ∈ ℝ^rank` per gene and `V ∈ ℝ^{rank × H}`
/// shared — `rank = 0` would be `Freeze` and `rank = H` is `Init` with a
/// detour, so both ends are refused. `lr_ratio` is the LoRA+ knob: `V`, which
/// starts at zero and is touched every step, takes `lr_ratio` times the row
/// factor's learning rate; `1` is plain LoRA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PresetMode {
    Init,
    Freeze,
    Lora { rank: usize, lr_ratio: f32 },
}

impl PresetMode {
    /// Whether the given rows keep their values for the whole fit.
    #[must_use]
    pub fn pins(&self) -> bool {
        !matches!(self, Self::Init)
    }

    /// The LoRA settings, when this is that mode.
    #[must_use]
    pub fn lora(&self) -> Option<(usize, f32)> {
        match *self {
            Self::Lora { rank, lr_ratio } => Some((rank, lr_ratio)),
            _ => None,
        }
    }

    /// One phrase for a log line: what happens to the given rows.
    #[must_use]
    pub fn describe(&self) -> &'static str {
        match self {
            Self::Init => "start from the given table and train on",
            Self::Freeze => "pinned to the given table; the rest train",
            Self::Lora { .. } => {
                "anchored to the given table with a low-rank residual; the rest train"
            }
        }
    }

    /// Refuse a rank that degenerates to another mode or exceeds the width.
    pub fn validate(&self, h: usize) -> anyhow::Result<()> {
        if let Self::Lora { rank, lr_ratio } = *self {
            anyhow::ensure!(
                rank >= 1,
                "a LoRA residual needs rank ≥ 1 (rank 0 is freeze)"
            );
            anyhow::ensure!(
                rank < h,
                "a LoRA residual of rank {rank} is not low-rank at H={h} (rank H is init)"
            );
            anyhow::ensure!(
                lr_ratio.is_finite() && lr_ratio > 0.0,
                "the LoRA+ learning-rate ratio must be positive, got {lr_ratio}"
            );
        }
        Ok(())
    }
}

/// Rows of a table given from outside, by row id on the engine's own axis
/// (genes for the hierarchical phase, nodes for the PBG engine): `rows` is
/// `[ids.len() × H]` row-major, one row per entry of `ids`. What happens to a
/// listed row is the [`PresetMode`]; unlisted rows train freely in every mode.
#[derive(Clone, Debug)]
pub struct PresetRows {
    pub ids: Vec<u32>,
    pub rows: Vec<f32>,
    pub mode: PresetMode,
}

impl PresetRows {
    /// The same rows on another axis: every id mapped through `f`.
    #[must_use]
    pub fn map_ids(self, f: impl Fn(u32) -> u32) -> Self {
        Self {
            ids: self.ids.into_iter().map(f).collect(),
            rows: self.rows,
            mode: self.mode,
        }
    }

    /// Width of the rows, the loader having refused an empty match.
    #[must_use]
    pub fn width(&self) -> usize {
        self.rows.len() / self.ids.len().max(1)
    }
}

#[cfg(test)]
#[path = "preset_mode_tests.rs"]
mod preset_mode_tests;
