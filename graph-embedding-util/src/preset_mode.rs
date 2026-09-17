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
    Lora(LoraSpec),
}

impl PresetMode {
    /// Whether the given rows keep their values for the whole fit.
    #[must_use]
    pub fn pins(&self) -> bool {
        !matches!(self, Self::Init)
    }

    /// The LoRA settings, when this is that mode.
    #[must_use]
    pub fn lora(&self) -> Option<LoraSpec> {
        match self {
            Self::Lora(spec) => Some(*spec),
            _ => None,
        }
    }

    /// One phrase for a log line: what happens to the given rows.
    #[must_use]
    pub fn describe(&self) -> &'static str {
        match self {
            Self::Init => "start from the given table and train on",
            Self::Freeze => "pinned to the given table; the rest train",
            Self::Lora(_) => "anchored to the given table with a low-rank residual; the rest train",
        }
    }

    /// Refuse a rank that degenerates to another mode or exceeds the width.
    pub fn validate(&self, h: usize) -> anyhow::Result<()> {
        if let Self::Lora(LoraSpec {
            rank,
            lr_ratio,
            ridge,
        }) = *self
        {
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
            anyhow::ensure!(
                ridge.is_finite() && ridge >= 0.0,
                "the LoRA ridge must be non-negative, got {ridge}"
            );
        }
        Ok(())
    }
}

/// The settings of [`PresetMode::Lora`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoraSpec {
    pub rank: usize,
    /// LoRA+: the shared factor's learning rate over the row factor's.
    pub lr_ratio: f32,
    /// Per-epoch ridge weight PER ROW on the residual's row norm² — the
    /// shrinkage that makes it a residual rather than a second table; `0` =
    /// none. Per row because the data gradient on the shared factor is a sum
    /// over the rows, so one weight means the same thing at any table size.
    pub ridge: f32,
}

/// LoRA's usual small rank; a moderate LoRA+ ratio (the paper's 16 belongs to
/// transformers at far smaller rates and destabilised the shared factor
/// under AdamW); a per-row ridge strong enough that the residual stays below
/// the anchor's own scale under a row optimizer, where a weaker one tied on
/// cell-side structure while letting the residual outgrow the anchor.
impl Default for LoraSpec {
    fn default() -> Self {
        Self {
            rank: 16,
            lr_ratio: 4.0,
            ridge: 0.05,
        }
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
