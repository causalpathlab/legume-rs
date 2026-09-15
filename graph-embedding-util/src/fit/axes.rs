//! The cell-axis samplers phase 1 trains on.
//!
//! Returns OWNED data and lets `fit()` take the borrows. The alternative — handing back
//! a `&[PerBatchStratifiedCellSampler]` chosen between two locals — would borrow out of
//! the constructor's own frame, and working around that is what keeps this stage stuck
//! inline in a 1000-line function.

use super::config::{FitConfig, DEFAULT_STRATIFY_ALPHA_CELL};
use super::samplers::{build_active_samplers, subsample_cell_samplers_multilevel};
use crate::data::UnifiedData;
use crate::loss::PerBatchStratifiedCellSampler;
use log::info;

/// Everything the phase-1 cell axis is assembled from.
pub(super) struct AxisData {
    /// The FULL per-batch cell samplers. Always kept: the phase-2 projection visits
    /// every cell regardless of what phase 1 trained on.
    pub cell_samplers: Vec<PerBatchStratifiedCellSampler>,
    /// A separate, smaller cell view for phase 1 when `--phase1-cells-per-pb` asks for
    /// one. `None` means phase 1 uses `cell_samplers` as-is (or no cell axis at all).
    phase1_subsample: Option<Vec<PerBatchStratifiedCellSampler>>,
    /// Does phase 1 get a cell axis at all? False at the default
    /// `--phase1-cells-per-pb 0`, which shapes `E_feat` from pb aggregates only.
    pub use_cell_axis: bool,
}

impl AxisData {
    /// The cell samplers that shape `E_feat` in phase 1 — the subsample when there is
    /// one, the full set otherwise.
    pub(super) fn phase1_cell_samplers(&self) -> &[PerBatchStratifiedCellSampler] {
        self.phase1_subsample
            .as_deref()
            .unwrap_or(&self.cell_samplers)
    }
}

/// Build the phase-1 cell axis.
///
/// # `--phase1-cells-per-pb k`
///
/// `k` controls only what shapes `E_feat` in phase 1; phase 2 always projects every
/// cell.
///
/// * `k == 0` — suppress the cell axis entirely (pure-pb phase 1).
/// * `1 ≤ k < n_cells` — keep ≤ `k` cells per pb-sample at every collapse level
///   (union), shrinking the per-epoch step budget from `n_cells` to ≈ `k` × pb-samples
///   while preserving rare-cell coverage.
/// * `k ≥ n_cells` — no pb-sample can exceed `k`, so subsampling is a no-op and the
///   full set is used.
pub(super) fn build_axis_data(
    unified: &UnifiedData,
    cell_to_pb_per_level: &[Vec<usize>],
    config: &FitConfig,
) -> anyhow::Result<AxisData> {
    let (n_cells, n_features) = (unified.n_cells(), unified.n_features());
    let num_levels = cell_to_pb_per_level.len();

    let cell_samplers = build_active_samplers(unified, DEFAULT_STRATIFY_ALPHA_CELL)?;
    info!(
        "Phase-1 cell axis ({} cells × {} features, strat-cell α={}, {} active batch(es))",
        n_cells,
        n_features,
        DEFAULT_STRATIFY_ALPHA_CELL,
        cell_samplers.len()
    );

    let use_cell_axis = config.phase1_cells_per_pb != 0;
    let phase1_subsample: Option<Vec<PerBatchStratifiedCellSampler>> =
        (config.phase1_cells_per_pb >= 1 && config.phase1_cells_per_pb < n_cells).then(|| {
            subsample_cell_samplers_multilevel(
                &cell_samplers,
                cell_to_pb_per_level,
                config.phase1_cells_per_pb,
                DEFAULT_STRATIFY_ALPHA_CELL,
                config.seed,
            )
        });
    match &phase1_subsample {
        Some(sub) => {
            let kept: usize = sub.iter().map(|s| s.active_cells.len()).sum();
            info!(
                "Phase-1 cell subsampling: ≤{} cells per pb-sample (all {num_levels} levels) → \
                 {kept} of {n_cells} cells shape E_feat (phase 2 still projects all {n_cells})",
                config.phase1_cells_per_pb,
            );
        }
        // `k ≥ n_cells` is the silent case: subsampling was a no-op, so the full set is
        // already logged above and there is nothing to add.
        None if !use_cell_axis => info!(
            "Phase-1 cell axis SUPPRESSED (pure-pb): E_feat shaped by pb aggregates only; \
             phase 2 still projects all {n_cells} cells"
        ),
        None => {}
    }

    Ok(AxisData {
        cell_samplers,
        phase1_subsample,
        use_cell_axis,
    })
}
