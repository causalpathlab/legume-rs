//! The samplers and coarsenings the composite axes are built from — one cell axis plus
//! one per collapse level.
//!
//! Returns OWNED data and lets `fit()` take the borrows. The alternative — handing back
//! a `&[PerBatchStratifiedCellSampler]` chosen between two locals — would borrow out of
//! the constructor's own frame, and working around that is what keeps this stage stuck
//! inline in a 1000-line function.

use super::config::{FitConfig, DEFAULT_STRATIFY_ALPHA_CELL, DEFAULT_STRATIFY_ALPHA_PB};
use super::samplers::{build_active_samplers, subsample_cell_samplers_multilevel};
use crate::coarsen::{identity_axis, AxisCoarsenings};
use crate::data::UnifiedData;
use crate::loss::{
    build_stratified_sampler, FeatPairing, PerBatchStratifiedCellSampler, StratifiedSampler,
};
use log::info;

/// Everything the composite axes are assembled from.
pub(super) struct AxisData {
    pub cell_axis_coarsening: AxisCoarsenings,
    /// The FULL per-batch cell samplers. Always kept: the phase-2 projection visits
    /// every cell regardless of what phase 1 trained on.
    pub cell_samplers: Vec<PerBatchStratifiedCellSampler>,
    /// A separate, smaller cell view for phase 1 when `--phase1-cells-per-pb` asks for
    /// one. `None` means phase 1 uses `cell_samplers` as-is (or no cell axis at all).
    pub phase1_subsample: Option<Vec<PerBatchStratifiedCellSampler>>,
    /// `(coarsening, sampler)` per collapse level, coarsest → finest.
    pub level_axes: Vec<(AxisCoarsenings, StratifiedSampler)>,
    /// Does phase 1 get a cell axis at all? False at the default
    /// `--phase1-cells-per-pb 0`, which shapes `E_feat` from pb aggregates only.
    pub use_cell_axis: bool,
}

impl AxisData {
    /// The cell samplers that shape `E_feat` in phase 1 — the subsample when there is
    /// one, the full set otherwise.
    pub fn phase1_cell_samplers(&self) -> &[PerBatchStratifiedCellSampler] {
        self.phase1_subsample
            .as_deref()
            .unwrap_or(&self.cell_samplers)
    }
}

/// Build the cell axis and every pseudobulk axis.
///
/// # `--phase1-cells-per-pb k`
///
/// `k` controls only what shapes `E_feat` in phase 1; phase 2 always projects every
/// cell.
///
/// * `k == 0` — suppress the cell axis entirely (pure-pb phase 1). The default.
/// * `1 ≤ k < n_cells` — keep ≤ `k` cells per pb-sample at every collapse level
///   (union), shrinking the per-epoch step budget from `n_cells` to ≈ `k` × pb-samples
///   while preserving rare-cell coverage.
/// * `k ≥ n_cells` — no pb-sample can exceed `k`, so subsampling is a no-op and the
///   full set is used.
pub(super) fn build_axis_data(
    unified: &UnifiedData,
    pb_blobs: &[UnifiedData],
    cell_to_pb_per_level: &[Vec<usize>],
    config: &FitConfig,
) -> anyhow::Result<AxisData> {
    let (n_cells, n_features) = (unified.n_cells(), unified.n_features());
    let num_levels = pb_blobs.len();

    let cell_samplers = build_active_samplers(
        unified,
        DEFAULT_STRATIFY_ALPHA_CELL,
        config.cell_weight_mult.as_deref(),
    )?;
    info!(
        "Composite axis cell ({} cells × {} features, strat-cell α={}, {} active batch(es))",
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
                config.cell_weight_mult.as_deref(),
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

    // β-sharing (gem): sample phase-1 positives by GENE at the spliced count, and emit
    // the paired unspliced edge so δ_g trains at that frequency — the identity stays
    // spliced-driven, with no double-bite from nascent abundance. `None` for bge, which
    // is per-row.
    let pairing = config.feat_factor.as_ref().map(|spec| FeatPairing {
        row_to_gene: &spec.row_to_gene,
        unspliced_rows: &spec.unspliced_rows,
    });

    let mut level_axes: Vec<(AxisCoarsenings, StratifiedSampler)> = Vec::with_capacity(num_levels);
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        let stratified = build_stratified_sampler(
            &pb.triplets,
            n_pb,
            n_features,
            DEFAULT_STRATIFY_ALPHA_PB,
            pairing.as_ref(),
        )
        .ok_or_else(|| {
            anyhow::anyhow!(
                "pb_l{level_idx}: stratified sampler build failed (no positives or empty \
                 feature pool)"
            )
        })?;
        info!(
            "Composite axis pb_l{} ({} pseudobulks × {} features, stratified α={}, {} active pb(s))",
            level_idx,
            n_pb,
            n_features,
            DEFAULT_STRATIFY_ALPHA_PB,
            stratified.active_pbs.len()
        );
        level_axes.push((identity_axis(n_pb), stratified));
    }

    Ok(AxisData {
        cell_axis_coarsening: identity_axis(n_cells),
        cell_samplers,
        phase1_subsample,
        level_axes,
        use_cell_axis,
    })
}
