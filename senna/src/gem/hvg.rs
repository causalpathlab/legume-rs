//! Pooled gene-level HVG projection weights for `senna gem`.
//!
//! Unlike a plain row axis, a gem gene may carry several rows (one per
//! track: the base count, an unspliced offset, one pair of channel rows per
//! co-measured modality). HVG selection has to rank GENES, pooling every one
//! of a gene's rows first — otherwise a gene whose variance shows up only in
//! a modality track (never in its own counts) would never earn its
//! projection weight. [`gem_hvg_row_weights`] does that pooling over a
//! [`TrackPlan`] and hands back a per-ROW weight vector, 1.0 only on the
//! BASE row (`count/spliced`) of a selected gene — every other row of that
//! gene, on every other track, carries weight 0. Non-base rows never carry
//! projection weight regardless of selection: the hierarchical trainer
//! already restricts its random-projection sketch to base rows when
//! `n_tracks > 1` (a masked clone of the feature axis), so this is the belt
//! to that suspender, not a second, independent decision.

use data_beans::utilities::name_matching::GeneIndex;
use data_beans_alg::hvg::{
    load_must_train, select_hvg_by_stats, union_indices, HvgCliArgs, MustTrainFeatures,
};
use data_beans_alg::sparse_streaming::streaming_sparse_running_stats;
use graph_embedding_util as ge;
use log::info;
use matrix_util::traits::RunningStatOps;
use rustc_hash::FxHashSet;

use crate::gem::tracks::TrackPlan;

/// Per-row HVG projection weights over a gem feature axis, pooled per gene
/// across every track. `None` when selection is off and the axis has only
/// the base track (plain `senna bge` behaviour: no weighting at all).
///
/// - `--feature-list-file` REPLACES the ranking with exactly the named
///   genes, resolved against `plan.gene_names` (lenient matching, see
///   [`MustTrainFeatures::resolve_with`]).
/// - Otherwise, when `hvg.n_hvg > 0`, genes are ranked by NB dispersion-trend
///   excess over pooled per-gene `(mean, variance)`: every row's streaming
///   stats are computed once, then summed into `plan.row_gene` buckets, so a
///   gene's base, unspliced and modality rows all contribute to its rank.
/// - `--must-train-features` UNIONS a curated panel into the selection,
///   resolved the same way, regardless of which of the two rules produced
///   the base selection.
/// - Weights: `w[r] = 1.0` iff `plan.base_rows[r]` and `row_gene[r]` was
///   selected, else `0.0` — computed over ALL selected genes, `Some` even
///   when nothing was selected (an all-zero vector), so the caller never has
///   to re-derive "selection ran but kept nothing" from a `None`.
/// - Selection off (`n_hvg == 0` and no `--feature-list-file`): `Some` of
///   the base-row mask when the plan has more than the base track (so
///   modality/unspliced rows still get weight 0, matching what the trainer
///   restricts to anyway), else `None` — a plain spliced-only axis has
///   nothing to mask, matching `senna bge`'s own unweighted default.
pub(crate) fn gem_hvg_row_weights(
    unified: &ge::UnifiedData,
    plan: &TrackPlan,
    hvg: &HvgCliArgs,
    block_size: Option<usize>,
) -> anyhow::Result<Option<Vec<f32>>> {
    let selection_on = hvg.n_hvg > 0 || hvg.feature_list_file.is_some();
    if !selection_on {
        let has_non_base = plan.base_rows.iter().any(|&b| !b);
        return Ok(has_non_base.then(|| {
            plan.base_rows
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect()
        }));
    }

    let n_genes = plan.gene_names.len();
    let gene_index = GeneIndex::build(&plan.gene_names);

    let mut selected: Vec<usize> = if let Some(path) = hvg.feature_list_file.as_deref() {
        MustTrainFeatures::load(path)?.resolve_with(&gene_index)
    } else {
        let stat = streaming_sparse_running_stats(unified.count_backend(), block_size, "gem HVG")?;
        let (means, vars) = (stat.mean(), stat.variance());
        let mut gmean = vec![0f32; n_genes];
        let mut gvar = vec![0f32; n_genes];
        for (r, (&m, &v)) in means.iter().zip(vars.iter()).enumerate() {
            gmean[plan.row_gene[r] as usize] += m;
            gvar[plan.row_gene[r] as usize] += v;
        }
        select_hvg_by_stats(&gmean, &gvar, hvg.n_hvg)
    };

    if let Some(must_train) = load_must_train(hvg.must_train_features.as_deref(), selection_on)? {
        let forced = must_train.resolve_with(&gene_index);
        let added = union_indices(&mut selected, &forced);
        info!(
            "gem HVG: {added} gene(s) force-added on top of the selection ({} of the {} \
             matched were already selected)",
            forced.len() - added,
            forced.len()
        );
    }

    let keep: FxHashSet<usize> = selected.into_iter().collect();
    let mut w = vec![0.0f32; unified.n_features()];
    for (r, slot) in w.iter_mut().enumerate() {
        if plan.base_rows[r] && keep.contains(&(plan.row_gene[r] as usize)) {
            *slot = 1.0;
        }
    }
    let n_weighted = w.iter().filter(|&&x| x > 0.0).count();
    info!(
        "gem HVG (--n-hvg {}): {} of {n_genes} genes selected -> {n_weighted} of {} feature \
         row(s) carry the projection; every row still trains",
        hvg.n_hvg,
        keep.len(),
        unified.n_features()
    );
    Ok(Some(w))
}

#[cfg(test)]
#[path = "hvg/tests.rs"]
mod tests;
