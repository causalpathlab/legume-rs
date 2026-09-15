//! Everything upstream of training: the batch-corrected projection, the multilevel
//! collapse it feeds, and the per-level pseudobulk views phase 1's axes are built
//! from.
//!
//! One module because it is one dependency chain — the projection exists only to hash
//! cells into the collapse, and the collapse exists only to produce these blobs. None
//! of it touches a model, a Var or an optimizer.

use super::config::{FitConfig, TrackSpec};
use crate::data::UnifiedData;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

/// The collapse, ordered **coarsest → finest**, paired with the per-level pseudobulk
/// views and the cell→pb maps.
///
/// Kept as one struct rather than unpacked at the call site on purpose: `fit()` used to
/// split these three apart immediately and then thread the pieces through every
/// downstream stage separately, which is what pushed several of its would-be helper
/// functions past a defensible parameter count. They are one object; passing them as
/// one keeps the seams below it honest.
pub(super) struct Pseudobulks {
    pub collapsed_levels: Vec<data_beans_alg::collapse_data::CollapsedOut>,
    /// `cell_to_pb_per_level[l][c]` is cell `c`'s pseudobulk at level `l`.
    pub cell_to_pb_per_level: Vec<Vec<usize>>,
    /// One `UnifiedData` per level, on the unified feature axis.
    pub blobs: Vec<UnifiedData>,
}

/// Project, collapse, and materialize the per-level pseudobulk views.
///
/// `tracks` is the fit's already-validated feature-axis structure (`fit` builds and
/// checks it before calling): the projection sketches on its base track's rows.
///
/// `sort_dim` controls how many bits of the binary-sketched projection are used to hash
/// cells into the *finest* pb-sample partition, so `2^sort_dim` bounds the number of
/// distinct codes at that level. It is exposed directly on [`FitConfig`] for parity with
/// `senna topic` / `svd` rather than derived from a target count.
pub(super) fn build_pseudobulks(
    unified: &mut UnifiedData,
    config: &FitConfig,
    tracks: &TrackSpec,
) -> anyhow::Result<Pseudobulks> {
    let n_features = unified.n_features();
    let feature_to_backend = unified.feature_to_backend_row.clone();
    let batch_labels: Vec<Box<str>> = unified.batch_labels();

    let proj_out = project(unified, config, &batch_labels, tracks)?;

    info!(
        "Multilevel collapse (sort_dim={}, {} levels requested)...",
        config.sort_dim, config.num_levels
    );
    let collapse_out = collapse_columns_multilevel_with_hierarchy(
        unified.count_backend_mut(),
        &proj_out.proj,
        &batch_labels,
        &MultilevelParams {
            knn_pb_samples: config.knn_pb_samples,
            num_levels: config.num_levels.max(1),
            sort_dim: config.sort_dim,
            num_opt_iter: config.num_opt_iter,
            refine: config.refine.clone(),
            // Only `posterior_mean()` is ever read off this, so skip the sd / log_mean /
            // log_sd planes — that is the bulk of the coarsen-stage memory at high
            // pb-sample counts.
            output_calibration: matrix_param::traits::CalibrateTarget::MeanOnly,
            anchor_batches: config.anchor_batches.clone(),
            bulk_batches: config.bulk_batches.clone(),
            observe_panels: true,
            keep_finest_stats: config.emit_finest_collapse,
            pb_tree: None,
        },
    )?;
    let mut collapsed_levels = collapse_out.levels;
    let mut cell_to_pb_per_level = collapse_out.cell_to_pb_per_level;
    // The collapse emits finest-first. Reverse both so levels run coarsest..finest —
    // `senna topic` uses the same order, so its curriculum trains coarse first.
    collapsed_levels.reverse();
    cell_to_pb_per_level.reverse();

    // pb counts live on the unified feature axis. When the backend holds more rows than
    // that axis — an HVG mask having narrowed `unified.feature_names`, say — gather the
    // unified rows out of the backend's pb matrix; otherwise reuse it as-is.
    let mut blobs: Vec<UnifiedData> = Vec::with_capacity(collapsed_levels.len());
    for collapsed in &collapsed_levels {
        let pb_full: &DMatrix<f32> = match &collapsed.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => collapsed.mu_observed.posterior_mean(),
        };
        let pb_count_ds = gather_to_unified_axis(pb_full, n_features, &feature_to_backend);
        blobs.push(UnifiedData::from_pseudobulks(
            &pb_count_ds,
            unified.feature_names.clone(),
            unified.feature_to_backend_row.clone(),
        )?);
    }

    // The flat cell↔feature edge list is intentionally NOT built. The cell axis is always
    // `PerBatchStratified`, whose sampler streams columns in `build_active_samplers` and
    // is self-contained at sample time, so `unified.triplets` stays empty for it.
    Ok(Pseudobulks {
        collapsed_levels,
        cell_to_pb_per_level,
        blobs,
    })
}

/// The batch-corrected random projection the collapse hashes on, HVG-weighted when the
/// caller supplied weights.
///
/// On a multi-track feature axis the sketch runs on the BASE track's rows alone: the
/// other tracks are offsets from the base model, not independent measurements, and a
/// sketch that stacked them would hash cells partly on the offsets' own scale. The
/// collapse itself still runs on the FULL backend with this sketch, so refinement,
/// `mu_adjusted` and `δ` cover every row. One track ⇒ no mask, no clone, the previous
/// call.
fn project(
    unified: &UnifiedData,
    config: &FitConfig,
    batch_labels: &[Box<str>],
    tracks: &TrackSpec,
) -> anyhow::Result<data_beans_alg::random_projection::RandColProjOut> {
    info!(
        "Batch-corrected projection (proj_dim={}, {} batches)...",
        config.proj_dim,
        unified.n_batches()
    );
    let batch_arg = (unified.n_batches() > 1).then_some(batch_labels);
    let backend = unified.count_backend();
    // The projection runs on the full backend row axis, which may be wider than the
    // compact feature axis when a prior pass dropped features (e.g. the two-pass null-QC
    // refine in `senna bge`). Scatter the compact weights to backend rows; rows not in
    // the current feature axis get 0 so they sit out the projection basis. Identity —
    // and a no-op — when no subset has happened.
    let backend_w: Option<Vec<f32>> = match config.hvg_weights.as_deref() {
        None => None,
        Some(w) => {
            anyhow::ensure!(
                w.len() == unified.n_features(),
                "hvg_weights length {} != n_features {} (the HVG mask must be aligned to the \
                 unified feature axis BEFORE any subset/coarsening — pass full-axis weights from \
                 the wrapper)",
                w.len(),
                unified.n_features()
            );
            info!(
                "HVG-weighted projection: {} weighted features (>= 1.0)",
                w.iter().filter(|&&x| x > 0.0).count()
            );
            let mut backend_w = vec![0.0f32; backend.num_rows()];
            for (compact_i, &brow) in unified.feature_to_backend_row.iter().enumerate() {
                backend_w[brow] = w[compact_i];
            }
            Some(backend_w)
        }
    };

    let keep = base_track_row_mask(tracks, &unified.feature_to_backend_row, backend.num_rows());
    let Some(keep) = keep else {
        return project_backend(
            backend,
            config.proj_dim,
            config.block_size,
            batch_arg,
            backend_w.as_deref(),
            config.seed,
        );
    };
    info!(
        "Multi-track feature axis: sketching on the base track's {} of {} backend rows",
        keep.iter().filter(|&&k| k).count(),
        keep.len()
    );
    let mut view = backend.clone_for_collapse();
    view.mask_rows(&keep)?;
    // `mask_rows` RENUMBERS the kept rows compactly, so the weight vector has to be
    // subset the same way — a full-axis vector would misalign every row.
    let view_w = backend_w.map(|w| subset_kept(&w, &keep));
    project_backend(
        &view,
        config.proj_dim,
        config.block_size,
        batch_arg,
        view_w.as_deref(),
        config.seed,
    )
}

/// Which backend rows the sketch reads: the base track's, or `None` when the feature
/// axis is a single track and there is nothing to mask.
fn base_track_row_mask(
    tracks: &TrackSpec,
    feature_to_backend: &[usize],
    backend_rows: usize,
) -> Option<Vec<bool>> {
    if tracks.n_tracks() <= 1 {
        return None;
    }
    let mut keep = vec![false; backend_rows];
    for (feature, &t) in tracks.track_of_row.iter().enumerate() {
        if t == 0 {
            if let Some(&brow) = feature_to_backend.get(feature) {
                keep[brow] = true;
            }
        }
    }
    Some(keep)
}

/// `values` restricted to the `keep`ed positions, in the same order — the order
/// [`SparseIoVec::mask_rows`] renumbers them into.
fn subset_kept(values: &[f32], keep: &[bool]) -> Vec<f32> {
    values
        .iter()
        .zip(keep)
        .filter(|&(_, &k)| k)
        .map(|(&v, _)| v)
        .collect()
}

/// One random projection over `backend`, weighted when `row_weights` is given (length =
/// `backend.num_rows()`).
fn project_backend<T>(
    backend: &SparseIoVec,
    proj_dim: usize,
    block_size: Option<usize>,
    batch_arg: Option<&[T]>,
    row_weights: Option<&[f32]>,
    seed: u64,
) -> anyhow::Result<data_beans_alg::random_projection::RandColProjOut>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    match row_weights {
        None => backend
            .project_columns_with_batch_correction_seeded(proj_dim, block_size, batch_arg, seed),
        Some(w) => {
            backend.project_columns_weighted_seeded(proj_dim, block_size, batch_arg, w, seed)
        }
    }
}

/// Gather a backend-row matrix onto the compact unified feature axis. A clone when the
/// two already agree, which is every run without a feature subset.
pub(super) fn gather_to_unified_axis(
    backend: &DMatrix<f32>,
    n_features: usize,
    feature_to_backend: &[usize],
) -> DMatrix<f32> {
    if backend.nrows() == n_features {
        return backend.clone();
    }
    let cols = backend.ncols();
    let mut out = DMatrix::<f32>::zeros(n_features, cols);
    for (new_i, &old_i) in feature_to_backend.iter().enumerate() {
        for s in 0..cols {
            out[(new_i, s)] = backend[(old_i, s)];
        }
    }
    out
}

#[cfg(test)]
#[path = "setup_tests.rs"]
mod setup_tests;
