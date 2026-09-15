//! Per-**cell** phase-2 projection: re-fit every cell's embedding `θ_c` (and the
//! per-cell bias `b_cell`) against the frozen feature dictionary, via the shared
//! block Poisson-MAP SGD ([`block_sgd`]).

use super::block_sgd;
use super::encoder::{self, CellEncoder, DistillSpec};
use super::CellBatchFold;
use crate::fit::config::TrackSpec;
use crate::loss::PerBatchStratifiedCellSampler;
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use log::info;

/// What phase 2 hands back to [`crate::fit::fit`].
pub(crate) struct Phase2Result {
    /// Un-normalized MAP norm the empty-droplet QC keys on (`‖θ‖`).
    pub cell_nrms: Vec<f32>,
    /// The identity mean `θ̄` the gauge fix removed, `[h]`.
    ///
    /// **A frame marker, not a diagnostic.** `block_sgd` removes the population
    /// mean from every solved cell and folds it into `b_feat` (see
    /// [`super::block_sgd::GaugeShift`]); the phase-1 pseudobulk tables were read
    /// in the as-trained (un-gauged) frame, so `fit()` shifts them by this mean
    /// before they leave in the cells' frame.
    pub theta_mean: Vec<f32>,
    /// The distilled encoder that placed the cells, when `distill` was given;
    /// `None` when the block SGD did.
    pub cell_encoder: Option<CellEncoder>,
    /// The fitted intercept of every NON-base track, `[T - 1][n_cells]`; empty on
    /// a one-track feature axis. Track 0's is `b_cell`, stored on the model.
    pub other_intercepts: Vec<Vec<f32>>,
}

/// Flatten the per-batch samplers into one `(cell_id, features, counts)` list,
/// borrowing each cell's edge slices. Shared by the baseline and dual-axis
/// phase-2 projections (both walk the same active cells).
fn collect_sampler_cells(
    cell_samplers: &[PerBatchStratifiedCellSampler],
) -> Vec<(u32, &[u32], &[f32])> {
    let mut cells = Vec::new();
    for s in cell_samplers {
        for (i, &cell) in s.active_cells.iter().enumerate() {
            let cf = &s.per_cell[i];
            cells.push((cell, cf.features.as_slice(), cf.counts.as_slice()));
        }
    }
    cells
}

/// Phase 2 — project every cell onto the fixed feature dictionary and overwrite
/// the `e_cell` var. The per-cell bias is fitted (to absorb library size) and
/// written into the `b_cell` var alongside `e_cell`.
///
/// The solve itself is a **cell-block Poisson SGD** ([`block_sgd`]): with the
/// feature side frozen the objective is separable per cell, so a block of cells is
/// an independent problem and each Adam step is two dense matmuls against the one
/// shared `Eᵀ`. That formulation also affords the **full log-partition over every
/// feature**, which is what identifies `θ` — the previous per-cell Newton solve
/// fit only each cell's observed features and let a fixed ridge stand in for the
/// partition, leaving `‖θ‖` free to run away along whatever direction the
/// unobserved features carried.
///
/// The stored latent (`model.e_cell`) is the **raw** embedding — magnitude kept,
/// no unit-norm. A unit-normed store puts every cell on `S^{H−1}`, and a layout of
/// that is arcs and rings rather than the free clouds the same objective gives
/// `simba`; it also made the run's own cells a different object from the cells
/// the persisted encoder places at predict time. A consumer that wants direction
/// only takes it (cosine) as an explicit choice, so a zero-signal cell is never
/// turned into a fabricated unit vector.
///
/// One combined Poisson-MAP solve per cell → identity `e_cell`, stored raw.
///
/// `distill`, when given, replaces the block SGD by the distilled encoder
/// ([`encoder`]): the phase-1 pseudobulk tables are the targets, and every cell
/// is encoded in one pass. `None` keeps the block SGD.
///
/// See [`Phase2Result`] for what comes back.
#[allow(clippy::too_many_arguments)] // frozen dictionary + samplers + batch fold + distill spec
pub(crate) fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
    batch_fold: Option<CellBatchFold>,
    distill: Option<&DistillSpec<'_>>,
    tracks: &TrackSpec,
) -> anyhow::Result<Phase2Result> {
    use anyhow::Context;
    use candle_util::candle_core::Tensor;

    let h = model.embedding_dim;

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let feat_flat: Vec<f32> = model.e_feat.flatten_all()?.to_vec1()?;
    let cells = collect_sampler_cells(cell_samplers);

    info!(
        "Phase 2 — {} over {n_cells} cells ({} with edges) on {dev:?}, \
         full log-partition, ridge λ={lambda}{}",
        if distill.is_some() {
            "distilled pooled-gene encoder"
        } else {
            "cell-block Poisson SGD"
        },
        cells.len(),
        match &batch_fold {
            Some(bf) => format!(
                ", counts divided by the per-batch gene fold ({} batches)",
                bf.fold.n_batches()
            ),
            None => String::new(),
        }
    );

    let input = block_sgd::Phase2Input {
        feat: &feat_flat,
        b_feat: &b_feat,
        h,
        n_cells,
        lambda,
        dev,
        label: "Phase 2",
        // Cells: fold the common mode into b_feat (feature co-embedding depends
        // on it) — see [`block_sgd::Phase2Input::gauge_fix`].
        gauge_fix: true,
    };
    let (out, cell_encoder) = match distill {
        Some(spec) => {
            let (out, enc) = encoder::project_cells(&input, &cells, batch_fold, spec, tracks)?;
            (out, Some(enc))
        }
        None => {
            // The cold solve is one partition over the whole feature axis, so it
            // has no per-track intercept to give. Refusing here is what keeps a
            // multi-track fit from silently losing them.
            anyhow::ensure!(
                tracks.is_base(),
                "phase 2: the cold block SGD is single-partition — a multi-track \
                 feature axis needs the distilled encoder path"
            );
            (block_sgd::project_cells(&input, &cells, batch_fold)?, None)
        }
    };

    /////////////////////////////////////////////////
    // Fold the gauge shift back into `b_feat`     //
    /////////////////////////////////////////////////

    // `block_sgd` removed the population mean from each latent. That is only a
    // *re-gauge* — every score unchanged — if the matching `⟨e_f, θ̄⟩` goes into
    // the per-feature bias. Skipping this would silently change the model, and
    // `b_feat` is a real output (`feature_bias.parquet`) that the held-out gene
    // projection also solves against.
    let mut b_feat = b_feat;
    let n_features = b_feat.len();
    let tm = &out.gauge.theta_mean;
    for (f, b) in b_feat.iter_mut().enumerate() {
        let e_f = &feat_flat[f * h..(f + 1) * h];
        let shift: f32 = e_f.iter().zip(tm).map(|(e, m)| e * m).sum();
        *b += shift;
    }
    // The persisted encoder places predict-time cells; give it the same gauge
    // so a query and the run's cells share one frame.
    if let Some(enc) = cell_encoder.as_ref() {
        enc.shift_output(tm)?;
    }
    let b_feat_t = Tensor::from_vec(b_feat, n_features, dev)?;
    {
        let vars = varmap.data().lock().unwrap();
        vars.get("b_feat")
            .context("b_feat var missing")?
            .set(&b_feat_t)?;
    }
    model.b_feat = b_feat_t;

    // `cell_nrms` is the MAP norm the empty-droplet QC keys on. Post-gauge-fix
    // this is the distance from the population mean, which is the more useful
    // "how much signal" reading anyway.
    let cell_nrms: Vec<f32> = out.theta.chunks_exact(h).map(norm).collect();

    let e_t = Tensor::from_vec(out.theta, (n_cells, h), dev)?;
    let b_t = Tensor::from_vec(out.b_cell, n_cells, dev)?;
    {
        let vars = varmap.data().lock().unwrap();
        vars.get("e_cell")
            .context("e_cell var missing")?
            .set(&e_t)?;
        vars.get("b_cell")
            .context("b_cell var missing")?
            .set(&b_t)?;
    }
    model.e_cell = e_t;
    model.b_cell = b_t;

    Ok(Phase2Result {
        cell_nrms,
        theta_mean: out.gauge.theta_mean,
        cell_encoder,
        other_intercepts: out.other_intercepts,
    })
}

/// Euclidean norm of a slice.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
