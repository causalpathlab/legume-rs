//! Per-**cell** phase-2 projection: re-fit every cell's embedding `θ_c` (and the
//! per-cell bias `b_cell`) against the frozen feature dictionary, via the shared
//! block Poisson-MAP SGD ([`block_sgd`]). The pseudobulk sibling is
//! [`super::pseudobulk`]; both drive the same engine.

use super::block_sgd;
use super::encoder::{self, CellEncoder, DistillSpec};
use super::CellBatchFold;
use crate::loss::PerBatchStratifiedCellSampler;
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use log::info;

/// What phase 2 hands back to [`crate::fit::fit`].
pub(crate) struct Phase2Result {
    /// Un-normalized MAP norm the empty-droplet QC keys on (`‖θ‖`; identical to
    /// `‖latent‖` on the splice path, where the gem latent is stored raw).
    pub cell_nrms: Vec<f32>,
    /// The `[n_cells × h]` raw velocity increment `δ` (`None` on bge).
    pub velocity: Option<Vec<f32>>,
    /// The identity mean `θ̄` the gauge fix removed, `[h]`.
    ///
    /// **A frame marker, not a diagnostic.** Anything that compares the stored
    /// `e_cell` against a latent produced *outside* phase 2 must first agree on a
    /// frame. The pseudobulk landmarks in `pb_velocity` are the live case: they are
    /// read by the pb readout in a raw (un-gauged) frame, so the cell-lift has to
    /// add this back before taking `dist2(θ_c, θ_p)`. Comparing across the two
    /// frames displaces every cell by `‖θ̄‖` — 88 on the reference fit, against a
    /// median `‖θ‖` of 5.6 after centring.
    pub theta_mean: Vec<f32>,
    /// The distilled encoder that placed the cells, on the plain path; `None`
    /// when the block SGD did.
    pub cell_encoder: Option<CellEncoder>,
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
/// written into the `b_cell` var alongside `e_cell` (consistent with `senna gem`).
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
/// The stored latent (`model.e_cell`) is the **raw** embedding on every path —
/// magnitude kept, no unit-norm. A unit-normed store puts every cell on
/// `S^{H−1}`, and a layout of that is arcs and rings rather than the free
/// clouds the same objective gives `simba`; it also made the run's own cells a
/// different object from the cells the persisted encoder places at predict
/// time. A consumer that wants direction only takes it (cosine) as an explicit
/// choice, so a zero-signal cell is never turned into a fabricated unit vector.
///
/// Phase 2 projection. Without `unspliced_rows` (bge): one combined Poisson-MAP
/// per cell → identity `e_cell`, stored raw. With
/// `unspliced_rows` (gem β-sharing): identity is resolved by the **spliced** edges
/// (`e_cell = θ`, mature mRNA = current state), so `‖θ‖` stays the
/// activity/QC signal; then, holding θ fixed, the
/// cell's **unspliced** edges are fit for an analytic velocity increment `δ`
/// against the shared `β_g` and stored **raw** too (magnitude = speed, direction =
/// velocity). δ is a directed Poisson-MAP residual in θ's own frame (not a second
/// independent projection); a cell missing either modality gets `δ = 0`. The
/// nascent state is simply `θ + δ` = `latent + velocity` (not materialized). Any
/// per-gene velocity readout comes from the in-model `δ_g` (`--delta-l2`), not a
/// post-hoc aggregate.
///
/// `distill`, when given on the plain path, replaces the block SGD by the
/// distilled encoder ([`encoder`]): the phase-1 pseudobulk tables are the
/// targets, and every cell is encoded in one pass. The splice path (gem) keeps
/// the SGD regardless.
///
/// See [`Phase2Result`] for what comes back.
#[allow(clippy::too_many_arguments)] // frozen dictionary + samplers + batch fold + splice mask
pub(crate) fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
    batch_fold: Option<CellBatchFold>,
    unspliced_rows: Option<&[bool]>,
    joint: bool,
    distill: Option<&DistillSpec<'_>>,
) -> anyhow::Result<Phase2Result> {
    use anyhow::Context;
    use candle_util::candle_core::Tensor;

    let h = model.embedding_dim;

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let feat_flat: Vec<f32> = model.e_feat.flatten_all()?.to_vec1()?;
    let cells = collect_sampler_cells(cell_samplers);

    // The plain path with pseudobulk targets goes through the encoder; the
    // splice path (gem) and a call without targets keep the block SGD.
    let distill = match unspliced_rows {
        None => distill,
        Some(_) => None,
    };
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
        joint,
    };
    let (out, cell_encoder) = match distill {
        Some(spec) => {
            let (out, enc) = encoder::project_cells(&input, &cells, batch_fold, spec)?;
            (out, Some(enc))
        }
        None => (
            block_sgd::project_cells(&input, &cells, batch_fold, unspliced_rows)?,
            None,
        ),
    };

    /////////////////////////////////////////////////
    // Fold the gauge shift back into `b_feat`     //
    /////////////////////////////////////////////////

    // `block_sgd` removed the population mean from each latent. That is only a
    // *re-gauge* — every score unchanged — if the matching `⟨e_f, mean⟩` goes into
    // the per-feature bias. Skipping this would silently change the model, and
    // `b_feat` is a real output (`feature_bias.parquet`) that the held-out gene
    // projection also solves against.
    //
    // Which mean a row takes depends on which pass scored it: a spliced row is only
    // ever scored as `⟨e_f, θ⟩ + β_f + c`, an unspliced row as
    // `⟨e_f, θ + δ⟩ + β_f + c_u`. So spliced rows absorb `θ̄` and unspliced rows
    // absorb `θ̄ + δ̄`. Off the splice path every row is a "spliced" row.
    let mut b_feat = b_feat;
    let n_features = b_feat.len();
    let (tm, dm) = (&out.gauge.theta_mean, &out.gauge.delta_mean);
    for (f, b) in b_feat.iter_mut().enumerate() {
        let e_f = &feat_flat[f * h..(f + 1) * h];
        let is_unspliced = unspliced_rows.is_some_and(|un| un[f]);
        let shift: f32 = e_f
            .iter()
            .enumerate()
            .map(|(k, e)| e * (tm[k] + if is_unspliced { dm[k] } else { 0.0 }))
            .sum();
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
        velocity: out.velocity,
        theta_mean: out.gauge.theta_mean,
        cell_encoder,
    })
}

/// Euclidean norm of a slice.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
