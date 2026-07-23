//! Per-**cell** phase-2 projection: re-fit every cell's embedding `θ_c` (and the
//! per-cell bias `b_cell`) against the frozen feature dictionary, via the shared
//! block Poisson-MAP SGD ([`block_sgd`]). The pseudobulk sibling is
//! [`super::pseudobulk`]; both drive the same engine.

use super::block_sgd;
use super::CellBatchDivisor;
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
/// written into the `b_cell` var alongside `e_cell` (consistent with `faba gem`).
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
/// On the **bge** path the stored latent (`model.e_cell`) is the **L2 direction**
/// of the Poisson-MAP embedding — depth-robust and best for that pipeline's
/// Euclidean clustering (storing the magnitude there blurs cell types: the
/// magnitude axis ≈ profile specialization, roughly orthogonal to identity, a
/// measured ~7–11pt purity loss). The **gem** path instead stores the latent
/// **raw** (magnitude kept) — see below — and normalizes downstream (or uses
/// cosine) as an explicit clustering choice, so a zero-signal cell can't be turned
/// into a fabricated unit direction.
///
/// Phase 2 projection. Without `unspliced_rows` (bge): one combined Poisson-MAP
/// per cell → identity `e_cell`, stored as the L2 direction `dir(θ)`. With
/// `unspliced_rows` (gem β-sharing): identity is resolved by the **spliced** edges
/// (`e_cell = θ`, mature mRNA = current state) and stored **raw** — no post-hoc
/// unit-norm, so `‖θ‖` stays the activity/QC signal; then, holding θ fixed, the
/// cell's **unspliced** edges are fit for an analytic velocity increment `δ`
/// against the shared `β_g` and stored **raw** too (magnitude = speed, direction =
/// velocity). δ is a directed Poisson-MAP residual in θ's own frame (not a second
/// independent projection); a cell missing either modality gets `δ = 0`. The
/// nascent state is simply `θ + δ` = `latent + velocity` (not materialized). Any
/// per-gene velocity readout comes from the in-model `δ_g` (`--delta-l2`), not a
/// post-hoc aggregate.
///
/// See [`Phase2Result`] for what comes back.
#[allow(clippy::too_many_arguments)] // frozen dictionary + samplers + batch divisor + splice mask
pub(crate) fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
    batch_divisor: Option<CellBatchDivisor>,
    unspliced_rows: Option<&[bool]>,
    joint: bool,
) -> anyhow::Result<Phase2Result> {
    use anyhow::Context;
    use candle_util::candle_core::Tensor;

    let h = model.embedding_dim;

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let feat_flat: Vec<f32> = model.e_feat.flatten_all()?.to_vec1()?;
    let cells = collect_sampler_cells(cell_samplers);

    info!(
        "Phase 2 — cell-block Poisson SGD over {n_cells} cells ({} with edges) on {dev:?}, \
         full log-partition, ridge λ={lambda}{}",
        cells.len(),
        if batch_divisor.is_some() {
            ", μ_residual batch-divided"
        } else {
            ""
        }
    );

    let out = block_sgd::project_cells(
        &block_sgd::Phase2Input {
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
        },
        &cells,
        batch_divisor,
        unspliced_rows,
    )?;

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
    let b_feat_t = Tensor::from_vec(b_feat, n_features, dev)?;
    {
        let vars = varmap.data().lock().unwrap();
        vars.get("b_feat")
            .context("b_feat var missing")?
            .set(&b_feat_t)?;
    }
    model.b_feat = b_feat_t;

    // `cell_nrms` is the un-normalized MAP norm the empty-droplet QC keys on, so it
    // is always read off the RAW θ — before the bge path replaces the stored latent
    // with its unit direction. Post-gauge-fix this is the distance from the
    // population mean, which is the more useful "how much signal" reading anyway.
    let cell_nrms: Vec<f32> = out
        .theta
        .chunks_exact(h)
        .map(|t| t.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    let mut e_out = out.theta;
    if unspliced_rows.is_none() {
        // bge: store the L2 direction — depth-robust and best for that pipeline's
        // Euclidean clustering. gem keeps the magnitude (see the doc above).
        // In place — a per-row helper returning a `Vec` would allocate per cell.
        for row in e_out.chunks_exact_mut(h) {
            let n = norm(row);
            if n > 1e-8 {
                row.iter_mut().for_each(|x| *x /= n);
            }
        }
    }

    let e_t = Tensor::from_vec(e_out, (n_cells, h), dev)?;
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
    })
}

/// Euclidean norm of a slice — the `√Σx²` the bge L2-direction store needs.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
