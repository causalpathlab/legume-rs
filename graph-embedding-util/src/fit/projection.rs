use crate::cell_projection::{solve_cell_increment, solve_one_cell};
use crate::data::UnifiedData;
use crate::loss::PerBatchStratifiedCellSampler;
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use log::info;
use matrix_util::dmatrix_util::adjust_by_poisson_ratio;
use nalgebra::DMatrix;

mod cell_sgd;

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
    /// read before phase 2 by the Newton pb readout and are never re-gauged, so the
    /// cell-lift has to add this back before taking `dist2(θ_c, θ_p)`. Comparing
    /// across the two frames displaces every cell by `‖θ̄‖` — 88 on the reference
    /// fit, against a median `‖θ‖` of 5.6 after centring.
    pub theta_mean: Vec<f32>,
}

/// Ridge prior strength λ on `e_cell` in the phase-2 projection.
///
/// A **mild** Gaussian prior, not a load-bearing bound: the cell-block SGD
/// ([`cell_sgd`]) sums the log-partition over every feature, which is what
/// identifies `θ`. The per-pseudobulk and held-out-gene solves in
/// [`crate::cell_projection`] still fit observed features only, and there this
/// same λ *is* the only thing standing in for the partition.
pub(crate) const PHASE2_RIDGE: f32 = 1.0;

/// Phase-2 batch correction, mirroring `senna svd`/`topic`: divide each cell's
/// counts by its finest-pseudobulk `μ_residual` fold-factor before the
/// Poisson-MAP projection, so `e_cell` fits the de-batched signal. Built only
/// when the collapse fit a `μ_residual` (>1 batch); a no-op otherwise.
#[derive(Clone, Copy)]
pub(crate) struct CellBatchDivisor<'a> {
    /// `[n_features × n_pb]` batch fold-factor on the **unified** feature axis,
    /// so a cell's feature id indexes a row directly (no remap).
    pub mu_residual: &'a DMatrix<f32>,
    /// Cell id → finest-pseudobulk id (the `μ_residual` column to divide by).
    pub cell_to_pb: &'a [usize],
}

/// Divide one cell's `(feature, count)` edges by its pseudobulk batch fold-factor,
/// reusing matrix-util's [`adjust_by_poisson_ratio`] — the same self-normalizing
/// divide (`λ = Σx/Σd`, depth preserved for `b_cell`) `senna svd`/`topic` apply via
/// the `CscMatrix` trait, here straight on the cell's counts (no per-cell CSC).
/// `feats` index `μ_residual` rows directly.
fn adjust_cell_edges(
    feats: &[u32],
    counts: &[f32],
    pb: usize,
    mu_residual: &DMatrix<f32>,
) -> Vec<(u32, f32)> {
    let mut vals = counts.to_vec();
    adjust_by_poisson_ratio(&mut vals, |k| mu_residual[(feats[k] as usize, pb)]);
    feats.iter().copied().zip(vals).collect()
}

/// Flatten the per-batch samplers into one `(cell_id, features, counts)` list,
/// borrowing each cell's edge slices. Shared by the baseline and dual-axis
/// phase-2 projections (both walk the same active cells).
pub(crate) fn collect_sampler_cells(
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

/// One cell's `(feature, count)` edges, batch-divided by its pseudobulk
/// fold-factor when correction is on, else the raw edges.
pub(crate) fn cell_edges(
    cell: u32,
    feats: &[u32],
    counts: &[f32],
    batch_divisor: Option<CellBatchDivisor>,
) -> Vec<(u32, f32)> {
    match batch_divisor {
        Some(bd) => adjust_cell_edges(feats, counts, bd.cell_to_pb[cell as usize], bd.mu_residual),
        None => feats.iter().copied().zip(counts.iter().copied()).collect(),
    }
}

/// Phase 2 — project every cell onto the fixed feature dictionary and overwrite
/// the `e_cell` var. The per-cell bias is fitted (to absorb library size) and
/// written into the `b_cell` var alongside `e_cell` (consistent with `faba gem`).
///
/// The solve itself is a **cell-block Poisson SGD** ([`cell_sgd`]): with the
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

    let out = cell_sgd::project_cells(
        &cell_sgd::Phase2Input {
            feat: &feat_flat,
            b_feat: &b_feat,
            h,
            n_cells,
            lambda,
            dev,
        },
        &cells,
        batch_divisor,
        unspliced_rows,
    )?;

    /////////////////////////////////////////////////
    // Fold the gauge shift back into `b_feat`     //
    /////////////////////////////////////////////////

    // `cell_sgd` removed the population mean from each latent. That is only a
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

/// Euclidean norm of a slice — the `√Σx²` this module and [`cell_sgd`] both need.
pub(super) fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Split one node's `(feature, count)` edges by the unspliced mask and run the
/// dual analytic solve: identity `θ` from the spliced edges, then the velocity
/// increment `δ` from the unspliced edges holding `θ` fixed (same likelihood
/// frame, so `θ + δ` is coherent). The per-**pseudobulk** readout's solver — the
/// per-cell path is [`cell_sgd`]. `frozen_e` is row-major `[n_features × h]`.
///
/// Returns `(θ, δ)`; `δ` is **empty** — not zero-filled — when velocity is
/// undefined (empty identity or no unspliced edges), so callers can skip the copy
/// into a pre-zeroed buffer.
fn solve_node_splice(
    edges: &[(u32, f32)],
    unspliced_rows: &[bool],
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    lambda: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut spliced: Vec<(u32, f32)> = Vec::with_capacity(edges.len());
    let mut unspliced: Vec<(u32, f32)> = Vec::with_capacity(edges.len());
    for &(f, c) in edges {
        if unspliced_rows[f as usize] {
            unspliced.push((f, c));
        } else {
            spliced.push((f, c));
        }
    }
    let (theta, _b_c) = solve_one_cell(&spliced, frozen_e, frozen_b, h, lambda);
    let velocity = if norm(&theta) > 1e-8 && !unspliced.is_empty() {
        solve_cell_increment(&unspliced, &theta, frozen_e, frozen_b, h, lambda).0
    } else {
        Vec::new()
    };
    (theta, velocity)
}

/// Per-level pseudobulk phase-2 velocity readout: the analytic identity `θ_pb`
/// and velocity increment `δ_pb` for every pb node of one collapse level, each
/// flattened `[n_pb × h]` row-major. Produced by [`project_pbs_phase2`] for the
/// lineage-DAG path — `δ_pb` orients the pb-DAG structure term, `θ_pb` are the
/// latent landmarks the phase-2 cell lift attaches to.
pub struct PbLevelVelocity {
    pub n_pb: usize,
    /// Identity `θ_pb`, `[n_pb × h]` row-major (raw spliced Poisson-MAP).
    pub theta: Vec<f32>,
    /// Velocity `δ_pb`, `[n_pb × h]` row-major; zero rows where undefined.
    pub delta: Vec<f32>,
}

/// Phase-2 **pseudobulk** velocity readout (gem β-sharing / lineage-DAG path).
/// Analytically re-projects every pb node of every level onto the frozen feature
/// dictionary — identity `θ_pb` from its spliced aggregate, velocity `δ_pb` from
/// its unspliced aggregate with `θ_pb` fixed — exactly as [`project_cells_phase2`]
/// does per cell, but at pseudobulk resolution. Reuses the pb aggregates already
/// built for phase 1 (`pb_blobs[level].triplets`), which are already
/// batch-corrected, so no batch divisor is applied. `frozen_e` is row-major
/// `[n_features × h]`.
///
/// Returns one [`PbLevelVelocity`] per level, in `pb_blobs` order (coarsest→finest).
pub(crate) fn project_pbs_phase2(
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    pb_blobs: &[UnifiedData],
    unspliced_rows: &[bool],
    lambda: f64,
) -> anyhow::Result<Vec<PbLevelVelocity>> {
    use crate::progress::new_progress_bar;
    use indicatif::ParallelProgressIterator;
    use rayon::prelude::*;

    // One bar across every level's nodes, so the sequential level loop reads as a
    // single span of work rather than a bar that restarts per level.
    let solve_bar =
        new_progress_bar(pb_blobs.iter().map(UnifiedData::n_cells).sum::<usize>() as u64);
    solve_bar.set_message("phase-2 per-pseudobulk Poisson MAP");

    let mut out = Vec::with_capacity(pb_blobs.len());
    for pb in pb_blobs {
        let n_pb = pb.n_cells();
        // Group the pb's (feature, count) edges by pb-node id.
        let mut per_pb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_pb];
        for t in &pb.triplets {
            per_pb[t.cell as usize].push((t.feature, t.count));
        }
        let solved: Vec<(Vec<f32>, Vec<f32>)> = per_pb
            .par_iter()
            .progress_with(solve_bar.clone())
            .map(|edges| {
                solve_node_splice(edges, unspliced_rows, frozen_e, frozen_b, h, lambda)
            })
            .collect();
        let mut theta = vec![0f32; n_pb * h];
        let mut delta = vec![0f32; n_pb * h];
        for (p, (th, dl)) in solved.into_iter().enumerate() {
            let s = p * h;
            theta[s..s + h].copy_from_slice(&th);
            if !dl.is_empty() {
                delta[s..s + h].copy_from_slice(&dl);
            }
        }
        out.push(PbLevelVelocity { n_pb, theta, delta });
    }
    solve_bar.finish_and_clear();
    Ok(out)
}

#[cfg(test)]
mod tests;
