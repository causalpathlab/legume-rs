use crate::loss::PerBatchStratifiedCellSampler;
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use log::info;
use matrix_util::dmatrix_util::adjust_by_poisson_ratio;
use nalgebra::DMatrix;

/// Ridge prior strength λ on `e_cell` in the analytical phase-2 projection.
/// The Poisson MAP fits each cell's observed features and this Gaussian
/// prior stands in for the (infeasible) all-feature softmax partition.
pub(crate) const PHASE2_RIDGE: f32 = 1.0;

/// One cell's phase-2 solve result.
struct SolvedCell {
    /// Global cell id (row into `e_cell` / `b_cell`).
    cell: usize,
    /// The stored `e_cell` row. On the **bge** path this is the L2 direction
    /// `dir(θ)` (depth-robust for that pipeline); on the **gem** β-sharing (splice)
    /// path it is the **raw** identity MAP `θ` from the spliced edges — magnitude
    /// kept (no post-hoc unit-norm), so `‖θ‖` remains the activity/QC signal and a
    /// zero-signal cell stays at the origin instead of a fabricated unit direction.
    latent: Vec<f32>,
    /// Un-normalized raw-count MAP norm the empty-droplet QC keys on (`‖θ‖` on the
    /// splice path — identical to `‖latent‖` there since the gem latent is raw).
    nrm_map: f32,
    /// Fitted per-cell bias `b_c` (absorbs library size).
    b_c: f32,
    /// Raw velocity increment `δ` (gem splice path only; empty otherwise) — the
    /// Poisson-MAP shift explaining the unspliced edges with the identity held
    /// fixed. Magnitude = speed, direction = velocity; no normalization.
    velocity: Vec<f32>,
}

/// Phase-2 splice output for the β-sharing (`feat_factor`) path: the raw per-cell
/// velocity increment `δ`, flat `[n_cells × h]` row-major in global cell-id order
/// (cells absent from the samplers, or with no unspliced edges, stay zero).
pub(crate) struct SpliceProjection {
    /// Raw velocity increment `δ` per cell — the analytic Poisson-MAP shift fit to
    /// the unspliced edges with the identity `θ` held fixed. Magnitude = speed,
    /// direction = velocity (no normalization). The nascent state is simply
    /// `θ + δ` = `latent + velocity`, so it is not materialized separately.
    pub velocity: Vec<f32>,
}

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

/// One cell's `(feature, count)` edges, batch-divided by its pseudobulk
/// fold-factor when correction is on, else the raw edges.
fn cell_edges(
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

/// L2-normalize to a unit direction; returns the input unchanged when its norm
/// underflows (an all-zero / empty solve).
fn l2_direction(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Phase 2 — project every cell onto the fixed feature dictionary, in
/// parallel, and overwrite the `e_cell` var. The per-cell bias is fitted
/// (to absorb library size) and written into the `b_cell` var alongside
/// `e_cell` (consistent with `faba gem`).
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
/// Returns `(cell_nrms, splice)`, where `cell_nrms` is the un-normalized MAP norm
/// the empty-droplet QC keys on (`‖θ‖`, identical to `‖latent‖` on the splice path)
/// and `splice` carries the `[n_cells × h]` raw velocity buffer (`None` on bge).
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
) -> anyhow::Result<(Vec<f32>, Option<SpliceProjection>)> {
    use crate::cell_projection::{solve_cell_increment, solve_one_cell};
    use anyhow::Context;
    use candle_util::candle_core::Tensor;
    use rayon::prelude::*;

    let h = model.embedding_dim;

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let mut e_out: Vec<f32> = model.e_cell.flatten_all()?.to_vec1()?;
    let mut b_out: Vec<f32> = model.b_cell.to_vec1()?;
    let mut cell_nrms = vec![0f32; n_cells];

    let cells = collect_sampler_cells(cell_samplers);
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

    let feat_flat: Vec<f32> = model.e_feat.flatten_all()?.to_vec1()?;
    let solve = |edges: &[(u32, f32)]| solve_one_cell(edges, &feat_flat, &b_feat, h, lambda);
    // Velocity: analytic Poisson-MAP increment δ holding the identity `e_base`
    // fixed (same likelihood/frame as the identity solve, so θ + δ is coherent).
    let solve_incr = |e_base: &[f32], edges: &[(u32, f32)]| {
        solve_cell_increment(edges, e_base, &feat_flat, &b_feat, h, lambda)
    };
    let solved: Vec<SolvedCell> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            // Batch-divide the counts (μ_residual fold-factor) when correction is
            // on, then project. The fitted intercept `b_c` absorbs library size
            // and is kept (written to `b_cell`).
            let edges = cell_edges(cell, feats, counts, batch_divisor);
            match unspliced_rows {
                // bge: one combined projection = identity, stored as the L2 direction.
                None => {
                    let (e_map, b_c) = solve(&edges);
                    SolvedCell {
                        cell: cell as usize,
                        latent: l2_direction(&e_map),
                        nrm_map: norm(&e_map),
                        b_c,
                        velocity: Vec::new(),
                    }
                }
                // gem β-sharing: identity = RAW spliced θ (magnitude kept); velocity =
                // the RAW analytic increment δ explaining the unspliced edges with θ
                // held fixed. Both index the shared β_g, so δ is a directed residual in
                // θ's own frame. No post-hoc unit-norm on either — the nascent state is
                // just θ + δ. δ = 0 when identity is empty or there are no unspliced edges.
                Some(un) => {
                    let mut spliced: Vec<(u32, f32)> = Vec::with_capacity(edges.len());
                    let mut unspliced: Vec<(u32, f32)> = Vec::with_capacity(edges.len());
                    for &(f, c) in &edges {
                        if un[f as usize] {
                            unspliced.push((f, c));
                        } else {
                            spliced.push((f, c));
                        }
                    }
                    let (theta, b_s) = solve(&spliced);
                    let theta_n = norm(&theta);
                    let velocity = if theta_n > 1e-8 && !unspliced.is_empty() {
                        let (delta, _) = solve_incr(&theta, &unspliced);
                        delta
                    } else {
                        vec![0f32; h]
                    };
                    SolvedCell {
                        cell: cell as usize,
                        latent: theta,
                        nrm_map: theta_n,
                        b_c: b_s,
                        velocity,
                    }
                }
            }
        })
        .collect();

    let split = unspliced_rows.is_some();
    let mut velocity_flat = split.then(|| vec![0f32; n_cells * h]);
    for sc in solved {
        let s = sc.cell * h;
        e_out[s..s + h].copy_from_slice(&sc.latent);
        cell_nrms[sc.cell] = sc.nrm_map;
        b_out[sc.cell] = sc.b_c;
        if let Some(vf) = velocity_flat.as_mut() {
            vf[s..s + h].copy_from_slice(&sc.velocity);
        }
    }

    info!(
        "Phase 2 — {} (per-cell Poisson MAP, ridge λ={lambda}{})",
        if split {
            "latent = raw spliced θ (magnitude kept) + raw velocity increment δ"
        } else {
            "latent = L2 direction"
        },
        if batch_divisor.is_some() {
            ", μ_residual batch-divided"
        } else {
            ""
        }
    );

    let e_t = Tensor::from_vec(e_out, (n_cells, h), dev)?;
    let b_t = Tensor::from_vec(b_out, n_cells, dev)?;
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

    let splice = velocity_flat.map(|velocity| SpliceProjection { velocity });
    Ok((cell_nrms, splice))
}
