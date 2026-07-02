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
    /// L2-direction identity latent — the stored `e_cell` row. On the
    /// β-sharing (splice) path this is `dir(θ)` from the **spliced** edges only
    /// (mature mRNA = current cell state); otherwise the combined projection.
    dir: Vec<f32>,
    /// Un-normalized raw-count MAP norm the empty-droplet QC keys on (spliced
    /// magnitude `‖θ‖` on the splice path).
    nrm_map: f32,
    /// Fitted per-cell bias `b_c` (absorbs library size).
    b_c: f32,
    /// `dir(φ)` — the nascent/unspliced-only latent (splice path only; empty
    /// otherwise). Where transcription is currently pointed = the near future.
    nascent: Vec<f32>,
    /// Velocity `δ = dir(φ) − dir(θ)` (splice path only; empty otherwise).
    velocity: Vec<f32>,
}

/// Phase-2 splice outputs for the β-sharing (`feat_factor`) path. Both are flat
/// `[n_cells × h]` row-major in global cell-id order (cells absent from the
/// samplers stay zero).
pub(crate) struct SpliceProjection {
    /// Nascent/unspliced-only latent `dir(φ)` per cell.
    pub nascent: Vec<f32>,
    /// Velocity `δ = dir(φ) − dir(θ)` per cell — the spliced→unspliced
    /// (current→nascent) shift on the cell axis. Zero for a cell missing either
    /// modality (no velocity is defined).
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
/// The stored latent (`model.e_cell`) is the **L2 direction** of the baseline
/// Poisson-MAP embedding — depth-robust and best for clustering/UMAP. (Storing
/// the magnitude instead blurs cell types: the magnitude axis ≈ profile
/// specialization, roughly orthogonal to cell-type identity, so Euclidean
/// clustering mixes it in — a measured ~7–11pt purity loss.)
///
/// Returns `cell_nrms`, the **un-normalized** baseline MAP norm — the
/// empty-droplet QC keys on it (empties solve to ≈0 only on raw counts;
/// normalizing would upscale and hide them). The joint cell–gene plot is
/// handled post-hoc by [`crate::postprocess::feature_coembedding`], which
/// re-embeds features onto the cell manifold (this replaced the old per-cell
/// biplot gauge that lived here).
/// Phase 2 projection. Without `unspliced_rows` (bge): one combined Poisson-MAP
/// per cell → identity `e_cell`. With `unspliced_rows` (gem β-sharing): identity
/// is resolved by the **spliced** edges only (`e_cell = dir(θ)`, mature mRNA =
/// current state), and the same pass also positions each cell on its **unspliced**
/// edges (`φ`, nascent = near future) against the shared `β_g`, yielding the
/// nascent latent `dir(φ)` and the velocity `δ = dir(φ) − dir(θ)`. Both views
/// are L2-normalized first, so `δ` is a direction change, not a depth change; a
/// cell missing either modality gets `δ = 0` (no velocity defined).
///
/// Returns `(cell_nrms, splice)`, where `cell_nrms` is the un-normalized MAP norm
/// the empty-droplet QC keys on (spliced magnitude `‖θ‖` on the splice path) and
/// `splice` carries the flat `[n_cells × h]` nascent + velocity buffers (`None`
/// on the bge path).
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
    use crate::cell_projection::solve_one_cell;
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
    let solved: Vec<SolvedCell> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            // Batch-divide the counts (μ_residual fold-factor) when correction is
            // on, then project. The fitted intercept `b_c` absorbs library size
            // and is kept (written to `b_cell`).
            let edges = cell_edges(cell, feats, counts, batch_divisor);
            match unspliced_rows {
                // bge: one combined projection = identity.
                None => {
                    let (e_map, b_c) = solve(&edges);
                    SolvedCell {
                        cell: cell as usize,
                        dir: l2_direction(&e_map),
                        nrm_map: norm(&e_map),
                        b_c,
                        nascent: Vec::new(),
                        velocity: Vec::new(),
                    }
                }
                // gem β-sharing: identity = spliced θ, plus nascent φ and δ = φ − θ.
                // Both modalities index the shared β_g, so θ / φ are comparable.
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
                    let (phi, _) = solve(&unspliced);
                    let (theta_n, phi_n) = (norm(&theta), norm(&phi));
                    let td = l2_direction(&theta);
                    let pd = l2_direction(&phi);
                    // δ = dir(φ) − dir(θ); undefined (→ 0) if either view is empty.
                    let velocity = if theta_n > 1e-8 && phi_n > 1e-8 {
                        pd.iter().zip(&td).map(|(a, b)| a - b).collect()
                    } else {
                        vec![0f32; h]
                    };
                    let nascent = if phi_n > 1e-8 { pd } else { vec![0f32; h] };
                    SolvedCell {
                        cell: cell as usize,
                        dir: td,
                        nrm_map: theta_n,
                        b_c: b_s,
                        nascent,
                        velocity,
                    }
                }
            }
        })
        .collect();

    let split = unspliced_rows.is_some();
    let mut nascent_flat = split.then(|| vec![0f32; n_cells * h]);
    let mut velocity_flat = split.then(|| vec![0f32; n_cells * h]);
    for sc in solved {
        let s = sc.cell * h;
        e_out[s..s + h].copy_from_slice(&sc.dir);
        cell_nrms[sc.cell] = sc.nrm_map;
        b_out[sc.cell] = sc.b_c;
        if let Some(nf) = nascent_flat.as_mut() {
            nf[s..s + h].copy_from_slice(&sc.nascent);
        }
        if let Some(vf) = velocity_flat.as_mut() {
            vf[s..s + h].copy_from_slice(&sc.velocity);
        }
    }

    info!(
        "Phase 2 — {} (per-cell Poisson MAP, ridge λ={lambda}{})",
        if split {
            "identity = spliced θ direction (+ nascent φ, velocity δ=φ−θ)"
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

    let splice = match (nascent_flat, velocity_flat) {
        (Some(nascent), Some(velocity)) => Some(SpliceProjection { nascent, velocity }),
        _ => None,
    };
    Ok((cell_nrms, splice))
}
