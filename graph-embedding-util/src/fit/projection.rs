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

/// One cell's phase-2 solve result: `(cell_id, L2-direction latent,
/// raw-count MAP norm, per-cell bias b_c)`.
type SolvedCell = (usize, Vec<f32>, f32, f32);

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
#[allow(clippy::too_many_arguments)] // frozen dictionary + cell samplers + batch divisor + aux term
pub(crate) fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
    batch_divisor: Option<CellBatchDivisor>,
    aux: Option<&dyn crate::cell_projection::PerCellAuxTerm>,
) -> anyhow::Result<Vec<f32>> {
    use crate::cell_projection::solve_one_cell_aux;
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
    let solved: Vec<SolvedCell> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            // Baseline MAP → L2 direction (the latent) + the depth-coupled norm
            // the empty-droplet QC keys on. The fitted intercept `b_c` absorbs
            // library size and is kept (written to `b_cell`), consistent with
            // faba gem. When batch correction is on, the counts are first divided
            // by their pseudobulk fold-factor (depth preserved by the trait's
            // self-normalizing scale, so empties still solve to ≈0).
            let edges = cell_edges(cell, feats, counts, batch_divisor);
            // Joint MAP when an aux term (e.g. m6A binomial) is present, else the
            // plain Poisson solve. The aux term indexes its per-cell data by `cell`.
            let (e_map, b_c, _extra) =
                solve_one_cell_aux(cell as usize, &edges, &feat_flat, &b_feat, h, lambda, aux);
            let nrm_map = norm(&e_map);
            (cell as usize, l2_direction(&e_map), nrm_map, b_c)
        })
        .collect();
    for (cell, dir, nrm_map, b_c) in solved {
        e_out[cell * h..(cell + 1) * h].copy_from_slice(&dir);
        cell_nrms[cell] = nrm_map;
        b_out[cell] = b_c;
    }

    info!(
        "Phase 2 — latent = L2 direction (per-cell Poisson MAP, ridge λ={lambda}{})",
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
    Ok(cell_nrms)
}

/// Dual phase-2 projection for the axis δ (β-sharing spliced/unspliced model).
///
/// With the feature side fixed to a per-gene `β` (a gene's spliced and unspliced
/// rows BOTH embed as `β_g`), the splice signal cannot live on the gene side
/// (it would be non-identifiable against an equal-and-opposite cell-axis shift).
/// Instead we recover it on the CELL axis: position each cell against the frozen
/// `β` TWICE — once on its spliced edges (`θ`) and once on its unspliced edges
/// (`φ`) — using the same Poisson-MAP solve as [`project_cells_phase2`], then
/// store the depth-removed shift `δ_cell = dir(φ) − dir(θ)` (each projection
/// L2-normalized first, so δ measures a direction change, not a depth change).
/// Because both spliced and unspliced rows index the same `β_g`, the two solves
/// share one dictionary — `θ` / `φ` are directly comparable.
///
/// Returns a flat `[n_cells × H]` row-major buffer in global cell-id order
/// (cells absent from the samplers stay zero). `unspliced_rows[f]` flags the
/// unspliced feature rows; `batch_divisor` applies the same μ_residual divide as
/// the main projection before the modality split.
pub(crate) fn project_axis_delta(
    model: &JointEmbedModel,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    unspliced_rows: &[bool],
    lambda: f64,
    batch_divisor: Option<CellBatchDivisor>,
) -> anyhow::Result<Vec<f32>> {
    use crate::cell_projection::solve_one_cell;
    use rayon::prelude::*;

    let h = model.embedding_dim;
    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let feat_flat: Vec<f32> = model.e_feat.flatten_all()?.to_vec1()?;

    let cells = collect_sampler_cells(cell_samplers);

    let solved: Vec<(usize, Vec<f32>)> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            let edges = cell_edges(cell, feats, counts, batch_divisor);
            // Split this cell's edges by modality; both index the shared β_g.
            let mut spliced: Vec<(u32, f32)> = Vec::with_capacity(edges.len());
            let mut unspliced: Vec<(u32, f32)> = Vec::with_capacity(edges.len());
            for &(f, c) in &edges {
                if unspliced_rows[f as usize] {
                    unspliced.push((f, c));
                } else {
                    spliced.push((f, c));
                }
            }
            let (theta, _) = solve_one_cell(&spliced, &feat_flat, &b_feat, h, lambda);
            let (phi, _) = solve_one_cell(&unspliced, &feat_flat, &b_feat, h, lambda);
            let (td, pd) = (l2_direction(&theta), l2_direction(&phi));
            let delta: Vec<f32> = pd.iter().zip(&td).map(|(a, b)| a - b).collect();
            (cell as usize, delta)
        })
        .collect();

    let mut out = vec![0f32; n_cells * h];
    for (cell, delta) in solved {
        out[cell * h..(cell + 1) * h].copy_from_slice(&delta);
    }
    info!("Phase 2 — dual-projection axis δ (spliced θ vs unspliced φ) for {n_cells} cells");
    Ok(out)
}
