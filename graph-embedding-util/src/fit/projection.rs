use crate::loss::PerBatchStratifiedCellSampler;
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use log::info;

/// Ridge prior strength λ on `e_cell` in the analytical phase-2 projection.
/// The Poisson MAP fits each cell's observed features and this Gaussian
/// prior stands in for the (infeasible) all-feature softmax partition.
pub(crate) const PHASE2_RIDGE: f32 = 1.0;

/// One cell's phase-2 solve result: `(cell_id, L2-direction latent,
/// raw-count MAP norm, per-cell bias b_c)`.
type SolvedCell = (usize, Vec<f32>, f32, f32);

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
pub(crate) fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
) -> anyhow::Result<Vec<f32>> {
    use crate::cell_projection::solve_one_cell;
    use anyhow::Context;
    use candle_util::candle_core::Tensor;
    use rayon::prelude::*;

    let h = model.embedding_dim;

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let mut e_out: Vec<f32> = model.e_cell.flatten_all()?.to_vec1()?;
    let mut b_out: Vec<f32> = model.b_cell.to_vec1()?;
    let mut cell_nrms = vec![0f32; n_cells];

    let mut cells: Vec<(u32, &[u32], &[f32])> = Vec::new();
    for s in cell_samplers {
        for (i, &cell) in s.active_cells.iter().enumerate() {
            let cf = &s.per_cell[i];
            cells.push((cell, &cf.features, &cf.counts));
        }
    }

    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

    let feat_flat: Vec<f32> = model.e_feat.flatten_all()?.to_vec1()?;
    let solved: Vec<SolvedCell> = cells
        .par_iter()
        .map(|&(cell, feats, counts)| {
            // Baseline MAP on RAW counts → L2 direction (the latent) + the
            // depth-coupled norm the empty-droplet QC keys on. The fitted
            // intercept `b_c` absorbs library size and is kept (written to
            // `b_cell`), consistent with faba gem.
            let edges: Vec<(u32, f32)> = feats.iter().zip(counts).map(|(&f, &c)| (f, c)).collect();
            let (e_map, b_c) = solve_one_cell(&edges, &feat_flat, &b_feat, h, lambda);
            let nrm_map = norm(&e_map);
            let dir: Vec<f32> = if nrm_map > 1e-8 {
                e_map.iter().map(|x| x / nrm_map).collect()
            } else {
                e_map.clone()
            };
            (cell as usize, dir, nrm_map, b_c)
        })
        .collect();
    for (cell, dir, nrm_map, b_c) in solved {
        e_out[cell * h..(cell + 1) * h].copy_from_slice(&dir);
        cell_nrms[cell] = nrm_map;
        b_out[cell] = b_c;
    }

    info!("Phase 2 — latent = L2 direction (per-cell Poisson MAP, ridge λ={lambda})");

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
