use crate::loss::PerBatchStratifiedCellSampler;
use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use log::info;

/// Ridge prior strength λ on `e_cell` in the analytical phase-2 projection.
/// The Poisson MAP fits each cell's observed features and this Gaussian
/// prior stands in for the (infeasible) all-feature softmax partition.
pub(crate) const PHASE2_RIDGE: f32 = 1.0;

/// One cell's phase-2 solve result: `(cell_id, L2-direction latent, depth-free
/// biplot embedding, raw-count MAP norm, per-cell bias b_c)`.
type SolvedCell = (usize, Vec<f32>, Vec<f32>, f32, f32);

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
/// Separately returns a **biplot side embedding** (not the latent): the same
/// direction but with a *depth-free* magnitude — counts normalized to a common
/// total before the solve, so sequencing depth drops out (the raw-count MAP
/// magnitude tracks depth ≈0.8; normalized ≈0) — then co-scaled onto the gene
/// dictionary by one global gauge `median‖e_feat‖/median‖e_cell‖`. This puts
/// cells and genes on one scale for a joint cell–gene plot; it is not used for
/// clustering.
///
/// Returns `(cell_nrms, scaled)`. `cell_nrms` is the **un-normalized** baseline
/// MAP norm — the empty-droplet QC keys on it (empties solve to ≈0 only on raw
/// counts; normalizing to the common total would upscale and hide them).
/// `scaled` is the `[n_cells × H]` row-major biplot side embedding.
pub(crate) fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    use crate::cell_projection::solve_one_cell;
    use anyhow::Context;
    use candle_util::candle_core::Tensor;
    use rayon::prelude::*;

    let h = model.embedding_dim;

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let mut e_out: Vec<f32> = model.e_cell.flatten_all()?.to_vec1()?;
    let mut b_out: Vec<f32> = model.b_cell.to_vec1()?;
    let mut cell_nrms = vec![0f32; n_cells];
    let mut scaled = vec![0f32; n_cells * h]; // biplot side embedding (returned)

    let mut cells: Vec<(u32, &[u32], &[f32])> = Vec::new();
    for s in cell_samplers {
        for (i, &cell) in s.active_cells.iter().enumerate() {
            let cf = &s.per_cell[i];
            cells.push((cell, &cf.features, &cf.counts));
        }
    }

    // Option-1 target total = median cell total (counts rescaled to this).
    let mut totals: Vec<f32> = cells
        .iter()
        .map(|(_, _, c)| c.iter().sum::<f32>())
        .collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target_total = if totals.is_empty() {
        1.0
    } else {
        totals[totals.len() / 2].max(1.0)
    };

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
            // Depth-normalized solve (counts → common total) → the biplot
            // side embedding: depth-free magnitude, same direction.
            let tot: f32 = counts.iter().sum::<f32>().max(1e-6);
            let sc = target_total / tot;
            let edges_n: Vec<(u32, f32)> = feats
                .iter()
                .zip(counts)
                .map(|(&f, &c)| (f, c * sc))
                .collect();
            let (e_n, _) = solve_one_cell(&edges_n, &feat_flat, &b_feat, h, lambda);
            (cell as usize, dir, e_n, nrm_map, b_c)
        })
        .collect();
    for (cell, dir, e_n, nrm_map, b_c) in solved {
        e_out[cell * h..(cell + 1) * h].copy_from_slice(&dir);
        scaled[cell * h..(cell + 1) * h].copy_from_slice(&e_n);
        cell_nrms[cell] = nrm_map;
        b_out[cell] = b_c;
    }

    // Co-scale the biplot side embedding onto the gene dictionary's scale (one
    // global scalar — no per-cell tuning). The clustering latent (`e_out`) is
    // left as unit directions; only `scaled` is gauged.
    let feat_scale = median_row_norm(&feat_flat, h);
    let mut cell_scales: Vec<f32> = scaled
        .chunks_exact(h)
        .map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt())
        .filter(|&n| n > 1e-8)
        .collect();
    let cell_scale = median_of(&mut cell_scales);
    let gauge = if cell_scale > 1e-8 {
        feat_scale / cell_scale
    } else {
        1.0
    };
    info!(
        "Phase 2 — latent = L2 direction; biplot side embedding gauged ‖e_feat‖/‖e_cell‖ = {feat_scale:.3}/{cell_scale:.3} = {gauge:.3}"
    );
    if (gauge - 1.0).abs() > 1e-6 {
        for x in &mut scaled {
            *x *= gauge;
        }
    }

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
    Ok((cell_nrms, scaled))
}

/// Median of a slice (sorts in place; upper-median for even length). Diagnostic.
fn median_of(xs: &mut [f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Median L2 norm over the rows of a row-major `[n × h]` matrix. Diagnostic.
fn median_row_norm(flat: &[f32], h: usize) -> f32 {
    if h == 0 || flat.is_empty() {
        return 0.0;
    }
    let mut norms: Vec<f32> = flat
        .chunks_exact(h)
        .map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    median_of(&mut norms)
}
