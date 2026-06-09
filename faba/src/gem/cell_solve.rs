//! Analytical phase-2: project every cell onto the **frozen** feature
//! dictionary, in parallel.
//!
//! Once phase 1 has fixed the feature side (β, z, δ, γ, biases), each
//! cell's embedding is independent of every other cell — the cell-axis
//! objective decouples. So instead of running SGD over a shared `e_cell`
//! `VarMap`, we solve each cell directly and fan the cells out across cores
//! with rayon.
//!
//! **Objective (Poisson MAP on observed features).** For cell `c`, with
//! frozen feature embeddings `e_f` / biases `b_f`, model its observed
//! counts `n_f` as Poisson with rate `μ_f = exp(⟨e_f, e_c⟩ + b_f + b_c)`
//! and put a Gaussian (ridge) prior `N(0, 1/λ)` on `e_c`. The exact softmax
//! MLE would normalise over *all* features (the partition NCE only ever
//! approximated); at this scale that's infeasible, so we fit the cell's
//! observed features and let the ridge prior stand in for the partition
//! (bounding `e_c`, which fitting positives alone would push to ∞). The
//! per-cell intercept `b_c` absorbs library size.
//!
//! Each Newton/IRLS step is a small `(H+1)×(H+1)` solve:
//! ```text
//! θ = [e_c; b_c],  ẽ_f = [e_f; 1],  s_f = ⟨θ, ẽ_f⟩ + b_f,  μ_f = exp(s_f)
//! θ ← θ + (Σ_f μ_f ẽ_f ẽ_fᵀ + λP)⁻¹ (Σ_f (n_f − μ_f) ẽ_f − λP θ)
//! ```
//! with `P = diag(1,…,1, 0)` (ridge on `e_c`, not the intercept).

use super::common::candle_core;
use anyhow::{Context, Result};
use candle_core::Tensor;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::args::GemArgs;
use super::feature_table::FeatureTable;
use super::model::GemModel;
use super::pseudobulk::AxisPools;

const MAX_IRLS_ITERS: usize = 8;
const SCORE_CLAMP: f64 = 30.0;
/// Count-comp rows share the count bias slot (matches the sampler's
/// `COUNT_BIAS_MODALITY`); modifiers use their own column.
const COUNT_BIAS_MODALITY: u32 = 0;

/// One distinct feature-row identity (the embed_and_bias_rows inputs that
/// all collapse to the same `e_f` / `b_f`).
struct Identity {
    gene: u32,
    q_modality: u32,
    region: u32,
    is_agg: bool,
    bias_modality: u32,
}

/// Frozen `e_f` (row-major `[n_id × H]`) and `b_f` (`[n_id]`). The row
/// width `H` is passed alongside (the solver already threads it).
struct FrozenFeatures {
    e: Vec<f32>,
    b: Vec<f32>,
}

/// Solve `e_cell` (and `b_cell`) by projecting every cell onto the frozen
/// feature side, then overwrite the model's `e_cell` / `b_cell` vars.
pub fn solve_cell_embeddings(
    model: &mut GemModel,
    table: &FeatureTable,
    pools: &AxisPools,
    args: &GemArgs,
) -> Result<()> {
    let n_cells = model.n_cells;
    let h = model.embedding_dim;
    if n_cells == 0 {
        return Ok(());
    }

    // 1. Distinct identities + per-cell (identity, count) lists.
    let (ids, per_cell) = collect_identities(table, pools, n_cells);
    // 2. Frozen feature embeddings (one batched device pass → CPU).
    let frozen = embed_identities(model, &ids, h)?;
    // 3. Per-cell Poisson MAP IRLS, fanned out across cores.
    let lambda = args.phase2_ridge.max(0.0) as f64;
    let solved: Vec<(Vec<f32>, f32)> = per_cell
        .par_iter()
        .map(|feats| solve_one_cell(feats, &frozen, h, lambda))
        .collect();
    // 4. Assemble + write back into the model's vars (and cached fields).
    let mut e_flat = vec![0f32; n_cells * h];
    let mut b_flat = vec![0f32; n_cells];
    for (c, (e_c, b_c)) in solved.iter().enumerate() {
        e_flat[c * h..(c + 1) * h].copy_from_slice(e_c);
        b_flat[c] = *b_c;
    }
    let e_t = Tensor::from_vec(e_flat, (n_cells, h), &model.dev)?;
    let b_t = Tensor::from_vec(b_flat, n_cells, &model.dev)?;
    {
        let vars = model.varmap.data().lock().unwrap();
        vars.get("e_cell")
            .context("e_cell var missing")?
            .set(&e_t)?;
        vars.get("b_cell")
            .context("b_cell var missing")?
            .set(&b_t)?;
    }
    model.e_cell = e_t;
    model.b_cell = b_t;
    Ok(())
}

fn intern(
    ids: &mut Vec<Identity>,
    map: &mut FxHashMap<(u32, u32, u32, bool), u32>,
    id: Identity,
) -> u32 {
    let key = (id.gene, id.q_modality, id.region, id.is_agg);
    if let Some(&x) = map.get(&key) {
        return x;
    }
    let x = ids.len() as u32;
    map.insert(key, x);
    ids.push(id);
    x
}

/// Walk the cell-axis pools once: intern each distinct feature identity and
/// append `(identity, count)` to its cell's list. Mirrors how the sampler
/// builds positives (AGG → β_g; count-comp → splice modality, shared count
/// bias; modifier-comp → its own modality + region).
fn collect_identities(
    _table: &FeatureTable,
    pools: &AxisPools,
    n_cells: usize,
) -> (Vec<Identity>, Vec<Vec<(u32, f32)>>) {
    let mut ids: Vec<Identity> = Vec::new();
    let mut map: FxHashMap<(u32, u32, u32, bool), u32> = FxHashMap::default();
    let mut per_cell: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];

    // AGG anchor: e_f = β_g (gate masked), bias = b_agg[g].
    for i in 0..pools.agg.len() {
        let cell = pools.agg.axis_ids[i] as usize;
        if cell >= n_cells {
            continue;
        }
        let idx = intern(
            &mut ids,
            &mut map,
            Identity {
                gene: pools.agg.gene_ids[i],
                q_modality: 0,
                region: 0,
                is_agg: true,
                bias_modality: 0,
            },
        );
        per_cell[cell].push((idx, pools.agg.counts[i]));
    }

    // Count-comp: splice modality (≥1), region 0, shared count bias slot.
    for i in 0..pools.count_comp.len() {
        let cell = pools.count_comp.axis_ids[i] as usize;
        if cell >= n_cells {
            continue;
        }
        let idx = intern(
            &mut ids,
            &mut map,
            Identity {
                gene: pools.count_comp.gene_ids[i],
                q_modality: pools.count_comp.modality_ids[i],
                region: pools.count_comp.region_ids[i],
                is_agg: false,
                bias_modality: COUNT_BIAS_MODALITY,
            },
        );
        per_cell[cell].push((idx, pools.count_comp.counts[i]));
    }

    // Modifier-comp: its own modality + transcript-position region + bias.
    for (m, pool) in pools.modifier_comp_per_modality.iter().enumerate() {
        let m = m as u32;
        for i in 0..pool.len() {
            let cell = pool.axis_ids[i] as usize;
            if cell >= n_cells {
                continue;
            }
            let idx = intern(
                &mut ids,
                &mut map,
                Identity {
                    gene: pool.gene_ids[i],
                    q_modality: m,
                    region: pool.region_ids[i],
                    is_agg: false,
                    bias_modality: m,
                },
            );
            per_cell[cell].push((idx, pool.counts[i]));
        }
    }

    (ids, per_cell)
}

/// Compute the frozen `(e_f, b_f)` for each identity via the model's
/// `embed_and_bias_rows`, in chunks, then bring them to the CPU.
fn embed_identities(model: &GemModel, ids: &[Identity], h: usize) -> Result<FrozenFeatures> {
    const CHUNK: usize = 65536;
    let n = ids.len();
    let mut e = vec![0f32; n * h];
    let mut b = vec![0f32; n];
    let mut off = 0;
    for chunk in ids.chunks(CHUNK) {
        let gene: Vec<u32> = chunk.iter().map(|x| x.gene).collect();
        let q_mod: Vec<u32> = chunk.iter().map(|x| x.q_modality).collect();
        let region: Vec<u32> = chunk.iter().map(|x| x.region).collect();
        let bias_mod: Vec<u32> = chunk.iter().map(|x| x.bias_modality).collect();
        let is_agg: Vec<bool> = chunk.iter().map(|x| x.is_agg).collect();
        let (e_t, b_t) =
            model.embed_and_bias_rows(&gene, &gene, &q_mod, &region, &gene, &bias_mod, &is_agg)?;
        let e_rows = e_t.to_vec2::<f32>()?;
        let b_vals = b_t.to_vec1::<f32>()?;
        for (j, row) in e_rows.iter().enumerate() {
            e[(off + j) * h..(off + j + 1) * h].copy_from_slice(row);
        }
        b[off..off + chunk.len()].copy_from_slice(&b_vals);
        off += chunk.len();
    }
    Ok(FrozenFeatures { e, b })
}

/// Poisson MAP IRLS for one cell. `θ = [e_c; b_c]`; the ridge `λ` applies
/// to `e_c` only. Returns `(e_c, b_c)`. A cell with no observed features
/// gets the zero embedding (it carried no cell-axis signal).
fn solve_one_cell(
    feats: &[(u32, f32)],
    fz: &FrozenFeatures,
    h: usize,
    lambda: f64,
) -> (Vec<f32>, f32) {
    let d = h + 1;
    if feats.is_empty() {
        return (vec![0f32; h], 0.0);
    }
    let mut theta = DVector::<f64>::zeros(d);
    for _ in 0..MAX_IRLS_ITERS {
        let mut hess = DMatrix::<f64>::zeros(d, d);
        let mut grad = DVector::<f64>::zeros(d);
        for &(idx, n) in feats {
            let ef = &fz.e[idx as usize * h..(idx as usize + 1) * h];
            let bf = fz.b[idx as usize] as f64;
            // s = ⟨e_c, e_f⟩ + b_c + b_f
            let mut s = bf + theta[h];
            for (k, &efk) in ef.iter().enumerate() {
                s += theta[k] * efk as f64;
            }
            let mu = s.clamp(-SCORE_CLAMP, SCORE_CLAMP).exp();
            let resid = n as f64 - mu;
            // grad += resid · ẽ ;  hess += μ · ẽ ẽᵀ
            for (a, &efa) in ef.iter().enumerate() {
                let efa = efa as f64;
                grad[a] += resid * efa;
                for (bb, &efb) in ef.iter().enumerate() {
                    hess[(a, bb)] += mu * efa * efb as f64;
                }
                let cross = mu * efa;
                hess[(a, h)] += cross;
                hess[(h, a)] += cross;
            }
            grad[h] += resid;
            hess[(h, h)] += mu;
        }
        // Ridge prior on e_c (not the intercept), plus tiny PD jitter.
        for k in 0..h {
            hess[(k, k)] += lambda;
            grad[k] -= lambda * theta[k];
        }
        for k in 0..d {
            hess[(k, k)] += 1e-6;
        }
        let delta = match hess.clone().cholesky() {
            Some(ch) => ch.solve(&grad),
            None => hess.lu().solve(&grad).unwrap_or_else(|| DVector::zeros(d)),
        };
        theta += &delta;
        if delta.amax() < 1e-5 {
            break;
        }
    }
    let e_c: Vec<f32> = (0..h).map(|k| theta[k] as f32).collect();
    (e_c, theta[h] as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic: a few frozen features with known e_f, a planted cell e_c*,
    // Poisson counts at the noiseless rate. IRLS should recover e_c* closely.
    #[test]
    fn irls_recovers_planted_cell() {
        let h = 4;
        // 12 frozen features with assorted directions.
        let n_id = 12;
        let mut e = vec![0f32; n_id * h];
        let mut b = vec![0f32; n_id];
        for f in 0..n_id {
            for k in 0..h {
                // deterministic pseudo-random in [-0.5, 0.5]
                e[f * h + k] = (((f * 7 + k * 13) % 11) as f32 / 11.0) - 0.5;
            }
            b[f] = (((f * 5) % 7) as f32 / 7.0) - 0.3;
        }
        let fz = FrozenFeatures { e, b };
        let e_star = [0.8f32, -0.6, 0.4, 0.2];
        let b_star = 0.5f32;
        // noiseless Poisson rate as the "observed" count
        let feats: Vec<(u32, f32)> = (0..n_id)
            .map(|f| {
                let ef = &fz.e[f * h..(f + 1) * h];
                let s: f32 =
                    ef.iter().zip(&e_star).map(|(a, b)| a * b).sum::<f32>() + fz.b[f] + b_star;
                (f as u32, s.exp())
            })
            .collect();
        let (e_c, _b_c) = solve_one_cell(&feats, &fz, h, 1e-3);
        // recovered direction aligns with the plant (ridge shrinks scale a bit)
        let dot: f32 = e_c.iter().zip(&e_star).map(|(a, b)| a * b).sum();
        let na: f32 = e_c.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = e_star.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.97,
            "recovered cell embedding misaligned (cos={cos:.3})"
        );
    }

    #[test]
    fn empty_cell_is_zero() {
        let fz = FrozenFeatures {
            e: vec![0.0; 4],
            b: vec![0.0; 1],
        };
        let (e_c, b_c) = solve_one_cell(&[], &fz, 4, 1.0);
        assert_eq!(e_c, vec![0.0; 4]);
        assert_eq!(b_c, 0.0);
    }
}
