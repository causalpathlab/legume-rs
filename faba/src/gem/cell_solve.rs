//! Phase-2 for `faba gem`: project every cell onto the **frozen** feature
//! dictionary, in parallel. This is the gem adapter; the actual per-cell
//! Poisson-MAP IRLS solver is the shared, model-agnostic
//! [`graph_embedding_util::cell_projection`] (also used by `senna bge`).
//!
//! The gem-specific work here is turning the trained `GemModel` + the
//! cell-axis pools into the two inputs the solver wants: the frozen
//! feature embeddings `e_f` / `b_f`, and each cell's observed
//! `(feature, count)` list.

use super::common::candle_core;
use anyhow::{Context, Result};
use candle_core::Tensor;
use rustc_hash::FxHashMap;

use super::args::GemArgs;
use super::model::GemModel;
use super::pseudobulk::AxisPools;
use super::sampling::COUNT_BIAS_MODALITY;

/// One distinct feature-row identity — the `embed_and_bias_rows` inputs that
/// all collapse to the same frozen `e_f` / `b_f`. Mirrors how the sampler
/// builds positives (`sampling::push_agg_positive` / `push_component_positive`):
/// AGG → β_g; count-comp → splice modality + shared `COUNT_BIAS_MODALITY`
/// bias; modifier-comp → its own modality + region + bias.
struct Identity {
    gene: u32,
    q_modality: u32,
    region: u32,
    is_agg: bool,
    bias_modality: u32,
}

/// Solve `e_cell` (and `b_cell`) by projecting every cell onto the frozen
/// feature side, then overwrite the model's `e_cell` / `b_cell` vars.
pub fn solve_cell_embeddings(
    model: &mut GemModel,
    pools: &AxisPools,
    args: &GemArgs,
) -> Result<()> {
    let n_cells = model.n_cells;
    let h = model.embedding_dim;
    if n_cells == 0 {
        return Ok(());
    }

    // 1. Distinct identities + per-cell (identity, count) lists.
    let (ids, per_cell) = collect_identities(pools, n_cells);
    // 2. Frozen feature embeddings (one batched device pass → CPU).
    let (frozen_e, frozen_b) = embed_identities(model, &ids, h)?;
    // 3. Parallel per-cell Poisson-MAP projection (shared solver).
    let lambda = args.phase2_ridge.max(0.0) as f64;
    // gem scores with a per-cell bias (b_cell), so keep the fitted b_c.
    let (mut e_flat, b_flat) = graph_embedding_util::cell_projection::project_cells(
        &frozen_e, &frozen_b, &per_cell, h, lambda,
    );

    // L2-normalize each cell's embedding (depth correction). The Poisson-MAP
    // matches absolute counts, so the *norm* of `e_cell` leaks sequencing
    // depth (corr(‖e‖, b_cell) ≈ -0.95 on rep1) and a single depth direction
    // dominates ~82% of the variance — which collapses the downstream
    // archetypal topics (one giant central topic). The feature side is
    // trained with scale-invariant NCE, so only the DIRECTION is meaningful;
    // depth stays in the unpenalized `b_cell`. Mirrors the bge phase-2 fix
    // (geu commit 9142779). Near-empty cells solve to ~0 and stay 0 (and are
    // dropped by `--min-cell-nnz` at write time anyway).
    for c in 0..n_cells {
        let row = &mut e_flat[c * h..(c + 1) * h];
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for v in row.iter_mut() {
                *v /= nrm;
            }
        }
    }

    // 4. Write back into the model's vars (and cached fields).
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
/// append `(identity, count)` to its cell's list.
fn collect_identities(pools: &AxisPools, n_cells: usize) -> (Vec<Identity>, Vec<Vec<(u32, f32)>>) {
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
/// `embed_and_bias_rows`, in chunks, then bring them to the CPU. Returns
/// `(e [n_id × h] row-major, b [n_id])`.
fn embed_identities(model: &GemModel, ids: &[Identity], h: usize) -> Result<(Vec<f32>, Vec<f32>)> {
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
    Ok((e, b))
}
