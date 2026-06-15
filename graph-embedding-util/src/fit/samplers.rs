use super::config::FeatureNetworkConfig;
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{CellFeatureSampler, PerBatchStratifiedCellSampler};
use crate::progress::new_progress_bar;
use log::{info, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;

/// Threshold above which a pb level is treated as a per-cell partition
/// and pruned from the chain. `n_pbs / n_cells > 0.5` means avg pb size
/// < 2 — even if a few edges survive the pb-mismatch filter at that
/// level, the loss carries near-zero training signal.
pub(crate) const DEGENERATE_PB_RATIO: f32 = 0.5;

/// Derive a phase-1-only subsampled view of the per-batch stratified cell
/// samplers: keep at most `k` cells per pb-sample at EVERY collapse level
/// (`cell_to_pb_per_level`, coarsest..finest), unioned across levels. The
/// returned samplers cover only the kept cells; each batch's `cell_picker` is
/// rebuilt from the kept cells' recomputed degree weights (`degree^alpha_cell ·
/// mult`, degree recovered from `CellFeatureSampler.counts`), while the
/// negative marginal (`neg` / `feature_pool`) is cloned unchanged so negatives
/// stay drawn from the full per-batch feature pool.
///
/// Keeping ≤k per pb at *every* level (not just the finest) lets each level's
/// partition contribute diverse representatives — robust even when refinement
/// breaks strict nesting between adjacent levels. The finest level is batch-
/// aware, so every non-empty batch keeps ≥1 cell; empty batches are dropped
/// (the cell axis re-indexes at sample time and ignores the original batch id).
pub(crate) fn subsample_cell_samplers_multilevel(
    full: &[PerBatchStratifiedCellSampler],
    cell_to_pb_per_level: &[Vec<usize>],
    k: usize,
    alpha_cell: f32,
    cell_weight_mult: Option<&[f32]>,
    seed: u64,
) -> Vec<PerBatchStratifiedCellSampler> {
    let n_cells = cell_to_pb_per_level.first().map_or(0, |v| v.len());
    // Global keep bitmap: ≤k cells per pb-sample, per level, unioned.
    let mut keep = vec![false; n_cells];
    for (level, c2pb) in cell_to_pb_per_level.iter().enumerate() {
        let n_pb = c2pb.iter().copied().max().map_or(0, |m| m + 1);
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_pb];
        for (cell, &pb) in c2pb.iter().enumerate() {
            buckets[pb].push(cell as u32);
        }
        // Per-level seed so the K kept cells differ across levels (more union
        // diversity) yet stay reproducible across runs.
        let mut rng =
            StdRng::seed_from_u64(seed ^ (level as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for b in buckets.iter_mut() {
            // `partial_shuffle` does only k swaps (vs a full O(bucket) shuffle)
            // and hands back the k-element random subset directly.
            let chosen: &[u32] = if b.len() > k {
                b.partial_shuffle(&mut rng, k).0
            } else {
                &b[..]
            };
            for &c in chosen {
                keep[c as usize] = true;
            }
        }
    }

    // Filter each batch sampler to the kept cells; rebuild `cell_picker`.
    full.iter()
        .filter_map(|s| {
            let cap = s.active_cells.len();
            let mut active_cells: Vec<u32> = Vec::with_capacity(cap);
            let mut per_cell: Vec<CellFeatureSampler> = Vec::with_capacity(cap);
            let mut cell_w: Vec<f32> = Vec::with_capacity(cap);
            for (i, &c) in s.active_cells.iter().enumerate() {
                if !keep[c as usize] {
                    continue;
                }
                let cf = &s.per_cell[i];
                let degree: f32 = cf.counts.iter().sum();
                let mult = cell_weight_mult.map_or(1.0, |m| m[c as usize]);
                cell_w.push(degree.max(1e-8).powf(alpha_cell) * mult);
                per_cell.push(cf.clone());
                active_cells.push(c);
            }
            if active_cells.is_empty() {
                return None;
            }
            let cell_picker =
                WeightedIndex::new(cell_w).expect("non-empty subsampled cell weights");
            Some(PerBatchStratifiedCellSampler {
                cell_picker,
                active_cells,
                per_cell,
                neg: s.neg.clone(),
                feature_pool: s.feature_pool.clone(),
            })
        })
        .collect()
}

/// Build the stratified per-batch cell samplers and filter out empty
/// batches. Mirrors the previous `build_active_samplers` (flat) but
/// uses the two-stage `cell_picker` → `per_cell` draw — every cell in
/// a batch gets coverage proportional to `degree^alpha_cell` instead
/// of being drowned by deeply sequenced cells.
pub(crate) fn build_active_samplers(
    unified: &UnifiedData,
    feat_weights: &[f32],
    alpha_cell: f32,
    alpha_neg: f32,
    cell_weight_mult: Option<&[f32]>,
) -> anyhow::Result<Vec<PerBatchStratifiedCellSampler>> {
    // Build the per-batch stratified-cell samplers by **streaming columns**
    // from the backend, never materializing the flat cell↔feature edge list.
    // The strat-cell sampler groups edges by cell (`per_cell`), which is
    // exactly a column read — so the 5 GB flat triplet list (which only the
    // unused flat `PerBatch` path ever read) is skipped entirely. The HVG /
    // frozen subset is honored via `feature_to_backend_row`.
    let data = unified.count_backend();
    let n_cells = data.num_columns();
    let n_features = unified.n_features();
    let n_batches = unified.n_batches();
    let batch_membership = &unified.batch_membership;

    // backend compact row → unified id (u32::MAX ⇒ dropped by a subset).
    let backend_rows = data.num_rows();
    let mut backend_to_unified = vec![u32::MAX; backend_rows];
    for (uid, &brow) in unified.feature_to_backend_row.iter().enumerate() {
        if brow < backend_rows {
            backend_to_unified[brow] = uid as u32;
        }
    }

    // Per-batch accumulators, filled as cells stream in.
    let mut active_cells: Vec<Vec<u32>> = vec![Vec::new(); n_batches];
    let mut per_cell: Vec<Vec<CellFeatureSampler>> = (0..n_batches).map(|_| Vec::new()).collect();
    let mut cell_w: Vec<Vec<f32>> = vec![Vec::new(); n_batches];
    let mut feat_count: Vec<Vec<f32>> = (0..n_batches).map(|_| vec![0f32; n_features]).collect();

    // Slab width targets ~8M edges/slab. When nnz can't be reported
    // (num_non_zeros errs → 0) fall back to a fixed cell-count slab rather
    // than the whole matrix, so the streaming memory bound always holds.
    let chunk = match data.num_non_zeros() {
        Ok(nnz) if nnz > 0 => {
            let avg_per_col = (nnz / n_cells.max(1)).max(1);
            (8_000_000 / avg_per_col).clamp(1, n_cells.max(1))
        }
        _ => (1usize << 14).min(n_cells.max(1)),
    };
    let pb_bar = new_progress_bar(n_cells as u64);
    pb_bar.set_message("strat-cell sampler (streaming columns)");

    let mut start = 0usize;
    while start < n_cells {
        let end = (start + chunk).min(n_cells);
        let slab = end - start;
        // Group this slab's nonzeros by local column (cell). `for_each_triplet`
        // emits out_col relative to the passed `start..end`, i.e. 0..slab.
        let mut col_feats: Vec<Vec<u32>> = vec![Vec::new(); slab];
        let mut col_counts: Vec<Vec<f32>> = vec![Vec::new(); slab];
        let mut col_wts: Vec<Vec<f32>> = vec![Vec::new(); slab];
        let mut col_deg: Vec<f32> = vec![0.0; slab];
        data.for_each_triplet(start..end, slab, |brow, local_col, v| {
            if v == 0.0 {
                return;
            }
            let uid = backend_to_unified[brow as usize];
            if uid == u32::MAX {
                return;
            }
            let lc = local_col as usize;
            let cell = start + lc;
            let b = batch_membership[cell] as usize;
            col_feats[lc].push(uid);
            col_counts[lc].push(v);
            col_wts[lc].push((v * feat_weights[uid as usize]).max(1e-8));
            col_deg[lc] += v;
            feat_count[b][uid as usize] += v;
        })?;
        for lc in 0..slab {
            if col_feats[lc].is_empty() {
                continue;
            }
            let cell = (start + lc) as u32;
            let b = batch_membership[cell as usize] as usize;
            let picker =
                WeightedIndex::new(std::mem::take(&mut col_wts[lc])).expect("cell-feature weights");
            per_cell[b].push(CellFeatureSampler {
                features: std::mem::take(&mut col_feats[lc]),
                counts: std::mem::take(&mut col_counts[lc]),
                picker,
            });
            active_cells[b].push(cell);
            let mult = cell_weight_mult.map_or(1.0, |m| m[cell as usize]);
            cell_w[b].push(col_deg[lc].max(1e-8).powf(alpha_cell) * mult);
        }
        pb_bar.inc(slab as u64);
        start = end;
    }
    pb_bar.finish_and_clear();

    // Finalize one sampler per non-empty batch (re-indexed; the original batch
    // id isn't used at sample time).
    let mut empty: Vec<&str> = Vec::new();
    let mut active: Vec<PerBatchStratifiedCellSampler> = Vec::new();
    for b in 0..n_batches {
        if active_cells[b].is_empty() {
            empty.push(unified.batch_names[b].as_ref());
            continue;
        }
        let cell_picker = WeightedIndex::new(std::mem::take(&mut cell_w[b])).expect("cell weights");
        let fc = &feat_count[b];
        let feature_pool: Vec<u32> = (0..n_features as u32)
            .filter(|&f| fc[f as usize] > 0.0)
            .collect();
        let neg_w: Vec<f32> = feature_pool
            .iter()
            .map(|&f| fc[f as usize].powf(alpha_neg))
            .collect();
        let neg = WeightedIndex::new(neg_w).expect("batch feature pool");
        active.push(PerBatchStratifiedCellSampler {
            cell_picker,
            active_cells: std::mem::take(&mut active_cells[b]),
            per_cell: std::mem::take(&mut per_cell[b]),
            neg,
            feature_pool,
        });
    }
    if !empty.is_empty() {
        warn!(
            "Skipping {} batch(es) with no observed edges: {}",
            empty.len(),
            empty.join(", ")
        );
    }
    if active.is_empty() {
        anyhow::bail!("no non-empty batches available for sampling");
    }
    Ok(active)
}

pub(crate) fn build_smoother(
    feature_network: Option<FeatureNetworkConfig>,
    n_features: usize,
    embedding_dim: usize,
) -> anyhow::Result<Option<FeatureNetworkSmoother>> {
    let Some(FeatureNetworkConfig {
        graph,
        k_hops,
        alpha,
        refresh_epochs,
    }) = feature_network
    else {
        return Ok(None);
    };
    if graph.num_edges() == 0 {
        anyhow::bail!("feature network has 0 matched edges — check name resolution at the caller");
    }
    info!(
        "SGC smoothing: K={}, α={}, refresh={} epochs over {} edges",
        k_hops,
        alpha,
        refresh_epochs,
        graph.num_edges()
    );
    Ok(Some(FeatureNetworkSmoother::new(
        &graph,
        n_features,
        embedding_dim,
        alpha,
        k_hops,
        refresh_epochs,
    )?))
}
