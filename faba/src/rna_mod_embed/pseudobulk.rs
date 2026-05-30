//! Multilevel pseudobulk collapse driven by the count modality, plus
//! per-axis per-stratum aggregation of triplets for the sampler.
//!
//! Pipeline:
//!   1. Random projection of cells from the count file (batch-corrected).
//!   2. `collapse_columns_multilevel_with_hierarchy` on the count file
//!      → per-level `cell_to_pb` partitions. Other modalities inherit
//!      the partition trivially via the same `cell_to_pb_per_level`.
//!   3. Walk `unified.triplets` once, accumulate per-(gene, axis_id)
//!      counts for each stratum at every requested axis: the cell axis
//!      (identity partition; one unit per cell) and each pb level.
//!      All component rows within a (g, m) pair share an embedding
//!      (e_{g,m,r} = β_g ⊙ exp(Σ_k z_{g,k}·δ_{k,m,:} + γ_{m,r,:})), so aggregation
//!      loses nothing for sampling and dedupes work.
//!
//! The two-stage trainer uses the pb axes for stage 1 (curriculum
//! training of β/z/δ/γ with lower-variance pb signal) and the cell axis
//! for stage 2 (refining e_cell with β/z/δ/γ frozen).

use anyhow::Context;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::data::UnifiedData;
use log::info;
use rustc_hash::FxHashMap;

use super::args::RnaModEmbedArgs;
use super::feature_table::{FeatureTable, RowStratum};

/// One per-(axis, stratum) pool of count-weighted (gene_id, axis_id)
/// candidates. All component rows within a (g, m) pair share an
/// embedding, so we aggregate their counts up to per-gene granularity
/// here. `weights` carries the τ-tempered draws used by
/// `rand::distr::WeightedIndex`.
pub struct StratumPool {
    pub gene_ids: Vec<u32>,
    pub axis_ids: Vec<u32>,
    /// Transcript-position region bin per pool entry. Anchor strata
    /// (agg / count-comp) use region 0 as a sentinel — the model masks
    /// their log-deviation gate to exp(0) regardless. Satellite
    /// (modifier-comp) entries carry the real `region(component)`, so
    /// two same-gene components in different regions stay distinct.
    pub region_ids: Vec<u32>,
    pub counts: Vec<f32>,
    pub weights: Vec<f32>,
}

impl StratumPool {
    pub fn len(&self) -> usize {
        self.gene_ids.len()
    }
}

/// Aggregated draws on one right-hand axis, organised by stratum.
/// Re-used for the cell axis (identity partition; `n_units = n_cells`)
/// and each pb level (`n_units = n_pb_ℓ`).
pub struct AxisPools {
    /// Number of distinct right-hand-axis ids.
    pub n_units: usize,
    pub agg: StratumPool,
    pub count_comp: StratumPool,
    /// `modifier_comp_per_modality[m]` for modality id `m` (slot 0 =
    /// count modality is always empty by convention).
    pub modifier_comp_per_modality: Vec<StratumPool>,
    /// `modality_total_mass[m]` = Σ counts in
    /// `modifier_comp_per_modality[m]`. Used by the sampler's τ_M
    /// modality-balance step.
    pub modality_total_mass: Vec<f32>,
}

pub struct PseudobulkData {
    /// `cell_to_pb_per_level[ℓ][cell]` = pb id at level ℓ for the given
    /// unified cell. Order is **coarsest-first** (level 0 is the smallest
    /// number of pbs). Persisted to `{out}.cell_to_pb.parquet`.
    pub cell_to_pb_per_level: Vec<Vec<usize>>,
    /// Cell-axis pools (identity partition; one unit per cell).
    pub cell_pools: AxisPools,
    /// Per-pb-level pools, coarsest-first.
    pub pb_pools_per_level: Vec<AxisPools>,
}

impl PseudobulkData {
    pub fn n_levels(&self) -> usize {
        self.pb_pools_per_level.len()
    }
}

/// Build the multilevel partition + the per-axis sampling pools.
/// Mutates `unified.per_file_data[0]` (the count file) to register
/// batch membership and HNSW caches — same side-effect bge has.
pub fn build_pseudobulk(
    unified: &mut UnifiedData,
    table: &FeatureTable,
    args: &RnaModEmbedArgs,
) -> anyhow::Result<PseudobulkData> {
    ////////////////////////////////////////
    // 1. Batch labels + projection
    ////////////////////////////////////////
    let batch_labels: Vec<Box<str>> = unified
        .batch_membership
        .iter()
        .map(|&b| unified.batch_names[b as usize].clone())
        .collect();
    let batch_arg = (unified.n_batches() > 1).then_some(batch_labels.as_slice());

    info!(
        "projection (proj_dim={}, {} batches)...",
        args.proj_dim,
        unified.n_batches()
    );
    let proj = unified.per_file_data[0]
        .project_columns_with_batch_correction(args.proj_dim, None, batch_arg)
        .context("random projection on count file")?;

    ////////////////////////////////////////
    // 2. Multilevel collapse on count file
    ////////////////////////////////////////
    info!(
        "multilevel collapse (sort_dim={}, {} levels)...",
        args.sort_dim, args.num_levels
    );
    let collapse_out = collapse_columns_multilevel_with_hierarchy(
        &mut unified.per_file_data[0],
        &proj.proj,
        &batch_labels,
        &MultilevelParams {
            knn_pb_samples: args.knn_pb,
            num_levels: args.num_levels.max(1),
            sort_dim: args.sort_dim,
            num_opt_iter: args.num_opt_iter,
            refine: Some(Default::default()),
            output_calibration: matrix_param::traits::CalibrateTarget::MeanOnly,
        },
    )
    .context("multilevel collapse on count file")?;

    // Returned finest-first; flip to coarsest-first.
    let mut cell_to_pb_per_level = collapse_out.cell_to_pb_per_level;
    cell_to_pb_per_level.reverse();
    drop(collapse_out.levels);

    ////////////////////////////////////////
    // 3. Materialise triplets + aggregate
    ////////////////////////////////////////
    // `materialize_cell_triplets` is deferred so the collapse phase's
    // peak memory doesn't have to coexist with the triplet edge list.
    unified
        .materialize_cell_triplets()
        .context("materialize unified triplets")?;
    info!(
        "materialized {} cell↔feature triplets",
        unified.triplets.len()
    );

    let n_modalities = table.n_modalities();
    let n_cells = unified.n_cells();

    // Cell axis: identity partition.
    let cell_identity: Vec<usize> = (0..n_cells).collect();
    let cell_pools = aggregate_pools(
        unified,
        table,
        &cell_identity,
        n_cells,
        n_modalities,
        args.tau,
    );

    // Pb axes: one set of pools per level (coarsest-first).
    let pb_pools_per_level: Vec<AxisPools> = cell_to_pb_per_level
        .iter()
        .map(|cell_to_pb| {
            let n_pbs = cell_to_pb.iter().copied().max().map(|m| m + 1).unwrap_or(0);
            aggregate_pools(unified, table, cell_to_pb, n_pbs, n_modalities, args.tau)
        })
        .collect();

    info!(
        "cell axis: {} cells, {} agg / {} count-comp / {} modifier-comp draws",
        cell_pools.n_units,
        cell_pools.agg.len(),
        cell_pools.count_comp.len(),
        cell_pools
            .modifier_comp_per_modality
            .iter()
            .map(|p| p.len())
            .sum::<usize>()
    );
    for (i, lvl) in pb_pools_per_level.iter().enumerate() {
        info!(
            "pb level {} (coarse→fine): {} pbs, {} agg / {} count-comp / {} modifier-comp draws",
            i,
            lvl.n_units,
            lvl.agg.len(),
            lvl.count_comp.len(),
            lvl.modifier_comp_per_modality
                .iter()
                .map(|p| p.len())
                .sum::<usize>()
        );
    }

    Ok(PseudobulkData {
        cell_to_pb_per_level,
        cell_pools,
        pb_pools_per_level,
    })
}

/// Aggregate `unified.triplets` to per-(gene, axis_id) granularity per
/// stratum. `cell_to_axis_id[c]` gives the right-hand-axis id for
/// unified cell `c` — identity for the cell axis,
/// `cell_to_pb_per_level[ℓ]` for pb axes.
fn aggregate_pools(
    unified: &UnifiedData,
    table: &FeatureTable,
    cell_to_axis_id: &[usize],
    n_units: usize,
    n_modalities: usize,
    tau: f32,
) -> AxisPools {
    // Anchor accumulators keyed by (gene_id, axis_id) — region-agnostic.
    let mut count_comp: FxHashMap<(u32, u32), f32> = FxHashMap::default();
    let mut agg: FxHashMap<(u32, u32), f32> = FxHashMap::default();
    // Satellite accumulators keyed by (gene_id, component, axis_id) so
    // two same-gene components stay distinct pool entries. A side map
    // records each (gene, component)'s region for the emitted pool.
    let mut modifier_by_mod: Vec<FxHashMap<(u32, u32, u32), f32>> =
        (0..n_modalities).map(|_| FxHashMap::default()).collect();
    let mut modifier_region: Vec<FxHashMap<(u32, u32), u32>> =
        (0..n_modalities).map(|_| FxHashMap::default()).collect();

    for t in &unified.triplets {
        let row = t.feature as usize;
        let aid = cell_to_axis_id[t.cell as usize] as u32;
        let Some(stratum) = table.strata[row] else {
            continue;
        };
        let Some(g) = table.row_gene[row] else {
            continue;
        };
        match stratum {
            RowStratum::CountComp => {
                *count_comp.entry((g, aid)).or_insert(0.0) += t.count;
                *agg.entry((g, aid)).or_insert(0.0) += t.count;
            }
            RowStratum::ModifierComp => {
                if let Some(m) = table.row_modality[row] {
                    let comp = table.row_component[row].unwrap_or(0);
                    let region = table.row_region[row].unwrap_or(0);
                    *modifier_by_mod[m as usize]
                        .entry((g, comp, aid))
                        .or_insert(0.0) += t.count;
                    modifier_region[m as usize].insert((g, comp), region);
                }
            }
            RowStratum::Site => {}
        }
    }

    // Anchor pools: region sentinel 0 (masked in the model).
    let count_comp = anchor_pool_from_map(count_comp, tau);
    let agg = anchor_pool_from_map(agg, tau);
    let modifier_comp_per_modality: Vec<StratumPool> = modifier_by_mod
        .into_iter()
        .zip(modifier_region)
        .map(|(m, region_map)| satellite_pool_from_map(m, &region_map, tau))
        .collect();
    let modality_total_mass: Vec<f32> = modifier_comp_per_modality
        .iter()
        .map(|p| p.counts.iter().sum())
        .collect();

    AxisPools {
        n_units,
        agg,
        count_comp,
        modifier_comp_per_modality,
        modality_total_mass,
    }
}

/// τ-tempered sampling weight. τ=1 → strict count-prop; τ=0 → uniform
/// over rows with non-zero mass.
fn temper(c: f32, tau: f32) -> f32 {
    if c > 0.0 {
        c.powf(tau)
    } else {
        0.0
    }
}

/// Assemble a `StratumPool` from `(gene, axis, region, count)` entries,
/// τ-tempering the weights. Both anchor and satellite builders feed this
/// — they differ only in how they extract gene/axis/region from their
/// map key.
fn pool_from_entries(
    n_hint: usize,
    tau: f32,
    entries: impl Iterator<Item = (u32, u32, u32, f32)>,
) -> StratumPool {
    let mut gene_ids = Vec::with_capacity(n_hint);
    let mut axis_ids = Vec::with_capacity(n_hint);
    let mut region_ids = Vec::with_capacity(n_hint);
    let mut counts = Vec::with_capacity(n_hint);
    let mut weights = Vec::with_capacity(n_hint);
    for (g, aid, region, c) in entries {
        gene_ids.push(g);
        axis_ids.push(aid);
        region_ids.push(region);
        counts.push(c);
        weights.push(temper(c, tau));
    }
    StratumPool {
        gene_ids,
        axis_ids,
        region_ids,
        counts,
        weights,
    }
}

/// Anchor strata (agg / count-comp): keyed by (gene, axis), region
/// sentinel 0 (the model masks their gate to exp(0)).
fn anchor_pool_from_map(map: FxHashMap<(u32, u32), f32>, tau: f32) -> StratumPool {
    let n = map.len();
    pool_from_entries(n, tau, map.into_iter().map(|((g, aid), c)| (g, aid, 0, c)))
}

/// Satellite stratum (modifier-comp): keyed by (gene, component, axis);
/// each entry carries the component's transcript-position region from
/// the side map (default 0 when un-annotated).
fn satellite_pool_from_map(
    map: FxHashMap<(u32, u32, u32), f32>,
    region_map: &FxHashMap<(u32, u32), u32>,
    tau: f32,
) -> StratumPool {
    let n = map.len();
    pool_from_entries(
        n,
        tau,
        map.into_iter().map(|((g, comp, aid), c)| {
            let region = region_map.get(&(g, comp)).copied().unwrap_or(0);
            (g, aid, region, c)
        }),
    )
}
