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
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::args::GemArgs;
use super::feature_table::{FeatureTable, RowStratum};

/// One per-(axis, stratum) pool of count-weighted (gene_id, axis_id)
/// candidates. All component rows within a (g, m) pair share an
/// embedding, so we aggregate their counts up to per-gene granularity
/// here. `weights` carries the τ-tempered draws used by
/// `rand::distr::WeightedIndex`.
pub struct StratumPool {
    pub gene_ids: Vec<u32>,
    pub axis_ids: Vec<u32>,
    /// Per-entry modality id. The AGG pool uses the sentinel 0 (its rows
    /// are emitted via the masked-gate AGG path). The count-comp pool now
    /// carries the entry's **splice modality** (`spliced` / `unspliced`,
    /// each its own id ≥ 1) so the sampler can give them distinct δ
    /// directions; modifier pools carry their single modality.
    pub modality_ids: Vec<u32>,
    /// Transcript-position region bin per pool entry. Anchor strata
    /// (agg / count-comp) use region 0 as a sentinel. Satellite
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

    pub fn is_empty(&self) -> bool {
        self.gene_ids.is_empty()
    }

    /// Keep only entries whose right-hand-axis id (cell id for the cell
    /// axis) is marked in `keep`. Weights are carried through unchanged —
    /// each surviving (gene, cell) edge keeps its τ-tempered / Fisher-
    /// adjusted weight, so the subsampled draw distribution is just the
    /// full one restricted to the kept cells.
    fn filter_by_axis(&self, keep: &[bool]) -> StratumPool {
        let mut out = StratumPool {
            gene_ids: Vec::new(),
            axis_ids: Vec::new(),
            modality_ids: Vec::new(),
            region_ids: Vec::new(),
            counts: Vec::new(),
            weights: Vec::new(),
        };
        for i in 0..self.len() {
            if keep
                .get(self.axis_ids[i] as usize)
                .copied()
                .unwrap_or(false)
            {
                out.gene_ids.push(self.gene_ids[i]);
                out.axis_ids.push(self.axis_ids[i]);
                out.modality_ids.push(self.modality_ids[i]);
                out.region_ids.push(self.region_ids[i]);
                out.counts.push(self.counts[i]);
                out.weights.push(self.weights[i]);
            }
        }
        out
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
    /// Per-gene NB-Fisher housekeeping weights (length `n_genes`, in
    /// `(0, 1]`). All `1.0` when `--housekeeping-penalty 0`. Already
    /// folded into the anchor sampling pools; kept here only so the
    /// caller can persist them to `{out}.fisher_weights.parquet`.
    pub gene_fisher_weights: Vec<f32>,
    /// Per-gene ubiquity (fraction of cells expressing, length `n_genes`,
    /// in `(0, 1]`) from the cell-axis count pool. NOT folded into any
    /// pool or the model; persisted to `{out}.ubiquity.parquet` as a
    /// diagnostic / inverse-propensity signal (breadth complement to the
    /// NB-Fisher magnitude weight).
    pub gene_ubiquity: Vec<f32>,
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
    args: &GemArgs,
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
    let mut cell_pools = aggregate_pools(
        unified,
        table,
        &cell_identity,
        n_cells,
        n_modalities,
        args.tau,
    );

    // Pb axes: one set of pools per level (coarsest-first). Each level
    // walks `unified.triplets` once into independent thread-local maps, so
    // the levels are embarrassingly parallel — fan out across them rather
    // than making `num_levels` sequential passes over the edge list. The
    // shared reborrow keeps the closure capturing `&UnifiedData` (Sync),
    // not the outer `&mut`.
    let unified_ref: &UnifiedData = unified;
    let mut pb_pools_per_level: Vec<AxisPools> = cell_to_pb_per_level
        .par_iter()
        .map(|cell_to_pb| {
            let n_pbs = cell_to_pb.iter().copied().max().map(|m| m + 1).unwrap_or(0);
            aggregate_pools(
                unified_ref,
                table,
                cell_to_pb,
                n_pbs,
                n_modalities,
                args.tau,
            )
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

    ////////////////////////////////////////
    // 4. NB-Fisher housekeeping penalty
    ////////////////////////////////////////
    // Per-gene Fisher weights from the cell-axis count modality, folded
    // into the count-based anchor pools (agg + count-comp) so housekeeping
    // genes stop monopolising the shared program loadings z. See
    // `gene_weight`. A no-op (all weights 1.0) when penalty == 0.
    let n_genes = table.n_genes();
    // Per-gene ubiquity / Fisher from the cell-axis **AGG** pool — it holds
    // exactly one per-(gene, cell) total entry (spliced+unspliced summed),
    // the per-(gene, cell) granularity these statistics assume. The
    // count-comp pool is now split per splice modality (multiple entries
    // per gene/cell), so reading it here would double-count. (The reweight
    // below still mutates only `weights`, not `gene_ids`.) See `gene_weight`.
    let gene_ubiquity =
        super::gene_weight::ubiquity_from_count_pool(&cell_pools.agg, n_genes, n_cells);

    let gene_fisher_weights = if args.housekeeping_penalty > 0.0 {
        let w =
            super::gene_weight::fisher_weights_from_count_pool(&cell_pools.agg, n_genes, n_cells);
        apply_fisher_to_axis_pools(&mut cell_pools, &w, args.housekeeping_penalty);
        for lvl in pb_pools_per_level.iter_mut() {
            apply_fisher_to_axis_pools(lvl, &w, args.housekeeping_penalty);
        }
        log_fisher_summary(&w, table, args.housekeeping_penalty);
        w
    } else {
        vec![1.0; n_genes]
    };

    Ok(PseudobulkData {
        cell_to_pb_per_level,
        cell_pools,
        pb_pools_per_level,
        gene_fisher_weights,
        gene_ubiquity,
    })
}

/// Phase-1-only subsampled view of the cell-axis pools: keep at most `k`
/// cells per pb-sample at EVERY collapse level (`cell_to_pb_per_level`),
/// unioned across levels, then restrict each cell-axis `StratumPool` to
/// those cells. The full `cell_pools` are untouched (phase 2 still
/// projects every cell against the frozen dictionary).
///
/// Keeping ≤k per pb at *every* level (not just the finest) lets each
/// level's partition contribute diverse representatives — robust even when
/// refinement breaks strict nesting between adjacent levels. `n_units` of
/// the returned pools is the kept-cell count, so the trainer's auto
/// per-epoch budget shrinks accordingly. Mirrors `senna bge`'s
/// `subsample_cell_samplers_multilevel`.
pub(crate) fn subsample_cell_pools_multilevel(
    full: &AxisPools,
    cell_to_pb_per_level: &[Vec<usize>],
    k: usize,
    seed: u64,
) -> AxisPools {
    let n_cells = cell_to_pb_per_level.first().map_or(0, |v| v.len());
    // Global keep bitmap: ≤k cells per pb-sample, per level, unioned.
    let mut keep = vec![false; n_cells];
    for (level, c2pb) in cell_to_pb_per_level.iter().enumerate() {
        let n_pb = c2pb.iter().copied().max().map_or(0, |m| m + 1);
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_pb];
        for (cell, &pb) in c2pb.iter().enumerate() {
            buckets[pb].push(cell as u32);
        }
        // Per-level seed so the kept cells differ across levels (more union
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

    let kept = keep.iter().filter(|&&x| x).count();
    let agg = full.agg.filter_by_axis(&keep);
    let count_comp = full.count_comp.filter_by_axis(&keep);
    let modifier_comp_per_modality: Vec<StratumPool> = full
        .modifier_comp_per_modality
        .iter()
        .map(|p| p.filter_by_axis(&keep))
        .collect();
    let modality_total_mass: Vec<f32> = modifier_comp_per_modality
        .iter()
        .map(|p| p.counts.iter().sum())
        .collect();

    AxisPools {
        n_units: kept,
        agg,
        count_comp,
        modifier_comp_per_modality,
        modality_total_mass,
    }
}

/// Apply per-gene Fisher weights to an axis's two count-based anchor
/// pools (agg + count-comp). Modifier (m6A / A2I / pA) pools are left
/// untouched — that signal is balanced via `--tau-modality` and is what
/// the model is meant to recover.
fn apply_fisher_to_axis_pools(pools: &mut AxisPools, fisher: &[f32], penalty: f32) {
    super::gene_weight::apply_to_pool(&mut pools.agg, fisher, penalty);
    super::gene_weight::apply_to_pool(&mut pools.count_comp, fisher, penalty);
}

/// One-line diagnostic: how many genes the penalty meaningfully shrinks,
/// plus the most-attenuated names (the suspected housekeeping/ribosomal
/// drivers) so the run log shows what got down-weighted.
fn log_fisher_summary(weights: &[f32], table: &FeatureTable, penalty: f32) {
    let n = weights.len();
    if n == 0 {
        return;
    }
    let n_attenuated = weights.iter().filter(|&&w| w < 0.5).count();
    let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean = weights.iter().sum::<f32>() / n as f32;

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        weights[a]
            .partial_cmp(&weights[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<String> = idx
        .iter()
        .take(10)
        .map(|&g| format!("{}={:.3}", table.gene_names[g], weights[g]))
        .collect();

    info!(
        "NB-Fisher housekeeping penalty (exp={penalty}): {n_attenuated}/{n} genes w<0.5 \
         (min={min:.3}, mean={mean:.3}); most-attenuated: {}",
        top.join(", ")
    );
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
    // Count-comp accumulator keyed by (gene_id, splice_modality, axis_id)
    // so a gene's spliced and unspliced totals stay distinct pool entries
    // (each embeds with its own δ direction). AGG stays keyed by
    // (gene_id, axis_id) — the gene total, summed across splice types.
    let mut count_comp: FxHashMap<(u32, u32, u32), f32> = FxHashMap::default();
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
                // Splice modality (≥1) from the table; AGG sums all splice.
                let m = table.row_modality[row].unwrap_or(0);
                *count_comp.entry((g, m, aid)).or_insert(0.0) += t.count;
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

    // Anchor pools: region sentinel 0 (masked in the model). count-comp
    // carries its per-entry splice modality; AGG carries sentinel 0.
    let count_comp = count_pool_from_map(count_comp, tau);
    let agg = agg_pool_from_map(agg, tau);
    let modifier_comp_per_modality: Vec<StratumPool> = modifier_by_mod
        .into_iter()
        .zip(modifier_region)
        .enumerate()
        .map(|(modality, (m, region_map))| {
            satellite_pool_from_map(m, &region_map, modality as u32, tau)
        })
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

/// Assemble a `StratumPool` from `(gene, axis, modality, region, count)`
/// entries, τ-tempering the weights. All pool builders feed this — they
/// differ only in how they extract the fields from their map key.
fn pool_from_entries(
    n_hint: usize,
    tau: f32,
    entries: impl Iterator<Item = (u32, u32, u32, u32, f32)>,
) -> StratumPool {
    let mut gene_ids = Vec::with_capacity(n_hint);
    let mut axis_ids = Vec::with_capacity(n_hint);
    let mut modality_ids = Vec::with_capacity(n_hint);
    let mut region_ids = Vec::with_capacity(n_hint);
    let mut counts = Vec::with_capacity(n_hint);
    let mut weights = Vec::with_capacity(n_hint);
    for (g, aid, modality, region, c) in entries {
        gene_ids.push(g);
        axis_ids.push(aid);
        modality_ids.push(modality);
        region_ids.push(region);
        counts.push(c);
        weights.push(temper(c, tau));
    }
    StratumPool {
        gene_ids,
        axis_ids,
        modality_ids,
        region_ids,
        counts,
        weights,
    }
}

/// AGG anchor pool: keyed by (gene, axis); modality sentinel 0, region
/// sentinel 0 (the model masks the AGG gate to exp(0)).
fn agg_pool_from_map(map: FxHashMap<(u32, u32), f32>, tau: f32) -> StratumPool {
    let n = map.len();
    pool_from_entries(
        n,
        tau,
        map.into_iter().map(|((g, aid), c)| (g, aid, 0, 0, c)),
    )
}

/// Count-comp anchor pool: keyed by (gene, splice_modality, axis); each
/// entry carries its splice modality (≥1) so spliced/unspliced embed with
/// distinct δ directions. Region stays the sentinel 0.
fn count_pool_from_map(map: FxHashMap<(u32, u32, u32), f32>, tau: f32) -> StratumPool {
    let n = map.len();
    pool_from_entries(
        n,
        tau,
        map.into_iter().map(|((g, m, aid), c)| (g, aid, m, 0, c)),
    )
}

/// Satellite stratum (modifier-comp): keyed by (gene, component, axis);
/// each entry carries the component's transcript-position region from
/// the side map (default 0 when un-annotated). `modality` is the single
/// modality this pool was built for.
fn satellite_pool_from_map(
    map: FxHashMap<(u32, u32, u32), f32>,
    region_map: &FxHashMap<(u32, u32), u32>,
    modality: u32,
    tau: f32,
) -> StratumPool {
    let n = map.len();
    pool_from_entries(
        n,
        tau,
        map.into_iter().map(|((g, comp, aid), c)| {
            let region = region_map.get(&(g, comp)).copied().unwrap_or(0);
            (g, aid, modality, region, c)
        }),
    )
}
