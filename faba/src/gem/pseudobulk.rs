//! Multilevel pseudobulk collapse driven by the **spliced** count modality,
//! plus per-axis per-stratum aggregation of triplets for the sampler.
//!
//! Pipeline:
//!   1. Random projection of cells from a **spliced-only view** of the genes
//!      backend (`clone_for_collapse` + `mask_rows`), batch-corrected over the
//!      gene-sample batches. Because the collapse backend holds only spliced
//!      rows, both the projection and the refinement's gene-sum scoring are
//!      spliced-only — unspliced and satellite (m6A / A2I / pA) mass never
//!      enters the geometry.
//!   2. `collapse_columns_multilevel_with_hierarchy` on the spliced view
//!      → per-level `cell_to_pb` partitions over the genes cells.
//!   3. Walk the genes triplets (for `agg` / `count_comp`) and each
//!      **satellite** backend's triplets (for `modifier_comp`), accumulating
//!      per-(gene, axis_id) counts. Satellite columns attach to a pb by
//!      matching their cell id to a genes cell (`col_to_genes_cell`); columns
//!      with no genes match donate nothing (e.g. `mut` m6A on a sample that
//!      has no gene match). All component rows within a (g, m) pair share an
//!      embedding (e_{g,m,r} = β_g ⊙ exp(Σ_k z_{g,k}·δ_{k,m,:} + γ_{m,r,:})),
//!      so aggregation loses nothing for sampling and dedupes work.
//!
//! The two-stage trainer uses the pb axes for stage 1 (curriculum
//! training of β/z/δ/γ with lower-variance pb signal) and the cell axis
//! for stage 2 (refining e_cell with β/z/δ/γ frozen).

use anyhow::Context;
use data_beans_alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, CollapsedOut, MultilevelParams,
};
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::data::UnifiedData;
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::args::GemArgs;
use super::feature_table::{BackendRowMap, FeatureTable, RowStratum};

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
    /// Cell-axis pools (identity partition; one unit per cell). `Some` only
    /// when the sampler draws the cell axis (`use_phase1_cell_axis`); `None`
    /// in the default pure-pb path, where phase-2 streams from the backend
    /// rather than materialising this ~per-(gene,cell) object.
    pub cell_pools: Option<AxisPools>,
    /// Per-pb-level pools, coarsest-first.
    pub pb_pools_per_level: Vec<AxisPools>,
    /// Per-gene ubiquity (fraction of cells expressing, length `n_genes`,
    /// in `(0, 1]`) from the cell-axis count pool. NOT consumed by the
    /// model; persisted to `{out}.ubiquity.parquet` as a breadth diagnostic.
    pub gene_ubiquity: Vec<f32>,
    /// Genes/spliced batch fold-factor (finest-level `μ_residual`) for phase-2
    /// batch correction (divide each cell's counts by it before projection).
    /// `None` when there is ≤1 batch (no cross-batch counterfactual). In-memory
    /// only. See `cell_solve::BatchDivisor`.
    pub genes_residual: Option<GenesResidual>,
}

/// Finest-level genes/spliced batch fold-factor `μ_residual`, the raw material
/// for the phase-2 batch divide.
///
/// `mu_residual[(row, pb)]` is the posterior-mean fold-factor on a ratio scale
/// (~1); `row` indexes the masked spliced backend rows in ascending order,
/// mapped to a unified gene by `resid_row_to_gene[row]`; `pb` columns are
/// **finest-level** pb ids (== `PseudobulkData::cell_to_pb_per_level.last()`,
/// since that vec is coarsest-first). Consumers divide observed counts by this
/// (clamped, floored at 1.0 on a miss) so the de-batched signal is fit.
pub struct GenesResidual {
    pub mu_residual: DMatrix<f32>,
    pub resid_row_to_gene: Vec<u32>,
}

impl PseudobulkData {
    pub fn n_levels(&self) -> usize {
        self.pb_pools_per_level.len()
    }
}

/// One satellite (non-spliced) modality backend, paired with the data
/// needed to fold its mass into the spliced-driven pseudobulk. Held
/// **separately** from the genes backend — the collapse never sees
/// it; it only donates `modifier_comp` mass at aggregation time.
pub struct SatelliteData<'a> {
    /// The satellite backend, with `triplets` already materialized.
    pub unified: &'a UnifiedData,

    /// Row classification (global `(gene, modality, …)` ids) aligned
    /// to the satellite backend's compact rows.
    pub row_map: &'a BackendRowMap,

    /// Satellite column → genes cell id, or `None` when the column
    /// has no matching genes cell (its mass is dropped). Matched by
    /// `barcode@sample`.  In the refine pass this is recomputed
    /// against the subset genes cells.
    pub col_to_genes_cell: &'a [Option<usize>],
}

/// Backend row indices of the **spliced** features in the genes backend — the
/// rows that drive both the collapse geometry and the cell QC. Indexed at
/// backend scale via `feature_to_backend_row` (identity in the normal pass;
/// N_old in the refine pass). Falls back to every count row when there is no
/// `spliced` modality, so a count matrix without a splice split still works.
pub fn spliced_backend_rows(
    unified: &UnifiedData,
    table: &FeatureTable,
    genes_row_map: &BackendRowMap,
) -> Vec<usize> {
    let spliced_id = table.spliced_modality_id();
    let rows: Vec<usize> = (0..genes_row_map.stratum.len())
        .filter(|&uid| {
            genes_row_map.stratum[uid] == Some(RowStratum::CountComp)
                && spliced_id.is_some()
                && genes_row_map.modality[uid] == spliced_id
        })
        .map(|uid| unified.feature_to_backend_row[uid])
        .collect();
    if !rows.is_empty() {
        return rows;
    }
    // No spliced track — fall back to all count rows so the collapse + QC still
    // have geometry to work with.
    (0..genes_row_map.stratum.len())
        .filter(|&uid| genes_row_map.stratum[uid] == Some(RowStratum::CountComp))
        .map(|uid| unified.feature_to_backend_row[uid])
        .collect()
}

/// Build the multilevel partition + the per-axis sampling pools.
///
/// The collapse runs on a transient **spliced-only clone** of the
/// genes backend (`clone_for_collapse` + `mask_rows`), so `unified`'s
/// own backend is left untouched (its full rows are still needed for
/// `agg`/`count_comp` aggregation). `genes_row_map` classifies the
/// genes backend's rows; the spliced mask is derived from
/// it. `satellites` donate `modifier_comp` mass.
///
/// Both the normal pass and the `--refine` pass call this identically: the
/// refine pass drops only null *genes* (`subset_features`) and keeps every
/// cell, so cell id == backend column in both, triplets are materialized
/// fresh, and batch labels come straight from `unified`.
pub fn build_pseudobulk(
    unified: &mut UnifiedData,
    table: &FeatureTable,
    genes_row_map: &BackendRowMap,
    satellites: &[SatelliteData],
    args: &GemArgs,
    // Build the per-(gene, cell) cell-axis pools? Only needed when the sampler
    // draws the cell axis (`use_phase1_cell_axis`); the default pure-pb path
    // leaves it `None` and phase-2 streams from the backend instead, never
    // materialising the ~per-(gene,cell) pool (the 700k-cell OOM driver).
    build_cell_pools: bool,
) -> anyhow::Result<PseudobulkData> {
    ////////////////////////////////////////
    // 1. Batch labels + projection
    ////////////////////////////////////////
    // Batch labels at BACKEND-column scale (N == backend columns; the refine
    // pass keeps every cell, so this holds in both passes).
    let batch_labels: Vec<Box<str>> = unified.batch_labels();
    let batch_arg = (unified.n_batches() > 1).then_some(batch_labels.as_slice());

    // Spliced-only collapse view: clone the genes backend (matrices are
    // Arc-shared — cheap) and mask it down to the spliced rows. The clone
    // drives projection + collapse + refinement, so the geometry is spliced-
    // only by construction (no row-weights, no satellite/unspliced leakage),
    // while `unified`'s own backend keeps all rows for aggregation.
    //
    // The spliced keep-mask is indexed by **backend row** (not the compact
    // unified axis): in the refine pass the backend still has all N_old rows
    // while `genes_row_map` only covers the N_new live features, so each live
    // spliced feature is placed at its backend row via `feature_to_backend_row`
    // (identity in the normal pass).
    let spliced_rows = spliced_backend_rows(unified, table, genes_row_map);
    let mut spliced_keep = vec![false; unified.count_backend().num_rows()];
    for &r in &spliced_rows {
        spliced_keep[r] = true;
    }

    let mut spliced_backend = unified.count_backend().clone_for_collapse();
    spliced_backend
        .mask_rows(&spliced_keep)
        .context("masking collapse backend to spliced rows")?;

    info!(
        "projection (proj_dim={}, {} batches, {} spliced rows)...",
        args.collapse.proj_dim,
        unified.n_batches(),
        spliced_backend.num_rows(),
    );
    // Unweighted projection — the backend is already spliced-only.
    let row_weights = vec![1.0_f32; spliced_backend.num_rows()];
    let proj = spliced_backend
        .project_columns_weighted(args.collapse.proj_dim, None, batch_arg, &row_weights)
        .context("spliced random projection")?;

    ////////////////////////////////////////
    // 2. Multilevel collapse on the spliced view
    ////////////////////////////////////////
    info!(
        "multilevel collapse (sort_dim={}, {} levels)...",
        args.collapse.sort_dim, args.collapse.num_levels
    );
    let collapse_out = collapse_columns_multilevel_with_hierarchy(
        &mut spliced_backend,
        &proj.proj,
        &batch_labels,
        &MultilevelParams {
            knn_pb_samples: args.collapse.knn_pb,
            num_levels: args.collapse.num_levels.max(1),
            sort_dim: args.collapse.sort_dim,
            num_opt_iter: args.collapse.num_opt_iter,
            refine: Some(Default::default()),
            output_calibration: matrix_param::traits::CalibrateTarget::MeanOnly,
        },
    )
    .context("multilevel collapse on spliced view")?;
    drop(spliced_backend);

    // Returned finest-first; flip to coarsest-first.
    let mut cell_to_pb_per_level = collapse_out.cell_to_pb_per_level;
    cell_to_pb_per_level.reverse();

    // Retain the finest-level genes/spliced μ_residual for phase-2 batch
    // correction (the collapse only fits it with >1 batch). Its pb columns are
    // the finest level == `cell_to_pb_per_level.last()` (now coarsest-first).
    let genes_residual = (unified.n_batches() > 1)
        .then(|| extract_genes_residual(&collapse_out.levels, unified, table, genes_row_map))
        .flatten();
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

    // Cell axis: identity partition. Built only when the sampler needs it;
    // otherwise phase-2 streams from the backend (see `cell_solve`).
    let cell_pools: Option<AxisPools> = build_cell_pools.then(|| {
        let cell_identity: Vec<usize> = (0..n_cells).collect();
        aggregate_pools(
            unified,
            genes_row_map,
            satellites,
            &cell_identity,
            n_cells,
            n_modalities,
            args.train.tau,
        )
    });

    // Pb axes: one set of pools per level (coarsest-first). Each level
    // walks the genes + satellite triplets once into independent thread-local
    // maps, so the levels are embarrassingly parallel — fan out across them
    // rather than making `num_levels` sequential passes over the edge lists.
    // The shared reborrow keeps the closure capturing `&UnifiedData` (Sync),
    // not the outer `&mut`.
    let unified_ref: &UnifiedData = unified;
    let pb_pools_per_level: Vec<AxisPools> = cell_to_pb_per_level
        .par_iter()
        .map(|cell_to_pb| {
            let n_pbs = cell_to_pb.iter().copied().max().map(|m| m + 1).unwrap_or(0);
            aggregate_pools(
                unified_ref,
                genes_row_map,
                satellites,
                cell_to_pb,
                n_pbs,
                n_modalities,
                args.train.tau,
            )
        })
        .collect();

    match &cell_pools {
        Some(cp) => info!(
            "cell axis: {} cells, {} agg / {} count-comp / {} modifier-comp draws",
            cp.n_units,
            cp.agg.len(),
            cp.count_comp.len(),
            cp.modifier_comp_per_modality
                .iter()
                .map(|p| p.len())
                .sum::<usize>()
        ),
        None => info!("cell axis: {n_cells} cells (streamed in phase 2, pool not materialised)"),
    }
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
    // 4. Per-gene ubiquity (breadth diagnostic)
    ////////////////////////////////////////
    // = distinct cells expressing each CountComp gene / n_cells. When the cell
    // pool was built, read its AGG stratum (one per-(gene, cell) entry); when
    // streaming, count distinct (gene, cell) over the column-grouped triplets
    // directly (bit-identical). Abundance balance is the sampler's `count^τ`
    // tempering, not a separate down-weight (see `gene_weight`).
    let n_genes = table.n_genes();
    let gene_ubiquity = match &cell_pools {
        Some(p) => super::gene_weight::ubiquity_from_count_pool(&p.agg, n_genes, n_cells),
        None => ubiquity_from_triplets(unified, genes_row_map, n_genes, n_cells),
    };

    Ok(PseudobulkData {
        cell_to_pb_per_level,
        cell_pools,
        pb_pools_per_level,
        gene_ubiquity,
        genes_residual,
    })
}

/// Lift the finest-level genes/spliced μ_residual out of the collapse output
/// into a [`GenesResidual`], mapping each matrix row back to its unified gene.
///
/// `mask_rows` renumbers the kept spliced rows in **ascending** old-row order,
/// so μ_residual row `i` is the `i`-th smallest spliced backend row; we recover
/// its gene by sorting the same `spliced_backend_rows` used to build the mask.
/// Returns `None` (correction simply skipped) if the finest level carries no
/// μ_residual or the row counts disagree — never a wrong mapping.
fn extract_genes_residual(
    levels_finest_first: &[CollapsedOut],
    unified: &UnifiedData,
    table: &FeatureTable,
    genes_row_map: &BackendRowMap,
) -> Option<GenesResidual> {
    let finest = levels_finest_first.first()?;
    let mu_residual = finest.mu_residual.as_ref()?.posterior_mean().clone();

    // backend compact row → unified gene, over every count row.
    let mut row_to_gene: FxHashMap<usize, u32> = FxHashMap::default();
    for uid in 0..genes_row_map.stratum.len() {
        if genes_row_map.stratum[uid] == Some(RowStratum::CountComp) {
            if let Some(g) = genes_row_map.gene[uid] {
                row_to_gene.insert(unified.feature_to_backend_row[uid], g);
            }
        }
    }
    let mut kept_rows = spliced_backend_rows(unified, table, genes_row_map);
    kept_rows.sort_unstable();
    if kept_rows.len() != mu_residual.nrows() {
        log::warn!(
            "genes μ_residual rows ({}) ≠ spliced rows ({}); skipping phase-2 batch divide",
            mu_residual.nrows(),
            kept_rows.len()
        );
        return None;
    }
    let resid_row_to_gene: Vec<u32> = kept_rows.iter().map(|r| row_to_gene[r]).collect();
    info!(
        "retained genes μ_residual: {} spliced rows × {} finest pbs",
        mu_residual.nrows(),
        mu_residual.ncols()
    );
    Some(GenesResidual {
        mu_residual,
        resid_row_to_gene,
    })
}

/// Per-gene ubiquity (distinct cells expressing each CountComp gene / n_cells)
/// straight from `unified.triplets`, used when the cell pool isn't built.
/// Triplets are column-grouped (one cell's nonzeros contiguous), so a gene's
/// spliced + unspliced rows for the same cell are deduped with a `last_cell`
/// watermark — matching `ubiquity_from_count_pool(&agg)` exactly.
fn ubiquity_from_triplets(
    unified: &UnifiedData,
    genes_row_map: &BackendRowMap,
    n_genes: usize,
    n_cells: usize,
) -> Vec<f32> {
    let mut cells_with = vec![0u32; n_genes];
    let mut last_cell = vec![u32::MAX; n_genes];
    for t in &unified.triplets {
        let row = t.feature as usize;
        if genes_row_map.stratum[row] != Some(RowStratum::CountComp) {
            continue;
        }
        let Some(g) = genes_row_map.gene[row] else {
            continue;
        };
        let g = g as usize;
        if g < n_genes && last_cell[g] != t.cell {
            cells_with[g] += 1;
            last_cell[g] = t.cell;
        }
    }
    let inv_n = 1.0 / n_cells.max(1) as f32;
    cells_with
        .iter()
        .map(|&c| (c as f32 * inv_n).clamp(0.0, 1.0))
        .collect()
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

/// Aggregate the genes triplets (`agg` / `count_comp`) and each satellite
/// backend's triplets (`modifier_comp`) to per-(gene, axis_id) granularity per
/// stratum. `cell_to_axis_id[c]` gives the right-hand-axis id for **genes**
/// cell `c` — identity for the cell axis, `cell_to_pb_per_level[ℓ]` for pb
/// axes. Satellite columns are mapped to a genes cell via
/// `sat.col_to_genes_cell` first; unmatched columns contribute nothing.
fn aggregate_pools(
    genes: &UnifiedData,
    genes_row_map: &BackendRowMap,
    satellites: &[SatelliteData],
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

    // One accumulation rule, fed by both the genes triplets and each satellite's
    // triplets — the only per-source differences are the row-map and how the axis
    // id is resolved. Preserves the per-stratum `None`-handling: a ModifierComp
    // row with no modality is skipped; a CountComp row with no modality folds into
    // the AGG base (modality 0); a row with no gene is skipped.
    let mut accumulate = |stratum: RowStratum,
                          gene: Option<u32>,
                          modality: Option<u32>,
                          component: Option<u32>,
                          region: Option<u32>,
                          count: f32,
                          aid: u32| {
        let Some(g) = gene else { return };
        match stratum {
            RowStratum::CountComp => {
                // Splice modality (≥1) from the row map; AGG sums all splice.
                let m = modality.unwrap_or(0);
                *count_comp.entry((g, m, aid)).or_insert(0.0) += count;
                *agg.entry((g, aid)).or_insert(0.0) += count;
            }
            RowStratum::ModifierComp => {
                if let Some(m) = modality {
                    let comp = component.unwrap_or(0);
                    let region = region.unwrap_or(0);
                    *modifier_by_mod[m as usize]
                        .entry((g, comp, aid))
                        .or_insert(0.0) += count;
                    modifier_region[m as usize].insert((g, comp), region);
                }
            }
            RowStratum::Site => {}
        }
    };

    // Genes backend: agg + count_comp (+ any modifier rows, defensively).
    for t in &genes.triplets {
        let row = t.feature as usize;
        let Some(stratum) = genes_row_map.stratum[row] else {
            continue;
        };
        let aid = cell_to_axis_id[t.cell as usize] as u32;
        accumulate(
            stratum,
            genes_row_map.gene[row],
            genes_row_map.modality[row],
            genes_row_map.component[row],
            genes_row_map.region[row],
            t.count,
            aid,
        );
    }

    // Satellite backends: modifier_comp (+ any count rows), attached to the
    // matched genes cell via col_to_genes_cell; unmatched columns donate nothing.
    for sat in satellites {
        for t in &sat.unified.triplets {
            let Some(genes_cell) = sat.col_to_genes_cell[t.cell as usize] else {
                continue;
            };
            let row = t.feature as usize;
            let Some(stratum) = sat.row_map.stratum[row] else {
                continue;
            };
            let aid = cell_to_axis_id[genes_cell] as u32;
            accumulate(
                stratum,
                sat.row_map.gene[row],
                sat.row_map.modality[row],
                sat.row_map.component[row],
                sat.row_map.region[row],
                t.count,
                aid,
            );
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
