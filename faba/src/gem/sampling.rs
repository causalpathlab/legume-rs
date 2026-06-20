//! Stratum-balanced count-weighted positive sampler + random /
//! swap-gene-mode negative draws. Tensor-free; the loss converts these
//! edge batches into tensor batches via `model::embed_and_bias_rows`.
//!
//! One `SamplerState` carries pre-built distributions for **every
//! axis** (cell + each pb level). `draw_minibatch(axis, ...)` dispatches
//! to the right pool. Used by stage 1 (pb axes) and stage 2 (cell axis).

use rand::distr::{weighted::WeightedIndex, Distribution, Uniform};
use rand::seq::IndexedRandom;
use rand::Rng;

use super::args::GemArgs;
use super::feature_table::FeatureTable;
use super::model::{Axis, FeatureRows};
use super::pseudobulk::{AxisPools, PseudobulkData, StratumPool};

/// Implement `feature_rows()` for an edge-batch struct carrying the seven
/// `FeatureRows` index fields, borrowing them as a [`FeatureRows`] for
/// [`GemModel::embed_and_bias_rows`](super::model::GemModel::embed_and_bias_rows).
/// `PositiveEdges` and `NegativeEdges` share the exact field set, so the body is
/// generated once here rather than copy-pasted per type.
macro_rules! impl_feature_rows {
    ($t:ty) => {
        impl $t {
            pub fn feature_rows(&self) -> FeatureRows<'_> {
                FeatureRows {
                    gene_for_rho: &self.gene_for_rho,
                    gene_for_z: &self.gene_for_z,
                    modality_for_q: &self.modality_for_q,
                    region_for_delta: &self.region_for_delta,
                    gene_for_bias: &self.gene_for_bias,
                    modality_for_bias: &self.modality_for_bias,
                    is_agg: &self.is_agg,
                }
            }
        }
    };
}

/// Per-positive identity. Positives have `gene_for_rho == gene_for_z`;
/// negatives may decouple them (swap-z), carry a different
/// `modality_for_q` (swap-modality), etc.
pub struct PositiveEdges {
    pub gene_for_rho: Vec<u32>,
    pub gene_for_z: Vec<u32>,
    pub modality_for_q: Vec<u32>,
    /// Transcript-position region selecting γ_{m,r,:}. Anchor rows carry
    /// the sentinel 0 (masked in the model).
    pub region_for_delta: Vec<u32>,
    pub gene_for_bias: Vec<u32>,
    pub modality_for_bias: Vec<u32>,
    pub is_agg: Vec<bool>,
    /// Right-hand-axis id (cell id for `Axis::Cell`; pb id for `Axis::Pb`).
    pub axis_id: Vec<u32>,
}

impl PositiveEdges {
    fn with_capacity(cap: usize) -> Self {
        PositiveEdges {
            gene_for_rho: Vec::with_capacity(cap),
            gene_for_z: Vec::with_capacity(cap),
            modality_for_q: Vec::with_capacity(cap),
            region_for_delta: Vec::with_capacity(cap),
            gene_for_bias: Vec::with_capacity(cap),
            modality_for_bias: Vec::with_capacity(cap),
            is_agg: Vec::with_capacity(cap),
            axis_id: Vec::with_capacity(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.gene_for_rho.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gene_for_rho.is_empty()
    }
}

/// Per-negative identity, flattened `[B * k]`.
pub struct NegativeEdges {
    pub gene_for_rho: Vec<u32>,
    pub gene_for_z: Vec<u32>,
    pub modality_for_q: Vec<u32>,
    pub region_for_delta: Vec<u32>,
    pub gene_for_bias: Vec<u32>,
    pub modality_for_bias: Vec<u32>,
    pub is_agg: Vec<bool>,
    pub k: usize,
}

impl NegativeEdges {
    /// Empty `[B*k]` edge batch with capacity for `total` rows. `k` rides
    /// along so the loss can reshape into `[B, k]`.
    fn with_capacity(total: usize, k: usize) -> Self {
        NegativeEdges {
            gene_for_rho: Vec::with_capacity(total),
            gene_for_z: Vec::with_capacity(total),
            modality_for_q: Vec::with_capacity(total),
            region_for_delta: Vec::with_capacity(total),
            gene_for_bias: Vec::with_capacity(total),
            modality_for_bias: Vec::with_capacity(total),
            is_agg: Vec::with_capacity(total),
            k,
        }
    }

    /// Degenerate edge batch the loss path skips (k = 0). Used when a stratum
    /// has no gene distinct from the positive.
    fn empty() -> Self {
        Self::with_capacity(0, 0)
    }

    pub fn len(&self) -> usize {
        self.gene_for_rho.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gene_for_rho.is_empty()
    }
}

impl_feature_rows!(PositiveEdges);
impl_feature_rows!(NegativeEdges);

pub struct SubBatch {
    pub positives: PositiveEdges,
    pub rand: NegativeEdges,
    /// Swap-gene-mode negatives: keep β_g and (modality m, region r);
    /// borrow z from another gene. Tests the gene's K-program loading.
    pub swap_gene_mode: Option<NegativeEdges>,
    /// Swap-modality negatives: keep (gene, cell); swap the (modality,
    /// region) pair to another satellite axis. Tests that a satellite
    /// stays distinguishable from the base / other modalities.
    pub swap_modality: Option<NegativeEdges>,
}

pub struct Minibatch {
    pub anchor: Option<SubBatch>,
    pub modifier: Option<SubBatch>,
}

/// Pre-built distributions for one axis.
pub struct AxisDists {
    pub agg: Option<WeightedIndex<f32>>,
    pub count: Option<WeightedIndex<f32>>,
    pub modifier_modality: Option<WeightedIndex<f32>>,
    pub modifier_per_modality: Vec<Option<WeightedIndex<f32>>>,
}

pub struct SamplerState {
    pub cell: AxisDists,
    /// Phase-1-only subsampled cell-axis pools (`--phase1-cells-per-pb k`
    /// with `1 ≤ k < n_cells`). `Some` → the phase-1 cell axis draws from
    /// these (≤k cells per pb-sample) and `cell` above is built from them;
    /// `None` → the full `pb.cell_pools` are used (k = 0 suppresses the
    /// axis entirely, so it's never drawn; k ≥ n_cells is the full set).
    /// Phase 2 always uses the full `pb.cell_pools` regardless.
    pub phase1_cell_pools: Option<AxisPools>,
    pub pb_per_level: Vec<AxisDists>,
    pub modifier_modality_ids: Vec<u32>,
    pub measured_genes_per_modality: Vec<Vec<u32>>,
    /// `is_count_modality[m]` = true for the count-derived satellite
    /// modalities (`spliced` / `unspliced`). Used to route a count
    /// positive's random negatives to the count pool (filtered to the
    /// same splice modality) rather than the modifier pools.
    pub is_count_modality: Vec<bool>,
    /// Count satellite modality ids (`spliced` / `unspliced`). The
    /// swap-modality negatives for anchor positives are drawn from this
    /// set so spliced and unspliced separate directionally.
    pub count_modality_ids: Vec<u32>,
    pub n_regions: usize,
}

impl SamplerState {
    pub fn new(table: &FeatureTable, pb: &PseudobulkData, args: &GemArgs) -> Self {
        let n_modalities = table.n_modalities();
        let n_regions = table.n_regions.max(1);

        let modifier_modality_ids: Vec<u32> = (1..n_modalities as u32)
            .filter(|&m| !table.modifier_rows_by_modality[m as usize].is_empty())
            .collect();

        let measured_genes_per_modality: Vec<Vec<u32>> = (0..n_modalities)
            .map(|m| table.measured_genes_for_modality(m as u32))
            .collect();

        let tau_m = args.train.tau_modality.max(0.0);
        let build_dists = |pool: &AxisPools| -> AxisDists {
            let mod_weights: Vec<f32> = modifier_modality_ids
                .iter()
                .map(|&m| {
                    let mass = pool.modality_total_mass[m as usize].max(0.0);
                    if mass > 0.0 {
                        mass.powf(tau_m)
                    } else {
                        0.0
                    }
                })
                .collect();
            AxisDists {
                agg: weighted_index(&pool.agg.weights),
                count: weighted_index(&pool.count_comp.weights),
                modifier_modality: weighted_index(&mod_weights),
                modifier_per_modality: pool
                    .modifier_comp_per_modality
                    .iter()
                    .map(|p| weighted_index(&p.weights))
                    .collect(),
            }
        };

        // Phase-1 cell-axis subsampling (`--phase1-cells-per-pb k`). Only the
        // `1 ≤ k < n_cells` band builds a smaller view; k = 0 (axis
        // suppressed, never drawn) and k ≥ n_cells (no-op) both keep the full
        // pools. The cell dists below are built from whichever pool the
        // phase-1 cell axis will actually draw from.
        let k = args.collapse.phase1_cells_per_pb;
        // `None` cell pool ⇒ the cell axis is off (streaming phase-2 path);
        // n_cells 0 then disables subsampling and the cell dists stay empty.
        let n_cells = pb.cell_pools.as_ref().map_or(0, |p| p.n_units);
        let phase1_cell_pools: Option<AxisPools> = (k >= 1 && k < n_cells).then(|| {
            super::pseudobulk::subsample_cell_pools_multilevel(
                pb.cell_pools
                    .as_ref()
                    .expect("cell pool present when subsampling"),
                &pb.cell_to_pb_per_level,
                k,
                args.runtime.seed,
            )
        });
        let cell = match pb.cell_pools.as_ref() {
            Some(cp) => build_dists(phase1_cell_pools.as_ref().unwrap_or(cp)),
            // Cell axis never drawn (off) → empty dists, never sampled.
            None => AxisDists {
                agg: None,
                count: None,
                modifier_modality: None,
                modifier_per_modality: Vec::new(),
            },
        };
        let pb_per_level = pb.pb_pools_per_level.iter().map(build_dists).collect();

        Self {
            cell,
            phase1_cell_pools,
            pb_per_level,
            modifier_modality_ids,
            measured_genes_per_modality,
            is_count_modality: table.is_count_modality.clone(),
            count_modality_ids: table.count_modality_ids.clone(),
            n_regions,
        }
    }

    pub fn dists(&self, axis: Axis) -> &AxisDists {
        match axis {
            Axis::Cell => &self.cell,
            Axis::Pb(level) => &self.pb_per_level[level],
        }
    }

    /// Number of cell-axis units shaping the phase-1 budget: the kept-cell
    /// count when subsampling is active, else the full `n_cells`.
    pub fn n_cell_units(&self, n_cells: usize) -> usize {
        self.phase1_cell_pools
            .as_ref()
            .map_or(n_cells, |p| p.n_units)
    }
}

/// Pool the given axis draws from. The cell axis is **phase-1 only** (phase
/// 2 is the analytical `cell_solve` path), so when a subsampled phase-1
/// view exists it's used here; otherwise the full `pb.cell_pools`.
fn axis_pools<'a>(state: &'a SamplerState, pb: &'a PseudobulkData, axis: Axis) -> &'a AxisPools {
    match axis {
        Axis::Cell => state
            .phase1_cell_pools
            .as_ref()
            .or(pb.cell_pools.as_ref())
            .expect("cell axis drawn but no cell pool built"),
        Axis::Pb(level) => &pb.pb_pools_per_level[level],
    }
}

fn weighted_index(w: &[f32]) -> Option<WeightedIndex<f32>> {
    // `v > 0.0` is false for NaN, ≤ 0, and -∞; we want the same
    // exclusion semantics here. `v.partial_cmp(&0.0) != Some(Greater)`
    // matches NaN-as-not-positive without tripping
    // `neg_cmp_op_on_partial_ord`.
    if w.is_empty()
        || w.iter()
            .all(|&v| v.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
    {
        return None;
    }
    WeightedIndex::new(w).ok()
}

/// Draw one minibatch on the given axis. Total positive budget
/// `B = args.train.batch_size`, partitioned across strata by
/// `args.train.f_agg` / `args.train.f_count` (modifier gets the remainder).
pub fn draw_minibatch<R: Rng>(
    axis: Axis,
    state: &SamplerState,
    pb: &PseudobulkData,
    args: &GemArgs,
    rng: &mut R,
) -> Minibatch {
    let b_total = args.train.batch_size;
    let b_agg = (args.train.f_agg.clamp(0.0, 1.0) * b_total as f32).round() as usize;
    let b_count = (args.train.f_count.clamp(0.0, 1.0) * b_total as f32).round() as usize;
    let b_modifier = b_total.saturating_sub(b_agg + b_count);

    let pools = axis_pools(state, pb, axis);
    let dists = state.dists(axis);

    let anchor = build_anchor_sub_batch(state, pools, dists, b_agg, b_count, args, rng);
    let modifier = build_modifier_sub_batch(state, pools, dists, b_modifier, args, rng);

    Minibatch { anchor, modifier }
}

fn build_anchor_sub_batch<R: Rng>(
    state: &SamplerState,
    pools: &AxisPools,
    dists: &AxisDists,
    b_agg: usize,
    b_count: usize,
    args: &GemArgs,
    rng: &mut R,
) -> Option<SubBatch> {
    let mut positives = PositiveEdges::with_capacity(b_agg + b_count);

    if let Some(dist) = dists.agg.as_ref() {
        for _ in 0..b_agg {
            let i = dist.sample(rng);
            push_agg_positive(&mut positives, &pools.agg, i);
        }
    }

    if let Some(dist) = dists.count.as_ref() {
        for _ in 0..b_count {
            let i = dist.sample(rng);
            // δ/γ modality rides along from the pool entry — `spliced` /
            // `unspliced`, each its own id ≥ 1 (distinct δ direction). The
            // **bias** uses the shared count slot 0, so spliced/unspliced
            // can't be told apart by a per-modality bias column — the
            // directional contrast is forced into the embedding (a distinct
            // b_comp per splice would let the bias absorb it and δ never
            // learns to separate them).
            let modality = pools.count_comp.modality_ids[i];
            push_component_positive(
                &mut positives,
                &pools.count_comp,
                i,
                modality,
                COUNT_BIAS_MODALITY,
            );
        }
    }

    if positives.is_empty() {
        return None;
    }

    let rand = draw_random_negatives(state, pools, dists, &positives, args.train.n_rand, rng);
    // Swap-modality negatives over the count satellite modalities: keep the
    // gene/cell, swap the splice modality (spliced↔unspliced). This is the
    // contrast that actively separates the two splice directions (and base↔
    // splice for AGG positives). Swap-gene-mode is intentionally skipped
    // here — it would be a no-op (and corrupt the NCE) on AGG positives,
    // whose gate is masked so z is irrelevant.
    let swap_modality =
        draw_count_swap_modality_negatives(state, &positives, args.train.n_swap_modality, rng);

    Some(SubBatch {
        positives,
        rand,
        swap_gene_mode: None,
        swap_modality,
    })
}

fn build_modifier_sub_batch<R: Rng>(
    state: &SamplerState,
    pools: &AxisPools,
    dists: &AxisDists,
    b: usize,
    args: &GemArgs,
    rng: &mut R,
) -> Option<SubBatch> {
    if b == 0 {
        return None;
    }
    let mod_dist = dists.modifier_modality.as_ref()?;
    let modifier_mods = &state.modifier_modality_ids;
    if modifier_mods.is_empty() {
        return None;
    }

    let mut positives = PositiveEdges::with_capacity(b);

    for _ in 0..b {
        let m_idx = mod_dist.sample(rng);
        let m = modifier_mods[m_idx];
        let Some(pool_dist) = dists.modifier_per_modality[m as usize].as_ref() else {
            continue;
        };
        let i = pool_dist.sample(rng);
        push_component_positive(
            &mut positives,
            &pools.modifier_comp_per_modality[m as usize],
            i,
            m,
            /*bias_modality=*/ m,
        );
    }

    if positives.is_empty() {
        return None;
    }

    let rand = draw_random_negatives(state, pools, dists, &positives, args.train.n_rand, rng);
    let swap_gene_mode =
        draw_swap_gene_mode_negatives(state, &positives, args.train.n_swap_gene_mode, rng);
    let swap_modality =
        draw_swap_modality_negatives(state, &positives, args.train.n_swap_modality, rng);

    Some(SubBatch {
        positives,
        rand,
        swap_gene_mode,
        swap_modality,
    })
}

fn push_agg_positive(edges: &mut PositiveEdges, pool: &StratumPool, i: usize) {
    let g = pool.gene_ids[i];
    let a = pool.axis_ids[i];
    edges.gene_for_rho.push(g);
    edges.gene_for_z.push(g);
    edges.modality_for_q.push(0);
    edges.region_for_delta.push(0); // sentinel; masked for AGG
    edges.gene_for_bias.push(g);
    edges.modality_for_bias.push(0);
    edges.is_agg.push(true);
    edges.axis_id.push(a);
}

fn push_component_positive(
    edges: &mut PositiveEdges,
    pool: &StratumPool,
    i: usize,
    modality: u32,
    bias_modality: u32,
) {
    let g = pool.gene_ids[i];
    let a = pool.axis_ids[i];
    edges.gene_for_rho.push(g);
    edges.gene_for_z.push(g);
    edges.modality_for_q.push(modality);
    // Region rides along from the pool entry (satellite = real region;
    // count-comp anchor = sentinel 0).
    edges.region_for_delta.push(pool.region_ids[i]);
    edges.gene_for_bias.push(g);
    // `bias_modality` may differ from `modality`: count-splice rows use the
    // shared count bias slot so the bias can't absorb the splice contrast.
    edges.modality_for_bias.push(bias_modality);
    edges.is_agg.push(false);
    edges.axis_id.push(a);
}

const REJECT_RETRIES: usize = 8;

/// Bias slot shared by all count-comp rows (spliced/unspliced + AGG base).
/// Splice modalities get distinct δ directions but a *shared* bias column,
/// so the per-modality bias can't absorb the spliced↔unspliced contrast —
/// the directional signal is forced into the embedding. Shared with
/// `cell_solve` (phase-2 must project against the same frozen bias).
pub(crate) const COUNT_BIAS_MODALITY: u32 = 0;

fn draw_random_negatives<R: Rng>(
    state: &SamplerState,
    pools: &AxisPools,
    dists: &AxisDists,
    positives: &PositiveEdges,
    k: usize,
    rng: &mut R,
) -> NegativeEdges {
    let b = positives.len();
    let total = b * k;
    let mut out = NegativeEdges::with_capacity(total, k);
    if k == 0 {
        return out;
    }
    for i in 0..b {
        let pos_gene = positives.gene_for_rho[i];
        let pos_modality = positives.modality_for_q[i];
        let pos_region = positives.region_for_delta[i];
        let pos_is_agg = positives.is_agg[i];

        // Count satellites (`spliced` / `unspliced`, modality ≥ 1) and true
        // modifiers both have is_agg=false; route them by which kind of
        // modality the positive carries. Count negatives are restricted to
        // the **same** splice modality so they probe gene-cell identity
        // within spliced (or within unspliced), not across.
        let is_count = state
            .is_count_modality
            .get(pos_modality as usize)
            .copied()
            .unwrap_or(false);
        let (pool, dist_opt, require_modality) = if pos_is_agg {
            (&pools.agg, dists.agg.as_ref(), None)
        } else if is_count {
            (&pools.count_comp, dists.count.as_ref(), Some(pos_modality))
        } else {
            (
                &pools.modifier_comp_per_modality[pos_modality as usize],
                dists.modifier_per_modality[pos_modality as usize].as_ref(),
                None,
            )
        };

        for _ in 0..k {
            let Some(g_neg) = pool_sample_distinct(pool, dist_opt, pos_gene, require_modality, rng)
            else {
                // No gene distinct from the positive exists in this
                // stratum (e.g. only one distinct gene in the pool).
                // A "negative" equal to the positive has identical
                // e_f / b_f and corrupts the NCE softmax denominator,
                // so drop the whole edge batch: the caller's loss path
                // will skip score_negative_edges when k=0.
                return NegativeEdges::empty();
            };
            out.gene_for_rho.push(g_neg);
            out.gene_for_z.push(g_neg);
            out.modality_for_q.push(pos_modality);
            // Random negatives keep the positive's (modality, region) and
            // swap only the gene → they probe gene-cell identification.
            out.region_for_delta.push(pos_region);
            out.gene_for_bias.push(g_neg);
            // Count satellites share the count bias slot (see
            // `COUNT_BIAS_MODALITY`); modifiers keep their own column.
            let bias_modality = if is_count {
                COUNT_BIAS_MODALITY
            } else {
                pos_modality
            };
            out.modality_for_bias.push(bias_modality);
            out.is_agg.push(pos_is_agg);
        }
    }
    out
}

fn pool_sample_distinct<R: Rng>(
    pool: &StratumPool,
    dist: Option<&WeightedIndex<f32>>,
    exclude_gene: u32,
    require_modality: Option<u32>,
    rng: &mut R,
) -> Option<u32> {
    let dist = dist?;
    // An entry is admissible if its gene differs from the positive and (for
    // the count pool) it carries the required splice modality.
    let admissible = |idx: usize| {
        pool.gene_ids[idx] != exclude_gene
            && require_modality.is_none_or(|m| pool.modality_ids[idx] == m)
    };
    // First, count-weighted rejection draws (preserves the τ-tempered
    // sampling distribution in the common case).
    for _ in 0..REJECT_RETRIES {
        let idx = dist.sample(rng);
        if admissible(idx) {
            return Some(pool.gene_ids[idx]);
        }
    }
    // Rare collision storm (e.g. one gene dominates the τ-weighted mass, or
    // the required splice modality is sparse). Fall back to a deterministic
    // linear scan — return the first admissible gene, else None so the
    // caller drops the edge batch.
    (0..pool.len())
        .find(|&idx| admissible(idx))
        .map(|idx| pool.gene_ids[idx])
}

fn draw_swap_gene_mode_negatives<R: Rng>(
    state: &SamplerState,
    positives: &PositiveEdges,
    k: usize,
    rng: &mut R,
) -> Option<NegativeEdges> {
    if k == 0 {
        return None;
    }
    let b = positives.len();
    let mut out = NegativeEdges::with_capacity(b * k, k);
    for i in 0..b {
        let g = positives.gene_for_rho[i];
        let m = positives.modality_for_q[i];
        let r = positives.region_for_delta[i];
        let pool = &state.measured_genes_per_modality[m as usize];
        // Need ≥ 2 distinct measured genes for the modality to even
        // construct a single swap-z negative; if not, drop the whole
        // swap-z edges this minibatch (NCE loss skips it gracefully).
        if pool.len() < 2 {
            return None;
        }
        for _ in 0..k {
            let g_prime = swap_pick_distinct(pool, g, rng)?;
            out.gene_for_rho.push(g);
            out.gene_for_z.push(g_prime);
            out.modality_for_q.push(m);
            // Keep (modality, region); only z's source gene changes.
            out.region_for_delta.push(r);
            out.gene_for_bias.push(g);
            out.modality_for_bias.push(m);
            out.is_agg.push(false);
        }
    }
    Some(out)
}

/// Swap-modality negatives: keep (gene, cell) but substitute the
/// satellite axis `(modality, region)` with a different one. Forces the
/// model to keep a gene's satellite at `(m, r)` distinguishable from the
/// same gene's base/other-modality embeddings — without it, δ/γ could
/// collapse and every satellite would coincide with β_g.
///
/// The candidate axes are `{(m', r') : m' ∈ modifier modalities,
/// r' ∈ 0..R} \ {(m, r)}`. Drawing from the full grid (rather than only
/// observed components) is intentional: a negative need not correspond
/// to real data, and the grid guarantees a candidate exists whenever
/// there is ≥1 modifier modality and R ≥ 1 (excluding the singleton
/// degenerate case handled below).
fn draw_swap_modality_negatives<R: Rng>(
    state: &SamplerState,
    positives: &PositiveEdges,
    k: usize,
    rng: &mut R,
) -> Option<NegativeEdges> {
    if k == 0 {
        return None;
    }
    let mods = &state.modifier_modality_ids;
    let n_regions = state.n_regions.max(1) as u32;
    // Total satellite axes in the grid. Need ≥ 2 so at least one differs
    // from any positive's (m, r).
    let n_axes = mods.len() as u32 * n_regions;
    if n_axes < 2 {
        return None;
    }

    // Region picker built once for the whole sub-batch (n_regions is
    // constant), not per swap draw.
    let region_dist = Uniform::new(0, n_regions).ok()?;

    let b = positives.len();
    let mut out = NegativeEdges::with_capacity(b * k, k);
    for i in 0..b {
        let g = positives.gene_for_rho[i];
        let m = positives.modality_for_q[i];
        let r = positives.region_for_delta[i];
        for _ in 0..k {
            let (m_prime, r_prime) = swap_pick_axis(mods, n_regions, &region_dist, m, r, rng)?;
            out.gene_for_rho.push(g);
            out.gene_for_z.push(g); // keep the gene's own program loading
            out.modality_for_q.push(m_prime);
            out.region_for_delta.push(r_prime);
            // Bias keyed by (gene, swapped modality) — matches the
            // satellite's own b_comp column so the negative isn't trivially
            // separable on bias alone.
            out.gene_for_bias.push(g);
            out.modality_for_bias.push(m_prime);
            out.is_agg.push(false);
        }
    }
    Some(out)
}

/// Swap-modality negatives for the **count** stratum: keep (gene, cell)
/// and the gene's own program loading, but swap the splice modality to a
/// *different* count satellite (spliced↔unspliced). Region is the
/// count sentinel 0. This is the contrast that pushes the two splice
/// directions apart (and base↔splice for AGG positives, whose
/// `pos_modality = 0` is never itself a count modality). Returns `None`
/// when there are fewer than two count modalities (no contrast possible),
/// so the loss path skips it.
fn draw_count_swap_modality_negatives<R: Rng>(
    state: &SamplerState,
    positives: &PositiveEdges,
    k: usize,
    rng: &mut R,
) -> Option<NegativeEdges> {
    if k == 0 {
        return None;
    }
    let mods = &state.count_modality_ids;
    if mods.len() < 2 {
        return None;
    }
    let b = positives.len();
    let mut out = NegativeEdges::with_capacity(b * k, k);
    for i in 0..b {
        let g = positives.gene_for_rho[i];
        let pos_m = positives.modality_for_q[i];
        for _ in 0..k {
            // pos_m is either a count modality (count positive → ≥1 other
            // remains) or 0 (AGG positive → every count modality qualifies),
            // so with ≥2 count modalities this never fails.
            let m_prime = swap_pick_distinct(mods, pos_m, rng)?;
            out.gene_for_rho.push(g);
            out.gene_for_z.push(g); // keep the gene's own program loading
            out.modality_for_q.push(m_prime);
            out.region_for_delta.push(0); // count satellites are region-0
            out.gene_for_bias.push(g);
            // Shared count bias slot (same for spliced/unspliced) → the
            // contrast must be carried by the δ direction, not the bias.
            out.modality_for_bias.push(COUNT_BIAS_MODALITY);
            out.is_agg.push(false);
        }
    }
    Some(out)
}

/// Pick a satellite axis `(m', r')` distinct from `(m, r)` from the
/// `modalities × 0..n_regions` grid. Rejection-samples, then falls back
/// to a deterministic scan for the rare all-collision case. `region_dist`
/// is built once by the caller (`0..n_regions`) and reused across draws.
fn swap_pick_axis<R: Rng>(
    modalities: &[u32],
    n_regions: u32,
    region_dist: &Uniform<u32>,
    exclude_m: u32,
    exclude_r: u32,
    rng: &mut R,
) -> Option<(u32, u32)> {
    for _ in 0..REJECT_RETRIES {
        let m = *modalities.choose(rng)?;
        let r = region_dist.sample(rng);
        if (m, r) != (exclude_m, exclude_r) {
            return Some((m, r));
        }
    }
    for &m in modalities {
        for r in 0..n_regions {
            if m != exclude_m || r != exclude_r {
                return Some((m, r));
            }
        }
    }
    None
}

/// Uniform-with-rejection swap target picker used by swap-gene-mode.
/// After `REJECT_RETRIES` RNG draws, falls back to a deterministic
/// linear scan so we never silently degenerate to
/// `target == exclude`. Returns `None` only when the haystack has no
/// element distinct from `exclude`.
fn swap_pick_distinct<R: Rng>(haystack: &[u32], exclude: u32, rng: &mut R) -> Option<u32> {
    for _ in 0..REJECT_RETRIES {
        if let Some(&v) = haystack.choose(rng) {
            if v != exclude {
                return Some(v);
            }
        } else {
            return None; // empty haystack
        }
    }
    haystack.iter().copied().find(|&v| v != exclude)
}
