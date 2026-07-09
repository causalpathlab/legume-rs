//! Cell-feature (bipartite) NCE samplers and losses.
//!
//! Used by `senna gbe` and the chain trainer. The cell side is one of
//! pseudobulks, fine cells, or a per-batch stratification; the feature
//! side is always fine. All samplers produce a uniform [`EdgeBatch`]
//! shape so the downstream NCE loss is sampler-agnostic.

use crate::data::Triplet;
use crate::feature_network::{select_feat_emb, FeatureNetworkSmoother};
use crate::loss::logistic_nce;
use crate::model::JointEmbedModel;
use crate::progress::new_progress_bar;
use candle_util::candle_core::{Device, Result, Tensor};
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use indicatif::ParallelProgressIterator;
use rand::Rng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub struct EdgeBatch {
    pub coarse_cells: Vec<u32>,
    pub fine_feats: Vec<u32>,
    /// `[B*K]` row-major: negatives for positive `b` are at `[b*K..(b+1)*K]`.
    pub neg_feats: Vec<u32>,
    pub edge_weights: Vec<f32>,
    pub n_negatives: usize,
}

pub struct PerBatchSampler {
    pub pos: WeightedIndex<f32>,
    pub neg: WeightedIndex<f32>,
    /// Indices into the global `triplets` slice for this batch's positives.
    pub triplet_indices: Vec<u32>,
    /// Global feature ids that constitute this batch's negative pool
    /// (features observed in any cell of this batch).
    pub feature_pool: Vec<u32>,
}

pub struct EdgeBatchArgs<'a> {
    pub triplets: &'a [Triplet],
    pub batch_sampler: &'a PerBatchSampler,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub fine_feature_weights: Option<&'a [f32]>,
    pub batch_size: usize,
    pub n_negatives: usize,
}

pub fn sample_edge_batch(args: EdgeBatchArgs, rng: &mut impl Rng) -> EdgeBatch {
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    let sampler = args.batch_sampler;

    for _ in 0..args.batch_size {
        let local_idx = sampler.pos.sample(rng);
        let global_idx = sampler.triplet_indices[local_idx] as usize;
        let t = &args.triplets[global_idx];
        let c_coarse = args.cell_coarsening.fine_to_coarse[t.cell as usize] as u32;
        coarse_cells.push(c_coarse);
        fine_feats.push(t.feature);
        let w = args
            .fine_feature_weights
            .map_or(1.0, |w| w[t.feature as usize]);
        weights.push(w);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = sampler.neg.sample(rng);
        neg_feats.push(sampler.feature_pool[local]);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

///////////////////////////////////////////////////
// Per-batch stratified cell sampler (cell axis) //
///////////////////////////////////////////////////

/// Two-stage per-batch sampler for the cell axis. Stage 1 picks a cell
/// (within this batch) with `q(c) ∝ degree(c)^alpha_cell`; stage 2
/// picks a feature within that cell weighted by `count · fisher(f)`.
/// Mirrors [`StratifiedSampler`] but the outer stratum is fine cells
/// (not pseudobulks), and one sampler exists per batch. Negatives use
/// the same per-batch feature marginal (`count^alpha_neg`) as
/// [`PerBatchSampler`]. With `alpha_cell = 1`, this is approximately
/// equivalent to the flat sampler; with `alpha_cell = 0`, every cell
/// in the batch gets uniform coverage regardless of sequencing depth.
#[derive(Clone)]
pub struct PerBatchStratifiedCellSampler {
    /// Local-index picker into `active_cells`. Weights = `q(c)`.
    pub cell_picker: WeightedIndex<f32>,
    /// Global cell ids with ≥ 1 expressed feature in this batch, in
    /// stable order.
    pub active_cells: Vec<u32>,
    /// Per-active-cell feature sampler; aligned with `active_cells`.
    pub per_cell: Vec<CellFeatureSampler>,
    /// Negative pool: features with any nonzero count in this batch.
    pub neg: WeightedIndex<f32>,
    pub feature_pool: Vec<u32>,
}

#[derive(Clone)]
pub struct CellFeatureSampler {
    /// Global feature ids expressed in this cell.
    pub features: Vec<u32>,
    /// Raw counts aligned with `features` (the `count` per `(cell, feature)`
    /// edge, before any fisher/weight). Used by the analytical phase-2
    /// projection ([`crate::cell_projection`]); the sampler itself draws via
    /// `picker`.
    pub counts: Vec<f32>,
    /// `WeightedIndex` over `features`; weights = `count · fisher(f)`.
    pub picker: WeightedIndex<f32>,
}

/// Build one stratified cell sampler per batch. Returns one entry per
/// original batch id (`None` for batches with zero positives). Caller
/// filters to the active subset before wiring into `AxisSampler`.
// Deliberately takes primitives (not `&UnifiedData`) to keep the loss
// module decoupled from the data layer; the input count is inherent.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_per_batch_stratified_cell_samplers(
    triplets: &[Triplet],
    batch_membership: &[u32],
    n_batches: usize,
    n_features: usize,
    fisher_weights: &[f32],
    alpha_cell: f32,
    alpha_neg: f32,
    cell_weight_mult: Option<&[f32]>,
) -> Vec<Option<PerBatchStratifiedCellSampler>> {
    let bucket_bar = new_progress_bar(triplets.len() as u64);
    bucket_bar.set_message("bucketing triplets by batch (strat-cell)");
    let per_batch_indices: Vec<Vec<u32>> = triplets
        .par_iter()
        .enumerate()
        .progress_with(bucket_bar.clone())
        .fold(
            || vec![Vec::new(); n_batches],
            |mut acc, (i, t)| {
                let b = batch_membership[t.cell as usize] as usize;
                acc[b].push(i as u32);
                acc
            },
        )
        .reduce(
            || vec![Vec::new(); n_batches],
            |mut a, b| {
                for (av, bv) in a.iter_mut().zip(b.into_iter()) {
                    av.extend(bv);
                }
                a
            },
        );
    bucket_bar.finish_and_clear();

    let build_bar = new_progress_bar(n_batches as u64);
    build_bar.set_message("per-batch strat-cell sampler build");
    let samplers = per_batch_indices
        .into_par_iter()
        .progress_with(build_bar.clone())
        .map(|trip_indices| {
            if trip_indices.is_empty() {
                return None;
            }

            // Bucket triplets in this batch by cell. FxHashMap keyed on
            // global cell id; values are (feature, count) lists. Dense
            // n_cells scratch is wasteful here because each batch only
            // touches a fraction of the global cell axis.
            let mut per_cell_map: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
            let mut cell_degree: FxHashMap<u32, f32> = FxHashMap::default();
            let mut feat_count = vec![0f32; n_features];
            for &i in &trip_indices {
                let t = &triplets[i as usize];
                per_cell_map
                    .entry(t.cell)
                    .or_default()
                    .push((t.feature, t.count));
                *cell_degree.entry(t.cell).or_insert(0.0) += t.count;
                feat_count[t.feature as usize] += t.count;
            }

            let mut active_cells: Vec<u32> = per_cell_map.keys().copied().collect();
            active_cells.sort_unstable();

            let mut per_cell: Vec<CellFeatureSampler> = Vec::with_capacity(active_cells.len());
            let mut cell_w: Vec<f32> = Vec::with_capacity(active_cells.len());
            for &c in &active_cells {
                let edges = &per_cell_map[&c];
                let features: Vec<u32> = edges.iter().map(|&(f, _)| f).collect();
                let counts: Vec<f32> = edges.iter().map(|&(_, cnt)| cnt).collect();
                let weights: Vec<f32> = edges
                    .iter()
                    .map(|&(f, cnt)| (cnt * fisher_weights[f as usize]).max(1e-8))
                    .collect();
                let picker = WeightedIndex::new(weights).expect("non-empty cell-feature weights");
                per_cell.push(CellFeatureSampler {
                    features,
                    counts,
                    picker,
                });
                let mult = cell_weight_mult.map_or(1.0, |m| m[c as usize]);
                cell_w.push(cell_degree[&c].max(1e-8).powf(alpha_cell) * mult);
            }
            let cell_picker = WeightedIndex::new(cell_w).expect("non-empty cell weights");

            let feature_pool: Vec<u32> = (0..n_features as u32)
                .filter(|&f| feat_count[f as usize] > 0.0)
                .collect();
            let neg_w: Vec<f32> = feature_pool
                .iter()
                .map(|&f| feat_count[f as usize].powf(alpha_neg))
                .collect();
            let neg = WeightedIndex::new(neg_w).expect("non-empty batch feature pool");

            Some(PerBatchStratifiedCellSampler {
                cell_picker,
                active_cells,
                per_cell,
                neg,
                feature_pool,
            })
        })
        .collect();
    build_bar.finish_and_clear();
    samplers
}

pub struct PerBatchStratifiedEdgeBatchArgs<'a> {
    pub sampler: &'a PerBatchStratifiedCellSampler,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub fine_feature_weights: &'a [f32],
    pub batch_size: usize,
    pub n_negatives: usize,
}

/// Two-stage draw: pick cell by `degree^alpha_cell`, then feature
/// within cell by `count · fisher`. Output `EdgeBatch` shape matches
/// the flat and stratified pb samplers — downstream NCE doesn't care.
pub fn sample_per_batch_stratified_edge_batch(
    args: PerBatchStratifiedEdgeBatchArgs,
    rng: &mut impl Rng,
) -> EdgeBatch {
    let s = args.sampler;
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let lc = s.cell_picker.sample(rng);
        let c = s.active_cells[lc];
        let pf = &s.per_cell[lc];
        let lf = pf.picker.sample(rng);
        let f = pf.features[lf];
        let c_coarse = args.cell_coarsening.fine_to_coarse[c as usize] as u32;
        coarse_cells.push(c_coarse);
        fine_feats.push(f);
        weights.push(args.fine_feature_weights[f as usize]);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = s.neg.sample(rng);
        neg_feats.push(s.feature_pool[local]);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

/////////////////////////////////
// Stratified positive sampler //
/////////////////////////////////

/// Two-stage stratified sampler for pseudobulk axes. Stage 1 picks a
/// pb (stratum) by `q(p) ∝ pb_size(p)^alpha_pb`; stage 2 picks a
/// feature within that pb weighted by `μ_pf · fisher(f)`. Compared to
/// flat `WeightedIndex` over all super-edges, this guarantees every pb
/// gets training coverage proportional to `q(p)` (uniform when
/// `alpha_pb = 0`, count-proportional when `alpha_pb = 1`), instead of
/// being dominated by housekeeping-gene super-edges.
///
/// Negatives come from a single global pb-level feature marginal
/// (count^`alpha_neg`), since `pb_unified` collapses all pseudobulks
/// into one synthetic "all" batch.
pub struct StratifiedSampler {
    /// Picks a local pb index into `active_pbs`. Weights = `q(p)`.
    pub pb_picker: WeightedIndex<f32>,
    /// Global pb ids that have ≥ 1 expressed feature, in stable order.
    pub active_pbs: Vec<u32>,
    /// Per-active-pb feature sampler; aligned with `active_pbs`.
    pub per_pb: Vec<PbFeatureSampler>,
    /// Negative pool: features with any nonzero pb-level count.
    pub neg: WeightedIndex<f32>,
    pub feature_pool: Vec<u32>,
}

pub struct PbFeatureSampler {
    /// Global feature ids sampled within this pb. In per-row mode this is every
    /// expressed row; in β-sharing gene-paired mode ([`FeatPairing`]) it is one
    /// entry per gene — the **spliced** (identity) row.
    pub features: Vec<u32>,
    /// Gene-paired mode only (else empty): the **unspliced** row paired with each
    /// `features` entry (`u32::MAX` = the gene has no nascent track in this pb). A
    /// draw emits both `features[i]` and `paired[i]` so δ_g trains at the spliced
    /// sampling frequency.
    pub paired: Vec<u32>,
    /// `WeightedIndex` over `features`; per-row weights = `μ_pf · fisher(f)`,
    /// gene-paired weights = `spliced_count · fisher(spliced_row)`.
    pub picker: WeightedIndex<f32>,
}

/// β-sharing pairing for **spliced-driven** positive sampling (gem's feat_factor
/// path). Each gene is sampled by its SPLICED-track count, and a draw emits both
/// the spliced row (identity, scored vs `β_g`) and its paired unspliced row
/// (nascent, scored vs `β_g + δ_g`). `None` at build time → plain per-row sampling
/// (bge). Row-indexed, aligned to the unified feature axis.
pub struct FeatPairing<'a> {
    pub row_to_gene: &'a [u32],
    pub unspliced_rows: &'a [bool],
}

/// Group one stratum's `(row, count)` edges by gene for [`FeatPairing`]: returns
/// `(primary_rows, paired_rows, weights)`, one entry per gene present. The gene is
/// sampled by its **total** count (`spliced + unspliced` — the nascent fraction
/// up-weights actively-regulated genes and no gene is dropped for lacking a mature
/// track), `weight = total_count · fisher(primary)`. `primary` is the spliced
/// (identity) row when present (else the nascent row for a nascent-only gene);
/// `paired` is the unspliced row when both tracks exist (`u32::MAX` otherwise). A
/// draw emits both rows so β_g / δ_g train at the gene's sampling frequency.
fn gene_paired_entries(
    edges: &[(u32, f32)],
    fp: &FeatPairing,
    fisher_weights: &[f32],
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    // gene → (row, summed count) for each track.
    let mut spliced: FxHashMap<u32, (u32, f32)> = FxHashMap::default();
    let mut unspliced: FxHashMap<u32, (u32, f32)> = FxHashMap::default();
    for &(row, cnt) in edges {
        let g = fp.row_to_gene[row as usize];
        let table = if fp.unspliced_rows[row as usize] {
            &mut unspliced
        } else {
            &mut spliced
        };
        let e = table.entry(g).or_insert((row, 0.0));
        e.0 = row;
        e.1 += cnt;
    }
    let mut genes: Vec<u32> = spliced.keys().chain(unspliced.keys()).copied().collect();
    genes.sort_unstable();
    genes.dedup();
    let mut features = Vec::with_capacity(genes.len());
    let mut paired = Vec::with_capacity(genes.len());
    let mut weights = Vec::with_capacity(genes.len());
    for g in genes {
        let s = spliced.get(&g);
        let u = unspliced.get(&g);
        let total = s.map_or(0.0, |&(_, c)| c) + u.map_or(0.0, |&(_, c)| c);
        // Primary = spliced (identity) row when present, else the nascent row.
        let (primary, pair) = match (s, u) {
            (Some(&(srow, _)), Some(&(urow, _))) => (srow, urow),
            (Some(&(srow, _)), None) => (srow, u32::MAX),
            (None, Some(&(urow, _))) => (urow, u32::MAX),
            (None, None) => unreachable!("gene collected from a nonempty track"),
        };
        features.push(primary);
        paired.push(pair);
        weights.push((total * fisher_weights[primary as usize]).max(1e-8));
    }
    (features, paired, weights)
}

/// Build a stratified sampler for a pseudobulk axis. Returns `None`
/// when the axis has zero positives or fewer than two active pb's
/// (degenerate stratum).
#[must_use]
pub fn build_stratified_sampler(
    triplets: &[Triplet],
    n_pb: usize,
    n_features: usize,
    fisher_weights: &[f32],
    alpha_pb: f32,
    alpha_neg: f32,
    pairing: Option<&FeatPairing>,
) -> Option<StratifiedSampler> {
    if triplets.is_empty() {
        return None;
    }

    // Parallel bucket triplets by pb; per-thread local accumulators
    // (per_pb / pb_size / feat_count) then reduce. Per-pb edge lists
    // concat across threads; pb_size and feat_count sum elementwise.
    let bucket_bar = new_progress_bar(triplets.len() as u64);
    bucket_bar.set_message("bucketing triplets by pb");
    struct Bucket {
        per_pb: Vec<Vec<(u32, f32)>>,
        pb_size: Vec<f32>,
        feat_count: Vec<f32>,
    }
    let Bucket {
        per_pb,
        pb_size,
        feat_count,
    } = triplets
        .par_iter()
        .progress_with(bucket_bar.clone())
        .fold(
            || Bucket {
                per_pb: vec![Vec::new(); n_pb],
                pb_size: vec![0f32; n_pb],
                feat_count: vec![0f32; n_features],
            },
            |mut acc, t| {
                acc.per_pb[t.cell as usize].push((t.feature, t.count));
                acc.pb_size[t.cell as usize] += t.count;
                acc.feat_count[t.feature as usize] += t.count;
                acc
            },
        )
        .reduce(
            || Bucket {
                per_pb: vec![Vec::new(); n_pb],
                pb_size: vec![0f32; n_pb],
                feat_count: vec![0f32; n_features],
            },
            |mut a, b| {
                for (av, bv) in a.per_pb.iter_mut().zip(b.per_pb.into_iter()) {
                    av.extend(bv);
                }
                for (av, bv) in a.pb_size.iter_mut().zip(b.pb_size.into_iter()) {
                    *av += bv;
                }
                for (av, bv) in a.feat_count.iter_mut().zip(b.feat_count.into_iter()) {
                    *av += bv;
                }
                a
            },
        );
    bucket_bar.finish_and_clear();

    // Per-pb sampler build is embarrassingly parallel.
    let active_idx: Vec<usize> = (0..n_pb).filter(|&p| !per_pb[p].is_empty()).collect();
    if active_idx.is_empty() {
        return None;
    }
    let build_bar = new_progress_bar(active_idx.len() as u64);
    build_bar.set_message("per-pb sampler build");
    let built: Vec<(u32, PbFeatureSampler, f32)> = active_idx
        .par_iter()
        .progress_with(build_bar.clone())
        .map(|&p| {
            let edges = &per_pb[p];
            let (features, paired, weights) = match pairing {
                // β-sharing: one entry per gene, sampled by spliced count, paired
                // with its unspliced row (emitted together at draw time).
                Some(fp) => gene_paired_entries(edges, fp, fisher_weights),
                // Plain per-row: every expressed row sampled by count · fisher.
                None => {
                    let features: Vec<u32> = edges.iter().map(|&(f, _)| f).collect();
                    let weights: Vec<f32> = edges
                        .iter()
                        .map(|&(f, c)| (c * fisher_weights[f as usize]).max(1e-8))
                        .collect();
                    (features, Vec::new(), weights)
                }
            };
            let picker = WeightedIndex::new(weights).expect("non-empty pb feature weights");
            (
                p as u32,
                PbFeatureSampler {
                    features,
                    paired,
                    picker,
                },
                pb_size[p].max(1e-8).powf(alpha_pb),
            )
        })
        .collect();
    build_bar.finish_and_clear();

    let mut active_pbs: Vec<u32> = Vec::with_capacity(built.len());
    let mut per_pb_samplers: Vec<PbFeatureSampler> = Vec::with_capacity(built.len());
    let mut pb_q: Vec<f32> = Vec::with_capacity(built.len());
    for (p, s, q) in built {
        active_pbs.push(p);
        per_pb_samplers.push(s);
        pb_q.push(q);
    }
    let pb_picker = WeightedIndex::new(pb_q).expect("non-empty pb weights");

    // Negative-sampling basis per feature row. Per-row (bge): the row's own count.
    // Gene-paired (gem): the row's GENE total (spliced + unspliced) count · fisher — the
    // SAME per-gene weighting as the positive draw, so the NCE noise distribution matches
    // the data distribution (fisher on negatives too, not just positives). The pos/neg
    // asymmetry (fisher on positives only) otherwise biases the learned scores toward the
    // abundant genes.
    let neg_basis: Vec<f32> = match pairing {
        Some(fp) => {
            let mut gene_total = vec![0f32; n_features];
            for f in 0..n_features {
                gene_total[fp.row_to_gene[f] as usize] += feat_count[f];
            }
            (0..n_features)
                .map(|f| gene_total[fp.row_to_gene[f] as usize] * fisher_weights[f])
                .collect()
        }
        None => feat_count.clone(),
    };
    let feature_pool: Vec<u32> = (0..n_features as u32)
        .filter(|&f| feat_count[f as usize] > 0.0)
        .collect();
    if feature_pool.is_empty() {
        return None;
    }
    let neg_w: Vec<f32> = feature_pool
        .iter()
        .map(|&f| neg_basis[f as usize].powf(alpha_neg))
        .collect();
    let neg = WeightedIndex::new(neg_w).expect("non-empty negative pool");

    Some(StratifiedSampler {
        pb_picker,
        active_pbs,
        per_pb: per_pb_samplers,
        neg,
        feature_pool,
    })
}

pub struct StratifiedEdgeBatchArgs<'a> {
    pub sampler: &'a StratifiedSampler,
    pub fine_feature_weights: &'a [f32],
    pub batch_size: usize,
    pub n_negatives: usize,
}

/// Two-stage draw: pick pb by `q(p)`, then feature within pb. Output
/// `EdgeBatch` is interchangeable with [`sample_edge_batch`]'s — the
/// downstream NCE loss doesn't care how the batch was sampled.
pub fn sample_stratified_edge_batch(
    args: StratifiedEdgeBatchArgs,
    rng: &mut impl Rng,
) -> EdgeBatch {
    let s = args.sampler;
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let local_pb = s.pb_picker.sample(rng);
        let p = s.active_pbs[local_pb];
        let pf = &s.per_pb[local_pb];
        let local_f = pf.picker.sample(rng);
        let f = pf.features[local_f];
        coarse_cells.push(p);
        fine_feats.push(f);
        weights.push(args.fine_feature_weights[f as usize]);
        // β-sharing gene-paired draw: the entry above is the spliced (identity)
        // row, drawn by its spliced count. Emit the paired unspliced row into the
        // same batch (same pb) so δ_g trains at the spliced sampling frequency.
        if !pf.paired.is_empty() {
            let u = pf.paired[local_f];
            if u != u32::MAX {
                coarse_cells.push(p);
                fine_feats.push(u);
                weights.push(args.fine_feature_weights[u as usize]);
            }
        }
    }

    // Negatives scale with the actual positive count (gene-paired draws emit up to
    // 2× batch_size positives); the loss reads `neg_feats[b*K..(b+1)*K]` per positive.
    let n_pos = fine_feats.len();
    let mut neg_feats = Vec::with_capacity(n_pos * args.n_negatives);
    for _ in 0..(n_pos * args.n_negatives) {
        let local = s.neg.sample(rng);
        neg_feats.push(s.feature_pool[local]);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        edge_weights: weights,
        n_negatives: args.n_negatives,
    }
}

/////////////////////////////////////////////
// Nested chain sampler (cell-feature MVP) //
/////////////////////////////////////////////

/// Bottom-up nested chain sampler. Each chain starts from a real
/// `(cell, feature)` super-edge drawn by the existing per-batch
/// sampler, then walks **up** the pb-tree via stored parent maps to
/// derive a coordinated `pb_path` across every coarser level. The
/// resulting chain produces:
///
/// - one positive `(leaf_cell, leaf_feature)` for the cell axis;
/// - one positive `(P_k, leaf_feature)` for each pb level k via
///   `P_k = ancestor of leaf_cell at level k`;
/// - shared feature negatives drawn from the cell axis's batch pool.
///
/// All axes therefore share the same positive feature (and negatives)
/// per chain. The chain's per-axis cell-side index is the cell itself
/// (cell axis) or the cell's pb at that level (pb axes). One backward
/// step over the summed loss updates `E_feat` once with coherent
/// gradient contributions from every level — that's the variance-
/// reduction win versus independently sampling each axis.
///
/// MVP scope: single-level supergene (no gene-tree yet); pb-tree only.
/// Hard negatives via siblings are a future extension.
pub struct ChainSampler<'a> {
    pub batch_sampler: &'a PerBatchSampler,
    /// `cell_to_pb_per_level[k][cell] = pb id at level k`. Finest-first
    /// (matches the gbe convention after `collapsed_levels.reverse()`),
    /// so `pb_path[level=0]` is coarsest and `pb_path.last()` is finest.
    pub cell_to_pb_per_level: &'a [Vec<usize>],
}

pub struct ChainBatch {
    /// Length B; one leaf cell per chain. Pb-tree ancestors at each
    /// level are derived by indexing `cell_to_pb_per_level[level][cell]`
    /// at the call site — not stored on the batch to avoid the per-step
    /// `Vec<Vec<u32>>` allocation.
    pub leaf_cells: Vec<u32>,
    /// Shared feature side: one positive + K negatives per chain.
    /// Lifted into its own struct so [`nce_loss_chain`] can take it by
    /// value while the caller still borrows `leaf_cells` for the
    /// cell-axis [`ChainAxis`] entry.
    pub feats: ChainFeatureSide,
}

pub struct ChainFeatureSide {
    /// Length B; one positive feature per chain (shared across all axes).
    pub fine_feats: Vec<u32>,
    /// Length B*K row-major; shared negatives across all axes.
    pub neg_feats: Vec<u32>,
    pub edge_weights: Vec<f32>,
    pub n_negatives: usize,
}

pub struct ChainBatchArgs<'a> {
    pub triplets: &'a [Triplet],
    pub sampler: &'a ChainSampler<'a>,
    pub fine_feature_weights: &'a [f32],
    pub batch_size: usize,
    pub n_negatives: usize,
}

pub fn sample_chain_batch(args: ChainBatchArgs, rng: &mut impl Rng) -> ChainBatch {
    let bs = args.sampler.batch_sampler;
    let mut leaf_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);
    let mut weights = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let local_idx = bs.pos.sample(rng);
        let global_idx = bs.triplet_indices[local_idx] as usize;
        let t = &args.triplets[global_idx];
        leaf_cells.push(t.cell);
        fine_feats.push(t.feature);
        weights.push(args.fine_feature_weights[t.feature as usize]);
    }

    let mut neg_feats = Vec::with_capacity(args.batch_size * args.n_negatives);
    for _ in 0..(args.batch_size * args.n_negatives) {
        let local = bs.neg.sample(rng);
        neg_feats.push(bs.feature_pool[local]);
    }

    ChainBatch {
        leaf_cells,
        feats: ChainFeatureSide {
            fine_feats,
            neg_feats,
            edge_weights: weights,
            n_negatives: args.n_negatives,
        },
    }
}

/// One axis's cell-side resolution for chain scoring. The chain loss
/// gathers `e_cell.index_select(&indices)` and scores against the
/// shared (already-gathered) feature side. `indices` is a pre-built
/// `[B] u32` index tensor — building it in the caller (once per axis
/// per step) avoids a second `Vec<u32> → Tensor` round-trip inside the
/// loss.
pub struct ChainAxis<'a> {
    pub e_cell: &'a Tensor,
    pub b_cell: &'a Tensor,
    pub indices: &'a Tensor,
    pub lambda: f32,
    /// Used in `nce_loss_chain`'s error diagnostics.
    pub label: &'a str,
}

/// Score one chain batch across every axis using a single shared
/// positive/negative feature gather. `e_feat` / `b_feat` are
/// **shared** across axes (same Var); each axis differs only in its
/// cell-side embeddings table and the per-chain indices into it.
///
/// Returns the λ-weighted sum across axes. The per-edge weight uses
/// the cell-axis NCE's `Σw` normalization (Fisher-info-aware) so the
/// gradient magnitude is consistent with the existing per-axis losses.
pub fn nce_loss_chain(
    e_feat: &Tensor,
    b_feat: &Tensor,
    feats: ChainFeatureSide,
    axes: &[ChainAxis],
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let b = feats.fine_feats.len();
    if b == 0 || axes.is_empty() {
        return Ok(Tensor::zeros(
            (),
            candle_util::candle_core::DType::F32,
            dev,
        )?);
    }
    let k = feats.n_negatives;

    // Feature side gathered once for the whole chain step.
    let pos_feat_idx = Tensor::from_slice(&feats.fine_feats, b, dev)?;
    let e_feat_pos = select_feat_emb(smoother, e_feat, &pos_feat_idx)?;
    let b_feat_pos = b_feat.index_select(&pos_feat_idx, 0)?;

    let neg_feat_idx = Tensor::from_slice(&feats.neg_feats, b * k, dev)?;
    let e_feat_neg_flat = select_feat_emb(smoother, e_feat, &neg_feat_idx)?;
    let b_feat_neg_flat = b_feat.index_select(&neg_feat_idx, 0)?;
    let h = e_feat_neg_flat.dim(1)?;
    let e_feat_neg = e_feat_neg_flat.reshape((b, k, h))?;
    let b_feat_neg = b_feat_neg_flat.reshape((b, k))?;

    let w_sum: f32 = feats.edge_weights.iter().sum::<f32>().max(1e-8);
    let w_t = Tensor::from_vec(feats.edge_weights, b, dev)?;

    let mut total: Option<Tensor> = None;
    for axis in axes {
        let n = axis.indices.dims1()?;
        anyhow::ensure!(
            n == b,
            "chain axis {} indices ({n}) != batch_size ({b})",
            axis.label,
        );
        let e_cell_pos = axis.e_cell.index_select(axis.indices, 0)?;
        let b_cell_pos = axis.b_cell.index_select(axis.indices, 0)?;

        let pos_score =
            JointEmbedModel::score_diag(&e_feat_pos, &e_cell_pos, &b_feat_pos, &b_cell_pos)?;
        let neg_score =
            JointEmbedModel::score_negatives(&e_feat_neg, &e_cell_pos, &b_feat_neg, &b_cell_pos)?;
        let per_edge = logistic_nce(&pos_score, std::slice::from_ref(&neg_score))?;
        let weighted = (per_edge * w_t.clone())?;
        let axis_loss = (weighted.sum(0)? / f64::from(w_sum))?;
        let scaled = (axis_loss * f64::from(axis.lambda))?;
        total = Some(match total {
            Some(prev) => (prev + scaled)?,
            None => scaled,
        });
    }
    Ok(total.unwrap())
}

///////////////////////////////////////////////////////
// Standard per-batch sampler + bipartite NCE losses //
///////////////////////////////////////////////////////

/// Build a per-batch sampler. Each batch contributes the positive triplets
/// whose cells belong to that batch, and a negative pool restricted to the
/// features observed in those cells. Batches with zero observed edges are
/// returned as `None` (caller filters).
#[must_use]
pub fn build_per_batch_samplers(
    triplets: &[Triplet],
    batch_membership: &[u32],
    n_batches: usize,
    n_features: usize,
    fisher_weights: &[f32],
    alpha_neg: f32,
) -> Vec<Option<PerBatchSampler>> {
    // Parallel bucketing of triplet indices by batch. Each worker folds
    // its chunk into per-batch local Vecs, then reduce concats per batch.
    let bucket_bar = new_progress_bar(triplets.len() as u64);
    bucket_bar.set_message("bucketing triplets by batch");
    let per_batch_indices: Vec<Vec<u32>> = triplets
        .par_iter()
        .enumerate()
        .progress_with(bucket_bar.clone())
        .fold(
            || vec![Vec::new(); n_batches],
            |mut acc, (i, t)| {
                let b = batch_membership[t.cell as usize] as usize;
                acc[b].push(i as u32);
                acc
            },
        )
        .reduce(
            || vec![Vec::new(); n_batches],
            |mut a, b| {
                for (av, bv) in a.iter_mut().zip(b.into_iter()) {
                    av.extend(bv);
                }
                a
            },
        );
    bucket_bar.finish_and_clear();

    let build_bar = new_progress_bar(n_batches as u64);
    build_bar.set_message("per-batch sampler build");
    let samplers = per_batch_indices
        .into_par_iter()
        .progress_with(build_bar.clone())
        .map(|trip_indices| {
            if trip_indices.is_empty() {
                return None;
            }

            let pos_w: Vec<f32> = trip_indices
                .iter()
                .map(|&i| {
                    let t = &triplets[i as usize];
                    let w = fisher_weights[t.feature as usize];
                    (t.count * w).max(1e-8)
                })
                .collect();
            let pos = WeightedIndex::new(pos_w).expect("non-empty batch positives");

            // Per-batch feature marginal (count-weighted), then α-smoothed.
            // Dense scratch is cheaper than a HashMap at our feature counts.
            let mut feat_count = vec![0f32; n_features];
            for &i in &trip_indices {
                let t = &triplets[i as usize];
                feat_count[t.feature as usize] += t.count;
            }
            let feature_pool: Vec<u32> = (0..n_features as u32)
                .filter(|&f| feat_count[f as usize] > 0.0)
                .collect();
            let neg_w: Vec<f32> = feature_pool
                .iter()
                .map(|&f| feat_count[f as usize].powf(alpha_neg))
                .collect();
            let neg = WeightedIndex::new(neg_w).expect("non-empty batch feature pool");

            Some(PerBatchSampler {
                pos,
                neg,
                triplet_indices: trip_indices,
                feature_pool,
            })
        })
        .collect();
    build_bar.finish_and_clear();
    samplers
}

pub fn nce_loss(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    cell_coarse_to_fine: &[Vec<usize>],
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let (unique_cells, cell_pos_idx) = unique_with_index(&batch.coarse_cells);
    let (e_cell_u, b_cell_u) = model.pool_cells(&unique_cells, cell_coarse_to_fine, dev)?;

    let cell_idx_t = Tensor::from_vec(cell_pos_idx, b, dev)?;
    let e_cell_pos = e_cell_u.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = b_cell_u.index_select(&cell_idx_t, 0)?;

    nce_loss_with_cell_side(model, batch, e_cell_pos, b_cell_pos, smoother, dev)
}

/// Fast path for the identity-coarsening case (every "pb-sample" is
/// its own row). Skips `unique_with_index`, `pool_cells`, and the
/// scatter-add — a single `index_select` directly off `model.e_cell` /
/// `model.b_cell` is mathematically equivalent because each block has
/// exactly one fine child and `mean([x]) == x`. Composite training
/// hits this path on every axis (cell axis + every pseudobulk level
/// use [`crate::coarsen::identity_axis`]).
pub fn nce_loss_identity(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let cell_idx_t = Tensor::from_vec(batch.coarse_cells.clone(), b, dev)?;
    let e_cell_pos = model.e_cell.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = model.b_cell.index_select(&cell_idx_t, 0)?;
    nce_loss_with_cell_side(model, batch, e_cell_pos, b_cell_pos, smoother, dev)
}

/// Shared tail of [`nce_loss`] / [`nce_loss_identity`]: feature-side
/// gathers, raw bilinear (`E_feat[f] · E_cell[c] + b_feat`) scoring,
/// count-weighted log-σ aggregation. The cell-side embeddings come
/// pre-resolved (pooled or directly gathered).
fn nce_loss_with_cell_side(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    e_cell_pos: Tensor,
    b_cell_pos: Tensor,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    let k = batch.n_negatives;

    // Gather the feature rows scored THIS step. A β-sharing factored model
    // composes the row→gene→β gathers, so only the batch's rows (b + b·k) are
    // materialized — never the full [n_features, H] dictionary per step — and β
    // still gets gradients through the gather. A free model selects from the
    // `e_feat` Var (SGC-smoothed when a smoother is present; a factored model
    // never has one — see `fit::run`). Shared by pos + neg.
    let gather_feat = |idx: &Tensor| -> Result<Tensor> {
        match &model.factor {
            Some(f) => {
                let genes = f.row_to_gene.index_select(idx, 0)?;
                let base = f.beta.index_select(&genes, 0)?; // β_g
                match &f.splice_delta {
                    // unspliced rows: + δ_g (β_g + mask ⊙ δ_g); spliced: mask = 0.
                    Some((delta, mask)) => {
                        let d = delta.index_select(&genes, 0)?; // [b, H]
                        let m = mask.index_select(idx, 0)?; // [b, 1]
                        base.add(&d.broadcast_mul(&m)?)
                    }
                    None => Ok(base),
                }
            }
            None => select_feat_emb(smoother, &model.e_feat, idx),
        }
    };

    let pos_feat_idx_t = Tensor::from_slice(&batch.fine_feats, b, dev)?;
    let e_feat_pos = gather_feat(&pos_feat_idx_t)?;
    let b_feat_pos = model.b_feat.index_select(&pos_feat_idx_t, 0)?;

    let neg_feat_idx_t = Tensor::from_slice(&batch.neg_feats, b * k, dev)?;
    let e_feat_neg_flat = gather_feat(&neg_feat_idx_t)?;
    let b_feat_neg_flat = model.b_feat.index_select(&neg_feat_idx_t, 0)?;
    let h = e_feat_neg_flat.dim(1)?;
    let e_feat_neg = e_feat_neg_flat.reshape((b, k, h))?;
    let b_feat_neg = b_feat_neg_flat.reshape((b, k))?;

    // Raw bilinear scoring: score = E_feat[f]·E_cell[c] + b_feat (+ b_cell,
    // dropped by the bge driver). The cell row is shared by the pos and neg
    // scores. (Cosine + learnable temperature was tried — commit 12d758a —
    // but regressed cell-type recovery on real data vs the raw dot, so it
    // was removed.)
    let pos_score =
        JointEmbedModel::score_diag(&e_feat_pos, &e_cell_pos, &b_feat_pos, &b_cell_pos)?;
    let neg_score =
        JointEmbedModel::score_negatives(&e_feat_neg, &e_cell_pos, &b_feat_neg, &b_cell_pos)?;

    let per_edge = logistic_nce(&pos_score, std::slice::from_ref(&neg_score))?;

    // Normalize by Σw, not B: when most positives are housekeeping and
    // get downweighted, dividing by B leaves an O(mean(w)) gradient
    // attenuation that stalls learning.
    let w_sum: f32 = batch.edge_weights.iter().sum::<f32>().max(1e-8);
    let w_t = Tensor::from_vec(batch.edge_weights, b, dev)?;
    let weighted = (per_edge * w_t)?;
    weighted.sum(0)? / f64::from(w_sum)
}

fn unique_with_index(values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut seen: FxHashMap<u32, u32> = FxHashMap::default();
    let mut unique = Vec::new();
    let mut idx_map = Vec::with_capacity(values.len());
    for &v in values {
        let id = *seen.entry(v).or_insert_with(|| {
            let id = unique.len() as u32;
            unique.push(v);
            id
        });
        idx_map.push(id);
    }
    (unique, idx_map)
}

#[cfg(test)]
mod pairing_tests {
    use super::*;
    use rustc_hash::FxHashMap;

    // Rows: 0=gene0 spliced, 1=gene0 unspliced, 2=gene1 spliced (no nascent),
    //       3=gene2 unspliced (NO spliced → nascent-only, now KEPT).
    fn fixture() -> (Vec<u32>, Vec<bool>) {
        let row_to_gene = vec![0u32, 0, 1, 2];
        let unspliced_rows = vec![false, true, false, true];
        (row_to_gene, unspliced_rows)
    }

    #[test]
    fn gene_paired_entries_pairs_and_keeps_nascent_only() {
        let (row_to_gene, unspliced_rows) = fixture();
        let fp = FeatPairing {
            row_to_gene: &row_to_gene,
            unspliced_rows: &unspliced_rows,
        };
        let fisher = vec![1.0f32; 4];
        // pb has all four rows: gene0 spliced 5 + unspliced 2, gene1 spliced 3,
        // gene2 unspliced-only 4.
        let edges = vec![(0u32, 5.0f32), (1, 2.0), (2, 3.0), (3, 4.0)];
        let (features, paired, weights) = gene_paired_entries(&edges, &fp, &fisher);

        // One entry per gene present → gene0, gene1, gene2 (nascent-only kept).
        assert_eq!(features.len(), 3);
        let by_primary: FxHashMap<u32, (u32, f32)> = features
            .iter()
            .zip(paired.iter())
            .zip(weights.iter())
            .map(|((&f, &p), &w)| (f, (p, w)))
            .collect();
        // gene0: primary spliced row 0, paired unspliced row 1, weight = total (5+2).
        assert_eq!(by_primary[&0], (1, 7.0));
        // gene1: primary spliced row 2, no nascent pair, weight = 3.
        assert_eq!(by_primary[&2], (u32::MAX, 3.0));
        // gene2: nascent-only → primary is the unspliced row 3, no pair, weight = 4.
        assert_eq!(by_primary[&3], (u32::MAX, 4.0));
    }

    #[test]
    fn weight_sums_spliced_and_unspliced() {
        let (row_to_gene, unspliced_rows) = fixture();
        let fp = FeatPairing {
            row_to_gene: &row_to_gene,
            unspliced_rows: &unspliced_rows,
        };
        let fisher = vec![2.0f32, 1.0, 1.0, 1.0];
        // gene0 spliced 5 + unspliced 2 → total 7; fisher applied on the primary
        // (spliced) row (2.0) → 14. Confirms the unspliced count IS summed in.
        let edges = vec![(0u32, 5.0f32), (1, 2.0)];
        let (features, paired, weights) = gene_paired_entries(&edges, &fp, &fisher);
        assert_eq!(features, vec![0]);
        assert_eq!(paired, vec![1]);
        assert_eq!(weights, vec![14.0]);
    }
}
