//! Cell-feature (bipartite) NCE samplers and losses.
//!
//! Used by `senna gbe` and the chain trainer. The cell side is one of
//! pseudobulks, fine cells, or a per-batch stratification; the feature
//! side is always fine. All samplers produce a uniform [`EdgeBatch`]
//! shape so the downstream NCE loss is sampler-agnostic.

use crate::data::Triplet;
use crate::loss::{logistic_nce, softmax_nce, NceObjective};
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
    pub n_negatives: usize,
}

///////////////////////////////////////////////////
// Per-batch stratified cell sampler (cell axis) //
///////////////////////////////////////////////////

/// Two-stage per-batch sampler for the cell axis. Stage 1 picks a cell
/// (within this batch) with `q(c) ∝ degree(c)^alpha_cell`; stage 2
/// picks a feature within that cell weighted by `count`.
/// Mirrors [`StratifiedSampler`] but the outer stratum is fine cells
/// (not pseudobulks), and one sampler exists per batch. Negatives are drawn
/// UNIFORMLY over this batch's expressed-feature pool (abundance-independent),
/// as in [`PerBatchSampler`]. With `alpha_cell = 1`, this is approximately
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
    /// edge). Used by the analytical phase-2 projection
    /// ([`crate::cell_projection`]); the sampler itself draws via `picker`.
    pub counts: Vec<f32>,
    /// `WeightedIndex` over `features`; weights = `count`.
    pub picker: WeightedIndex<f32>,
}
pub struct PerBatchStratifiedEdgeBatchArgs<'a> {
    pub sampler: &'a PerBatchStratifiedCellSampler,
    pub cell_coarsening: &'a FeatureCoarsening,
    pub batch_size: usize,
    pub n_negatives: usize,
}

/// Two-stage draw: pick cell by `degree^alpha_cell`, then feature
/// within cell by `count`. Output `EdgeBatch` shape matches the flat
/// and stratified pb samplers — downstream NCE doesn't care.
pub fn sample_per_batch_stratified_edge_batch(
    args: PerBatchStratifiedEdgeBatchArgs,
    rng: &mut impl Rng,
) -> EdgeBatch {
    let s = args.sampler;
    let mut coarse_cells = Vec::with_capacity(args.batch_size);
    let mut fine_feats = Vec::with_capacity(args.batch_size);

    for _ in 0..args.batch_size {
        let lc = s.cell_picker.sample(rng);
        let c = s.active_cells[lc];
        let pf = &s.per_cell[lc];
        let lf = pf.picker.sample(rng);
        let f = pf.features[lf];
        let c_coarse = args.cell_coarsening.fine_to_coarse[c as usize] as u32;
        coarse_cells.push(c_coarse);
        fine_feats.push(f);
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
        n_negatives: args.n_negatives,
    }
}

/////////////////////////////////
// Stratified positive sampler //
/////////////////////////////////

/// Two-stage stratified sampler for pseudobulk axes. Stage 1 picks a
/// pb (stratum) by `q(p) ∝ pb_size(p)^alpha_pb`; stage 2 picks a
/// feature within that pb weighted by `μ_pf`. Compared to
/// flat `WeightedIndex` over all super-edges, this guarantees every pb
/// gets training coverage proportional to `q(p)` (uniform when
/// `alpha_pb = 0`, count-proportional when `alpha_pb = 1`), instead of
/// being dominated by housekeeping-gene super-edges.
///
/// Negatives are drawn UNIFORMLY over the single global pb-level pool of
/// expressed features (abundance-independent), since `pb_unified` collapses
/// all pseudobulks into one synthetic "all" batch.
pub struct StratifiedSampler {
    /// Picks a local pb index into `active_pbs`. Weights = `q(p)`.
    pub pb_picker: WeightedIndex<f32>,
    /// Global pb ids that have ≥ 1 expressed feature, in stable order.
    pub active_pbs: Vec<u32>,
    /// Per-active-pb feature sampler; aligned with `active_pbs`.
    pub per_pb: Vec<PbFeatureSampler>,
    /// Negative pool: features with any nonzero pb-level count.
    ///
    /// Two pickers over the SAME `feature_pool`, mixed 50/50 per draw, which is
    /// what SIMBA actually does: it "produces 100 negatives by corrupting the
    /// edge with a source or destination sampled uniformly from the nodes with
    /// the correct types for this relation and 100 by corrupting the edge with
    /// a source or destination node sampled with probability proportional to
    /// its degree" (inherited from PyTorch-BigGraph). This code previously took
    /// only the uniform half.
    ///
    /// The two are not interchangeable, because NCE learns a log-ratio against
    /// the noise distribution: at the optimum the score is
    /// `log p(f|c) − log q(f)`. Uniform `q` leaves raw abundance, so the
    /// nearest features to every cell are the globally most abundant ones;
    /// degree-proportional `q` divides abundance out entirely. Mixing gives
    /// degree-proportional behaviour at the abundant end and a uniform floor at
    /// the rare end — more abundance correction where housekeeping genes
    /// dominate, less where counts are too thin to estimate a degree.
    pub neg: WeightedIndex<f32>,
    /// The degree-proportional half of the negative distribution.
    pub neg_by_degree: WeightedIndex<f32>,
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
    /// `WeightedIndex` over `features`; per-row weights = `μ_pf`,
    /// gene-paired weights = the gene's `total` count.
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
/// track), `weight = total_count`. `primary` is the spliced
/// (identity) row when present (else the nascent row for a nascent-only gene);
/// `paired` is the unspliced row when both tracks exist (`u32::MAX` otherwise). A
/// draw emits both rows so β_g / δ_g train at the gene's sampling frequency.
fn gene_paired_entries(edges: &[(u32, f32)], fp: &FeatPairing) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
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
        weights.push(total.max(1e-8));
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
    alpha_pb: f32,
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
                Some(fp) => gene_paired_entries(edges, fp),
                // Plain per-row: every expressed row sampled by count.
                None => {
                    let features: Vec<u32> = edges.iter().map(|&(f, _)| f).collect();
                    let weights: Vec<f32> = edges.iter().map(|&(_, c)| c.max(1e-8)).collect();
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

    // Uniform negatives: every expressed feature row is equally likely as a
    // negative, independent of abundance (SIMBA's uniform edge-corruption option).
    // Note a uniform noise distribution also makes the gem gene-total pooling moot —
    // pooling only mattered to keep an abundance-weighted basis matched to the
    // gene-paired positives; uniform is uniform either way.
    let feature_pool: Vec<u32> = (0..n_features as u32)
        .filter(|&f| feat_count[f as usize] > 0.0)
        .collect();
    if feature_pool.is_empty() {
        return None;
    }
    let neg_w: Vec<f32> = vec![1.0; feature_pool.len()];
    let neg = WeightedIndex::new(neg_w).expect("non-empty negative pool");
    // Degree = the feature's pooled pb-level count. `max(1e-8)` only guards the
    // picker; every entry in `feature_pool` already has a nonzero count.
    let deg_w: Vec<f32> = feature_pool
        .iter()
        .map(|&f| feat_count[f as usize].max(1e-8))
        .collect();
    let neg_by_degree = WeightedIndex::new(deg_w).expect("non-empty negative pool");

    Some(StratifiedSampler {
        pb_picker,
        active_pbs,
        per_pb: per_pb_samplers,
        neg,
        neg_by_degree,
        feature_pool,
    })
}

pub struct StratifiedEdgeBatchArgs<'a> {
    pub sampler: &'a StratifiedSampler,
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

    for _ in 0..args.batch_size {
        let local_pb = s.pb_picker.sample(rng);
        let p = s.active_pbs[local_pb];
        let pf = &s.per_pb[local_pb];
        let local_f = pf.picker.sample(rng);
        let f = pf.features[local_f];
        coarse_cells.push(p);
        fine_feats.push(f);
        // β-sharing gene-paired draw: the entry above is the spliced (identity)
        // row, drawn by its spliced count. Emit the paired unspliced row into the
        // same batch (same pb) so δ_g trains at the spliced sampling frequency.
        if !pf.paired.is_empty() {
            let u = pf.paired[local_f];
            if u != u32::MAX {
                coarse_cells.push(p);
                fine_feats.push(u);
            }
        }
    }

    // Negatives scale with the actual positive count (gene-paired draws emit up to
    // 2× batch_size positives); the loss reads `neg_feats[b*K..(b+1)*K]` per positive.
    let n_pos = fine_feats.len();
    let n_neg = n_pos * args.n_negatives;
    let mut neg_feats = Vec::with_capacity(n_neg);
    // Half uniform, half degree-proportional — SIMBA's 100 + 100 split. Drawn by
    // alternating rather than by a coin flip so the ratio is exact for any
    // `n_negatives`, including odd ones.
    for i in 0..n_neg {
        let local = if i % 2 == 0 {
            s.neg.sample(rng)
        } else {
            s.neg_by_degree.sample(rng)
        };
        neg_feats.push(s.feature_pool[local]);
    }

    EdgeBatch {
        coarse_cells,
        fine_feats,
        neg_feats,
        n_negatives: args.n_negatives,
    }
}

pub fn nce_loss(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    cell_coarse_to_fine: &[Vec<usize>],
    objective: NceObjective,
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

    nce_loss_with_cell_side(model, batch, e_cell_pos, b_cell_pos, objective, dev)
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
    objective: NceObjective,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let cell_idx_t = Tensor::from_slice(&batch.coarse_cells, b, dev)?;
    let e_cell_pos = model.e_cell.index_select(&cell_idx_t, 0)?;
    let b_cell_pos = model.b_cell.index_select(&cell_idx_t, 0)?;
    nce_loss_with_cell_side(model, batch, e_cell_pos, b_cell_pos, objective, dev)
}

/// Gather the feature-embedding rows for `idx`, applying the per-gene gate(s) when
/// enabled. A β-sharing factored model composes `β̃ + mask·δ̃` (identity + velocity, each
/// an independently gated effect — see [`JointEmbedModel::factored_feat_rows`]) via the
/// row→gene gathers, so only the batch's rows (`b + b·k`) are materialized — never the
/// full `[n_features, H]` dictionary — and gradients still reach `β`/`δ` and the gate
/// logits. A free model selects from the raw `e_feat` Var (SGC-smoothed when a smoother
/// is present; a factored model never has one — see `fit::run`) and gates that.
///
/// When gated, the base is reparam-sampled (`μ + σ·ε`) and multiplied by the gate's
/// weight table: this epoch's `z ~ Bern(pip)` when a selection pass installed one, else
/// the learned `α = σ(S/τ)` — an INDEPENDENT inclusion probability per (feature, dim),
/// with no normalizer and no null column. Gradients reach both the base and, on the
/// learned path, the logits. Ungated with no `pip` it is the plain gather. This is the
/// single feature-gather point for the bge + gem trainers.
pub fn gather_feature_rows(model: &JointEmbedModel, idx: &Tensor) -> Result<Tensor> {
    // Adapter: compose the gathered rows from the fixed dictionary and the
    // shared map, then let the same gate/effect machinery as the free path
    // apply on top. Mutually exclusive with `factor` by construction.
    if let Some(a) = &model.adapter {
        let mut mu = a.rho.index_select(idx, 0)?.matmul(&a.w)?;
        if let Some(r) = &a.residual {
            mu = mu.add(&r.index_select(idx, 0)?)?;
        }
        let logstd = model
            .e_feat_logstd
            .as_ref()
            .map(|l| l.index_select(idx, 0))
            .transpose()?;
        let w = model.gathered_gate_weights(
            crate::model::GateKind::Identity,
            model.s_feat.as_ref(),
            idx,
        )?;
        return model.gated_rows(&mu, logstd.as_ref(), w.as_ref(), true);
    }
    match &model.factor {
        Some(f) => {
            let genes = f.row_to_gene.index_select(idx, 0)?;
            let mask = f
                .splice_delta
                .as_ref()
                .map(|(_, m)| m.index_select(idx, 0)) // [b, 1] unspliced selector
                .transpose()?;
            model.factored_feat_rows(f, &genes, mask.as_ref(), true)
        }
        None => {
            // Gate reads the RAW Var (`e_feat_raw`) once the gate is on, so a post-
            // phase-1 materialize can overwrite `e_feat` without double-gating here.
            let raw = model.e_feat_raw.as_ref().unwrap_or(&model.e_feat);
            let mu = raw.index_select(idx, 0)?; // effect mean μ
            let logstd = model
                .e_feat_logstd
                .as_ref()
                .map(|l| l.index_select(idx, 0))
                .transpose()?;
            let w = model.gathered_gate_weights(
                crate::model::GateKind::Identity,
                model.s_feat.as_ref(),
                idx,
            )?;
            model.gated_rows(&mu, logstd.as_ref(), w.as_ref(), true)
        }
    }
}

/// Shared tail of [`nce_loss`] / [`nce_loss_identity`]: feature-side
/// gathers, raw bilinear (`E_feat[f] · E_cell[c] + b_feat`) scoring,
/// unweighted-mean log-σ aggregation over the batch. The cell-side
/// embeddings come pre-resolved (pooled or directly gathered).
fn nce_loss_with_cell_side(
    model: &JointEmbedModel,
    batch: EdgeBatch,
    e_cell_pos: Tensor,
    b_cell_pos: Tensor,
    objective: NceObjective,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.coarse_cells.len();
    let k = batch.n_negatives;

    // Gather the feature rows scored THIS step (see `gather_feature_rows`). Shared
    // by pos + neg.
    let gather_feat = |idx: &Tensor| gather_feature_rows(model, idx);

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

    let negs = std::slice::from_ref(&neg_score);
    let per_edge = match objective {
        NceObjective::Logistic => logistic_nce(&pos_score, negs)?,
        NceObjective::Softmax => softmax_nce(&pos_score, negs)?,
    };

    // Unweighted mean over the batch's positives (pure count-weighted training:
    // the count weighting lives in the sampler's positive draw, not the loss).
    per_edge.mean(0)
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
        // pb has all four rows: gene0 spliced 5 + unspliced 2, gene1 spliced 3,
        // gene2 unspliced-only 4.
        let edges = vec![(0u32, 5.0f32), (1, 2.0), (2, 3.0), (3, 4.0)];
        let (features, paired, weights) = gene_paired_entries(&edges, &fp);

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
        // gene0 spliced 5 + unspliced 2 → total 7. Confirms the unspliced count
        // IS summed into the (pure count) weight.
        let edges = vec![(0u32, 5.0f32), (1, 2.0)];
        let (features, paired, weights) = gene_paired_entries(&edges, &fp);
        assert_eq!(features, vec![0]);
        assert_eq!(paired, vec![1]);
        assert_eq!(weights, vec![7.0]);
    }
}
