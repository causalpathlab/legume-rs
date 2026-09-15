//! Amortized phase 2: one encoder places every pair, and every cell, on the
//! frozen dictionary.
//!
//! The exact solver ([`super::newton_polish`]) settles the same strictly
//! convex problem once per node — the multinomial MAP of the pooled counts
//! against `e_feat` — and pairs outnumber cells by an order of magnitude.
//! The map from counts to that optimum is smooth, so a small encoder trained
//! on the same objective places every pair in one forward pass, on the
//! device, and the per-pair solver becomes the check on it (and the finisher
//! of its rare misses) rather than the workhorse.
//!
//! The MAP is a function of one statistic. Write `S = Σ_g n_g e_g` for the
//! count-weighted sum of dictionary rows, `N = Σ_g n_g` for the depth and
//! `c = Σ_g n_g b_g`; then
//!
//! ```text
//! s_g    = ⟨e_g, z⟩ + b_g
//! L(z)   = N · lse_g(s_g) − ⟨S, z⟩ − c + (λ/2)‖z‖²
//! ```
//!
//! so the counts enter the objective only through `(S, N, c)`, three exact
//! sums over a cell's nonzeros, and pool additively over a pair. Nothing here
//! is ever dense over the genes on the count side: the corpus carries each
//! cell's `(S, N, c)`, a step's input is `[2B, D + 1]`, and the one `G`-wide
//! tensor is the partition `lse_g` — a GEMM output on the device, the same
//! sum the exact solver forms, never an upload.
//!
//! The encoder is a cell trunk applied twice and a [`SymmetricPairHead`]: the
//! trunk reads `[S/N ‖ ln(N + 1)]`, standardised by population moments saved
//! with the encoder, through two layer-normed ReLU layers into a cell code
//! `h`; the head — a gated mixture of linear experts over
//! `[(h_u + h_v)/2 ‖ h_u ⊙ h_v ‖ the pair's own statistic]` — turns two codes
//! into the pair code `z_uv`. Symmetric by construction, and with no
//! per-gene parameter, so the trained encoder transfers to any re-aligned
//! axis. Normalisation is per row (layer norm), never per batch: a batch
//! norm's running variance for a unit that rarely fires sits near zero, and
//! the rare input that fires it is then blown up at inference — the wild
//! placement a shared map must never produce. And the placement is kept
//! inside the ball the optimum provably lies in: at the optimum
//! `λz = S − N p̄` with `p̄` a convex combination of dictionary rows, so
//! `‖z‖ ≤ 2N·max_g‖e_g‖/λ` — binding only for a row with a handful of
//! counts, exactly where a shared map would otherwise extrapolate.
//!
//! Every placement also leaves with a certificate: the Newton decrement
//! `½ ∇ᵀH⁻¹∇` at the placement, with `∇ = N p̄ − S + λz` and
//! `H = N·Cov_p(e) + λI` — the excess likelihood the local quadratic model
//! puts on it, in nats, exact for a quadratic and the standard Newton
//! stopping statistic. A row it puts far out is a row the map extrapolated
//! on — a rare composition, a near-empty profile — and the caller finishes
//! those few exactly.
//!
//! `β_uv` is never learned: given `z` the intercept is `ln N − lse(s)` in
//! closed form. The self-pair term is what makes the cell's own placement a
//! trained quantity: `z_uu` is the MAP of the doubled profile `2x_u`, whose
//! composition is `x_u`'s, so `.cell_embedding.parquet` comes from the same
//! map as every pair. A row with no counts is put at the origin afterwards,
//! where the solver puts it.

use super::{PairDictionary, SCORE_CLAMP};
use crate::util::common::*;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{
    layer_norm, linear, AdamW, LayerNorm, LayerNormConfig, Linear, Module, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};
use candle_util::encoder::{SymmetricPairHead, SymmetricPairHeadArgs};
use candle_util::nn::seed_uniform_vars;
use candle_util::vae::{clip_and_step_dense, PhaseTimers};
use matrix_util::rand_util::mix_seed;
use matrix_util::utils::{cosine, quantiles};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

///////////////
// Constants //
///////////////

/// The var-name prefix the encoder is saved under.
const VAR_PREFIX: &str = "pair_enc";
/// The non-var tensors saved beside the vars: the statistic's population
/// moments and the encoder's own hyperparameters `[L, K, λ]`.
const STAT_MEAN_TENSOR: &str = "pair_enc.stat_mean";
const STAT_STD_TENSOR: &str = "pair_enc.stat_std";
const HPARAMS_TENSOR: &str = "pair_enc.hparams";
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const GRAD_CLIP: f64 = 5.0;
/// A step with fewer pairs than this is skipped.
const MIN_STEP_PAIRS: usize = 16;
/// Cells placed per pass at inference, at most.
pub(crate) const CELL_BLOCK: usize = 4096;
/// Elements of one `[rows × G]` partition block, whatever `G` is: every
/// scoring pass — a training step's pairs and endpoints, a cell block, a pair
/// block — takes as many rows as fit this budget, so a whole-transcriptome
/// axis takes fewer rows per pass and a panel takes the caps, and the device
/// holds a fixed amount of dense work either way.
const BLOCK_ELEMENTS: usize = 1 << 25;
/// Optimizer steps a run makes at least, whatever its pair count: a small
/// sample at the batch cap would otherwise train for a handful of steps.
const MIN_STEPS: usize = 512;
/// Pairs held out for the per-epoch evaluation line and the closing check
/// against the exact solver.
const CHECK_PAIRS: usize = 2048;

/// The encoder's shape and training budget. `cage` runs [`Default`]; the
/// closing check against the exact solver says how well it did, and a tiny
/// fixture shrinks it.
#[derive(Debug, Clone)]
pub struct PairEncoderSpec {
    /// Width `L` of the cell code.
    pub trunk_width: usize,
    /// Experts `K` in the pair head; `1` is a plain linear head.
    pub n_experts: usize,
    /// Passes over the pairs, at least: a run makes [`MIN_STEPS`] optimizer
    /// steps whatever its pair count. `0` does not train.
    pub epochs: usize,
    /// Pairs per optimizer step, at most: a step is also held to
    /// [`BLOCK_ELEMENTS`] on the gene axis.
    pub batch: usize,
}

impl Default for PairEncoderSpec {
    fn default() -> Self {
        Self {
            trunk_width: 64,
            n_experts: 4,
            epochs: 3,
            batch: 4096,
        }
    }
}

/// Rows of a `[rows × g]` block within [`BLOCK_ELEMENTS`], at most `cap` and
/// at least one.
fn rows_within_budget(g: usize, cap: usize) -> usize {
    (BLOCK_ELEMENTS / g.max(1)).clamp(1, cap.max(1))
}

////////////
// Corpus //
////////////

/// One cell's batch-divided counts on the active axis, sorted by position,
/// with the statistic the objective reads them through.
#[derive(Clone)]
pub(crate) struct CellRow {
    pub genes: Vec<u32>,
    pub counts: Vec<f32>,
    /// `N = Σ_g n_g`.
    pub total: f32,
    /// `S = Σ_g n_g e_g`, `[D]`.
    pub sums: Vec<f32>,
    /// `c = Σ_g n_g b_g`: the data term's constant.
    pub offset: f32,
}

impl CellRow {
    /// From a `(global gene, count)` profile sorted by gene with no
    /// duplicates — what [`crate::util::gene_axis::GeneAxis::pool_profile`]
    /// hands out — keeping the genes the dictionary carries.
    pub(crate) fn from_profile(dict: &PairDictionary, profile: &[(u32, f32)]) -> Self {
        debug_assert!(
            profile.windows(2).all(|w| w[0].0 < w[1].0),
            "a cell profile must be sorted by gene with no duplicates"
        );
        let (genes, counts): (Vec<u32>, Vec<f32>) = dict.to_local(profile).into_iter().unzip();
        let (sums, total, offset) = dict.statistic(&genes, &counts);
        Self {
            genes,
            counts,
            total,
            sums,
            offset,
        }
    }

    /// The pooled `(position, count)` profile of two rows, sorted.
    pub(crate) fn pooled(&self, other: &CellRow) -> Vec<(u32, f32)> {
        let mut out = Vec::with_capacity(self.genes.len() + other.genes.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.genes.len() && j < other.genes.len() {
            match self.genes[i].cmp(&other.genes[j]) {
                std::cmp::Ordering::Less => {
                    out.push((self.genes[i], self.counts[i]));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push((other.genes[j], other.counts[j]));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push((self.genes[i], self.counts[i] + other.counts[j]));
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend(
            self.genes[i..]
                .iter()
                .copied()
                .zip(self.counts[i..].iter().copied()),
        );
        out.extend(
            other.genes[j..]
                .iter()
                .copied()
                .zip(other.counts[j..].iter().copied()),
        );
        out
    }

    /// The self-pair: the row pooled with itself.
    pub(crate) fn doubled(&self) -> Vec<(u32, f32)> {
        self.genes
            .iter()
            .zip(&self.counts)
            .map(|(&g, &n)| (g, 2.0 * n))
            .collect()
    }
}

/// Every cell's `(S, N, c)` on the device, row `i` for cell `i`: the whole
/// corpus's, uploaded once, gathered per block by endpoint id.
struct CorpusStats {
    /// `[n × D]`.
    sums: Tensor,
    /// `[n]` each.
    totals: Tensor,
    offsets: Tensor,
}

impl CorpusStats {
    fn new(corpus: &[CellRow], d: usize, dev: &Device) -> anyhow::Result<Self> {
        let n = corpus.len();
        let sums: Vec<f32> = corpus.iter().flat_map(|r| r.sums.iter().copied()).collect();
        Ok(Self {
            sums: Tensor::from_vec(sums, (n, d), dev)?,
            totals: Tensor::from_vec(corpus.iter().map(|r| r.total).collect(), n, dev)?,
            offsets: Tensor::from_vec(corpus.iter().map(|r| r.offset).collect(), n, dev)?,
        })
    }

    /// The rows of `ids` → `(sums, totals, offsets)`.
    fn gather(&self, ids: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        Ok((
            self.sums.index_select(ids, 0)?,
            self.totals.index_select(ids, 0)?,
            self.offsets.index_select(ids, 0)?,
        ))
    }
}

/// Population mean and standard deviation of every cell's statistic
/// `[S/N ‖ ln(N + 1)]` over the corpus — `(mean, std)`, `[D + 1]` each, the
/// std floored so a constant coordinate is left alone.
fn stat_moments(d: usize, corpus: &[CellRow]) -> (Vec<f32>, Vec<f32>) {
    let (sum, sq) = corpus
        .par_iter()
        .fold(
            || (vec![0f64; d + 1], vec![0f64; d + 1]),
            |(mut sum, mut sq), row| {
                let n = row.total.max(1.0);
                for (j, &v) in row.sums.iter().enumerate() {
                    let v = f64::from(v / n);
                    sum[j] += v;
                    sq[j] += v * v;
                }
                let ln_n = f64::from((row.total + 1.0).ln());
                sum[d] += ln_n;
                sq[d] += ln_n * ln_n;
                (sum, sq)
            },
        )
        .reduce(
            || (vec![0f64; d + 1], vec![0f64; d + 1]),
            |(mut a, mut b), (c, e)| {
                a.iter_mut().zip(&c).for_each(|(x, y)| *x += y);
                b.iter_mut().zip(&e).for_each(|(x, y)| *x += y);
                (a, b)
            },
        );
    let n = corpus.len().max(1) as f64;
    let mean: Vec<f32> = sum.iter().map(|&v| (v / n) as f32).collect();
    let std: Vec<f32> = sum
        .iter()
        .zip(&sq)
        .map(|(&s, &q)| ((q / n - (s / n).powi(2)).max(0.0).sqrt() as f32).max(1e-3))
        .collect();
    (mean, std)
}

////////////////
// Placements //
////////////////

/// Where a set of nodes (pairs, or cells) landed: latent, intercept, and the
/// certificate — the Newton decrement at the placement, the excess in nats
/// the local quadratic model puts on it over the optimum, zero there.
pub(crate) struct Placement {
    /// `[n × D]`.
    pub latent: Mat,
    pub bias: Vec<f32>,
    pub gap: Vec<f32>,
}

impl Placement {
    pub(crate) fn len(&self) -> usize {
        self.bias.len()
    }

    pub(crate) fn set(&mut self, i: usize, theta: &[f32], beta: f32, gap: f32) {
        for (j, &t) in theta.iter().enumerate().take(self.latent.ncols()) {
            self.latent[(i, j)] = t;
        }
        self.bias[i] = beta;
        self.gap[i] = gap;
    }
}

/// Every pair's and every cell's placement, in the callers' orders.
pub(crate) struct Encoded {
    pub pairs: Placement,
    pub cells: Placement,
}

////////////
// Device //
////////////

/// The frozen side on the device: the dictionary for the composition mean,
/// its transpose for the partition, its row outer products for the
/// curvature, the log-rate offset and the statistic's population moments.
struct DeviceDict {
    /// `[G × D]`.
    e_gd: Tensor,
    /// `[D × G]`.
    e_hd: Tensor,
    /// `[G × D²]`, row `g` = `e_g e_gᵀ` flattened.
    ee_gdd: Tensor,
    /// `[1 × G]`.
    b_1d: Tensor,
    /// `[1 × (D + 1)]` each.
    stat_mean: Tensor,
    stat_std: Tensor,
    /// `max_g ‖e_g‖`.
    max_row_norm: f32,
    g: usize,
    d: usize,
}

impl DeviceDict {
    fn new(
        dict: &PairDictionary,
        stat_mean: &[f32],
        stat_std: &[f32],
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (g, d) = (dict.n_active(), dict.d);
        anyhow::ensure!(
            stat_mean.len() == d + 1 && stat_std.len() == d + 1,
            "pair encoder: statistic moments have {} and {} entries, expected {}",
            stat_mean.len(),
            stat_std.len(),
            d + 1
        );
        let e_gd = Tensor::from_slice(&dict.feat, (g, d), dev)?;
        let ee_gdd = e_gd
            .unsqueeze(2)?
            .broadcast_mul(&e_gd.unsqueeze(1)?)?
            .reshape((g, d * d))?;
        Ok(Self {
            e_hd: e_gd.t()?.contiguous()?,
            ee_gdd,
            e_gd,
            b_1d: Tensor::from_slice(&dict.b, (1, g), dev)?,
            stat_mean: Tensor::from_slice(stat_mean, (1, d + 1), dev)?,
            stat_std: Tensor::from_slice(stat_std, (1, d + 1), dev)?,
            max_row_norm: dict
                .feat
                .chunks_exact(d)
                .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
                .fold(0f32, f32::max),
            g,
            d,
        })
    }

    fn device(&self) -> &Device {
        self.b_1d.device()
    }
}

/////////////////
// The encoder //
/////////////////

/// Statistic → cell code: two layer-normed ReLU layers and a linear head.
struct CellTrunk {
    /// `[D + 1] → L` and `L → L`, each followed by layer norm and ReLU.
    layers: Vec<(Linear, LayerNorm)>,
    /// `L → L`, linear: the code is signed.
    head: Linear,
}

impl CellTrunk {
    fn new(stat_dim: usize, width: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(2);
        let mut in_dim = stat_dim;
        for k in 0..2 {
            let lin = linear(in_dim, width, vb.pp(format!("fc.{k}")))?;
            let norm = layer_norm(width, LayerNormConfig::default(), vb.pp(format!("ln.{k}")))?;
            layers.push((lin, norm));
            in_dim = width;
        }
        Ok(Self {
            layers,
            head: linear(width, width, vb.pp("head"))?,
        })
    }

    /// Standardised statistic `[n, D + 1]` → code `[n, L]`.
    fn forward(&self, stat: &Tensor) -> anyhow::Result<Tensor> {
        let mut h = stat.clone();
        for (lin, norm) in &self.layers {
            h = norm.forward(&lin.forward(&h)?)?.relu()?;
        }
        Ok(self.head.forward(&h)?)
    }
}

/// The trained trunk and pair head on a device dictionary.
pub(crate) struct PairEncoder {
    cell: CellTrunk,
    head: SymmetricPairHead,
    varmap: VarMap,
    dict: DeviceDict,
    /// The ridge `λ` the encoder was fitted under: the objective's, the
    /// certificate's, and the norm ball's.
    ridge: f32,
    /// `2·max_g‖e_g‖ / λ`: the optimum's norm bound per pooled count.
    cap_per_count: f64,
}

/// One node's `(S, N, c)` on the device, `[n × D]`, `[n]`, `[n]`.
struct Stats {
    sums: Tensor,
    totals: Tensor,
    offsets: Tensor,
}

impl Stats {
    /// Two endpoints pooled: every sum adds.
    fn pooled(&self, other: &Stats) -> anyhow::Result<Stats> {
        Ok(Stats {
            sums: (&self.sums + &other.sums)?,
            totals: (&self.totals + &other.totals)?,
            offsets: (&self.offsets + &other.offsets)?,
        })
    }

    /// The self-pair: doubled.
    fn doubled(&self) -> anyhow::Result<Stats> {
        self.pooled(self)
    }
}

/// A block of pairs through the trunk and the head: what training,
/// evaluation and inference all read.
struct PairForward {
    /// `[2B, L]` cell codes, the `u`s then the `v`s.
    h: Tensor,
    /// The `2B` endpoints' own statistics.
    endpoints: Stats,
    /// `[B, 2L + D + 1]` pair features.
    features: Tensor,
    /// `[B, D]` pair codes.
    z_uv: Tensor,
    /// The pairs' pooled statistics.
    pairs: Stats,
}

/// What training reports back.
pub(crate) struct TrainStats {
    pub steps: usize,
    pub nll_per_count: f32,
}

impl PairEncoder {
    /// An untrained encoder on `dict`, its statistic standardised by the
    /// corpus, its vars seeded from `seed`.
    pub(crate) fn build(
        dict: &PairDictionary,
        corpus: &[CellRow],
        trunk_width: usize,
        n_experts: usize,
        ridge: f32,
        seed: u64,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (mean, std) = stat_moments(dict.d, corpus);
        let this = Self::construct(
            DeviceDict::new(dict, &mean, &std, dev)?,
            trunk_width,
            n_experts,
            ridge,
            dev,
        )?;
        // candle's `VarBuilder` initialises from an unseeded stream; the
        // layer norms keep their defaults.
        seed_uniform_vars(&this.varmap, seed, |name| name.contains(".ln."))?;
        Ok(this)
    }

    fn construct(
        dict: DeviceDict,
        trunk_width: usize,
        n_experts: usize,
        ridge: f32,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            trunk_width > 0,
            "pair encoder: trunk width must be positive"
        );
        anyhow::ensure!(ridge > 0.0, "pair encoder: the ridge must be positive");
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev).pp(VAR_PREFIX);
        // The statistic and the log depth: into the trunk per cell, and
        // beside the two codes into the head per pair.
        let stat_dim = dict.d + 1;
        let cell = CellTrunk::new(stat_dim, trunk_width, vb.pp("cell"))?;
        let head = SymmetricPairHead::new(
            SymmetricPairHeadArgs {
                code_dim: trunk_width,
                out_dim: dict.d,
                n_experts,
                extra_dim: stat_dim,
            },
            vb.pp("head"),
        )?;
        let cap_per_count = 2.0 * f64::from(dict.max_row_norm) / f64::from(ridge);
        Ok(Self {
            cell,
            head,
            varmap,
            dict,
            ridge,
            cap_per_count,
        })
    }

    /// Rebuild a saved encoder on `dict`, with the widths and the ridge the
    /// file records.
    pub(crate) fn load(dict: &PairDictionary, path: &str, dev: &Device) -> anyhow::Result<Self> {
        let tensors = candle_util::candle_core::safetensors::load(path, dev)?;
        let side = |name: &str| -> anyhow::Result<Vec<f32>> {
            Ok(tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("{path}: no `{name}` tensor"))?
                .flatten_all()?
                .to_vec1()?)
        };
        let (mean, std) = (side(STAT_MEAN_TENSOR)?, side(STAT_STD_TENSOR)?);
        let hparams = side(HPARAMS_TENSOR)?;
        anyhow::ensure!(
            hparams.len() == 3,
            "{path}: `{HPARAMS_TENSOR}` has {} entries, expected [L, K, λ]",
            hparams.len()
        );
        let mut this = Self::construct(
            DeviceDict::new(dict, &mean, &std, dev)?,
            hparams[0] as usize,
            hparams[1] as usize,
            hparams[2],
            dev,
        )?;
        // Matches by name and ignores the side tensors, which are not vars.
        this.varmap.load(path)?;
        Ok(this)
    }

    /// Write every var and the side tensors to one safetensors file.
    pub(crate) fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        tensors.insert(
            STAT_MEAN_TENSOR.to_string(),
            self.dict.stat_mean.flatten_all()?,
        );
        tensors.insert(
            STAT_STD_TENSOR.to_string(),
            self.dict.stat_std.flatten_all()?,
        );
        let hparams = [
            self.trunk_width() as f32,
            self.n_experts() as f32,
            self.ridge,
        ];
        tensors.insert(
            HPARAMS_TENSOR.to_string(),
            Tensor::new(&hparams, self.dict.device())?,
        );
        candle_util::candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub(crate) fn trunk_width(&self) -> usize {
        self.head.code_dim()
    }

    pub(crate) fn n_experts(&self) -> usize {
        self.head.n_experts()
    }

    pub(crate) fn ridge(&self) -> f32 {
        self.ridge
    }

    /// `[S / max(N, 1) ‖ ln(N + 1)]` standardised by the population moments
    /// → `[n, D + 1]`.
    fn stat_of(&self, stats: &Stats) -> anyhow::Result<Tensor> {
        let n1 = stats.totals.unsqueeze(1)?;
        let mean = stats.sums.broadcast_div(&n1.clamp(1.0, f64::INFINITY)?)?;
        let ln_n = (&n1 + 1.0)?.log()?;
        let raw = Tensor::cat(&[&mean, &ln_n], 1)?;
        Ok(raw
            .broadcast_sub(&self.dict.stat_mean)?
            .broadcast_div(&self.dict.stat_std)?)
    }

    /// Cell codes → `[n, L]`.
    fn cell_codes(&self, stats: &Stats) -> anyhow::Result<Tensor> {
        self.cell.forward(&self.stat_of(stats)?)
    }

    /// The pair code of two endpoints from their codes and pooled statistic:
    /// the head's output kept inside the optimum's ball for the pooled
    /// depth. Also returns the pair features, for the gate readout.
    fn pair_code(
        &self,
        h_u: &Tensor,
        h_v: &Tensor,
        pooled: &Stats,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let stat = self.stat_of(pooled)?;
        let features = self.head.features(h_u, h_v, Some(&stat))?;
        let z = self.head.forward(h_u, h_v, Some(&stat))?;
        let cap = pooled
            .totals
            .unsqueeze(1)?
            .affine(self.cap_per_count, 0.0)?; // [B, 1]
        let norm = z.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1]
                                                     // min(1, cap / ‖z‖), with ‖z‖ = 0 left alone.
        let scale = (cap / (norm + 1e-12)?)?.clamp(0.0, 1.0)?;
        Ok((z.broadcast_mul(&scale)?, features))
    }

    /// The self-pair of every row of a block: its code with itself, at the
    /// doubled depth.
    fn self_pair_code(&self, h: &Tensor, stats: &Stats) -> anyhow::Result<Tensor> {
        Ok(self.pair_code(h, h, &stats.doubled()?)?.0)
    }

    /// A block of pairs, from their endpoints' statistics (the `u`s then the
    /// `v`s), through the trunk and the head.
    fn pair_forward(&self, endpoints: Stats) -> anyhow::Result<PairForward> {
        let b = endpoints.totals.dim(0)? / 2;
        let h = self.cell_codes(&endpoints)?;
        let half = |t: &Tensor, k: usize| t.narrow(0, k * b, b);
        let (h_u, h_v) = (half(&h, 0)?, half(&h, 1)?);
        let side = |k: usize| -> anyhow::Result<Stats> {
            Ok(Stats {
                sums: half(&endpoints.sums, k)?,
                totals: half(&endpoints.totals, k)?,
                offsets: half(&endpoints.offsets, k)?,
            })
        };
        let pairs = side(0)?.pooled(&side(1)?)?;
        let (z_uv, features) = self.pair_code(&h_u, &h_v, &pairs)?;
        Ok(PairForward {
            h,
            endpoints,
            features,
            z_uv,
            pairs,
        })
    }

    /// Scores `z·e_hd + b` → `[n, G]`.
    fn scores(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        Ok(z.matmul(&self.dict.e_hd)?.broadcast_add(&self.dict.b_1d)?)
    }

    /// The profiled multinomial NLL of placements `z` for the nodes `stats`,
    /// per row: `N·lse(s) − ⟨S, z⟩ − c`.
    fn nll(&self, z: &Tensor, scores: &Tensor, stats: &Stats) -> anyhow::Result<Tensor> {
        let lse = scores.log_sum_exp(1)?;
        let data = (z * &stats.sums)?.sum(1)?;
        Ok((((&stats.totals * lse)? - data)? - &stats.offsets)?)
    }

    /// The Newton decrement `½ ∇ᵀH⁻¹∇` per row for placements `z` of the
    /// nodes `stats`, with `∇ = N p̄ − S + λz` and `H = N·Cov_p(e) + λI`
    /// under the composition `p` the scores fit: the composition's mean and
    /// second moment come off the device, the `D × D` solves run here.
    fn decrement(&self, z: &Tensor, scores: &Tensor, stats: &Stats) -> anyhow::Result<Vec<f32>> {
        let d = self.dict.d;
        let p = candle_util::candle_nn::ops::softmax(scores, 1)?; // [B, G]
        let pbar: Vec<f32> = p.matmul(&self.dict.e_gd)?.flatten_all()?.to_vec1()?; // [B, D]
        let m2: Vec<f32> = p.matmul(&self.dict.ee_gdd)?.flatten_all()?.to_vec1()?; // [B, D²]
        let z: Vec<f32> = z.flatten_all()?.to_vec1()?;
        let sums: Vec<f32> = stats.sums.flatten_all()?.to_vec1()?;
        let totals: Vec<f32> = stats.totals.to_vec1()?;
        let ridge = self.ridge;
        Ok(totals
            .par_iter()
            .enumerate()
            .map(|(i, &n)| {
                let (pb, zi, si) = (
                    &pbar[i * d..(i + 1) * d],
                    &z[i * d..(i + 1) * d],
                    &sums[i * d..(i + 1) * d],
                );
                let grad = nalgebra::DVector::<f32>::from_iterator(
                    d,
                    (0..d).map(|j| n * pb[j] - si[j] + ridge * zi[j]),
                );
                let hess = nalgebra::DMatrix::<f32>::from_fn(d, d, |r, c| {
                    let cov = m2[i * d * d + r * d + c] - pb[r] * pb[c];
                    n * cov + if r == c { ridge } else { 0.0 }
                });
                match hess.cholesky() {
                    Some(chol) => 0.5 * grad.dot(&chol.solve(&grad)),
                    None => f32::INFINITY,
                }
            })
            .collect())
    }

    /// One step's loss on a block of pairs: the pairs' objective plus the
    /// endpoints' own, each with its ridge.
    fn step_loss(&self, endpoints: Stats) -> anyhow::Result<Tensor> {
        let half_ridge = f64::from(self.ridge) / 2.0;
        let fwd = self.pair_forward(endpoints)?;
        let nll_uv = self.nll(&fwd.z_uv, &self.scores(&fwd.z_uv)?, &fwd.pairs)?;
        let ridge_uv = fwd.z_uv.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;
        let z_uu = self.self_pair_code(&fwd.h, &fwd.endpoints)?;
        let nll_uu = self.nll(&z_uu, &self.scores(&z_uu)?, &fwd.endpoints.doubled()?)?;
        let ridge_uu = z_uu.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;
        Ok(((nll_uv + ridge_uv)?.mean_all()? + (nll_uu + ridge_uu)?.mean_all()?)?)
    }

    /// The endpoint statistics of a chunk of pairs, the `u` rows then the
    /// `v` rows, gathered on the device.
    fn endpoint_stats(&self, corpus: &CorpusStats, chunk: &[(u32, u32)]) -> anyhow::Result<Stats> {
        let ids: Vec<u32> = chunk
            .iter()
            .map(|&(u, _)| u)
            .chain(chunk.iter().map(|&(_, v)| v))
            .collect();
        let ids = Tensor::from_vec(ids, chunk.len() * 2, self.dict.device())?;
        let (sums, totals, offsets) = corpus.gather(&ids)?;
        Ok(Stats {
            sums,
            totals,
            offsets,
        })
    }

    /// NLL per pooled count over `pairs`, and the mean expert weights: the
    /// checkpoint reading, and whether the gate still routes.
    fn evaluate(
        &self,
        corpus: &CorpusStats,
        pairs: &[(u32, u32)],
        batch: usize,
    ) -> anyhow::Result<(f32, Vec<f32>)> {
        let (mut nll, mut count) = (0f64, 0f64);
        let mut pi_sum = vec![0f64; self.head.n_experts()];
        for chunk in pairs.chunks(batch.max(1)) {
            let fwd = self.pair_forward(self.endpoint_stats(corpus, chunk)?)?;
            let pi: Vec<f32> = self.head.gate(&fwd.features)?.sum(0)?.to_vec1()?;
            for (s, p) in pi_sum.iter_mut().zip(&pi) {
                *s += f64::from(*p);
            }
            nll += f64::from(
                self.nll(&fwd.z_uv, &self.scores(&fwd.z_uv)?, &fwd.pairs)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            count += f64::from(fwd.pairs.totals.sum_all()?.to_scalar::<f32>()?);
        }
        let n_rows = pairs.len().max(1) as f64;
        let pi_mean = pi_sum.iter().map(|&s| (s / n_rows) as f32).collect();
        Ok(((nll / count.max(1.0)) as f32, pi_mean))
    }

    /// Train on the pairs' likelihood (and the endpoints' own), from the
    /// current vars.
    pub(crate) fn train(
        &self,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        spec: &PairEncoderSpec,
        seed: u64,
    ) -> anyhow::Result<TrainStats> {
        let n_pairs = edges.len();
        // A step scores its pairs and their two endpoints.
        let batch = rows_within_budget(3 * self.dict.g, spec.batch).max(MIN_STEP_PAIRS);
        let steps_per_epoch = n_pairs.div_ceil(batch).max(1);
        let epochs = if spec.epochs == 0 {
            0
        } else {
            spec.epochs.max(MIN_STEPS.div_ceil(steps_per_epoch))
        };
        let stats = CorpusStats::new(corpus, self.dict.d, self.dict.device())?;
        let mut adam = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: LEARNING_RATE,
                weight_decay: WEIGHT_DECAY,
                ..Default::default()
            },
        )?;

        // A fixed, seeded subset scored in evaluation mode after every epoch.
        let check: Vec<(u32, u32)> = {
            let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x0043_4845_434b));
            (0..CHECK_PAIRS.min(n_pairs))
                .map(|_| edges[rng.random_range(0..n_pairs)])
                .collect()
        };
        let (nll0, _) = self.evaluate(&stats, &check, batch)?;
        info!(
            "Pair encoder: {n_pairs} pairs, L={}, K={}, {epochs} epochs of {batch}-pair steps, \
             ridge λ={}; NLL/count before {nll0:.4}",
            self.trunk_width(),
            self.n_experts(),
            self.ridge
        );

        let mut order: Vec<usize> = (0..n_pairs).collect();
        let mut timers = PhaseTimers::default();
        let (mut steps, mut skipped) = (0usize, 0usize);
        let mut nll_per_count = nll0;
        let bar = new_progress_bar((epochs * steps_per_epoch) as u64).with_message("pair encoder");
        for epoch in 0..epochs {
            order.shuffle(&mut StdRng::seed_from_u64(mix_seed(seed, epoch as u64)));
            let mut loss_sum = 0f64;
            let mut n_steps = 0usize;
            for chunk in order.chunks(batch) {
                if chunk.len() < MIN_STEP_PAIRS {
                    bar.inc(1);
                    continue;
                }
                let t = std::time::Instant::now();
                let pairs: Vec<(u32, u32)> = chunk.iter().map(|&i| edges[i]).collect();
                let endpoints = self.endpoint_stats(&stats, &pairs)?;
                timers.precompute += t.elapsed();

                let t = std::time::Instant::now();
                let loss = self.step_loss(endpoints)?;
                timers.encoder_fwd += t.elapsed();

                let t = std::time::Instant::now();
                let grads = loss.backward()?;
                timers.backward += t.elapsed();

                let t = std::time::Instant::now();
                if !clip_and_step_dense(&mut adam, grads, GRAD_CLIP)? {
                    skipped += 1;
                }
                timers.optimize += t.elapsed();

                loss_sum += f64::from(loss.to_scalar::<f32>()?);
                n_steps += 1;
                steps += 1;
                bar.inc(1);
            }
            let (nll, pi) = self.evaluate(&stats, &check, batch)?;
            nll_per_count = nll;
            info!(
                "Pair encoder epoch {}/{epochs}: mean step loss {:.2}, held-out NLL/count {:.4}, expert usage {}",
                epoch + 1,
                loss_sum / n_steps.max(1) as f64,
                nll,
                pi.iter()
                    .map(|p| format!("{p:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        bar.finish_and_clear();
        timers.log_summary();
        if skipped > 0 {
            warn!("Pair encoder: {skipped}/{steps} steps were skipped for non-finite gradients");
        }
        Ok(TrainStats {
            steps,
            nll_per_count,
        })
    }

    /// Place every cell, then every pair, in evaluation mode. `cell_block`
    /// cells share one pass and `pair_block` pairs one, at most: both are
    /// held to [`BLOCK_ELEMENTS`] on the gene axis.
    pub(crate) fn encode_all(
        &self,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        pair_block: usize,
        cell_block: usize,
    ) -> anyhow::Result<Encoded> {
        let dev = self.dict.device();
        let (n_cells, g, d) = (corpus.len(), self.dict.g, self.dict.d);
        let cell_block = rows_within_budget(g, cell_block);
        let pair_block = rows_within_budget(g, pair_block);
        let stats = CorpusStats::new(corpus, d, dev)?;

        // Cells: the codes stay on the device for the pair pass; the
        // self-pairs are written out.
        let mut codes = Vec::new();
        let mut cell_latent = Mat::zeros(n_cells, d);
        let mut cell_bias = vec![0f32; n_cells];
        let mut cell_gap = vec![0f32; n_cells];
        let bar = new_progress_bar(n_cells as u64).with_message("encoding cells");
        for (lb, ub) in
            matrix_util::utils::generate_minibatch_intervals(n_cells, 0, Some(cell_block))
        {
            let ids = Tensor::from_vec((lb as u32..ub as u32).collect::<Vec<u32>>(), ub - lb, dev)?;
            let (sums, totals, offsets) = stats.gather(&ids)?;
            let own = Stats {
                sums,
                totals,
                offsets,
            };
            let h = self.cell_codes(&own)?;
            let z = self.self_pair_code(&h, &own)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            // The self-pair's objective is over the doubled profile.
            let gap = self.decrement(&z, &scores, &own.doubled()?)?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, c) in (lb..ub).enumerate() {
                let total = corpus[c].total;
                if total > 0.0 {
                    for j in 0..d {
                        cell_latent[(c, j)] = z_host[i * d + j];
                    }
                    cell_bias[c] = (total.ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                    cell_gap[c] = gap[i];
                }
            }
            codes.push(h);
            bar.inc((ub - lb) as u64);
        }
        bar.finish_and_clear();
        let codes = Tensor::cat(&codes, 0)?;

        // Pairs: the two endpoints' codes and statistics gathered on the
        // device, through the head.
        let n_pairs = edges.len();
        let mut pair_latent = Mat::zeros(n_pairs, d);
        let mut pair_bias = vec![0f32; n_pairs];
        let mut pair_gap = vec![0f32; n_pairs];
        let bar = new_progress_bar(n_pairs as u64).with_message("encoding pairs");
        for (lb, ub) in
            matrix_util::utils::generate_minibatch_intervals(n_pairs, 0, Some(pair_block))
        {
            let chunk = &edges[lb..ub];
            let b = chunk.len();
            let ids = |pick: fn(&(u32, u32)) -> u32| -> anyhow::Result<Tensor> {
                Ok(Tensor::from_vec(
                    chunk.iter().map(pick).collect::<Vec<u32>>(),
                    b,
                    dev,
                )?)
            };
            let (u, v) = (ids(|e| e.0)?, ids(|e| e.1)?);
            let (h_u, h_v) = (codes.index_select(&u, 0)?, codes.index_select(&v, 0)?);
            let side = |ids: &Tensor| -> anyhow::Result<Stats> {
                let (sums, totals, offsets) = stats.gather(ids)?;
                Ok(Stats {
                    sums,
                    totals,
                    offsets,
                })
            };
            let pooled = side(&u)?.pooled(&side(&v)?)?;
            let (z, _) = self.pair_code(&h_u, &h_v, &pooled)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let gap = self.decrement(&z, &scores, &pooled)?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, &(u, v)) in chunk.iter().enumerate() {
                let total = corpus[u as usize].total + corpus[v as usize].total;
                if total > 0.0 {
                    for j in 0..d {
                        pair_latent[(lb + i, j)] = z_host[i * d + j];
                    }
                    pair_bias[lb + i] = (total.ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                    pair_gap[lb + i] = gap[i];
                }
            }
            bar.inc(b as u64);
        }
        bar.finish_and_clear();

        Ok(Encoded {
            pairs: Placement {
                latent: pair_latent,
                bias: pair_bias,
                gap: pair_gap,
            },
            cells: Placement {
                latent: cell_latent,
                bias: cell_bias,
                gap: cell_gap,
            },
        })
    }
}

///////////////////////////////
// The check against the MAP //
///////////////////////////////

/// A seeded sample of pairs solved exactly once, so any placement of those
/// pairs can be compared with the optimum.
pub(crate) struct ExactCheck {
    /// Pair index, its optimum `θ`, the NLL there, and its norm.
    solved: Vec<(usize, Vec<f32>, f32, f32)>,
    /// Each sampled pair's local profile.
    profiles: Vec<Vec<(u32, f32)>>,
}

/// How far a placement sits from the exact per-pair optimum on the sample.
pub(crate) struct AmortizationGap {
    pub mean_cosine: f32,
    /// Placement NLL over exact NLL, summed over the sample.
    pub nll_ratio: f32,
    /// Median and largest `‖z‖` of the placement and of the solver.
    pub norm_encoder: (f32, f32),
    pub norm_exact: (f32, f32),
}

impl ExactCheck {
    pub(crate) fn new(
        dict: &PairDictionary,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        ridge: f32,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x0047_4150));
        let sample: Vec<usize> = (0..CHECK_PAIRS.min(edges.len()))
            .map(|_| rng.random_range(0..edges.len()))
            .filter(|&e| {
                let (u, v) = edges[e];
                corpus[u as usize].total + corpus[v as usize].total > 0.0
            })
            .collect();
        let profiles: Vec<Vec<(u32, f32)>> = sample
            .par_iter()
            .map(|&e| {
                let (u, v) = edges[e];
                corpus[u as usize].pooled(&corpus[v as usize])
            })
            .collect();
        let solved = sample
            .par_iter()
            .zip(&profiles)
            .map(|(&e, obs)| {
                let (theta, _, _) = super::solve_exact(obs, dict, ridge);
                let norm = theta.iter().map(|v| v * v).sum::<f32>().sqrt();
                let nll = dict.nll(obs, &theta);
                (e, theta, nll, norm)
            })
            .collect();
        Self { solved, profiles }
    }

    pub(crate) fn n_pairs(&self) -> usize {
        self.solved.len()
    }

    /// Compare a placement of every pair (rows in `edges` order) with the
    /// sample's optima. The dictionary is the one the check was built on.
    pub(crate) fn compare(&self, dict: &PairDictionary, latent: &Mat) -> AmortizationGap {
        let per_pair: Vec<(f32, f32, f32)> = self
            .solved
            .par_iter()
            .zip(&self.profiles)
            .map(|((e, theta, _, _), obs)| {
                let z: Vec<f32> = latent.row(*e).iter().copied().collect();
                (
                    cosine(&z, theta),
                    dict.nll(obs, &z),
                    z.iter().map(|v| v * v).sum::<f32>().sqrt(),
                )
            })
            .collect();
        let n = per_pair.len().max(1) as f32;
        let (enc, exact) = per_pair
            .iter()
            .zip(&self.solved)
            .fold((0f64, 0f64), |(a, b), (p, s)| {
                (a + f64::from(p.1), b + f64::from(s.2))
            });
        let median_max = |v: Vec<f32>| {
            let q = quantiles(&v, &[0.5, 1.0]);
            (q[0], q[1])
        };
        AmortizationGap {
            mean_cosine: per_pair.iter().map(|p| p.0).sum::<f32>() / n,
            nll_ratio: if exact != 0.0 {
                (enc / exact) as f32
            } else {
                f32::NAN
            },
            norm_encoder: median_max(per_pair.iter().map(|p| p.2).collect()),
            norm_exact: median_max(self.solved.iter().map(|s| s.3).collect()),
        }
    }
}
