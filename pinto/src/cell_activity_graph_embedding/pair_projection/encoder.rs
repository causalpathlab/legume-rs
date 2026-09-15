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
//! The encoder is the `senna bge` phase-2 read applied twice: a cell trunk
//! reads each endpoint's counts through the dictionary (attention over genes
//! on an Anscombe gate, then a small ReLU stack and a linear head) into a
//! cell code `h`, and a [`SymmetricPairHead`] — a gated mixture of linear
//! experts over `[(h_u + h_v)/2 ‖ h_u ⊙ h_v]` — turns two codes into the
//! pair code `z_uv`. Symmetric by construction, and with no per-gene
//! parameter: the only gene-sized objects are the dictionary and the gate's
//! per-gene mean, so the trained trunk transfers to any re-aligned axis. The
//! trunk normalises per row (layer norm), never per batch: a batch norm's
//! running variance for a unit that rarely fires sits near zero, and the
//! rare input that fires it is then blown up at inference — the wild
//! placement a shared map must never produce. And the placement is kept
//! inside the ball the optimum provably lies in: at the optimum
//! `λθ = N(m − p̄)` with both means convex combinations of dictionary rows,
//! so `‖θ‖ ≤ 2N·max_g‖e_g‖/λ` — binding only for a row with a handful of
//! counts, exactly where a shared map would otherwise extrapolate.
//!
//! Every placement also leaves with a certificate: the objective is
//! `λ`-strongly convex, so `‖∇‖²/(2λ)` at the placement bounds, in nats, how
//! far its likelihood sits above the optimum's. The bound is loose (it knows
//! `λ`, not the curvature `N·Cov`), but a row it puts far out is a row the
//! map extrapolated on — a rare composition, a near-empty profile — and the
//! caller finishes those few exactly.
//!
//! The MAP is a function of one statistic: the count-weighted mean of the
//! dictionary rows, `Σ_g n_g e_g / N`, and the depth `N`. The attention pool
//! is a different summary of the same counts, so the trunk also takes that
//! statistic directly, and the pair head takes the pair's own — the two
//! endpoints' sums pooled, which is additive and therefore symmetric. What
//! the trained layers add is the nonlinearity from statistic to optimum.
//!
//! ```text
//! s_uv,g = ⟨e_g, z_uv⟩ + b_g
//! L_uv   = N_uv · lse_g(s_uv,g) − Σ_g n_uv,g s_uv,g + (λ/2)‖z_uv‖²        n_uv = x_u + x_v
//! L_uu   = 2 · [N_u · lse_g(s_uu,g) − Σ_g x_u,g s_uu,g] + (λ/2)‖z_uu‖²     the self-pair
//! ```
//!
//! The partition is summed exactly over the active axis — one `[B, D]·[D, G]`
//! matmul — so nothing is sampled, and `β_uv` is never learned: given `z` the
//! intercept is `ln N − lse(s)` in closed form. The self-pair term is what
//! makes the cell's own placement a trained quantity: `z_uu` is the MAP of the
//! doubled profile `2x_u`, whose composition is `x_u`'s, so
//! `.cell_embedding.parquet` comes from the same map as every pair. A row with
//! no counts is put at the origin afterwards, where the solver puts it: the
//! gate would otherwise give an empty row a constant code.

use super::{PairDictionary, SCORE_CLAMP};
use crate::util::common::*;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{
    layer_norm, linear, AdamW, LayerNorm, LayerNormConfig, Linear, Module, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};
use candle_util::encoder::{dense_pool, scatter_pool, SymmetricPairHead, SymmetricPairHeadArgs};
use candle_util::feature_embedding::FeatureEmbedding;
use candle_util::loss::multinomial_nll_profiled;
use candle_util::nn::seed_uniform_vars;
use candle_util::vae::{clip_and_step_dense, PhaseTimers};
use candle_util::value_transform::anscombe_residual;
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
/// A step with fewer pairs than this is skipped: the gate standardises over
/// the block, and needs a block to do it over.
const MIN_STEP_PAIRS: usize = 16;
/// Cells encoded per forward pass at inference; the dense block is this many
/// rows by the active axis.
pub(crate) const CELL_BLOCK: usize = 4096;
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
    /// Passes over the pairs.
    pub epochs: usize,
    /// Pairs per optimizer step.
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

////////////
// Corpus //
////////////

/// One cell's batch-divided counts on the active axis, sorted by position.
#[derive(Clone)]
pub(crate) struct CellRow {
    pub genes: Vec<u32>,
    pub counts: Vec<f32>,
    pub total: f32,
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
        let total = counts.iter().sum();
        Self {
            genes,
            counts,
            total,
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

/// Scatter rows into a dense `[n × g]` row-major buffer, in parallel over the
/// rows, returning each row's total.
fn densify(rows: &[&CellRow], g: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0f32; rows.len() * g];
    let totals: Vec<f32> = x
        .par_chunks_mut(g)
        .zip(rows.par_iter())
        .map(|(dst, row)| {
            for (&p, &c) in row.genes.iter().zip(&row.counts) {
                dst[p as usize] += c;
            }
            row.total
        })
        .collect();
    (x, totals)
}

/// Population mean and standard deviation of every cell's statistic
/// `[Σ_g n_g e_g / N ‖ ln(N + 1)]` over the corpus — `(mean, std)`, `[D + 1]`
/// each, the std floored so a constant coordinate is left alone.
fn stat_moments(dict: &PairDictionary, corpus: &[CellRow]) -> (Vec<f32>, Vec<f32>) {
    let d = dict.d;
    let feat = &dict.feat;
    let (sum, sq) = corpus
        .par_iter()
        .fold(
            || (vec![0f64; d + 1], vec![0f64; d + 1], vec![0f32; d]),
            |(mut sum, mut sq, mut m), row| {
                m.fill(0.0);
                for (&g, &n) in row.genes.iter().zip(&row.counts) {
                    let e = &feat[g as usize * d..(g as usize + 1) * d];
                    for (acc, &v) in m.iter_mut().zip(e) {
                        *acc += n * v;
                    }
                }
                let n = row.total.max(1.0);
                for (j, &v) in m.iter().enumerate() {
                    let v = f64::from(v / n);
                    sum[j] += v;
                    sq[j] += v * v;
                }
                let ln_n = f64::from((row.total + 1.0).ln());
                sum[d] += ln_n;
                sq[d] += ln_n * ln_n;
                (sum, sq, m)
            },
        )
        .map(|(sum, sq, _)| (sum, sq))
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
/// certificate — `‖∇‖²/(2λ)` at the placement, an upper bound in nats on how
/// far its likelihood sits above the optimum's (the objective is
/// `λ`-strongly convex), zero at the optimum.
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

/// The frozen side on the device: the table for the pool and the statistic,
/// its transpose for the partition, the log-rate offset, the gate's per-gene
/// mean and the statistic's population moments.
struct DeviceDict {
    features: Arc<FeatureEmbedding>,
    /// `[G × D]`.
    e_gd: Tensor,
    /// `[D × G]`.
    e_hd: Tensor,
    /// `[1 × G]`.
    b_1d: Tensor,
    /// `[1 × G]`, `exp(b)`: the mean count per cell of each gene.
    mean_1d: Tensor,
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
        Ok(Self {
            e_hd: e_gd.t()?.contiguous()?,
            features: FeatureEmbedding::fixed(e_gd.clone()),
            e_gd,
            b_1d: Tensor::from_slice(&dict.b, (1, g), dev)?,
            mean_1d: Tensor::from_slice(&dict.mean, (1, g), dev)?,
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
}

/////////////////
// The encoder //
/////////////////

/// Counts → cell code: attention over genes through the dictionary on an
/// Anscombe gate, joined by the sufficient statistic, through two
/// layer-normed ReLU layers and a linear head. No per-gene parameter.
struct CellTrunk {
    /// `[1, D]`, the attention query over the dictionary.
    attn_query: Tensor,
    /// `[D + stat] → L` and `L → L`, each followed by layer norm and ReLU.
    layers: Vec<(Linear, LayerNorm)>,
    /// `L → L`, linear: the code is signed.
    head: Linear,
}

impl CellTrunk {
    fn new(d: usize, width: usize, stat_dim: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let attn_query = vb.get_with_hints(
            (1, d),
            "attn.query",
            candle_util::candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let mut layers = Vec::with_capacity(2);
        let mut in_dim = d + stat_dim;
        for k in 0..2 {
            let lin = linear(in_dim, width, vb.pp(format!("fc.{k}")))?;
            let norm = layer_norm(width, LayerNormConfig::default(), vb.pp(format!("ln.{k}")))?;
            layers.push((lin, norm));
            in_dim = width;
        }
        Ok(Self {
            attn_query,
            layers,
            head: linear(width, width, vb.pp("head"))?,
        })
    }

    /// Attention-pool a dense `[n, G]` block over the genes → `[n, D]`.
    fn pool(&self, dict: &DeviceDict, x: &Tensor) -> anyhow::Result<Tensor> {
        let scale = 1.0 / (dict.d as f64).sqrt();
        let gate = anscombe_residual(x, None, Some(&dict.mean_1d))?;
        let rq_d = scatter_pool::query_over_features(&dict.features, &self.attn_query)?;
        let scores = dense_pool::attention_scores_dense(&gate, &rq_d, None, scale)?;
        let attn = candle_util::candle_nn::ops::softmax(&scores, 1)?;
        Ok(dense_pool::pool_dense(&attn, &gate, &dict.features)?)
    }

    /// `[pool ‖ stat]` → code `[n, L]`.
    fn forward(&self, dict: &DeviceDict, x: &Tensor, stat: &Tensor) -> anyhow::Result<Tensor> {
        let pooled = self.pool(dict, x)?;
        let mut h = Tensor::cat(&[&pooled, stat], 1)?;
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

/// One block of endpoint rows (the `u`s then the `v`s) through the trunk and
/// the head: what training, evaluation and inference all read.
struct PairForward {
    /// `[2B, L]` cell codes.
    h: Tensor,
    /// `[2B, D]` dictionary sums per endpoint.
    sums: Tensor,
    /// `[B, 2L + stat]` pair features.
    features: Tensor,
    /// `[B, D]` pair codes.
    z_uv: Tensor,
    /// `[B, G]` pooled counts and `[B]` pooled depths.
    n_uv: Tensor,
    big_n: Tensor,
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
        let (mean, std) = stat_moments(dict, corpus);
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
        // The pooled statistic and the log depth ride beside the attention
        // pool into the trunk, and beside the two codes into the head.
        let stat_dim = dict.d + 1;
        let cell = CellTrunk::new(dict.d, trunk_width, stat_dim, vb.pp("cell"))?;
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
        let dev = self.dict.mean_1d.device();
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
        tensors.insert(HPARAMS_TENSOR.to_string(), Tensor::new(&hparams, dev)?);
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

    /// `[sums / max(N, 1) ‖ ln(N + 1)]` from dictionary sums `[n, D]` and
    /// depths `[n]`, standardised by the population moments.
    fn stat_of(&self, sums: &Tensor, totals: &Tensor) -> anyhow::Result<Tensor> {
        let n1 = totals.unsqueeze(1)?;
        let mean = sums.broadcast_div(&n1.clamp(1.0, f64::INFINITY)?)?;
        let ln_n = (&n1 + 1.0)?.log()?;
        let raw = Tensor::cat(&[&mean, &ln_n], 1)?;
        Ok(raw
            .broadcast_sub(&self.dict.stat_mean)?
            .broadcast_div(&self.dict.stat_std)?)
    }

    /// Cell codes of a dense block → `[n, L]`, with the block's dictionary
    /// sums `[n, D]` for the pair statistic.
    fn cell_codes(&self, x: &Tensor, totals: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let sums = x.matmul(&self.dict.e_gd)?;
        let stat = self.stat_of(&sums, totals)?;
        let h = self.cell.forward(&self.dict, x, &stat)?;
        Ok((h, sums))
    }

    /// The pair code of two endpoints from their codes, dictionary sums and
    /// depths: the head's output kept inside the optimum's ball for the
    /// pooled depth. Also returns the pair features, for the gate readout.
    fn pair_code(
        &self,
        h_u: &Tensor,
        h_v: &Tensor,
        s_u: &Tensor,
        s_v: &Tensor,
        n_u: &Tensor,
        n_v: &Tensor,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let big_n = (n_u + n_v)?;
        let stat = self.stat_of(&(s_u + s_v)?, &big_n)?;
        let features = self.head.features(h_u, h_v, Some(&stat))?;
        let z = self.head.forward(h_u, h_v, Some(&stat))?;
        let cap = big_n.unsqueeze(1)?.affine(self.cap_per_count, 0.0)?; // [B, 1]
        let norm = z.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1]
                                                     // min(1, cap / ‖z‖), with ‖z‖ = 0 left alone.
        let scale = (cap / (norm + 1e-12)?)?.clamp(0.0, 1.0)?;
        Ok((z.broadcast_mul(&scale)?, features))
    }

    /// The self-pair of every row of a block: its code with itself, at the
    /// doubled depth.
    fn self_pair_code(&self, h: &Tensor, sums: &Tensor, totals: &Tensor) -> anyhow::Result<Tensor> {
        Ok(self.pair_code(h, h, sums, sums, totals, totals)?.0)
    }

    /// A block of `2B` endpoint rows (the `u`s then the `v`s) through the
    /// trunk and the head.
    fn pair_forward(&self, x: &Tensor, totals: &Tensor) -> anyhow::Result<PairForward> {
        let b = x.dim(0)? / 2;
        let (h, sums) = self.cell_codes(x, totals)?;
        let (h_u, h_v) = (h.narrow(0, 0, b)?, h.narrow(0, b, b)?);
        let (s_u, s_v) = (sums.narrow(0, 0, b)?, sums.narrow(0, b, b)?);
        let (n_u, n_v) = (totals.narrow(0, 0, b)?, totals.narrow(0, b, b)?);
        let (z_uv, features) = self.pair_code(&h_u, &h_v, &s_u, &s_v, &n_u, &n_v)?;
        Ok(PairForward {
            n_uv: (x.narrow(0, 0, b)? + x.narrow(0, b, b)?)?,
            big_n: (n_u + n_v)?,
            h,
            sums,
            features,
            z_uv,
        })
    }

    /// Scores `z·e_hd + b` → `[n, G]`.
    fn scores(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        Ok(z.matmul(&self.dict.e_hd)?.broadcast_add(&self.dict.b_1d)?)
    }

    /// `‖∇‖²/(2λ)` per row for placements `z` of profiles with pooled
    /// dictionary sums `sums` and depths `n`: `∇ = N(p̄(z) − m) + λz` with
    /// `p̄` the mean dictionary row under the fitted composition and
    /// `m = sums / N`.
    fn certificate(
        &self,
        z: &Tensor,
        scores: &Tensor,
        sums: &Tensor,
        n: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let p = candle_util::candle_nn::ops::softmax(scores, 1)?; // [B, G]
        let n1 = n.unsqueeze(1)?;
        let pred = p.matmul(&self.dict.e_gd)?.broadcast_mul(&n1)?; // N·p̄
        let grad = ((pred - sums)? + z.affine(f64::from(self.ridge), 0.0)?)?;
        Ok(grad
            .sqr()?
            .sum(1)?
            .affine(1.0 / (2.0 * f64::from(self.ridge)), 0.0)?)
    }

    /// One step's loss on a dense block of `2B` endpoint rows: the pairs'
    /// objective plus the endpoints' own, each with its ridge.
    fn step_loss(&self, x: &Tensor, totals: &Tensor) -> anyhow::Result<Tensor> {
        let half_ridge = f64::from(self.ridge) / 2.0;
        let fwd = self.pair_forward(x, totals)?;
        let nll_uv = multinomial_nll_profiled(&fwd.n_uv, &self.scores(&fwd.z_uv)?, &fwd.big_n)?;
        let ridge_uv = fwd.z_uv.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;
        let z_uu = self.self_pair_code(&fwd.h, &fwd.sums, totals)?;
        let nll_uu = multinomial_nll_profiled(x, &self.scores(&z_uu)?, totals)?.affine(2.0, 0.0)?;
        let ridge_uu = z_uu.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;
        Ok(((nll_uv + ridge_uv)?.mean_all()? + (nll_uu + ridge_uu)?.mean_all()?)?)
    }

    /// The dense `[2B × G]` block of a chunk of pairs: the `u` rows then the
    /// `v` rows, and their totals, on the device.
    fn endpoint_block(
        &self,
        corpus: &[CellRow],
        chunk: &[(u32, u32)],
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let rows: Vec<&CellRow> = chunk
            .iter()
            .map(|&(u, _)| &corpus[u as usize])
            .chain(chunk.iter().map(|&(_, v)| &corpus[v as usize]))
            .collect();
        let (x, totals) = densify(&rows, self.dict.g);
        let dev = self.dict.mean_1d.device();
        Ok((
            Tensor::from_vec(x, (rows.len(), self.dict.g), dev)?,
            Tensor::from_vec(totals, rows.len(), dev)?,
        ))
    }

    /// NLL per pooled count over `pairs`, and the mean expert weights: the
    /// checkpoint reading, and whether the gate still routes.
    fn evaluate(
        &self,
        corpus: &[CellRow],
        pairs: &[(u32, u32)],
        batch: usize,
    ) -> anyhow::Result<(f32, Vec<f32>)> {
        let (mut nll, mut count) = (0f64, 0f64);
        let mut pi_sum = vec![0f64; self.head.n_experts()];
        for chunk in pairs.chunks(batch.max(1)) {
            let (x, totals) = self.endpoint_block(corpus, chunk)?;
            let fwd = self.pair_forward(&x, &totals)?;
            let pi: Vec<f32> = self.head.gate(&fwd.features)?.sum(0)?.to_vec1()?;
            for (s, p) in pi_sum.iter_mut().zip(&pi) {
                *s += f64::from(*p);
            }
            nll += f64::from(
                multinomial_nll_profiled(&fwd.n_uv, &self.scores(&fwd.z_uv)?, &fwd.big_n)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            count += f64::from(fwd.big_n.sum_all()?.to_scalar::<f32>()?);
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
        let batch = spec.batch.max(MIN_STEP_PAIRS);
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
        let (nll0, _) = self.evaluate(corpus, &check, batch)?;
        info!(
            "Pair encoder: {n_pairs} pairs, L={}, K={}, {} epochs of {batch}-pair steps, ridge λ={}; \
             NLL/count before {nll0:.4}",
            self.trunk_width(),
            self.n_experts(),
            spec.epochs,
            self.ridge
        );

        let mut order: Vec<usize> = (0..n_pairs).collect();
        let mut timers = PhaseTimers::default();
        let (mut steps, mut skipped) = (0usize, 0usize);
        let mut nll_per_count = nll0;
        let bar = new_progress_bar((spec.epochs * n_pairs.div_ceil(batch)) as u64)
            .with_message("pair encoder");
        for epoch in 0..spec.epochs {
            order.shuffle(&mut StdRng::seed_from_u64(mix_seed(seed, epoch as u64)));
            let mut loss_sum = 0f64;
            let mut n_steps = 0usize;
            for chunk in order.chunks(batch) {
                if chunk.len() < MIN_STEP_PAIRS {
                    bar.inc(1);
                    continue;
                }
                let pairs: Vec<(u32, u32)> = chunk.iter().map(|&i| edges[i]).collect();
                let t = std::time::Instant::now();
                let (x, totals) = self.endpoint_block(corpus, &pairs)?;
                timers.precompute += t.elapsed();

                let t = std::time::Instant::now();
                let loss = self.step_loss(&x, &totals)?;
                timers.decoder_fwd += t.elapsed();

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
            let (nll, pi) = self.evaluate(corpus, &check, batch)?;
            nll_per_count = nll;
            info!(
                "Pair encoder epoch {}/{}: mean step loss {:.2}, held-out NLL/count {:.4}, expert usage {}",
                epoch + 1,
                spec.epochs,
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
    /// cells share one forward pass.
    pub(crate) fn encode_all(
        &self,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        pair_block: usize,
        cell_block: usize,
    ) -> anyhow::Result<Encoded> {
        let dev = self.dict.mean_1d.device().clone();
        let (n_cells, g, d) = (corpus.len(), self.dict.g, self.dict.d);

        // Cells: the codes, dictionary sums and depths stay on the device for
        // the pair pass; the self-pairs are written out.
        let mut codes = Vec::new();
        let mut sums_all = Vec::new();
        let mut cell_latent = Mat::zeros(n_cells, d);
        let mut cell_bias = vec![0f32; n_cells];
        let mut cell_gap = vec![0f32; n_cells];
        let bar = new_progress_bar(n_cells as u64).with_message("encoding cells");
        for (lb, ub) in
            matrix_util::utils::generate_minibatch_intervals(n_cells, 0, Some(cell_block.max(1)))
        {
            let rows: Vec<&CellRow> = corpus[lb..ub].iter().collect();
            let (x, totals) = densify(&rows, g);
            let x = Tensor::from_vec(x, (ub - lb, g), &dev)?;
            let totals_t = Tensor::from_slice(&totals, ub - lb, &dev)?;
            let (h, sums) = self.cell_codes(&x, &totals_t)?;
            let z = self.self_pair_code(&h, &sums, &totals_t)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            // The self-pair's objective is over the doubled profile.
            let gap: Vec<f32> = self
                .certificate(
                    &z,
                    &scores,
                    &sums.affine(2.0, 0.0)?,
                    &totals_t.affine(2.0, 0.0)?,
                )?
                .to_vec1()?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, c) in (lb..ub).enumerate() {
                if totals[i] > 0.0 {
                    for j in 0..d {
                        cell_latent[(c, j)] = z_host[i * d + j];
                    }
                    cell_bias[c] = (totals[i].ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                    cell_gap[c] = gap[i];
                }
            }
            codes.push(h);
            sums_all.push(sums);
            bar.inc((ub - lb) as u64);
        }
        bar.finish_and_clear();
        let codes = Tensor::cat(&codes, 0)?;
        let sums_all = Tensor::cat(&sums_all, 0)?;
        let totals_all = Tensor::from_vec(
            corpus.iter().map(|r| r.total).collect::<Vec<f32>>(),
            n_cells,
            &dev,
        )?;

        // Pairs: the two endpoints' codes gathered on the device, through the head.
        let n_pairs = edges.len();
        let mut pair_latent = Mat::zeros(n_pairs, d);
        let mut pair_bias = vec![0f32; n_pairs];
        let mut pair_gap = vec![0f32; n_pairs];
        let bar = new_progress_bar(n_pairs as u64).with_message("encoding pairs");
        for (lb, ub) in
            matrix_util::utils::generate_minibatch_intervals(n_pairs, 0, Some(pair_block.max(1)))
        {
            let chunk = &edges[lb..ub];
            let b = chunk.len();
            let ids = |pick: fn(&(u32, u32)) -> u32| -> anyhow::Result<Tensor> {
                Ok(Tensor::from_vec(
                    chunk.iter().map(pick).collect::<Vec<u32>>(),
                    b,
                    &dev,
                )?)
            };
            let (u, v) = (ids(|e| e.0)?, ids(|e| e.1)?);
            let (h_u, h_v) = (codes.index_select(&u, 0)?, codes.index_select(&v, 0)?);
            let (s_u, s_v) = (sums_all.index_select(&u, 0)?, sums_all.index_select(&v, 0)?);
            let (n_u, n_v) = (
                totals_all.index_select(&u, 0)?,
                totals_all.index_select(&v, 0)?,
            );
            let (z, _) = self.pair_code(&h_u, &h_v, &s_u, &s_v, &n_u, &n_v)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let gap: Vec<f32> = self
                .certificate(&z, &scores, &(&s_u + &s_v)?, &(&n_u + &n_v)?)?
                .to_vec1()?;
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
