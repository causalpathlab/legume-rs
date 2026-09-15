//! Amortized phase 2: one encoder places every pair, and every cell, on the
//! frozen dictionary.
//!
//! [`super::solve_pair`] solves the same strictly convex problem once per
//! pair — the multinomial MAP of the pooled counts against `e_feat` — and
//! pairs outnumber cells by an order of magnitude. The map from counts to
//! that optimum is smooth, so a small encoder trained on the same objective
//! places every pair in one forward pass, on the device, and the per-pair
//! solver becomes the check on it rather than the workhorse.
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

use super::{PairDictionary, ProjectionArgs, SCORE_CLAMP};
use crate::util::common::*;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, Module};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_util::encoder::{dense_pool, scatter_pool, SymmetricPairHead, SymmetricPairHeadArgs};
use candle_util::feature_embedding::FeatureEmbedding;
use candle_util::vae::{clip_and_step_dense, PhaseTimers};
use candle_util::value_transform::anscombe_residual;
use matrix_util::rand_util::{mix_seed, name_seed};
use rand::rngs::{SmallRng, StdRng};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

///////////////
// Constants //
///////////////

/// The var-name prefix the encoder is saved under.
const VAR_PREFIX: &str = "pair_enc";
/// The statistic moments' tensor names inside the saved file.
const STAT_MEAN_TENSOR: &str = "pair_enc.stat_mean";
const STAT_STD_TENSOR: &str = "pair_enc.stat_std";
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
/// Adam steps the closing check gives the exact solver, with the partition
/// summed exactly: enough to call its answer the optimum.
const CONVERGED_STEPS: usize = 1500;

/// What the encoder arm takes from the command line.
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
    /// Pairs drawn per epoch; `0` walks every pair.
    pub train_pairs: usize,
}

/////////////
// Corpus  //
/////////////

/// One cell's batch-divided counts on the active axis, sorted by position.
pub(crate) struct CellRow {
    pub genes: Vec<u32>,
    pub counts: Vec<f32>,
    pub total: f32,
}

impl CellRow {
    /// From a `(global gene, count)` profile, keeping the genes the dictionary
    /// carries and merging duplicates.
    pub(crate) fn from_profile(dict: &PairDictionary, profile: &[(u32, f32)]) -> Self {
        let mut local = dict.to_local(profile);
        local.sort_unstable_by_key(|&(g, _)| g);
        let mut genes = Vec::with_capacity(local.len());
        let mut counts: Vec<f32> = Vec::with_capacity(local.len());
        for (g, n) in local {
            if genes.last() == Some(&g) {
                *counts.last_mut().unwrap() += n;
            } else {
                genes.push(g);
                counts.push(n);
            }
        }
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

/// Exact multinomial NLL of a local profile at `z`, on the host: the
/// objective both the solver and the encoder minimise, up to the ridge.
pub(crate) fn pair_nll(dict: &PairDictionary, obs: &[(u32, f32)], z: &[f32]) -> f32 {
    let d = dict.d();
    let feat = dict.feat();
    let b = dict.b();
    let (mut max, mut acc) = (f32::NEG_INFINITY, 0f32);
    let mut scores = vec![0f32; b.len()];
    for (g, s) in scores.iter_mut().enumerate() {
        let dot: f32 = feat[g * d..(g + 1) * d]
            .iter()
            .zip(z)
            .map(|(&e, &t)| e * t)
            .sum();
        *s = (dot + b[g]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
        if *s > max {
            acc *= (max - *s).exp();
            max = *s;
        }
        acc += (*s - max).exp();
    }
    let lse = max + acc.ln();
    let (mut total, mut data) = (0f32, 0f32);
    for &(g, n) in obs {
        total += n;
        data += n * scores[g as usize];
    }
    total * lse - data
}

/////////////
// Device  //
/////////////

/// The frozen side on the device: the table for the pool, its transpose for
/// the partition, the log-rate offset and the gate's per-gene mean.
struct DeviceDict {
    features: Arc<FeatureEmbedding>,
    /// `[G × D]`, the table the sufficient statistic pools.
    e_gd: Tensor,
    /// `[D × G]`.
    e_hd: Tensor,
    /// `[1 × G]`.
    b_1d: Tensor,
    /// `[1 × G]`, `exp(b)`: the mean count per cell of each gene.
    mean_1d: Tensor,
    /// `[1 × (D + 1)]` each: the population mean and standard deviation of
    /// the cell statistic, so the trunk reads it standardised.
    stat_mean: Tensor,
    stat_std: Tensor,
    /// `max_g ‖e_g‖`.
    max_row_norm: f32,
    g: usize,
    d: usize,
}

impl DeviceDict {
    /// `stat_mean` / `stat_std` are `[D + 1]` each, from [`stat_moments`].
    fn new(
        dict: &PairDictionary,
        stat_mean: &[f32],
        stat_std: &[f32],
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (g, d) = (dict.n_active(), dict.d());
        anyhow::ensure!(
            stat_mean.len() == d + 1 && stat_std.len() == d + 1,
            "pair encoder: statistic moments have {} and {} entries, expected {}",
            stat_mean.len(),
            stat_std.len(),
            d + 1
        );
        let e_gd = Tensor::from_slice(dict.feat(), (g, d), dev)?;
        Ok(Self {
            e_hd: e_gd.t()?.contiguous()?,
            features: FeatureEmbedding::fixed(e_gd.clone()),
            e_gd,
            b_1d: Tensor::from_slice(dict.b(), (1, g), dev)?,
            mean_1d: Tensor::from_slice(dict.mean(), (1, g), dev)?,
            stat_mean: Tensor::from_slice(stat_mean, (1, d + 1), dev)?,
            stat_std: Tensor::from_slice(stat_std, (1, d + 1), dev)?,
            max_row_norm: dict
                .feat()
                .chunks_exact(d)
                .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
                .fold(0f32, f32::max),
            g,
            d,
        })
    }
}

/// Population mean and standard deviation of every cell's statistic
/// `[Σ_g n_g e_g / N ‖ ln(N + 1)]` over the corpus — `(mean, std)`, `[D + 1]`
/// each, the std floored so a constant coordinate is left alone.
pub(crate) fn stat_moments(dict: &PairDictionary, corpus: &[CellRow]) -> (Vec<f32>, Vec<f32>) {
    let d = dict.d();
    let feat = dict.feat();
    let (sum, sq) = corpus
        .par_iter()
        .fold(
            || (vec![0f64; d + 1], vec![0f64; d + 1]),
            |(mut sum, mut sq), row| {
                let mut m = vec![0f32; d];
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
    fn new(
        dict: &DeviceDict,
        width: usize,
        stat_dim: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let d = dict.d;
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
    /// `2·max_g‖e_g‖ / λ`: the optimum's norm bound per pooled count.
    cap_per_count: f64,
    /// The ridge `λ`, for the gradient bound.
    ridge: f64,
}

/// What the inference pass hands back, in the callers' orders.
pub(crate) struct Encoded {
    /// `[n_pairs × D]`.
    pub pair_latent: Mat,
    pub pair_bias: Vec<f32>,
    /// Per pair, `‖∇‖²/(2λ)` at the placement: an upper bound in nats on
    /// how far its likelihood sits above the optimum's (the objective is
    /// `λ`-strongly convex), zero at the optimum.
    pub pair_gap: Vec<f32>,
    /// `[n_cells × D]`.
    pub cell_latent: Mat,
    pub cell_bias: Vec<f32>,
    pub cell_gap: Vec<f32>,
}

/// What training reports back.
pub(crate) struct TrainStats {
    pub epochs: usize,
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
        this.seed_vars(seed)?;
        Ok(this)
    }

    fn construct(
        dict: DeviceDict,
        trunk_width: usize,
        n_experts: usize,
        ridge: f32,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let cap_per_count = 2.0 * f64::from(dict.max_row_norm) / f64::from(ridge.max(1e-6));
        anyhow::ensure!(
            trunk_width > 0,
            "pair encoder: trunk width must be positive"
        );
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev).pp(VAR_PREFIX);
        // The pooled statistic and the log depth ride beside the attention
        // pool into the trunk, and beside the two codes into the head.
        let stat_dim = dict.d + 1;
        let cell = CellTrunk::new(&dict, trunk_width, stat_dim, vb.pp("cell"))?;
        let head = SymmetricPairHead::new(
            SymmetricPairHeadArgs {
                code_dim: trunk_width,
                out_dim: dict.d,
                n_experts,
                extra_dim: stat_dim,
            },
            vb.pp("head"),
        )?;
        Ok(Self {
            cell,
            head,
            varmap,
            dict,
            cap_per_count,
            ridge: f64::from(ridge.max(1e-6)),
        })
    }

    /// Re-draw every linear weight and the attention query — uniform in
    /// `±1/√fan_in`, biases at zero — from name-keyed sub-streams of `seed`,
    /// so a run replays and adding a var never shifts another's draw. The
    /// layer norms keep their defaults.
    fn seed_vars(&self, seed: u64) -> anyhow::Result<()> {
        let tbl = self.varmap.data().lock().unwrap();
        for (name, var) in tbl.iter() {
            if name.contains(".ln.") {
                continue;
            }
            let dims = var.dims().to_vec();
            let n: usize = dims.iter().product();
            let draw: Vec<f32> = if name.ends_with(".bias") {
                vec![0f32; n]
            } else {
                let bound = (1.0 / *dims.last().unwrap_or(&1) as f64).sqrt();
                let mut rng = StdRng::seed_from_u64(name_seed(seed, name));
                (0..n)
                    .map(|_| ((rng.random::<f64>() * 2.0 - 1.0) * bound) as f32)
                    .collect()
            };
            var.set(&Tensor::from_vec(draw, dims.as_slice(), var.device())?)?;
        }
        Ok(())
    }

    /// Rebuild a saved encoder on `dict`. The widths come from the file: the
    /// expert map is `[K·D, 2L]`, and `D` must be the dictionary's.
    pub(crate) fn load(
        dict: &PairDictionary,
        path: &str,
        ridge: f32,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let tensors = candle_util::candle_core::safetensors::load(path, dev)?;
        let name = format!("{VAR_PREFIX}.head.experts.weight");
        let experts = tensors
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("{path}: no `{name}` tensor"))?;
        let moment = |name: &str| -> anyhow::Result<Vec<f32>> {
            Ok(tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("{path}: no `{name}` tensor"))?
                .flatten_all()?
                .to_vec1()?)
        };
        let (mean, std) = (moment(STAT_MEAN_TENSOR)?, moment(STAT_STD_TENSOR)?);
        let (kd, width) = experts.dims2()?;
        let d = dict.d();
        let stat_dim = d + 1;
        anyhow::ensure!(
            kd.is_multiple_of(d) && width > stat_dim && (width - stat_dim).is_multiple_of(2),
            "{path}: the pair head is [{kd} × {width}], not a mixture over a {d}-dim dictionary"
        );
        let mut this = Self::construct(
            DeviceDict::new(dict, &mean, &std, dev)?,
            (width - stat_dim) / 2,
            kd / d,
            ridge,
            dev,
        )?;
        // Matches by name and ignores the moment tensors, which are not vars.
        this.varmap.load(path)?;
        Ok(this)
    }

    /// Write every var and the statistic moments to one safetensors file.
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
        candle_util::candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub(crate) fn trunk_width(&self) -> usize {
        self.head.code_dim()
    }

    pub(crate) fn n_experts(&self) -> usize {
        self.head.n_experts()
    }

    /// The sufficient statistic of a dense block: the count-weighted sums of
    /// the dictionary rows `[n, D]`, and `[mean ‖ ln(N + 1)]` → `[n, D + 1]`.
    fn statistic(&self, x: &Tensor, totals: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let sums = x.matmul(&self.dict.e_gd)?;
        let stat = self.stat_of(&sums, totals)?;
        Ok((sums, stat))
    }

    /// `[sums / max(N, 1) ‖ ln(N + 1)]` from pooled sums `[n, D]` and depths
    /// `[n]`, standardised by the population moments.
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
        let (sums, stat) = self.statistic(x, totals)?;
        let h = self.cell.forward(&self.dict, x, &stat)?;
        Ok((h, sums))
    }

    /// `‖∇‖²/(2λ)` per row for placements `z` of profiles with pooled
    /// dictionary sums `sums` and depths `n`: `∇ = N(p̄(z) − m) + λz` with
    /// `p̄` the mean dictionary row under the fitted composition and
    /// `m = sums / N`. An upper bound, in nats, on the placement's excess
    /// likelihood over the optimum.
    fn gap_bound(
        &self,
        z: &Tensor,
        scores: &Tensor,
        sums: &Tensor,
        n: &Tensor,
    ) -> anyhow::Result<Vec<f32>> {
        let p = candle_util::candle_nn::ops::softmax(scores, 1)?; // [B, G]
        let n1 = n.unsqueeze(1)?;
        let pred = p.matmul(&self.dict.e_gd)?.broadcast_mul(&n1)?; // N·p̄
        let grad = ((pred - sums)? + z.affine(self.ridge, 0.0)?)?;
        Ok(grad
            .sqr()?
            .sum(1)?
            .affine(1.0 / (2.0 * self.ridge), 0.0)?
            .to_vec1()?)
    }

    /// The pair code of two endpoints: the head's output kept inside the
    /// optimum's ball for the pooled depth `n`.
    fn pair_code(
        &self,
        h_u: &Tensor,
        h_v: &Tensor,
        stat: &Tensor,
        n: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let z = self.head.forward(h_u, h_v, Some(stat))?;
        let cap = n.unsqueeze(1)?.affine(self.cap_per_count, 0.0)?; // [B, 1]
        let norm = z.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1]
                                                     // min(1, cap / ‖z‖), with ‖z‖ = 0 left alone.
        let scale = (cap / (norm + 1e-12)?)?.clamp(0.0, 1.0)?;
        Ok(z.broadcast_mul(&scale)?)
    }

    /// The pair's own statistic from its endpoints' sums and depths.
    fn pair_stat(
        &self,
        sums_u: &Tensor,
        sums_v: &Tensor,
        n_u: &Tensor,
        n_v: &Tensor,
    ) -> anyhow::Result<Tensor> {
        self.stat_of(&(sums_u + sums_v)?, &(n_u + n_v)?)
    }

    /// Scores `z·e_hd + b` → `[n, G]`.
    fn scores(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        Ok(z.matmul(&self.dict.e_hd)?.broadcast_add(&self.dict.b_1d)?)
    }

    /// `N · lse(s) − Σ x·s` per row.
    fn multinomial_nll(x: &Tensor, s: &Tensor, totals: &Tensor) -> anyhow::Result<Tensor> {
        let lse = s.log_sum_exp(1)?;
        let data = (x * s)?.sum(1)?;
        Ok(((totals * lse)? - data)?)
    }

    /// One step's loss on a dense block of `2B` endpoint rows (the `u`s then
    /// the `v`s): the pairs' objective plus the endpoints' own.
    fn step_loss(&self, x: &Tensor, totals: &Tensor, half_ridge: f64) -> anyhow::Result<Tensor> {
        let b = x.dim(0)? / 2;
        let (h, sums) = self.cell_codes(x, totals)?;
        let (h_u, h_v) = (h.narrow(0, 0, b)?, h.narrow(0, b, b)?);
        let (s_u, s_v) = (sums.narrow(0, 0, b)?, sums.narrow(0, b, b)?);
        let (n_u, n_v) = (totals.narrow(0, 0, b)?, totals.narrow(0, b, b)?);
        let big_n = (&n_u + &n_v)?;
        let z_uv = self.pair_code(&h_u, &h_v, &self.pair_stat(&s_u, &s_v, &n_u, &n_v)?, &big_n)?;
        let n_uv = (x.narrow(0, 0, b)? + x.narrow(0, b, b)?)?;
        let nll_uv = Self::multinomial_nll(&n_uv, &self.scores(&z_uv)?, &big_n)?;
        let ridge_uv = z_uv.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;

        let z_uu = self.pair_code(
            &h,
            &h,
            &self.pair_stat(&sums, &sums, totals, totals)?,
            &totals.affine(2.0, 0.0)?,
        )?;
        let nll_uu = Self::multinomial_nll(x, &self.scores(&z_uu)?, totals)?.affine(2.0, 0.0)?;
        let ridge_uu = z_uu.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;

        Ok(((nll_uv + ridge_uv)?.mean_all()? + (nll_uu + ridge_uu)?.mean_all()?)?)
    }

    /// Evaluation-mode NLL per pooled count over `pairs`, and the mean expert
    /// weights: the checkpoint reading, and whether the gate still routes.
    fn evaluate(
        &self,
        corpus: &[CellRow],
        pairs: &[(u32, u32)],
        batch: usize,
    ) -> anyhow::Result<(f32, Vec<f32>)> {
        let dev = self.dict.mean_1d.device();
        let (mut nll, mut count) = (0f64, 0f64);
        let mut pi_sum = vec![0f64; self.head.n_experts()];
        let mut n_rows = 0usize;
        for chunk in pairs.chunks(batch.max(1)) {
            let (x, totals) = self.endpoint_block(corpus, chunk);
            let b = chunk.len();
            let x = Tensor::from_vec(x, (2 * b, self.dict.g), dev)?;
            let totals_t = Tensor::from_slice(&totals, 2 * b, dev)?;
            let (h, sums) = self.cell_codes(&x, &totals_t)?;
            let (h_u, h_v) = (h.narrow(0, 0, b)?, h.narrow(0, b, b)?);
            let (s_u, s_v) = (sums.narrow(0, 0, b)?, sums.narrow(0, b, b)?);
            let (n_u, n_v) = (totals_t.narrow(0, 0, b)?, totals_t.narrow(0, b, b)?);
            let stat = self.pair_stat(&s_u, &s_v, &n_u, &n_v)?;
            let f = self.head.features(&h_u, &h_v, Some(&stat))?;
            let pi: Vec<f32> = self.head.gate(&f)?.sum(0)?.to_vec1()?;
            for (s, p) in pi_sum.iter_mut().zip(&pi) {
                *s += f64::from(*p);
            }
            n_rows += b;
            let big_n = (&n_u + &n_v)?;
            let z_uv = self.pair_code(&h_u, &h_v, &stat, &big_n)?;
            let n_uv = (x.narrow(0, 0, b)? + x.narrow(0, b, b)?)?;
            nll += f64::from(
                Self::multinomial_nll(&n_uv, &self.scores(&z_uv)?, &big_n)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            count += totals.iter().map(|&t| f64::from(t)).sum::<f64>();
        }
        let pi_mean = pi_sum
            .iter()
            .map(|&s| (s / n_rows.max(1) as f64) as f32)
            .collect();
        Ok(((nll / count.max(1.0)) as f32, pi_mean))
    }

    /// The dense `[2B × G]` block of a chunk of pairs: the `u` rows then the
    /// `v` rows, and their totals.
    fn endpoint_block(&self, corpus: &[CellRow], chunk: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
        let rows: Vec<&CellRow> = chunk
            .iter()
            .map(|&(u, _)| &corpus[u as usize])
            .chain(chunk.iter().map(|&(_, v)| &corpus[v as usize]))
            .collect();
        densify(&rows, self.dict.g)
    }

    /// Train on the pairs' likelihood (and the endpoints' own), from the
    /// current vars.
    pub(crate) fn train(
        &self,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        spec: &PairEncoderSpec,
        ridge: f32,
        seed: u64,
    ) -> anyhow::Result<TrainStats> {
        let dev = self.dict.mean_1d.device().clone();
        let n_pairs = edges.len();
        let batch = spec.batch.max(MIN_STEP_PAIRS);
        let half_ridge = f64::from(ridge) / 2.0;
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
            "Pair encoder: {n_pairs} pairs, L={}, K={}, {} epochs of {batch}-pair steps, ridge λ={ridge}; \
             NLL/count before {nll0:.4}",
            self.trunk_width(),
            self.n_experts(),
            spec.epochs
        );

        let per_epoch = if spec.train_pairs == 0 {
            n_pairs
        } else {
            spec.train_pairs.min(n_pairs)
        };
        let mut order: Vec<usize> = (0..n_pairs).collect();
        let mut timers = PhaseTimers::default();
        let (mut steps, mut skipped) = (0usize, 0usize);
        let mut nll_per_count = nll0;
        let bar = new_progress_bar((spec.epochs * per_epoch.div_ceil(batch)) as u64)
            .with_message("pair encoder");
        for epoch in 0..spec.epochs {
            order.shuffle(&mut StdRng::seed_from_u64(mix_seed(seed, epoch as u64)));
            let mut loss_sum = 0f64;
            let mut n_steps = 0usize;
            for chunk in order[..per_epoch].chunks(batch) {
                if chunk.len() < MIN_STEP_PAIRS {
                    bar.inc(1);
                    continue;
                }
                let pairs: Vec<(u32, u32)> = chunk.iter().map(|&i| edges[i]).collect();
                let t = std::time::Instant::now();
                let (x, totals) = self.endpoint_block(corpus, &pairs);
                let b = pairs.len();
                let x = Tensor::from_vec(x, (2 * b, self.dict.g), &dev)?;
                let totals = Tensor::from_slice(&totals, 2 * b, &dev)?;
                timers.precompute += t.elapsed();

                let t = std::time::Instant::now();
                let loss = self.step_loss(&x, &totals, half_ridge)?;
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
            epochs: spec.epochs,
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
        let (n_cells, g, d, l) = (corpus.len(), self.dict.g, self.dict.d, self.trunk_width());

        // Cells: codes and dictionary sums kept on the host for the pair
        // pass, self-pairs written out.
        let mut codes = vec![0f32; n_cells * l];
        let mut sums_host = vec![0f32; n_cells * d];
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
            let z = self.pair_code(
                &h,
                &h,
                &self.pair_stat(&sums, &sums, &totals_t, &totals_t)?,
                &totals_t.affine(2.0, 0.0)?,
            )?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            // The self-pair's objective is over the doubled profile.
            let gap = self.gap_bound(
                &z,
                &scores,
                &sums.affine(2.0, 0.0)?,
                &totals_t.affine(2.0, 0.0)?,
            )?;
            codes[lb * l..ub * l].copy_from_slice(&h.flatten_all()?.to_vec1::<f32>()?);
            sums_host[lb * d..ub * d].copy_from_slice(&sums.flatten_all()?.to_vec1::<f32>()?);
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
            bar.inc((ub - lb) as u64);
        }
        bar.finish_and_clear();

        // Pairs: two gathered codes per pair through the head.
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
            // Rows of a host table gathered by endpoint.
            let gather = |table: &[f32], w: usize, pick: fn(&(u32, u32)) -> u32| -> Vec<f32> {
                let mut out = vec![0f32; b * w];
                out.par_chunks_mut(w)
                    .zip(chunk.par_iter())
                    .for_each(|(dst, e)| {
                        let c = pick(e) as usize;
                        dst.copy_from_slice(&table[c * w..(c + 1) * w]);
                    });
                out
            };
            let h_u = Tensor::from_vec(gather(&codes, l, |e| e.0), (b, l), &dev)?;
            let h_v = Tensor::from_vec(gather(&codes, l, |e| e.1), (b, l), &dev)?;
            let s_u = Tensor::from_vec(gather(&sums_host, d, |e| e.0), (b, d), &dev)?;
            let s_v = Tensor::from_vec(gather(&sums_host, d, |e| e.1), (b, d), &dev)?;
            let depth = |pick: fn(&(u32, u32)) -> u32| -> Vec<f32> {
                chunk
                    .iter()
                    .map(|e| corpus[pick(e) as usize].total)
                    .collect()
            };
            let n_u = Tensor::from_vec(depth(|e| e.0), b, &dev)?;
            let n_v = Tensor::from_vec(depth(|e| e.1), b, &dev)?;
            let z = self.pair_code(
                &h_u,
                &h_v,
                &self.pair_stat(&s_u, &s_v, &n_u, &n_v)?,
                &(&n_u + &n_v)?,
            )?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let gap = self.gap_bound(&z, &scores, &(&s_u + &s_v)?, &(&n_u + &n_v)?)?;
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
            pair_latent,
            pair_bias,
            pair_gap,
            cell_latent,
            cell_bias,
            cell_gap,
        })
    }
}

///////////////////////////////
// The check against the MAP //
///////////////////////////////

/// How far the amortized placement sits from the exact per-pair optimum on a
/// seeded sample of pairs.
pub(crate) struct AmortizationGap {
    pub n_pairs: usize,
    pub mean_cosine: f32,
    /// Encoder NLL over exact NLL, summed over the sample.
    pub nll_ratio: f32,
    /// Mean cosine between the solver's own answer at the run's step budget
    /// and its converged answer: how well the direction is determined at all.
    pub solver_self_cosine: f32,
    /// Median and largest `‖z‖` from the encoder and from the solver.
    pub norm_encoder: (f32, f32),
    pub norm_exact: (f32, f32),
}

/// One checked pair: cosine to the optimum, encoder NLL, exact NLL, the
/// solver's budget-vs-converged cosine, encoder norm, exact norm.
type PairCheck = (f32, f32, f32, f32, f32, f32);

/// `(median, max)` of a sample of norms.
fn median_max(mut v: Vec<f32>) -> (f32, f32) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match v.len() {
        0 => (0.0, 0.0),
        n => (v[n / 2], v[n - 1]),
    }
}

/// Solve a seeded sample of pairs exactly (exhaustive partition) and compare
/// the encoder's placement with the optimum.
pub(crate) fn amortization_gap(
    dict: &PairDictionary,
    corpus: &[CellRow],
    edges: &[(u32, u32)],
    encoded: &Mat,
    args: &ProjectionArgs,
    seed: u64,
) -> AmortizationGap {
    let n = CHECK_PAIRS.min(edges.len());
    let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x0047_4150));
    let sample: Vec<usize> = (0..n).map(|_| rng.random_range(0..edges.len())).collect();
    let budget = ProjectionArgs {
        ridge: args.ridge,
        steps: args.steps.max(1),
        gene_sample: 0,
    };
    let converged = ProjectionArgs {
        ridge: args.ridge,
        steps: CONVERGED_STEPS.max(budget.steps),
        gene_sample: 0,
    };
    let cosine = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            1.0
        }
    };
    let norm = |a: &[f32]| a.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Per pair: cosine to the optimum, encoder NLL, exact NLL, the solver's
    // own budget-vs-converged cosine, encoder norm, exact norm.
    let per_pair: Vec<PairCheck> = sample
        .par_iter()
        .map(|&e| {
            let (u, v) = edges[e];
            let obs = corpus[u as usize].pooled(&corpus[v as usize]);
            let mut rng = SmallRng::seed_from_u64(mix_seed(seed, e as u64));
            let (theta, _) = super::solve_pair(&obs, dict, &converged, None, &mut rng);
            let mut rng = SmallRng::seed_from_u64(mix_seed(seed, e as u64));
            let (theta_budget, _) = super::solve_pair(&obs, dict, &budget, None, &mut rng);
            let z: Vec<f32> = encoded.row(e).iter().copied().collect();
            (
                cosine(&z, &theta),
                pair_nll(dict, &obs, &z),
                pair_nll(dict, &obs, &theta),
                cosine(&theta_budget, &theta),
                norm(&z),
                norm(&theta),
            )
        })
        .collect();
    let mean = |f: &dyn Fn(&PairCheck) -> f32| -> f32 {
        per_pair.iter().map(f).sum::<f32>() / n.max(1) as f32
    };
    let (enc, exact) = per_pair.iter().fold((0f64, 0f64), |(a, b), p| {
        (a + f64::from(p.1), b + f64::from(p.2))
    });
    AmortizationGap {
        n_pairs: n,
        mean_cosine: mean(&|p| p.0),
        nll_ratio: if exact != 0.0 {
            (enc / exact) as f32
        } else {
            f32::NAN
        },
        solver_self_cosine: mean(&|p| p.3),
        norm_encoder: median_max(per_pair.iter().map(|p| p.4).collect()),
        norm_exact: median_max(per_pair.iter().map(|p| p.5).collect()),
    }
}
