//! Localized attention from genes to their cis peaks: the learned links.
//!
//! ```text
//! s_gp = −γ · ln(d_gp + c) + (ρ_g A) · (φ_p B)       A, B: [H × r]
//! π_gp = softmax_{p ∈ cis(g)} s_gp
//! ψ_g  = Σ_p π_gp φ_p
//! L    = mean_g ‖ρ_g − ψ_g‖²
//! ```
//!
//! `ρ` (gene rows) and `φ` (peak rows) are frozen; only the low-rank content
//! map `A Bᵀ` and the distance kernel's `γ` and `c` train. A step takes a batch
//! of genes, gathers their cis pairs once, and runs a segmented softmax per
//! gene, so memory is `pairs in the batch × H` and nothing is `genes × peaks`.

use super::cis::CisPairs;
use crate::common::*;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Rank `r` of the content map (capped at `H`).
    pub rank: usize,
    pub epochs: usize,
    pub genes_per_step: usize,
    pub learning_rate: f64,
    pub seed: u64,
    /// Scale of the random `A`, `B` start; `0` starts at the distance prior
    /// alone (and, `A = B = 0` being a saddle, never leaves it).
    pub init_scale: f64,
    pub init_gamma: f64,
    pub init_pseudocount: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            epochs: 100,
            genes_per_step: 512,
            learning_rate: 0.01,
            seed: 42,
            init_scale: 0.1,
            init_gamma: 1.0,
            init_pseudocount: 5_000.0,
        }
    }
}

/// The trainable part: content map factors and log kernel constants.
pub struct LocalAttention {
    a: Var,
    b: Var,
    log_gamma: Var,
    log_c: Var,
}

/// One batch of genes' pairs, gathered.
struct Batch {
    /// Pair index into `CisPairs` for every gathered pair, in order.
    pair: Vec<usize>,
    /// Batch-local gene of each pair.
    local: Vec<u32>,
    peak: Vec<u32>,
    dist: Vec<f64>,
    /// `CisPairs` gene index of each batch-local gene.
    genes: Vec<u32>,
}

impl Batch {
    fn new(pairs: &CisPairs, genes: &[usize]) -> Self {
        let mut b = Self {
            pair: Vec::new(),
            local: Vec::new(),
            peak: Vec::new(),
            dist: Vec::new(),
            genes: Vec::new(),
        };
        for &g in genes {
            let r = pairs.gene(g);
            if r.is_empty() {
                continue;
            }
            let l = b.genes.len() as u32;
            b.genes.push(g as u32);
            for k in r {
                b.pair.push(k);
                b.local.push(l);
                b.peak.push(pairs.peak[k]);
                b.dist.push(pairs.dist[k] as f64);
            }
        }
        b
    }
}

impl LocalAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        h: usize,
        rank: usize,
        init_scale: f64,
        init_gamma: f64,
        init_pseudocount: f64,
        dtype: DType,
        dev: &Device,
        seed: u64,
    ) -> anyhow::Result<Self> {
        let r = rank.clamp(1, h);
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, init_scale.max(0.0) / (h as f64).sqrt())?;
        let mut factor = || -> anyhow::Result<Var> {
            let v: Vec<f64> = (0..h * r).map(|_| normal.sample(&mut rng)).collect();
            Ok(Var::from_tensor(
                &Tensor::from_vec(v, (h, r), dev)?.to_dtype(dtype)?,
            )?)
        };
        let (a, b) = (factor()?, factor()?);
        let scalar = |x: f64| -> anyhow::Result<Var> {
            Ok(Var::from_tensor(&Tensor::new(x, dev)?.to_dtype(dtype)?)?)
        };
        Ok(Self {
            a,
            b,
            log_gamma: scalar(init_gamma.ln())?,
            log_c: scalar(init_pseudocount.ln())?,
        })
    }

    /// The trained variables: `A`, `B`, `ln γ`, `ln c`.
    #[must_use]
    pub fn vars(&self) -> Vec<Var> {
        vec![
            self.a.clone(),
            self.b.clone(),
            self.log_gamma.clone(),
            self.log_c.clone(),
        ]
    }

    #[must_use]
    pub fn gamma(&self) -> f64 {
        scalar_f64(self.log_gamma.as_tensor()).exp()
    }

    #[must_use]
    pub fn pseudocount(&self) -> f64 {
        scalar_f64(self.log_c.as_tensor()).exp()
    }

    /// Shares `π` `[pairs]` and pooled rows `ψ` `[genes × H]` of one batch.
    fn forward(
        &self,
        rho: &Tensor,
        phi: &Tensor,
        batch: &Batch,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let dev = rho.device();
        let dtype = rho.dtype();
        let n_genes = batch.genes.len();
        let local = Tensor::from_slice(&batch.local, batch.local.len(), dev)?;
        let genes = Tensor::from_slice(&batch.genes, n_genes, dev)?;
        let peaks = Tensor::from_slice(&batch.peak, batch.peak.len(), dev)?;
        let dist = Tensor::from_slice(&batch.dist, batch.dist.len(), dev)?.to_dtype(dtype)?;

        let rho_b = rho.index_select(&genes, 0)?; // [G_b, H]
        let phi_k = phi.index_select(&peaks, 0)?; // [K, H]
        let q = rho_b.matmul(self.a.as_tensor())?.index_select(&local, 0)?; // [K, r]
        let k = phi_k.matmul(self.b.as_tensor())?; // [K, r]
        let content = (q * k)?.sum(1)?;
        let gamma = self.log_gamma.as_tensor().exp()?;
        let c = self.log_c.as_tensor().exp()?;
        let log_kernel = dist
            .broadcast_add(&c)?
            .log()?
            .broadcast_mul(&gamma)?
            .neg()?;
        let score = (content + log_kernel)?;

        // Segmented softmax: subtract each gene's (detached) max, which leaves
        // the shares and their gradient unchanged.
        let host = score.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        let mut seg_max = vec![f64::NEG_INFINITY; n_genes];
        for (&l, &s) in batch.local.iter().zip(&host) {
            seg_max[l as usize] = seg_max[l as usize].max(s);
        }
        let shift: Vec<f64> = batch.local.iter().map(|&l| seg_max[l as usize]).collect();
        let shift = Tensor::from_vec(shift, batch.local.len(), dev)?.to_dtype(dtype)?;
        let ex = (score - shift)?.exp()?;
        let denom = Tensor::zeros(n_genes, dtype, dev)?.index_add(&local, &ex, 0)?;
        let pi = (ex / denom.index_select(&local, 0)?)?;
        let psi = Tensor::zeros((n_genes, phi.dim(1)?), dtype, dev)?.index_add(
            &local,
            &phi_k.broadcast_mul(&pi.unsqueeze(1)?)?,
            0,
        )?;
        Ok((pi, psi))
    }

    /// Mean squared disagreement `‖ρ_g − ψ_g‖²` over the batch's genes that
    /// have cis peaks. `rho` is `[genes × H]` aligned with `pairs`, `phi`
    /// `[peaks × H]`.
    pub fn batch_loss(
        &self,
        rho: &Tensor,
        phi: &Tensor,
        pairs: &CisPairs,
        genes: &[usize],
    ) -> anyhow::Result<Tensor> {
        let batch = Batch::new(pairs, genes);
        anyhow::ensure!(
            !batch.genes.is_empty(),
            "no gene in the batch has cis peaks"
        );
        let (_, psi) = self.forward(rho, phi, &batch)?;
        let idx = Tensor::from_slice(&batch.genes, batch.genes.len(), rho.device())?;
        let rho_b = rho.index_select(&idx, 0)?;
        Ok((rho_b - psi)?.sqr()?.sum(1)?.mean(0)?)
    }
}

fn scalar_f64(t: &Tensor) -> f64 {
    t.to_dtype(DType::F64)
        .and_then(|t| t.to_scalar::<f64>())
        .unwrap_or(f64::NAN)
}

fn to_tensor(m: &DMatrix<f32>, dev: &Device) -> anyhow::Result<Tensor> {
    Ok(Tensor::from_slice(
        m.transpose().as_slice(),
        m.shape(),
        dev,
    )?)
}

/// A fitted attention: shares per pair (in `CisPairs` order), the learned
/// kernel, and the loss after each epoch.
pub struct AttentionFit {
    pub pi: Vec<f32>,
    pub gamma: f64,
    pub pseudocount: f64,
    pub loss: Vec<f64>,
}

/// Train the attention and read every gene's shares. `rho` `[genes × H]` is
/// aligned with `pairs`; `phi` is `[peaks × H]`.
pub fn fit_attention(
    rho: &DMatrix<f32>,
    phi: &DMatrix<f32>,
    pairs: &CisPairs,
    cfg: &AttentionConfig,
) -> anyhow::Result<AttentionFit> {
    let h = rho.ncols();
    anyhow::ensure!(
        phi.ncols() == h,
        "gene rows are {h}-dimensional, peak rows {}",
        phi.ncols()
    );
    anyhow::ensure!(
        rho.nrows() == pairs.n_genes(),
        "{} gene rows for {} genes",
        rho.nrows(),
        pairs.n_genes()
    );
    let dev = Device::Cpu;
    let (rho_t, phi_t) = (to_tensor(rho, &dev)?, to_tensor(phi, &dev)?);
    let att = LocalAttention::new(
        h,
        cfg.rank,
        cfg.init_scale,
        cfg.init_gamma,
        cfg.init_pseudocount,
        DType::F32,
        &dev,
        cfg.seed,
    )?;
    let mut opt = AdamW::new(
        att.vars(),
        ParamsAdamW {
            lr: cfg.learning_rate,
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        },
    )?;

    let mut genes: Vec<usize> = (0..pairs.n_genes())
        .filter(|&g| !pairs.gene(g).is_empty())
        .collect();
    let step = cfg.genes_per_step.max(1);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut loss = Vec::with_capacity(cfg.epochs);
    for _ in 0..cfg.epochs {
        genes.shuffle(&mut rng);
        let (mut total, mut n) = (0f64, 0usize);
        for chunk in genes.chunks(step) {
            let l = att.batch_loss(&rho_t, &phi_t, pairs, chunk)?;
            total += scalar_f64(&l) * chunk.len() as f64;
            n += chunk.len();
            opt.backward_step(&l)?;
        }
        loss.push(total / n.max(1) as f64);
    }

    let mut pi = vec![0f32; pairs.n_pairs()];
    genes.sort_unstable();
    for chunk in genes.chunks(step) {
        let batch = Batch::new(pairs, chunk);
        let (p, _) = att.forward(&rho_t, &phi_t, &batch)?;
        for (&k, v) in batch.pair.iter().zip(p.to_vec1::<f32>()?) {
            pi[k] = v;
        }
    }
    Ok(AttentionFit {
        pi,
        gamma: att.gamma(),
        pseudocount: att.pseudocount(),
        loss,
    })
}

/// `ψ_g = Σ_p π_gp φ_p` on the host, `[genes × H]`; genes without pairs get 0.
#[must_use]
pub fn pool(pairs: &CisPairs, pi: &[f32], phi: &DMatrix<f32>) -> DMatrix<f32> {
    let mut psi = DMatrix::zeros(pairs.n_genes(), phi.ncols());
    for g in 0..pairs.n_genes() {
        for k in pairs.gene(g) {
            let p = pairs.peak[k] as usize;
            for c in 0..phi.ncols() {
                psi[(g, c)] += pi[k] * phi[(p, c)];
            }
        }
    }
    psi
}
