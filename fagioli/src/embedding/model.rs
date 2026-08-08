//! The low-rank model and its score.
//!
//! # The score is a matched filter
//!
//! Both axes are whitened ([`super::whiten`]), so per eigen-coordinate the null
//! is `ž_k ~ N(0, s²_k I_T)` with `s_k` the noise scale from [`super::noise`],
//! and the alternative shifts the mean to `μ̌_k`. The log Bayes factor is then
//!
//! ```text
//! s(ž_k) = [ <ž_k, μ̌_k> − ½‖μ̌_k‖² ] / s²_k
//! ```
//!
//! — a matched filter, with no determinant term because the two hypotheses
//! share a covariance. Both densities are proper, so this is *exactly*
//! normalised: the learned offset in [`super::train`] has nothing to absorb and
//! should converge to zero, which makes it a free calibration check rather than
//! a nuisance parameter.
//!
//! # The mean structure
//!
//! ```text
//! μ̌_b = (X̃_b U_b) V̌'          X̃_b: (K_b, p_b),  U_b: (p_b, H),  V̌: (T, H)
//! ```
//!
//! `X̃_b U_b` is the only per-block matmul and is `(K_b, H)` — small. `V̌` is
//! shared across blocks and is what couples them; `U_b` is block-local, which is
//! also what makes the leave-one-block-out jackknife cheap.
//!
//! # What is *not* here
//!
//! The BSLMM dense arm. Marginalising a per-program dense component contributes
//! `σ²_d,h · X̃X̃' = σ²_d,h · D̃²` to the covariance, which is diagonal and would
//! cost only `H` parameters — but the resulting score needs an `H x H` solve
//! whose factorisation depends on the learned `Σ_d`, and candle has no
//! differentiable Cholesky or eigendecomposition. v1 therefore fits the mean
//! structure, which is exact at `Σ_d = 0`; the dense coefficient is still
//! measured and reported by
//! [`crate::summary_stats::calibration`]. [`EmbedModel::score`] is the single
//! seam where a dense term, a heavy-tailed null, or a likelihood objective
//! would enter.

use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::VarBuilder;
use candle_util::sgvb::{SpikeSlabVar, SusieVar};
use candle_util::sgvb::VariationalDistribution;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;

use super::noise::NoiseModel;
use super::whiten::WhitenedBlock;

/// Which variational family carries the sparse variant loadings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum UPrior {
    /// Independent Bernoulli per (variant, program). No categorical anywhere,
    /// so the softmax degeneracy that appears past a few thousand categories
    /// cannot arise. Gives no credible sets.
    SpikeSlab,
    /// `L` single-effect components per block, each a categorical over that
    /// block's variants. The softmax is bounded by the block size rather than
    /// the genome, which is what `--max-block-snps` is for. Gives credible sets.
    Susie,
}

/// Knobs for a fit.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Program dimension H.
    pub embedding_dim: usize,
    /// Negatives drawn per positive.
    pub num_negatives: usize,
    /// Prior inclusion probability for the spike-slab on `U`.
    pub prior_inclusion: f64,
    /// Which variational family carries `U`.
    pub u_prior: UPrior,
    /// Single-effect components per block, for [`UPrior::Susie`].
    pub num_components: usize,
    /// Dirichlet concentration for the SuSiE selection prior.
    pub prior_alpha: f64,
    pub learning_rate: f64,
    pub num_iterations: usize,
    /// Global-norm gradient clip; `None` disables.
    pub grad_clip: Option<f64>,
    /// Model a marginalised polygenic component alongside the sparse one.
    pub dense_arm: bool,
    /// Weight on the soft gauge penalty `‖V̌'V̌ − I‖²_F`. The dense arm's score
    /// requires it (it diagonalises `A'A` to `Σ_d`), but it is applied
    /// independently: orthonormal columns are also what identify `V̌` at all,
    /// so it is a regulariser in its own right. Set 0 to disable.
    pub gauge_weight: f64,
    pub seed: u64,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 20,
            num_negatives: 5,
            prior_inclusion: 0.01,
            u_prior: UPrior::SpikeSlab,
            num_components: 5,
            prior_alpha: 1.0,
            learning_rate: 0.01,
            num_iterations: 500,
            grad_clip: Some(10.0),
            dense_arm: false,
            gauge_weight: 0.0,
            seed: 42,
        }
    }
}

/// Constant, per-block tensors the model reads but never learns.
pub struct BlockTensors {
    /// `X̃_b`, shape (K_b, p_b).
    pub x_design: Tensor,
    /// `ž_b`, shape (K_b, T).
    pub z_white: Tensor,
    /// `1/s²_k`, shape (K_b, 1) — broadcasts over traits.
    pub inv_var: Tensor,
    /// `r_k = d̃²_k / s²_k`, shape (K_b, 1). Sets how much polygenic variance a
    /// coordinate carries: the dense component enters as `d̃²`, the sparse mean
    /// as `d̃`, which is what makes the two separable.
    pub ratio: Tensor,
    pub rank: usize,
}

/// The sparse variant loadings for one block.
///
/// Both families expose a posterior mean and a per-(variant, program)
/// inclusion probability, so the score and the readouts are identical; they
/// differ in how selection is parameterised and therefore in what the
/// inclusion probabilities mean.
pub(crate) enum VariantLoadings {
    /// Independent gates: `pip` is a marginal inclusion probability.
    SpikeSlab(SpikeSlabVar),
    /// Single-effect components: `pip` aggregates `L` categoricals, so mass is
    /// forced to compete within a block and credible sets follow.
    Susie(SusieVar),
}

impl VariantLoadings {
    fn mean(&self) -> Result<Tensor> {
        Ok(match self {
            Self::SpikeSlab(v) => v.mean()?,
            Self::Susie(v) => v.mean()?,
        })
    }

    fn pip(&self) -> Result<Tensor> {
        Ok(match self {
            Self::SpikeSlab(v) => v.pip()?,
            Self::Susie(v) => v.pip()?,
        })
    }

    fn kl(&self, prior_inclusion: f64, prior_alpha: f64) -> Result<Tensor> {
        use candle_util::sgvb::IndependentGateVariational;
        Ok(match self {
            Self::SpikeSlab(v) => v.kl_bernoulli(prior_inclusion)?,
            Self::Susie(v) => v.kl_categorical(prior_alpha)?,
        })
    }
}

/// The embedding: a shared trait side, a block-local sparse variant side.
pub struct EmbedModel {
    /// `V̌ = Ω^{-1/2} Λ_N V`, shape (T, H). Learned in whitened coordinates.
    v_check: Tensor,
    /// Sparse `U_b` per block.
    u_blocks: Vec<VariantLoadings>,
    /// NCE offset. Should stay near zero for a correctly normalised score.
    offset: Tensor,
    /// `log σ²_d,h`, shape (H,). The entire polygenic arm: one variance per
    /// program, shared by every variant. `None` when the dense arm is off.
    log_sigma_d: Option<Tensor>,
    pub blocks: Vec<BlockTensors>,
    pub num_traits: usize,
    pub config: EmbedConfig,
}

impl EmbedModel {
    pub fn new(
        vb: &VarBuilder,
        blocks: &[WhitenedBlock],
        noise: &NoiseModel,
        config: EmbedConfig,
        device: &Device,
    ) -> Result<Self> {
        let num_traits = blocks.first().map_or(0, WhitenedBlock::num_traits);
        let h = config.embedding_dim;

        let v_check = vb.get_with_hints(
            (num_traits, h),
            "v_check",
            candle_util::candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;
        let offset = vb.get_with_hints(
            (1,),
            "nce_offset",
            candle_util::candle_nn::Init::Const(0.0),
        )?;

        // The polygenic arm costs H parameters in total, not M x H: variants
        // share a prior, not a value.
        let log_sigma_d = config
            .dense_arm
            .then(|| {
                vb.get_with_hints((h,), "log_sigma_d", candle_util::candle_nn::Init::Const(-2.0))
            })
            .transpose()?;

        let mut u_blocks = Vec::with_capacity(blocks.len());
        let mut tensors = Vec::with_capacity(blocks.len());

        for (bi, b) in blocks.iter().enumerate() {
            u_blocks.push(match config.u_prior {
                UPrior::SpikeSlab => VariantLoadings::SpikeSlab(SpikeSlabVar::new(
                    vb.pp(format!("u_{bi}")),
                    b.num_snps,
                    h,
                    config.prior_inclusion.clamp(1e-6, 0.5),
                )?),
                // The categorical runs over this block's variants only, never
                // the genome, so its dimension is bounded by --max-block-snps.
                UPrior::Susie => VariantLoadings::Susie(SusieVar::new_with_null(
                    vb.pp(format!("u_{bi}")),
                    config.num_components.max(1),
                    b.num_snps,
                    h,
                )?),
            });

            let inv_var: Vec<f32> = noise.scale[bi]
                .iter()
                .map(|s| 1.0 / (s * s).max(f32::MIN_POSITIVE))
                .collect();

            let ratio: Vec<f32> = b
                .d_sq
                .iter()
                .zip(noise.scale[bi].iter())
                .map(|(&d2, &s)| (d2 + noise.lambda as f32) / (s * s).max(f32::MIN_POSITIVE))
                .collect();

            tensors.push(BlockTensors {
                x_design: b.x_design.to_tensor(device)?.contiguous()?,
                z_white: b.z_white.to_tensor(device)?.contiguous()?,
                inv_var: Tensor::from_slice(&inv_var, (b.rank(), 1), device)?,
                ratio: Tensor::from_slice(&ratio, (b.rank(), 1), device)?,
                rank: b.rank(),
            });
        }

        Ok(Self {
            v_check,
            u_blocks,
            offset,
            log_sigma_d,
            blocks: tensors,
            num_traits,
            config,
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// `μ̌_b = (X̃_b E[U_b]) V̌'`, shape (K_b, T).
    pub fn mean_structure(&self, block: usize) -> Result<Tensor> {
        let u = self.u_blocks[block].mean()?; // (p_b, H)
        let xu = self.blocks[block].x_design.matmul(&u)?; // (K_b, H)
        Ok(xu.matmul(&self.v_check.t()?)?) // (K_b, T)
    }

    /// Per-eigen-coordinate log Bayes factor for `z`, shape (K_b,).
    ///
    /// This is the single seam for changing the score: a dense covariance term,
    /// a heavy-tailed null, or a switch to a likelihood objective all replace
    /// this function and nothing else.
    pub fn score(&self, block: usize, z: &Tensor, mean: &Tensor) -> Result<Tensor> {
        let bt = &self.blocks[block];
        // Matched filter: <ž_k, μ̌_k> − ½‖μ̌_k‖², weighted by 1/s²_k.
        let cross = (z * mean)?.sum(1)?; // (K,)
        let quad = mean.sqr()?.sum(1)?; // (K,)
        let inv_var = bt.inv_var.squeeze(1)?; // (K,)
        let mut s = ((cross - (quad * 0.5)?)? * &inv_var)?;

        // Polygenic arm. Under the gauge V̌'V̌ = I the covariance
        // Σ_k = s²_k I + d̃²_k V̌ Σ_d V̌' inverts elementwise, so this adds a
        // quadratic reward for residual energy along program directions and a
        // log-determinant penalty -- no factorisation anywhere.
        if let Some(log_sigma_d) = &self.log_sigma_d {
            let sigma_d = log_sigma_d.exp()?.reshape((1, ()))?; // (1, H)
            let ratio = &bt.ratio; // (K, 1)
            // r_k σ²_dh, shape (K, H)
            let r_sig = ratio.broadcast_mul(&sigma_d)?;
            let one_plus = (&r_sig + 1.0)?;
            let w = r_sig.broadcast_div(&one_plus)?; // (K, H)

            // Residual projected onto the program directions.
            let resid = (z - mean)?; // (K, T)
            let proj = resid.matmul(&self.v_check)?; // (K, H)

            let dense = ((proj.sqr()? * w)?.sum(1)? * &inv_var)?;
            let logdet = one_plus.log()?.sum(1)?;
            s = ((s + (dense * 0.5)?)? - (logdet * 0.5)?)?;
        }

        Ok(s.broadcast_add(&self.offset)?)
    }

    /// Soft gauge penalty `‖V̌'V̌ − I‖²_F`.
    ///
    /// The dense arm's score assumes `V̌` has orthonormal columns -- that is what
    /// makes `A'A = Σ_d` diagonal and the whole thing elementwise. It is also
    /// the gauge that makes `V̌` identified at all, since `U→UA, V̌→V̌A⁻ᵀ` leaves
    /// `UV̌'` unchanged, and what repairs the singular Hessian in a jackknife.
    pub fn gauge_penalty(&self) -> Result<Tensor> {
        let h = self.config.embedding_dim;
        let gram = self.v_check.t()?.matmul(&self.v_check)?; // (H, H)
        let eye = Tensor::eye(h, gram.dtype(), gram.device())?;
        Ok((gram - eye)?.sqr()?.sum_all()?)
    }

    /// Fitted polygenic variance per program, `σ²_d,h`.
    pub fn dense_variance(&self) -> Result<Option<Vec<f32>>> {
        match &self.log_sigma_d {
            Some(t) => Ok(Some(t.exp()?.to_vec1::<f32>()?)),
            None => Ok(None),
        }
    }

    /// Selection KL for a block's `U`: Bernoulli or categorical by family.
    pub fn kl_selection(&self, block: usize) -> Result<Tensor> {
        self.u_blocks[block].kl(self.config.prior_inclusion, self.config.prior_alpha)
    }


    pub fn offset_value(&self) -> Result<f32> {
        Ok(self.offset.flatten_all()?.to_vec1::<f32>()?[0])
    }

    /// `V̌` as a host matrix, shape (T, H).
    pub fn v_check_matrix(&self) -> Result<DMatrix<f32>> {
        tensor_to_matrix(&self.v_check)
    }

    /// `E[U_b] = π ⊙ μ`, shape (p_b, H).
    pub fn u_mean_matrix(&self, block: usize) -> Result<DMatrix<f32>> {
        tensor_to_matrix(&self.u_blocks[block].mean()?)
    }

    /// Posterior inclusion probabilities for a block, shape (p_b, H).
    pub fn u_pip_matrix(&self, block: usize) -> Result<DMatrix<f32>> {
        tensor_to_matrix(&self.u_blocks[block].pip()?)
    }

    /// `V̌ (U'U) V̌'`, the `A`-invariant trait geometry the diagnostic judges.
    ///
    /// `U'U` is accumulated across blocks, since programs are global while `U`
    /// is stored block-locally.
    pub fn trait_geometry(&self) -> Result<DMatrix<f32>> {
        let h = self.config.embedding_dim;
        let mut utu = DMatrix::<f32>::zeros(h, h);
        for b in 0..self.num_blocks() {
            let u = self.u_mean_matrix(b)?;
            utu += u.transpose() * u;
        }
        let v = self.v_check_matrix()?;
        Ok(&v * utu * v.transpose())
    }
}

fn tensor_to_matrix(t: &Tensor) -> Result<DMatrix<f32>> {
    let dims = t.dims();
    let (r, c) = (dims[0], if dims.len() > 1 { dims[1] } else { 1 });
    let flat = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    // candle is row-major, nalgebra column-major.
    Ok(DMatrix::from_fn(r, c, |i, j| flat[i * c + j]))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
