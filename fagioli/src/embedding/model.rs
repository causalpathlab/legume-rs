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
use candle_util::sgvb::SpikeSlabVar;
use candle_util::sgvb::VariationalDistribution;
use matrix_util::traits::ConvertMatOps;
use nalgebra::DMatrix;

use super::noise::NoiseModel;
use super::whiten::WhitenedBlock;

/// Knobs for a fit.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Program dimension H.
    pub embedding_dim: usize,
    /// Negatives drawn per positive.
    pub num_negatives: usize,
    /// Prior inclusion probability for the spike-slab on `U`.
    pub prior_inclusion: f64,
    pub learning_rate: f64,
    pub num_iterations: usize,
    /// Global-norm gradient clip; `None` disables.
    pub grad_clip: Option<f64>,
    pub seed: u64,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 20,
            num_negatives: 5,
            prior_inclusion: 0.01,
            learning_rate: 0.01,
            num_iterations: 500,
            grad_clip: Some(10.0),
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
    pub rank: usize,
    pub num_snps: usize,
}

/// The embedding: a shared trait side, a block-local sparse variant side.
pub struct EmbedModel {
    /// `V̌ = Ω^{-1/2} Λ_N V`, shape (T, H). Learned in whitened coordinates.
    v_check: Tensor,
    /// Sparse `U_b` per block.
    u_blocks: Vec<SpikeSlabVar>,
    /// NCE offset. Should stay near zero for a correctly normalised score.
    offset: Tensor,
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

        let mut u_blocks = Vec::with_capacity(blocks.len());
        let mut tensors = Vec::with_capacity(blocks.len());

        for (bi, b) in blocks.iter().enumerate() {
            u_blocks.push(SpikeSlabVar::new(
                vb.pp(format!("u_{bi}")),
                b.num_snps,
                h,
                config.prior_inclusion.clamp(1e-6, 0.5),
            )?);

            let inv_var: Vec<f32> = noise.scale[bi]
                .iter()
                .map(|s| 1.0 / (s * s).max(f32::MIN_POSITIVE))
                .collect();

            tensors.push(BlockTensors {
                x_design: b.x_design.to_tensor(device)?.contiguous()?,
                z_white: b.z_white.to_tensor(device)?.contiguous()?,
                inv_var: Tensor::from_slice(&inv_var, (b.rank(), 1), device)?,
                rank: b.rank(),
                num_snps: b.num_snps,
            });
        }

        Ok(Self {
            v_check,
            u_blocks,
            offset,
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
        // <ž_k, μ̌_k> − ½‖μ̌_k‖², weighted by 1/s²_k, summed over traits.
        let cross = (z * mean)?.sum(1)?; // (K_b,)
        let quad = mean.sqr()?.sum(1)?; // (K_b,)
        let raw = (cross - (quad * 0.5)?)?;
        let w = self.blocks[block].inv_var.squeeze(1)?; // (K_b,)
        Ok((raw * w)?.broadcast_add(&self.offset)?)
    }

    /// Bernoulli selection KL for a block's `U`.
    pub fn kl_selection(&self, block: usize) -> Result<Tensor> {
        use candle_util::sgvb::IndependentGateVariational;
        Ok(self.u_blocks[block].kl_bernoulli(self.config.prior_inclusion)?)
    }

    pub fn v_check_tensor(&self) -> &Tensor {
        &self.v_check
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
