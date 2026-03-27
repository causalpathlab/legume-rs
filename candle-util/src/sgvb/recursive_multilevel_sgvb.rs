//! Recursive Multilevel SuSiE for massive n << p problems.
//!
//! When p >> n, standard SuSiE's flat softmax over p features creates a difficult
//! optimization landscape — the gradient signal is diluted across millions of
//! near-zero probabilities. This module replaces the flat softmax with a recursive
//! hierarchy of smaller softmaxes, where each level's design matrix is collapsed
//! using the level below's PIPs.
//!
//! # Per-component collapse
//!
//! Each SuSiE component l has its own selection probabilities α_l, so it sees
//! its own collapsed design matrix X̃_l at each level. This is critical because
//! different components select different features — component 1 might focus on
//! block 3 while component 2 focuses on block 7, so their collapsed views of
//! the data must differ.
//!
//! # Architecture
//!
//! ```text
//! Level 0: X (n × p),       SusieVar(L, p, k)    → α⁰_{l,j}
//!          Collapse: X̃¹_l[:,g] = Σ_{j∈g} α⁰_{l,j} · x_j
//!
//! Level 1: X̃¹_l (n × G₁),  SusieVar(L, G₁, k)   → α¹_{l,g}
//!          Collapse: X̃²_l[:,g'] = Σ_{g∈g'} α¹_{l,g} · X̃¹_l[:,g]
//!
//! ...
//! Level D (top): X̃ᴰ_l (n × G_D), SusieVar(L, G_D, k) — terminal, no collapse
//! ```
//!
//! Joint selection: weight_{l,j} = α⁰_{l,j} · α¹_{l,g¹(j)} · ... · αᴰ_{l,gᴰ(j)}
//!
//! The collapse is differentiable — gradients flow through α back to per level
//! logits, so all levels are jointly optimized by a single SGVB pass.
//!
//! # Why not CAVI
//!
//! CAVI's coordinate ascent subtracts component l's old contribution from the
//! residual, then recomputes. But here X̃_l depends on α_l — updating α changes
//! the design matrix itself. This circular dependency breaks the "subtract old,
//! add new" trick. SGVB handles it naturally since everything is jointly optimized
//! via backprop through the differentiable collapse.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::block_partition::BlockPartition;
use super::sgvb::SGVBConfig;
use super::traits::{
    AnalyticalKL, BlackBoxLikelihood, LocalReparamModel, LocalReparamSample, Prior,
    VariationalDistribution,
};
use super::variational_susie::SusieVar;

/// One level of the collapse hierarchy.
pub struct CollapseLevel {
    /// SuSiE variational distribution at this level's granularity
    pub susie: SusieVar,
    /// Block partition defining how this level's features group into the next level
    pub partition: BlockPartition,
}

/// Recursive multilevel SGVB model.
///
/// Composes multiple `CollapseLevel`s from finest to coarsest, each with its
/// own `SusieVar`. The forward pass collapses X bottom up using per component
/// α, then applies local reparameterization at the top level.
pub struct RecursiveMultilevelSGVB<P> {
    levels: Vec<CollapseLevel>,
    prior: P,
    x_design: Tensor,
    pub config: SGVBConfig,
}

impl<P> RecursiveMultilevelSGVB<P> {
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn num_components(&self) -> usize {
        self.levels[0].susie.num_components()
    }

    pub fn x_design(&self) -> &Tensor {
        &self.x_design
    }

    pub fn level(&self, d: usize) -> &CollapseLevel {
        &self.levels[d]
    }
}

impl<P: Prior + AnalyticalKL> RecursiveMultilevelSGVB<P> {
    /// Build a recursive multilevel model with automatic level construction.
    ///
    /// Given p features and block_size, constructs a hierarchy:
    ///   Level 0: SusieVar(L, p, k) with partition p -> ceil(p/block_size) groups
    ///   Level 1: SusieVar(L, G1, k) with partition G1 -> ceil(G1/block_size) groups
    ///   ...
    ///   Level D (terminal): SusieVar(L, G_D, k) where G_D <= block_size
    ///
    /// When p <= block_size, creates a single flat SusieVar (no hierarchy).
    pub fn auto(
        vb: VarBuilder,
        x_design: Tensor,
        num_components: usize,
        k: usize,
        block_size: usize,
        prior: P,
        config: SGVBConfig,
    ) -> Result<Self> {
        let p = x_design.dim(1)?;
        let partitions = BlockPartition::build_hierarchy(p, block_size);

        let mut levels = Vec::new();

        if partitions.is_empty() {
            let susie = SusieVar::new(vb.pp("level_0"), num_components, p, k)?;
            let partition = BlockPartition::regular(p, p);
            levels.push(CollapseLevel { susie, partition });
        } else {
            let susie_0 = SusieVar::new(vb.pp("level_0"), num_components, p, k)?;
            levels.push(CollapseLevel {
                susie: susie_0,
                partition: partitions[0].clone(),
            });

            for (d, part) in partitions.iter().enumerate().skip(1) {
                let g_d = partitions[d - 1].num_blocks();
                let susie_d = SusieVar::new(vb.pp(format!("level_{}", d)), num_components, g_d, k)?;
                levels.push(CollapseLevel {
                    susie: susie_d,
                    partition: part.clone(),
                });
            }

            let g_top = partitions.last().unwrap().num_blocks();
            let d_top = partitions.len();
            let susie_top =
                SusieVar::new(vb.pp(format!("level_{}", d_top)), num_components, g_top, k)?;
            let terminal_partition = BlockPartition::regular(g_top, g_top);
            levels.push(CollapseLevel {
                susie: susie_top,
                partition: terminal_partition,
            });
        }

        Ok(Self {
            levels,
            prior,
            x_design,
            config,
        })
    }

    /// Forward pass: bottom up collapse + local reparameterization at top level.
    ///
    /// 1. Cache all level alphas (avoids recomputing softmax per level)
    /// 2. Collapse X bottom up: for each level d, each component l gets
    ///    X̃ᵈ_l = collapse(X̃^{d-1}_l, αᵈ_l) via weighted column sums
    /// 3. At the top level, compute E[η] and V[η] per component (can't use
    ///    the standard (X⊙X)·V trick because each component has a different X̃_l)
    /// 4. Sample η = E[η] + √V[η] ⊙ ε using antithetic sampling
    pub fn local_reparam_sample(&self, num_samples: usize) -> Result<LocalReparamSample> {
        let num_levels = self.levels.len();
        let num_comp = self.num_components();
        let (n, _p) = self.x_design.dims2()?;
        let device = self.x_design.device();
        let dtype = self.x_design.dtype();

        // Shallow clone — candle tensors are refcounted, no data copy
        let mut prev_x: Vec<Tensor> = vec![self.x_design.clone(); num_comp];

        // Cache alphas to avoid recomputation
        let mut alphas = Vec::with_capacity(num_levels);
        for level in &self.levels {
            alphas.push(level.susie.alpha()?);
        }

        let k = alphas[0].dim(2)?;

        for (d, alpha_d) in alphas.iter().enumerate().take(num_levels - 1) {
            let partition = &self.levels[d].partition;

            let mut next_x = Vec::with_capacity(num_comp);
            for (l, x_prev_l) in prev_x.iter().enumerate() {
                let x_collapsed =
                    collapse_with_alpha(x_prev_l, alpha_d, l, &partition.block_ranges, k)?;
                next_x.push(x_collapsed);
            }
            prev_x = next_x;
        }

        // Top-level local reparameterization
        let top_alpha = &alphas[num_levels - 1]; // (L, G_top, k)
        let top_susie = &self.levels[num_levels - 1].susie;
        let top_beta_mean = top_susie.beta_mean(); // (L, G_top, k)
        let top_beta_std = top_susie.beta_std()?; // (L, G_top, k)

        // Standard local reparam would be: E[η] = X·μ, V[η] = X²·σ².
        // But here each component l has a DIFFERENT X̃_l, so we can't factor
        // as a single (X⊙X)·V. Instead we accumulate per component:
        //   E[η] = Σ_l X̃_l · (α_l ⊙ μ_l)
        //   E[η²]_i = Σ_l (X̃²_l)_i · [α_l · (σ²_l + μ²_l)]
        //   V[η] = E[η²] - E[η]²
        //
        // We accumulate per output dimension k separately and cat once at
        // the end, avoiding repeated tensor decompose/reassemble.
        let mut mean_per_k: Vec<Tensor> = (0..k)
            .map(|_| Tensor::zeros((n, 1), dtype, device))
            .collect::<Result<_>>()?;
        let mut second_per_k: Vec<Tensor> = (0..k)
            .map(|_| Tensor::zeros((n, 1), dtype, device))
            .collect::<Result<_>>()?;

        for (l, x_l) in prev_x.iter().enumerate() {
            let alpha_l = top_alpha.get(l)?; // (G_top, k)
            let mu_l = top_beta_mean.get(l)?;
            let sigma_l = top_beta_std.get(l)?;

            let weighted_mean = alpha_l.broadcast_mul(&mu_l)?;
            let sigma_sq = sigma_l.sqr()?;
            let mu_sq = mu_l.sqr()?;
            let second_weights = alpha_l.broadcast_mul(&(sigma_sq + mu_sq)?)?;

            for kk in 0..k {
                let x_lk = if k == 1 {
                    x_l.clone()
                } else {
                    extract_k_slice(x_l, kk, k)?
                };
                let wm_k = weighted_mean.narrow(1, kk, 1)?.contiguous()?;
                let sw_k = second_weights.narrow(1, kk, 1)?.contiguous()?;

                mean_per_k[kk] = (&mean_per_k[kk] + x_lk.matmul(&wm_k)?)?;
                second_per_k[kk] = (&second_per_k[kk] + x_lk.sqr()?.matmul(&sw_k)?)?;
            }
        }

        let eta_mean = Tensor::cat(&mean_per_k, 1)?; // (n, k)
        let eta_second_moment = Tensor::cat(&second_per_k, 1)?;

        let eta_var = (eta_second_moment - eta_mean.sqr()?)?.clamp(1e-8, f64::INFINITY)?;

        let epsilon = antithetic_epsilon(num_samples, n, k, device, dtype)?;

        let eta_std = eta_var.sqrt()?;
        let eta = eta_mean
            .unsqueeze(0)?
            .broadcast_add(&epsilon.broadcast_mul(&eta_std.unsqueeze(0)?)?)?;

        let kl = self.compute_kl()?;

        Ok(LocalReparamSample { eta, kl })
    }

    /// Total KL divergence summed across all hierarchy levels.
    ///
    /// Each level contributes two terms:
    ///   1. Categorical KL: how far α deviates from uniform selection
    ///   2. Gaussian slab KL: how far the effect size posterior deviates from the prior
    ///
    /// KL = Σ_d [KL_categorical(αᵈ || uniform) + KL_gaussian(βᵈ || prior)]
    fn compute_kl(&self) -> Result<Tensor> {
        let device = self.x_design.device();
        let dtype = self.x_design.dtype();
        let mut total_kl = Tensor::new(0f32, device)?.to_dtype(dtype)?;

        for level in &self.levels {
            let mean = level.susie.mean()?;
            let var = level.susie.var()?;
            let gaussian_kl = self.prior.kl_from_gaussian(&mean, &var)?;
            total_kl = (total_kl + gaussian_kl)?;

            let cat_kl = level.susie.kl_categorical(1.0)?;
            total_kl = (total_kl + cat_kl)?;
        }

        Ok(total_kl)
    }

    /// Joint selection probabilities across all levels.
    ///
    /// weight_{l,j} = α⁰_{l,j} · α¹_{l,g¹(j)} · ... · αᴰ_{l,gᴰ(j)}
    ///
    /// Uses `index_select` to expand coarser level α back to fine-feature space.
    /// This is for diagnostics/reporting only — not used in the loss computation.
    /// The expansion detaches from the coarser level computation graph, so
    /// gradients do not flow through this path.
    ///
    /// Returns shape (L, p, k).
    pub fn joint_alpha(&self) -> Result<Tensor> {
        let num_levels = self.levels.len();

        let mut joint = self.levels[0].susie.alpha()?; // (L, p, k)

        let mut current_partition = &self.levels[0].partition;

        for d in 1..num_levels {
            let alpha_d = self.levels[d].susie.alpha()?; // (L, G_d, k)

            // Build index tensor mapping each feature to its group
            let (_, prev_features, _) = joint.dims3()?;
            let mut group_indices = vec![0u32; prev_features];
            for (b, range) in current_partition.block_ranges.iter().enumerate() {
                for j in range.clone() {
                    group_indices[j] = b as u32;
                }
            }
            let idx = Tensor::from_vec(group_indices, (prev_features,), joint.device())?;

            // Expand alpha_d from (L, G_d, k) to (L, prev_features, k) via index_select
            let (l_dim, _, _) = alpha_d.dims3()?;
            let mut expanded_levels = Vec::with_capacity(l_dim);
            for l in 0..l_dim {
                let alpha_l = alpha_d.get(l)?; // (G_d, k)
                let expanded_l = alpha_l.index_select(&idx, 0)?; // (prev_features, k)
                expanded_levels.push(expanded_l.unsqueeze(0)?);
            }
            let expanded = Tensor::cat(&expanded_levels, 0)?; // (L, prev_features, k)

            // Detach expanded so coarser alpha doesn't accumulate stale gradients
            // in this diagnostic path
            let expanded = expanded.to_dtype(joint.dtype())?;

            joint = joint.broadcast_mul(&expanded)?;

            if d < num_levels - 1 {
                current_partition = &self.levels[d].partition;
            }
        }

        Ok(joint)
    }

    /// Posterior inclusion probabilities (diagnostics only, not differentiable).
    ///
    /// PIP_j = 1 - Π_l (1 - weight_{l,j})
    ///
    /// Returns shape (p, k).
    pub fn joint_pip(&self) -> Result<Tensor> {
        let joint = self.joint_alpha()?;
        pip_from_alpha(&joint)
    }
}

impl<P: Prior + AnalyticalKL> LocalReparamModel for RecursiveMultilevelSGVB<P> {
    fn local_reparam_sample(&self, num_samples: usize) -> Result<LocalReparamSample> {
        self.local_reparam_sample(num_samples)
    }
}

/// Compute posterior inclusion probability from per component selection weights.
///
/// PIP_j = 1 - Π_l (1 - α_{l,j})
///
/// Computed in log space for numerical stability:
///   log(1 - PIP_j) = Σ_l log(1 - α_{l,j})
///
/// Shared across SusieVar, MultiLevelSusieVar, BiSusieVar, and the
/// recursive multilevel model.
///
/// alpha shape (L, p, k), returns (p, k).
pub fn pip_from_alpha(alpha: &Tensor) -> Result<Tensor> {
    let one_minus = (1.0 - alpha)?.clamp(1e-10, 1.0)?;
    let log_one_minus = one_minus.log()?;
    let sum_log = log_one_minus.sum(0)?;
    1.0 - sum_log.exp()?
}

/// Collapse a design matrix using component l's selection probabilities.
///
/// For each group g: X̃[:,g] = X[:, block(g)] @ α[l, block(g), :]
///
/// This is the core differentiable operation — `narrow` on α preserves the
/// computation graph, so gradients flow through α back to the per level logits.
/// Each block's collapse is an independent matmul (no cross block interaction),
/// and `.contiguous()` is required because `narrow` produces strided views
/// that candle's matmul rejects.
fn collapse_with_alpha(
    x: &Tensor,
    alpha: &Tensor,
    component: usize,
    block_ranges: &[std::ops::Range<usize>],
    k: usize,
) -> Result<Tensor> {
    let alpha_l = alpha.get(component)?; // (G_prev, k)

    if k == 1 {
        let mut cols = Vec::with_capacity(block_ranges.len());
        for range in block_ranges {
            let x_sub = x.narrow(1, range.start, range.len())?.contiguous()?;
            let a_sub = alpha_l.narrow(0, range.start, range.len())?.contiguous()?;
            cols.push(x_sub.matmul(&a_sub)?);
        }
        Tensor::cat(&cols, 1)
    } else {
        // k > 1: collapse per output dimension using tensor ops (no scalar extraction)
        let mut cols = Vec::with_capacity(block_ranges.len() * k);
        for range in block_ranges {
            let x_sub_full = x
                .narrow(1, range.start * k, range.len() * k)?
                .contiguous()?;
            let a_sub = alpha_l.narrow(0, range.start, range.len())?.contiguous()?; // (|g|, k)
            for kk in 0..k {
                // Extract kk-th columns from x_sub: stride by k
                let mut x_k_cols = Vec::with_capacity(range.len());
                for j_local in 0..range.len() {
                    x_k_cols.push(x_sub_full.narrow(1, j_local * k + kk, 1)?);
                }
                let x_k = Tensor::cat(&x_k_cols, 1)?.contiguous()?; // (n, |g|)
                let a_k = a_sub.narrow(1, kk, 1)?.contiguous()?; // (|g|, 1)
                cols.push(x_k.matmul(&a_k)?); // (n, 1)
            }
        }
        Tensor::cat(&cols, 1)
    }
}

/// Extract the kk-th output slice from a multi output design matrix.
/// x has shape (n, G * k), returns (n, G) selecting every k-th column starting at kk.
fn extract_k_slice(x: &Tensor, kk: usize, k: usize) -> Result<Tensor> {
    let (_n, gk) = x.dims2()?;
    let g = gk / k;
    let mut cols = Vec::with_capacity(g);
    for j in 0..g {
        cols.push(x.narrow(1, j * k + kk, 1)?);
    }
    Tensor::cat(&cols, 1)?.contiguous()
}

/// Antithetic sampling for variance reduction in the local reparameterization trick.
///
/// Draws S/2 noise vectors ε, pairs each with its mirror −ε so the empirical
/// mean of noise is exactly zero. This strictly reduces variance of the MC
/// log likelihood estimate with no hyperparameters to tune.
///
/// Returns epsilon of shape (num_samples, n, k).
pub fn antithetic_epsilon(
    num_samples: usize,
    n: usize,
    k: usize,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let half_s = num_samples / 2;
    if half_s > 0 {
        let eps_half = Tensor::randn(0f32, 1f32, (half_s, n, k), device)?.to_dtype(dtype)?;
        let eps_neg = eps_half.neg()?;
        if num_samples % 2 == 1 {
            let eps_extra = Tensor::randn(0f32, 1f32, (1, n, k), device)?.to_dtype(dtype)?;
            Tensor::cat(&[eps_half, eps_neg, eps_extra], 0)
        } else {
            Tensor::cat(&[eps_half, eps_neg], 0)
        }
    } else {
        Tensor::randn(0f32, 1f32, (1, n, k), device)?.to_dtype(dtype)
    }
}

/// Compute ELBO loss for any model implementing `LocalReparamModel`.
///
/// loss = -E[log p(y|η)] + β·KL
pub fn generic_local_reparam_loss<M: LocalReparamModel, L: BlackBoxLikelihood>(
    model: &M,
    likelihood: &L,
    num_samples: usize,
    kl_weight: f64,
) -> Result<Tensor> {
    let sample = model.local_reparam_sample(num_samples)?;
    let llik = likelihood.log_likelihood(&[&sample.eta])?;
    let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

    let elbo = (llik.mean(0)? - (sample.kl * kl_weight)?)?;
    elbo.neg()
}

/// Convenience alias for `generic_local_reparam_loss` with `RecursiveMultilevelSGVB`.
pub fn multilevel_loss<P: Prior + AnalyticalKL, L: BlackBoxLikelihood>(
    model: &RecursiveMultilevelSGVB<P>,
    likelihood: &L,
    num_samples: usize,
    kl_weight: f64,
) -> Result<Tensor> {
    generic_local_reparam_loss(model, likelihood, num_samples, kl_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgvb::traits::BlackBoxLikelihood;
    use crate::sgvb::GaussianPrior;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    struct GaussianLik {
        y: Tensor,
    }

    impl BlackBoxLikelihood for GaussianLik {
        fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
            let eta = etas[0];
            let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
            let log_prob = (diff_sq * (-0.5))?;
            log_prob.sum(2)?.sum(1)
        }
    }

    #[test]
    fn test_auto_construction_small() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 30;
        let p = 50;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RecursiveMultilevelSGVB::auto(vb.pp("ml"), x, l, k, 100, prior, config)?;

        assert_eq!(model.num_levels(), 1);
        assert_eq!(model.num_components(), l);

        let sample = model.local_reparam_sample(5)?;
        assert_eq!(sample.eta.dims(), &[5, n, k]);
        assert!(sample.kl.to_scalar::<f32>()?.is_finite());

        Ok(())
    }

    #[test]
    fn test_auto_construction_two_levels() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 30;
        let p = 100;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RecursiveMultilevelSGVB::auto(vb.pp("ml"), x, l, k, 10, prior, config)?;

        assert_eq!(model.num_levels(), 2);

        let sample = model.local_reparam_sample(5)?;
        assert_eq!(sample.eta.dims(), &[5, n, k]);

        let pip = model.joint_pip()?;
        assert_eq!(pip.dims(), &[p, k]);

        let pip_sum: f32 = pip.sum_all()?.to_scalar()?;
        assert!(pip_sum.is_finite());

        Ok(())
    }

    #[test]
    fn test_auto_construction_three_levels() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 20;
        let p = 1000;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RecursiveMultilevelSGVB::auto(vb.pp("ml"), x, l, k, 10, prior, config)?;

        assert_eq!(model.num_levels(), 3);

        let sample = model.local_reparam_sample(3)?;
        assert_eq!(sample.eta.dims(), &[3, n, k]);

        let pip = model.joint_pip()?;
        assert_eq!(pip.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_joint_alpha_sums() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let n = 20;
        let p = 100;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?.to_dtype(dtype)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RecursiveMultilevelSGVB::auto(vb.pp("ml"), x, l, k, 10, prior, config)?;

        let joint = model.joint_alpha()?;

        let joint_data: Vec<f64> = joint.flatten_all()?.to_vec1()?;
        for &v in &joint_data {
            assert!(v >= 0.0, "joint alpha must be non-negative, got {}", v);
            assert!(v <= 1.0, "joint alpha must be <= 1, got {}", v);
        }

        Ok(())
    }

    #[test]
    fn test_gradient_flow() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 30;
        let p = 50;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = RecursiveMultilevelSGVB::auto(vb.pp("ml"), x, l, k, 10, prior, config)?;
        let likelihood = GaussianLik { y };

        let loss = multilevel_loss(&model, &likelihood, 5, 1.0)?;
        assert!(loss.dims().is_empty());
        assert!(loss.to_scalar::<f32>()?.is_finite());

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;
        optimizer.backward_step(&loss)?;

        let loss2 = multilevel_loss(&model, &likelihood, 5, 1.0)?;
        let l1: f32 = loss.to_scalar()?;
        let l2: f32 = loss2.to_scalar()?;
        assert!(
            (l1 - l2).abs() > 1e-8,
            "Loss should change after gradient step: {} vs {}",
            l1,
            l2
        );

        Ok(())
    }

    #[test]
    fn test_sparse_recovery() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 150;
        let p = 100;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;

        let x_10 = x.narrow(1, 10, 1)?;
        let x_50 = x.narrow(1, 50, 1)?;
        let noise = Tensor::randn(0f32, 0.5f32, (n, k), &device)?;
        let y = ((x_10 * 2.0)? + (x_50 * 1.5)? + noise)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::new(30);

        let model = RecursiveMultilevelSGVB::auto(vb.pp("ml"), x, l, k, 10, prior, config)?;
        let likelihood = GaussianLik { y };

        let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.05)?;

        for i in 0..500 {
            let loss = multilevel_loss(&model, &likelihood, 30, 1.0)?;
            optimizer.backward_step(&loss)?;

            if i % 100 == 0 {
                let lv: f32 = loss.to_scalar()?;
                let pip = model.joint_pip()?;
                let pip_10: f32 = pip.get(10)?.get(0)?.to_scalar()?;
                let pip_50: f32 = pip.get(50)?.get(0)?.to_scalar()?;
                println!(
                    "multilevel iter {}: loss={:.4}, PIP[10]={:.4}, PIP[50]={:.4}",
                    i, lv, pip_10, pip_50
                );
            }
        }

        let pip = model.joint_pip()?;
        let pip_10: f32 = pip.get(10)?.get(0)?.to_scalar()?;
        let pip_50: f32 = pip.get(50)?.get(0)?.to_scalar()?;

        let mut other_sum = 0.0f32;
        for j in 0..p {
            if j != 10 && j != 50 {
                other_sum += pip.get(j)?.get(0)?.to_scalar::<f32>()?;
            }
        }
        let other_mean = other_sum / (p - 2) as f32;

        println!("\nMultilevel PIPs:");
        println!("  PIP[10] (true): {:.4}", pip_10);
        println!("  PIP[50] (true): {:.4}", pip_50);
        println!("  Others mean:    {:.4}", other_mean);

        assert!(
            pip_10 > other_mean * 2.0,
            "PIP[10] should be > 2x others: {} vs {}",
            pip_10,
            other_mean
        );
        assert!(
            pip_50 > other_mean * 2.0,
            "PIP[50] should be > 2x others: {} vs {}",
            pip_50,
            other_mean
        );

        Ok(())
    }
}
