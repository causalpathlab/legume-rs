use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::traits::VariationalDistribution;
use super::variant_tree::VariantTree;

/// Multi-level SuSiE variational distribution with hierarchical softmax.
///
/// Replaces the flat softmax over p variants with a tree-structured
/// hierarchical softmax. At each tree level, log-softmax selects among
/// children. Joint selection probability is the product along the path.
///
/// Each SuSiE component has a per-feature effect size (L, p, k),
/// giving the optimizer full flexibility over effect sizes.
pub struct MultiLevelSusieVar {
    /// Variant tree structure
    tree: VariantTree,
    /// Selection logits per tree level: logits[d] has shape (L, G_d, C_d, k)
    logits_per_level: Vec<Tensor>,
    /// Validity masks per level: masks[d] has shape (G_d, C_d)
    masks_per_level: Vec<Tensor>,
    /// Flat path indices per level: path_indices[d] has shape (p,) as u32 tensor
    path_indices: Vec<Tensor>,
    /// Effect size means, shape (L, p, k) — per-feature effect per component
    beta_mean: Tensor,
    /// Effect size log-stds, shape (L, p, k)
    beta_ln_std: Tensor,
    /// Number of SuSiE components L
    num_components: usize,
    /// Temperature for softmax smoothing
    temperature: f64,
}

impl MultiLevelSusieVar {
    /// Create a new multi-level SuSiE variational distribution.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for creating trainable parameters
    /// * `tree` - Variant tree defining the hierarchical grouping
    /// * `num_components` - Number of single-effect components L
    /// * `k` - Number of output dimensions
    /// * `temperature` - Softmax temperature (1.0 = standard, >1 = smoother)
    pub fn new(
        vb: VarBuilder,
        tree: VariantTree,
        num_components: usize,
        k: usize,
        temperature: f64,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let mut logits_per_level = Vec::with_capacity(tree.depth);
        let mut masks_per_level = Vec::with_capacity(tree.depth);
        let mut path_indices = Vec::with_capacity(tree.depth);

        for (d, level) in tree.levels.iter().enumerate() {
            // Trainable logits: (L, num_groups, max_children, k)
            let logits = vb.get_with_hints(
                (num_components, level.num_groups, level.max_children, k),
                &format!("logits_level_{}", d),
                candle_nn::Init::Const(0.0),
            )?;
            logits_per_level.push(logits);

            // Mask: (num_groups, max_children) as f32/f64
            let mask_data: Vec<f32> = level
                .mask
                .iter()
                .flat_map(|row| row.iter().map(|&b| if b { 1.0 } else { 0.0 }))
                .collect();
            let mask =
                Tensor::from_vec(mask_data, (level.num_groups, level.max_children), &device)?
                    .to_dtype(dtype)?;
            masks_per_level.push(mask);

            // Path indices: (p,) as u32
            let idx_data: Vec<u32> = level.flat_path_indices.iter().map(|&i| i as u32).collect();
            let idx = Tensor::from_vec(idx_data, (tree.num_variants,), &device)?;
            path_indices.push(idx);
        }

        // Per-feature effect size: (L, p, k)
        let beta_mean = vb.get_with_hints(
            (num_components, tree.num_variants, k),
            "beta_mean",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;

        let beta_ln_std = vb.get_with_hints(
            (num_components, tree.num_variants, k),
            "beta_ln_std",
            candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            tree,
            logits_per_level,
            masks_per_level,
            path_indices,
            beta_mean,
            beta_ln_std,
            num_components,
            temperature,
        })
    }

    /// Compute log selection probabilities via hierarchical softmax.
    ///
    /// log α[l, j, k] = Σ_d log_softmax(logits_d / τ)[l, group_d(j), child_d(j), k]
    ///
    /// Returns shape (L, p, k).
    pub fn log_alpha(&self) -> Result<Tensor> {
        let p = self.tree.num_variants;
        let (l_dim, _, _, k) = self.logits_per_level[0].dims4()?;
        let dtype = self.logits_per_level[0].dtype();
        let device = self.logits_per_level[0].device();

        let mut total = Tensor::zeros((l_dim, p, k), dtype, device)?;

        for d in 0..self.tree.depth {
            let logits = &self.logits_per_level[d]; // (L, G_d, C_d, k)
            let mask = &self.masks_per_level[d]; // (G_d, C_d)
            let indices = &self.path_indices[d]; // (p,) u32

            // Apply temperature
            let scaled = if (self.temperature - 1.0).abs() > 1e-10 {
                (logits / self.temperature)?
            } else {
                logits.clone()
            };

            // Apply mask: set invalid children to -inf before softmax
            // mask is (G_d, C_d), need to broadcast to (L, G_d, C_d, k)
            let neg_inf_mask = ((1.0 - mask)? * (-1e30))?.unsqueeze(0)?.unsqueeze(3)?; // (1, G_d, C_d, 1)
            let masked_logits = scaled.broadcast_add(&neg_inf_mask)?; // (L, G_d, C_d, k)

            // Log-softmax over children dimension (dim=2)
            let log_sm = candle_nn::ops::log_softmax(&masked_logits, 2)?; // (L, G_d, C_d, k)

            // Reshape to (L, G_d * C_d, k) for gather
            let (_, g_d, c_d, _) = log_sm.dims4()?;
            let flat = log_sm.reshape((l_dim, g_d * c_d, k))?; // (L, G_d*C_d, k)

            // Gather: select flat[l, indices[j], k] for each variant j
            // index_select on dim=1 with indices of shape (p,)
            let gathered = flat.index_select(indices, 1)?; // (L, p, k)

            total = (total + gathered)?;
        }

        Ok(total)
    }

    /// Get selection probabilities α = exp(log_alpha).
    /// Returns shape (L, p, k).
    pub fn alpha(&self) -> Result<Tensor> {
        self.log_alpha()?.exp()
    }

    /// Get posterior inclusion probabilities.
    /// PIP[j, k] = 1 - Π_l (1 - α[l, j, k])
    /// Returns shape (p, k).
    pub fn pip(&self) -> Result<Tensor> {
        let alpha = self.alpha()?;
        let one_minus_alpha = (1.0 - &alpha)?.clamp(1e-10, 1.0)?;
        let log_one_minus = one_minus_alpha.log()?;
        let sum_log = log_one_minus.sum(0)?; // (p, k)
        1.0 - sum_log.exp()?
    }

    /// Get the mean of θ: E[θ_j] = Σ_l α_l[j] · μ_l[j]
    /// Returns shape (p, k).
    pub fn theta_mean(&self) -> Result<Tensor> {
        let alpha = self.alpha()?; // (L, p, k)
        let weighted = alpha.broadcast_mul(&self.beta_mean)?; // (L, p, k)
        weighted.sum(0) // (p, k)
    }

    /// Set temperature (for annealing during training).
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Get current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get effect size means per component. Shape (L, p, k).
    pub fn beta_mean(&self) -> &Tensor {
        &self.beta_mean
    }

    /// Get effect size stds per component. Shape (L, p, k).
    pub fn beta_std(&self) -> Result<Tensor> {
        self.beta_ln_std.exp()
    }

    /// Get number of components L.
    pub fn num_components(&self) -> usize {
        self.num_components
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        self.logits_per_level[0].device()
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.logits_per_level[0].dtype()
    }
}

impl VariationalDistribution for MultiLevelSusieVar {
    /// Get the mean of θ.
    fn mean(&self) -> Result<Tensor> {
        self.theta_mean()
    }

    /// Get the variance of θ.
    /// Var[θ_j] = Σ_l [α_l[j] · (σ²_l[j] + μ²_l[j])] - (Σ_l α_l[j] · μ_l[j])²
    fn var(&self) -> Result<Tensor> {
        let alpha = self.alpha()?; // (L, p, k)
        let mu = &self.beta_mean; // (L, p, k)
        let sigma_sq = (&self.beta_ln_std * 2.0)?.exp()?; // (L, p, k)

        let mu_sq = mu.sqr()?; // (L, p, k)

        // E[θ²] = Σ_l α_l[j] · (σ²_l[j] + μ²_l[j])
        let second_moment_l = alpha.broadcast_mul(&(&sigma_sq + &mu_sq)?)?; // (L, p, k)
        let second_moment = second_moment_l.sum(0)?; // (p, k)

        // E[θ]² = (Σ_l α_l[j] · μ_l[j])²
        let first_moment_l = alpha.broadcast_mul(mu)?; // (L, p, k)
        let first_moment = first_moment_l.sum(0)?; // (p, k)
        let first_moment_sq = first_moment.sqr()?;

        (second_moment - first_moment_sq)?.clamp(1e-8, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_multilevel_susie_shapes() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let l = 3;
        let p = 100;
        let k = 2;

        let tree = VariantTree::regular(p, 10);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = MultiLevelSusieVar::new(vb, tree, l, k, 1.0)?;

        let alpha = susie.alpha()?;
        assert_eq!(alpha.dims(), &[l, p, k]);

        let pip = susie.pip()?;
        assert_eq!(pip.dims(), &[p, k]);

        let theta_mean = susie.theta_mean()?;
        assert_eq!(theta_mean.dims(), &[p, k]);

        let var = susie.var()?;
        assert_eq!(var.dims(), &[p, k]);

        Ok(())
    }

    #[test]
    fn test_alpha_sums_to_one() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 3;
        let p = 50;
        let k = 2;

        let tree = VariantTree::regular(p, 10);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = MultiLevelSusieVar::new(vb, tree, l, k, 1.0)?;

        let alpha = susie.alpha()?; // (L, p, k)
        let alpha_sum = alpha.sum(1)?; // (L, k)

        for i in 0..l {
            for j in 0..k {
                let sum: f64 = alpha_sum.get(i)?.get(j)?.to_scalar()?;
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Alpha should sum to 1 for l={}, k={}, got {}",
                    i,
                    j,
                    sum
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_alpha_sums_with_uneven_groups() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let l = 2;
        let p = 23;
        let k = 1;

        let tree = VariantTree::regular(p, 10);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = MultiLevelSusieVar::new(vb, tree, l, k, 1.0)?;

        let alpha = susie.alpha()?;
        let alpha_sum = alpha.sum(1)?;

        for i in 0..l {
            let sum: f64 = alpha_sum.get(i)?.get(0)?.to_scalar()?;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Alpha should sum to 1 for l={}, got {}",
                i,
                sum
            );
        }

        Ok(())
    }

    #[test]
    fn test_temperature_effect() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        let p = 20;
        let k = 1;

        let tree = VariantTree::regular(p, 5);

        // With uniform logits (init=0), temperature doesn't change uniform distribution
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let susie_sharp = MultiLevelSusieVar::new(vb, tree.clone(), 1, k, 0.1)?;
        let alpha_sharp = susie_sharp.alpha()?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let susie_smooth = MultiLevelSusieVar::new(vb, tree, 1, k, 10.0)?;
        let alpha_smooth = susie_smooth.alpha()?;

        // Both sum to 1
        let sum_sharp: f64 = alpha_sharp.sum(1)?.get(0)?.get(0)?.to_scalar()?;
        let sum_smooth: f64 = alpha_smooth.sum(1)?.get(0)?.get(0)?.to_scalar()?;
        assert!((sum_sharp - 1.0).abs() < 1e-5);
        assert!((sum_smooth - 1.0).abs() < 1e-5);

        // Both uniform at init
        let val: f64 = alpha_sharp.get(0)?.get(0)?.get(0)?.to_scalar()?;
        assert!(
            (val - 1.0 / p as f64).abs() < 1e-5,
            "Expected ~{}, got {}",
            1.0 / p as f64,
            val
        );

        Ok(())
    }

    #[test]
    fn test_with_linear_model() -> Result<()> {
        use crate::sgvb::traits::BlackBoxLikelihood;
        use crate::sgvb::{local_reparam_loss, GaussianPrior, LinearModelSGVB, SGVBConfig};

        let device = Device::Cpu;
        let dtype = DType::F32;

        let n = 30;
        let p = 20;
        let k = 1;
        let l = 2;

        let x = Tensor::randn(0f32, 1f32, (n, p), &device)?;
        let y = Tensor::randn(0f32, 1f32, (n, k), &device)?;

        let tree = VariantTree::regular(p, 5);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = MultiLevelSusieVar::new(vb.pp("susie"), tree, l, k, 1.0)?;

        let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
        let config = SGVBConfig::default();

        let model = LinearModelSGVB::from_variational(susie, x, prior, config);

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
        let likelihood = GaussianLik { y };

        let loss = local_reparam_loss(&model, &likelihood, 10, 1.0)?;
        assert!(loss.dims().is_empty());

        let eta_mean = model.eta_mean()?;
        assert_eq!(eta_mean.dims(), &[n, k]);

        let coef_mean = model.coef_mean()?;
        assert_eq!(coef_mean.dims(), &[p, k]);

        Ok(())
    }
}
