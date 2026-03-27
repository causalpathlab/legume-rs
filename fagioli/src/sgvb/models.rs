use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::VarBuilder;
use candle_util::sgvb::{
    block_partition::BlockPartition, BiSusieVar, FixedGaussianPrior, LocalReparamSample,
    MultilevelSusieSGVB, RegressionSGVB, SGVBConfig, SusieVar, VariationalDistribution,
};

use super::config::ModelType;
use super::training::local_reparam_with_design;

pub(crate) enum GeneticModel {
    Susie(RegressionSGVB<SusieVar, FixedGaussianPrior>),
    BiSusie(RegressionSGVB<BiSusieVar, FixedGaussianPrior>),
    MultilevelSusie(MultilevelSusieSGVB<FixedGaussianPrior>),
}

pub(crate) struct GeneticModelSpec<'a> {
    pub vb: &'a VarBuilder<'a>,
    pub x_design: Tensor,
    pub prior: FixedGaussianPrior,
    pub sgvb_config: SGVBConfig,
    pub model_type: ModelType,
    pub num_components: usize,
    pub p: usize,
    pub k: usize,
    pub partitions: Option<&'a [BlockPartition]>,
}

impl GeneticModel {
    /// When `spec.partitions` is `Some`, builds a `MultilevelSusieSGVB` using the
    /// provided LD-aware partition hierarchy. Otherwise builds a flat model.
    pub fn new(spec: GeneticModelSpec) -> Result<Self> {
        if let Some(parts) = spec.partitions {
            anyhow::ensure!(
                spec.model_type == ModelType::Susie,
                "Multilevel regression only supports SuSiE (not BiSuSiE)"
            );
            let model = MultilevelSusieSGVB::new_with_partitions(
                spec.vb.pp("ml_susie"),
                spec.x_design,
                spec.prior,
                parts.to_vec(),
                spec.num_components,
                spec.k,
                spec.sgvb_config,
            )?;
            return Ok(Self::MultilevelSusie(model));
        }
        Ok(match spec.model_type {
            ModelType::Susie => {
                let var_dist =
                    SusieVar::new(spec.vb.pp("susie"), spec.num_components, spec.p, spec.k)?;
                Self::Susie(RegressionSGVB::from_variational(
                    var_dist,
                    spec.x_design,
                    spec.prior,
                    spec.sgvb_config,
                ))
            }
            ModelType::BiSusie => {
                let var_dist =
                    BiSusieVar::new(spec.vb.pp("bisusie"), spec.num_components, spec.p, spec.k)?;
                Self::BiSusie(RegressionSGVB::from_variational(
                    var_dist,
                    spec.x_design,
                    spec.prior,
                    spec.sgvb_config,
                ))
            }
        })
    }

    /// For flat models, `x_batch` is the (possibly subsampled) design matrix.
    /// For multilevel, `x_batch` is ignored — the model uses its internal design.
    pub fn local_reparam_sample(
        &self,
        num_samples: usize,
        x_batch: &Tensor,
    ) -> Result<LocalReparamSample> {
        match self {
            Self::Susie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::BiSusie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::MultilevelSusie(m) => Ok(m.forward(num_samples)?),
        }
    }

    pub fn kl_categorical(&self, prior_alpha: f64, device: &Device) -> Result<Tensor> {
        match self {
            Self::Susie(m) => Ok(m.variational.kl_categorical(prior_alpha)?),
            Self::BiSusie(_) => Ok(Tensor::zeros((), DType::F32, device)?),
            // Multilevel model includes all categorical KL in its forward() KL term
            Self::MultilevelSusie(_) => Ok(Tensor::zeros((), DType::F32, device)?),
        }
    }

    pub fn extract_results(&self) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Susie(m) => extract_flat_results(m.variational.pip()?, &m.variational),
            Self::BiSusie(m) => extract_flat_results(m.variational.pip()?, &m.variational),
            Self::MultilevelSusie(m) => {
                let pip = m.joint_pip()?;
                let (eff_mean, eff_std) = multilevel_effects(m)?;
                Ok((pip, eff_mean, eff_std))
            }
        }
    }
}

fn extract_flat_results(
    pip: Tensor,
    var_dist: &impl VariationalDistribution,
) -> Result<(Tensor, Tensor, Tensor)> {
    let eff_mean = var_dist.mean()?;
    let eff_std = var_dist.var()?.sqrt()?;
    Ok((pip, eff_mean, eff_std))
}

/// Compute per-feature effect mean and std from a multilevel model.
///
/// Expands the top-level group effects back to per-feature space using
/// the joint selection probabilities from all hierarchy levels.
fn multilevel_effects(model: &MultilevelSusieSGVB<FixedGaussianPrior>) -> Result<(Tensor, Tensor)> {
    let joint_alpha = model.joint_alpha()?; // (L, p, k)
    let num_levels = model.num_levels();
    let top = model.level(num_levels - 1);
    let top_mu = top.variational.beta_mean(); // (L, G_top, k)
    let top_sigma_sq = top.variational.beta_std()?.sqr()?; // (L, G_top, k)

    // Build feature → top-level group mapping by composing partitions bottom-up.
    let (_, p, _) = joint_alpha.dims3()?;
    let mut group_of_feature: Vec<u32> = (0..p as u32).collect();
    for d in 0..(num_levels - 1) {
        let block_map = model.level(d).partition.feature_to_block();
        for g in group_of_feature.iter_mut() {
            *g = block_map[*g as usize];
        }
    }

    let idx = Tensor::from_vec(group_of_feature, (p,), joint_alpha.device())?;

    // Expand top-level betas to per-feature: (L, G_top, k) → (L, p, k)
    let l_dim = model.num_components();
    let mu_sq = top_mu.sqr()?;
    let second_moment = (&top_sigma_sq + &mu_sq)?;

    let expand = |tensor: &Tensor| -> candle_util::candle_core::Result<Tensor> {
        let parts: Vec<Tensor> = (0..l_dim)
            .map(|l| tensor.get(l)?.index_select(&idx, 0)?.unsqueeze(0))
            .collect::<candle_util::candle_core::Result<_>>()?;
        Tensor::cat(&parts, 0)
    };
    let expanded_mu = expand(top_mu)?;
    let expanded_second = expand(&second_moment)?;

    let eff_mean = joint_alpha.broadcast_mul(&expanded_mu)?.sum(0)?;
    let eff_var = (joint_alpha.broadcast_mul(&expanded_second)?.sum(0)? - eff_mean.sqr()?)?
        .clamp(1e-8, f64::INFINITY)?;

    Ok((eff_mean, eff_var.sqrt()?))
}
