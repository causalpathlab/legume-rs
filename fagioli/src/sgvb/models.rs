use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::VarBuilder;
use candle_util::sgvb::{
    block_partition::BlockPartition, AnalyticalKL, BiSusieVar, FixedGaussianPrior, GaussianPrior,
    LocalReparamSample, MultilevelPartitionParams, MultilevelSusieSGVB, Prior, RegressionSGVB,
    SGVBConfig, SusieVar, VariationalDistribution,
};

use super::config::ModelType;
use super::training::local_reparam_with_design;

pub(crate) enum GeneticModel {
    Susie(RegressionSGVB<SusieVar, FixedGaussianPrior>),
    BiSusie(RegressionSGVB<BiSusieVar, FixedGaussianPrior>),
    MultilevelSusie(MultilevelSusieSGVB<FixedGaussianPrior>),
    // Learnable per-component prior variants
    SusieLearnable {
        model: RegressionSGVB<SusieVar, FixedGaussianPrior>,
        component_priors: Vec<GaussianPrior>,
    },
    MultilevelSusieLearnable {
        model: MultilevelSusieSGVB<FixedGaussianPrior>,
        component_priors: Vec<GaussianPrior>,
    },
}

pub(crate) struct GeneticModelSpec<'a> {
    pub vb: &'a VarBuilder<'a>,
    pub x_design: Tensor,
    pub sgvb_config: SGVBConfig,
    pub model_type: ModelType,
    pub num_components: usize,
    pub p: usize,
    pub k: usize,
    pub partitions: Option<&'a [BlockPartition]>,
    pub init_prior_std: f32,
    pub learn_prior_var: bool,
}

/// Dummy prior τ for learnable variants. The KL from this prior is zeroed out
/// in `local_reparam_sample` — the actual value is irrelevant. Set large as a
/// safety net in case the zeroing is accidentally removed.
const DUMMY_PRIOR_TAU: f32 = 100.0;

impl GeneticModel {
    pub fn new(spec: GeneticModelSpec) -> Result<Self> {
        if spec.learn_prior_var {
            return Self::new_learnable(spec);
        }
        let prior = FixedGaussianPrior::new(spec.init_prior_std);
        if let Some(parts) = spec.partitions {
            anyhow::ensure!(
                spec.model_type == ModelType::Susie,
                "Multilevel regression only supports SuSiE (not BiSuSiE)"
            );
            let model = MultilevelSusieSGVB::new_with_partitions(
                spec.vb.pp("ml_susie"),
                spec.x_design,
                prior,
                parts.to_vec(),
                MultilevelPartitionParams {
                    num_components: spec.num_components,
                    k: spec.k,
                    config: spec.sgvb_config,
                    gate_epsilon: None,
                },
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
                    prior,
                    spec.sgvb_config,
                ))
            }
            ModelType::BiSusie => {
                let var_dist =
                    BiSusieVar::new(spec.vb.pp("bisusie"), spec.num_components, spec.p, spec.k)?;
                Self::BiSusie(RegressionSGVB::from_variational(
                    var_dist,
                    spec.x_design,
                    prior,
                    spec.sgvb_config,
                ))
            }
        })
    }

    fn new_learnable(spec: GeneticModelSpec) -> Result<Self> {
        anyhow::ensure!(
            spec.model_type == ModelType::Susie,
            "Learnable prior variance only supports SuSiE (not BiSuSiE)"
        );

        // Per-component learnable priors
        let component_priors: Vec<GaussianPrior> = (0..spec.num_components)
            .map(|l| GaussianPrior::new(spec.vb.pp(format!("prior_{l}")), spec.init_prior_std))
            .collect::<candle_util::candle_core::Result<_>>()?;

        // Use a dummy wide prior in the model — real KL comes from component_priors
        let dummy_prior = FixedGaussianPrior::new(DUMMY_PRIOR_TAU);

        if let Some(parts) = spec.partitions {
            let model = MultilevelSusieSGVB::new_with_partitions(
                spec.vb.pp("ml_susie"),
                spec.x_design,
                dummy_prior,
                parts.to_vec(),
                MultilevelPartitionParams {
                    num_components: spec.num_components,
                    k: spec.k,
                    config: spec.sgvb_config,
                    gate_epsilon: None,
                },
            )?;
            Ok(Self::MultilevelSusieLearnable {
                model,
                component_priors,
            })
        } else {
            let var_dist = SusieVar::new(spec.vb.pp("susie"), spec.num_components, spec.p, spec.k)?;
            let model = RegressionSGVB::from_variational(
                var_dist,
                spec.x_design,
                dummy_prior,
                spec.sgvb_config,
            );
            Ok(Self::SusieLearnable {
                model,
                component_priors,
            })
        }
    }

    pub fn local_reparam_sample(
        &self,
        num_samples: usize,
        x_batch: &Tensor,
    ) -> Result<LocalReparamSample> {
        match self {
            Self::Susie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::BiSusie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::MultilevelSusie(m) => Ok(m.forward(num_samples)?),
            Self::SusieLearnable { model, .. } => {
                // Zero out the dummy prior's KL — real KL comes from per_component_kl()
                let mut sample = local_reparam_with_design(model, num_samples, x_batch)?;
                sample.kl = Tensor::zeros((), DType::F32, sample.kl.device())?;
                Ok(sample)
            }
            Self::MultilevelSusieLearnable { model, .. } => {
                let mut sample = model.forward(num_samples)?;
                sample.kl = Tensor::zeros((), DType::F32, sample.kl.device())?;
                Ok(sample)
            }
        }
    }

    /// Per-component KL divergence for learnable prior variants.
    ///
    /// For fixed-prior variants, returns zero (the KL is already in the sample).
    /// For learnable variants, the sample's KL is zeroed out in `local_reparam_sample`,
    /// so this provides the full KL: Σ_l KL(q(β_l) || N(0, τ_l²)).
    pub fn per_component_kl(&self) -> Result<Tensor> {
        match self {
            Self::SusieLearnable {
                model,
                component_priors,
            } => compute_per_component_kl(&model.variational, component_priors),
            Self::MultilevelSusieLearnable {
                model,
                component_priors,
            } => {
                let var = &model.level(0).variational;
                compute_per_component_kl(var, component_priors)
            }
            _ => {
                let device = self.get_device()?;
                Ok(Tensor::zeros((), DType::F32, &device)?)
            }
        }
    }

    pub fn kl_categorical(&self, prior_alpha: f64, device: &Device) -> Result<Tensor> {
        match self {
            Self::Susie(m) => Ok(m.variational.kl_categorical(prior_alpha)?),
            Self::SusieLearnable { model, .. } => {
                Ok(model.variational.kl_categorical(prior_alpha)?)
            }
            _ => Ok(Tensor::zeros((), DType::F32, device)?),
        }
    }

    pub fn extract_results(&self) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Susie(m) => extract_flat_results(m.variational.pip()?, &m.variational),
            Self::BiSusie(m) => extract_flat_results(m.variational.pip()?, &m.variational),
            Self::SusieLearnable { model, .. } => {
                extract_flat_results(model.variational.pip()?, &model.variational)
            }
            Self::MultilevelSusie(m) => multilevel_results(m),
            Self::MultilevelSusieLearnable { model, .. } => multilevel_results(model),
        }
    }

    fn get_device(&self) -> Result<Device> {
        Ok(match self {
            Self::Susie(m) => m.variational.device().clone(),
            Self::BiSusie(m) => m.variational.device().clone(),
            Self::SusieLearnable { model, .. } => model.variational.device().clone(),
            Self::MultilevelSusie(m) => m.x_design.device().clone(),
            Self::MultilevelSusieLearnable { model, .. } => model.x_design.device().clone(),
        })
    }
}

/// Compute per-component KL: Σ_l KL(q(β_l) || N(0, τ_l²)).
fn compute_per_component_kl(var: &SusieVar, component_priors: &[GaussianPrior]) -> Result<Tensor> {
    let beta_mean = var.beta_mean(); // (L, p, k)
    let beta_var = var.beta_std()?.sqr()?; // (L, p, k)
    let device = var.device();
    let mut total_kl = Tensor::zeros((), DType::F32, device)?;

    for (comp, prior_l) in component_priors.iter().enumerate() {
        let mu_l = beta_mean.get(comp)?;
        let var_l = beta_var.get(comp)?;
        total_kl = (total_kl + prior_l.kl_from_gaussian(&mu_l, &var_l)?)?;
    }

    Ok(total_kl)
}

fn extract_flat_results(
    pip: Tensor,
    var_dist: &impl VariationalDistribution,
) -> Result<(Tensor, Tensor, Tensor)> {
    let eff_mean = var_dist.mean()?;
    let eff_std = var_dist.var()?.sqrt()?;
    Ok((pip, eff_mean, eff_std))
}

fn multilevel_results<P: Prior + AnalyticalKL>(
    model: &MultilevelSusieSGVB<P>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let pip = model.joint_pip()?;
    let joint_alpha = model.joint_alpha()?;
    let num_levels = model.num_levels();
    let top = model.level(num_levels - 1);
    let top_mu = top.variational.beta_mean();
    let top_sigma_sq = top.variational.beta_std()?.sqr()?;

    let (_, p, _) = joint_alpha.dims3()?;
    let mut group_of_feature: Vec<u32> = (0..p as u32).collect();
    for d in 0..(num_levels - 1) {
        let block_map = model.level(d).partition.feature_to_block();
        for g in group_of_feature.iter_mut() {
            *g = block_map[*g as usize];
        }
    }

    let idx = Tensor::from_vec(group_of_feature, (p,), joint_alpha.device())?;
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

    Ok((pip, eff_mean, eff_var.sqrt()?))
}
