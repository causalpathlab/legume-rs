use anyhow::Result;
use candle_util::candle_core::Tensor;
use candle_util::candle_nn::VarBuilder;
use candle_util::sgvb::{
    BiSusieVar, FixedGaussianPrior, IndependentGateVariational, LocalReparamSample, RegressionSGVB,
    SGVBConfig, SpikeSlabVar, SusieVar, VariationalDistribution,
};

use super::config::ModelType;
use super::training::local_reparam_with_design;

pub(crate) enum GeneticModel {
    Susie(RegressionSGVB<SusieVar, FixedGaussianPrior>),
    BiSusie(RegressionSGVB<BiSusieVar, FixedGaussianPrior>),
    SpikeSlab(RegressionSGVB<SpikeSlabVar, FixedGaussianPrior>),
}

pub(crate) struct GeneticModelSpec<'a> {
    pub vb: &'a VarBuilder<'a>,
    pub x_design: Tensor,
    pub sgvb_config: SGVBConfig,
    pub model_type: ModelType,
    pub num_components: usize,
    pub p: usize,
    pub k: usize,
    pub init_prior_std: f32,
}

impl GeneticModel {
    pub fn new(spec: GeneticModelSpec) -> Result<Self> {
        let prior = FixedGaussianPrior::new(spec.init_prior_std);
        Ok(match spec.model_type {
            ModelType::Susie => {
                let var_dist = SusieVar::new_with_null(
                    spec.vb.pp("susie"),
                    spec.num_components,
                    spec.p,
                    spec.k,
                )?;
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
            ModelType::SpikeSlab => {
                let var_dist = SpikeSlabVar::new(spec.vb.pp("ss"), spec.p, spec.k, 0.01)?;
                Self::SpikeSlab(RegressionSGVB::from_variational(
                    var_dist,
                    spec.x_design,
                    prior,
                    spec.sgvb_config,
                ))
            }
        })
    }

    pub fn local_reparam_sample(
        &self,
        num_samples: usize,
        x_batch: &Tensor,
    ) -> Result<LocalReparamSample> {
        match self {
            Self::Susie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::BiSusie(m) => local_reparam_with_design(m, num_samples, x_batch),
            Self::SpikeSlab(m) => local_reparam_with_design(m, num_samples, x_batch),
        }
    }

    /// Selection KL: categorical for SuSiE, Bernoulli for spike-slab.
    pub fn kl_selection(&self, prior_alpha: f64) -> Result<Tensor> {
        Ok(match self {
            Self::Susie(m) => m.variational.kl_categorical(prior_alpha)?,
            Self::BiSusie(m) => m.variational.kl_categorical(prior_alpha)?,
            Self::SpikeSlab(m) => m.variational.kl_bernoulli(prior_alpha)?,
        })
    }

    pub fn extract_results(&self) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Susie(m) => extract_flat_results(m.variational.pip()?, &m.variational),
            Self::BiSusie(m) => extract_flat_results(m.variational.pip()?, &m.variational),
            Self::SpikeSlab(m) => extract_flat_results(m.variational.pip()?, &m.variational),
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
