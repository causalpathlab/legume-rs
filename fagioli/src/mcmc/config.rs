/// MCMC prior type for sparse regression.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum McmcPriorType {
    Susie,
    #[value(alias = "spikeslab", alias = "ss")]
    SpikeSlab,
}

/// Configuration for MCMC block fitting.
#[derive(Debug, Clone)]
pub struct McmcFitConfig {
    pub prior_type: McmcPriorType,
    pub num_components: usize,
    pub prior_var: f32,
    pub logit_var: f32,
    pub n_samples: usize,
    pub warmup: usize,
    pub thin: usize,
    pub estimate_effect_var: bool,
}
