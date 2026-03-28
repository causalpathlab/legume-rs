use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

use super::model::McmcModel;

/// Configuration for the MCMC runner.
pub struct McmcConfig {
    /// Number of posterior samples to collect.
    pub n_samples: usize,
    /// Warmup (burn-in) iterations.
    pub warmup: usize,
    /// Thinning interval.
    pub thin: usize,
    /// RNG seed.
    pub seed: u64,
}

/// Run a single MCMC chain.
pub fn run_mcmc<M: McmcModel>(model: &M, config: &McmcConfig) -> M::Result {
    let total = config.warmup + config.n_samples * config.thin;
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut state = model.init(&mut rng);
    let mut samples = Vec::with_capacity(config.n_samples);

    for iter in 0..total {
        model.sweep(&mut state, &mut rng);
        if iter >= config.warmup && (iter - config.warmup) % config.thin == 0 {
            samples.push(model.collect(&state));
        }
    }

    model.summarize(samples)
}

/// Run multiple independent chains in parallel.
/// Each chain gets `seed + chain_idx` for reproducibility.
pub fn run_mcmc_parallel<M: McmcModel + Sync>(
    model: &M,
    config: &McmcConfig,
    n_chains: usize,
) -> Vec<M::Result>
where
    M::State: Send,
    M::Result: Send,
{
    (0..n_chains)
        .into_par_iter()
        .map(|i| {
            let chain_config = McmcConfig {
                n_samples: config.n_samples,
                warmup: config.warmup,
                thin: config.thin,
                seed: config.seed.wrapping_add(i as u64),
            };
            run_mcmc(model, &chain_config)
        })
        .collect()
}
