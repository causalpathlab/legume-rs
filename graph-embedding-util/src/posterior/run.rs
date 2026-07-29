//! The `--posterior` / `--mcmc` flag group.
//!
//! Orchestration used to live here too, as a post-hoc pass that sampled gene
//! anchors against the frozen **cell** side. That is gone: MCMC in this crate runs
//! at the pseudobulk level ([`super::pb_gibbs`]), inside `fit`, before phase 2.
//! What remains is the flag both CLIs flatten — a `clap` `Args`, the same pattern
//! `data_beans::qc_lib::QcArgs` uses — so the two keep one help text and one
//! resolution rule.

/// Default retained draws per chain when `--posterior` is given without `--mcmc`.
pub const DEFAULT_SAMPLES: usize = 200;

/// Default frozen negative-slate size — the number of other-side rows summed in
/// the Poisson rate normalizer. It is the Monte-Carlo resolution of that
/// normalizer, not a subsample of the data: every observed count still enters
/// exactly.
pub const DEFAULT_PARTITION: usize = 1024;

/// The `--posterior` / `--mcmc` flag group, flattened by each CLI's args struct.
///
/// The help covers both callers. Anything true of only one of them — gem's second
/// gate, bge's co-embed shrinkage — belongs on that CLI's own flag, not here.
#[derive(clap::Args, Debug, Clone)]
pub struct PosteriorArgs {
    #[arg(
        long = "posterior",
        alias = "mcmc",
        alias = "jitter",
        value_name = "N",
        num_args = 0..=1,
        default_missing_value = "200",
        help = "Sample phase 1 instead of stopping at its point estimate, N retained\n\
                sweeps (bare --posterior uses 200). Off unless given.",
        long_help = "Replace phase 1's SGD point estimate with a sampled one. N is the number of\n\
                     RETAINED sweeps (warmup is N/2 more), the sampler's analogue of `-i/--epochs`.\n\
                     Bare `--posterior` uses 200. Omit it and the run is byte-identical to one\n\
                     built before this existed.\n\
                     \n\
                     This is a two-sided blocked Gibbs over the PSEUDOBULK model, warm-started\n\
                     from the SGD fit and run before phase 2. It alternates the gene side given\n\
                     the pseudobulks and the pseudobulks given the genes, over every collapse\n\
                     level at once, and writes its posterior means back into the model — so the\n\
                     dictionary, phase 2 and the co-embedding all read the sampled fit rather\n\
                     than a second set of tables.\n\
                     \n\
                     Selection lives on the feature side: each (gene, dim) gets a posterior\n\
                     inclusion probability, alongside the per-dim slab variance σ₀h² and sparsity\n\
                     π₀h learned from the data. Inclusion is INDEPENDENT per dim, so a gene may\n\
                     load several and its row does NOT sum to 1. A pseudobulk is a location, not\n\
                     a selection, so that side is sampled without a spike-and-slab.\n\
                     \n\
                     A model carrying more than one feature-side gate has every one of them\n\
                     sampled, each with its own σ₀h² and π₀h. They are different objects — a\n\
                     loading and a deviation from it are not on one scale — so a single shared\n\
                     sparsity prior would force them to agree.\n\
                     \n\
                     READ THE OUTPUT AGAINST THE EFFECTIVE RANK. When the embedding dimension\n\
                     far exceeds the rank the embedding actually uses, the likelihood carries no\n\
                     information about the surplus dims and their inclusion indicators simply\n\
                     reproduce the prior — every gene then loads something and the probabilities\n\
                     stop discriminating. The run reports effective rank, per-dim hypers and\n\
                     their ESS so that case is visible.\n\
                     \n\
                     Cost scales with H likelihood passes per anchor per sweep, so a whole\n\
                     dictionary runs for a long time — Ctrl+C returns partial results, and a\n\
                     smaller N is the way to shorten an exploratory run.\n\
                     Writes {out}.feature_pip.parquet + {out}.feature_posterior_mean.parquet\n\
                     (gem: one pair per gate, keyed by gene).\n\
                     `--mcmc` and `--jitter` are accepted aliases."
    )]
    pub posterior: Option<usize>,
}

/// A resolved posterior request. Reaching this type at all means the posterior
/// is ON — [`PosteriorArgs::resolve`] returns `None` for `off`, so no downstream
/// code has to re-test for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosteriorPlan {
    /// Retained draws per chain; warmup is `n_samples / 2` on top.
    pub n_samples: usize,
    /// Base seed, so a reproducible fit gives a reproducible posterior. Each
    /// sampler derives a distinct stream from it.
    pub seed: u64,
}

impl PosteriorArgs {
    /// Resolve `--posterior [N]` into a plan. `None` means the posterior is off,
    /// so no downstream code has to re-test for it.
    pub fn resolve(&self, seed: u64) -> anyhow::Result<Option<PosteriorPlan>> {
        let Some(n_samples) = self.posterior else {
            return Ok(None);
        };
        anyhow::ensure!(n_samples > 0, "--posterior must be > 0 (got {n_samples})");
        Ok(Some(PosteriorPlan { n_samples, seed }))
    }
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;
