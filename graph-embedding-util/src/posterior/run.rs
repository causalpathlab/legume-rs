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
        help = "SAMPLE phase 1 instead of training it — N retained sweeps, and SGD\n\
                does not run (bare --posterior uses 200). Off unless given.",
        long_help = "Phase 1 is SGD XOR sampling, and this flag picks sampling. SGD does NOT\n\
                     run: --epochs keeps its value but stops applying to this phase. N is the\n\
                     number of RETAINED sweeps (warmup is N/2 more), the sampler's analogue of\n\
                     `-i/--epochs`. Bare `--posterior` uses 200. Omit the flag entirely and the\n\
                     run is byte-identical to one built before this existed.\n\
                     \n\
                     WHY EXCLUSIVE, rather than sampling around the SGD fit. An initialization\n\
                     cannot bias a CONVERGED chain — it only sets burn-in — so warm-starting\n\
                     from an optimum is either harmless or fatal. It is fatal here: when the\n\
                     embedding's effective rank is far below its dimension the surplus\n\
                     directions have a flat likelihood, the chain random-walks in them, and\n\
                     nothing washes out at any practical sweep count. The output would then\n\
                     describe curvature around wherever SGD landed rather than a posterior. The\n\
                     price is that burn-in is now yours to check, which is what the reported R̂\n\
                     is for.\n\
                     \n\
                     This is a two-sided blocked Gibbs over the PSEUDOBULK model, run before\n\
                     phase 2. It alternates the gene side given the pseudobulks and the\n\
                     pseudobulks given the genes, over every collapse level at once, and writes\n\
                     its posterior means back into the model — so the dictionary, phase 2 and\n\
                     the co-embedding all read the sampled fit rather than a second set of\n\
                     tables. Phase 2 still runs: it is an analytical Poisson-MAP projection, not\n\
                     SGD.\n\
                     \n\
                     Requires the pure-pseudobulk phase 1 (--phase1-cells-per-pb 0, the\n\
                     default): a cell axis is trained only by SGD, so there is no cell block to\n\
                     sample and one would be left at its initialization. Cannot be combined\n\
                     with --lineage-dag, which refines a trained fit. Requires\n\
                     --nce-objective softmax: the sampled likelihood is the profiled Poisson,\n\
                     which is the same estimand as sampled-softmax but not as logistic, so\n\
                     sampling a logistic fit would report a posterior for a different model.\n\
                     All three are hard errors rather than silent degradations.\n\
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
                     Cost is one column pass per dim per sweep over the whole anchor axis, so a\n\
                     full dictionary runs for a long time — Ctrl+C returns partial results, and\n\
                     a smaller N is the way to shorten an exploratory run. The run reports its\n\
                     own bracket-fallback count; a large fraction there means coordinates are\n\
                     stalling rather than moving, and the numbers should not be read as a\n\
                     converged posterior.\n\
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
