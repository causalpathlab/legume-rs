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
        help = "SAMPLE the phase-1 feature SELECTION, then fit the loading by SGD\n\
                under it — N retained sweeps (bare --posterior uses 200).",
        long_help = "The sampler chooses WHICH features load each dim; SGD then fits HOW MUCH.\n\
                     N is the number of RETAINED sweeps (warmup is N/2 more). Bare\n\
                     `--posterior` uses 200.\n\
                     \n\
                     WHY BOTH, rather than one or the other. A learned gate does not train:\n\
                     the KL that would drive selection sits far below the true ELBO, the\n\
                     sigmoid passes a fraction of the gradient wherever the gate is inert,\n\
                     and — decisively — a large share of features are never drawn as NCE\n\
                     positives at all, so their gate receives EXACTLY zero gradient and\n\
                     reports its initialization forever. The sampler has no such blind spot:\n\
                     its column pass touches every anchor on every sweep. So selection comes\n\
                     from sampling, where it works, and the loading from SGD, which is far\n\
                     faster and moves to the GPU.\n\
                     \n\
                     The inclusion probabilities are applied as a per-(feature, dim) mask:\n\
                     `z ~ Bern(pip)`, redrawn once per EPOCH (not per minibatch — z is a\n\
                     latent for the dataset), with the mean `pip` used at output so the\n\
                     written dictionary matches what training averaged over. Features with\n\
                     pip 0 are masked permanently: their loading never trains and the\n\
                     dictionary carries exact zeros.\n\
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
                     Writes one PIP table and one posterior-mean table PER GATE, keyed by the\n\
                     feature the gate is over. A single-gate model gets\n\
                     {out}.feature_pip.parquet and {out}.feature_posterior_mean.parquet; a model\n\
                     with two gates (gem's identity β_g and velocity δ_g) gets\n\
                     {out}.beta_pip.parquet, {out}.beta_posterior_mean.parquet,\n\
                     {out}.delta_pip.parquet and {out}.delta_posterior_mean.parquet, keyed by\n\
                     gene rather than by feature row.\n\
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
