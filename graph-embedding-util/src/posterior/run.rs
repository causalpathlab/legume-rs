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

    #[arg(
        long = "stick-alpha",
        value_name = "ALPHA",
        default_value_t = crate::posterior::dim_block::DEFAULT_STICK_ALPHA,
        requires = "posterior",
        help = "Truncated-IBP concentration α — the expected number of dims a feature\n\
                loads. Independent of --embedding-dim.",
        long_help = "Concentration α for the TRUNCATED INDIAN BUFFET PROCESS on the per-dim\n\
                     inclusion rates, which is the DEFAULT selection prior.\n\
                     \n\
                     Stick-breaking (Teh, Görür & Ghahramani 2007) draws v_j ~ Beta(α, 1) and\n\
                     sets each dim's inclusion rate to the running product ∏_{j≤h} v_j, so the\n\
                     rates DECREASE with the dim index and surplus dims are squeezed off by\n\
                     construction. α is the expected number of dims a feature loads, and it is\n\
                     independent of --embedding-dim: measured on BM1, doubling H from 16 to 32\n\
                     moved the active-dim count only 10 -> 12, against 16 -> 32 under the\n\
                     unordered alternative. So H is a TRUNCATION, not a tuning knob.\n\
                     \n\
                     WHY IT IS THE DEFAULT. With tens of thousands of features on every dim, an\n\
                     independent Beta prior carries O(1) pseudo-counts against O(10^4)\n\
                     observations and is swamped, so every unused dim re-estimates the SAME\n\
                     rate rather than collapsing — measured flat at 0.787-0.930 across 16 dims\n\
                     while the likelihood supported ~3.4 of them. Stick-breaking is a\n\
                     structural constraint, which data cannot outvote. It also mixes BETTER on\n\
                     the sparsity parameter (split-R-hat 3.09 -> 1.35), because each stick\n\
                     pools across every dim above it.\n\
                     \n\
                     A SIDE EFFECT WORTH KNOWING: the dims become ordered, which removes the\n\
                     dim-permutation gauge and makes them comparable across runs the way PCA\n\
                     components are. And a feature with NO counts now reverts to each dim's\n\
                     population rate rather than to a flat null — on the leading dim that is\n\
                     about a coin flip at α = 1. Lower α if that matters for your read.\n\
                     \n\
                     Pass --no-stick-breaking for the previous independent-Beta-per-dim prior."
    )]
    pub stick_alpha: f64,

    #[arg(
        long = "no-stick-breaking",
        requires = "posterior",
        help = "Use an independent Beta prior per dim instead of the truncated IBP.",
        long_help = "Revert the selection prior to an independent Beta(a,b) on every dim — no\n\
                     ordering, no coupling between dims. This is what shipped before the IBP\n\
                     became the default, and it is kept for A/B: with many features per dim it\n\
                     is swamped by the data, so it neither imposes sparsity nor collapses\n\
                     unused dims. See --stick-alpha."
    )]
    pub no_stick_breaking: bool,
}

/// A resolved posterior request. Reaching this type at all means the posterior
/// is ON — [`PosteriorArgs::resolve`] returns `None` for `off`, so no downstream
/// code has to re-test for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorPlan {
    /// Retained draws per chain; warmup is `n_samples / 2` on top.
    pub n_samples: usize,
    /// Base seed, so a reproducible fit gives a reproducible posterior. Each
    /// sampler derives a distinct stream from it.
    pub seed: u64,
    /// `Some(α)` = truncated-IBP stick-breaking on the per-dim inclusion rates.
    pub stick_alpha: Option<f64>,
}

impl PosteriorArgs {
    /// Resolve `--posterior [N]` into a plan. `None` means the posterior is off,
    /// so no downstream code has to re-test for it.
    pub fn resolve(&self, seed: u64) -> anyhow::Result<Option<PosteriorPlan>> {
        let Some(n_samples) = self.posterior else {
            return Ok(None);
        };
        anyhow::ensure!(n_samples > 0, "--posterior must be > 0 (got {n_samples})");
        anyhow::ensure!(
            self.stick_alpha > 0.0 && self.stick_alpha.is_finite(),
            "--stick-alpha must be a positive, finite concentration (got {})",
            self.stick_alpha
        );
        Ok(Some(PosteriorPlan {
            n_samples,
            seed,
            stick_alpha: (!self.no_stick_breaking).then_some(self.stick_alpha),
        }))
    }
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;
