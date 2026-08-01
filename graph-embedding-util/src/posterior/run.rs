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
        help = "SAMPLE the phase-1 feature SELECTION; SGD then fits the loading",
        long_help = "The sampler chooses WHICH features load each dim. SGD then fits HOW MUCH.\n\
                     N is the number of RETAINED sweeps; warmup adds N/2 more.\n\
                     A bare `--posterior` uses 200.\n\
                     \n\
                     WHY BOTH, rather than one or the other. A learned gate does not train.\n\
                     The KL that would drive selection sits far below the true ELBO.\n\
                     The sigmoid passes only a fraction of the gradient wherever the gate is inert.\n\
                     Decisively, many features are never drawn as NCE positives.\n\
                     Their gate receives EXACTLY zero gradient.\n\
                     It reports its initialization forever.\n\
                     \n\
                     The sampler has no such blind spot.\n\
                     Its column pass touches every anchor on every sweep.\n\
                     So selection comes from sampling, where it works.\n\
                     The loading comes from SGD, which is far faster, and moves to the GPU.\n\
                     \n\
                     Inclusion probabilities apply as a per-(feature, dim) mask.\n\
                     `z ~ Bern(pip)` is redrawn once per EPOCH, not per minibatch,\n\
                     because z is a latent for the dataset. The mean `pip` is used at output.\n\
                     So the written dictionary matches what training averaged over.\n\
                     \n\
                     Features with pip 0 are masked permanently. Their loading never trains.\n\
                     The dictionary carries exact zeros for them.\n\
                     \n\
                     This is a two-sided blocked Gibbs over the PSEUDOBULK model.\n\
                     It runs before phase 2. It alternates two conditionals,\n\
                     over every collapse level at once: the gene side given the pseudobulks,\n\
                     and the pseudobulks given the genes.\n\
                     It writes its posterior means back into the model. So the dictionary,\n\
                     phase 2 and the co-embedding all read the sampled fit.\n\
                     None of them reads a second set of tables. Phase 2 still runs afterwards,\n\
                     as a block Poisson-MAP SGD.\n\
                     \n\
                     This requires the pure-pseudobulk phase 1. That is --phase1-cells-per-pb 0,\n\
                     the default. A cell axis is trained only by SGD.\n\
                     There is no cell block to sample. One would sit at its initialization.\n\
                     \n\
                     It cannot combine with --lineage-dag. That flag refines a trained fit.\n\
                     \n\
                     It requires --nce-objective softmax.\n\
                     The sampled likelihood is the profiled Poisson.\n\
                     That matches sampled-softmax as an estimand. It does not match logistic.\n\
                     Sampling a logistic fit would report the wrong posterior.\n\
                     \n\
                     All three are hard errors, not silent degradations.\n\
                     \n\
                     Selection lives on the feature side.\n\
                     Each (gene, dim) gets a posterior inclusion probability,\n\
                     alongside the per-dim slab variance σ₀h².\n\
                     Inclusion is INDEPENDENT per dim. A gene may load several,\n\
                     and its row does NOT sum to 1.\n\
                     \n\
                     A pseudobulk is a location, not a selection.\n\
                     That side is therefore sampled without a spike-and-slab.\n\
                     The per-dim inclusion RATES come from the truncated IBP.\n\
                     That is the default; see --stick-alpha.\n\
                     \n\
                     A model may carry more than one feature-side gate. Every one is sampled,\n\
                     each with its own σ₀h² and π₀h. They are different objects.\n\
                     A loading and a deviation from it are not on one scale.\n\
                     A single shared sparsity prior would force them to agree.\n\
                     \n\
                     READ THE OUTPUT AGAINST THE EFFECTIVE RANK.\n\
                     The embedding dimension may far exceed the rank actually used.\n\
                     The likelihood then carries no information about surplus dims.\n\
                     Their inclusion indicators fall back on the prior.\n\
                     \n\
                     Under the default IBP, that prior DECAYS with the dim index.\n\
                     Surplus dims are pushed toward zero, not toward a shared rate.\n\
                     Under --no-stick-breaking they instead reproduce one flat rate,\n\
                     and stop discriminating.\n\
                     \n\
                     The run reports effective rank, per-dim hypers and their ESS.\n\
                     The case is visible either way.\n\
                     \n\
                     Cost is one column pass per dim per sweep.\n\
                     It covers the whole anchor axis, so a full dictionary runs long.\n\
                     Ctrl+C returns partial results.\n\
                     A smaller N is the way to shorten an exploratory run.\n\
                     \n\
                     The run reports its own bracket-fallback count.\n\
                     A large fraction there means coordinates are stalling, rather than moving.\n\
                     Those numbers should not be read as a converged posterior.\n\
                     \n\
                     Writes one PIP table and one posterior-mean table PER GATE.\n\
                     They are keyed by the feature the gate is over.\n\
                     \n\
                     A single-gate model gets {out}.feature_pip.parquet and {out}.feature_posterior_mean.parquet.\n\
                     A two-gate model — gem's identity β_g and velocity δ_g —\n\
                     gets {out}.beta_pip.parquet, {out}.beta_posterior_mean.parquet,\n\
                     {out}.delta_pip.parquet and {out}.delta_posterior_mean.parquet.\n\
                     Those are keyed by gene rather than by feature row.\n\
                     \n\
                     `--mcmc` and `--jitter` are accepted aliases."
    )]
    pub posterior: Option<usize>,

    #[arg(
        long = "stick-alpha",
        value_name = "ALPHA",
        conflicts_with = "no_stick_breaking",
        default_value_t = crate::posterior::dim_block::DEFAULT_STICK_ALPHA,
        requires = "posterior",
        help = "Truncated-IBP concentration α: expected dims a feature loads",
        long_help = "Concentration α for the TRUNCATED INDIAN BUFFET PROCESS.\n\
                     It sits on the per-dim inclusion rates.\n\
                     That process is the DEFAULT selection prior.\n\
                     \n\
                     Stick-breaking follows Teh,\n\
                     Görür & Ghahramani 2007. Each dim's inclusion rate is the running product ∏_{j≤h} v_j.\n\
                     The sticks are v_j ~ Beta(α, 1), held at their prior mean.\n\
                     So the rate at dim h is (α/(α+1))^(h+1). It decreases with the dim index.\n\
                     Surplus dims are squeezed off by construction.\n\
                     \n\
                     α is the expected number of dims a feature loads.\n\
                     It is independent of --embedding-dim. Measured on BM1,\n\
                     doubling H from 16 to 32 barely moved it.\n\
                     The active-dim count went 10 -> 12. The unordered alternative moved 16 -> 32 instead.\n\
                     So H is a TRUNCATION, not a tuning knob.\n\
                     \n\
                     WHY IT IS THE DEFAULT. Every dim carries tens of thousands of features.\n\
                     An independent Beta prior brings O(1) pseudo-counts against O(10^4) observations,\n\
                     so it is swamped. Every unused dim then settles on the SAME rate.\n\
                     None of them collapses. Measured,\n\
                     that was flat at 0.787-0.930 across 16 dims,\n\
                     while the likelihood supported about 3.4 of them.\n\
                     The ladder is a structural constraint. Data cannot outvote it.\n\
                     \n\
                     ALPHA IS CHOSEN, NOT FITTED.\n\
                     The rates are held at the stick-breaking prior mean.\n\
                     Nothing resamples them. So there is no chain to converge,\n\
                     and no per-dim R-hat to read.\n\
                     \n\
                     Letting them adapt was measured on BM1.\n\
                     It moved the dictionary's effective rank under 5%, from 9.05 to 8.64,\n\
                     while making the fit LESS sparse. It was not worth the machinery.\n\
                     \n\
                     H MUST BE LARGE RELATIVE TO ALPHA. The ladder is geometric,\n\
                     with ratio alpha/(alpha+1). At alpha = 1,\n\
                     sixteen dims already carry all the mass.\n\
                     At alpha = 5 they carry only about 95%.\n\
                     --embedding-dim is then still truncating the prior.\n\
                     Keep H well above alpha.\n\
                     \n\
                     A SIDE EFFECT WORTH KNOWING. The dims become ordered.\n\
                     That removes the dim-permutation gauge.\n\
                     It makes them comparable across runs, as PCA components are.\n\
                     \n\
                     A feature with NO counts reverts to each dim's population rate.\n\
                     It does not revert to a flat null.\n\
                     On the leading dim that is about a coin flip at α = 1. Lower α if that matters for your read.\n\
                     \n\
                     Pass --no-stick-breaking for the previous independent-Beta-per-dim prior."
    )]
    pub stick_alpha: f64,

    #[arg(
        long = "no-stick-breaking",
        requires = "posterior",
        help = "Use an independent Beta prior per dim instead of the truncated IBP.",
        long_help = "Revert the selection prior to an independent Beta(a,b) per dim.\n\
                     There is then no ordering, and no coupling between dims.\n\
                     \n\
                     This shipped before the IBP became the default.\n\
                     It is kept for A/B comparison.\n\
                     With many features per dim it is swamped by the data.\n\
                     So it neither imposes sparsity nor collapses unused dims.\n\
                     See --stick-alpha."
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

impl PosteriorPlan {
    /// The sampler config this plan implies.
    ///
    /// Lives here because `PosteriorPlan` is the resolved form of the flags and
    /// `PbGibbsConfig` is their only consumer — so the `n_samples / 2` warmup rule sits
    /// beside the docstring that states it, instead of being spelled out in `senna` and
    /// `faba` separately. Both CLIs had drifted into byte-identical copies of this, and
    /// a new posterior knob meant editing two crates to reach two binaries.
    #[must_use]
    pub fn pb_gibbs_config(&self) -> super::pb_gibbs::PbGibbsConfig {
        let mut cfg =
            super::pb_gibbs::PbGibbsConfig::new(self.n_samples, self.n_samples / 2, self.seed);
        cfg.stick_alpha = self.stick_alpha;
        cfg
    }
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
