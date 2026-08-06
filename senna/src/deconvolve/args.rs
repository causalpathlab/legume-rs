//! CLI arguments for `senna deconvolve`.

use clap::Args;

#[derive(Args, Debug)]
pub struct DeconvolveArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest with a feature embedding:\n\
                `senna bge --skip-etm` or `masked-topic`",
        long_help = "Run manifest from `senna bge`, with or without --skip-etm.\n\
                     Every bge run records the per-gene loading ρ.\n\
                     Older runs kept it only under --skip-etm; those still work.\n\
                     \n\
                     Topic-family runs are not supported.\n\
                     Their ρ pairs with the topic embeddings under a softmax head.\n\
                     It does not pair with cell positions under a Poisson rate."
    )]
    pub from: Box<str>,

    #[arg(
        long = "bulk",
        required = true,
        num_args = 1..,
        help = "One or more bulk count matrices (parquet/tsv; genes × samples)"
    )]
    pub bulk: Vec<Box<str>>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (default: `--from` with `.senna.json`/`.json` stripped, `.deconv`)"
    )]
    pub out: Option<Box<str>>,

    /////////////
    // Sampler //
    /////////////
    #[arg(
        long = "warmup",
        default_value_t = 500,
        help = "Gibbs warmup (burn-in) sweeps, discarded before collection"
    )]
    pub warmup: usize,

    #[arg(
        long = "draws",
        default_value_t = 500,
        help = "Posterior draws collected after warmup (thinned)"
    )]
    pub draws: usize,

    #[arg(
        long = "thin",
        default_value_t = 1,
        help = "Keep one draw per `thin` sweeps"
    )]
    pub thin: usize,

    #[arg(
        long = "seed",
        default_value_t = 42,
        help = "RNG seed (one stream per sample and per chain)"
    )]
    pub seed: u64,

    #[arg(
        long = "frac-prior-shape",
        help = "Gamma prior shape a0 per component; auto-scaled unless set",
        long_help = "Gamma prior shape a0 on each component's abundance, in pseudo-counts.\n\
                     \n\
                     It is what stops a component being driven to zero.\n\
                     The gene allocation is winner-take-all between overlapping profiles.\n\
                     A component that loses ground early can be extinguished for good.\n\
                     With few components that rarely happens, and a0 = 1 is a fine prior.\n\
                     With many it happens readily, and it costs real accuracy.\n\
                     \n\
                     The archetype default is 2*sqrt(mean sample counts / R).\n\
                     That holds each component a couple of sampling-noise units off zero.\n\
                     It also scales with depth, which a fixed value cannot:\n\
                     a constant would be negligible on a deep bulk and swamp a shallow one.\n\
                     \n\
                     On a real pseudobulk benchmark, accuracy falls off either side.\n\
                     Pearson r runs 0.68, 0.71, 0.78, 0.60, 0.44 at a0 = 1, 10, 100, 1e3, 1e4.\n\
                     Too little lets components die.\n\
                     Too much pulls every sample toward the same uniform composition."
    )]
    pub frac_prior_shape: Option<f32>,

    #[arg(
        long = "frac-prior-rate",
        default_value_t = 1.0,
        help = "Gamma prior rate b0 on cell-type abundances w (weak: 1.0)"
    )]
    pub frac_prior_rate: f32,

    #[arg(
        long = "nb-dispersion",
        default_value_t = 10000.0,
        help = "Negative-binomial dispersion r (size);\n\
                smaller = more overdispersion (default ≈ Poisson)",
        long_help = "Per-(gene,sample) overdispersion on the Poisson rate.\n\
                     A Gamma(r,r) factor ε multiplies the rate:\n\
                     y ~ Poisson(ε·Σ_c w_c μ_{g,c}), so Var(y) = λ + λ²/r.\n\
                     Small r absorbs reference and gene misfit into ε.\n\
                     Large r recovers Poisson.\n\
                     \n\
                     Held fixed: freely sampling r is non-identifiable against w.\n\
                     ε competes with w through the per-type exposure.\n\
                     It is a knob, not a hyperparameter."
    )]
    pub nb_dispersion: f32,

    #[arg(
        long = "count-scale",
        default_value_t = 1.0,
        help = "Likelihood tempering τ ∈ (0,1]; smaller gives wider intervals",
        long_help = "Power-posterior temperature:\n\
                     all count sufficient statistics are scaled by τ (likelihood^τ),\n\
                     so the posterior reflects τ·(observed counts) of independent evidence.\n\
                     τ=1 uses raw counts (tight, often overconfident at high depth);\n\
                     τ<1 widens credible intervals (variance ∝ 1/τ).\n\
                     Calibrate against held-out coverage."
    )]
    pub count_scale: f32,

    #[arg(
        long = "expression-every",
        default_value_t = 10,
        help = "Accumulate the expression tensor every N collected draws (1 = every draw)",
        long_help = "Thinning for the per-cell-type expression tensor only.\n\
                     Fractions are collected on every draw; they are cheap.\n\
                     The expression tensor costs one pass of genes x components per draw,\n\
                     which dominates the run once the reference has many components.\n\
                     Raise this to trade expression-tensor precision for speed."
    )]
    pub expression_every: usize,

    ////////////////
    // Monitoring //
    ////////////////
    #[arg(
        long = "trace-every",
        default_value_t = 10,
        help = "Write a fraction trace row every N sweeps, warmup included (0 disables)"
    )]
    pub trace_every: usize,

    #[arg(
        long = "checkpoint-every",
        default_value_t = 100,
        help = "Re-write the fraction table from the running mean every N sweeps (0 disables)"
    )]
    pub checkpoint_every: usize,

    /////////////////////////////////
    // Archetype reference sources //
    /////////////////////////////////
    #[arg(
        long = "sc-data",
        num_args = 1..,
        help = "Single-cell count matrices behind the archetypes (zarr/h5)"
    )]
    pub sc_data: Vec<Box<str>>,

    #[arg(
        long = "annotation",
        help = "Cell x celltype annotation parquet; defaults to the one named in `--from`",
        long_help = "Soft per-cell annotation, cells x cell types, with a leading name column.\n\
                     Both annotate layouts are accepted.\n\
                     The posterior table and the label-stability table both work.\n\
                     A hard membership table also works, read as a one-hot posterior.\n\
                     \n\
                     Rows are averaged within an archetype.\n\
                     A cell with an uncertain label passes that uncertainty to its archetype."
    )]
    pub annotation: Option<Box<str>>,

    #[arg(
        long = "archetypes",
        num_args = 1..,
        default_values_t = [150usize, 300, 600],
        help = "Target archetype counts; one pooled chain per value",
        long_help = "How finely the reference cells are collapsed into archetypes.\n\
                     Leiden resolution is binary-searched to reach each target.\n\
                     \n\
                     Several values run several chains and pool their draws.\n\
                     The partition is a nuisance parameter, not something the data pins down.\n\
                     The granularity does move the answer.\n\
                     Averaging over a few granularities beats conditioning on one.\n\
                     Pooling also gives a between-chain R-hat.\n\
                     When the chains disagree, the reported spread shows it.\n\
                     \n\
                     Pass a single value to condition on one partition.\n\
                     Targets that leave too few cells per archetype are merged back down,\n\
                     so asking for too many is safe."
    )]
    pub archetypes: Vec<usize>,

    #[arg(
        long = "archetype-min-cells",
        default_value_t = 20,
        help = "Merge archetypes below this many cells into the nearest surviving one"
    )]
    pub archetype_min_cells: usize,

    #[arg(
        long = "archetype-shrink",
        default_value_t = 5.0,
        help = "Pseudo-count shrinking each archetype profile toward the pooled profile",
        long_help = "Empirical-Bayes shrinkage of the archetype profiles.\n\
                     A gene seen in no cell of an archetype would otherwise have rate zero,\n\
                     and any bulk count on that gene then has nowhere to go.\n\
                     Each profile is pulled toward the pooled profile by this many counts,\n\
                     which keeps every rate strictly positive.\n\
                     Larger values blur the archetypes together."
    )]
    pub archetype_shrink: f32,

    #[arg(
        long = "archetype-cells",
        help = "Restrict archetypes to the cell names in this file (one per line)",
        long_help = "Build the archetype profiles from a subset of cells only.\n\
                     The point is leakage.\n\
                     When a bulk sample is itself made of reference cells,\n\
                     the reference can read back the counts it is meant to explain,\n\
                     and the accuracy reported is not real.\n\
                     Passing the complement of the bulk cells here gives a clean estimate."
    )]
    pub archetype_cells: Option<Box<str>>,
}

/// Plain (non-clap) sampler settings threaded into the Gibbs core.
pub struct SamplerConfig {
    pub warmup: usize,
    pub draws: usize,
    pub thin: usize,
    pub seed: u64,
    pub a0: f32,
    pub b0: f32,
    /// NB dispersion r (`--nb-dispersion`), held fixed.
    pub nb_r: f32,
    /// Likelihood tempering τ (`--count-scale`).
    pub tau: f32,
    /// Collect the expression tensor every this many retained draws.
    pub expression_every: usize,
}

/// Plain (non-clap) archetype-construction settings.
pub struct ArchetypeConfig<'a> {
    pub sc_data: &'a [Box<str>],
    pub annotation: Option<&'a str>,
    pub min_cells: usize,
    pub shrink: f32,
    pub cells: Option<&'a str>,
    pub seed: u64,
}

impl DeconvolveArgs {
    #[must_use]
    pub fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            warmup: self.warmup,
            draws: self.draws,
            thin: self.thin,
            seed: self.seed,
            // Resolved against the reference size in `deconvolve::run`, which is
            // the first place the component count and the bulk depth are known.
            a0: self.frac_prior_shape.unwrap_or(1.0),
            b0: self.frac_prior_rate,
            nb_r: self.nb_dispersion,
            tau: self.count_scale,
            expression_every: self.expression_every.max(1),
        }
    }

    #[must_use]
    pub fn archetype_config(&self) -> ArchetypeConfig<'_> {
        ArchetypeConfig {
            sc_data: &self.sc_data,
            annotation: self.annotation.as_deref(),
            min_cells: self.archetype_min_cells,
            shrink: self.archetype_shrink,
            cells: self.archetype_cells.as_deref(),
            seed: self.seed,
        }
    }

    #[must_use]
    pub fn monitor_config(&self) -> crate::deconvolve::monitor::MonitorConfig {
        crate::deconvolve::monitor::MonitorConfig {
            trace_every: self.trace_every,
            checkpoint_every: self.checkpoint_every,
        }
    }
}
