//! CLI arguments for `senna deconvolve`.

use clap::{Args, ValueEnum};

/// How the per-component gene reference is built.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ReferenceMode {
    /// One anchor per cell type, `μ = exp(ρ·t + a)` reconstructed from the embedding.
    LowRank,
    /// Many empirical profiles collapsed from annotated cells.
    Archetype,
}

/// Prior covariance shape for the per-cell-type anchor `t_c`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum AnchorCov {
    /// `Σ_c = σ_c²·I` — one scale per type from the marker-coordinate scatter.
    Isotropic,
    /// Shrunk empirical `H×H` covariance of the type's marker coordinates.
    Full,
}

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
        short = 'm',
        long = "markers",
        help = "Marker-gene TSV: `gene<TAB>celltype` per line (required for `--reference low-rank`)"
    )]
    pub markers: Option<Box<str>>,

    #[arg(
        long = "reference",
        value_enum,
        default_value_t = ReferenceMode::LowRank,
        help = "How the gene reference is built: reconstructed anchors, or empirical archetypes",
        long_help = "Which reference the sampler allocates bulk counts against.\n\
                     \n\
                     `low-rank` reconstructs one profile per cell type from the embedding,\n\
                     as exp(rho_g . t_c + a_g), with the anchors t_c resampled under a\n\
                     marker-derived prior. It is exact on a well-specified reference.\n\
                     Its weakness is that the reconstruction is rank-limited, so it cannot\n\
                     match a real bulk profile even at the true cell-type centroid.\n\
                     \n\
                     `archetype` collapses the annotated cells into many empirical profiles\n\
                     and maps them onto cell types through the annotation posterior.\n\
                     There is no reconstruction step, so that ceiling does not apply.\n\
                     It needs the single-cell counts and an annotation, not markers."
    )]
    pub reference: ReferenceMode,

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
        help = "RNG seed (per-sample + anchor chains)"
    )]
    pub seed: u64,

    #[arg(
        long = "frac-prior-shape",
        help = "Gamma prior shape a0 per component (default: 1.0 low-rank, auto-scaled archetype)",
        long_help = "Gamma prior shape a0 on each component's abundance, in pseudo-counts.\n\
                     \n\
                     It is what stops a component being driven to zero. The gene allocation\n\
                     is a winner-take-all dynamic between overlapping profiles, so a component\n\
                     that loses ground early can be extinguished and never recover.\n\
                     With few components that rarely happens, and a0 = 1 is a fine weak prior.\n\
                     With many, it happens readily and costs real accuracy.\n\
                     \n\
                     The archetype default is therefore 2*sqrt(mean sample counts / R),\n\
                     which holds each component a couple of sampling-noise units off zero\n\
                     and scales with depth. A fixed value cannot: it would be negligible on\n\
                     a deep bulk and would swamp a shallow one.\n\
                     \n\
                     Measured on a real pseudobulk benchmark at R = 180, fraction accuracy\n\
                     runs r = 0.68, 0.71, 0.78, 0.60, 0.44 at a0 = 1, 10, 100, 1000, 10000.\n\
                     The optimum is interior: too little lets components die, too much pulls\n\
                     every sample toward the same uniform composition."
    )]
    pub frac_prior_shape: Option<f32>,

    #[arg(
        long = "frac-prior-rate",
        default_value_t = 1.0,
        help = "Gamma prior rate b0 on cell-type abundances w (weak: 1.0)"
    )]
    pub frac_prior_rate: f32,

    #[arg(
        long = "project-ridge",
        default_value_t = 1.0,
        help = "Ridge λ for the Poisson projection of bulk into the embedding"
    )]
    pub project_ridge: f64,

    #[arg(
        long = "init-iters",
        default_value_t = 50,
        help = "Frank-Wolfe iterations for the simplex fraction initialization"
    )]
    pub init_iters: usize,

    #[arg(
        long = "nb-dispersion",
        default_value_t = 10000.0,
        help = "Negative-binomial dispersion r (size);\n\
                smaller = more overdispersion (default ≈ Poisson)",
        long_help = "Per-(gene,sample) overdispersion via a Gamma(r,r) multiplicative factor ε on the Poisson rate:\n\
                     y ~ Poisson(ε·Σ_c w_c μ_{g,c}), Var(y)=λ+λ²/r.\n\
                     Small r absorbs reference/gene misfit into ε; r → ∞ recovers Poisson.\n\
                     Held fixed: freely sampling r is non-identifiable against the fractions,\n\
                     since ε competes with w through the per-type exposure, so it is a knob,\n\
                     not a hyperparameter."
    )]
    pub nb_dispersion: f32,

    #[arg(
        long = "count-scale",
        default_value_t = 1.0,
        help = "Effective-count multiplier τ ∈ (0,1] tempering the likelihood (smaller → wider CIs)",
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
                     Both annotate layouts are accepted: the posterior table, and the\n\
                     bootstrap label-stability table. A hard membership table also works,\n\
                     and is read as a one-hot posterior.\n\
                     Rows are averaged within an archetype, so a cell whose label is\n\
                     uncertain contributes that uncertainty to the archetype it lands in."
    )]
    pub annotation: Option<Box<str>>,

    #[arg(
        long = "archetypes",
        num_args = 1..,
        default_values_t = [150usize, 300, 600],
        help = "Target archetype counts; one chain per value, pooled (Leiden resolution is searched)",
        long_help = "How finely the reference cells are collapsed into archetypes.\n\
                     Leiden resolution is binary-searched to reach each target.\n\
                     \n\
                     Several values run several chains and pool their draws.\n\
                     The partition is a nuisance parameter, not something the data pins down,\n\
                     and the granularity does move the answer, so averaging over a few\n\
                     granularities is better than conditioning on one.\n\
                     Pooling also gives a between-chain R-hat: if the chains disagree,\n\
                     the reported spread reflects that rather than hiding it.\n\
                     \n\
                     Pass a single value to condition on one partition.\n\
                     Targets that leave fewer than `--archetype-min-cells` cells per\n\
                     archetype are merged back down, so asking for too many is safe."
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
                     Each profile is pulled toward the pooled profile by this many\n\
                     pseudo-counts, which keeps every rate strictly positive.\n\
                     Larger values blur the archetypes together."
    )]
    pub archetype_shrink: f32,

    #[arg(
        long = "archetype-cells",
        help = "Restrict archetypes to the cell names in this file (one per line)",
        long_help = "Build the archetype profiles from a subset of cells only.\n\
                     The point is leakage: when a bulk sample is itself made of reference\n\
                     cells, an empirical reference can read back the very counts it is\n\
                     meant to explain, and the accuracy reported is not real.\n\
                     Passing the complement of the bulk cells here gives a clean estimate."
    )]
    pub archetype_cells: Option<Box<str>>,

    //////////////////////////////////////////////////
    // Anchors (annotate-by-projection uncertainty) //
    //////////////////////////////////////////////////
    #[arg(
        long = "anchor-cov",
        value_enum,
        default_value_t = AnchorCov::Isotropic,
        help = "Anchor prior covariance: isotropic scale, or shrunk full H×H"
    )]
    pub anchor_cov: AnchorCov,

    #[arg(
        long = "anchor-prior-scale",
        default_value_t = 0.1,
        help = "Multiplier on the anchor prior spread; larger lets anchors drift from their markers",
        long_help = "Multiplier on the marker-derived anchor spread.\n\
                     Larger values let the sampler move each anchor further from its markers.\n\
                     \n\
                     That freedom is mostly harmful.\n\
                     The reference is low-rank, so it cannot match the bulk exactly.\n\
                     A free anchor absorbs the misfit by drifting off the true position.\n\
                     Composition is what pays for the improved fit.\n\
                     \n\
                     On a real pseudobulk benchmark, accuracy falls away as anchors loosen.\n\
                     Pearson r runs 0.75, 0.55, 0.36, 0.11 at 0.1, 0.2, 0.3, 0.5.\n\
                     It reaches ~0 at 1.0, which was the old default.\n\
                     On a well-specified synthetic reference the same change costs little.\n\
                     There r moves 0.989 -> 0.984, and the systematic bias halves."
    )]
    pub anchor_prior_scale: f32,
}

/// Plain (non-clap) sampler settings threaded into the Gibbs core.
pub struct SamplerConfig {
    pub warmup: usize,
    pub draws: usize,
    pub thin: usize,
    pub seed: u64,
    pub a0: f32,
    pub b0: f32,
    pub project_ridge: f64,
    pub init_iters: usize,
    /// NB dispersion r (`--nb-dispersion`), held fixed.
    pub nb_r: f32,
    /// Likelihood tempering τ (`--count-scale`).
    pub tau: f32,
    /// Collect the expression tensor every this many retained draws.
    pub expression_every: usize,
}

/// Plain (non-clap) anchor-prior settings.
pub struct AnchorConfig {
    pub cov: AnchorCov,
    pub scale: f32,
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
            project_ridge: self.project_ridge,
            init_iters: self.init_iters,
            nb_r: self.nb_dispersion,
            tau: self.count_scale,
            expression_every: self.expression_every.max(1),
        }
    }

    #[must_use]
    pub fn anchor_config(&self) -> AnchorConfig {
        AnchorConfig {
            cov: self.anchor_cov,
            scale: self.anchor_prior_scale,
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
