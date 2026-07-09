//! CLI arguments for `senna deconvolve`.

use clap::{Args, ValueEnum};

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
        help = "Run manifest with a feature embedding: `senna bge --skip-etm` or `masked-topic`",
        long_help = "Run manifest exposing a per-gene embedding ρ.\n\
                     `senna bge --skip-etm` is exact: the raw Poisson ρ is persisted as\n\
                     dictionary.parquet (default ETM bge overwrites it with β — re-run with\n\
                     --skip-etm). `masked-topic` is supported as a transfer approximation\n\
                     (its ρ was trained under a softmax-ETM head)."
    )]
    pub from: Box<str>,

    #[arg(
        short = 'm',
        long = "markers",
        required = true,
        help = "Marker-gene TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

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

    //////////////////////////////////////////////////////////////////////
    // Sampler                                                            //
    //////////////////////////////////////////////////////////////////////
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
        default_value_t = 1.0,
        help = "Gamma prior shape a0 on cell-type abundances w (weak: 1.0)"
    )]
    pub frac_prior_shape: f32,

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
        help = "Negative-binomial dispersion r (size); smaller = more overdispersion (default ≈ Poisson)",
        long_help = "Per-(gene,sample) overdispersion via a Gamma(r,r) multiplicative factor ε\n\
                     on the Poisson rate: y ~ Poisson(ε·Σ_c w_c μ_{g,c}), Var(y)=λ+λ²/r. Small r\n\
                     absorbs reference/gene misfit into ε; r → ∞ recovers Poisson. Held fixed:\n\
                     freely sampling r is non-identifiable against the fractions (ε competes with\n\
                     w through the per-type exposure), so it is a knob, not a hyperparameter."
    )]
    pub nb_dispersion: f32,

    #[arg(
        long = "count-scale",
        default_value_t = 1.0,
        help = "Effective-count multiplier τ ∈ (0,1] tempering the likelihood (smaller → wider CIs)",
        long_help = "Power-posterior temperature: all count sufficient statistics are scaled by τ\n\
                     (likelihood^τ), so the posterior reflects τ·(observed counts) of independent\n\
                     evidence. τ=1 uses raw counts (tight, often overconfident at high depth);\n\
                     τ<1 widens credible intervals (variance ∝ 1/τ). Calibrate against held-out\n\
                     coverage."
    )]
    pub count_scale: f32,

    //////////////////////////////////////////////////////////////////////
    // Anchors (annotate-by-projection uncertainty)                       //
    //////////////////////////////////////////////////////////////////////
    #[arg(
        long = "anchor-cov",
        value_enum,
        default_value_t = AnchorCov::Isotropic,
        help = "Anchor prior covariance: isotropic scale, or shrunk full H×H"
    )]
    pub anchor_cov: AnchorCov,

    #[arg(
        long = "anchor-prior-scale",
        default_value_t = 1.0,
        help = "Multiplier on the anchor prior spread (larger → looser, wider fraction CIs)"
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
}

/// Plain (non-clap) anchor-prior settings.
pub struct AnchorConfig {
    pub cov: AnchorCov,
    pub scale: f32,
}

impl DeconvolveArgs {
    #[must_use]
    pub fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            warmup: self.warmup,
            draws: self.draws,
            thin: self.thin,
            seed: self.seed,
            a0: self.frac_prior_shape,
            b0: self.frac_prior_rate,
            project_ridge: self.project_ridge,
            init_iters: self.init_iters,
            nb_r: self.nb_dispersion,
            tau: self.count_scale,
        }
    }

    #[must_use]
    pub fn anchor_config(&self) -> AnchorConfig {
        AnchorConfig {
            cov: self.anchor_cov,
            scale: self.anchor_prior_scale,
        }
    }
}
