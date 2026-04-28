use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::common_io::mkdir_parent;

use fagioli::mcmc::{McmcFitConfig, McmcPriorType, McmcSampler};
use fagioli::summary_stats::common::{
    estimate_adaptive_prior_vars, parse_prior_vars, prepare_sumstat_input, run_blocks_and_write,
    CommonSumstatArgs,
};

#[derive(Args, Debug, Clone)]
pub struct FitSumstatMcmcArgs {
    #[command(flatten)]
    pub common: CommonSumstatArgs,

    // ── MCMC-specific parameters ─────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        help_heading = "MCMC Sampler",
        default_value = "susie",
        help = "MCMC prior: 'susie' or 'spike-slab'",
        long_help = "Sparse prior for MCMC fine-mapping:\n\n\
            - susie: SoftmaxNormal prior — ESS samples logits with softmax\n\
              transformation, enforcing single-effect per component.\n\
            - spike-slab: BernoulliNormal prior — ESS samples logits with\n\
              sigmoid transformation, allowing independent per-SNP selection.\n\n\
            Default: susie."
    )]
    pub prior: McmcPriorType,

    #[arg(
        long,
        help_heading = "MCMC Sampler",
        default_value = "2000",
        help = "Number of posterior samples to collect",
        long_help = "Number of posterior samples to collect after warmup.\n\
            Total sweeps = warmup + n_samples × thin. More samples reduce\n\
            Monte Carlo noise in PIP estimates but increase runtime. Default: 2000."
    )]
    pub n_samples: usize,

    #[arg(
        long,
        help_heading = "MCMC Sampler",
        default_value = "1000",
        help = "Warmup (burn-in) iterations before collecting samples",
        long_help = "Number of burn-in sweeps discarded before sample collection.\n\
            Allows the chain to reach the stationary distribution. Default: 1000."
    )]
    pub warmup: usize,

    #[arg(
        long,
        help_heading = "MCMC Sampler",
        default_value = "2",
        help = "Thinning interval (collect every N-th sweep)",
        long_help = "Collect a sample every N-th sweep after warmup to reduce\n\
            autocorrelation between posterior samples. Default: 2."
    )]
    pub thin: usize,

    #[arg(
        long,
        help_heading = "MCMC Sampler",
        default_value = "1.0",
        help = "Prior variance on inclusion logits",
        long_help = "Prior variance for the Gaussian prior on raw inclusion logits.\n\
            Controls ESS step size: larger values allow bigger jumps in\n\
            inclusion probability space. Default: 1.0."
    )]
    pub logit_var: f32,

    #[arg(
        long,
        help_heading = "MCMC Sampler",
        default_value_t = false,
        help = "Estimate the effect size prior variance via Gibbs",
        long_help = "When enabled, the effect size prior variance is sampled via a\n\
            conjugate inverse-gamma Gibbs update instead of being fixed.\n\
            This adapts the prior to the data, improving calibration when\n\
            the initial prior_var (from LDSC h²) is misspecified.\n\
            Prior: InvGamma(0.01, 0.01). Default: disabled."
    )]
    pub estimate_prior_var: bool,
}

pub fn fit_sumstat_mcmc(args: &FitSumstatMcmcArgs) -> Result<()> {
    mkdir_parent(&args.common.output)?;
    info!("Starting fit-sumstat-mcmc");

    let input = prepare_sumstat_input(&args.common)?;

    // Determine prior variance
    let mut prior_vars = parse_prior_vars(&args.common.prior_var)?;
    if prior_vars.is_empty() {
        prior_vars =
            estimate_adaptive_prior_vars(&input, args.common.num_components, args.common.lambda);
    }
    // For MCMC, use the median of the adaptive grid (single prior_var)
    let prior_var = if prior_vars.len() == 1 {
        prior_vars[0]
    } else {
        prior_vars[prior_vars.len() / 2]
    };
    info!("MCMC prior_var: {:.4}", prior_var);

    let prior_type = args.prior;
    info!(
        "MCMC prior: {:?}, L={}, logit_var={}, n_samples={}, warmup={}, thin={}",
        prior_type,
        args.common.num_components,
        args.logit_var,
        args.n_samples,
        args.warmup,
        args.thin,
    );

    let fitter = McmcSampler::new(McmcFitConfig {
        prior_type,
        num_components: args.common.num_components,
        prior_var,
        logit_var: args.logit_var,
        n_samples: args.n_samples,
        warmup: args.warmup,
        thin: args.thin,
        estimate_effect_var: args.estimate_prior_var,
    });

    let extra_params = serde_json::json!({
        "command": "fit-sumstat-mcmc",
        "method": "mcmc",
        "prior": format!("{:?}", prior_type),
        "prior_var": prior_var,
        "logit_var": args.logit_var,
        "n_samples": args.n_samples,
        "warmup": args.warmup,
        "thin": args.thin,
        "estimate_prior_var": args.estimate_prior_var,
    });

    run_blocks_and_write(&input, &fitter, &args.common, extra_params)?;

    info!("fit-sumstat-mcmc completed successfully");
    Ok(())
}
