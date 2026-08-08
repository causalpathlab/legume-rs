use anyhow::Result;
use clap::Args;
use log::{info, warn};
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;

use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::score::{assemble_u, PanelStandardization};
use fagioli::embedding::train::train;
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::io::results::write_parameters;
use fagioli::sgvb::ComputeDevice;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::{prepare_sumstat_input, CommonSumstatArgs};

#[derive(Args, Debug, Clone)]
#[command(long_about = "Embed GWAS summary statistics into a shared variant-trait program space.

The model factors the direct, LD-free effect matrix as B = U V'. Variants and
traits are placed in one H-dimensional program space, and the fit is trained
contrastively against a null calibrated from the reference panel.

Writes:
  {prefix}.variant_embedding.parquet   variant loadings U, one row per SNP
  {prefix}.trait_embedding.parquet     trait loadings V, one row per trait
  {prefix}.panel.tsv.gz                standardization needed to score a cohort
  {prefix}.whiteness.tsv.gz            whether lambda whitened the null
  {prefix}.parameters.json             every setting, plus fit diagnostics

No trait-geometry verdict is written. The estimator for it is not reliable on
realistic LD, so the fitted geometry is reported without an independent check.")]
pub struct EmbedSumstatArgs {
    #[command(flatten)]
    pub common: CommonSumstatArgs,

    // ── Model ────────────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help_heading = "Embedding",
        help = "Number of latent programs H",
        long_help = "Number of latent programs H shared by variants and traits.\n\
                     \n\
                     The model factors the direct effect matrix as B = U V'.\n\
                     A variant's loading u_g and a trait's loading v_t meet in\n\
                     this space, so beta_gt is their inner product.\n\
                     \n\
                     H bounds how much pleiotropy the fit can express.\n\
                     Set it below the trait count for a low-rank geometry."
    )]
    pub embedding_dim: usize,

    #[arg(
        long,
        default_value = "5",
        help_heading = "Embedding",
        help = "Negative samples drawn per positive",
        long_help = "Negatives drawn per positive for the contrastive objective.\n\
                     \n\
                     Negatives come from the calibrated null, so their\n\
                     distribution is known rather than estimated.\n\
                     More negatives lower the gradient variance.\n\
                     They cost one extra score evaluation each."
    )]
    pub num_negatives: usize,

    #[arg(
        long,
        default_value = "0.01",
        help_heading = "Embedding",
        help = "Prior inclusion probability for the spike-slab on U",
        long_help = "Prior probability that a variant loads on a program.\n\
                     \n\
                     Each (variant, program) pair carries its own Bernoulli\n\
                     gate, so there is no categorical over variants.\n\
                     Lower values shrink more variants to zero.\n\
                     The prior enters the objective through its KL term."
    )]
    pub prior_inclusion: f64,

    #[arg(
        long,
        value_enum,
        default_value = "spike-slab",
        help_heading = "Embedding",
        help = "Variational family for the variant loadings",
        long_help = "How selection over variants is parameterised.\n\
                     \n\
                     spike-slab gives each (variant, program) pair its own\n\
                     Bernoulli gate. There is no categorical anywhere, so the\n\
                     degeneracy that appears when a softmax spans many\n\
                     thousands of categories cannot arise.\n\
                     It reports marginal inclusion probabilities only.\n\
                     \n\
                     susie places L single-effect components in each block.\n\
                     Each is a categorical over that block's variants, so the\n\
                     softmax is bounded by --max-block-snps and never by the\n\
                     genome. Mass competes within a block, which is what makes\n\
                     credible sets meaningful.\n\
                     Use --num-components to set L."
    )]
    pub u_prior: UPrior,

    #[arg(
        long,
        default_value = "1.0",
        help_heading = "Embedding",
        help = "Dirichlet concentration for the SuSiE selection prior",
        long_help = "Concentration of the Dirichlet prior on the selection\n\
                     probabilities within a block.\n\
                     \n\
                     Values below one favour concentrated selection.\n\
                     Values above one spread mass across variants.\n\
                     Ignored when --u-prior is spike-slab."
    )]
    pub prior_alpha: f64,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Embedding",
        help = "Model a polygenic component alongside the sparse one",
        long_help = "Add a marginalised polygenic component to the model.\n\
                     \n\
                     Variants share a prior rather than a value, so the whole\n\
                     arm costs H parameters and not one per variant.\n\
                     It enters the covariance, where the sparse part enters\n\
                     the mean, which is what separates the two.\n\
                     \n\
                     The two arms compete for the same signal.\n\
                     Leave it off unless a polygenic background is expected."
    )]
    pub dense_arm: bool,

    #[arg(
        long,
        default_value = "0.0",
        help_heading = "Embedding",
        help = "Weight on the orthonormality penalty for the trait embedding",
        long_help = "Penalty weight holding the trait embedding orthonormal.\n\
                     \n\
                     U V' is unchanged by U -> UA and V -> VA', so the program\n\
                     directions are otherwise arbitrary.\n\
                     Fixing the gauge makes a single program interpretable.\n\
                     \n\
                     The polygenic arm requires it and sets it automatically.\n\
                     It constrains the fit, so it can cost some accuracy."
    )]
    pub gauge_weight: f64,

    // ── Optimization ─────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0.01",
        help_heading = "Optimization",
        help = "AdamW learning rate",
        long_help = "Step size for the AdamW optimizer.\n\
                     \n\
                     The objective is contrastive, so no single quantity is\n\
                     being minimised to convergence.\n\
                     Watch the reported loss rather than expecting a plateau."
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value = "500",
        help_heading = "Optimization",
        help = "Number of gradient steps",
        long_help = "Gradient steps taken over the whole set of blocks.\n\
                     \n\
                     Every block contributes at each step, so this is not an\n\
                     epoch count over minibatches.\n\
                     Cost grows linearly in it."
    )]
    pub num_iterations: usize,

    #[arg(
        long,
        default_value = "10.0",
        help_heading = "Optimization",
        help = "Global gradient-norm clip (0 disables)",
        long_help = "Rescale gradients whose global norm exceeds this value.\n\
                     \n\
                     The contrastive loss spikes when a batch of negatives\n\
                     lands close to the fitted mean.\n\
                     Clipping keeps one such step from moving the fit."
    )]
    pub grad_clip: f64,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help_heading = "Optimization",
        help = "Hardware device: cpu, cuda, or metal",
        long_help = "Device the fit runs on.\n\
                     \n\
                     Blocks are held resident, so memory grows with the panel.\n\
                     cuda and metal require the matching build feature."
    )]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help_heading = "Optimization",
        help = "GPU device index",
        long_help = "Which GPU to use when --device is cuda or metal.\n\
                     Ignored on cpu."
    )]
    pub device_no: usize,
}

pub fn embed_sumstat(args: &EmbedSumstatArgs) -> Result<()> {
    mkdir_parent(&args.common.output)?;
    info!("Starting embed-sumstat");

    // ── Step 1: Panel, z-scores, LD blocks ───────────────────────────────
    let input = prepare_sumstat_input(&args.common)?;
    let num_traits = input.zscores.ncols();

    // ── Step 2: Calibrate the null and choose lambda ─────────────────────
    let report = calibrate_input(&input)
        .ok_or_else(|| anyhow::anyhow!("Could not calibrate the null; no block was large enough"))?;
    let lambda = match args.common.lambda {
        Some(l) => {
            info!("Using the supplied lambda {:.4}", l);
            l
        }
        None => report.noise.lambda_white(),
    };
    if report.misspecified {
        warn!(
            "The null did not whiten, so the reference panel's LD does not describe \
             this study's noise. Treat the embedding as provisional."
        );
    }

    // ── Step 3: Whiten ───────────────────────────────────────────────────
    // Sample overlap is left as the identity: it cancels in the noise model and
    // only ever entered through the trait axis, which is correct for disjoint
    // cohorts. Overlapping cohorts need Omega, and the estimator for it is not
    // trustworthy on realistic LD yet.
    let blocks = whiten_blocks(&input, None, lambda)?;
    anyhow::ensure!(!blocks.is_empty(), "No LD block was large enough to fit");

    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);

    // ── Step 4: Fit ──────────────────────────────────────────────────────
    let device = args.device.to_device(args.device_no)?;
    let gauge_weight = if args.dense_arm && args.gauge_weight <= 0.0 {
        info!("Polygenic arm requested, so enabling the orthonormality gauge");
        10.0
    } else {
        args.gauge_weight
    };

    let config = EmbedConfig {
        embedding_dim: args.embedding_dim,
        num_negatives: args.num_negatives,
        prior_inclusion: args.prior_inclusion,
        u_prior: args.u_prior,
        num_components: args.common.num_components,
        prior_alpha: args.prior_alpha,
        learning_rate: args.learning_rate,
        num_iterations: args.num_iterations,
        grad_clip: (args.grad_clip > 0.0).then_some(args.grad_clip),
        dense_arm: args.dense_arm,
        gauge_weight,
        seed: args.common.seed,
    };
    let fit = train(&blocks, &noise, &config, &device)?;

    // ── Step 5: Write ────────────────────────────────────────────────────
    let starts: Vec<usize> = blocks.iter().map(|b| b.snp_start).collect();
    let u = assemble_u(&fit.u_mean, &starts, input.geno.num_snps());
    let program_names: Vec<Box<str>> = (0..args.embedding_dim)
        .map(|h| Box::from(format!("program_{h}")))
        .collect();

    u.to_parquet_with_names(
        &format!("{}.variant_embedding.parquet", args.common.output),
        (Some(&input.geno.snp_ids), Some("snp_id")),
        Some(&program_names),
    )?;

    let trait_names: Vec<Box<str>> = (0..num_traits)
        .map(|t| Box::from(format!("trait_{t}")))
        .collect();
    fit.v_check.to_parquet_with_names(
        &format!("{}.trait_embedding.parquet", args.common.output),
        (Some(&trait_names), Some("trait")),
        Some(&program_names),
    )?;

    // Panel standardization, so a new cohort can be scored without the panel.
    let panel = PanelStandardization::from_panel(&input.geno);
    write_panel(&format!("{}.panel.tsv.gz", args.common.output), &panel)?;

    write_whiteness(
        &format!("{}.whiteness.tsv.gz", args.common.output),
        &report.curve,
    )?;

    let params = serde_json::json!({
        "command": "embed-sumstat",
        "sumstat_file": args.common.sumstat_file,
        "bed_prefix": args.common.bed_prefix,
        "chromosome": args.common.chromosome,
        "num_individuals": input.geno.num_individuals(),
        "num_snps": input.geno.num_snps(),
        "num_traits": num_traits,
        "num_blocks_fitted": blocks.len(),
        "embedding_dim": args.embedding_dim,
        "num_negatives": args.num_negatives,
        "prior_inclusion": args.prior_inclusion,
        "u_prior": format!("{:?}", args.u_prior),
        "num_components": args.common.num_components,
        "prior_alpha": args.prior_alpha,
        "dense_arm": args.dense_arm,
        "gauge_weight": gauge_weight,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "seed": args.common.seed,
        "lambda": lambda,
        "lambda_se": report.lambda_se,
        "noise_c": report.noise.c,
        "noise_tau": report.noise.tau,
        "whiteness_deviation": report.deviation,
        "misspecified": report.misspecified,
        "nce_offset": fit.offset,
        "gauge_residual": fit.gauge_residual,
        "dense_variance": fit.dense_variance,
        "final_loss": fit.loss_trace.last(),
    });
    write_parameters(
        &format!("{}.parameters.json", args.common.output),
        &params,
    )?;

    if fit.offset.abs() > 0.5 {
        warn!(
            "The contrastive offset settled away from zero, which usually means U has \
             more freedom than the data supports. Consider a smaller --embedding-dim \
             or a smaller --prior-inclusion."
        );
    }

    info!("embed-sumstat completed successfully");
    Ok(())
}

/// Write the panel's standardization constants, keyed by variant.
fn write_panel(path: &str, panel: &PanelStandardization) -> Result<()> {
    use matrix_util::common_io::open_buf_writer;
    use std::io::Write;

    let mut w = open_buf_writer(path)?;
    writeln!(w, "snp_id\tchr\tpos\ta1\ta2\tmean\tsd")?;
    for j in 0..panel.num_snps() {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}",
            panel.snp_ids[j],
            panel.chromosomes[j],
            panel.positions[j],
            panel.allele1[j],
            panel.allele2[j],
            panel.mean[j],
            panel.sd[j],
        )?;
    }
    w.flush()?;
    info!("Wrote panel standardization to {}", path);
    Ok(())
}

/// Write the whiteness curve, which shows whether lambda whitened the null.
fn write_whiteness(
    path: &str,
    curve: &[fagioli::summary_stats::calibration::WhitenessBin],
) -> Result<()> {
    use matrix_util::common_io::open_buf_writer;
    use std::io::Write;

    let mut w = open_buf_writer(path)?;
    writeln!(w, "d_sq\tmean_z_sq\tnum_coordinates")?;
    for b in curve {
        writeln!(w, "{:.6}\t{:.6}\t{}", b.d_sq, b.mean_z_sq, b.count)?;
    }
    w.flush()?;
    info!("Wrote whiteness curve to {}", path);
    Ok(())
}
