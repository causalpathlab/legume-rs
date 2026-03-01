use crate::collapse_cocoa_data::*;
use crate::common::*;
use crate::input::*;
use crate::randomly_partition_data::*;
use crate::stat::*;

use clap::Parser;
use matrix_param::io::*;
use matrix_util::common_io::write_lines;
use matrix_util::traits::{IoOps, MatOps};
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    #[arg(
        required = true,
        help = "Data files of either `.zarr` `.h` format",
        long_help = "Data files of either `.zarr` or `.h5` format. \n\
		     All the formats in the given list should be identical. \n\
		     You can convert `.mtx` to `.zarr` or `.h5` using the `data-beans`"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'i',
        long,
        value_delimiter = ',',
        required = true,
        help = "Individual membership file names (comma-separated).",
        long_help = "Individual membership files (comma-separated file names). \n\
		     Each line in each file can specify: \n\
		     * just  individual ID or\n\
		     * (1) Cell and (2) individual ID pair."
    )]
    indv_files: Vec<Box<str>>,

    #[arg(
        short = 'e',
        long,
        required = true,
        help = "Exposure assignment file.",
        long_help = "Each line corresponds to: \n\
		     (1) individual name and (2) exposure name."
    )]
    exposure_assignment_file: Box<str>,

    #[arg(
        short = 't',
        long = "topic-assignment-files",
        value_delimiter = ',',
        help = "Latent topic assignment file names (comma-separated).",
        long_help = "Latent topic assignment files (comma-separated file names). \n\
		     Each line in each file can specify:\n\
		     * just topic name or \n\
		     * (1) cell and (2) topic name pair."
    )]
    topic_assignment_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'r',
        long,
        value_delimiter = ',',
        help = "Latent topic proportion file names (comma-separated).",
        long_help = "Latent topic proportion files (comma-separated file names). \n\
		     Each file contains a full `cell x topic` matrix."
    )]
    topic_proportion_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value = "logit",
        help = "Is topic proportion matrix of probability?",
        long_help = "Specify if the topic proportion matrix is of probability type. \n\
		     Default is `logit`-valued."
    )]
    topic_proportion_value: TopicValue,

    #[arg(
        short = 'n',
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each condition.",
        long_help = "Specify the number of k-nearest neighbours within each condition."
    )]
    knn: usize,

    #[arg(
        short = 'p',
        long,
        default_value_t = 10,
        help = "Projection dimension to account for confounding factors.",
        long_help = "Projection dimension to account for confounding factors."
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing.",
        long_help = "Block size for parallel processing."
    )]
    block_size: usize,

    #[arg(
        long,
        help = "Number of iterations for optimization.",
        long_help = "Number of iterations for optimization."
    )]
    num_opt_iter: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Hyperparameter a0 in Gamma(a0, b0).",
        long_help = "Hyperparameter a0 in Gamma(a0, b0)."
    )]
    a0: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Hyperparameter b0 in Gamma(a0, b0).",
        long_help = "Hyperparameter b0 in Gamma(a0, b0)."
    )]
    b0: f32,

    #[arg(
        short,
        long,
        required = true,
        help = "Output directory.",
        long_help = "Output directory."
    )]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of exposure-label permutations for empirical p-values."
    )]
    n_permutations: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "Random seed for permutation testing."
    )]
    permutation_seed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all the columns data.",
        long_help = "Preload all the columns data."
    )]
    preload_data: bool,

    #[arg(
        long,
        help = "Known confounder matrix file (tsv.gz, n_indv x n_covar).",
        long_help = "Provide a known individual-level confounder matrix V instead of\n\
                     discovering confounders by random projection. The file should be\n\
                     a tab-delimited matrix (n_indv x n_covar) in .tsv.gz format.\n\
                     Rows correspond to individuals 0, 1, 2, ... in order."
    )]
    confounder_file: Option<Box<str>>,
}

/////////////////////////////////////
// Run CoCoA differential analysis //
/////////////////////////////////////

pub fn run_cocoa_diff(args: DiffArgs) -> anyhow::Result<()> {
    let mut data = read_input_data(InputDataArgs {
        data_files: args.data_files,
        indv_files: Some(args.indv_files),
        topic_assignment_files: args.topic_assignment_files,
        topic_proportion_files: args.topic_proportion_files,
        exposure_assignment_file: Some(args.exposure_assignment_file),
        preload_data: args.preload_data,
        topic_value: args.topic_proportion_value,
    })?;

    if data.cell_topic.ncols() > 1 {
        info!("normalizing cell topic proportion");
        data.cell_topic.sum_to_one_rows_inplace();
    }

    info!("Assign cells to pseudobulk samples to calibrate the null distribution");

    if let Some(ref conf_file) = args.confounder_file {
        info!("Loading known confounders from: {}", conf_file);
        let confounder_v = Mat::from_tsv(conf_file, None)?;
        info!(
            "Confounder matrix: {} individuals x {} covariates",
            confounder_v.nrows(),
            confounder_v.ncols()
        );
        data.sparse_data
            .assign_pseudobulk_with_known_confounders(&confounder_v, &data.cell_to_indv)?;
    } else {
        data.sparse_data.assign_pseudobulk_individuals(
            args.proj_dim,
            args.block_size,
            &data.cell_to_indv,
        )?;
    }

    let indv_names = data.sparse_data.batch_names().unwrap();
    let indv_to_exposure = data
        .indv_to_exposure
        .ok_or(anyhow::anyhow!("Missing exposure information"))?;
    let exposure_id = data
        .exposure_id
        .ok_or(anyhow::anyhow!("Missing exposure information"))?;
    let n_exposure = exposure_id.len();
    let topic_names = data.sorted_topic_names;

    let exposure_assignment: Vec<usize> = indv_names
        .iter()
        .map(|indv| {
            if let Some(exposure) = indv_to_exposure.get(indv) {
                exposure_id[exposure]
            } else {
                warn!("No exposure was assigned for sample {}, but it's kept for controlling confounders.", indv);
                n_exposure
            }
        })
        .collect();

    let indv_exposure_names: Vec<Box<str>> = indv_names
        .iter()
        .map(|indv| {
            if let Some(exposure) = indv_to_exposure.get(indv) {
                (indv.to_string() + "_" + exposure).into()
            } else {
                indv.to_string().into()
            }
        })
        .collect();

    let n_genes = data.sparse_data.num_rows();
    let n_topics = data.cell_topic.ncols();
    let gene_names = data.sparse_data.row_names()?;

    let cocoa_input = CocoaCollapseIn {
        n_genes,
        n_topics,
        knn: args.knn,
        n_opt_iter: args.num_opt_iter,
        hyper_param: Some((args.a0, args.b0)),
        cell_topic_nk: data.cell_topic,
        exposure_assignment: &exposure_assignment,
    };

    info!("Collecting statistics...");
    let cocoa_stat = data.sparse_data.collect_cocoa_stat(&cocoa_input)?;

    info!("Optimizing parameters...");
    let parameters = cocoa_stat.estimate_parameters()?;

    // Compute real contrast before consuming parameters
    let real_contrast = if args.n_permutations > 0 {
        Some(compute_exposure_contrast(&parameters, &exposure_assignment))
    } else {
        None
    };

    info!("Writing down the estimates...");

    let mut tau = Vec::with_capacity(parameters.len());
    let mut shared = Vec::with_capacity(parameters.len());
    let mut residual = Vec::with_capacity(parameters.len());

    for param in parameters.into_iter() {
        tau.push(param.exposure);
        shared.push(param.shared);
        residual.push(param.residual);
    }

    to_parquet(
        &tau,
        (Some(&gene_names), Some("gene")),
        (Some(&indv_exposure_names), Some("individual_exposure")),
        Some(&topic_names),
        &format!("{}.effect.parquet", args.out),
    )?;

    // Permutation testing
    if let Some(real_contrast) = real_contrast {
        info!(
            "Building match cache for {} permutations...",
            args.n_permutations
        );
        let cache = MatchCache::build(&data.sparse_data, args.knn)?;

        let mut null_sum = vec![0f32; n_genes];
        let mut null_sum_sq = vec![0f32; n_genes];

        let mut rng = rand::rngs::StdRng::seed_from_u64(args.permutation_seed);

        for p in 0..args.n_permutations {
            let mut perm_exposure = exposure_assignment.clone();
            perm_exposure.shuffle(&mut rng);

            let perm_stat = cache.replay_with_exposure(
                &cocoa_input.cell_topic_nk,
                &perm_exposure,
                n_genes,
                n_topics,
                args.num_opt_iter,
                Some((args.a0, args.b0)),
            )?;
            let perm_params = perm_stat.estimate_parameters()?;
            let perm_contrast = compute_exposure_contrast(&perm_params, &perm_exposure);

            for g in 0..n_genes {
                null_sum[g] += perm_contrast[g];
                null_sum_sq[g] += perm_contrast[g] * perm_contrast[g];
            }
            info!("Permutation {}/{}", p + 1, args.n_permutations);
        }

        // Compute z-scores and p-values
        let np = args.n_permutations as f32;
        let lines: Vec<Box<str>> = std::iter::once("gene\tcontrast\tz_score\tpvalue".into())
            .chain((0..n_genes).map(|g| {
                let null_mean = null_sum[g] / np;
                let null_var = null_sum_sq[g] / np - null_mean * null_mean;
                let null_std = null_var.max(1e-10).sqrt();
                let z = (real_contrast[g] - null_mean) / null_std;
                let pval = z_to_pvalue(z);
                let name = &gene_names[g];
                format!("{}\t{}\t{}\t{}", name, real_contrast[g], z, pval).into()
            }))
            .collect();

        let perm_file = format!("{}.perm.tsv.gz", args.out);
        write_lines(&lines, &perm_file)?;
        info!("Wrote permutation results to {}", perm_file);
    }

    info!("Done");
    Ok(())
}
