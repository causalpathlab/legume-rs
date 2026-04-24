use crate::collapse_cocoa_data::*;
use crate::common::*;
use crate::input::*;
use crate::randomly_partition_data::*;
use crate::stat::*;

use clap::Parser;
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;
use matrix_param::io::*;
use matrix_util::common_io::write_lines;
use matrix_util::traits::{IoOps, MatOps};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

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
        help = "Known covariate matrix file (tsv.gz, n_indv x n_covar).",
        long_help = "Provide a known individual-level covariate matrix V instead of\n\
                     discovering confounders by random projection. The file should be\n\
                     a tab-delimited matrix (n_indv x n_covar) in .tsv.gz format.\n\
                     Rows correspond to individuals 0, 1, 2, ... in order."
    )]
    #[arg(conflicts_with = "adjustment_data_files")]
    covariate_file: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Separate SC data files (.zarr/.h5) for confounder adjustment.",
        long_help = "Provide separate single-cell data (e.g., scRNA-seq) for computing\n\
                     the confounder-adjustment projection. KNN matching will be based on\n\
                     these data, while y1/y0 counts come from the primary data files.\n\
                     Both datasets must have the same cells in the same column order."
    )]
    adjustment_data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable residual collider adjustment of topic proportions.",
        long_help = "By default, the exposure-driven shift is removed from topic logits\n\
                     before analysis to break collider bias (X -> A <- U). Use this flag\n\
                     to disable this adjustment, e.g. when cell type is known not to be\n\
                     a collider or for comparison experiments.\n\
                     \n\
                     Reference: adapted from residual collider stratification,\n\
                     Hartwig et al. (2023) Eur J Epidemiol."
    )]
    no_residualize_topics: bool,

    #[arg(
        long = "no-adjust-housekeeping",
        default_value_t = false,
        help = "Disable NB-Fisher housekeeping gene adjustment.",
        long_help = "By default, y1/y0 sufficient statistics are row-scaled by NB-Fisher\n\
                     weights w_g = 1 / (1 + π_g · s̄ · φ(μ_g)) after accumulation, so\n\
                     τ, μ, γ posteriors contract toward the prior for housekeeping\n\
                     (high-mean / high-dispersion) genes. Matches pinto's adjustment."
    )]
    no_adjust_housekeeping: bool,

    #[arg(
        long = "refine",
        default_value_t = false,
        help = "Refine cell→pseudobulk membership via senna's multilevel DC-Poisson pass.",
        long_help = "When set, cocoa routes pseudobulk assignment through the same\n\
                     multilevel path senna uses (collapse_columns_multilevel_vec with\n\
                     BBKNN + DC-Poisson refinement). Cells are reassigned across\n\
                     sibling pseudobulks by Poisson likelihood under NB-Fisher gene\n\
                     weighting, so each pseudobulk is more internally coherent.\n\
                     Only applies when using the default pseudobulk path\n\
                     (no --covariate-file and no --adjustment-data-files)."
    )]
    refine: bool,

    #[arg(
        long = "refine-num-levels",
        default_value_t = 2,
        help = "Number of coarsening levels for multilevel refinement."
    )]
    refine_num_levels: usize,

    #[arg(
        long = "refine-knn-super-cells",
        default_value_t = 10,
        help = "BBKNN fan-out for multilevel refinement."
    )]
    refine_knn_super_cells: usize,
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

    // Build individual name -> index mapping early (needed for residualization)
    let indv_to_exposure = data
        .indv_to_exposure
        .take()
        .ok_or(anyhow::anyhow!("Missing exposure information"))?;
    let exposure_id = data
        .exposure_id
        .take()
        .ok_or(anyhow::anyhow!("Missing exposure information"))?;
    let n_exposure = exposure_id.len();

    // Map individual names to numeric indices (filter unmatched "NA" cells)
    let unique_indv_names: Vec<Box<str>> = data
        .cell_to_indv
        .iter()
        .filter(|s| !s.is_empty() && s.as_ref() != "NA")
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let indv_name_to_index: HashMap<Box<str>, usize> = unique_indv_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();
    // Cell -> individual index mapping (usize::MAX for unmatched cells)
    let cell_to_individual_index: Vec<usize> = data
        .cell_to_indv
        .iter()
        .map(|name| indv_name_to_index.get(name).copied().unwrap_or(usize::MAX))
        .collect();

    // Individual -> exposure group mapping
    let individual_exposure_group: Vec<usize> = unique_indv_names
        .iter()
        .map(|indv| {
            if let Some(exposure) = indv_to_exposure.get(indv) {
                exposure_id[exposure]
            } else {
                n_exposure // unassigned individuals kept for confounding control
            }
        })
        .collect();

    // Residual collider adjustment: remove the exposure-driven shift from
    // topic proportions to break collider bias (X -> A <- U).
    //
    // By default ON. Use --no-residualize-topics to disable.
    //
    // Reference: adapted from residual collider stratification,
    //   Hartwig et al. (2023) Eur J Epidemiol
    //   "Avoiding collider bias in MR when performing stratified analyses"
    if !args.no_residualize_topics && data.cell_topic.ncols() > 1 {
        info!("Residualizing topic proportions to remove exposure-driven collider bias");
        let shifts = remove_exposure_effect_from_topic_proportions(
            &mut data.cell_topic,
            &cell_to_individual_index,
            &individual_exposure_group,
        );
        for (k, &shift) in shifts.iter().enumerate() {
            info!("  topic {}: max exposure shift removed = {:.4}", k, shift);
        }
    }

    if data.cell_topic.ncols() > 1 {
        info!("normalizing cell topic proportion");
        data.cell_topic.sum_to_one_rows_inplace();
    }

    info!("Assign cells to pseudobulk samples to calibrate the null distribution");

    if let Some(ref cov_file) = args.covariate_file {
        info!("Loading known covariates from: {}", cov_file);
        let covariate_v = Mat::from_tsv(cov_file, None)?;
        info!(
            "Covariate matrix: {} individuals x {} covariates",
            covariate_v.nrows(),
            covariate_v.ncols()
        );
        data.sparse_data
            .assign_pseudobulk_with_known_confounders(&covariate_v, &data.cell_to_indv)?;
    } else if let Some(ref adj_files) = args.adjustment_data_files {
        info!("Loading adjustment data from {} file(s)", adj_files.len());
        let adj_data = read_adjustment_data(adj_files, args.preload_data)?;
        if adj_data.num_columns() != data.sparse_data.num_columns() {
            return Err(anyhow::anyhow!(
                "Adjustment data has {} cells but test data has {} cells — they must match",
                adj_data.num_columns(),
                data.sparse_data.num_columns()
            ));
        }
        data.sparse_data.assign_pseudobulk_from_adjustment_data(
            &adj_data,
            args.proj_dim,
            args.block_size,
            &data.cell_to_indv,
        )?;
    } else if args.refine {
        info!(
            "Refining pseudobulk assignment via multilevel DC-Poisson (num_levels={}, knn_super_cells={})",
            args.refine_num_levels, args.refine_knn_super_cells
        );
        let mut refine_settings =
            crate::randomly_partition_data::RefineSettings::with_proj_dim(args.proj_dim);
        refine_settings.num_levels = args.refine_num_levels;
        refine_settings.knn_super_cells = args.refine_knn_super_cells;
        data.sparse_data.assign_pseudobulk_individuals_refined(
            args.proj_dim,
            args.block_size,
            &data.cell_to_indv,
            &refine_settings,
        )?;
    } else {
        data.sparse_data.assign_pseudobulk_individuals(
            args.proj_dim,
            args.block_size,
            &data.cell_to_indv,
        )?;
    }

    let indv_names = data.sparse_data.batch_names().unwrap();
    let topic_names = data.sorted_topic_names;

    // Build exposure assignment indexed by batch order (for downstream use)
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

    let gene_weights: Option<Vec<f32>> = if args.no_adjust_housekeeping {
        None
    } else {
        info!("Computing NB-Fisher housekeeping weights (--no-adjust-housekeeping to disable)");
        let w = compute_nb_fisher_weights(&data.sparse_data, Some(args.block_size))?;
        let wmin = w.iter().cloned().fold(f32::INFINITY, f32::min);
        let wmax = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        info!(
            "NB-Fisher weights: {} genes, min={:.4}, max={:.4}",
            w.len(),
            wmin,
            wmax
        );
        Some(w)
    };

    let cocoa_input = CocoaCollapseIn {
        n_genes,
        n_topics,
        knn: args.knn,
        n_opt_iter: args.num_opt_iter,
        hyper_param: Some((args.a0, args.b0)),
        cell_topic_nk: data.cell_topic,
        exposure_assignment: &exposure_assignment,
        gene_weights: gene_weights.as_deref(),
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

        // Pre-generate shuffled exposure assignments (sequential, for reproducibility)
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.permutation_seed);
        let permuted_exposures: Vec<Vec<usize>> = (0..args.n_permutations)
            .map(|_| {
                let mut perm = exposure_assignment.clone();
                perm.shuffle(&mut rng);
                perm
            })
            .collect();

        // Run permutations in parallel — MatchCache is thread-safe (&self)
        let hyper_param = Some((args.a0, args.b0));
        let num_opt_iter = args.num_opt_iter;
        let cell_topic = &cocoa_input.cell_topic_nk;
        let n_perm = args.n_permutations;
        let perm_gene_weights = gene_weights.as_deref();

        let perm_contrasts: Vec<Vec<f32>> = permuted_exposures
            .into_par_iter()
            .enumerate()
            .map(|(p, perm_exposure)| -> anyhow::Result<Vec<f32>> {
                let perm_stat = cache.replay_with_exposure(
                    cell_topic,
                    &perm_exposure,
                    n_genes,
                    n_topics,
                    num_opt_iter,
                    hyper_param,
                    perm_gene_weights,
                )?;
                let perm_params = perm_stat.estimate_parameters()?;
                let contrast = compute_exposure_contrast(&perm_params, &perm_exposure);
                info!("Permutation {}/{}", p + 1, n_perm);
                Ok(contrast)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Reduce: accumulate null distribution statistics
        let mut null_sum = vec![0f32; n_genes];
        let mut null_sum_sq = vec![0f32; n_genes];
        for contrast in &perm_contrasts {
            for g in 0..n_genes {
                null_sum[g] += contrast[g];
                null_sum_sq[g] += contrast[g] * contrast[g];
            }
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
