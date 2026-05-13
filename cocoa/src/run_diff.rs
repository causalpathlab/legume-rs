use crate::cnv_call::{self, CnvArgs};
use crate::collapse_cocoa_data::*;
use crate::common::*;
use crate::input::*;
use crate::randomly_partition_data::*;
use crate::stat::*;

use clap::Parser;
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;
use matrix_param::io::*;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::{IoOps, MatOps};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    #[arg(
        required = true,
        help = "Single-cell data files (.zarr / .h5)",
        long_help = "Single-cell sparse data files in `.zarr` or `.h5` format.\n\
                     All files in the list must share the same format and gene order.\n\
                     Convert `.mtx` to `.zarr` / `.h5` with the `data-beans` CLI."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'i',
        long,
        value_delimiter = ',',
        required = true,
        help = "Individual membership files (comma-separated)",
        long_help = "Individual membership files (comma-separated). Each line is either:\n  \
                     * an individual ID (one per cell, in cell order), or\n  \
                     * a (cell, individual ID) pair."
    )]
    indv_files: Vec<Box<str>>,

    #[arg(
        short = 'e',
        long,
        required = true,
        help = "Exposure assignment file",
        long_help = "Each line is a (individual name, exposure name) pair."
    )]
    exposure_assignment_file: Box<str>,

    #[arg(
        short = 't',
        long,
        value_delimiter = ',',
        help = "Latent topic assignment files (comma-separated)",
        long_help = "Latent topic assignment files (comma-separated). Each line is either:\n  \
                     * a topic name (one per cell, in cell order), or\n  \
                     * a (cell, topic name) pair."
    )]
    topic_assignment_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'r',
        long,
        value_delimiter = ',',
        help = "Latent topic proportion files (comma-separated)",
        long_help = "Latent topic proportion files (comma-separated). Each file is a full\n\
                     `cell × topic` matrix."
    )]
    topic_proportion_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value = "logit",
        help = "Scale of the topic-proportion matrix (logit or prob)"
    )]
    topic_proportion_value: TopicValue,

    #[arg(
        short = 'n',
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each condition"
    )]
    knn: usize,

    #[arg(
        short = 'p',
        long,
        default_value_t = 10,
        help = "Projection dimension for confounder factors"
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel column reads"
    )]
    block_size: usize,

    #[arg(long, help = "Number of iterations for optimization")]
    num_opt_iter: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Hyperparameter a0 in Gamma(a0, b0)"
    )]
    a0: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Hyperparameter b0 in Gamma(a0, b0)"
    )]
    b0: f32,

    #[arg(
        short,
        long,
        required = true,
        value_name = "PREFIX",
        help = "Output file name prefix"
    )]
    output: Box<str>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of exposure-label permutations for empirical p-values"
    )]
    n_permutations: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "Random seed for permutation testing"
    )]
    permutation_seed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all column (cell) data into memory before fitting"
    )]
    preload_data: bool,

    #[arg(
        long,
        help = "Known covariate matrix file (tsv.gz, n_indv × n_covar)",
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
        help = "Separate SC data files (.zarr/.h5) for confounder adjustment",
        long_help = "Provide separate single-cell data (e.g., scRNA-seq) for computing\n\
                     the confounder-adjustment projection. KNN matching will be based on\n\
                     these data, while y1/y0 counts come from the primary data files.\n\
                     Both datasets must have the same cells in the same column order."
    )]
    adjustment_data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable residual collider adjustment of topic proportions",
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
        long,
        default_value_t = false,
        help = "Disable NB-Fisher housekeeping gene adjustment",
        long_help = "By default, y1/y0 sufficient statistics are row-scaled by NB-Fisher\n\
                     weights w_g = 1 / (1 + π_g · s̄ · φ(μ_g)) after accumulation, so\n\
                     τ, μ, γ posteriors contract toward the prior for housekeeping\n\
                     (high-mean / high-dispersion) genes. Matches pinto's adjustment."
    )]
    no_adjust_housekeeping: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Refine cell → pseudobulk membership via senna's multilevel DC-Poisson pass",
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
        long,
        default_value_t = 2,
        help = "Number of coarsening levels for multilevel refinement"
    )]
    refine_num_levels: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "BBKNN fan-out for multilevel refinement"
    )]
    refine_knn_pb_samples: usize,

    #[command(flatten)]
    cnv: CnvArgs,
}

/////////////////////////////////////
// Run CoCoA differential analysis //
/////////////////////////////////////

pub fn run_cocoa_diff(args: DiffArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.output)?;

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
            "Refining pseudobulk assignment via multilevel DC-Poisson (num_levels={}, knn_pb_samples={})",
            args.refine_num_levels, args.refine_knn_pb_samples
        );
        let mut refine_settings =
            crate::randomly_partition_data::RefineSettings::with_proj_dim(args.proj_dim);
        refine_settings.num_levels = args.refine_num_levels;
        refine_settings.knn_pb_samples = args.refine_knn_pb_samples;
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

    // CNV calling on the cocoa-residual (indv × topic) signal
    let cnv_run = cnv_call::run_cnv_calling(
        &args.cnv,
        &parameters,
        &indv_names,
        &topic_names,
        &gene_names,
        &args.output,
    )?;

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
        &format!("{}.effect.parquet", args.output),
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
        let mut contrast_col = vec![0f32; n_genes];
        let mut z_col = vec![0f32; n_genes];
        let mut p_col = vec![0f32; n_genes];
        for g in 0..n_genes {
            let null_mean = null_sum[g] / np;
            let null_var = null_sum_sq[g] / np - null_mean * null_mean;
            let null_std = null_var.max(1e-10).sqrt();
            let z = (real_contrast[g] - null_mean) / null_std;
            contrast_col[g] = real_contrast[g];
            z_col[g] = z;
            p_col[g] = z_to_pvalue(z);
        }

        // Optional CNV concordance columns
        let concordance = cnv_run.as_ref().map(|(cnv_result, signal)| {
            cnv_call::compute_deg_concordance(signal, cnv_result, n_genes)
        });

        // Assemble [n_genes × n_cols] numeric matrix and write as parquet,
        // matching the format used elsewhere in cocoa (effect.parquet).
        let mut col_names: Vec<Box<str>> = vec![
            "contrast".to_string().into(),
            "z_score".to_string().into(),
            "pvalue".to_string().into(),
        ];
        let mut columns: Vec<DVec> = vec![
            DVec::from(contrast_col),
            DVec::from(z_col),
            DVec::from(p_col),
        ];
        if let Some(c) = concordance.as_ref() {
            col_names.extend(
                ["cnv_concordance_r", "cnv_concordance_p", "cnv_state"]
                    .iter()
                    .map(|s| (*s).to_string().into()),
            );
            columns.push(DVec::from(c.r.clone()));
            columns.push(DVec::from(c.p.clone()));
            columns.push(DVec::from_iterator(
                n_genes,
                c.state.iter().map(|&x| x as f32),
            ));
        }
        let perm_mat = Mat::from_columns(&columns);
        let perm_file = format!("{}.perm.parquet", args.output);
        perm_mat.to_parquet_with_names(
            &perm_file,
            (Some(&gene_names), Some("gene")),
            Some(&col_names),
        )?;
        info!("Wrote permutation results to {}", perm_file);
    }

    info!("Done");
    Ok(())
}
