use crate::collapse_cocoa_data::*;
use crate::common::*;
use crate::input::*;
use crate::randomly_partition_data::*;
use crate::stat::*;

use crate::stat::control_factors::learn_control_factors;
use crate::stat::estimability::*;
use crate::stat::propensity::{fit_propensity, SWAP_SWEEPS};
use crate::stat::propensity_effect::{
    estimate_exposure_effect, estimate_log_effect, ExposureEffect,
};
use clap::Parser;
use data_beans::utilities::name_matching::GeneIndex;
use legume_numeric::matrix::common_io::{mkdir_parent, read_name_list};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use legume_numeric::matrix::traits::{IoOps, MatOps};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

#[cfg(test)]
mod tests;

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
        long_help = "Latent topic proportion files (comma-separated).\n\
                     Each file is a full `cell × topic` matrix."
    )]
    topic_proportion_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value = "logit",
        help = "Scale of the topic-proportion matrix (logit or prob)"
    )]
    topic_proportion_value: TopicValue,

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
        long_help = "Provide a known individual-level covariate matrix V of\n\
                     pre-exposure confounders. V enters the stage-2 effect\n\
                     estimator (propensity and outcome model); the pseudobulks of\n\
                     stage 1 are built from cell state regardless.\n\
                     \n\
                     The file is a tab-delimited matrix, n_indv x n_covar, in .tsv.gz format.\n\
                     Rows correspond to individuals 0, 1, 2, and so on, in order."
    )]
    covariate_file: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Separate SC data files (.zarr/.h5) for confounder adjustment",
        long_help = "Provide separate single-cell data, such as scRNA-seq.\n\
                     Its projection defines the cell-state pseudobulks.\n\
                     The counts still come from the primary data files.\n\
                     Both datasets must hold the same cells, in the same order."
    )]
    adjustment_data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Negative-control genes for learning individual confounders",
        long_help = "A file with one gene name per line: genes affected by\n\
                     individual-level confounders but not by the exposure.\n\
                     Their per-topic individual log rates give confounder factors\n\
                     (RUV-g), which enter both the outcome model and the\n\
                     propensity behind the conditional permutation."
    )]
    control_genes: Option<Box<str>>,

    #[arg(
        long,
        value_name = "auto|K",
        help = "Use top principal components of expression as confounders",
        long_help = "Learn individual-level confounders V as the top principal\n\
                     components of individual expression (all genes), when no\n\
                     control genes are given. Assumes the components carry no\n\
                     exposure signal. `auto` picks the number by the eigenvalue\n\
                     ratio; a number K keeps exactly K. Off unless given."
    )]
    confounder_pcs: Option<PcCount>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum individuals per pseudobulk; fewer drops it from the topic",
        long_help = "A pseudobulk contributes to a topic only if at least this many\n\
                     individuals have cells of that topic in it. Fewer cannot\n\
                     separate the cell-state rate from the individuals' levels.\n\
                     Kept and dropped counts are written to {out}.stage1.parquet."
    )]
    min_individuals_per_pb: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Levels a poorly mixed pseudobulk may merge up the code tree",
        long_help = "Pseudobulks are the leaves of a binary code tree over cell states.\n\
                     A leaf with fewer than --min-individuals-per-pb individuals\n\
                     merges with its sibling into the parent, for at most this many levels.\n\
                     Pseudobulks still poorly mixed after merging are dropped per topic."
    )]
    pb_merge_levels: usize,
}

/// Number of principal components: `auto` or a count.
#[derive(Clone, Copy, Debug)]
pub enum PcCount {
    Auto,
    Fixed(usize),
}

impl std::str::FromStr for PcCount {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(PcCount::Auto),
            _ => match s.parse::<usize>() {
                Ok(k) if k > 0 => Ok(PcCount::Fixed(k)),
                _ => Err(format!("expected `auto` or a positive count, got `{}`", s)),
            },
        }
    }
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
    anyhow::ensure!(
        n_exposure >= 2,
        "need at least two exposure groups, found {}",
        n_exposure
    );

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
    // Always on; a no-op on hard one-hot topics. Runs BEFORE topic-conditioned
    // CoCoA matching.
    //
    // Reference: adapted from residual collider stratification,
    //   Hartwig et al. (2023) Eur J Epidemiol
    //   "Avoiding collider bias in MR when performing stratified analyses"
    if data.cell_topic.ncols() > 1 {
        if topics_look_one_hot(&data.cell_topic) {
            warn!(
                "Topic matrix looks one-hot (hard assignments); \
                 residual collider adjustment is a no-op after row-normalization. \
                 Use soft proportions (-r) for residualization to affect matching weights."
            );
        }
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

    info!("Assign cells to pseudobulks by cell state");

    let mut covariate_file_v: Option<Mat> = None;
    if let Some(ref cov_file) = args.covariate_file {
        info!("Loading known covariates from: {}", cov_file);
        let covariate_v = Mat::from_tsv(cov_file, None)?;
        info!(
            "Covariate matrix: {} individuals x {} covariates (stage 2 only)",
            covariate_v.nrows(),
            covariate_v.ncols()
        );
        covariate_file_v = Some(covariate_v);
    }

    // Stage 1 pseudobulks come from cell state only, across individuals;
    // individual-level confounders enter stage 2.
    let n_indv_named = unique_indv_names.len();
    let spec = PartitionSpec {
        proj_dim: args.proj_dim,
        bits: partition_bits(data.sparse_data.num_columns(), n_indv_named, args.proj_dim),
        merge_levels: args.pb_merge_levels,
        min_individuals: args.min_individuals_per_pb,
        block_size: args.block_size,
    };
    if let Some(ref adj_files) = args.adjustment_data_files {
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
            &spec,
            &data.cell_to_indv,
        )?;
    } else {
        data.sparse_data
            .assign_pseudobulk_across_individuals(&spec, &data.cell_to_indv)?;
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
                (indv.to_string() + "_" + exposure.as_ref()).into()
            } else {
                indv.to_string().into()
            }
        })
        .collect();

    let n_genes = data.sparse_data.num_rows();
    let n_topics = data.cell_topic.ncols();
    let gene_names = data.sparse_data.row_names()?;

    let cocoa_input = CocoaCollapseIn {
        min_individuals_per_pb: args.min_individuals_per_pb,
        n_genes,
        n_topics,
        n_opt_iter: args.num_opt_iter,
        hyper_param: Some((args.a0, args.b0)),
        cell_topic_nk: data.cell_topic,
    };

    info!("Collecting statistics...");
    let cocoa_stat = data.sparse_data.collect_cocoa_stat(&cocoa_input)?;
    write_mixing_report(
        &cocoa_stat,
        &topic_names,
        args.min_individuals_per_pb,
        &format!("{}.stage1.parquet", args.output),
    )?;

    ////////////////////////////////////////////////////////
    // Individual-level confounders V enter through the   //
    // propensity P(X | V): a covariate of the stage-2    //
    // effect model and the law of the conditional        //
    // permutation. Factors learned from control genes or //
    // top PCs come from the label-free stage-1 sums.     //
    ////////////////////////////////////////////////////////

    let control_rows: Vec<usize> = match args.control_genes {
        Some(ref file) => read_control_rows(file, &gene_names)?,
        None => vec![],
    };
    let adjust = build_individual_covariates(
        covariate_file_v.as_ref(),
        &control_rows,
        args.confounder_pcs,
        &cocoa_stat,
        &indv_names,
        &exposure_assignment,
        n_exposure,
    )?;
    let propensity = match &adjust {
        Some(v) => {
            let prop = fit_propensity(v, &exposure_assignment, n_exposure)?;
            let mut col_names: Vec<Box<str>> = vec!["".into(); n_exposure];
            for (name, &idx) in exposure_id.iter() {
                col_names[idx] = name.clone();
                info!(
                    "propensity: group {} effective sample size {:.1}",
                    name, prop.ess[idx]
                );
            }
            let file = format!("{}.propensity.parquet", args.output);
            prop.prob.to_parquet_with_names(
                &file,
                (Some(&indv_names), Some("individual")),
                Some(&col_names),
            )?;
            info!(
                "exposure effect adjusted by the propensity on {} confounders",
                v.ncols()
            );
            Some(prop)
        }
        // no confounders: the ratio of means
        None => None,
    };
    // unclipped propensities enter the effect model
    let e = propensity.as_ref().map(|p| &p.prob_raw);

    // effects are relative to the reference level 0
    let mut level_names: Vec<Box<str>> = vec!["".into(); n_exposure];
    for (name, &idx) in exposure_id.iter() {
        level_names[idx] = name.clone();
    }
    let level_names = &level_names[..];

    // Stage 1 fit: the pseudobulk baseline gives each individual's expected
    // count at its own cell states, m(d,i). It uses no labels, so it is fit
    // once and shared by every permutation draw.
    info!("Fitting the stage-1 baseline...");
    let parameters = cocoa_stat.estimate_baseline();
    // Stage 2: tau and delta per topic, deconfounded by V when given.
    info!("Estimating the exposure effect (stage 2)...");
    let effects: Vec<ExposureEffect> = parameters
        .iter()
        .enumerate()
        .map(|(k, p)| {
            estimate_exposure_effect(
                cocoa_stat.indv_y1_stat(k),
                &p.indv_offset,
                &exposure_assignment,
                n_exposure,
                e,
                true,
            )
        })
        .collect();
    info!(
        "stage 2 converged within {} iterations",
        effects.iter().map(|e| e.iterations).max().unwrap_or(0)
    );
    ////////////////////////////////////////////////////////
    // Effects that cannot be estimated are flagged and   //
    // left out (NA): per (gene, topic) from the observed //
    // data, per run from the overlap and the null.       //
    ////////////////////////////////////////////////////////
    let mut mask = EstimableMask::from_topic_flags(
        &effects
            .iter()
            .enumerate()
            .map(|(k, eff)| {
                topic_flags(
                    cocoa_stat.indv_y1_stat(k),
                    &eff.used,
                    &exposure_assignment,
                    n_exposure,
                    MIN_INDIVIDUALS_PER_LEVEL,
                )
            })
            .collect::<Vec<_>>(),
    );
    if let Some(prop) = &propensity {
        if prop.ess.iter().any(|&ess| ess < MIN_ARM_ESS) {
            warn!(
                "an exposure arm has propensity effective sample size below {}: \
                 no gene is tested",
                MIN_ARM_ESS
            );
            mask.flag_all(Flag::WeakOverlap);
        }
    }

    // Relabelings for the permutation test (sequential, for
    // reproducibility): drawn from the propensity when there is one, a
    // uniform shuffle otherwise.
    let permuted_exposures: Vec<Vec<usize>> = if args.n_permutations == 0 {
        Vec::new()
    } else if let Some(prop) = &propensity {
        info!("Conditional permutation from the fitted propensity");
        prop.conditional_permutations(
            &exposure_assignment,
            args.n_permutations,
            SWAP_SWEEPS,
            args.permutation_seed,
        )
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.permutation_seed);
        (0..args.n_permutations)
            .map(|_| {
                let mut perm = exposure_assignment.clone();
                perm.shuffle(&mut rng);
                perm
            })
            .collect()
    };
    // with few draws, repeats are expected even from a healthy sampler
    let min_distinct = MIN_DISTINCT_PERMUTATIONS.min(args.n_permutations.div_ceil(2));
    if args.n_permutations > 0 && permutations_degenerate(&permuted_exposures, min_distinct) {
        warn!(
            "the permutation draws hold fewer than {} distinct relabelings: \
             no gene is tested",
            min_distinct
        );
        mask.flag_all(Flag::DegenerateNull);
    }
    for (flag, count) in mask.summary(n_genes) {
        warn!("{} genes left out: {}", count, flag.as_str());
    }
    let gene_flags: Vec<Box<str>> = (0..n_genes)
        .map(|d| mask.gene_flag(d).map_or("", |f| f.as_str()).into())
        .collect();

    let group_contrast = mask.mean_log_effect(effects.iter().map(|e| &e.psi));
    let unadjusted_contrast = e.map(|_| {
        mask.mean_log_effect(
            log_effects(
                &cocoa_stat,
                &parameters,
                &exposure_assignment,
                n_exposure,
                None,
                true,
            )
            .iter(),
        )
    });
    let is_control: Option<Vec<f32>> = (!control_rows.is_empty()).then(|| {
        let mut flag = vec![0f32; n_genes];
        for &g in &control_rows {
            flag[g] = 1.0;
        }
        flag
    });

    info!("Writing down the estimates...");

    let log_mean: Vec<f32> = (0..n_genes)
        .map(|g| parameters.iter().map(|p| p.log_mean[g]).sum::<f32>() / parameters.len() as f32)
        .collect();

    write_effects(
        &effects,
        &mask,
        &gene_names,
        &topic_names,
        level_names,
        &format!("{}.effect.parquet", args.output),
    )?;
    write_deltas(
        &effects,
        &gene_names,
        &topic_names,
        &indv_names,
        &indv_exposure_names,
        &format!("{}.delta.parquet", args.output),
    )?;

    {
        let file = format!("{}.contrast.parquet", args.output);
        let adjusted: Vec<Vec<f32>> = level_columns(&group_contrast);
        let unadjusted: Option<Vec<Vec<f32>>> = unadjusted_contrast.as_ref().map(level_columns);
        let mut columns: Vec<(Box<str>, Column)> = Vec::new();
        for (l, c) in adjusted.iter().enumerate() {
            columns.push((level_col("contrast", level_names, l + 1), Column::F32(c)));
        }
        columns.push(("log_mean".into(), Column::F32(&log_mean)));
        if let Some(u) = &unadjusted {
            for (l, c) in u.iter().enumerate() {
                let name = format!("{}_unadjusted", level_col("contrast", level_names, l + 1));
                columns.push((name.into(), Column::F32(c)));
            }
        }
        if let Some(c) = &is_control {
            columns.push(("control".into(), Column::F32(c)));
        }
        columns.push(("flag".into(), Column::Str(&gene_flags)));
        write_named_table(&file, "gene", &gene_names, &columns)?;
        info!(
            "Wrote contrasts against reference level {} to {}",
            level_names[0], file
        );
    }

    // Permutation testing
    if args.n_permutations > 0 {
        // Stage 1 (sums and baseline) does not depend on the labels, so every
        // draw reuses it; only stage 2 sees the new labels.
        let n_perm = args.n_permutations;

        let perm_contrasts: Vec<Mat> = permuted_exposures
            .into_par_iter()
            .enumerate()
            .map(|(p, perm_exposure)| -> anyhow::Result<Mat> {
                // draws already run in parallel
                let contrast = mask.mean_log_effect(
                    log_effects(
                        &cocoa_stat,
                        &parameters,
                        &perm_exposure,
                        n_exposure,
                        e,
                        false,
                    )
                    .iter(),
                );
                info!("Permutation {}/{}", p + 1, n_perm);
                Ok(contrast)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let perm_file = format!("{}.perm.parquet", args.output);
        write_permutation_tests(
            &group_contrast,
            &perm_contrasts,
            &gene_flags,
            &gene_names,
            level_names,
            &perm_file,
        )?;
        info!("Wrote permutation results to {}", perm_file);
    }

    info!("Done");
    Ok(())
}

/// Stage-2 log effects `psi` per topic, from the stage-1 statistics and the
/// baseline offsets; `e` is the unclipped propensity, `None` without
/// confounders.
fn log_effects(
    stat: &CocoaStat,
    params: &[Baseline],
    exposure: &[usize],
    n_levels: usize,
    e: Option<&Mat>,
    parallel: bool,
) -> Vec<Mat> {
    params
        .iter()
        .enumerate()
        .map(|(k, p)| {
            estimate_log_effect(
                stat.indv_y1_stat(k),
                &p.indv_offset,
                exposure,
                n_levels,
                e,
                parallel,
            )
        })
        .collect()
}

/// Columns of a gene x level matrix as vectors.
fn level_columns(m: &Mat) -> Vec<Vec<f32>> {
    m.column_iter()
        .map(|c| c.iter().cloned().collect())
        .collect()
}

/// Column name for level `l`. With two levels there is one contrast and its
/// columns keep the plain `prefix` (`contrast`, `pvalue`, ...); with more,
/// every non-reference level gets `prefix_<level>`. The README lists the full
/// schema.
fn level_col(prefix: &str, level_names: &[Box<str>], l: usize) -> Box<str> {
    if level_names.len() == 2 {
        prefix.into()
    } else {
        format!("{}_{}", prefix, level_names[l]).into()
    }
}

/// tau per (gene, topic) for every level, and the log effect of every
/// non-reference level.
fn write_effects(
    effects: &[ExposureEffect],
    mask: &EstimableMask,
    gene_names: &[Box<str>],
    topic_names: &[Box<str>],
    level_names: &[Box<str>],
    file: &str,
) -> anyhow::Result<()> {
    let k = level_names.len();
    let mut genes = Vec::new();
    let mut topics = Vec::new();
    let mut tau: Vec<Vec<f32>> = vec![Vec::new(); k];
    let mut log_effect: Vec<Vec<f32>> = vec![Vec::new(); k - 1];
    let mut flags: Vec<Box<str>> = Vec::new();
    for (t, (e, topic)) in effects.iter().zip(topic_names).enumerate() {
        for (d, gene) in gene_names.iter().enumerate() {
            genes.push(gene.clone());
            topics.push(topic.clone());
            flags.push(mask.flag(d, t).map_or("", |f| f.as_str()).into());
            for (l, t) in tau.iter_mut().enumerate() {
                t.push(e.tau[(d, l)]);
            }
            for (l, c) in log_effect.iter_mut().enumerate() {
                c.push(e.psi[(d, l + 1)]);
            }
        }
    }
    let mut columns: Vec<(Box<str>, Column)> = vec![("topic".into(), Column::Str(&topics))];
    for (l, t) in tau.iter().enumerate() {
        columns.push((format!("tau_{}", level_names[l]).into(), Column::F32(t)));
    }
    for (l, c) in log_effect.iter().enumerate() {
        columns.push((level_col("log_effect", level_names, l + 1), Column::F32(c)));
    }
    columns.push(("flag".into(), Column::Str(&flags)));
    write_named_table(file, "gene", &genes, &columns)
}

/// Per level: z against the permutation null (mean and sd of the draws) and
/// its normal p-value. With more than one non-reference level, also a global
/// test that every level equals the reference: the sum of squared z over
/// levels, with a permutation p-value from the same draws.
fn write_permutation_tests(
    observed: &Mat,
    draws: &[Mat],
    gene_flags: &[Box<str>],
    gene_names: &[Box<str>],
    level_names: &[Box<str>],
    file: &str,
) -> anyhow::Result<()> {
    let (n_genes, r) = observed.shape();
    let b = draws.len() as f32;
    let mut mean = Mat::zeros(n_genes, r);
    let mut sq = Mat::zeros(n_genes, r);
    for dr in draws {
        mean += dr;
        sq += dr.component_mul(dr);
    }
    mean /= b;
    let sd = Mat::from_fn(n_genes, r, |d, l| {
        (sq[(d, l)] / b - mean[(d, l)].powi(2)).max(1e-10).sqrt()
    });
    let z = Mat::from_fn(n_genes, r, |d, l| {
        (observed[(d, l)] - mean[(d, l)]) / sd[(d, l)]
    });

    let mut cols: Vec<(Box<str>, Vec<f32>)> = Vec::new();
    for l in 0..r {
        let c: Vec<f32> = observed.column(l).iter().cloned().collect();
        let zl: Vec<f32> = z.column(l).iter().cloned().collect();
        let pl: Vec<f32> = zl
            .iter()
            .zip(gene_flags)
            .map(|(&v, f)| {
                if f.is_empty() {
                    z_to_pvalue(v)
                } else {
                    f32::NAN
                }
            })
            .collect();
        cols.push((level_col("contrast", level_names, l + 1), c));
        cols.push((level_col("z_score", level_names, l + 1), zl));
        cols.push((level_col("pvalue", level_names, l + 1), pl));
    }
    if r > 1 {
        let stat = |m: &Mat, d: usize| -> f32 {
            (0..r)
                .map(|l| ((m[(d, l)] - mean[(d, l)]) / sd[(d, l)]).powi(2))
                .sum()
        };
        let global: Vec<f32> = (0..n_genes).map(|d| stat(observed, d)).collect();
        let pglobal: Vec<f32> = (0..n_genes)
            .map(|d| {
                if !gene_flags[d].is_empty() {
                    return f32::NAN;
                }
                let exceed = draws.iter().filter(|m| stat(m, d) >= global[d]).count();
                (1 + exceed) as f32 / (1.0 + b)
            })
            .collect();
        cols.push(("global_stat".into(), global));
        cols.push(("global_pvalue".into(), pglobal));
    }
    let mut columns: Vec<(Box<str>, Column)> = cols
        .iter()
        .map(|(n, v)| (n.clone(), Column::F32(v)))
        .collect();
    columns.push(("flag".into(), Column::Str(gene_flags)));
    write_named_table(file, "gene", gene_names, &columns)
}

/// delta per (gene, individual, topic), for the individuals in the contrast.
fn write_deltas(
    effects: &[ExposureEffect],
    gene_names: &[Box<str>],
    topic_names: &[Box<str>],
    indv_names: &[Box<str>],
    indv_exposure_names: &[Box<str>],
    file: &str,
) -> anyhow::Result<()> {
    let mut genes = Vec::new();
    let (mut indv, mut indv_exp, mut topics, mut delta) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (e, topic) in effects.iter().zip(topic_names) {
        for i in (0..indv_names.len()).filter(|&i| e.used[i]) {
            for (d, gene) in gene_names.iter().enumerate() {
                genes.push(gene.clone());
                indv.push(indv_names[i].clone());
                indv_exp.push(indv_exposure_names[i].clone());
                topics.push(topic.clone());
                delta.push(e.delta[(d, i)]);
            }
        }
    }
    write_named_table(
        file,
        "gene",
        &genes,
        &[
            ("individual".into(), Column::Str(&indv)),
            ("individual_exposure".into(), Column::Str(&indv_exp)),
            ("topic".into(), Column::Str(&topics)),
            ("delta".into(), Column::F32(&delta)),
        ],
    )
}

/// Rows of the genes listed in `file` (one name per line).
/// Rows of the genes listed in `file`, matched as `--feature-list-file`
/// matches them (symbol, Ensembl id, `ENSG_SYMBOL`, case).
fn read_control_rows(file: &str, gene_names: &[Box<str>]) -> anyhow::Result<Vec<usize>> {
    let names = read_name_list(file)?;
    let index = GeneIndex::build(gene_names);
    let mut rows: Vec<usize> = names.iter().filter_map(|g| index.match_gene(g)).collect();
    rows.sort_unstable();
    rows.dedup();
    info!(
        "control genes: {} listed, {} found among {} genes",
        names.len(),
        rows.len(),
        gene_names.len()
    );
    anyhow::ensure!(
        rows.len() >= 2,
        "fewer than two control genes found in the data"
    );
    Ok(rows)
}

/// Individual x covariate matrix in batch order from the covariate file
/// (row `r` belongs to the individual named `r`, as for the pseudobulk
/// partition) and the control-gene factors, standardized over individuals.
/// `None` when neither source is given.
fn build_individual_covariates(
    file_v: Option<&Mat>,
    control_rows: &[usize],
    pcs: Option<PcCount>,
    stat: &CocoaStat,
    indv_names: &[Box<str>],
    exposure_assignment: &[usize],
    n_exposure: usize,
) -> anyhow::Result<Option<Mat>> {
    let n_indv = indv_names.len();
    let mut cols: Vec<DVec> = Vec::new();

    if let Some(v) = file_v {
        for (i, name) in indv_names.iter().enumerate() {
            let row: Option<usize> = name.parse().ok().filter(|&r: &usize| r < v.nrows());
            anyhow::ensure!(
                row.is_some() || exposure_assignment[i] >= n_exposure,
                "individual {} has no row in the covariate file",
                name
            );
        }
        for c in 0..v.ncols() {
            cols.push(DVec::from_fn(n_indv, |i, _| {
                indv_names[i]
                    .parse::<usize>()
                    .ok()
                    .filter(|&r| r < v.nrows())
                    .map_or(0.0, |r| v[(r, c)])
            }));
        }
    }

    // control genes if given; otherwise, on request, the top principal
    // components of all genes, assumed to carry no exposure signal
    let all_rows: Vec<usize>;
    let (factor_rows, n_factors, source) = match (control_rows.is_empty(), pcs) {
        (false, _) => (control_rows, None, "control-gene factors"),
        (true, None) => (&[][..], None, ""),
        (true, Some(k)) => {
            all_rows = (0..stat.indv_y1_stat(0).nrows()).collect();
            let n = match k {
                PcCount::Auto => None,
                PcCount::Fixed(k) => Some(k),
            };
            (&all_rows[..], n, "top principal components")
        }
    };

    if !factor_rows.is_empty() {
        let learned = learn_control_factors(stat, factor_rows, n_factors)?;
        let top: Vec<String> = learned
            .singular_values
            .iter()
            .take(6)
            .map(|s| format!("{:.2}", s))
            .collect();
        info!(
            "{}: {} kept from {} features (singular values {})",
            source,
            learned.factors.ncols(),
            learned.n_features,
            top.join(", ")
        );
        cols.extend(learned.factors.column_iter().map(|c| c.into_owned()));
    }

    if cols.is_empty() {
        return Ok(None);
    }
    // standardize, dropping constant columns
    cols.retain(|c| c.variance() > 1e-12);
    Ok((!cols.is_empty()).then(|| Mat::from_columns(&cols).scale_columns()))
}

/// Per topic: pseudobulks and cells kept or dropped for pooling too few
/// individuals, and how many individuals the kept pseudobulks link. Warns
/// when a topic drops much of its data or pools few individuals.
fn write_mixing_report(
    stat: &CocoaStat,
    topic_names: &[Box<str>],
    min_individuals: usize,
    file: &str,
) -> anyhow::Result<()> {
    let n = stat.n_topics();
    let (mut kept, mut dropped) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut cells_kept, mut cells_dropped) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut median = Vec::with_capacity(n);
    for (k, topic) in topic_names.iter().enumerate().take(n) {
        let mix = stat.mixing(k);
        let mut per = mix.individuals_per_pseudobulk.clone();
        per.sort_unstable();
        let med = per.get(per.len() / 2).copied().unwrap_or(0) as f32;
        let total = mix.cells_kept + mix.cells_dropped;
        if total > 0.0 && mix.cells_dropped > 0.2 * total {
            warn!(
                "topic {}: {:.0}% of cells sit in pseudobulks with fewer than {} individuals \
                 and were dropped",
                topic,
                100.0 * mix.cells_dropped / total,
                min_individuals
            );
        }
        if mix.pseudobulks_kept == 0 {
            warn!(
                "topic {}: no pseudobulk pools {} individuals; its effects are not estimable",
                topic, min_individuals
            );
        }
        kept.push(mix.pseudobulks_kept as f32);
        dropped.push(mix.pseudobulks_dropped as f32);
        cells_kept.push(mix.cells_kept);
        cells_dropped.push(mix.cells_dropped);
        median.push(med);
    }
    write_named_table(
        file,
        "topic",
        topic_names,
        &[
            ("pseudobulks_kept".into(), Column::F32(&kept)),
            ("pseudobulks_dropped".into(), Column::F32(&dropped)),
            ("cells_kept".into(), Column::F32(&cells_kept)),
            ("cells_dropped".into(), Column::F32(&cells_dropped)),
            (
                "median_individuals_per_pseudobulk".into(),
                Column::F32(&median),
            ),
        ],
    )
}
