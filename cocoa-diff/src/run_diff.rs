use crate::collapse_data::*;
use crate::common::*;
use crate::input::*;
use crate::randomly_partition_data::*;

use matrix_param::io::*;
use matrix_util::common_io::basename;
use matrix_util::common_io::ReadLinesOut;
use matrix_util::common_io::{extension, read_lines_of_words_delim};
use matrix_util::dmatrix_util::concatenate_vertical;
use matrix_util::parquet::peek_parquet_field_names;
use matrix_util::traits::IoOps;
use matrix_util::traits::MatOps;
use matrix_util::traits::MatWithNames;

use clap::Parser;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    /// data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `data-beans` command.
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// individual membership files (comma-separated file names). Each
    /// line in each file can specify individual ID or cell and
    /// individual ID pair.
    #[arg(long, short = 'i', value_delimiter(','))]
    indv_files: Vec<Box<str>>,

    /// latent topic assignment files (comma-separated file names)
    /// Each line in each file can specify topic name or cell and
    /// topic name pair.
    #[arg(long, short = 't', value_delimiter(','))]
    topic_assignment_files: Option<Vec<Box<str>>>,

    /// latent topic proportion files (comma-separated file names)
    /// Each file contains a full `cell x topic` matrix.
    #[arg(long, short = 'r', value_delimiter(','))]
    topic_proportion_files: Option<Vec<Box<str>>>,

    /// each line corresponds to (1) individual name and (2) exposure name
    #[arg(long, short = 'e')]
    exposure_assignment_file: Box<str>,

    /// #k-nearest neighbours within each condition
    #[arg(long, short = 'n', default_value_t = 10)]
    knn: usize,

    /// projection dimension to account for confounding factors.
    #[arg(long, short = 'p', default_value_t = 10)]
    proj_dim: usize,

    /// block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of iterations for optimization
    #[arg(long)]
    num_opt_iter: Option<usize>,

    /// hyperparameter a0 in Gamma(a0,b0)
    #[arg(long, default_value_t = 1.0)]
    a0: f32,

    /// hyperparameter b0 in Gamma(a0,b0)
    #[arg(long, default_value_t = 1.0)]
    b0: f32,

    /// output directory
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// preload all the columns data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

/// Run CoCoA differential analysis
pub fn run_cocoa_diff(args: DiffArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let mut data = read_input_data(InputDataArgs {
        data_files: args.data_files,
        indv_files: Some(args.indv_files),
        topic_assignment_files: args.topic_assignment_files,
        topic_proportion_files: args.topic_proportion_files,
        exposure_assignment_file: Some(args.exposure_assignment_file),
        preload_data: args.preload_data,
    })?;

    if data.cell_topic.ncols() > 1 {
        info!("normalizing cell topic proportion");
        data.cell_topic.sum_to_one_rows_inplace();
    }

    info!("Assign cells to pseudobulk samples to calibrate the null distribution");

    data.sparse_data.assign_pseudobulk_individuals(
        args.proj_dim,
        args.block_size,
        &data.cell_to_indv,
    )?;

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

    let cocoa_input = &CocoaCollapseIn {
        n_genes: data.sparse_data.num_rows()?,
        n_topics: data.cell_topic.ncols(),
        knn: args.knn,
        n_opt_iter: args.num_opt_iter,
        hyper_param: Some((args.a0, args.b0)),
        cell_topic_nk: data.cell_topic,
        exposure_assignment: &exposure_assignment,
    };

    info!("Collecting statistics...");
    let cocoa_stat = data.sparse_data.collect_cocoa_stat(cocoa_input)?;

    info!("Optimizing parameters...");
    let parameters = cocoa_stat.estimate_parameters()?;

    info!("Writing down the estimates...");
    let gene_names = data.sparse_data.row_names()?;

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
        Some(&gene_names),
        Some(&indv_exposure_names),
        Some(&topic_names),
        &format!("{}.effect.parquet", args.out),
    )?;

    // these can take too much space...
    // to_parquet(
    //     &shared,
    //     Some(&gene_names),
    //     None,
    // 	Some(&topic_names),
    //     &format!("{}.pb.shared.parquet", args.out),
    // )?;

    // to_parquet(
    //     &residual,
    //     Some(&gene_names),
    //     None,
    // 	Some(&topic_names),
    //     &format!("{}.pb.residual.parquet", args.out),
    // )?;

    info!("Done");
    Ok(())
}
