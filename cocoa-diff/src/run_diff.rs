use crate::collapse_data::*;
use crate::common::*;
use crate::randomly_partition_data::*;

pub use clap::Parser;

use matrix_param::io::*;
use matrix_util::common_io::basename;
pub use matrix_util::common_io::{extension, read_lines, read_lines_of_words};
use matrix_util::dmatrix_util::concatenate_vertical;
use matrix_util::traits::IoOps;
pub use std::sync::Arc;

use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    /// data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `asap-data build` command.
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// individual membership files (comma-separated file names). Each
    /// individual membership file should match with each data file.
    #[arg(long, short, value_delimiter(','))]
    indv_files: Vec<Box<str>>,

    /// latent topic assignment files (comma-separated file names)
    /// Each topic file should match with each data file or None.
    #[arg(long, short, value_delimiter(','))]
    topic_assignment_files: Option<Vec<Box<str>>>,

    /// each line corresponds to (1) individual name and (2) exposure name
    #[arg(long, short)]
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

    let mut data = parse_arg_input_data(args.clone())?;

    if data.cell_topic.ncols() > 1 {
        info!("normalizing cell topic proportion");
        data.cell_topic
            .row_iter_mut()
            .for_each(|mut r| r.unscale_mut(r.sum()));
    }

    info!("Assign cells to pseudobulk samples to calibrate the null distribution");

    data.sparse_data.assign_pseudobulk_individuals(
        args.proj_dim,
        args.block_size,
        &data.cell_to_indv,
    )?;

    let indv_names = data.sparse_data.batch_names().unwrap();
    let indv_to_exposure = data.indv_to_exposure;
    let exposure_id = data.exposure_id;
    let n_exposure = exposure_id.len();

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
    let cocoa_stat = data.sparse_data.collect_stat(cocoa_input)?;

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
        Some(&indv_names),
        &format!("{}.effect.parquet", args.out),
    )?;

    to_parquet(
        &shared,
        Some(&gene_names),
        None,
        &format!("{}.pb.shared.parquet", args.out),
    )?;

    to_parquet(
        &residual,
        Some(&gene_names),
        None,
        &format!("{}.pb.residual.parquet", args.out),
    )?;

    info!("Done");
    Ok(())
}

struct ArgInputData {
    sparse_data: SparseIoVec,
    cell_to_indv: Vec<Box<str>>,
    cell_topic: Mat,
    indv_to_exposure: HashMap<Box<str>, Box<str>>,
    exposure_id: HashMap<Box<str>, usize>,
}

fn parse_arg_input_data(args: DiffArgs) -> anyhow::Result<ArgInputData> {
    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    if args.indv_files.len() != args.data_files.len() {
        return Err(anyhow::anyhow!("# sample files != # of data files"));
    }

    let (exposure, _) = read_lines_of_words(&args.exposure_assignment_file, -1)?;

    let indv_to_exposure = exposure
        .into_iter()
        .filter(|w| w.len() > 1)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect::<HashMap<_, _>>();

    let exposure_id: HashMap<_, usize> = indv_to_exposure
        .values()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(id, val)| (val, id))
        .collect();

    let n_exposure = exposure_id.len();
    let mut exposure_name = vec![String::from("").into_boxed_str(); n_exposure];
    for (x, &id) in exposure_id.iter() {
        if id < n_exposure {
            exposure_name[id] = x.clone();
        }
    }
    info!("{} exposure groups", n_exposure);

    let mut sparse_data = SparseIoVec::new();
    let mut cell_to_indv = vec![];
    let mut topic_vec = vec![];

    let data_files = args.data_files;
    let indv_files = args.indv_files;
    let topic_files = match args.topic_assignment_files {
        Some(vec) => vec.into_iter().map(Some).collect(),
        None => vec![None; data_files.len()],
    };

    for ((this_data_file, indv_file), topic_file) in
        data_files.into_iter().zip(indv_files).zip(topic_files)
    {
        info!("Importing: {}, {}", this_data_file, indv_file);

        match extension(&this_data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", this_data_file)),
        };

        let mut this_data = open_sparse_matrix(&this_data_file, &backend)?;

        if args.preload_data {
            this_data.preload_columns()?;
        }

        let ndata = this_data.num_columns().unwrap_or(0);

        let this_indv = read_lines(&indv_file)?;
        let this_topic = match topic_file.as_ref() {
            Some(_file) => Mat::from_tsv(_file, None)?,
            None => Mat::from_element(ndata, 1, 1.0),
        };

        if this_indv.len() != ndata || this_topic.nrows() != ndata {
            return Err(anyhow::anyhow!(
                "{} and {} don't match",
                indv_file,
                this_data_file,
            ));
        }

        cell_to_indv.extend(this_indv);
        let data_name = basename(&this_data_file)?;
        sparse_data.push(Arc::from(this_data), Some(data_name))?;
        topic_vec.push(this_topic);
    }
    info!("Total {} data sets combined", sparse_data.len());

    let cell_topic = concatenate_vertical(topic_vec.as_slice())?;

    Ok(ArgInputData {
        sparse_data,
        cell_to_indv,
        cell_topic,
        indv_to_exposure,
        exposure_id,
    })
}
