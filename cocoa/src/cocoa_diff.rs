use crate::cocoa_collapse::*;
use crate::cocoa_common::*;

pub use clap::Parser;
pub use log::info;
pub use matrix_util::common_io::{extension, read_lines, read_lines_of_words};
use matrix_util::traits::IoOps;
pub use std::sync::Arc;

use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `asap-data build` command.
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Sample membership files (comma-separated names). Each bach
    /// file should match with each data file.
    #[arg(long, short, value_delimiter(','))]
    sample_files: Vec<Box<str>>,

    /// Each line corresponds to (1) sample name and (2) exposure name
    #[arg(long, short)]
    exposure_assignment_file: Box<str>,

    /// Latent topic assignment file
    #[arg(long, short)]
    topic_assignment_file: Option<Box<str>>,

    /// #k-nearest neighbours within each condition
    #[arg(long, short = 'n', default_value_t = 10)]
    knn: usize,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 30)]
    proj_dim: usize,

    /// Block_size for parallel processing
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

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

/// Run CoCoA differential analysis
pub fn run_cocoa_diff(args: DiffArgs) -> anyhow::Result<()> {
    let data = read_input_data(args.clone())?;

    let sample_to_cells = (0..data.sparse_data.num_batches())
        .map(|b| data.sparse_data.batch_to_columns(b).unwrap().clone())
        .collect::<Vec<_>>();

    let n_topics = data.cell_topic.ncols();

    let cocoa_input = &CocoaCollapseIn {
        n_genes: data.sparse_data.num_rows()?,
        n_samples: data.sparse_data.num_batches(),
        n_topics,
        knn: args.knn,
        n_opt_iter: args.num_opt_iter,
        hyper_param: Some((args.a0, args.b0)),
        cell_topic_nk: data.cell_topic,
        sample_to_cells: &sample_to_cells,
        sample_to_exposure: &data.sample_to_exposure,
    };

    let cocoa_stat = data.sparse_data.collect_stat(cocoa_input)?;

    let parameters = cocoa_stat.estimate_parameters()?;

    let out_dir = args.out.to_string() + "/";

    // for k in 0..n_topics {
    //     let out = cocoa_stat.optimize_each_topic(k)?;
    // }
    Ok(())
}

struct DiffData {
    sparse_data: SparseIoVec,
    sample_to_exposure: Vec<usize>,
    cell_topic: Mat,
}

fn read_input_data(args: DiffArgs) -> anyhow::Result<DiffData> {
    use asap_embed::asap_collapse_data::CollapsingOps;
    use asap_embed::asap_random_projection::RandProjOps;

    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    if args.sample_files.len() != args.data_files.len() {
        return Err(anyhow::anyhow!("# sample files != # of data files"));
    }

    let (exposure, _) = read_lines_of_words(&args.exposure_assignment_file, 0)?;

    let sample_to_exposure: HashMap<_, _> = exposure
        .into_iter()
        .filter(|w| w.len() > 1)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();

    let exposure_id: HashMap<_, usize> = sample_to_exposure
        .values()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(id, val)| (val, id))
        .collect();

    let n_exposure = exposure_id.len();
    info!("{} exposure groups", n_exposure);

    let mut sparse_data = SparseIoVec::new();
    let mut cell_to_sample = vec![];

    for (this_data_file, sample_file) in args.data_files.into_iter().zip(args.sample_files) {
        info!("Importing: {}, {}", this_data_file, sample_file);

        match extension(&this_data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", this_data_file)),
        };

        let this_data = open_sparse_matrix(&this_data_file, &backend)?;
        let this_data_samples = read_lines(&sample_file)?;

        if this_data_samples.len() != sparse_data.num_columns().unwrap_or(0) {
            return Err(anyhow::anyhow!(
                "{} and {} don't match",
                sample_file,
                this_data_file,
            ));
        }

        cell_to_sample.extend(this_data_samples);
        sparse_data.push(Arc::from(this_data))?;
    }
    info!("Total {} data sets combined", sparse_data.len());

    let proj_out = sparse_data.project_columns(args.proj_dim, Some(args.block_size))?;
    let proj_kn = &proj_out.proj;

    sparse_data.register_batches(proj_kn, &cell_to_sample)?;

    let samples = sparse_data
        .batch_names()
        .ok_or(anyhow::anyhow!("empty sample name in sparse data"))?;

    let sample_to_exposure: Vec<usize> = samples
        .iter()
        .map(|s| {
            if let Some(exposure) = sample_to_exposure.get(s) {
                exposure_id[exposure]
            } else {
                n_exposure
            }
        })
        .collect();

    info!(
        "Found {} samples with exposure assignments",
        sample_to_exposure.len()
    );

    let cell_topic = match args.topic_assignment_file.as_ref() {
        Some(file) => {
            info!("cell topic data {}", file);
            Mat::from_tsv(file, None)?
        }
        None => {
            info!("ignoring cell types");
            Mat::from_element(sparse_data.num_columns()?, 1, 1.0)
        }
    };

    if cell_topic.nrows() != sparse_data.num_columns()? {
        return Err(anyhow::anyhow!(
            "topic assignment matrix should be a tab-separated file with {} rows",
            sparse_data.num_columns()?
        ));
    }

    Ok(DiffData {
        sparse_data,
        sample_to_exposure,
        cell_topic,
    })
}
