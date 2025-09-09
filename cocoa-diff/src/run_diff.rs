use crate::collapse_data::*;
use crate::common::*;
use crate::randomly_partition_data::*;

pub use clap::Parser;

use matrix_param::io::*;
use matrix_util::common_io::basename;
use matrix_util::common_io::ReadLinesOut;
use matrix_util::common_io::{extension, read_lines_of_words_delim};
use matrix_util::dmatrix_util::concatenate_vertical;
use matrix_util::parquet::peek_parquet_field_names;
use matrix_util::traits::IoOps;
use matrix_util::traits::MatWithNames;
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

struct ArgInputData {
    sparse_data: SparseIoVec,
    cell_to_indv: Vec<Box<str>>,
    cell_topic: Mat,
    sorted_topic_names: Vec<Box<str>>,
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

    let exposure =
        read_lines_of_words_delim(&args.exposure_assignment_file, &['\t', ',', ' '], -1)?.lines;

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

    let data_files = args.data_files;
    let indv_files = args.indv_files;
    let topic_assignment_files = match args.topic_assignment_files {
        Some(vec) => vec.into_iter().map(Some).collect(),
        None => vec![None; data_files.len()],
    };
    let topic_proportion_files = match args.topic_proportion_files {
        Some(vec) => vec.into_iter().map(Some).collect(),
        None => vec![None; data_files.len()],
    };

    info!("Looking into the topic files...");

    let mut topic_names = HashSet::<Box<str>>::new();

    for (a_file, p_file) in topic_assignment_files
        .iter()
        .zip(topic_proportion_files.iter())
    {
        if let Some(a_file) = a_file {
            let _names = read_lines_of_words_delim(a_file.as_ref(), &['\t', ',', ' '], -1)?
                .lines
                .iter()
                .filter_map(|line| (line.len() > 1).then(|| line[1].clone()))
                .collect::<HashSet<_>>();
            for _x in _names {
                topic_names.insert(_x);
            }
        }

        if let Some(p_file) = p_file {
            let ext = extension(p_file)?;
            if ext.as_ref() == "parquet" {
                let _names = peek_parquet_field_names(p_file)?;
                for _x in _names {
                    topic_names.insert(_x);
                }
            }
        }
    }

    let mut sorted_topic_names: Vec<Box<str>> = topic_names.into_iter().collect();
    sorted_topic_names.sort();

    let topic_names: HashMap<Box<str>, usize> = sorted_topic_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();

    info!("Found {} topics", topic_names.len());

    let mut topic_vec = vec![];

    let mut sparse_data = SparseIoVec::new();
    let mut cell_to_indv = vec![];

    for (f, this_data_file) in data_files.iter().enumerate() {
        let this_indv_file = indv_files[f].clone();

        info!("Importing: {}, {}", this_data_file, this_indv_file);

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

        let this_indv = read_lines_of_words_delim(&this_indv_file, &['\t', ',', ' '], -1)?.lines;

        if this_indv.is_empty() {
            return Err(anyhow::anyhow!(
                "empty assignment file: {}",
                &this_indv_file
            ));
        }

        let this_indv: Vec<Box<str>> = match this_indv[0].len() {
            1 => this_indv.into_iter().map(|w| w[0].clone()).collect(),
            _ => {
                let cell_to_indv: HashMap<_, _> = this_indv
                    .into_iter()
                    .filter_map(|w| (w.len() > 1).then(|| (w[0].clone(), w[1].clone())))
                    .collect();

                let missing = "NA".to_string().into_boxed_str();

                this_data
                    .column_names()?
                    .iter()
                    .map(|c| {
                        cell_to_indv
                            .get(c)
                            .cloned()
                            .unwrap_or_else(|| missing.clone())
                    })
                    .collect()
            }
        };

        if this_indv.len() != ndata {
            return Err(anyhow::anyhow!(
                "{} and {} don't match",
                this_indv_file,
                this_data_file,
            ));
        }

        let topic_a_file = &topic_assignment_files[f];
        let topic_p_file = &topic_proportion_files[f];

        let mut this_topic = if topic_names.is_empty() {
            Mat::from_element(ndata, 1, 1.0)
        } else {
            Mat::zeros(ndata, topic_names.len())
        };

        let cells_to_rows = this_data
            .column_names()?
            .into_iter()
            .enumerate()
            .map(|(i, x)| (x, i))
            .collect::<HashMap<_, _>>();

        if let Some(a_file) = topic_a_file {
            info!("importing topic information from {}", a_file);

            let ReadLinesOut { lines, header: _ } =
                read_lines_of_words_delim(a_file.as_ref(), &['\t', ',', ' '], -1)?;

            for words in lines {
                if words.len() > 1 {
                    if let Some(&r) = cells_to_rows.get(&words[0]) {
                        if let Some(&c) = topic_names.get(&words[1]) {
                            this_topic[(r, c)] = 1.0;
                        }
                    }
                }
            }
        }

        if let Some(p_file) = topic_p_file {
            let ext = extension(p_file)?;
            if ext.as_ref() == "parquet" {
                info!("importing topic information from {}", p_file);

                let MatWithNames { mat, rows, cols } = Mat::from_parquet(&p_file)?;

                for j in 0..mat.ncols() {
                    if let Some(&c) = topic_names.get(&cols[j]) {
                        for i in 0..mat.nrows() {
                            if let Some(&r) = cells_to_rows.get(&rows[i]) {
                                this_topic[(r, c)] = mat[(i, j)];
                            }
                        }
                    }
                }
            }
        }

        let data_name = basename(&this_data_file)?;
        sparse_data.push(Arc::from(this_data), Some(data_name))?;
        cell_to_indv.extend(this_indv);
        topic_vec.push(this_topic);
    }

    info!("Total {} data sets combined", sparse_data.len());

    let cell_topic = concatenate_vertical(topic_vec.as_slice())?;

    Ok(ArgInputData {
        sparse_data,
        cell_to_indv,
        cell_topic,
        sorted_topic_names,
        indv_to_exposure,
        exposure_id,
    })
}
