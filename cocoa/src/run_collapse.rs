use crate::common::*;
use crate::input::*;

use clap::Parser;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::io::ParamIo;
use matrix_param::traits::TwoStatParam;
use matrix_util::dmatrix_util::concatenate_horizontal;
use matrix_util::dmatrix_util::subset_rows;
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug, Clone)]
pub struct CollapseArgs {
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
        help = "Individual membership file names (comma-separated).",
        long_help = "Individual membership files (comma-separated file names). \n\
		     Each line in each file can specify: \n\
		     * just  individual ID or\n\
		     * (1) Cell and (2) individual ID pair."
    )]
    indv_files: Option<Vec<Box<str>>>,

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
        long = "topic-proportion-files",
        value_delimiter = ',',
        help = "Latent topic proportion file names (comma-separated).",
        long_help = "Latent topic proportion files (comma-separated file names). \n\
		     Each file contains a full `cell x topic` matrix."
    )]
    topic_proportion_files: Option<Vec<Box<str>>>,

    #[arg(
        long = "topic-proportion-value",
        default_value = "logit",
        help = "Is topic proportion matrix of probability?",
        long_help = "Specify if the topic proportion matrix is of probability type. \n\
		     Default is `logit`-valued."
    )]
    topic_proportion_value: TopicValue,

    #[arg(
        long = "block-size",
        default_value_t = 100,
        help = "Block size for parallel processing.",
        long_help = "Block size for parallel processing."
    )]
    block_size: usize,

    #[arg(
        long = "a0",
        default_value_t = 1.0,
        help = "Hyperparameter a0 in Gamma(a0, b0).",
        long_help = "Hyperparameter a0 in Gamma(a0, b0)."
    )]
    a0: f32,

    #[arg(
        long = "b0",
        default_value_t = 1.0,
        help = "Hyperparameter b0 in Gamma(a0, b0).",
        long_help = "Hyperparameter b0 in Gamma(a0, b0)."
    )]
    b0: f32,

    #[arg(
        short,
        long = "out",
        required = true,
        help = "Output file name.",
        long_help = "Output file name."
    )]
    out: Box<str>,

    #[arg(
        long = "preload-data",
        default_value_t = false,
        help = "Preload all the columns data.",
        long_help = "Preload all the columns data."
    )]
    preload_data: bool,

    #[arg(
        short,
        long = "verbose",
        help = "Verbosity.",
        long_help = "Increase output verbosity."
    )]
    verbose: bool,
}

pub fn run_collapse(args: CollapseArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // read data without `exposure`
    let data = read_input_data(InputDataArgs {
        data_files: args.data_files.clone(),
        indv_files: args.indv_files.clone(),
        topic_assignment_files: args.topic_assignment_files,
        topic_proportion_files: args.topic_proportion_files,
        exposure_assignment_file: None,
        preload_data: args.preload_data,
        topic_value: args.topic_proportion_value,
    })?;

    let cell_topic = &data.cell_topic;
    let ngenes = data.sparse_data.num_rows()?;
    let ntopics = cell_topic.ncols();

    // break down into individual level collapsing
    if let Some(indv_names) = data.sparse_data.group_keys() {
        let nindv = indv_names.len();

        let mut topic_indv_stat = TopicIndvStat {
            count_gene_topic_indv: Mat::zeros(ngenes, ntopics * nindv),
            count_topic_indv: DVec::zeros(ntopics * nindv),
        };

        data.sparse_data.visit_columns_by_group(
            &collect_topic_indv_stat_visitor,
            &cell_topic,
            &mut topic_indv_stat,
        )?;

        let mut gamma = GammaMatrix::new((ngenes, ntopics), args.a0, args.b0);
        gamma.update_stat(
            &topic_indv_stat.count_gene_topic_indv,
            &concatenate_horizontal(&vec![topic_indv_stat.count_topic_indv; ngenes])?.transpose(),
        );
        gamma.calibrate();

        let gene_names = data.sparse_data.row_names()?;
        let out_file = format!("{}.parquet", args.out.replace(".parquet", ""));

        // Create column names with {indv}_{topic} format
        let indv_topic_names: Vec<Box<str>> = indv_names
            .iter()
            .flat_map(|indv| {
                (0..ntopics)
                    .map(|topic| format!("{}_{}", indv, topic).into_boxed_str())
                    .collect::<Vec<_>>()
            })
            .collect();

        gamma.to_parquet(Some(&gene_names), Some(&indv_topic_names), &out_file)?;
    }

    {
        let mut topic_stat = TopicStat {
            count_gene_topic: Mat::zeros(ngenes, ntopics),
            count_topic: DVec::zeros(ntopics),
        };

        data.sparse_data.visit_columns_by_block(
            &collect_topic_stat_visitor,
            &cell_topic,
            &mut topic_stat,
            Some(args.block_size),
        )?;

        let mut gamma = GammaMatrix::new((ngenes, ntopics), args.a0, args.b0);
        gamma.update_stat(
            &topic_stat.count_gene_topic,
            &concatenate_horizontal(&vec![topic_stat.count_topic; ngenes])?.transpose(),
        );
        gamma.calibrate();

        let gene_names = data.sparse_data.row_names()?;
        let out_file = format!("{}.parquet", args.out.replace(".parquet", ""));
        gamma.to_parquet(Some(&gene_names), None, &out_file)?;
    }

    Ok(())
}

//////////////////////////////////////////////////////
// collapsing within each topic and individual pair //
//////////////////////////////////////////////////////

struct TopicIndvStat {
    count_gene_topic_indv: Mat,
    count_topic_indv: DVec,
}

fn collect_topic_indv_stat_visitor(
    indv_id: usize,  // individual id
    cells: &[usize], // cells within this individual
    data: &SparseIoVec,
    cell_topic_nk: &Mat,
    arc_stat: Arc<Mutex<&mut TopicIndvStat>>,
) -> anyhow::Result<()> {
    let y_gn = data.read_columns_csc(cells.iter().cloned())?;

    let z_nk = subset_rows(cell_topic_nk, cells.iter().cloned())?;
    let z_k = z_nk.row_sum().transpose();
    let y_gk = y_gn * z_nk;

    let kk = cell_topic_nk.ncols();

    let mut stat = arc_stat.lock().expect("lock");

    let lb = kk * indv_id;
    let ub = kk * (indv_id + 1);
    let mut y_gk_target = stat.count_gene_topic_indv.columns_range_mut(lb..ub);
    y_gk_target += &y_gk;

    for (j, p) in (lb..ub).enumerate() {
        stat.count_topic_indv[p] += z_k[j];
    }

    Ok(())
}

//////////////////////////////////
// collapsing within each topic //
//////////////////////////////////

struct TopicStat {
    count_gene_topic: Mat,
    count_topic: DVec,
}

fn collect_topic_stat_visitor(
    bound: (usize, usize),
    data: &SparseIoVec,
    cell_topic_nk: &Mat,
    arc_stat: Arc<Mutex<&mut TopicStat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;

    let y_gn = data.read_columns_csc(lb..ub)?;
    let z_nk = subset_rows(cell_topic_nk, lb..ub)?;

    let z_k = z_nk.row_sum().transpose();
    let y_gk = y_gn * z_nk;

    let mut stat = arc_stat.lock().expect("lock");
    stat.count_gene_topic += y_gk;
    stat.count_topic += z_k;
    Ok(())
}
