mod fit_srt_svd;
mod fit_srt_topic;
mod srt_cell_pairs;
mod srt_collapse_pairs;
mod srt_common;
mod srt_random_projection;
mod srt_routines_latent_representation;
mod srt_routines_post_process;
mod srt_routines_pre_process;

use fit_srt_svd::*;
use fit_srt_topic::*;

use clap::{Parser, Subcommand};

/// Proximity-based Interaction Network analysis to dissect Tissue
/// Organizations
///
/// Data files of either `.zarr` or `.h5` format. All the formats in
/// the given list should be identical. We can convert `.mtx` to
/// `.zarr` or `.h5` using `data-beans from-mtx` or similar commands.
///
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// embedding by randomized singular value decomposition
    Svd(SrtSvdArgs),
    /// embedding by fitting a topic model
    Topic(SrtTopicArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Svd(args) => {
            fit_srt_svd(args)?;
        }
        Commands::Topic(args) => {
            fit_srt_topic(args)?;
        }
    }

    Ok(())
}

// #[derive(Parser, Debug, Clone)]
// #[command(version, about, long_about, term_width = 80)]
// ///
// /// Embedding spatially resolved transcriptomic (SRT) data.
// ///
// struct PintoMainArgs {
//     // /// Data files of either `.zarr` or `.h5` format. All the formats
//     // /// in the given list should be identical. We can convert `.mtx`
//     // /// to `.zarr` or `.h5` using `data-beans from-mtx` command.
//     // #[arg(required = true, value_delimiter(','))]
//     // data_files: Vec<Box<str>>,

//     // /// An auxiliary cell coordinate file. Each coordinate file should
//     // /// correspond to each data file. Each line contains barcode, x, y, ...
//     // /// coordinates. We could include more columns.
//     // #[arg(long = "coord", short = 'c', required = true, value_delimiter(','))]
//     // coord_files: Vec<Box<str>>,

//     // /// Indicate the cell coordinate columns in the `coord` files (comma separated)
//     // #[arg(long = "coord_column_indices", value_delimiter(','))]
//     // coord_columns: Option<Vec<usize>>,

//     // /// The columns names in the `coord` files (comma separated)
//     // #[arg(
//     //     long = "coord_column_names",
//     //     value_delimiter(','),
//     //     default_value = "pxl_row_in_fullres,pxl_col_in_fullres"
//     // )]
//     // coord_column_names: Vec<Box<str>>,

//     // /// batch membership files (comma-separated names). Each bach file
//     // /// should correspond to each data file.
//     // #[arg(long, short = 'b', value_delimiter(','))]
//     // batch_files: Option<Vec<Box<str>>>,

//     // /// Random projection dimension to project the data.
//     // #[arg(long, short = 'p', default_value_t = 50)]
//     // proj_dim: usize,

//     // /// Use top `S` components of projection. #samples < `2^S+1`.
//     // #[arg(long, short = 'd', default_value_t = 10)]
//     // sort_dim: usize,

//     // /// #k-nearest neighbours for spectral embedding for spatial coordinates
//     // #[arg(short = 'k', long, default_value_t = 10)]
//     // knn_spatial: usize,

//     // /// #k-nearest neighbours batches
//     // #[arg(long, default_value_t = 3)]
//     // knn_batches: usize,

//     // /// #k-nearest neighbours within each batch
//     // #[arg(long, default_value_t = 10)]
//     // knn_cells: usize,

//     // /// #downsampling columns per each collapsed sample. If None, no
//     // /// downsampling.
//     // #[arg(long, short = 's')]
//     // down_sample: Option<usize>,

//     // // /// optimization iterations
//     // // #[arg(long, default_value_t = 15)]
//     // // iter_opt: usize,
//     // /// Output header
//     // #[arg(long, short, required = true)]
//     // out: Box<str>,

//     // /// Block_size for parallel processing
//     // #[arg(long, default_value_t = 100)]
//     // block_size: usize,

//     // /// number of latent topics
//     // #[arg(short = 't', long, default_value_t = 10)]
//     // n_latent_topics: usize,

//     // /// targeted number of row feature modules (to speed up)
//     // #[arg(short = 'r', long, default_value_t = 512)]
//     // n_row_modules: usize,

//     // /// encoder layers
//     // #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,128,128])]
//     // encoder_layers: Vec<usize>,

//     // /// intensity levels for frequency embedding
//     // #[arg(long, default_value_t = 20)]
//     // vocab_size: usize,

//     // /// intensity embedding dimension
//     // #[arg(long, default_value_t = 10)]
//     // vocab_emb: usize,

//     // /// # training epochs
//     // #[arg(long, short = 'i', default_value_t = 1000)]
//     // epochs: usize,

//     // /// Minibatch size
//     // #[arg(long, default_value_t = 100)]
//     // minibatch_size: usize,

//     // #[arg(long, default_value_t = 1e-3)]
//     // learning_rate: f32,

//     // /// candle device
//     // #[arg(long, value_enum, default_value = "cpu")]
//     // device: ComputeDevice,

//     // /// preload all the columns data
//     // #[arg(long, default_value_t = false)]
//     // preload_data: bool,

//     // /// verbosity
//     // #[arg(long, short)]
//     // verbose: bool,
// }
