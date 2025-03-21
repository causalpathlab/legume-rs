use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use clap::{Parser, ValueEnum};

fn main() {
    println!("Hello, world!");
}

#[derive(Parser, Debug, Clone)]
#[command(name = "SRT network", version, about, long_about, term_width = 80)]
/// 
/// Generate network data from spatially-resolved transcriptomic (SRT) data.
/// 
struct SRTNetworkArgs {


    // 

    // /// Data files of either `.zarr` or `.h5` format. All the formats
    // /// in the given list should be identical. We can convert `.mtx`
    // /// to `.zarr` or `.h5` using `asap-data build` command.
    // #[arg(required = true)]
    // data_files: Vec<Box<str>>,

    // /// Random projection dimension to project the data.
    // #[arg(long, short, default_value_t = 30)]
    // proj_dim: usize,

    // /// Output header
    // #[arg(long, short, required = true)]
    // out: Box<str>,

    // /// Use top `S` components of projection. #samples < `2^S+1`.
    // #[arg(long, short = 'd', default_value_t = 10)]
    // sort_dim: usize,

    // /// Batch membership files (comma-separated names). Each bach file
    // /// should correspond to each data file.
    // #[arg(long, short, value_delimiter(','))]
    // batch_files: Option<Vec<Box<str>>>,

    // /// Reference batch name (comma-separated names)
    // #[arg(short = 'r', long, value_delimiter(','))]
    // reference_batch: Option<Vec<Box<str>>>,

    // /// #k-nearest neighbours within each batch
    // #[arg(long, short = 'n', default_value_t = 10)]
    // knn: usize,

    // /// #downsampling columns per each collapsed sample. If None, no
    // /// downsampling.
    // #[arg(long, short = 's')]
    // down_sample: Option<usize>,

    // /// optimization iterations
    // #[arg(long, default_value_t = 100)]
    // iter_opt: usize,

    // /// Block_size for parallel processing
    // #[arg(long, default_value_t = 100)]
    // block_size: usize,

    // /// Number of latent topics
    // #[arg(short = 'k', long, default_value_t = 10)]
    // latent_topics: usize,

    // /// Encoder layers
    // #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![32,128,1024,128])]
    // encoder_layers: Vec<usize>,

    // /// # pre-training epochs
    // #[arg(long, default_value_t = 100)]
    // pretrain_epochs: usize,

    // /// # training epochs
    // #[arg(long, short = 'i', default_value_t = 1000)]
    // epochs: usize,

    // /// Minibatch size
    // #[arg(long, default_value_t = 100)]
    // minibatch_size: usize,

    // #[arg(long, default_value_t = 1e-3)]
    // learning_rate: f32,

    // /// Candle device
    // #[arg(long, value_enum, default_value = "cpu")]
    // device: ComputeDevice,

    // /// Decodeer model: topic (softmax) model, poisson MF, or nystrom
    // /// projection
    // #[arg(long, short = 'm', value_enum, default_value = "topic")]
    // decoder_model: DecoderModel,

    // /// Save intermediate projection results
    // #[arg(long)]
    // save_intermediate: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}
