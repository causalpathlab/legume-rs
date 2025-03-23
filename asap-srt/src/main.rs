use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use clap::{Parser, ValueEnum};
use log::info;
use matrix_util::knn_match::*;

#[derive(Parser, Debug, Clone)]
#[command(name = "SRT network", version, about, long_about, term_width = 80)]
///
/// Generate network data from spatially-resolved transcriptomic (SRT) data.
///
struct SRTNetworkArgs {
    /// Data file of either `.zarr` or `.h5` format. We can convert
    /// `.mtx` to `.zarr` or `.h5` using `asap-data build` command.
    #[arg(required = true)]
    data_file: Box<str>,

    /// An auxiliary cell coordinate file. Each
    /// coordinate file should correspond to each data file. Each line
    /// of each file contains (1) cell barcode (column name) (2) x (3)
    /// y coordinates. We could include more columns.
    #[arg(long = "coord", short = 'p')]
    coord_file: Box<str>,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    // /// #downsampling columns per each collapsed sample. If None, no
    // /// downsampling.
    // #[arg(long, short = 's')]
    // down_sample: Option<usize>,
    /// optimization iterations
    #[arg(long, default_value_t = 10)]
    iter_opt: usize,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

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

struct SRT {}

fn build_coordinate_map(args: &SRTNetworkArgs) -> anyhow::Result<()> {
    // args.coord_file;

    // let (cell_names, coords) = read_coordinate_file(&args.coord_file)?;
    // Ok((cell_names, coords))
    todo!("");
}

struct CoordMap {}

// use HashMap::<

fn generate_srt_triplets(args: &SRTNetworkArgs) -> anyhow::Result<()> {
    todo!("");
}

fn main() -> anyhow::Result<()> {
    let args: SRTNetworkArgs = SRTNetworkArgs::parse();

    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }

    info!("creating a HNSW map for fast look-up ...");

    // let columns = batch_cells
    //     .iter()
    //     .map(|&c| feature_matrix.column(c))
    //     .collect::<Vec<_>>();

    // ColumnDict::<usize>::from_dvector_views(columns, batch_cells.clone())

    // 2. think about a visitor pattern
    //
    // for each pair (i -> j)
    //
    // x(:,i), neighbours of i
    // x(:,j), neighbours of j
    //

    info!("");

    Ok(())
}
