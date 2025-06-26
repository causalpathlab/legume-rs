use crate::embed_common::*;
use crate::routines_latent_representation::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;

#[derive(Args, Debug)]
pub struct DeconvArgs {
    /// single-cell data files (`zarr` or `h5`)
    #[arg(required = true)]
    sc_data_files: Vec<Box<str>>,

    /// bulk data files (`.parquet`, `.tsv.gz`, `.csv.gz`)
    /// where the first column corresponds to gene names
    #[arg(short, long, required = true)]
    bulk_data_files: Vec<Box<str>>,

    /// random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 15)]
    iter_opt: usize,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_deconv(args: &DeconvArgs) -> anyhow::Result<()> {
    unimplemented!("take both bulk and single cell data");
    Ok(())
}
