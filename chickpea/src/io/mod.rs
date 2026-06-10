pub mod fragments;
pub mod multiome;

use clap::Args;
use data_beans::sparse_io::SparseIo;
use data_beans::sparse_io::SparseIoBackend;
use log::info;
use matrix_util::common_io::mkdir_parent;
use rustc_hash::FxHashSet;

#[derive(Args, Debug)]
pub struct FromMultiomeArgs {
    /// Input 10x Multiome HDF5 file (filtered_feature_bc_matrix.h5)
    pub h5_file: Box<str>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix for backend files",
        long_help = "Output prefix for backend files.\n\
                     Produces {out}.rna.zarr and {out}.atac.zarr"
    )]
    pub out: Box<str>,

    #[arg(
        long,
        help = "Optional ATAC fragments file (e.g. atac_fragments.tsv.gz)",
        long_help = "Optional ATAC fragments file (e.g. atac_fragments.tsv.gz).\n\
                     When provided, writes {out}.fragments.tsv.gz filtered to matched barcodes."
    )]
    pub fragments: Option<Box<str>>,
}

pub fn run_from_multiome(args: &FromMultiomeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let backend = SparseIoBackend::Zarr;

    let backends = multiome::convert_multiome_h5(&args.h5_file, &args.out, &backend)?;

    info!(
        "RNA backend: {} rows x {} cols",
        backends.rna.num_rows().unwrap_or(0),
        backends.rna.num_columns().unwrap_or(0),
    );
    info!(
        "ATAC backend: {} rows x {} cols",
        backends.atac.num_rows().unwrap_or(0),
        backends.atac.num_columns().unwrap_or(0),
    );

    // Optionally filter fragments file to ATAC barcodes
    if let Some(ref frag_file) = args.fragments {
        let atac_barcodes: FxHashSet<Box<str>> =
            backends.atac.column_names()?.into_iter().collect();

        let frag_output = format!("{}.fragments.tsv.gz", args.out);
        fragments::filter_fragments(frag_file, &frag_output, Some(&atac_barcodes))?;
    }

    info!("Done");
    Ok(())
}
