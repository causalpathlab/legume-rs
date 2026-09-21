//! `peak-to-gene` CLI args and orchestrator.
//!
//! Association via `graph-embedding-util` is not wired yet. The rSVD / SuSiE /
//! knockoff / TMLE path has been removed.

use crate::common::*;

#[derive(Args, Debug)]
pub struct PeakToGeneArgs {
    /* Input — kept as the forward CLI surface for the ge-util pipeline */
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "RNA sparse matrices (zarr/h5), comma-separated"
    )]
    rna_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "ATAC sparse matrices (zarr/h5), comma-separated"
    )]
    atac_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file in RNA-then-ATAC order"
    )]
    batch_files: Option<Vec<Box<str>>>,

    /// Shared cell QC (on by default; `--no-qc` to disable).
    #[command(flatten)]
    qc: data_beans::qc_lib::QcArgs,

    #[arg(
        long,
        default_value_t = 500000,
        help = "Cis-window in bp around each gene TSS (peak midpoint distance)"
    )]
    cis_window: i64,

    #[arg(
        long,
        help = "Gene coordinates TSV (gene<TAB>chr<TAB>tss).\n\
                From sim-link gene_coords.tsv.gz"
    )]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF annotation for gene TSS. Alternative to --gene-coords"
    )]
    gff_file: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix (E2G-like peaks / clusters / peak_gene parquet when wired)"
    )]
    out: Box<str>,
}

pub fn run_peak_to_gene(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    let _ = (
        &args.rna_files,
        &args.atac_files,
        &args.batch_files,
        &args.qc,
        args.cis_window,
        &args.gene_coords,
        &args.gff_file,
    );
    anyhow::bail!(
        "peak-to-gene via graph-embedding-util is not wired yet \
         (rSVD / SuSiE / knockoff / TMLE removed). Stages: abc_map → embed_ge → \
         cluster/refine → parquet_out. See chickpea/todo.md and \
         docs/superpowers/plans/2026-09-21-chickpea-ge-util-p2g.md \
         (out prefix would be {})",
        args.out
    );
}
