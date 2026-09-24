//! `peak-to-gene` CLI arguments and the run.

use crate::common::*;
use crate::p2g::cis::AbcKernel;
use crate::p2g::input::load_gene_coords_tsv;
use crate::p2g::two_track::{TwoTrackConfig, TwoTrackInput};
use crate::p2g::workflow::{run_links, LinkConfig};
use data_beans::sparse_io::open_sparse_matrix_by_path;
use genomic_data::coordinates::load_gene_tss;
use rustc_hash::FxHashSet;

#[derive(Args, Debug)]
#[command(group = clap::ArgGroup::new("positions")
    .required(true)
    .args(["gene_coords", "gff_file"]))]
pub struct PeakToGeneArgs {
    /* Input */
    #[arg(long, help = "RNA gene counts (zarr/h5)")]
    rna: Box<str>,

    #[arg(long, help = "ATAC peak counts (zarr/h5) for the same barcodes")]
    atac: Box<str>,

    #[arg(long, help = "Batch labels, one per barcode")]
    batch: Option<Box<str>>,

    /// Shared cell QC, driven by the RNA counts (on by default; `--no-qc` to
    /// disable). Failed cells still inform the embedding, but are left out of
    /// clustering, accessibility rates and every cell table.
    #[command(flatten)]
    qc: data_beans::qc_lib::QcArgs,

    /* Gene positions (one required) */
    #[arg(long, help = "Gene TSS table: gene, chr, tss, with a header")]
    gene_coords: Option<Box<str>>,

    #[arg(long, help = "GFF/GTF annotation for gene TSS")]
    gff_file: Option<Box<str>>,

    /* Cis candidates and the ABC contact prior */
    #[arg(
        long,
        default_value_t = 500_000,
        help = "Max peak-midpoint distance (bp) to a gene's TSS"
    )]
    cis_window: i64,

    #[arg(
        long,
        default_value_t = 200,
        help = "Max candidate peaks per gene, nearest first"
    )]
    max_cis: usize,

    #[arg(long, default_value_t = 1.0, help = "Contact exponent γ in (d + c)^-γ")]
    contact_gamma: f32,

    #[arg(long, default_value_t = 5000.0, help = "Contact pseudocount c (bp)")]
    contact_pseudocount: f32,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Weight of the gene-to-cis-peak alignment, per unit",
        long_help = "Weight of the gene-to-cis-peak alignment, per unit.\n\
                     Pulls each gene's profile across pseudobulks toward the\n\
                     predicted accessibility of its cis peaks, so genes and\n\
                     peaks share one feature space. 0 turns it off."
    )]
    align_weight: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Share of each gene's score taken from its cis peaks, in [0, 1)",
        long_help = "Share of each gene's score taken from its cis peaks, in [0, 1).\n\
                     A gene with cis pairs scores (1 − mix) of its own row plus\n\
                     mix of its cis peaks' predicted accessibility, and the\n\
                     written gene rows carry the same mixture. 0 keeps the gene\n\
                     likelihood exact."
    )]
    mix: f32,

    /* Multiome embedding (bge multiome recipe) */
    #[arg(long, default_value_t = 128, help = "Embedding dimension")]
    embedding_dim: usize,

    #[arg(long, default_value_t = 1000, help = "Embedding epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 3, help = "Pseudobulk levels")]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Binary sort dimension; about 2^sort_dim finest pseudobulks"
    )]
    sort_dim: usize,

    #[arg(long, default_value_t = 50, help = "Random projection dimension")]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "L2 weight decay on the gene and peak embeddings",
        long_help = "L2 weight decay on the gene and peak embeddings.\n\
                     Every feature row a phase-1 step touches — each modality's\n\
                     module and gene rows — shrinks by 1 − lr·wd before its\n\
                     update, so directions the data never pins down decay.\n\
                     The pseudobulk rows and biases never decay. 0 turns it off."
    )]
    weight_decay: f64,

    #[arg(long, default_value_t = 1024, help = "RNA gene modules")]
    feature_modules: usize,

    #[arg(
        long,
        default_value_t = 10_000,
        help = "ATAC peak modules",
        long_help = "ATAC peak modules. Each peak's row is its module's row, so\n\
                     this sets how finely peaks resolve; on the order of the\n\
                     genes with their own rows keeps the two sides comparable."
    )]
    peak_modules: usize,

    #[arg(
        long,
        default_value_t = 16,
        help = "Cells per finest pseudobulk drawn into training"
    )]
    phase1_cells_per_pb: usize,

    #[arg(
        long,
        default_value_t = 100_000,
        help = "ATAC modality is module-only when it has at least this many peaks (0 = off)"
    )]
    module_only_min_rows: usize,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device index (CUDA/Metal)")]
    device_no: usize,

    /* Clusters, seed, output */
    #[arg(
        long,
        help = "Target number of cell clusters (default: Leiden decides)"
    )]
    n_clusters: Option<usize>,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    seed: u64,

    #[arg(long, short, help = "Output prefix for {out}.*.parquet")]
    out: Box<str>,
}

pub fn run_peak_to_gene(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let rna = open_sparse_matrix_by_path(&args.rna)?;
    let gene_names = rna.row_names()?;
    let positions = match (&args.gene_coords, &args.gff_file) {
        (Some(tsv), _) => load_gene_coords_tsv(tsv, &gene_names)?,
        (None, Some(gff)) => load_gene_tss(gff, &gene_names)?,
        (None, None) => unreachable!("clap requires --gene-coords or --gff-file"),
    };
    info!(
        "Gene positions: {} of {} RNA genes placed",
        positions.iter().filter(|p| p.is_some()).count(),
        gene_names.len()
    );

    let keep: Option<FxHashSet<Box<str>>> = match args.qc.to_config() {
        Some(cfg) => {
            let barcodes = rna.column_names()?;
            let mut v = SparseIoVec::new();
            v.push(Arc::from(rna), None)?;
            let report = data_beans::qc_lib::compute_qc(&v, &cfg, None)?;
            Some(
                report
                    .emit_idx_unmasked()
                    .into_iter()
                    .map(|c| barcodes[c].clone())
                    .collect(),
            )
        }
        None => None,
    };

    let kernel = AbcKernel {
        window: args.cis_window,
        max_per_gene: args.max_cis,
        gamma: args.contact_gamma,
        pseudocount: args.contact_pseudocount,
    };
    let input = TwoTrackInput {
        rna_file: &args.rna,
        atac_file: &args.atac,
        batch_file: args.batch.as_deref(),
        gene_positions: &positions,
        kernel: &kernel,
        work_prefix: &args.out,
    };
    let cfg = LinkConfig {
        embed: TwoTrackConfig {
            embedding_dim: args.embedding_dim,
            epochs: args.epochs,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            proj_dim: args.proj_dim,
            feature_modules: args.feature_modules,
            peak_modules: args.peak_modules,
            phase1_cells_per_pb: args.phase1_cells_per_pb,
            module_only_min_rows: args.module_only_min_rows,
            align_weight: args.align_weight,
            mix: args.mix,
            weight_decay: args.weight_decay,
            seed: args.seed,
            device: args.device.clone(),
            device_no: args.device_no,
        },
        n_clusters: args.n_clusters,
        ..LinkConfig::default()
    };
    let summary = run_links(&input, &cfg, keep.as_ref())?;
    info!(
        "Done: {} genes, {} peaks, {} cis pairs, {} clusters → {}.links.parquet",
        summary.n_genes, summary.n_peaks, summary.n_pairs, summary.n_clusters, args.out
    );
    Ok(())
}
