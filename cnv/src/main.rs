//! `canna` — copy-number profiles from single-cell expression backends
//! (crate/lib: `cnv`).

use anyhow::Context;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use clap::{Args, Parser, Subcommand};
use cnv::cell_profile::{run_cell_profiles, CellProfileConfig};
use cnv::clone_call::{call_clones, write_clone_table, CloneCallConfig};
use cnv::gene_loci::GeneLocusIndex;
use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use log::{info, warn};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "canna",
    version,
    about = "canna — copy-number variation from single-cell expression",
    long_about = "Reads `data-beans` backends (.zarr.zip / .zarr / .h5) and writes\n\
                  copy-number profiles and clone strata for `--cnv-clones` consumers.\n\
                  \n\
                  Subcommands:\n  \
                  \x20 infercnv — per-cell inferCNV log-ratio profiles on a genomic-interval axis\n  \
                  \x20 clones   — donor-private CNV strata (`{out}.clones.tsv.gz`) for\n               \
                  senna / pinto `--cnv-clones`"
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Increase output verbosity")]
    verbose: bool,

    #[arg(
        long = "n-threads",
        visible_aliases = ["threads", "num-threads"],
        global = true,
        value_name = "N",
        help = "Limit the number of CPU threads (rayon global pool)"
    )]
    n_threads: Option<usize>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Per-cell inferCNV profiles against a normal reference",
        long_about = "For every query cell: depth-normalise, take the log-ratio against the\n\
                      per-gene mean of the reference (normal) cells, clip, average over a\n\
                      sliding window of genes within each chromosome, and median-centre.\n\
                      \n\
                      Output is a new data-beans backend `{out}.zarr.zip` whose rows are\n\
                      genomic intervals (`chr:start-end`; one gene each, or `--bin-size`\n\
                      tiles — prefer `1000000` / 1 Mb on large cohorts) and whose columns\n\
                      are the query cells. `{out}.features.tsv.gz`\n\
                      maps each row back to its genes; `{out}.cells.tsv.gz` has per-cell\n\
                      depth and mean |log-ratio| (CNV burden).\n\
                      \n\
                      With no `--ref`, the query cohort mean is the baseline: any CNV shared\n\
                      by every cell becomes invisible.\n\
                      \n\
                      Example:\n  \
                      canna infercnv --gff gencode.v46.gtf.gz \\\n    \
                      --ref Control1.zarr.zip Control2.zarr.zip \\\n    \
                      --out aml001.cnv AML001.zarr.zip"
    )]
    Infercnv(InferCnvArgs),
    #[command(
        about = "Donor-private CNV clone strata from inferCNV profiles",
        long_about = "Cluster cells on a genomic sketch of the inferCNV log-ratio\n\
                      (`--bin-size 0` = per-chromosome means, the inferCNV default;\n\
                      `>0` = fixed bp tiles), then keep a cluster as a putative clone\n\
                      only if it is both donor-enclosing and genomically structured\n\
                      (elevated segmental CN vs a size-matched null). Shared or flat\n\
                      clusters dump into stratum 0 (the mixable bucket).\n\
                      \n\
                      Gate tuning: a false clone is under-integration (cheap); a missed\n\
                      clone lets batch δ eat a private program (the motivating failure).\n\
                      Prefer a permissive `--k-max` / `--min-cells` and let the mixture\n\
                      (BIC over K=1..=n_eligible clusters) dump weak clusters to 0.\n\
                      \n\
                      Reads an existing CNV backend (`--from`), or runs `infercnv`\n\
                      first on `--ref` / QUERY and then calls clones.\n\
                      \n\
                      Writes `{out}.clones.tsv.gz` (cell, donor, cluster, stratum, …).\n\
                      Pass that file as `--cnv-clones` to senna (topic / masked-* /\n\
                      vae / svd / bge / gem / joint-*) or pinto (cage / lc / dsvd)\n\
                      so collapse cannot mix across clone boundaries."
    )]
    Clones(CloneArgs),
}

#[derive(Args, Debug, Clone)]
struct InferCnvArgs {
    #[arg(
        required = true,
        value_name = "QUERY",
        help = "Query backend files (.zarr.zip / .zarr / .h5); cells to profile"
    )]
    query: Vec<Box<str>>,

    #[arg(
        long,
        num_args = 1..,
        value_name = "REF",
        help = "Reference (normal) backend files; their per-gene mean is the diploid baseline"
    )]
    r#ref: Vec<Box<str>>,

    #[arg(long, help = "GFF/GTF with `gene` features (gene_id, gene_name)")]
    gff: Box<str>,

    #[arg(
        short,
        long,
        help = "Output prefix; writes {out}.zarr.zip, {out}.features.tsv.gz, {out}.cells.tsv.gz"
    )]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 101,
        help = "Smoothing window in genes (odd; ≤1 disables)"
    )]
    window: usize,

    #[arg(
        long,
        default_value_t = 3.0,
        help = "Clip per-gene log-ratio at ±clip before smoothing (≤0 disables)"
    )]
    clip: f32,

    #[arg(
        long,
        default_value_t = 1e4,
        help = "Depth-normalisation target: ln(1 + scale·x/depth)"
    )]
    scale: f32,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Keep genes whose mean raw count over reference cells is ≥ this"
    )]
    min_mean_expr: f32,

    #[arg(
        long,
        default_value_t = 0,
        help = "Average smoothed genes into fixed genomic tiles of this many bp (0 = one row per gene)",
        long_help = "Average smoothed genes into fixed genomic tiles of this many bp.\n\
                     Default 0 keeps classic inferCNV gene-level rows.\n\
                     For large cohorts, prefer `--bin-size 1000000` (1 Mb):\n\
                     after a 101-gene window the signal is already ~Mb-scale,\n\
                     and autosomes compress to ~3k rows instead of ~15–20k genes."
    )]
    bin_size: i64,

    #[arg(long, default_value_t = 1000, help = "Cells per streamed block")]
    block_size: usize,

    #[arg(long, help = "Do not median-centre each cell after smoothing")]
    no_center: bool,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "chrX,chrY,chrM",
        value_name = "CHR[,CHR..]",
        help = "Chromosomes to drop from the gene axis (inferCNV default X,Y,M; `none` keeps all)"
    )]
    exclude_chr: Vec<Box<str>>,

    #[arg(long, help = "Preload backend columns into memory")]
    preload: bool,
}

#[derive(Args, Debug, Clone)]
struct CloneArgs {
    #[arg(
        value_name = "QUERY",
        help = "Query backends (expression); ignored when `--from` is set"
    )]
    query: Vec<Box<str>>,

    #[arg(
        long,
        num_args = 1..,
        value_name = "REF",
        help = "Reference (normal) backends for inferCNV; ignored with `--from`"
    )]
    r#ref: Vec<Box<str>>,

    #[arg(long, help = "GFF/GTF; required unless `--from`")]
    gff: Option<Box<str>>,

    #[arg(
        long,
        help = "Existing inferCNV backend (`.zarr.zip`); skip the expression pass"
    )]
    from: Option<Box<str>>,

    #[arg(
        short,
        long,
        help = "Output prefix; writes {out}.clones.tsv.gz (and inferCNV artifacts when not `--from`)"
    )]
    out: Box<str>,

    #[arg(long, default_value_t = 101)]
    window: usize,
    #[arg(long, default_value_t = 3.0)]
    clip: f32,
    #[arg(long, default_value_t = 1e4)]
    scale: f32,
    #[arg(long, default_value_t = 0.1)]
    min_mean_expr: f32,
    #[arg(
        long,
        default_value_t = 0,
        help = "Genomic tile size in bp for inferCNV rows and the clone sketch (0 = inferCNV default)",
        long_help = "Same `--bin-size` as `canna infercnv` (default 0 = classic inferCNV).\n\
                     When running inferCNV first: average smoothed genes into fixed genomic\n\
                     tiles of this many bp (0 = one row per gene). Prefer `1000000` (1 Mb)\n\
                     on large cohorts.\n\
                     For the clone sketch (also with `--from`): `0` = one dim per chromosome;\n\
                     `>0` = one dim per tile of the interval midpoint."
    )]
    bin_size: i64,
    #[arg(long, default_value_t = 1000)]
    block_size: usize,
    #[arg(long)]
    no_center: bool,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "chrX,chrY,chrM",
        value_name = "CHR[,CHR..]"
    )]
    exclude_chr: Vec<Box<str>>,
    #[arg(long)]
    preload: bool,

    #[arg(
        long,
        default_value_t = 8,
        help = "k-means K on the genomic sketch (overclustering is fine; lean permissive)",
        long_help = "k-means K on the genomic sketch (`--bin-size` dims). Overclustering is\n\
                     fine: weak clusters dump to stratum 0. A false clone is under-integration\n\
                     (cheap); a missed clone lets δ absorb private CN — lean permissive."
    )]
    k_max: usize,
    #[arg(
        long,
        help = "Optional purity floor on top of the clone-score mixture (omit = mixture only)"
    )]
    min_purity: Option<f32>,
    #[arg(
        long,
        default_value_t = 50,
        help = "Minimum cells to keep a cluster as a clone (lean permissive)",
        long_help = "Clusters smaller than this cannot be clones. Lean permissive:\n\
                     a missed small clone is worse than a false one that fails to mix."
    )]
    min_cells: usize,
    #[arg(
        long = "segmental-z",
        visible_alias = "spatial-z",
        help = "Optional segmental-CN z floor on top of the mixture (omit = mixture only)",
        long_help = "Optional floor on the segmental (genomic) z-score of a cluster's\n\
                     chromosome-sketch L1 vs a size-matched null. This is genomic\n\
                     roughness, not tissue spatial coordinates (unlike pinto's spatial z).\n\
                     `--spatial-z` is a deprecated alias."
    )]
    spatial_z: Option<f32>,
    #[arg(
        long,
        default_value_t = 32,
        help = "Null permutations for the segmental-CN z-score"
    )]
    n_perm: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn run_infercnv(args: &InferCnvArgs) -> anyhow::Result<()> {
    let has_ref = !args.r#ref.is_empty();
    if !has_ref {
        warn!("no --ref given: using the query cohort mean as baseline; shared CNVs will be invisible");
    }

    // Reference backends first, then query, in one shared-row load.
    let mut files: Vec<Box<str>> = args.r#ref.clone();
    files.extend(args.query.iter().cloned());

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files,
        preload: args.preload,
        ..Default::default()
    })?;
    let data = loaded.data;

    let per_backend = data.num_columns_by_data()?;
    let n_ref_cols: usize = per_backend.iter().take(args.r#ref.len()).sum();
    let n_total = data.num_columns();
    let (ref_cols, query_cols): (Vec<usize>, Vec<usize>) = if has_ref {
        ((0..n_ref_cols).collect(), (n_ref_cols..n_total).collect())
    } else {
        ((0..n_total).collect(), (0..n_total).collect())
    };
    info!(
        "loaded {} genes; {} reference cells, {} query cells",
        data.num_rows(),
        ref_cols.len(),
        query_cols.len()
    );

    let row_names = data.row_names()?;
    let loci = GeneLocusIndex::from_gff(&args.gff)
        .with_context(|| format!("reading {}", args.gff))?
        .resolve_all(&row_names);

    let cfg = CellProfileConfig {
        window: args.window,
        clip: args.clip,
        scale: args.scale,
        min_mean_expr: args.min_mean_expr,
        bin_size: args.bin_size,
        block_size: args.block_size,
        center: !args.no_center,
        exclude_chr: args
            .exclude_chr
            .iter()
            .filter(|c| !c.is_empty() && !c.eq_ignore_ascii_case("none"))
            .cloned()
            .collect(),
    };
    let outs = run_cell_profiles(&data, &ref_cols, &query_cols, &loci, &cfg, &args.out)?;
    info!(
        "done: {} ({} intervals × {} cells)",
        outs.backend, outs.n_rows, outs.n_cells
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    auxiliary_data::logging::init_logger(cli.verbose);

    if let Some(n) = cli.n_threads {
        anyhow::ensure!(n >= 1, "--n-threads must be >= 1");
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    match &cli.command {
        Commands::Infercnv(args) => run_infercnv(args),
        Commands::Clones(args) => run_clones(args),
    }
}

fn run_clones(args: &CloneArgs) -> anyhow::Result<()> {
    let backend_path = if let Some(from) = args.from.as_ref() {
        from.to_string()
    } else {
        anyhow::ensure!(
            !args.query.is_empty(),
            "QUERY backends required unless --from"
        );
        let gff = args
            .gff
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--gff is required unless --from"))?;
        let infer = InferCnvArgs {
            query: args.query.clone(),
            r#ref: args.r#ref.clone(),
            gff: gff.clone(),
            out: args.out.clone(),
            window: args.window,
            clip: args.clip,
            scale: args.scale,
            min_mean_expr: args.min_mean_expr,
            bin_size: args.bin_size,
            block_size: args.block_size,
            no_center: args.no_center,
            exclude_chr: args.exclude_chr.clone(),
            preload: args.preload,
        };
        run_infercnv(&infer)?;
        format!("{}.zarr.zip", args.out)
    };

    let opened = try_open_or_convert(&backend_path)?;
    let mut data = SparseIoVec::new();
    data.push(Arc::from(opened), None)?;
    let cfg = CloneCallConfig {
        bin_size: args.bin_size,
        k_max: args.k_max,
        min_purity: args.min_purity,
        min_cells: args.min_cells,
        spatial_z: args.spatial_z,
        n_perm: args.n_perm,
        seed: args.seed,
        ..Default::default()
    };
    let rows = call_clones(&data, &cfg)?;
    let n_clone = rows.iter().filter(|r| r.stratum > 0).count();
    let n_strata = rows.iter().map(|r| r.stratum).max().unwrap_or(0);
    let path = format!("{}.clones.tsv.gz", args.out);
    write_clone_table(&rows, &path)?;
    info!(
        "wrote {path} ({} cells, {n_clone} in {} donor-private clone(s))",
        rows.len(),
        n_strata
    );
    Ok(())
}
