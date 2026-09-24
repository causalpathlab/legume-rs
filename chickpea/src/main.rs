use chickpea::common::*;
use chickpea::p2g;
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");
const ENV_HELP: &str = "Environment variables:\n  RUST_LOG=info  Enable logging to stderr";

fn colorize_logo_line(line: &str) -> String {
    line.replace(':', &":".truecolor(210, 180, 120).to_string())
        .replace('*', &"*".truecolor(240, 210, 140).to_string())
        .replace('o', &"o".truecolor(160, 120, 60).to_string())
        .replace('.', &".".truecolor(180, 160, 110).to_string())
        .replace('^', &"^".truecolor(180, 160, 110).to_string())
        .replace('\'', &"'".truecolor(180, 160, 110).to_string())
        .replace('/', &"/".truecolor(139, 90, 43).to_string())
        .replace('\\', &"\\".truecolor(139, 90, 43).to_string())
        .replace('_', &"_".truecolor(139, 90, 43).to_string())
        .replace('=', &"=".green().to_string())
        .replace('>', &">".bright_green().to_string())
        .replace('-', &"-".green().to_string())
        .replace('|', &"|".green().to_string())
        .replace('▀', &"▀".truecolor(100, 180, 100).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(" {}", "Multi-Omic Linkage Analysis".bold());
    println!();
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "chickpea —\n\
             peak-to-gene cis-regulatory linkage for paired single-cell RNA + ATAC",
    long_about = "chickpea — peak-to-gene cis-regulatory linkage\n\
                  \n\
                  Links ATAC peaks to RNA genes in paired single-cell RNA + ATAC data.\n\
                  Peaks become gene features: each gene gets an ATAC track,\n\
                  embedded next to its RNA track against shared pseudobulks.\n\
                  Each gene then attends over its cis peaks, and the shares are the links.\n\
                  \n\
                  Usage:\n\
                  data-beans-sim multiome -o sim && chickpea peak-to-gene --rna sim.rna.zarr \\\n\
                  --atac sim.atac.zarr --gene-coords sim.gene_coords.tsv.gz -o out",
    term_width = 80
)]
struct Cli {
    #[arg(
        short = 'v',
        long,
        global = true,
        help = "Enable verbose logging",
        long_help = "Enable verbose logging to stderr. Equivalent to setting RUST_LOG=info."
    )]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Link cis peaks to genes over a multiome embedding (bge-style)
    #[command(
        long_about = "Link ATAC peaks to RNA genes.\n\
                      \n\
                      RNA genes and ATAC peaks are embedded on one multiome axis\n\
                      (ATAC module-only, like senna bge --multiome). Each gene's\n\
                      score mixes in the accessibility of its cis peaks, pooled\n\
                      through ReLU gates on distance (--mix), and its profile is\n\
                      pulled toward it (--align-weight), so genes and peaks share\n\
                      one feature space.\n\
                      A link's score is its gate times the correlation across\n\
                      pseudobulks of the peak module's score and the gene's.\n\
                      \n\
                      {out}.links.parquet has gene, peak, distance, abc, gate,\n\
                      corr and score (gate × corr).\n\
                      {out}.links_by_cluster.parquet has gene_idx/peak_idx/cluster shares.\n\
                      Gene, peak and cell embeddings and cell clusters are written too.",
        after_long_help = ENV_HELP,
        aliases = ["p2g", "peak2gene"]
    )]
    PeakToGene(p2g::PeakToGeneArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    // The shared logger routes log lines above the progress bars (one
    // MultiProgress for the whole workspace), so they do not corrupt each other.
    data_beans::aux::logging::init_logger(cli.verbose);

    match &cli.commands {
        Commands::PeakToGene(args) => p2g::run_peak_to_gene(args),
    }
}
