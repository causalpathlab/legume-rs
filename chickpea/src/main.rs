mod common;
mod run_embed;
mod run_spectral;

use crate::common::*;
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

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
    println!(
        " {}",
        "Bipartite Embedding for Single-Cell Multi-Omic Analysis".bold()
    );
    println!();
}

/// Bipartite Embedding for Single-Cell Multi-Omic Analysis
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Bipartite spectral SVD — fast cell + feature embeddings
    #[command(alias = "svd")]
    Spectral(run_spectral::SpectralArgs),

    /// Bipartite VAE — joint embedding via bilateral coarsening
    Embed(run_embed::EmbedArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Spectral(args) => run_spectral::run_spectral(args),
        Commands::Embed(args) => run_embed::run_embed(args),
    }
}
