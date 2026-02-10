mod common;
mod input;
mod run_cis_linking;
mod run_trans_linking;

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
        "Cis-regulatory Hi-C and Chromatin-linked Kinetics for Peak-gene Enrichment Analysis"
            .bold()
    );
    println!();
}

/// Cis-regulatory Hi-C and Chromatin-linked Kinetics for Peak-gene Enrichment Analysis
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Cis-regulatory linking from peaks to target genes
    #[command(alias = "cis")]
    CisLinking,

    /// Trans-regulatory linking from peaks to target genes
    #[command(alias = "trans")]
    TransLinking,
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::CisLinking => {
            todo!("cis-linking not yet implemented");
        }
        Commands::TransLinking => {
            todo!("trans-linking not yet implemented");
        }
    }
}
