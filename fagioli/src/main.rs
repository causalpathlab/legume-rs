mod sim_eqtl;

use sim_eqtl::*;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line
        .replace('●', &"●".bright_yellow().to_string())
        .replace('╱', &"╱".truecolor(0, 100, 0).to_string())
        .replace('╲', &"╲".truecolor(0, 100, 0).to_string())
        .replace('(', &"(".truecolor(0, 100, 0).to_string())
        .replace(')', &")".truecolor(0, 100, 0).to_string())
        .replace('\\', &"\\".truecolor(0, 100, 0).to_string())
        .replace('/', &"/".truecolor(0, 100, 0).to_string())
        .replace('~', &"~".truecolor(101, 67, 33).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    // Factored Association study with Genetic Information
    // for Optimal Linear Interpolation
    println!("  {}", "fagioli".bold());
    println!();
}

#[derive(Parser)]
#[command(name = "fagioli")]
#[command(about = "Genetic association analysis toolkit")]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run eQTL simulation with cell type heterogeneity
    SimEqtl(SimulationArgs),
}

fn main() -> Result<()> {
    env_logger::init();

    // Show logo if help is requested
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::SimEqtl(args) => {
            sim_eqtl(args)?;
        }
    }

    Ok(())
}
