mod gene_network;
mod link_community;
mod propensity;
mod svd;
mod util;

use clap::{Parser, Subcommand};
use colored::Colorize;
use link_community::fit::*;
use propensity::*;
use svd::fit::*;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('▄', &"▄".truecolor(190, 100, 70).to_string())
        .replace('▓', &"▓".truecolor(217, 119, 87).to_string())
        .replace('█', &"█".truecolor(180, 120, 60).to_string())
        .replace('▀', &"▀".truecolor(190, 100, 70).to_string())
        .replace('━', &"━".truecolor(0, 100, 0).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(
        " {}",
        "Proximity-based Interaction Network --> Tissue Organization".bold()
    );
    println!();
}

/// PINTO
#[derive(Parser, Debug)]
#[command(
    version,
    about = "PINTO",
    long_about = "Proximity-based Interaction Network analysis to dissect Tissue Organizations\n\n\
                  PINTO discovers cell-cell interaction patterns from transcriptomics\n\
                  data via link community detection. Communities are assigned to edges\n\
                  (cell-cell interactions) using gene module-based profiles or\n\
                  dimensionality reduction (SVD).\n\n\
                  PINTO supports two modes:\n\
                  - Spatial mode (recommended): provide coordinate files to build\n\
                    a spatial KNN graph from cell positions.\n\
                  - Expression mode: omit --coord to build a KNN graph from\n\
                    random-projected gene expression embeddings. 2D coordinates\n\
                    are generated via force-directed layout for visualization.\n\n\
                  FILE FORMATS:\n\n\
                  Data files:\n\
                  - Must be `.zarr` or `.h5` (HDF5) format\n\
                  - Convert from `.mtx` using: data-beans from-mtx input.mtx output.zarr\n\
                  - Multiple files can be provided (comma-separated)\n\n\
                  Coordinate files (recommended, one per data file):\n\
                  - CSV, TSV, or space-delimited text files (or .parquet)\n\
                  - First column: cell/barcode names (must match data file)\n\
                  - Subsequent columns: spatial coordinates (x, y, etc.)\n\
                  - Header row optional (auto-detected or specify with --coord-header-row)\n\
                  - Default column names: pxl_row_in_fullres, pxl_col_in_fullres (10X Visium)\n\
                  - Use --coord-column-names for different column headers\n\
                  - When omitted, KNN graph is built from expression embeddings\n\n\
                  Batch files (optional, one per data file):\n\
                  - Plain text file, one batch label per line\n\
                  - Must have one line for each cell in the corresponding data file\n\
                  - If not provided, each data file is treated as a separate batch\n\n\
                  WORKFLOWS:\n\n\
                  1. SVD-based cell-level analysis (spatial):\n\
                     pinto dsvd data.zarr -c coords.csv -o out\n\n\
                  2. SVD-based cell-level analysis (expression only):\n\
                     pinto dsvd data.zarr -o out\n\n\
                  3. Link community model (gene module-based):\n\
                     pinto lc data.zarr -c coords.csv -k 20 -o out\n\n\
                  4. Link community with gene-pair network:\n\
                     pinto lc data.zarr -c coords.csv -o out --gene-network net.tsv\n\n\
                  All subcommands accept --coord or work without it.\n\
                  Use prop subcommand for re-clustering with different K.\n\n\
                  Choose dsvd for cell-level shared/difference patterns, or lc for\n\
                  probabilistic link community detection (with gene modules or\n\
                  gene-pair interaction profiles via --gene-network).",
    term_width = 80
)]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        alias = "dsvd",
        about = "Gene-level shared/difference analysis by SVD",
        long_about = "Gene-level cell-cell interaction analysis by SVD with\n\
                      shared/difference channels.\n\n\
                      Model:\n\
                      \x20 For each cell pair e=(i,j) and gene g:\n\
                      \x20   sigma_e^g = log1p(x_ig) + log1p(x_jg)    shared\n\
                      \x20   delta_e^g = |log1p(x_ig) - log1p(x_jg)|  difference\n\
                      \x20 Pairs grouped into S pseudobulk samples via\n\
                      \x20 graph-constrained coarsening.\n\
                      \x20 Per sample s, gene g:\n\
                      \x20   Y_s^g = sum_{e in s} sigma_e^g  (or delta_e^g)\n\
                      \x20   Y_s^g | mu_g ~ Poisson(n_s * mu_g)\n\
                      \x20   mu_g ~ Gamma(a0, b0)   collapsed out\n\n\
                      Algorithm:\n\
                      \x20 1. Load data X [G x N] and coordinates [N x D]\n\
                      \x20    (if no coordinates, use expression embeddings)\n\
                      \x20 2. Estimate batch effects delta [G x B]\n\
                      \x20 3. Build KNN graph -> E cell pairs\n\
                      \x20    (spatial KNN from coordinates, or expression KNN\n\
                      \x20     from random-projected gene expression)\n\
                      \x20 4. Random projection of cells [N x P]\n\
                      \x20 5. Graph coarsening -> assign pairs to S samples\n\
                      \x20 6. Collapse: accumulate sigma/delta per gene per sample\n\
                      \x20    Sigma[g,s] += log1p(x_ig) + log1p(x_jg)\n\
                      \x20    Delta[g,s] += |log1p(x_ig) - log1p(x_jg)|\n\
                      \x20 7. Fit Poisson-Gamma -> posterior log means\n\
                      \x20    mu_hat[g,s] = E[ln mu_g | Y_s^g]\n\
                      \x20 8. Stack M = [mu_shared; mu_diff] [2G x S]\n\
                      \x20 9. Randomized SVD: M = U S V^T, keep top T cols\n\
                      \x20 10. Nystrom: for each pair e=(i,j):\n\
                      \x20     z_e = basis_shared^T * sigma_e + basis_diff^T * delta_e\n\
                      \x20     z_e <- z_e / ||z_e||   (L2 normalize)\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effects (when multi-batch)\n\
                      - {out}.coord_pairs.parquet: cell pair coordinates\n\
                      - {out}.basis.parquet: SVD basis (2G x T)\n\
                      - {out}.latent.parquet: per-pair latent codes (E x T)\n\
                      - {out}.propensity.parquet: cell propensity (N x K)\n\
                      - {out}.gene_topic.parquet: gene-topic Poisson-Gamma statistics (G x K)"
    )]
    DeltaSvd(SrtDeltaSvdArgs),

    #[command(
        alias = "prop",
        about = "Estimate vertex propensity from edge clusters (standalone)",
        long_about = "Estimate vertex (cell) propensity scores from edge\n\
                      (cell-pair) cluster assignments.\n\n\
                      NOTE: dsvd now produces propensity and edge cluster\n\
                      outputs inline. Use this subcommand only when you need\n\
                      a different K or separate expression data.\n\n\
                      Model:\n\
                      \x20 Given latent codes z_e [E x T] from delta-svd:\n\
                      \x20   c_e = argmin_k ||z_e - centroid_k||  K-means cluster\n\
                      \x20 For each vertex i:\n\
                      \x20   p_i[k] = |{e incident to i : c_e = k}| / degree(i)\n\
                      \x20 Optionally, cluster-specific gene expression:\n\
                      \x20   mu_{g,k} ~ Gamma(a0, b0) with pseudocount sums\n\n\
                      Algorithm:\n\
                      \x20 1. Load latent codes Z [E x T] from .latent.parquet\n\
                      \x20 2. Load cell pair names from .coord_pairs.parquet\n\
                      \x20 3. K-means on Z^T -> cluster assignment c_e for each edge\n\
                      \x20 4. For each vertex i, count edges per cluster:\n\
                      \x20    p_i[k] = count(c_e=k for e incident to i) / deg(i)\n\
                      \x20 5. dominant_cluster[i] = argmax_k p_i[k]\n\
                      \x20 6. If expression data provided:\n\
                      \x20    weighted gene sums per cluster -> Poisson-Gamma\n\n\
                      Inputs:\n\
                      - .latent.parquet (from delta-svd)\n\
                      - .coord_pairs.parquet (cell pair names)\n\
                      - Optionally, expression data (.zarr or .h5)\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity (N x K)\n\
                      - {out}.edge_cluster.parquet: edge cluster assignments\n\
                      - {out}.genes.parquet: cluster-specific gene expression (when expr_data_files provided)"
    )]
    Propensity(SrtPropensityArgs),

    #[command(
        alias = "lc",
        about = "Link community model via collapsed Gibbs sampling",
        long_about = "Link community model for cell-cell interaction analysis.\n\n\
                      Treats transcriptomics data as G separate weighted networks\n\
                      on N cells. Community membership is assigned at the link\n\
                      level via collapsed Gibbs sampling with a Poisson-Gamma\n\
                      conjugate model.\n\n\
                      Supports spatial coordinates or expression-only mode\n\
                      (omit --coord to build KNN from expression embeddings).\n\n\
                      Edge profiles can be built in three ways:\n\
                      \x20 1. Gene modules (default): genes clustered into M modules,\n\
                      \x20    y_e[m] = sum_{g in module m} (x_{g,i} + x_{g,j})\n\
                      \x20 2. Gene-pair network (--gene-network): interaction deltas\n\
                      \x20    y_e[p] = sum of positive co-expression deltas per pair\n\
                      \x20 3. Random projection (--n-gene-modules 0): y_e = W^T(x_i+x_j)\n\n\
                      Model:\n\
                      \x20 Given N cells, G genes, KNN graph with E edges.\n\
                      \x20 Link community assignment:\n\
                      \x20   z_e in {1..K}\n\
                      \x20   y_e^m | z_e=k ~ Poisson(s_e * mu_{m,k})\n\
                      \x20   mu_{m,k} ~ Gamma(a0, b0)   collapsed out analytically\n\
                      \x20   s_e = sum_m y_e^m          link size factor\n\n\
                      Algorithm:\n\
                      \x20 1. Load data X [G x N] and coordinates [N x D]\n\
                      \x20 2. Build KNN graph -> E edges\n\
                      \x20 3. Estimate batch effects (multi-batch only)\n\
                      \x20 4. Build edge profiles (gene modules, gene-pair, or projection)\n\
                      \x20 5. Multi-level coarsening -> super-edges\n\
                      \x20 6. Collapsed Gibbs sampling on coarsest super-edges\n\
                      \x20 7. Progressive encoder training (coarse to fine)\n\
                      \x20 8. EM refinement + greedy finalization on full edges\n\
                      \x20 9. Output: cell propensity, gene-topic stats, assignments\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: soft membership (N x K)\n\
                      - {out}.gene_modules.parquet: gene module assignments (when using modules)\n\
                      - {out}.gene_graph.parquet: gene-gene edges (when using --gene-network)\n\
                      - {out}.link_community.parquet: link community assignments\n\
                      - {out}.scores.parquet: score trace\n\
                      - {out}.coord_pairs.parquet: cell pair coordinates\n\
                      - {out}.delta.parquet: batch effects (when multi-batch)"
    )]
    LinkCommunity(SrtLinkCommunityArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    crate::util::common::init_logger(cli.verbose);

    match &cli.commands {
        Commands::Propensity(args) => {
            fit_srt_propensity(args)?;
        }
        Commands::DeltaSvd(args) => {
            fit_srt_delta_svd(args)?;
        }
        Commands::LinkCommunity(args) => {
            fit_srt_link_community(args)?;
        }
    }

    Ok(())
}
