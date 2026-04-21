mod gene_network;
mod link_community;
mod propensity;
mod svd;
mod util;

#[cfg(test)]
mod test_support;

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
    about = "PINTO - Proximity-based Interaction Network for Tissue Organization",
    long_about = "PINTO discovers cell-cell interaction patterns from spatial\n\
                  transcriptomics via link community detection on cell-pair graphs.\n\n\
                  SUBCOMMANDS:\n\n\
                  \x20 lc    Link community model (recommended)\n\
                  \x20       Assigns each cell-cell edge to a community via collapsed\n\
                  \x20       Gibbs sampling on compressed all-gene edge profiles.\n\n\
                  \x20 dsvd  Delta-SVD model\n\
                  \x20       Cell-pair shared/difference analysis via Poisson-Gamma\n\
                  \x20       SVD on pseudobulk co-expression.\n\n\
                  \x20 prop  Propensity (standalone)\n\
                  \x20       Re-cluster edge latent codes from dsvd with different K.\n\n\
                  QUICK START:\n\n\
                  \x20 # Prepare data (convert MTX to HDF5):\n\
                  \x20 data-beans from-mtx -r features.tsv.gz -c barcodes.tsv.gz \\\n\
                  \x20   matrix.mtx.gz --backend hdf5 -o data.h5\n\n\
                  \x20 # Link community (spatial, 10x Visium):\n\
                  \x20 pinto lc data.h5 -c tissue_positions.csv -o results\n\n\
                  \x20 # Link community (expression-only, no coordinates):\n\
                  \x20 pinto lc data.h5 -o results\n\n\
                  \x20 # Delta-SVD:\n\
                  \x20 pinto dsvd data.h5 -c coords.csv -o results\n\n\
                  INPUT FILES:\n\n\
                  \x20 Data:   .h5 or .zarr (genes x cells, sparse). Multiple files\n\
                  \x20         comma-separated for multi-sample: s1.h5,s2.h5\n\
                  \x20 Coords: CSV/TSV/parquet, first column = barcode, rest = x,y,...\n\
                  \x20         Default columns: pxl_row_in_fullres,pxl_col_in_fullres\n\
                  \x20         Omit -c for expression-only mode.\n\
                  \x20 Batch:  -b labels.txt (one label per cell per line, optional)\n\n\
                  OUTPUT: All outputs are .parquet files with {out} prefix.\n\
                  \x20 Use --help on each subcommand for output file details.",
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
                      - {out}.gene_topic.parquet: gene-topic Poisson-Gamma statistics (G x K).\n\
                      \x20 Housekeeping-adjusted by default (row-scaled by 1/(bg[g]+ε));\n\
                      \x20 pass --no-adjust-housekeeping for raw rates"
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
                      - {out}.genes.parquet: cluster-specific gene expression (when expr_data_files provided).\n\
                      \x20 Housekeeping-adjusted by default (row-scaled by 1/(bg[g]+ε));\n\
                      \x20 pass --no-adjust-housekeeping for raw rates"
    )]
    Propensity(SrtPropensityArgs),

    #[command(
        alias = "lc",
        about = "Link community model via collapsed Gibbs sampling",
        long_about = "Link community detection for spatial transcriptomics.\n\n\
                      Assigns each cell-cell edge to one of K communities based on\n\
                      per-edge expression profiles, then derives per-cell soft membership.\n\n\
                      QUICK START:\n\n\
                      \x20 # Typical spatial run (10x Visium):\n\
                      \x20 pinto lc data.h5 -c tissue_positions.csv -o out\n\n\
                      \x20 # More communities:\n\
                      \x20 pinto lc data.h5 -c coords.csv -o out --n-communities 25\n\n\
                      \x20 # Expression-only (no coordinates):\n\
                      \x20 pinto lc data.h5 -o out\n\n\
                      \x20 # With external gene-pair network:\n\
                      \x20 pinto lc data.h5 -c coords.csv -o out \\\n\
                      \x20   --gene-network biogrid_pairs.tsv --n-outer-iter 3\n\n\
                      \x20 # Multi-sample with batch correction:\n\
                      \x20 pinto lc s1.h5,s2.h5 -c c1.csv,c2.csv -o out\n\n\
                      INPUT FILES:\n\n\
                      \x20 data.h5 / data.zarr   Genes-by-cells sparse matrix.\n\
                      \x20                        Convert from MTX: data-beans from-mtx in.mtx out.h5\n\
                      \x20 -c coords.csv          Cell coordinates (barcode,x,y).\n\
                      \x20                        Omit for expression-only mode.\n\n\
                      EDGE PROFILE MODES:\n\n\
                      \x20 Compressed all-gene profile (default):\n\
                      \x20   y_e = W^T(x_i + x_j), W = G × --proj-dim Gaussian basis.\n\
                      \x20   Every profile dim is a full linear combination of ALL genes\n\
                      \x20   (no genes dropped); M = proj-dim just compresses the gene axis.\n\
                      \x20   Optionally zero basis rows for genes below --min-gene-count.\n\n\
                      \x20 Gene-pair network (--gene-network file.tsv):\n\
                      \x20   External gene-gene edges (two-column TSV).\n\
                      \x20   Edge profile = positive co-expression deltas per pair.\n\
                      \x20   Pairs collapsed into modules if count > --n-edge-modules.\n\
                      \x20   --n-outer-iter > 1 re-estimates modules from community rates.\n\n\
                      ALGORITHM:\n\n\
                      \x20 1. Build spatial KNN graph (or expression KNN if no coords)\n\
                      \x20 2. Batch effect estimation (multi-sample only)\n\
                      \x20 3. Multi-level graph coarsening\n\
                      \x20 4. Build edge profiles (projection or gene-pair)\n\
                      \x20 5. Optional IDF reweighting against empirical marginal\n\
                      \x20 6. Collapsed Gibbs on coarsest super-edges\n\
                      \x20 7. Transfer labels to full resolution + EM Gibbs + greedy\n\
                      \x20    (with gene-pair modules: outer EM re-clusters modules)\n\
                      \x20 8. Extract cell propensity + gene-topic statistics\n\n\
                      See `pinto lc --help` for individual flag docs.\n\n\
                      OUTPUT FILES:\n\n\
                      \x20 {out}.propensity.parquet      Cell community membership [N × K]\n\
                      \x20 {out}.gene_topic.parquet      Gene-topic rates [G × K]\n\
                      \x20                                (housekeeping-adjusted by 1/(bg[g]+ε) by default;\n\
                      \x20                                 pass --no-adjust-housekeeping for raw)\n\
                      \x20 {out}.link_community.parquet  Edge community assignments [E × 3]\n\
                      \x20 {out}.coord_pairs.parquet     Cell pair coordinates\n\
                      \x20 {out}.scores.parquet          Score trace per iteration\n\
                      \x20 {out}.delta.parquet           Batch effects (multi-sample only)\n\
                      \x20 {out}.gene_graph.parquet      Gene-gene pairs (gene-pair mode only)"
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
