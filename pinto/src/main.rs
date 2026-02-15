mod edge_profiles;
mod fit_srt_delta_svd;
mod fit_srt_gene_pair_svd;
mod fit_srt_link_community;
mod fit_srt_propensity;
mod link_community_gibbs;
mod link_community_model;
mod srt_cell_pairs;
mod srt_common;
mod srt_estimate_batch_effects;
mod srt_gene_graph;
mod srt_gene_pairs;
mod srt_graph_coarsen;
mod srt_input;
mod srt_knn_graph;

use clap::{Parser, Subcommand};
use colored::Colorize;
use fit_srt_delta_svd::*;
use fit_srt_gene_pair_svd::*;
use fit_srt_link_community::*;
use fit_srt_propensity::*;

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
                  PINTO discovers spatial interaction patterns from spatially-resolved\n\
                  transcriptomics data via link community detection. Communities are\n\
                  assigned to edges (cell-cell interactions) using gene module-based\n\
                  profiles or dimensionality reduction (SVD).\n\n\
                  Data files must be `.zarr` or `.h5` (`data-beans`) format. \n\
		  Or convert `.mtx` files using `data-beans from-mtx`.",
    term_width = 80
)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
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
                      \x20 2. Estimate batch effects delta [G x B]\n\
                      \x20 3. Build spatial KNN graph -> E cell pairs\n\
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
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.dictionary.parquet: SVD dictionary (2G x T)\n\
                      - {out}.latent.parquet: per-pair latent codes (E x T)"
    )]
    DeltaSvd(SrtDeltaSvdArgs),

    #[command(
        about = "Gene-gene interaction patterns analysis by SVD",
        long_about = "Gene-gene interaction analysis by randomized SVD.\n\n\
                      Discovers gene-gene co-expression patterns within spatial\n\
                      neighbourhoods by building a gene-gene graph and decomposing\n\
                      interaction deltas via SVD.\n\n\
                      Model:\n\
                      \x20 Gene-gene graph: edges (g1,g2) from KNN on posterior\n\
                      \x20 means or external network (e.g., BioGRID).\n\
                      \x20 For each cell j and gene pair (g1,g2):\n\
                      \x20   delta_j = (x_{g1,j} - mu_g1) * (x_{g2,j} - mu_g2)\n\
                      \x20   delta_j^+ = max(delta_j, 0)   positive interaction\n\
                      \x20 Aggregate per sample s:\n\
                      \x20   Y_s^{g1,g2} = sum_{j in s} delta_j^+\n\
                      \x20   Y_s | mu ~ Poisson(n_s * mu), mu ~ Gamma(a0, b0)\n\n\
                      Algorithm:\n\
                      \x20 1. Load data X [G x N] and coordinates [N x D]\n\
                      \x20 2. Build spatial KNN graph -> E cell pairs\n\
                      \x20 3. Random projection + binary sort -> S samples\n\
                      \x20 4. Preliminary collapse: gene x sample sums\n\
                      \x20 5. Poisson-Gamma on gene sums -> posterior means\n\
                      \x20 6. Build gene-gene KNN graph from posterior means\n\
                      \x20    (or load external network)\n\
                      \x20 7. For each cell j, each gene pair (g1,g2):\n\
                      \x20    delta_j = (x_{g1,j}-mu_g1)*(x_{g2,j}-mu_g2)\n\
                      \x20    accumulate delta^+ into sample s(j)\n\
                      \x20 8. Poisson-Gamma on gene-pair stats -> log means\n\
                      \x20 9. Randomized SVD on [n_edges x S] log means\n\
                      \x20 10. Nystrom: per-cell projection, average to pairs\n\
                      \x20     z_j = sum_{(g1,g2)} delta_j^+ * basis[g1:g2]\n\
                      \x20     z_e = (z_i + z_j) / 2, then L2 normalize\n\n\
                      Outputs:\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coords\n\
                      - {out}.gene_graph.parquet: gene-gene graph edges\n\
                      - {out}.dictionary.parquet: SVD dictionary (n_edges x T)\n\
                      - {out}.latent.parquet: per-pair latent codes (E x T)"
    )]
    GenePairDeltaSvd(SrtGenePairSvdArgs),

    #[command(
        about = "Estimate vertex propensity from edge clusters",
        long_about = "Estimate vertex (cell) propensity scores from edge\n\
                      (cell-pair) cluster assignments.\n\n\
                      Model:\n\
                      \x20 Given latent codes z_e [E x T] from delta-svd or\n\
                      \x20 gene-pair-delta-svd:\n\
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
                      - .latent.parquet (from delta-svd or gene-pair-delta-svd)\n\
                      - .coord_pairs.parquet (cell pair names)\n\
                      - Optionally, expression data (.zarr or .h5)\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity (N x K)\n\
                      - {out}.edge_cluster.parquet: edge cluster assignments"
    )]
    Propensity(SrtPropensityArgs),

    #[command(
        about = "Link community model via collapsed Gibbs sampling",
        long_about = "Link community model for spatial cell-cell interaction analysis.\n\n\
                      Treats spatial transcriptomics data as G separate weighted\n\
                      networks on N cells. Community membership is assigned at the\n\
                      link level via collapsed Gibbs sampling with a Poisson-Gamma\n\
                      conjugate model.\n\n\
                      Model:\n\
                      \x20 Given N cells, G genes, KNN spatial graph with E edges.\n\
                      \x20 For each edge e=(i,j), build module-count profile:\n\
                      \x20   y_e[m] = sum_{g in module m} (x_{g,i} + x_{g,j})\n\
                      \x20 Link community assignment:\n\
                      \x20   z_e in {1..K}\n\
                      \x20   y_e^m | z_e=k ~ Poisson(s_e * mu_{m,k})\n\
                      \x20   mu_{m,k} ~ Gamma(a0, b0)   collapsed out analytically\n\
                      \x20   s_e = sum_m y_e^m          link size factor\n\n\
                      Algorithm:\n\
                      \x20 1. Load data X [G x N] and coordinates [N x D]\n\
                      \x20 2. Estimate batch effects delta [G x B]\n\
                      \x20 3. Build spatial KNN graph -> E edges\n\
                      \x20 4. Discover gene modules via random sketch clustering\n\
                      \x20 5. Build edge profiles: y_e[m] = sum of module m counts\n\
                      \x20 6. Coarsen: cell KMeans -> super-edges, sum profiles\n\
                      \x20 7. Collapsed Gibbs sampling on coarsest super-edges:\n\
                      \x20    for each sweep:\n\
                      \x20      for each edge e:\n\
                      \x20        remove e from community z_e\n\
                      \x20        for t = 1..K:\n\
                      \x20          delta_t = sum_m [score(a0+E_{t,m}+y_e^m,\n\
                      \x20            b0+T_t+s_e) - score(a0+E_{t,m}, b0+T_t)]\n\
                      \x20        z_e ~ Categorical(softmax(delta))\n\
                      \x20        add e to community z_e\n\
                      \x20 8. Transfer labels to finer levels, refine\n\
                      \x20 9. Greedy finalization (argmax instead of sample)\n\
                      \x20 10. Output:\n\
                      \x20     node_membership[i,k] = frac of i's edges in k\n\
                      \x20     gene_modules[g,k] = mean expression in community k\n\n\
                      Outputs:\n\
                      - {out}.node_membership.parquet: soft membership (N x K)\n\
                      - {out}.gene_modules.parquet: gene modules (G x K)\n\
                      - {out}.link_community.parquet: link community assignments\n\
                      - {out}.scores.parquet: score trace\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.delta.parquet: batch effects (when multi-batch)"
    )]
    LinkCommunity(SrtLinkCommunityArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Propensity(args) => {
            fit_srt_propensity(args)?;
        }
        Commands::DeltaSvd(args) => {
            fit_srt_delta_svd(args)?;
        }
        Commands::GenePairDeltaSvd(args) => {
            fit_srt_gene_pair_svd(args)?;
        }
        Commands::LinkCommunity(args) => {
            fit_srt_link_community(args)?;
        }
    }

    Ok(())
}
