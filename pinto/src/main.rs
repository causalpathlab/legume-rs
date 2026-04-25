mod gene_network;
mod link_community;
mod lr_activity;
mod plot;
mod propensity;
mod svd;
mod util;

#[cfg(test)]
mod test_support;

use clap::{Parser, Subcommand};
use colored::Colorize;
use link_community::fit::*;
use lr_activity::{fit_srt_lr_activity, SrtLrActivityArgs};
use plot::{make_srt_plot, SrtPlotArgs};
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
                      \x20 Columns: 0 .. K-1, cluster (argmax), entropy (Shannon, nats).\n\
                      - {out}.gene_topic.parquet: gene-topic Poisson-Gamma statistics (G x K).\n\
                      \x20 Housekeeping-adjusted by default (row-scaled by 1/(bg[g]+ε));\n\
                      \x20 pass --no-adjust-housekeeping for raw rates\n\
                      - {out}.metadata.json: information-flow manifest used by\n\
                      \x20 `pinto plot` and `pinto lr-activity` (lists every parquet)."
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
                      \x20 Columns: propensity_0 .. propensity_{K-1}, cluster (argmax),\n\
                      \x20 entropy (Shannon, nats), plus optional coord trailer.\n\
                      - {out}.edge_cluster.parquet: edge cluster assignments\n\
                      - {out}.genes.parquet: cluster-specific gene expression (when expr_data_files provided).\n\
                      \x20 Housekeeping-adjusted by default (row-scaled by 1/(bg[g]+ε));\n\
                      \x20 pass --no-adjust-housekeeping for raw rates\n\
                      - {out}.metadata.json: information-flow manifest used by\n\
                      \x20 `pinto plot`."
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
                      \x20 # With external gene-gene network:\n\
                      \x20 pinto lc data.h5 -c coords.csv -o out \\\n\
                      \x20   --gene-network biogrid_pairs.tsv\n\n\
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
                      \x20 Gene-network module-pair profile (--gene-network file.tsv):\n\
                      \x20   External gene-gene edges (two-column TSV), optionally SNN-\n\
                      \x20   augmented, k-core-trimmed, Leiden-clustered into gene modules.\n\
                      \x20   Edge profile is SPARSE over module-pairs (a, b) with entries\n\
                      \x20   max(0, x_{i,a}·x_{j,b} + x_{i,b}·x_{j,a} − X_i·X_j · deg(a)·deg(b)/(2W)²).\n\
                      \x20   Controls: --snn-min-shared, --gene-trim-min-degree,\n\
                      \x20   --gene-modules-resolution.\n\n\
                      ALGORITHM:\n\n\
                      \x20 1. Build spatial KNN graph (or expression KNN if no coords)\n\
                      \x20 2. Batch effect estimation (multi-sample only)\n\
                      \x20 3. Multi-level graph coarsening\n\
                      \x20 4. Resolve gene modules (projection or SNN + k-core + Leiden)\n\
                      \x20 5. Build sparse edge profiles (projection or module-pair residual)\n\
                      \x20 6. V-cycle Gibbs + greedy across coarsening levels\n\
                      \x20 7. Component-EM + greedy on full fine-resolution edges\n\
                      \x20 8. Extract cell propensity + gene-topic statistics (+ BHC)\n\n\
                      See `pinto lc --help` for individual flag docs.\n\n\
                      OUTPUT FILES:\n\n\
                      \x20 {out}.propensity.parquet      Cell community membership [N × K]\n\
                      \x20                                Columns: 0 .. K-1, plus `entropy`\n\
                      \x20                                (Shannon entropy of each row, nats).\n\
                      \x20 {out}.gene_topic.parquet      Gene-topic rates [G × K]\n\
                      \x20                                (housekeeping-adjusted by 1/(bg[g]+ε) by default;\n\
                      \x20                                 pass --no-adjust-housekeeping for raw)\n\
                      \x20 {out}.link_community.parquet  Edge community assignments [E × 3]\n\
                      \x20 {out}.coord_pairs.parquet     Cell pair coordinates\n\
                      \x20 {out}.scores.parquet          Score trace per iteration\n\
                      \x20 {out}.delta.parquet           Batch effects (multi-sample only)\n\
                      \x20 {out}.gene_graph.parquet      Gene-gene pairs (gene-pair mode only)\n\
                      \x20 {out}.L{l}.*.parquet          Per-cascade-level outputs (unless --no-level-outputs)\n\
                      \x20 {out}.bhc.*.parquet           BHC consensus outputs (unless --no-bhc)\n\
                      \x20 {out}.metadata.json           Information-flow manifest:\n\
                      \x20                                lists every parquet, level tags,\n\
                      \x20                                BHC presence, and (when set by\n\
                      \x20                                lr-activity) the lr_activity JSON.\n\
                      \x20                                Pass this path to `pinto plot --from`\n\
                      \x20                                or `pinto lr-activity --lc-prefix`."
    )]
    LinkCommunity(SrtLinkCommunityArgs),

    #[command(
        alias = "p",
        about = "Plot spatial scatter from pinto lc/dsvd/prop outputs",
        long_about = "Render publication-quality PDFs (+ SVG/PNG) from pinto\n\
                      outputs. Works on `pinto lc`, `pinto dsvd`, and\n\
                      `pinto prop` runs. Defaults to flat-top hexagon markers\n\
                      that tile tightly; size adapts to plot density.\n\n\
                      INPUT (--from):\n\n\
                      \x20 Pass either a `{prefix}.metadata.json` (preferred —\n\
                      \x20 carries level list, BHC presence, and any lr_activity\n\
                      \x20 JSON) or a bare `{prefix}` (auto-globs *.parquet).\n\n\
                      PER-LEVEL × PER-CORE PLOTS (always):\n\n\
                      \x20 community.pdf                one color per community\n\
                      \x20 propensity.argmax.pdf        size ∝ propensity (capped at hex tile),\n\
                      \x20                              color = argmax community\n\
                      \x20 propensity.community{k}.pdf  per-community soft-membership\n\
                      \x20                              (size scales 0 → tile by propensity)\n\
                      \x20 mesh.pdf                     cell-cell edges (lc only; --no-mesh skips)\n\
                      \x20 markers.topic{k}.{gene}.heatmap.pdf       grayscale on log1p expr\n\
                      \x20                                           (darker = higher)\n\
                      \x20 markers.topic{k}.{gene}.by-community.pdf  color by argmax,\n\
                      \x20                                           size ∝ log expr\n\n\
                      OPT-IN: --show-interfaces (per (level, core)):\n\n\
                      \x20 interfaces.pdf  All cells; radius scaled by entropy\n\
                      \x20                 quantile rank (within core), single dark\n\
                      \x20                 gray fill — high-entropy boundary cells\n\
                      \x20                 stand out as full hex tiles, low-entropy\n\
                      \x20                 interior cells fade to 0.\n\
                      \x20 interfaces.tsv  Per focal cell: dominant community,\n\
                      \x20                 1- and 2-hop neighbor mix, top-N marker\n\
                      \x20                 genes per neighbor community.\n\
                      \x20 Tunables: --entropy-quantile, --neighborhood-hops,\n\
                      \x20            --max-interface-cells, --interface-top-genes.\n\n\
                      LR-ACTIVITY OVERLAY (auto-discovered):\n\n\
                      \x20 When the metadata.json carries an `outputs.lr_activity`\n\
                      \x20 path (set automatically by `pinto lr-activity`), one\n\
                      \x20 PDF is written per (core × significant LR pair):\n\
                      \x20   lr.core{batch}.lr.B{batch}.C{community}.{L}-{R}.pdf\n\
                      \x20 Layout:\n\
                      \x20   - Faint hex tiling of all core cells (tissue context)\n\
                      \x20   - Per-community CC convex hulls (thin gray outlines)\n\
                      \x20     for the pair's community only\n\
                      \x20   - Quiver of L→R arrows along edges incident to a\n\
                      \x20     boundary cell (1-hop expanded). Arrow direction\n\
                      \x20     comes from per-edge L+R expression argmax (needs --data).\n\
                      \x20   - Color = diverging blue↔red on edge coexpression\n\
                      \x20     `sqrt(L·R)` minus the per-pair edge mean (centered\n\
                      \x20     on 0 = typical edge of this pair).\n\
                      \x20 Tunables: --lr-top-pairs, --lr-commit-threshold,\n\
                      \x20            --lr-hull-min-cells, --no-lr-hulls,\n\
                      \x20            --no-lr-overlay, --lr-coexpr-bins,\n\
                      \x20            --lr-activity-json (override path).\n\n\
                      Levels: `final`, `L0..Ln` (V-cycle), `bhc`. Cores: one\n\
                      per batch label (read from coord_pairs.parquet).\n\n\
                      Outlier handling is robust by default: coordinate bounds,\n\
                      color scales, and size scales all use percentile clipping\n\
                      (see --coord-clip, --expr-clip).\n\n\
                      A JSON manifest listing every emitted file is written to\n\
                      {out}.plot.manifest.json."
    )]
    Plot(SrtPlotArgs),

    #[command(
        aliases = ["lra", "test-lr"],
        about = "Posthoc directional ligand→receptor activity test per link community",
        long_about = "Tests whether a user-supplied directional ligand→receptor list shows\n\
                      elevated activity within each link community from a prior `pinto lc`\n\
                      run. Statistic is conditional entropy H(R_receiver | L_sender)\n\
                      over edges in each (batch × community × connected component),\n\
                      aggregated with inverse-variance weights across components and\n\
                      compared to a gene-swap null matched on (mean expression, global\n\
                      Moran's I) over the same edge graph.\n\n\
                      QUICK START:\n\n\
                      \x20 # After a `pinto lc` run at prefix `out/run1`:\n\
                      \x20 pinto lr-activity data.h5 -c coords.csv -o out/run1.lr \\\n\
                      \x20   --lc-prefix out/run1 --lr-pairs cellchat_pairs.tsv\n\n\
                      INPUTS:\n\n\
                      \x20 --lc-prefix   prefix of a prior `pinto lc` run (reads its\n\
                      \x20               {prefix}.link_community.parquet +\n\
                      \x20               {prefix}.coord_pairs.parquet, and back-fills\n\
                      \x20               the lr_activity path into {prefix}.metadata.json\n\
                      \x20               so `pinto plot` can auto-discover it).\n\
                      \x20 --lr-pairs    two-column TSV/CSV: ligand gene, receptor gene.\n\
                      \x20               Gene names are resolved against the data\n\
                      \x20               row-names; the resolved canonical names are\n\
                      \x20               persisted in the JSON sidecar.\n\n\
                      OUTPUTS:\n\n\
                      \x20 {out}.lr_activity.parquet — columns:\n\
                      \x20   batch, community, ligand, receptor,\n\
                      \x20   n_edges, n_components, ce_obs, ce_null_mean, ce_null_sd,\n\
                      \x20   z, p_empirical, q_bh\n\n\
                      \x20 {out}.lr_activity.json — sidecar consumed by `pinto plot`:\n\
                      \x20   summary stats per pair (with `ligand_resolved` /\n\
                      \x20   `receptor_resolved` row-name aliases) PLUS, for each\n\
                      \x20   significant pair (q_bh < --json-q-threshold), the\n\
                      \x20   participating-edge endpoints under a deduped per-stratum\n\
                      \x20   block. Disable with --emit-json=false.\n\n\
                      \x20 BATCH LABELS:\n\
                      \x20   `all`     single-batch run pseudo-label (no --batch-files).\n\
                      \x20   `pooled`  cross-batch pooled rows; emitted only when\n\
                      \x20             ≥ 2 real batches exist (would just duplicate\n\
                      \x20             the per-batch stats otherwise). BH is applied\n\
                      \x20             within each batch stratum."
    )]
    LrActivity(SrtLrActivityArgs),
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
        Commands::Plot(args) => {
            make_srt_plot(args)?;
        }
        Commands::LrActivity(args) => {
            fit_srt_lr_activity(args)?;
        }
    }

    Ok(())
}
