mod annotate;
mod cell_activity_graph_embedding;
mod gene_network;
mod link_community;
mod lr_activity;
mod plot;
mod propensity;
mod svd;
mod util;

#[cfg(test)]
mod test_support;

use annotate::{run_annotate, AnnotateArgs};
use cell_activity_graph_embedding::{
    fit_cell_activity_graph_embedding, CellActivityGraphEmbeddingArgs,
};
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
    long_about = "PINTO discovers cell-cell interaction patterns.\n\
                  It reads spatial transcriptomics.\n\
                  It detects link communities on cell-pair graphs.\n\n\
                  SUBCOMMANDS:\n\n\
                  \x20 lc    Link community model (recommended)\n\
                  \x20       Assigns each cell-cell edge to a community via collapsed\n\
                  \x20       Gibbs sampling on compressed all-gene edge profiles.\n\n\
                  \x20 dsvd  Delta-SVD model\n\
                  \x20       Cell-pair shared/difference analysis via Poisson-Gamma\n\
                  \x20       SVD on pseudobulk co-expression.\n\n\
                  \x20 prop  Propensity (standalone)\n\
                  \x20       Re-cut a cage/dsvd edge latent at a different K.\n\n\
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
        long_about = "Gene-level cell-cell interaction analysis by SVD.\n\
                      It uses shared and difference channels.\n\n\
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
                      \x20     z_e <- z_e / ||z_e||   (L2 normalize)\n\
                      \x20 11. Cut z into link communities, then take each\n\
                      \x20     cell's propensity as its incident-edge fraction\n\n\
                      --edge-cluster-method picks the cut.\n\
                      leiden is the default,\n\
                      deciding the count from --leiden-resolution.\n\
                      kmeans instead uses a fixed --n-edge-clusters.\n\
                      `pinto prop` re-cuts the same latent at a fixed K.\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effects (when multi-batch)\n\
                      - {out}.coord_pairs.parquet: cell pair coordinates\n\
                      - {out}.basis.parquet: SVD basis (2G x T)\n\
                      - {out}.latent.parquet: per-pair latent codes (E x T)\n\
                      - {out}.propensity.parquet: cell propensity (N x K)\n\
                      \x20 Columns: 0 .. K-1, cluster (argmax), entropy (Shannon, nats).\n\
                      - {out}.link_community.parquet: per-edge community labels\n\
                      - {out}.gene_community.parquet: gene-community Poisson-Gamma statistics (G x K).\n\
                      \x20 Rows are scaled by the NB Fisher-info weight\n\
                      \x20 w_g = 1 / (1 + π_g · s̄ · φ(μ_g)), which attenuates\n\
                      \x20 high-mean high-dispersion genes. There is no flag for it.\n\
                      - {out}.pinto.json: information-flow manifest used by\n\
                      \x20 `pinto plot` and `pinto lr-activity` (lists every parquet)."
    )]
    DeltaSvd(SrtDeltaSvdArgs),

    #[command(
        alias = "prop",
        about = "Estimate vertex propensity from edge clusters (standalone)",
        long_about = "Estimate vertex (cell) propensity scores from edge\n\
                      (cell-pair) cluster assignments.\n\n\
                      NOTE: cage and dsvd produce propensity and edge outputs inline.\n\
                      Use this subcommand to re-cut the same latent,\n\
                      or for separate expression data.\n\
                      It is the fixed-K path:\n\
                      --edge-cluster-method kmeans --n-edge-clusters K.\n\n\
                      Model:\n\
                      \x20 Given latent codes z_e [E x T] from cage or delta-svd:\n\
                      \x20   c_e = the pair's community, cut by leiden (default)\n\
                      \x20        or by kmeans, argmin_k ||z_e - centroid_k||\n\
                      \x20 For each vertex i:\n\
                      \x20   p_i[k] = |{e incident to i : c_e = k}| / degree(i)\n\
                      \x20 Optionally, cluster-specific gene expression:\n\
                      \x20   mu_{g,k} ~ Gamma(a0, b0) with pseudocount sums\n\n\
                      Algorithm:\n\
                      \x20 1. Load latent codes Z [E x T] from .latent.parquet\n\
                      \x20 2. Load cell pair names from .coord_pairs.parquet\n\
                      \x20 3. Cluster Z^T -> assignment c_e for each edge\n\
                      \x20 4. For each vertex i, count edges per cluster:\n\
                      \x20    p_i[k] = count(c_e=k for e incident to i) / deg(i)\n\
                      \x20 5. dominant_cluster[i] = argmax_k p_i[k]\n\
                      \x20 6. If expression data provided:\n\
                      \x20    weighted gene sums per cluster -> Poisson-Gamma\n\n\
                      Inputs (all passed by flag; there is no positional arg):\n\
                      - -z/--latent-data-file: .latent.parquet (from cage or delta-svd)\n\
                      - -e/--coord-pair-file: .coord_pairs.parquet (cell pair names)\n\
                      - -d/--expr-data-files: expression data (.zarr or .h5), optional\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity (N x K)\n\
                      \x20 Columns: C0 .. C{K-1}, cluster (argmax),\n\
                      \x20 entropy (Shannon, nats), plus optional coord trailer.\n\
                      - {out}.link_community.parquet: per-edge community labels\n\
                      - {out}.genes.parquet: cluster-specific gene expression (with -d/--expr-data-files).\n\
                      \x20 Rows are scaled by the NB Fisher-info weight\n\
                      \x20 w_g = 1 / (1 + π_g · s̄ · φ(μ_g)). There is no flag for it.\n\
                      - {out}.pinto.json: information-flow manifest used by\n\
                      \x20 `pinto plot`."
    )]
    Propensity(SrtPropensityArgs),

    #[command(
        alias = "lc",
        about = "Link community model via collapsed Gibbs sampling",
        long_about = "Link community detection for spatial transcriptomics.\n\n\
                      Each cell-cell edge is assigned to one of K communities.\n\
                      The assignment reads per-edge expression profiles.\n\
                      Per-cell soft membership follows from those labels.\n\n\
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
                      \x20   y_e = W^T(x_i + x_j), W = rows × --proj-dim Gaussian basis.\n\
                      \x20   Rows, not genes: on a {gene}/count/{spliced,unspliced}\n\
                      \x20   matrix there are two rows per gene, and both carry the\n\
                      \x20   same gene-level count filter and NB weight so a gene is\n\
                      \x20   never split across the projection.\n\
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
                      \x20 8. Extract cell propensity + gene-community statistics (+ cosine dictionary merge)\n\n\
                      See `pinto lc --help` for individual flag docs.\n\n\
                      OUTPUT FILES:\n\n\
                      \x20 {out}.propensity.parquet      Cell community membership [N × K]\n\
                      \x20                                Columns: 0 .. K-1, plus `entropy`\n\
                      \x20                                (Shannon entropy of each row, nats).\n\
                      \x20 {out}.gene_community.parquet      Gene-community rates [G × K]\n\
                      \x20                                (rows scaled by the NB Fisher-info weight\n\
                      \x20                                 w_g = 1/(1 + π_g·s̄·φ(μ_g)); no flag)\n\
                      \x20                                Keyed by the bare GENE name: on a matrix of\n\
                      \x20                                {gene}/count/{spliced,unspliced} rows the two\n\
                      \x20                                tracks are pooled. `cage` keeps its own copy of\n\
                      \x20                                this table on the matrix rows instead.\n\
                      \x20 {out}.link_community.parquet  Edge community assignments [E × 3]\n\
                      \x20 {out}.coord_pairs.parquet     Cell pair coordinates\n\
                      \x20 {out}.scores.parquet          Per-sweep diagnostics (level, sweep,\n\
                      \x20                                score, n_edges, total_mass,\n\
                      \x20                                mutual_information). `score` is the\n\
                      \x20                                plug-in Poisson DC-SBM log-likelihood\n\
                      \x20                                Σ_kg f(D_kg) − Σ_k f(V_k) with\n\
                      \x20                                f(x)=x·ln x, where D_kg is the\n\
                      \x20                                edge-weighted gene degree in community k\n\
                      \x20                                and V_k = Σ_g D_kg is its volume\n\
                      \x20                                (equivalently −Σ_k V_k · H(p_k), nats).\n\
                      \x20                                Higher = better; `score/total_mass`\n\
                      \x20                                is the\n\
                      \x20                                mass-weighted mean per-community\n\
                      \x20                                log-likelihood per edge unit.\n\
                      \x20 {out}.delta.parquet           Batch effects (multi-sample only)\n\
                      \x20 {out}.gene_graph.parquet      Gene-gene pairs (gene-pair mode only)\n\
                      \x20 {out}.L{l}.*.parquet          Per-cascade-level outputs (unless --no-level-outputs)\n\
                      \x20 {out}.draft.*.parquet         Pre-merge fine partition (when dictionary merge collapsed)\n\
                      \x20 {out}.dict_merges.parquet     Cosine merge tree over the gene-community dictionary\n\
                      \x20 {out}.dict_merges.cut.parquet Fine→super community remap from --merge-cut\n\
                      \x20 {out}.pinto.json           Information-flow manifest:\n\
                      \x20                                lists every parquet, level tags,\n\
                      \x20                                dict-merge presence, and (when set by\n\
                      \x20                                lr-activity) the lr_activity JSON.\n\
                      \x20                                Pass this path to `pinto plot --from`\n\
                      \x20                                or `pinto lr-activity --lc-prefix`."
    )]
    LinkCommunity(SrtLinkCommunityArgs),

    #[command(
        alias = "cge",
        about = "Activity-gated cell-graph embedding (cage)",
        long_about = "Learn per-cell embeddings on the spatial cell-cell graph.\n\
                      cage visits one gene at a time.\n\
                      Each gene defines a per-cell activity vector.\n\
                      That vector gates a shared multi-scale cell-cell hierarchy.\n\n\
                      A per-gene per-dim selection is SAMPLED by block Gibbs.\n\
                      It runs against a pseudobulk Poisson.\n\
                      The resulting inclusion probabilities become DROP RATES.\n\
                      A fresh z ~ Bern(pip) is drawn each epoch.\n\
                      Every epoch therefore trains a different sub-network.\n\
                      The rates re-estimate against the live embedding.\n\
                      See --selection-refresh-epochs.\n\n\
                      --embedding-dim is an UPPER BOUND.\n\
                      The stick-breaking (IBP) prior orders dims by admittance.\n\
                      The data decides how many are really used.\n\
                      Chain levels differ only in their negative pools.\n\
                      This is embedding-only. There is no count decoder.\n\n\
                      NOTE --n-hvg no longer subsets the trained gene axis.\n\
                      It weights the random projection instead.\n\
                      That projection builds the coarsening hierarchy.\n\
                      senna bge and senna gem do the same.\n\
                      Every gene is trained and present in every output table.\n\
                      Use --genes-per-epoch to cap per-epoch cost instead.\n\n\
                      SPLICE CHANNELS are recognised on the feature axis.\n\
                      Rows named {gene}/count/spliced pair with their\n\
                      {gene}/count/unspliced counterpart.\n\
                      A gene's two rows are ONE gene everywhere the model fits.\n\
                      Their counts are summed before the log1p activity.\n\
                      Gene-side output tables are keyed by the bare gene name.\n\
                      {out}.gene_community.parquet stays on the matrix rows,\n\
                      so it still lists a gene's two channels separately.\n\
                      --n-hvg counts ROWS, then widens to whole genes,\n\
                      so a gene is never half-weighted in the projection.\n\
                      A matrix mixing channel rows with plain rows is rejected.\n\
                      A {gene}/count/total row is the usual cause.\n\n\
                      With both tracks present, the sampler gains a second gate.\n\
                      It samples a nascent DEVIATION on top of each gene loading:\n\
                      spliced scores the loading, unspliced the loading plus delta.\n\
                      That is `senna gem`'s sign, recorded as delta_base in the manifest.\n\
                      `senna gem-encoder` uses the opposite base, so the two\n\
                      delta tables are not comparable without reading that field.\n\
                      A gene needs counts on BOTH tracks to identify delta at all.\n\
                      Genes that do not are written NaN, never a number.\n\
                      The manifest reports how many qualified.\n\
                      See --independent-delta-gate for the un-nested arm.\n\n\
                      After training, every CELL PAIR is projected.\n\
                      The target is the frozen gene embedding.\n\
                      Its pooled counts x_gu + x_gv are fit by Poisson MAP.\n\
                      That gives a per-pair latent e_uv.\n\
                      The solve is rayon-parallel, one D+1 problem per pair,\n\
                      with a sampled log-partition.\n\n\
                      Clustering those pairs gives link communities.\n\
                      A cell's propensity is its incident-edge fraction.\n\
                      That is the same definition `lc` and `dsvd` use.\n\
                      --edge-cluster-method picks the cut.\n\
                      leiden is the default,\n\
                      deciding the count from --leiden-resolution.\n\
                      kmeans instead uses a fixed --n-edge-clusters.\n\n\
                      Outputs:\n\
                      \x20 {out}.cell_embedding.parquet  cell × embedding_dim\n\
                      \x20 {out}.cell_bias.parquet       per-cell scalar\n\
                      \x20 {out}.feature_embedding.parquet  feature × embedding_dim\n\
                      \x20 {out}.feature_posterior_mean.parquet  feature × dim (E[z*beta])\n\
                      \x20 {out}.delta_feature_embedding.parquet gene × dim (E[z*delta])\n\
                      \x20                              (splice-channelized input only)\n\
                      \x20 {out}.delta_selection.parquet  gene × dim delta inclusion\n\
                      \x20                              (splice-channelized input only)\n\
                      \x20 {out}.pseudobulk_cells.parquet  cell × (coords, super-cell, e_pb)\n\
                      \x20                              (--gate-mode sampled only)\n\
                      \x20 {out}.gene_bias.parquet       per-gene scalar\n\
                      \x20 {out}.coord_pairs.parquet     cell pair list, tagged by kind\n\
                      \x20 {out}.latent.parquet          cell pair × embedding_dim\n\
                      \x20 {out}.propensity.parquet      cell × K, + cluster, entropy\n\
                      \x20 {out}.link_community.parquet  per-edge community\n\
                      \x20 {out}.gene_community.parquet  gene × K Poisson-Gamma rates\n\
                      \x20 {out}.scores.parquet          per-epoch loss trace\n\
                      \x20 {out}.fisher_weights.parquet  per-ROW NB precisions w_r\n\
                      \x20 {out}.delta.parquet           batch effects (multi-batch only)\n\
                      \x20 {out}.pinto.json           manifest"
    )]
    Cage(CellActivityGraphEmbeddingArgs),

    #[command(
        visible_alias = "cage-annotate",
        about = "Marker-set cell-type annotation by projection (any embedding run)",
        long_about = "Firm cell-type annotation via the shared term-ORA core.\n\
                      This is the embedding-grounded twin of `senna annotate-by-projection` and `senna annotate-gem`.\n\
                      \n\
                      Each marker-defined type is embedded as a centroid.\n\
                      That centroid is an IDF-weighted mean.\n\
                      It averages the type's marker feature embeddings.\n\
                      Every cell is hard-assigned to its nearest centroid.\n\
                      Distance outliers are then pruned. The cells are Leiden-clustered.\n\
                      Each cluster × term is then tested.\n\
                      The test is hypergeometric over-representation, permutation-calibrated.\n\
                      --obo adds optional TreeBH Cell-Ontology calling.\n\
                      \n\
                      Inputs are `{prefix}.feature_embedding.parquet` and `{prefix}.cell_embedding.parquet`.\n\
                      Any pinto embedding run supplies them, `cage` among them.\n\
                      To annotate anything else,\n\
                      point --feature-embedding and --cell-embedding at explicit paths.\n\
                      \n\
                      Outputs follow the shared per-cell contract:\n\
                      {out}.annot.{parquet,membership.tsv,argmax.tsv}.\n\
                      The cluster × term p/q/Q matrices ship alongside."
    )]
    Annotate(AnnotateArgs),

    #[command(
        alias = "p",
        about = "Plot spatial scatter from pinto lc/dsvd/prop outputs",
        long_about = "Render publication-quality PDFs, and SVG or PNG.\n\
                      It reads outputs from lc, dsvd, prop and cage.\n\
                      Markers default to tightly tiling flat-top hexagons.\n\
                      Their size adapts to plot density.\n\n\
                      INPUT (--from):\n\n\
                      \x20 Pass either a `{prefix}.pinto.json` (preferred,\n\
                      \x20 carries level list, dict-merge presence, and any lr_activity\n\
                      \x20 JSON) or a bare `{prefix}` (auto-globs *.parquet).\n\n\
                      PER-LEVEL × PER-CORE PLOTS (default = PDF only):\n\n\
                      Final level (full suite):\n\
                      \x20 propensity/{level}.argmax.propensity.pdf  size ∝ propensity, color = argmax\n\
                      \x20 propensity/{level}.community{k}.pdf       per-community soft-membership\n\
                      \x20 mesh/{level}.pdf                          cell-cell edges (lc only)\n\
                      \x20 markers/{level}.community{k}.{gene}.pdf   log1p expr heatmap\n\
                      \x20                                           with that community's hull outline\n\
                      Intermediate `L*` levels:\n\
                      \x20 propensity/{level}.argmax.propensity.pdf  only\n\
                      Draft level:\n\
                      \x20 propensity/{level}.argmax.propensity.pdf + mesh/{level}.pdf\n\n\
                      Pass --svg / --png to also emit those formats.\n\n\
                      OPT-IN: --show-interfaces (per (level, core)):\n\n\
                      \x20 interfaces.pdf  All cells; radius scaled by entropy\n\
                      \x20                 quantile rank (within core), single dark\n\
                      \x20                 gray fill. High-entropy boundary cells\n\
                      \x20                 stand out as full hex tiles, low-entropy\n\
                      \x20                 interior cells fade to 0.\n\
                      \x20 interfaces.tsv  Per focal cell: dominant community,\n\
                      \x20                 1- and 2-hop neighbor mix, top-N marker\n\
                      \x20                 genes per neighbor community.\n\
                      \x20 Tunables: --entropy-quantile, --neighborhood-hops,\n\
                      \x20            --max-interface-cells, --interface-top-genes.\n\n\
                      LR-ACTIVITY OVERLAY (auto-discovered):\n\n\
                      \x20 When the .pinto.json carries an `outputs.lr_activity`\n\
                      \x20 path (set automatically by `pinto lr-activity`), one\n\
                      \x20 PDF is written per (core × significant LR pair):\n\
                      \x20   lr.core{batch}.lr.B{batch}.C{community}.{L}-{R}.pdf\n\
                      \x20 Layout:\n\
                      \x20   - Faint hex tiling of all core cells (tissue context)\n\
                      \x20   - Per-community CC convex hulls (thin gray outlines)\n\
                      \x20     for the pair's community only\n\
                      \x20   - Quiver of L→R arrows along edges incident to a\n\
                      \x20     boundary cell (1-hop expanded).\n\
                      \x20     With --lr-color-mode=coexpr, arrow direction is the\n\
                      \x20     ANNOTATED role wherever a contact realizes exactly\n\
                      \x20     one: it runs from the ligand-carrying cell to the\n\
                      \x20     receptor-carrying cell (bookkeeping, not inference;\n\
                      \x20     mutual contacts have no side and keep the display\n\
                      \x20     heuristic below). In the other colour modes,\n\
                      \x20     direction comes from per-edge L+R expression argmax\n\
                      \x20     (needs --data), a display heuristic only.\n\
                      \x20   - Color: the default --lr-color-mode=log-ratio maps\n\
                      \x20     log((R+1)/(L+1)) on a red↔blue ramp.\n\
                      \x20     With --lr-color-mode=coexpr, the ramp shows co-detection\n\
                      \x20     instead, centered on the per-pair edge mean:\n\
                      \x20     red where both genes are detected across the contact,\n\
                      \x20     blue where only one side is.\n\
                      \x20     That is the same co-detection notion\n\
                      \x20     the `lra --edge-scores-only` table is built on\n\
                      \x20     (a shared concept; plot does not read that table).\n\
                      \x20 Tunables: --lr-top-pairs, --lr-commit-threshold,\n\
                      \x20            --no-lr-overlay, --lr-coexpr-bins,\n\
                      \x20            --lr-activity-json (override path).\n\n\
                      Levels are `final`, `L0..Ln` and `draft`.\n\
                      There is one core per batch label.\n\
                      Batch labels are read from coord_pairs.parquet.\n\n\
                      Outlier handling is robust by default: coordinate bounds,\n\
                      color scales, and size scales all use percentile clipping\n\
                      (see --coord-clip, --expr-clip).\n\n\
                      A JSON manifest listing every emitted file is written to\n\
                      {out}.plot.manifest.json."
    )]
    Plot(SrtPlotArgs),

    #[command(
        aliases = ["lra", "test-lr"],
        about = "Posthoc ligand-receptor co-activity test per link community",
        long_about = "Tests a user-supplied ligand-receptor list.\n\
                      It asks whether each pair is co-active along the contacts of a\n\
                      link community from a prior lc run, one community at a time.\n\
                      The statistic is symmetric in the pair: both orientations of\n\
                      every within-community edge are counted, so no endpoint plays a\n\
                      privileged role, and edges bridging two communities sit out.\n\n\
                      DESIGN:\n\
                      \x20 1. Cells are collapsed into pseudobulk samples =\n\
                      \x20    (batch × propensity-bin), where the propensity bin is the\n\
                      \x20    sign-LSH binary code of an SVD'd random projection of gene\n\
                      \x20    expression (data-beans-alg::binary_sort_columns).\n\
                      \x20 2. Each cell carries soft membership over the link communities:\n\
                      \x20    the fraction of its within-community edge instances in each.\n\
                      \x20 3. Per (community, sample) we accumulate membership-weighted\n\
                      \x20    gene sums for the LR genes: one pseudobulk profile per\n\
                      \x20    sample per community, with weight w = membership mass.\n\
                      \x20 4. Statistic per (batch, community, LR pair): weighted covariance\n\
                      \x20    of `log1p(w_g · pb_mean)` between L and R across samples,\n\
                      \x20    sample-weighted by w. Per-gene `w_g` are NB-Fisher-info\n\
                      \x20    weights (same as propensity / lc).\n\
                      \x20 5. Null: sample-level permutation of L within propensity-stratified\n\
                      \x20    buckets (top --shuffle-stratify-dim bits of the propensity\n\
                      \x20    code). The same shuffle σ_k is applied to every pair so\n\
                      \x20    cross-pair dependence is preserved.\n\
                      \x20 6. Inference: Efron-Tibshirani restandardize stat_obs against\n\
                      \x20    per-stratum (median, MAD) of stat_obs (z_re / p_re), then\n\
                      \x20    Westfall-Young single-step minP for FWER (fwer_wy).\n\n\
                      QUICK START:\n\n\
                      \x20 # Shortest form, reading inputs from a prior pinto lc .pinto.json:\n\
                      \x20 pinto lra --from out/run1.pinto.json --lr-pairs cellchat_pairs.tsv\n\n\
                      \x20   `--from <.pinto.json>` auto-fills `--lc-prefix`, `--out` (=\n\
                      \x20   `<prefix>.lra`), and the positional data files from\n\
                      \x20   the metadata. Any of those passed explicitly on the CLI win.\n\n\
                      \x20 # Long form, same effect, fully explicit:\n\
                      \x20 pinto lr-activity data.h5 -o out/run1.lr \\\n\
                      \x20   --lc-prefix out/run1 --lr-pairs cellchat_pairs.tsv\n\n\
                      INPUTS:\n\n\
                      \x20 --lc-prefix   prefix of a prior `pinto lc` run (reads its\n\
                      \x20               {prefix}.link_community.parquet +\n\
                      \x20               {prefix}.coord_pairs.parquet, and back-fills\n\
                      \x20               the lr_activity path into {prefix}.pinto.json\n\
                      \x20               so `pinto plot` can auto-discover it).\n\
                      \x20 --lr-pairs    two-column TSV/CSV: ligand gene, receptor gene.\n\
                      \x20               Gene names are resolved against the data\n\
                      \x20               row-names; the resolved canonical names are\n\
                      \x20               persisted in the JSON sidecar.\n\n\
                      KEY KNOBS:\n\n\
                      \x20 --propensity-dim         d for binary-sort propensity codes\n\
                      \x20                          (default 10 → ≤1024 samples per batch).\n\
                      \x20 --shuffle-stratify-dim   top bits of propensity used for\n\
                      \x20                          permutation buckets (default 4 → 16\n\
                      \x20                          buckets; 0 disables stratification).\n\
                      \x20 --n-permutations         number of sample shuffles (default 1000).\n\n\
                      OUTPUTS:\n\n\
                      \x20 {out}.lr_activity.parquet, columns:\n\
                      \x20   batch, community, ligand, receptor, n_samples,\n\
                      \x20   stat_obs (weighted covariance of log1p(w·pb)),\n\
                      \x20   null_mean, null_sd, z, p_empirical, p_z, z_re, p_re,\n\
                      \x20   fwer_wy.\n\
                      \x20   z_re/p_re: Efron-Tibshirani restandardization of\n\
                      \x20     stat_obs against per-stratum (median, MAD).\n\
                      \x20   fwer_wy: Westfall-Young single-step minP\n\
                      \x20     (joint sample permutation across pairs in a stratum;\n\
                      \x20     FWER-controlled).\n\
                      \x20   community: the link community id. It joins directly\n\
                      \x20     against link_community.parquet and\n\
                      \x20     propensity.parquet.\n\
                      \x20   The statistic is symmetric in the pair; no direction\n\
                      \x20     may be read off any row.\n\n\
                      \x20 {out}.lr_activity.json, the sidecar consumed by `pinto plot`:\n\
                      \x20   summary stats per pair (with `ligand_resolved` /\n\
                      \x20   `receptor_resolved` row-name aliases) PLUS, for each\n\
                      \x20   significant pair (fwer_wy < --json-fwer-threshold), the\n\
                      \x20   participating-edge endpoints under a deduped per-stratum\n\
                      \x20   block. Disable with --emit-json=false.\n\n\
                      \x20 BATCH LABELS:\n\
                      \x20   `all`     single-batch run pseudo-label (no --batch-files).\n\
                      \x20   `pooled`  cross-batch pooled rows; emitted only when\n\
                      \x20             ≥ 2 real batches exist (would just duplicate\n\
                      \x20             the per-batch stats otherwise). WY shuffles are\n\
                      \x20             still bucketed per (batch, propensity-bin).\n\n\
                      EDGE SCORES (--edge-scores-only):\n\n\
                      \x20 Skips the test entirely and writes {out}.lr_scores.parquet,\n\
                      \x20 one row per (batch, community, ligand, receptor).\n\
                      \x20 The estimand: the probability that ligand and receptor are\n\
                      \x20 co-detected across a physical contact of that community,\n\
                      \x20 BEYOND each side's independent activity. Every contact\n\
                      \x20 contributes both orientations; each instance is classified\n\
                      \x20 by endpoint detection into a 2x2 table, and the score is\n\
                      \x20 the posterior log odds ratio under a Jeffreys +1/2 prior:\n\
                      \x20   log_or    = ln[(n11+.5)(n00+.5)/((n10+.5)(n01+.5))]\n\
                      \x20   log_or_se = sqrt(sum of 1/(cell+.5))\n\
                      \x20 log_or is symmetric in the pair by construction.\n\n\
                      \x20 Direction is reported as CONFIGURATION, not inferred:\n\
                      \x20 the pair file names the ligand, so a contact where the\n\
                      \x20 roles sit on opposite cells identifies its ligand side\n\
                      \x20 outright. Per row:\n\
                      \x20   n_oneway  contacts with the ligand on exactly one side\n\
                      \x20   n_mutual  contacts co-detected both ways (no side)\n\
                      \x20   (the 2x2's n11 counts oriented instances,\n\
                      \x20    so n11 = 2*n_mutual + n_oneway)\n\
                      \x20   role_purity  mean over active cells of\n\
                      \x20     |sent - received| / (sent + received):\n\
                      \x20     1 = cells specialize as sender or receiver here,\n\
                      \x20     0 = every cell plays both roles equally.\n\
                      \x20 These are configuration facts from annotated roles.\n\
                      \x20 They say which cells carry which side,\n\
                      \x20 never that signalling flowed, and a static snapshot\n\
                      \x20 cannot say more. Spot-level platforms mix cells within\n\
                      \x20 a spot and deflate role_purity by construction;\n\
                      \x20 compare it across cores of one platform,\n\
                      \x20 never across platforms.\n\n\
                      \x20 The margins ship beside it:\n\
                      \x20 lig_rate and rec_rate are the detection rates\n\
                      \x20 of each side over the contact instances.\n\
                      \x20 They are rates over contacts, not cell fractions:\n\
                      \x20 a cell counts once per contact it participates in.\n\
                      \x20 Use them as covariates to isolate the interaction;\n\
                      \x20 they are activity phenotypes in their own right.\n\
                      \x20 No test and no null: these are descriptive phenotypes.\n\n\
                      \x20 Pivot to a batch x (pair, community) matrix in R:\n\
                      \x20   dcast(dt, batch ~ ligand + receptor + community,\n\
                      \x20         value.var = \"log_or\")\n\n\
                      \x20 Caveats. A prior-dominated pair is NaN in both columns:\n\
                      \x20 no co-detection observed and none expected\n\
                      \x20 means the row is unmeasurable, not zero.\n\
                      \x20 The SE counts each physical contact once,\n\
                      \x20 but contacts sharing a cell are still correlated,\n\
                      \x20 so treat log_or_se as a relative precision weight,\n\
                      \x20 not a calibrated interval.\n\
                      \x20 A fully saturated table grows with the contact count;\n\
                      \x20 compare such rows through their SE, never by magnitude.\n\
                      \x20 Filter or precision-weight on log_or_se and n_edges\n\
                      \x20 downstream (no threshold is applied here),\n\
                      \x20 and keep mean_log_depth in the covariate set.\n\
                      \x20 Community ids come from the `pinto lc` fit,\n\
                      \x20 so the lc artifacts are part of the phenotype definition.\n\
                      \x20 Freeze them alongside any analysis of these scores."
    )]
    LrActivity(SrtLrActivityArgs),
}

/// Expand `pinto lra --from <.pinto.json>` into the full positional /
/// flag form clap expects.
///
/// The user-friendly `--from` is not a real `clap` arg on `pinto lra` — it's
/// preprocessed here so the rest of the CLI surface (`SrtInputArgs` and
/// friends) stays unchanged. When `--from foo.pinto.json` is detected
/// after the `lra` / `lr-activity` / `test-lr` subcommand:
///
///   - `--lc-prefix`, `--out`, and the positional `data_files` are
///     injected from the metadata when not already on the CLI;
///   - `--from <path>` is removed before clap sees it.
///
/// Anything the user explicitly passed wins: only missing fields are filled.
fn expand_lra_from_metadata(mut args: Vec<String>) -> anyhow::Result<Vec<String>> {
    const LRA_NAMES: &[&str] = &["lra", "lr-activity", "test-lr"];

    let Some(lra_pos) = args.iter().position(|a| LRA_NAMES.contains(&a.as_str())) else {
        return Ok(args);
    };

    let from_pos = (lra_pos + 1..args.len()).find(|&i| {
        let a = &args[i];
        a == "--from" || a.starts_with("--from=") || a == "-f"
    });
    let Some(from_pos) = from_pos else {
        return Ok(args);
    };

    let meta_path: String = if let Some(rest) = args[from_pos].strip_prefix("--from=") {
        let p = rest.to_string();
        args.drain(from_pos..from_pos + 1);
        p
    } else {
        if from_pos + 1 >= args.len() {
            anyhow::bail!("--from requires a path argument");
        }
        let p = args[from_pos + 1].clone();
        args.drain(from_pos..from_pos + 2);
        p
    };

    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(&meta_path))?;

    // Inspect what's already on the CLI (post-drain) so we don't clobber
    // explicit user overrides.
    let (has_lc_prefix, has_out, has_positional) = {
        let tail = &args[lra_pos + 1..];
        let has_flag = |needles: &[&str]| -> bool {
            tail.iter().any(|a| {
                needles
                    .iter()
                    .any(|n| a == n || a.starts_with(&format!("{n}=")))
            })
        };
        let mut positional = false;
        let mut i = 0;
        while i < tail.len() {
            let a = &tail[i];
            if a.starts_with('-') {
                // "--flag value" pair → skip both. "--flag=value" or short bool → skip one.
                if !a.contains('=') && i + 1 < tail.len() && !tail[i + 1].starts_with('-') {
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                positional = true;
                break;
            }
        }
        (
            has_flag(&["--lc-prefix"]),
            has_flag(&["--out", "-o"]),
            positional,
        )
    };

    if !has_lc_prefix {
        args.push("--lc-prefix".to_string());
        args.push(meta.prefix.clone());
    }
    if !has_out {
        args.push("--out".to_string());
        args.push(format!("{}.lra", meta.prefix));
    }
    if !has_positional {
        match meta.data_files.as_ref() {
            Some(files) if !files.is_empty() => {
                for f in files {
                    args.push(f.clone());
                }
            }
            _ => anyhow::bail!(
                ".pinto.json {meta_path} has no data_files; pass them as positional args, \
                 or re-run pinto lc/dsvd to regenerate metadata"
            ),
        }
    }

    Ok(args)
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let argv = expand_lra_from_metadata(std::env::args().collect())?;
    let cli = Cli::parse_from(argv);

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
        Commands::Cage(args) => {
            fit_cell_activity_graph_embedding(args)?;
        }
        Commands::Annotate(args) => {
            run_annotate(args)?;
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
