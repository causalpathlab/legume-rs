#![allow(
    // `embed_common` is deliberately shaped as a prelude module.
    clippy::wildcard_imports,
    // Counts / dimensions / IDs routinely cross usize↔f32/f64; the
    // values always fit and the casts are intentional.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // Not every `Result`-returning helper needs a `# Errors` stanza.
    clippy::missing_errors_doc,
    // Training / fit functions are naturally long; splitting them for
    // a line-count lint would fragment logical phases.
    clippy::too_many_lines,
    // CLI struct fields intentionally share a `phate_` prefix so the
    // clap flag names (`--phate-t`, `--phate-knn`) are self-documenting.
    clippy::struct_field_names,
    // Config / args structs are typically built once at the call site
    // and consumed — passing by value is part of the ownership-forward
    // API style used across the crate.
    clippy::needless_pass_by_value,
    // Local `use`/`const`/`enum` items scoped to where they're relevant
    // read more naturally than hoisting them to the top of a function.
    clippy::items_after_statements,
    // Binding-name similarity is noisy for domain-driven names like
    // `dist`/`d`, `stress`/`prev_stress`.
    clippy::similar_names,
    // Math code uses short names (`n`, `i`, `j`, `k`, `d`) where the
    // semantics come from surrounding indices (row/col/dim).
    clippy::many_single_char_names,
)]

mod anchor_common;
mod annotate;
mod cluster;
mod cluster_aggregation;
mod cluster_bhc;
mod cnv_pseudobulk;
mod embed_common;
mod empirical_dict;
mod eval_topic;
mod fit_clustering;
mod fit_indexed_topic;
mod fit_joint_topic;
mod fit_pseudotime;
mod fit_topic;
mod geometry;
mod hvg;
mod logging;
mod marker_support;
mod output_helpers;
mod postprocess;
mod predict;
mod predict_tmle;
mod principal_graph;
mod refine_weighting;
mod run_manifest;
mod senna_input;
mod svd;
mod topic;
mod tree_layout;

use annotate::{annotate_run, AnnotateArgs};
use embed_common::*;
use eval_topic::*;
use fit_clustering::*;
use fit_indexed_topic::*;
use fit_joint_topic::*;
use fit_pseudotime::{run_pseudotime, PseudotimeArgs};
use fit_topic::*;
use postprocess::*;
use predict::{predict_model, PredictArgs};
use svd::*;

use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('@', &"@".bright_yellow().to_string())
        .replace('◠', &"◠".bright_yellow().to_string())
        .replace('◡', &"◠".bright_yellow().to_string())
        .replace('_', &"_".bright_yellow().to_string())
        .replace('(', &"(".bright_yellow().to_string())
        .replace(')', &")".bright_yellow().to_string())
        .replace('{', &"{".bright_yellow().to_string())
        .replace('}', &"}".bright_yellow().to_string())
        .replace('\\', &"\\".bright_yellow().to_string())
        .replace('/', &"/".bright_yellow().to_string())
        .replace('|', &"|".green().to_string())
        .replace('‖', &"‖".green().to_string())
        .replace('~', &"~".truecolor(101, 67, 33).to_string())
}

fn print_logo() {
    let intro = [
        "",
        "",
        "SENNA",
        "Stochastic data Embedding with",
        "Nearest Neighbourhood Adjustment",
        "",
    ];

    let logo_lines: Vec<_> = LOGO.lines().collect();
    let max_lines = logo_lines.len().max(intro.len());

    for i in 0..max_lines {
        let logo_part = if i < logo_lines.len() {
            colorize_logo_line(logo_lines[i])
        } else {
            " ".repeat(13) // width of logo box
        };

        let text_part = if i < intro.len() { intro[i] } else { "" };

        println!("{logo_part}  {text_part}");
    }
    println!();
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "SENNA — single-cell embedding (SVD / topic), annotation, trajectory, and plotting.",
    long_about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment.\n\n\
                  Input: sparse backends in `.zarr` or `.h5` (convert from Matrix Market\n\
                  with `data-beans from-mtx`).\n\n\
                  Pipeline (each step writes its outputs back to the run manifest\n\
                  `{prefix}.senna.json`, so downstream commands need no extra flags):\n\n  \
                  1. Train embedding   senna topic | indexed-topic | svd\n                       \
                                       senna joint-topic | joint-svd       (multi-modality)\n  \
                  2. Held-out inference senna predict                       (apply trained model)\n  \
                  3. Cluster cells     senna clustering --from run.senna.json\n  \
                  4. Annotate cells    senna annotate   --from run.senna.json -m markers.tsv\n  \
                  5. Trajectory        senna pseudotime --from run.senna.json\n  \
                  6. 2D layout         senna layout {phate|tsne|umap} --from run.senna.json\n  \
                  7. Scatter plot      senna plot       --from run.senna.json\n  \
                  8. Topic diagnostics senna plot-topic --from run.senna.json\n\n\
                  `senna plot` auto-runs steps 3 + 6 on demand."
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Verbose logging")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    // ─────────── 1. Train embedding (writes the run manifest) ───────────
    #[command(
        about = "Train: topic-model embedding (VAE).",
        long_about = "Probabilistic topic-model embedding.\n\n\
                      Stages: (1) batch-aware pseudobulk collapsing, (2) encoder-decoder\n\
                      VAE via SGD, (3) per-cell topic inference. Decoders: multinomial,\n\
                      negative-binomial, vMF (combine via comma-separated --decoder).\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.metadata.json, {out}.senna.json (run manifest)."
    )]
    Topic(TopicArgs),

    #[command(
        about = "Train: topic-model with adaptive top-K feature windows (~4-7× faster).",
        long_about = "Same pipeline as `topic`, but encoder/decoder operate on a per-cell\n\
                      top-K feature window instead of the full D × K dictionary.\n\
                      Useful for very large gene sets.\n\n\
                      Writes the same artifacts as `topic`.",
        visible_alias = "itopic"
    )]
    IndexedTopic(IndexedTopicArgs),

    #[command(
        about = "Train: Nyström SVD embedding.",
        long_about = "Three stages: (1) batch-aware pseudobulk collapsing, (2) randomized SVD,\n\
                      (3) per-cell Nyström projection.\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.senna.json."
    )]
    Svd(SvdArgs),

    #[command(
        about = "Train: joint topic model across modalities (independent or delta decoder).",
        long_about = "Joint topic-model embedding over a stack of modalities sharing cells.\n\
                      Data files are a row-major (modality × batch) table; -m sets the\n\
                      modality-row count.\n\n\
                      Decoder types:\n  \
                      independent — each modality keeps its own dictionary; features may differ.\n  \
                      delta       — shared base + cumulative chain deltas\n              \
                                    (modality m = softmax(z @ (W_base + Σ δ_1..m));\n              \
                                    requires shared features across modalities).\n\n\
                      Writes {out}.latent.parquet, {out}.senna.json."
    )]
    JointTopic(JointTopicArgs),

    #[command(
        about = "Train: joint Nyström SVD across modalities.",
        long_about = "Joint SVD over a stack of modalities sharing cells. Data files form\n\
                      a row-major (modality × batch) table; -m sets the modality-row count.\n\
                      Cells must be shared; features may differ.\n\n\
                      Writes {out}.latent.parquet, {out}.senna.json."
    )]
    JointSvd(JointSvdArgs),

    // ─────────── 2. Held-out inference ───────────
    #[command(
        about = "Apply a trained topic / indexed-topic model to held-out data.",
        long_about = "Latent inference + per-cell predictive log-likelihood on a separate\n\
                      backend file. Auto-dispatches dense vs indexed via metadata.json.\n\
                      Handles gene-set misalignment via flexible name matching and\n\
                      re-estimates per-batch delta from the frozen dictionary.\n\n\
                      Latent modes: encoder-only (default), encoder+refine, decoder-only."
    )]
    Predict(PredictArgs),

    #[command(about = "[deprecated] Alias for `senna predict`.")]
    EvalTopic(EvalTopicArgs),

    // ─────────── 3. Cluster / annotate / trajectory (run on a manifest) ───────────
    #[command(
        about = "Cluster cells on the manifest's latent (kmeans / leiden / hsblock).",
        long_about = "Cluster cells using `manifest.outputs.latent`.\n\n\
                      Algorithms:\n  \
                      kmeans  — requires -k.\n  \
                      leiden  — graph-based, auto-k.\n  \
                      hsblock — hierarchical SBM (2^(depth-1) clusters).\n\n\
                      Writes {out}.clusters.parquet and updates `manifest.cluster.clusters`."
    )]
    Clustering(ClusteringArgs),

    #[command(
        about = "Annotate cells via cluster-level marker enrichment.",
        long_about = "Pipeline: (re)cluster on the manifest's latent (Leiden if no clusters\n\
                      exist) → NB-Fisher-adjusted per-cluster mean expression (streamed\n\
                      from raw counts) → weighted-KS marker enrichment with cross-cluster\n\
                      simplex normalization (housekeeping suppression) → softmax-normalized\n\
                      per-cluster Q matrix → cluster-broadcast per-cell labels.\n\n\
                      Usage: senna annotate --from run.senna.json -m markers.tsv -o out\n\n\
                      Updates `manifest.annotate.{argmax,annotation,...}` so subsequent\n\
                      `senna plot` runs colour cells by predicted cell type by default.\n\
                      Writes {out}.argmax.tsv, {out}.annotation.parquet, {out}.cluster_*.parquet."
    )]
    Annotate(AnnotateArgs),

    #[command(
        about = "Pseudotime via Monocle-3-style principal graph (SimplePPT) on the latent.",
        long_about = "Port of Mao et al. 2015 SimplePPT applied to `manifest.outputs.latent`.\n\n\
                      (1) k-means init K centroids,\n\
                      (2) iterate: soft-assign cells → MST over centroids → solve\n    \
                          (D_R + γL) Y = R^T Z for centroid coords,\n\
                      (3) project each cell onto its nearest tree edge,\n\
                      (4) Dijkstra geodesic from a chosen root → pseudotime.\n\n\
                      Outputs {out}.pseudotime.parquet and {out}.principal_graph.{nodes,edges}.parquet."
    )]
    Pseudotime(PseudotimeArgs),

    // ─────────── 4. Layout + plotting ───────────
    #[command(
        about = "2D layout of cells (tsne / umap / phate) over batch-corrected pseudobulks.",
        long_about = "Builds PBs via batch-corrected multi-level collapsing, computes\n\
                      PB-PB cosine similarity on log1p-CPM gene vectors, lays out via\n\
                      the chosen method, and projects every cell via Nyström.\n\n\
                      Updates `manifest.layout.{cell_coords, pb_coords, pb_gene_mean}` so\n\
                      `senna plot --from ...` picks the layout up automatically.\n\n\
                      Pick a method: `senna layout {phate|tsne|umap} --from run.senna.json`.",
        visible_alias = "lay",
        subcommand_required = true,
        arg_required_else_help = true
    )]
    Layout {
        #[command(subcommand)]
        cmd: LayoutCmd,
    },

    #[command(
        about = "Publication scatter plot from a run manifest (SVG/PNG/PDF).",
        long_about = "`senna plot --from run.senna.json` reads cell_coords, topics,\n\
                      annotation, clusters, labels, and palette from the manifest and\n\
                      renders a 300-dpi rasterized scatter with vector text labels.\n\n\
                      Auto-fills missing pieces:\n  \
                      • no `layout.cell_coords` → runs `senna layout phate` first.\n  \
                      • `--colour-by cluster` but no clusters → runs Leiden on the latent.\n\n\
                      --colour-by cluster (default) | annotation | topic | pb-id | pseudotime.\n\
                      Default flips to `annotation` once `senna annotate` populates the\n\
                      manifest, so cells are coloured + labelled by predicted cell type.\n\n\
                      Outputs: {out}.plot.{svg,png,pdf} (PDF default; pass --svg / --png\n\
                      for those formats)."
    )]
    Plot(PlotArgs),

    #[command(
        about = "Topic-model diagnostics: per-batch structure bars + gene × topic dictionary.",
        long_about = "Admixture-style stacked-bar structure plots per batch (panel width\n\
                      ∝ #cells), plus a gene × topic dictionary summary (Hinton ≤ 100\n\
                      genes; viridis heatmap above).\n\n\
                      Usage: senna plot-topic --from run.senna.json\n\n\
                      PDF only by default; pass --svg / --png to also emit those.\n\
                      Outputs land under {out}.plots/{struct,dict}/.",
        visible_alias = "pt"
    )]
    PlotTopic(PlotTopicArgs),
}

#[derive(Subcommand, Debug)]
enum LayoutCmd {
    #[command(about = "PHATE diffusion embedding of pseudobulks (recommended default).")]
    Phate(LayoutPhateArgs),
    #[command(about = "t-SNE of pseudobulks on raw-gene similarity (random init).")]
    Tsne(LayoutTsneArgs),
    #[command(about = "UMAP-style SGD of pseudobulks over the fuzzy kNN graph.")]
    Umap(LayoutUmapArgs),
}

fn main() -> anyhow::Result<()> {
    // Show logo if help is requested
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    logging::init_logger(cli.verbose);

    match &cli.commands {
        Commands::Svd(args) => {
            fit_svd(args)?;
        }
        Commands::Topic(args) => {
            fit_topic_model(args)?;
        }
        Commands::IndexedTopic(args) => {
            fit_indexed_topic_model(args)?;
        }
        Commands::JointTopic(args) => {
            fit_joint_topic_model(args)?;
        }

        Commands::Annotate(args) => {
            annotate_run(args)?;
        }
        Commands::Predict(args) => {
            predict_model(args)?;
        }
        Commands::EvalTopic(args) => {
            eval_topic_model(args)?;
        }
        Commands::JointSvd(args) => {
            fit_joint_svd(args)?;
        }
        Commands::Layout { cmd } => match cmd {
            LayoutCmd::Tsne(args) => {
                fit_layout_tsne(args)?;
            }
            LayoutCmd::Umap(args) => {
                fit_layout_umap(args)?;
            }
            LayoutCmd::Phate(args) => {
                fit_layout_phate(args)?;
            }
        },
        Commands::Clustering(args) => {
            run_clustering(args)?;
        }
        Commands::Pseudotime(args) => {
            run_pseudotime(args)?;
        }
        Commands::Plot(args) => {
            fit_plot(args)?;
        }
        Commands::PlotTopic(args) => {
            fit_plot_topic(args)?;
        }
    }

    info!("Done");
    Ok(())
}
