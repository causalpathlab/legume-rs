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
mod fit_topic;
mod geometry;
mod hvg;
mod logging;
mod marker_support;
mod postprocess;
mod predict;
mod predict_tmle;
mod refine_weighting;
mod run_manifest;
mod senna_input;
mod svd;
mod topic;

use annotate::{annotate_run, AnnotateArgs};
use embed_common::*;
use eval_topic::*;
use fit_clustering::*;
use fit_indexed_topic::*;
use fit_joint_topic::*;
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
    about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment",
    long_about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment.\n\n\
                  Input: sparse backends in `.zarr` or `.h5` format.\n\
                  Convert from Matrix Market with `data-beans from-mtx`."
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Verbose logging")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Embed cells by Nyström SVD with batch-aware pseudobulk collapsing",
        long_about = "Nyström SVD embedding in three stages:\n\
                      (1) batch-aware multi-level pseudobulk collapsing,\n\
                      (2) randomized SVD on the collapsed matrix,\n\
                      (3) projection of every cell onto the learned basis."
    )]
    Svd(SvdArgs),

    #[command(
        about = "Embed cells by probabilistic topic modelling (VAE, multi-decoder)",
        long_about = "Topic-model embedding in three stages:\n\
                      (1) batch-aware multi-level pseudobulk collapsing,\n\
                      (2) encoder-decoder VAE training via SGD,\n\
                      (3) encoder inference of per-cell latent topic proportions.\n\n\
                      Supports multinomial, negative-binomial, and vMF decoders\n\
                      (jointly when comma-separated via --decoder)."
    )]
    Topic(TopicArgs),

    #[command(
        about = "Topic model with adaptive top-K feature windows (~4-7× faster decoder)",
        long_about = "Same three-stage pipeline as `topic`, but the encoder and\n\
                      decoder operate on a per-cell top-K feature window instead\n\
                      of the dense D × K dictionary. Useful for very large gene sets.",
        visible_alias = "itopic"
    )]
    IndexedTopic(IndexedTopicArgs),

    #[command(
        about = "Annotate cells via cluster-level marker enrichment",
        long_about = "Cluster cells (Leiden on cosine-KNN of the manifest's latent),\n\
                      compute NB-Fisher-adjusted per-cluster mean expression by\n\
                      streaming raw counts from the manifest's zarr, then run a\n\
                      weighted-KS marker enrichment with cross-cluster simplex\n\
                      normalization for housekeeping suppression. The FDR-sparse\n\
                      nClusters × C Q matrix is softmax-normalized per cluster;\n\
                      per-cell labels come from cluster-broadcast Q.\n\n\
                      Usage: senna annotate --from run.senna.json -m markers.tsv -o out\n\n\
                      Provide --clusters <PATH> (or run `senna cluster --from ...`\n\
                      first) to skip the internal Leiden pass.",
        visible_alias = "annotate"
    )]
    Annotate(AnnotateArgs),

    #[command(
        about = "Apply a trained topic / indexed-topic model to held-out data",
        long_about = "Run latent inference + per-cell predictive log-likelihood\n\
                      on a separate held-out backend file. Auto-dispatches between\n\
                      the dense (`topic`) and indexed (`indexed-topic`) paths via\n\
                      the model.json metadata. Handles gene-set misalignment via\n\
                      flexible name matching, re-estimates per-batch delta from\n\
                      the frozen dictionary, and supports three latent modes:\n\
                      encoder-only (default), encoder+refine, and decoder-only."
    )]
    Predict(PredictArgs),

    #[command(
        about = "[deprecated] Use `senna predict`",
        long_about = "Deprecated alias for `senna predict`. The new `predict`\n\
                      subcommand handles both `topic` and `indexed-topic` models\n\
                      and additionally computes per-cell predictive log-likelihood."
    )]
    EvalTopic(EvalTopicArgs),

    #[command(
        about = "Joint Nyström SVD across multiple modalities",
        long_about = "Nyström SVD on a stack of modalities sharing the same cells.\n\
                      Data files are arranged row-major as a (modality × batch) table;\n\
                      use -m to set the number of modality rows."
    )]
    JointSvd(JointSvdArgs),

    #[command(
        about = "Joint topic model across multiple modalities (independent or delta decoder)",
        long_about = "Joint topic-model embedding across a stack of modalities.\n\n\
                      Data files are arranged row-major as a (modality × batch) table;\n\
                      use -m to set the number of modality rows.\n\n\
                      Decoder types:\n\
                        independent — each modality has its own topic dictionary;\n\
                                      features may differ across modalities.\n\
                        delta       — shared base dictionary + cumulative chain deltas;\n\
                                      modality m = softmax(z @ (W_base + Σ δ_1..m)).\n\
                                      Requires shared features across all modalities."
    )]
    JointTopic(JointTopicArgs),

    #[command(
        about = "Compute 2D layout in raw-gene-space (tsne or phate)",
        long_about = "Topic-agnostic 2D layout: build PBs via batch-corrected\n\
                      multi-level collapsing, compute PB-PB cosine similarity on\n\
                      log1p-CPM gene vectors, then lay out via t-SNE or PHATE.\n\
                      Cells placed by cheap Nyström in random-projection space.\n\n\
                      Preferred invocation uses a run manifest produced by\n\
                      `senna topic` (or svd / itopic / joint-*):\n    \
                      senna layout phate --from run.senna.json\n\
                      The manifest supplies data files + output prefix, and is\n\
                      updated in place with `layout.cell_coords`, `layout.pb_coords`,\n\
                      and `layout.pb_gene_mean` paths — downstream\n\
                      `senna plot --from run.senna.json` then just works.\n\n\
                      Pick a method: `senna layout tsne ...` or `senna layout phate ...`.",
        visible_alias = "lay",
        subcommand_required = true,
        arg_required_else_help = true
    )]
    Layout {
        #[command(subcommand)]
        cmd: LayoutCmd,
    },

    #[command(
        about = "Publication scatter plot from `senna layout` coords (SVG/PNG/PDF)",
        long_about = "Render a publication-quality scatter: per-group rasterized\n\
                      layers (tiny-skia, 300 dpi) with transparent background,\n\
                      optional convex hull polygons, and vector text labels at\n\
                      per-group medians (fully editable in Illustrator/Inkscape).\n\n\
                      Preferred invocation uses a run manifest from\n\
                      `senna topic` (and optionally `senna annotate`):\n    \
                      senna plot --from run.senna.json\n\
                      If the manifest has no `layout.cell_coords` yet, plot will\n\
                      auto-run `senna layout phate` against it first. Everything\n\
                      else (cell_coords, topics, annotation, labels, colour_by,\n\
                      palette) is read from the manifest; individual CLI flags\n\
                      still override when passed.\n\n\
                      Group source selectable via\n\
                      --colour-by cluster|pb-id|topic|annotation. When\n\
                      `senna annotate` has populated the manifest and no\n\
                      explicit --colour-by is given, `annotation` is the default\n\
                      so cells are coloured + labelled by predicted cell type.\n\
                      Hull polygons are off by default (scRNA groups are rarely\n\
                      separable in 2D); enable with --hull for debugging.\n\
                      Outputs: {out}.plot.svg, {out}.plot.png, {out}.plot.pdf."
    )]
    Plot(PlotArgs),

    #[command(
        about = "Topic structure-bar (per batch) + gene × topic dictionary plots",
        long_about = "Admixture-style stacked-bar structure plots per batch (panel\n\
                      width ∝ #cells), plus a gene × topic dictionary summary\n\
                      (Hinton ≤ 100 genes; viridis-bin heatmap above).\n\n\
                      Preferred invocation:\n    \
                      senna plot-topic --from run.senna.json\n\
                      Defaults to PDF only — pass --svg / --png to also emit those.\n\
                      Outputs land under {out}.plots/{struct,dict}/.",
        visible_alias = "pt"
    )]
    PlotTopic(PlotTopicArgs),

    #[command(
        about = "Cluster cells on latent topic / SVD representations",
        long_about = "Cluster cells using a latent matrix from `senna topic` or `senna svd`.\n\n\
                      Algorithms:\n\
                        kmeans  — k-means (default; requires -k)\n\
                        leiden  — graph-based community detection (auto-k)\n\
                        hsblock — hierarchical stochastic block model (2^(depth-1) clusters)"
    )]
    Clustering(ClusteringArgs),
}

#[derive(Subcommand, Debug)]
enum LayoutCmd {
    #[command(about = "PB landmarks laid out by t-SNE on raw-gene similarity (random init)")]
    Tsne(LayoutTsneArgs),
    #[command(about = "PB landmarks laid out by UMAP-style SGD on the fuzzy kNN graph")]
    Umap(LayoutUmapArgs),
    #[command(about = "PB landmarks laid out by PHATE diffusion on raw-gene features")]
    Phate(LayoutPhateArgs),
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
