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
mod cluster;
mod cnv_pseudobulk;
mod embed_common;
mod eval_topic;
mod fit_clustering;
mod fit_indexed_topic;
mod fit_joint_topic;
mod fit_topic;
mod geometry;
mod logging;
mod marker_support;
mod postprocess;
mod run_manifest;
mod senna_input;
mod svd;
mod topic;

use embed_common::*;
use eval_topic::*;
use fit_clustering::*;
use fit_indexed_topic::*;
use fit_joint_topic::*;
use fit_topic::*;
use postprocess::*;
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
        about = "Annotate cells via anchor PBs + marker margin scoring",
        long_about = "Runs after `senna viz`. Picks anchor pseudobulks by greedy\n\
                      Gram-Schmidt on the whitened reconstruction features, labels\n\
                      each anchor against user markers via a margin rule (top1 vs\n\
                      top2 mean z-score), and soft-assigns cells to anchors by\n\
                      cosine in whitened latent space. Anchors that miss the\n\
                      margin become `novel_i` and contribute no celltype mass.\n\n\
                      Modes:\n\
                        direct      — senna annotate -g dict -z latent -p pb_mean_latent -m markers -o out\n\
                        interactive — add -I to iteratively augment markers\n\
                        LLM-assist  — --suggest-only out.json, then --apply-suggestions in.json",
        visible_alias = "annotate"
    )]
    AnnotateTopic(AnnotateTopicArgs),

    #[command(
        about = "Apply a trained topic model to new data",
        long_about = "Run encoder inference with a previously trained topic model.\n\
                      Handles gene-set misalignment, re-estimates per-batch delta\n\
                      from the frozen dictionary, and runs on CPU with rayon."
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
                      `senna topic` + `senna layout`:\n    \
                      senna plot --from run.senna.json\n\
                      Everything else (cell_coords, topics, labels, colour_by,\n\
                      palette) is read from the manifest; individual CLI flags\n\
                      still override when passed.\n\n\
                      Group source selectable via --colour-by cluster|pb-id|topic.\n\
                      Hull polygons are off by default (scRNA groups are rarely\n\
                      separable in 2D); enable with --hull for debugging.\n\
                      Outputs: {out}.plot.svg, {out}.plot.png, {out}.plot.pdf."
    )]
    Plot(PlotArgs),

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
    Tsne(VisualizeTsneArgs),
    #[command(about = "PB landmarks laid out by PHATE diffusion on raw-gene features")]
    Phate(VisualizePhateArgs),
    #[command(
        about = "PB landmarks laid out by MST + Fruchterman–Reingold on PB-PB similarity"
    )]
    Mst(LayoutMstArgs),
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

        Commands::AnnotateTopic(args) => {
            annotate_topics(args)?;
        }
        Commands::EvalTopic(args) => {
            eval_topic_model(args)?;
        }
        Commands::JointSvd(args) => {
            fit_joint_svd(args)?;
        }
        Commands::Layout { cmd } => match cmd {
            LayoutCmd::Tsne(args) => {
                fit_visualize_tsne(args)?;
            }
            LayoutCmd::Phate(args) => {
                fit_visualize_phate(args)?;
            }
            LayoutCmd::Mst(args) => {
                fit_layout_mst(args)?;
            }
        },
        Commands::Clustering(args) => {
            run_clustering(args)?;
        }
        Commands::Plot(args) => {
            fit_plot(args)?;
        }
    }

    info!("Done");
    Ok(())
}
