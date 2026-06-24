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
mod bge;
mod cluster;
mod cluster_aggregation;
mod cluster_bhc;
mod clustering;
mod cnv_pseudobulk;
mod embed_common;
mod empirical_dict;
mod eval_topic;
mod fne;
mod geometry;
mod hvg;
mod impute;
mod joint_topic;
mod logging;
mod marker_support;
mod masked_topic;
mod output_helpers;
mod postprocess;
mod predict;
mod predict_tmle;
mod principal_graph;
mod pseudotime;
mod refine_weighting;
mod resolve_embedding_space;
mod run_manifest;
mod senna_input;
mod svd;
mod topic;
mod tree_layout;
mod vae;

use annotate::{
    annotate_by_enrichment, annotate_by_projection, annotate_ontology, AnnotateArgs,
    AnnotateOntologyArgs, AnnotateProjectArgs,
};
use bge::{fit_bge, BgeArgs};
use clustering::*;
use embed_common::*;
use eval_topic::*;
use fne::{fit_fne, FneArgs};
use impute::{impute_model, ImputeArgs};
use joint_topic::*;
use masked_topic::*;
use postprocess::*;
use predict::{predict_model, PredictArgs};
use pseudotime::{run_pseudotime, PseudotimeArgs};
use resolve_embedding_space::{resolve_embedding_space, RestArgs};
use svd::*;
use topic::cmd::*;
use vae::*;

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
                  1. Train embedding   senna topic | masked-topic | svd\n                       \
                                       senna joint-topic | joint-svd       (multi-modality)\n  \
                  2. Held-out inference senna predict                       (apply trained model)\n  \
                  3. Cluster cells     senna clustering --from run.senna.json\n  \
                  4. Annotate cells    senna annotate-by-enrichment --from run.senna.json -m markers.tsv\n  \
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
        about = "Train topic-model embedding (VAE).",
        long_about = "Probabilistic topic-model embedding.\n\n\
                      Stages: (1) batch-aware pseudobulk collapsing, (2) encoder-decoder\n\
                      VAE via SGD, (3) per-cell topic inference. Decoders: multinomial,\n\
                      negative-binomial, vMF (combine via comma-separated --decoder).\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.model.json, {out}.senna.json (run manifest)."
    )]
    Topic(TopicArgs),

    #[command(
        name = "masked-topic",
        about = "Train a masked-imputation embedded topic model (foundation-style).",
        long_about = "Embedded topic model trained by masked-gene imputation — no ELBO,\n\
                      no posterior collapse. Encoder and decoder share a learned per-gene\n\
                      symbol embedding ρ ∈ ℝ^{D×H} (Dieng et al. 2020, ETM); the encoder\n\
                      pools a per-cell top-K feature window by single-query attention.\n\n\
                      Training: each cell's top-K genes are split into visible / masked;\n\
                      θ_n = softmax(encoder(visible)) (deterministic, no KL); the NB head\n\
                      imputes the held-out genes with μ = residual · ℓ · (θ·β), where\n\
                      β_kg = softmax_g(α_k · ρ_g) and φ_g is a per-gene dispersion. The\n\
                      masked objective (not a KL bottleneck) is what prevents collapse,\n\
                      so it scales with more data. Inference is encoder-only.\n\n\
                      Writes the same artifacts as `topic`, plus\n\
                      `{out}.feature_embedding.parquet` (ρ) and `{out}.dispersion.parquet`.",
        visible_aliases = ["mtm"],
        aliases = ["itopic", "indexed-topic", "etm"]
    )]
    MaskedTopic(MaskedTopicArgs),

    #[command(
        name = "masked-vae",
        about = "Train a masked-imputation Gaussian VAE (BERT-style, continuous latent).",
        long_about = "Masked-imputation VAE: the Gaussian-latent sibling of `masked-topic`.\n\
                      Same pipeline (PB-collapse training, shared per-gene ρ embedding, NB\n\
                      ETM head, encoder-only cell inference), but the encoder emits a\n\
                      reparameterized Gaussian latent z (no simplex softmax) regularized by\n\
                      a KL term — a true variational bottleneck. exp(z) drives the NB head's\n\
                      per-topic intensities (μ_g = ℓ·Σ_t exp(z_t)·β_{t,g}), so the masked\n\
                      objective + KL train an unconstrained continuous embedding while\n\
                      reusing the masked decoder unchanged. Held-out genes are imputed; the\n\
                      masked objective (not just the KL) keeps the latent from collapsing.\n\n\
                      Writes the same artifacts as `masked-topic`. NB objective only.",
        visible_aliases = ["bert"]
    )]
    MaskedVae(MaskedTopicArgs),

    #[command(
        about = "Train an scVI-style Gaussian VAE (continuous factor model).",
        long_about = "Gaussian (scVI-style) VAE — the continuous-latent sibling of\n\
                      `topic`. Same pipeline (batch-aware pseudobulk collapse → dense VAE),\n\
                      but the encoder emits an unconstrained Gaussian latent z (no simplex\n\
                      projection) and the NB decoder maps z → π = softmax_d(z·W) → μ =\n\
                      library·π. Outputs are continuous factors (cell × factor) and gene ×\n\
                      factor loadings, not topic proportions + a topic-gene dictionary.\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.model.json, {out}.senna.json (run manifest)."
    )]
    Vae(VaeArgs),

    #[command(
        about = "Train Nyström SVD embedding.",
        long_about = "Three stages: (1) batch-aware pseudobulk collapsing, (2) randomized SVD,\n\
                      (3) per-cell Nyström projection.\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.senna.json."
    )]
    Svd(SvdArgs),

    #[command(
        about = "Train joint topic model across modalities (independent or delta decoder).",
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
        about = "Train joint Nyström SVD across modalities.",
        long_about = "Joint SVD over a stack of modalities sharing cells. Data files form\n\
                      a row-major (modality × batch) table; -m sets the modality-row count.\n\
                      Cells must be shared; features may differ.\n\n\
                      Writes {out}.latent.parquet, {out}.senna.json."
    )]
    JointSvd(JointSvdArgs),

    #[command(
        about = "Train graph-based embedding (count-NCE, modality-agnostic).",
        long_about = "Joint embedding of features and cells in a single H-dim\n\
                      space via discriminative count-NCE on a sketch-coarsened\n\
                      pseudobulk (cell, feature) bipartite graph. Each input file\n\
                      contributes its rows to a shared feature axis; cell barcodes\n\
                      union across files. Modality-agnostic — works for any number\n\
                      of count panels (RNA, ATAC, protein, …). Bilinear\n\
                      `E_f · E_c + b_f + b_c` scoring with per-file rebalanced\n\
                      sampling and same-file hard negatives.\n\n\
                      Trains in two phases: (1) embed features + pseudobulks to\n\
                      learn the gene side, then (2) freeze it and densely fit each\n\
                      cell's embedding — every cell is swept ~once/epoch and the\n\
                      per-cell fit is separable (embarrassingly parallel).\n\n\
                      Writes {out}.{latent,dictionary,feature_bias,cell_bias}.parquet,\n\
                      {out}.senna.json.",
        alias = "embed-graph",
        alias = "gbe"
    )]
    Bge(BgeArgs),

    #[command(
        about = "Train a continuous Miller-Griffiths-Jordan-style latent feature model \
                 on an explicit feature-feature edge list (no expression data).",
        long_about = "Consumes a TSV/CSV of feature-feature edges (BioGRID, STRING, \
                      KEGG, synthetic-lethality, regulatory) and learns per-feature \
                      latent embeddings E ∈ ℝ^{D×H} via a continuous Miller-Griffiths-\
                      Jordan link-prediction model:\n  \n  \
                      s(i, j) = (E_i ⊙ γ) · E_j + b_i + b_j\n  \n\
                      Trains with binary cross-entropy + degree^α negative sampling \
                      (node2vec convention). Symmetric by construction.\n\n\
                      Writes {out}.feature_embedding.parquet (+ feature_bias, gamma, \
                      log_likelihood, senna.json). The output shape matches the freeze \
                      loader used by `senna masked-topic \
                      --freeze-feature-embedding`, so an `fne` run is a direct gene-side \
                      input to downstream cell-side training."
    )]
    Fne(FneArgs),

    #[command(
        name = "resolve-embedding-space",
        visible_alias = "rest",
        about = "Freeze a topic run's cell proportions θ and learn a shared cell+gene \
                 embedding from the counts (Resolve Embedding Space for Topic-models).",
        long_about = "Mirror of bge with the roles flipped: takes a finished topic-family \
                      run via --from, FREEZES its cell topic proportions θ, and trains a \
                      gene embedding ρ ∈ ℝ^{D×H} + topic embedding α ∈ ℝ^{K×H} against the \
                      raw counts (bipartite NCE). The cell embedding is derived, frozen-θ: \
                      Z = θ·α. This recasts the topic result into a metric H-space where \
                      genes, topics, and cells coexist:\n  \n  \
                      score(cell c, gene g) = (θ_c·α)·ρ_g + b_g\n  \n\
                      Writes {out}.{feature_embedding,cell_embedding,latent,topic_embedding}\
                      .parquet + senna.json (kind=resolve-embedding-space), so \
                      `senna annotate-by-projection --from {out}.senna.json` can annotate by \
                      projecting markers into the shared space — which a raw topic run \
                      cannot do. H defaults to K but may exceed it."
    )]
    ResolveEmbeddingSpace(RestArgs),

    // ─────────── 2. Held-out inference ───────────
    #[command(
        about = "Apply a trained topic / masked-topic / vae model to held-out data.",
        long_about = "Latent inference + per-cell predictive log-likelihood on a separate\n\
                      backend file. Auto-dispatches dense / indexed via model.json.\n\
                      Handles gene-set misalignment via flexible name matching and\n\
                      re-estimates per-batch delta from the frozen dictionary.\n\n\
                      Latent modes: encoder-only (default), encoder+refine, decoder-only."
    )]
    Predict(PredictArgs),

    #[command(
        about = "Impute full-feature counts on new (sparse-panel) cells via kNN over a reference latent.",
        long_about = "Two-stage post-hoc imputation:\n  \
                      1. Project new sparse-panel data through the trained\n  \
                         masked-topic encoder → θ_new [N_new, K] (runs the\n  \
                         predict pipeline internally).\n  \
                      2. For each new cell, find K nearest reference cells in\n  \
                         θ-space (L2 over the topic simplex), softmax-weight\n  \
                         their distances, and accumulate the reference cells'\n  \
                         full-feature counts.\n\n\
                      Writes {out}.imputed.parquet (N_new × n_ref_features)."
    )]
    Impute(ImputeArgs),

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
        name = "annotate-by-enrichment",
        visible_aliases = ["annotate-by-topic", "ann-by-topic", "ann-by-enrich", "annot-by-enrich"],
        about = "Annotate cells via cluster-level marker enrichment.",
        long_about = "Pipeline: (re)cluster on the manifest's latent (Leiden if no clusters\n\
                      exist) → NB-Fisher-adjusted per-cluster mean expression (streamed\n\
                      from raw counts) → weighted-KS marker enrichment with cross-cluster\n\
                      simplex normalization (housekeeping suppression) → softmax-normalized\n\
                      per-cluster Q matrix → cluster-broadcast per-cell labels.\n\n\
                      Usage: senna annotate-by-enrichment --from run.senna.json -m markers.tsv -o out\n\n\
                      Updates `manifest.annotate.{argmax,annotation,...}` so subsequent\n\
                      `senna plot` runs colour cells by predicted cell type by default.\n\
                      Writes {out}.argmax.tsv, {out}.annotation.parquet, {out}.cluster_*.parquet."
    )]
    Annotate(AnnotateArgs),

    #[command(
        name = "annotate-by-projection",
        visible_aliases = ["ann-by-proj", "annot-by-proj"],
        about = "Light cell-type annotation by marker projection (embedding runs).",
        long_about = "Projection-based complement to `senna annotate-by-enrichment`: embeds\n\
                      each marker-defined cell type as the L2-normalized centroid of its\n\
                      marker feature embeddings (the H-space the cells live in), then\n\
                      cosine-scores every cell → per-cell soft posterior. Per-cell,\n\
                      clustering-free; a permutation null (random gene sets) gives\n\
                      null-standardized z-scores (p = pnorm(-z)).\n\n\
                      Reads `outputs.feature_embedding` + `outputs.latent` from a\n\
                      bge / fne / resolve-embedding-space `run.senna.json`.\n\n\
                      Usage: senna annotate-by-projection --from run.senna.json -m markers.tsv\n\n\
                      Writes {out}.{kind}_annot.{posterior,zscore,type_embedding}.parquet."
    )]
    AnnotateProject(AnnotateProjectArgs),

    #[command(
        name = "annotate-ontology",
        visible_aliases = ["ann-ontology", "annot-ontology"],
        about = "Hierarchical multi-resolution cell-type calling on the Cell Ontology (TreeBH).",
        long_about = "Post-processes an `annotate-by-enrichment` run: places each cluster on the\n\
                      Cell Ontology is_a tree at the deepest resolution the data supports,\n\
                      abstaining on sibling ties and flagging clusters no marker explains\n\
                      (TreeBH; Bogomolov, Peterson, Benjamini & Sabatti, Biometrika 2021).\n\
                      Scores Φ(−z) on the permutation z (else restandardized ES), Simes-combined\n\
                      up the tree. Writes {out}.ontology_assignment.tsv + .ontology_node_mass.parquet.\n\
                      `annotate-by-enrichment --obo --label-cl` does the same inline (no re-run).\n\n\
                      Usage: senna annotate-ontology --from run.senna.json \\\n\
                        --label-cl label_cl.tsv --obo cl-basic.obo"
    )]
    AnnotateOntology(AnnotateOntologyArgs),

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
                      Default flips to `annotation` once `senna annotate-by-enrichment` populates the\n\
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

    #[command(
        about = "Watson/Crick mirrored genomic-activity ideograms (Strand-seq style).",
        long_about = "For each cell type, draw per-chromosome gene activity split by strand:\n\
                      forward/Watson genes as a filled pileup rising upward, reverse/Crick\n\
                      genes mirrored downward around a shared chromosome axis.\n\n\
                      Usage: senna plot-strand --from run.senna.json --gtf gencode.gtf\n\n\
                      Activity defaults to a gene × cell-type matrix derived from\n\
                      `senna annotate-by-enrichment` outputs; override with --activity. One figure per\n\
                      cell type (chromosomes stacked) plus an optional consensus, under\n\
                      {out}.strand/. PDF only by default; pass --svg / --png.",
        visible_alias = "ps"
    )]
    PlotStrand(PlotStrandArgs),
}

#[derive(Subcommand, Debug)]
enum LayoutCmd {
    #[command(about = "PHATE diffusion embedding of pseudobulks (recommended default).")]
    Phate(LayoutPhateArgs),
    #[command(about = "t-SNE of pseudobulks on raw-gene similarity (random init).")]
    Tsne(LayoutTsneArgs),
    #[command(about = "UMAP-style SGD of pseudobulks over the fuzzy kNN graph.")]
    Umap(LayoutUmapArgs),
    #[command(
        about = "Reingold-Tilford tree layout from a pseudotime run.",
        long_about = "Reads the principal graph + root node from `manifest.pseudotime`\n\
                      (written by `senna fit-pseudotime`), then produces a top-down tree\n\
                      layout where y is geodesic pseudotime and x is sibling order.\n\n\
                      Writes manifest.pseudotime.tree_{cell_coords,nodes_2d}."
    )]
    Tree(LayoutTreeArgs),
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
        Commands::Bge(args) => {
            fit_bge(args)?;
        }
        Commands::Fne(args) => {
            fit_fne(args)?;
        }
        Commands::ResolveEmbeddingSpace(args) => {
            resolve_embedding_space(args)?;
        }
        Commands::Topic(args) => {
            fit_topic_model(args)?;
        }
        Commands::MaskedTopic(args) => {
            fit_masked_topic_model(args)?;
        }
        Commands::MaskedVae(args) => {
            fit_masked_vae_model(args)?;
        }
        Commands::Vae(args) => {
            fit_vae_model(args)?;
        }
        Commands::JointTopic(args) => {
            fit_joint_topic_model(args)?;
        }

        Commands::Annotate(args) => {
            annotate_by_enrichment(args)?;
        }
        Commands::AnnotateProject(args) => {
            annotate_by_projection(args)?;
        }
        Commands::AnnotateOntology(args) => {
            annotate_ontology(args)?;
        }
        Commands::Predict(args) => {
            predict_model(args)?;
        }
        Commands::Impute(args) => {
            impute_model(args)?;
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
            LayoutCmd::Tree(args) => {
                fit_layout_tree(args)?;
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
        Commands::PlotStrand(args) => {
            fit_plot_strand(args)?;
        }
    }

    info!("Done");
    Ok(())
}
