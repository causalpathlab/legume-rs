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
mod annotate_gem;
mod assoc;
mod bge;
mod cluster;
mod cluster_aggregation;
mod cluster_bhc;
mod clustering;
mod cnv_pseudobulk;
mod counterfactual;
mod deconvolve;
mod embed_common;
mod empirical_dict;
mod eval_topic;
mod fne;
mod gem;
mod gem_encoder;
mod gem_manifest;
mod geometry;
mod hvg;
mod impute;
mod joint_topic;
mod lineage;
mod lineage_plot;
mod logging;
mod marker_support;
mod masked_topic;
mod output_helpers;
mod pb_reference;
mod postprocess;
mod predict;
mod predict_tmle;
mod principal_graph;
mod probe;
mod pseudotime;
mod refine_weighting;
mod resolve_embedding_space;
mod run_manifest;
mod senna_input;
mod svd;
mod topic;
mod tree_layout;
mod update;
mod vae;

use annotate::{
    annotate_by_enrichment, annotate_by_projection, annotate_ontology, AnnotateArgs,
    AnnotateOntologyArgs, AnnotateProjectionArgs,
};
use annotate_gem::run::{run_annotate as run_annotate_gem, AnnotateArgs as AnnotateGemArgs};
use assoc::run::{run_assoc, AssocArgs};
use bge::{fit_bge, BgeArgs};
use clustering::*;
use deconvolve::DeconvolveArgs;
use embed_common::*;
use eval_topic::*;
use fne::{fit_fne, FneArgs};
use gem::args::GemArgs;
use gem::run::run_gem_embedding;
use gem_encoder::args::GemEncoderArgs;
use gem_encoder::run::run_gem_encoder;
use impute::{impute_model, ImputeArgs};
use joint_topic::*;
use lineage::args::LineageArgs;
use lineage::run::run_lineage;
use lineage_plot::{run_plot as run_lineage_plot, PlotArgs as LineagePlotArgs};
use masked_topic::*;
use postprocess::*;
use predict::{predict_model, PredictArgs};
use probe::{run_probe, ProbeArgs};
use pseudotime::{run_pseudotime, PseudotimeArgs};
use resolve_embedding_space::{resolve_embedding_space, RestArgs};
use svd::*;
use topic::cmd::*;
use update::{run_update, UpdateArgs};
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
    about = "SENNA — single-cell embedding (SVD / topic), annotation, trajectory,\n\
             and plotting.",
    long_about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment.\n\
                  \n\
                  Input: sparse backends in `.zarr` or `.h5`.\n\
                  Convert from Matrix Market with `data-beans from-mtx`.\n\
                  \n\
                  Each step writes its outputs back to the run manifest `{prefix}.senna.json`.\n\
                  Downstream commands read data and batch files from it.\n\
                  Steps 3 and 5 still need their own --latent / --out.\n\
                  \n  \
                  1. Train embedding   senna topic | masked-topic | svd | bge\n                       \
                  senna joint-topic | joint-svd   (multi-modality)\n  \
                  2. Held-out inference senna predict            (apply trained model)\n  \
                  3. Cluster cells     senna clustering --from run.senna.json --latent L --out O\n  \
                  4. Annotate cells    senna annotate-by-enrichment --from run.senna.json\n                       \
                  -m markers.tsv\n  \
                  5. Trajectory        senna pseudotime --from run.senna.json --out O\n  \
                  6. 2D layout         senna layout {phate|tsne|umap} --from run.senna.json\n  \
                  7. Scatter plot      senna plot       --from run.senna.json\n  \
                  8. Topic diagnostics senna plot-topic --from run.senna.json\n\
                  \n\
                  `senna plot` auto-runs steps 3 + 6 on demand.\n\
                  \n\
                  Bulk deconvolution is a side branch off a `bge` run.\n\
                  It needs an annotation and bulk counts, plus the single-cell\n\
                  counts the reference profiles are measured from.\n\
                  \n  \
                  senna deconvolve --from bge.senna.json --annotation A --bulk bulk.parquet\n\
                  \n\
                  Artifact naming: a slot name fixes the axis, never the numeric scale.\n\
                  `feature_loading` is the per-gene loading rho, and it is signed.\n\
                  `dictionary` is a topic dictionary in LOG space, or SVD signed loadings.\n\
                  Reading one as the other yields NaN, so check `kind` before assuming.\n\
                  See senna/docs/deconvolve.md and the run_manifest module docs."
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
        long_about = "Probabilistic topic-model embedding.\n\
                      \n\
                      Stages:\n\
                      \x20 1. batch-aware pseudobulk collapsing\n\
                      \x20 2. encoder-decoder VAE via SGD\n\
                      \x20 3. per-cell topic inference\n\
                      \n\
                      Decoders are multinom, nb and nbmixture (the default).\n\
                      Combine them with a comma-separated --decoder.\n\
                      \n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.model.json, {out}.senna.json (run manifest)."
    )]
    Topic(TopicArgs),

    #[command(
        name = "masked-topic",
        about = "Train a masked-imputation embedded topic model (foundation-style).",
        long_about = "Embedded topic model trained by masked-gene imputation. There is no ELBO,\n\
                      and no posterior collapse.\n\
                      Encoder and decoder share a per-gene embedding ρ ∈ ℝ^{D×H}.\n\
                      That follows Dieng et al. 2020 (ETM).\n\
                      The encoder pools a per-cell top-K window by attention.\n\
                      \n\
                      Training splits each cell's top-K genes into visible and masked.\n\
                      θ_n = softmax(encoder(visible)), deterministic and KL-free.\n\
                      The NB head imputes held-out genes with μ = residual · ℓ · (θ·β).\n\
                      There β_kg = softmax_g(α_k · ρ_g). φ_g is a per-gene dispersion.\n\
                      \n\
                      The masked objective prevents collapse, not a KL bottleneck.\n\
                      So it scales with more data. Inference is encoder-only.\n\
                      \n\
                      Writes the same artifacts as `topic`.\n\
                      It adds `{out}.feature_embedding.parquet` (ρ) and `{out}.dispersion.parquet`.",
        visible_aliases = ["mtm"],
        aliases = ["itopic", "indexed-topic", "etm"]
    )]
    MaskedTopic(MaskedTopicArgs),

    #[command(
        name = "masked-vae",
        about = "Train a masked-imputation Gaussian VAE (BERT-style, continuous latent).",
        long_about = "Masked-imputation VAE.\n\
                      It is the Gaussian-latent sibling of `masked-topic`.\n\
                      The pipeline is the same. PB-collapse training, a shared ρ embedding,\n\
                      an NB ETM head, encoder-only inference.\n\
                      \n\
                      The encoder differs.\n\
                      It emits a reparameterized Gaussian latent z. There is no simplex softmax;\n\
                      a KL term regularizes it. That is a true variational bottleneck.\n\
                      exp(z) drives the NB head's per-topic intensities,\n\
                      μ_g = ℓ·Σ_t exp(z_t)·β_{t,g}.\n\
                      \n\
                      Masked objective and KL together train that embedding,\n\
                      which stays unconstrained and continuous.\n\
                      The masked decoder is reused unchanged.\n\
                      Held-out genes are imputed.\n\
                      The masked objective keeps the latent from collapsing.\n\
                      The KL alone does not.\n\
                      \n\
                      Writes the same artifacts as `masked-topic`.\n\
                      The NB objective is the only one available.",
        visible_aliases = ["bert"]
    )]
    MaskedVae(MaskedTopicArgs),

    #[command(
        name = "masked-sbp",
        about = "Train a masked-imputation topic model with a stick-breaking-process simplex.",
        long_about = "Stick-breaking-process (SBP) sibling of `masked-topic`.\n\
                      The masked-imputation pipeline is the same. A shared ρ embedding,\n\
                      an NB ETM head, a deterministic KL-free objective, encoder-only inference.\n\
                      \n\
                      The encoder differs. It maps logits through a stick-breaking simplex,\n\
                      not a softmax: θ_k = v_k·∏_{j<k}(1−v_j) with v_k = σ(η_k).\n\
                      \n\
                      Topics are therefore no longer exchangeable.\n\
                      Early sticks carry more mass a priori.\n\
                      That gives an intrinsic ordering and a self-pruning tail:\n\
                      later topics shrink toward 0 unless the data needs them. It is a soft,\n\
                      differentiable way to over-provision K and prune.\n\
                      \n\
                      Writes the same artifacts as `masked-topic`.",
        visible_aliases = ["sbp"]
    )]
    MaskedSbp(MaskedTopicArgs),

    #[command(
        about = "Train an scVI-style Gaussian VAE (continuous factor model).",
        long_about = "Gaussian (scVI-style) VAE. It is the continuous-latent sibling of `topic`.\n\
                      The pipeline is the same: batch-aware pseudobulk collapse,\n\
                      then a dense VAE.\n\
                      \n\
                      The encoder emits an unconstrained Gaussian latent z,\n\
                      with no simplex projection.\n\
                      The NB decoder maps z → π = softmax_d(z·W) → μ = library·π.\n\
                      \n\
                      Outputs are continuous factors and loadings.\n\
                      They are cell × factor and gene × factor.\n\
                      They are not topic proportions and a topic-gene dictionary.\n\
                      \n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.model.json, {out}.senna.json (run manifest)."
    )]
    Vae(VaeArgs),

    #[command(
        about = "Train Nyström SVD embedding.",
        long_about = "Three stages:\n\
                      \x20 1. batch-aware pseudobulk collapsing\n\
                      \x20 2. randomized SVD\n\
                      \x20 3. per-cell Nyström projection\n\
                      \n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.senna.json."
    )]
    Svd(SvdArgs),

    #[command(
        about = "Train joint topic model across modalities (independent or delta decoder).",
        long_about = "Joint topic-model embedding over modalities sharing cells.\n\
                      Data files form a row-major (modality × batch) table.\n\
                      -m sets the modality-row count.\n\
                      \n\
                      Decoder types:\n  \
                      independent — each modality keeps its own dictionary; features may differ.\n  \
                      delta       — shared base + cumulative chain deltas\n              \
                      (modality m = softmax(z @ (W_base + Σ δ_1..m));\n              \
                      requires shared features across modalities).\n\
                      \n\
                      Writes {out}.latent.parquet, {out}.senna.json."
    )]
    JointTopic(JointTopicArgs),

    #[command(
        about = "Train joint Nyström SVD across modalities.",
        long_about = "Joint SVD over a stack of modalities sharing cells.\n\
                      Data files form a row-major (modality × batch) table.\n\
                      -m sets the modality-row count.\n\
                      Cells must be shared; features may differ.\n\
                      \n\
                      Writes {out}.latent.parquet, {out}.senna.json."
    )]
    JointSvd(JointSvdArgs),

    #[command(
        about = "Train graph-based embedding (count-NCE, modality-agnostic).",
        long_about = "Joint embedding of features and cells in one H-dim space.\n\
                      It uses discriminative count-NCE.\n\
                      The graph is a sketch-coarsened pseudobulk bipartite graph,\n\
                      over (cell, feature) pairs.\n\
                      \n\
                      Each input file contributes its rows to a shared feature axis.\n\
                      Cell barcodes union across files. The method is modality-agnostic.\n\
                      Any number of count panels works: RNA, ATAC, protein. Scoring is bilinear:\n\
                      `E_f · E_c + b_f + b_c`.\n\
                      \n\
                      Positives are drawn by a two-stage stratified sampler.\n\
                      Stage 1 picks a pseudobulk with q(p) ∝ pb_size(p)^alpha_pb.\n\
                      Stage 2 picks a feature within it, weighted by μ_pf.\n\
                      Negatives are drawn UNIFORMLY over the global pool.\n\
                      That pool holds every expressed feature.\n\
                      They are therefore abundance-independent.\n\
                      \n\
                      Training runs in two phases. Phase 1 embeds features and pseudobulks.\n\
                      That learns the gene side.\n\
                      Phase 2 freezes that and densely fits each cell embedding.\n\
                      Every cell is swept about once per epoch. The per-cell fit is separable,\n\
                      so it is embarrassingly parallel.\n\
                      \n\
                      Writes {out}.senna.json,\n\
                      plus {out}.{cell_embedding,dictionary,feature_embedding,\n  \
                      feature_bias,cell_bias}.parquet.\n\
                      The H-space cell embedding Z is always {out}.cell_embedding.parquet.\n\
                      \n\
                      Unless --skip-etm, an ETM is resolved too.\n\
                      That adds {out}.{latent,topic_embedding}.parquet, with latent = log θ.",
        alias = "embed-graph",
        alias = "gbe"
    )]
    Bge(BgeArgs),

    #[command(
        about = "Latent feature model over a feature-feature edge list.",
        long_about = "Learns per-feature latent embeddings from an edge list.\n\
                      No expression data is involved.\n\
                      \n\
                      Input is a TSV/CSV of feature-feature edges. BioGRID, STRING,\n\
                      KEGG and regulatory networks all fit.\n\
                      \n\
                      Embeddings E ∈ ℝ^{D×H} come from a link-prediction model.\n\
                      That model is a continuous Miller-Griffiths-Jordan.\n  \
                      \n  \
                      s(i, j) = (E_i ⊙ γ) · E_j + b_i + b_j\n  \
                      \n\
                      Training is binary cross-entropy. Negative sampling is degree^α,\n\
                      the node2vec convention. The model is symmetric by construction.\n\
                      \n\
                      Writes {out}.feature_embedding.parquet. feature_bias, gamma,\n\
                      log_likelihood and senna.json ship too.\n\
                      \n\
                      The output shape matches the freeze loader.\n\
                      That is `senna masked-topic --freeze-feature-embedding`.\n\
                      An `fne` run is a direct gene-side input downstream."
    )]
    Fne(FneArgs),

    #[command(
        name = "resolve-embedding-space",
        visible_alias = "rest",
        about = "Recast a topic run into a shared cell+gene embedding.",
        long_about = "Mirror of bge with the roles flipped.\n\
                      It takes a finished topic-family run via --from.\n\
                      That run's cell topic proportions θ are FROZEN.\n\
                      \n\
                      It then trains ρ ∈ ℝ^{D×H} and α ∈ ℝ^{K×H}.\n\
                      Both fit the raw counts by bipartite NCE.\n\
                      The cell embedding is derived from frozen θ, as Z = θ·α.\n  \
                      \n  \
                      score(cell c, gene g) = (θ_c·α)·ρ_g + b_g\n  \
                      \n\
                      Writes senna.json with kind=resolve-embedding-space.\n\
                      It also writes {out}.feature_embedding.parquet,\n\
                      {out}.cell_embedding.parquet and {out}.topic_embedding.parquet.\n\
                      \n\
                      The result is a metric H-space. Genes, topics and cells coexist in it.\n\
                      Downstream clustering and `senna annotate-by-enrichment` read it.\n\
                      H defaults to K, but may exceed it."
    )]
    ResolveEmbeddingSpace(RestArgs),

    // ─────────── 2. Held-out inference ───────────
    #[command(
        about = "Apply a trained topic / masked-topic / vae model to held-out data.",
        long_about = "Latent inference on a separate backend file.\n\
                      It also reports per-cell predictive log-likelihood.\n\
                      \n\
                      Dense and indexed models are auto-dispatched via model.json.\n\
                      Gene-set misalignment is handled by flexible name matching.\n\
                      Per-batch delta is re-estimated from the frozen dictionary.\n\
                      \n\
                      Latent modes are encoder-only (the default), encoder+refine,\n\
                      and decoder-only."
    )]
    Predict(PredictArgs),

    #[command(
        about = "Drift probe: novelty verdict for held-out data vs a trained masked model.",
        long_about = "Read-only drift probe — the covered-vs-new gate.\n\
                      \n\
                      It scores each query cell's predictive fit. The model may be masked-topic,\n\
                      masked-vae or masked-sbp. A null is calibrated from --calibration,\n\
                      in-distribution. Query cells below the null tail are flagged.\n\
                      A batch-level covered/novel verdict is emitted.\n\
                      \n\
                      Usage:\n\
                      senna probe --model M --calibration ref.zarr query.zarr -o out\n  \
                      Writes {out}.probe.tsv (per-cell fit + flag)."
    )]
    Probe(ProbeArgs),

    #[command(
        about = "Absorb new samples into a trained model by continuing its training.",
        long_about = "Continue a trained run over a larger cohort.\n\
                      \n\
                      The parent's manifest records both the data it was trained on\n\
                      and the arguments it was trained with, so the update re-runs\n\
                      that same fit over `recorded + new` data with warm start on.\n\
                      Only the new files and the output prefix are named here.\n\
                      \n\
                      Every family trains on pseudobulks, so the old cohort is\n\
                      replayed exactly and old-vs-new batch effects are matched at\n\
                      cell resolution. The cost is that each round re-reads every\n\
                      previously absorbed cell.\n\
                      \n\
                      Usage:\n\
                      senna update new.zarr --model M_v1 -o M_v2\n\
                      \n\
                      Families: topic, masked-topic, masked-sbp, masked-vae, vae.\n\
                      For svd this re-fits on the union — there are no weights to\n\
                      warm-start. bge is unsupported: it saves no checkpoint."
    )]
    Update(UpdateArgs),

    #[command(
        about = "Impute full-feature counts on new cells by kNN over a reference latent.",
        long_about = "Two-stage post-hoc imputation:\n  \
                      1. Project new sparse-panel data through the trained\n  \
                      \x20  masked-topic encoder, giving θ_new [N_new, K].\n  \
                      \x20  This runs the predict pipeline internally.\n  \
                      2. For each new cell, find its K nearest reference cells\n  \
                      \x20  in θ-space, by L2 over the topic simplex.\n  \
                      \x20  Softmax-weight their distances, then accumulate\n  \
                      \x20  those reference cells' full-feature counts.\n\
                      \n\
                      Writes {out}.imputed.parquet (N_new × n_ref_features)."
    )]
    Impute(ImputeArgs),

    #[command(about = "[deprecated] Alias for `senna predict`.")]
    EvalTopic(EvalTopicArgs),

    // ─────────── 3. Cluster / annotate / trajectory (run on a manifest) ───────────
    #[command(
        about = "Cluster cells on the manifest's latent (kmeans / leiden / hsblock).",
        long_about = "Cluster cells using `manifest.outputs.latent`.\n\
                      \n\
                      Algorithms:\n  \
                      kmeans  — requires -k.\n  \
                      leiden  — graph-based, auto-k.\n  \
                      hsblock — hierarchical SBM (2^(depth-1) clusters).\n\
                      \n\
                      Writes {out}.clusters.parquet and updates `manifest.cluster.clusters`."
    )]
    Clustering(ClusteringArgs),

    #[command(
        name = "annotate-by-enrichment",
        visible_aliases = ["annotate-by-topic", "ann-by-topic", "ann-by-enrich", "annot-by-enrich"],
        about = "Annotate cells via cluster-level marker enrichment.",
        long_about = "Pipeline:\n  \
                      1. (re)cluster on the manifest's latent, Leiden when no\n  \
                      \x20  clusters exist yet.\n  \
                      2. NB-Fisher-adjusted per-cluster mean expression,\n  \
                      \x20  streamed from raw counts.\n  \
                      3. weighted-KS marker enrichment, with cross-cluster simplex\n  \
                      \x20  normalization to suppress housekeeping genes.\n  \
                      4. softmax-normalized per-cluster Q matrix.\n  \
                      5. cluster-broadcast per-cell labels.\n\
                      \n\
                      Usage:\n\
                      senna annotate-by-enrichment -f run.senna.json -m markers.tsv -o out\n\
                      \n\
                      Updates `manifest.annotate.{argmax,annotation,...}`.\n\
                      Later `senna plot` runs then colour by predicted cell type.\n\
                      Writes {out}.argmax.tsv, {out}.annotation.parquet and {out}.cluster_*.parquet."
    )]
    Annotate(AnnotateArgs),

    #[command(
        name = "annotate-ontology",
        visible_aliases = ["ann-ontology", "annot-ontology"],
        about = "Hierarchical multi-resolution cell-type calling on the Cell Ontology (TreeBH).",
        long_about = "Post-processes an `annotate-by-enrichment` run.\n\
                      Each cluster is placed on the Cell Ontology is_a tree,\n\
                      at the deepest resolution the data supports. Sibling ties abstain.\n\
                      Clusters that no marker explains are flagged. The method is TreeBH,\n\
                      after Bogomolov, Peterson, Benjamini & Sabatti, Biometrika 2021.\n\
                      \n\
                      Scores are Φ(−z) on the permutation z,\n\
                      or the restandardized ES when that is unavailable,\n\
                      Simes-combined up the tree.\n\
                      Writes {out}.ontology_assignment.tsv and {out}.ontology_node_mass.parquet.\n\
                      `annotate-by-enrichment --obo --label-cl` does the same inline,\n\
                      with no re-run needed.\n\
                      \n\
                      Usage:\n\
                      senna annotate-ontology -f run.senna.json \\\n\
                      \x20 --label-cl label_cl.tsv --obo cl-basic.obo"
    )]
    AnnotateOntology(AnnotateOntologyArgs),

    #[command(
        name = "annotate-by-projection",
        visible_aliases = ["ann-by-proj", "annot-by-proj"],
        about = "Annotate cells via firm marker over-representation on the co-embedding.",
        long_about = "Embedding-grounded alternative to `annotate-by-enrichment`.\n\
                      It suits runs with a co-embedded gene space:\n\
                      bge, fne or resolve-embedding-space.\n\
                      \n\
                      Pipeline:\n  \
                      1. build each type's IDF-weighted marker centroid.\n  \
                      2. assign each cell to its Euclidean nearest centroid.\n  \
                      3. drop distance outliers in QC.\n  \
                      4. Leiden-cluster the cells.\n  \
                      5. test cluster × term hypergeometric over-representation,\n  \
                      \x20  permutation-calibrated.\n  \
                      6. broadcast the per-cluster call to cells.\n\
                      \n\
                      --obo and --label-cl add an optional TreeBH ontology.\n\
                      Raw counts are never re-read.\n\
                      That makes this complementary to enrichment.\n\
                      Enrichment is raw-count-grounded instead.\n\
                      \n\
                      Usage:\n\
                      senna annotate-by-projection -f run.senna.json -m markers.tsv -o out\n\
                      Writes {out}.{argmax.tsv,membership.tsv,annot.parquet,cluster_term_*.parquet,\n\
                      null_calibration.tsv}; updates `manifest.annotate.*`."
    )]
    AnnotateByProjection(AnnotateProjectionArgs),

    #[command(
        name = "deconvolve",
        visible_aliases = ["deconv", "deconvolution"],
        about = "Deconvolve bulk samples into cell-type fractions + per-type expression.",
        long_about = "Hierarchical-Bayes bulk deconvolution against an empirical reference.\n\
                      \n\
                      Annotated single cells are collapsed into archetypes, and each\n\
                      archetype's gene profile is measured from its member cells.\n\
                      Profiles are measured rather than reconstructed, so nothing caps\n\
                      how well a composition can fit.\n\
                      A full Gibbs sampler then runs a multinomial gene split with\n\
                      Gamma-Poisson conjugate abundances.\n\
                      \n\
                      Several archetype granularities are pooled, so the partition is\n\
                      averaged over rather than conditioned on.\n\
                      \n\
                      Usage:\n\
                      senna deconvolve --from run.senna.json --annotation A --bulk bulk.parquet\n  \
                      Writes {out}.{fractions,fractions_ci,abundance,residual}.tsv,\n\
                      {out}.expression/*.parquet, and component diagnostics.\n\
                      \n\
                      Reported fractions are mRNA shares, not cell shares, and their\n\
                      range is compressed. See senna/docs/deconvolve.md for the limits."
    )]
    Deconvolve(DeconvolveArgs),

    #[command(
        about = "Pseudotime via Monocle-3-style principal graph (SimplePPT) on the latent.",
        long_about = "Port of Mao et al. 2015 SimplePPT applied to `manifest.outputs.latent`.\n\
                      \n\
                      (1) k-means init K centroids, (2) iterate:\n\
                      soft-assign cells → MST over centroids → solve\n    \
                      (D_R + γL) Y = R^T Z for centroid coords,\n\
                      (3) project each cell onto its nearest tree edge,\n\
                      (4) Dijkstra geodesic from a chosen root → pseudotime.\n\
                      \n\
                      Outputs {out}.pseudotime.parquet.\n\
                      It also writes {out}.principal_graph.{nodes,edges}.parquet."
    )]
    Pseudotime(PseudotimeArgs),

    // ─────────── 4. Layout + plotting ───────────
    #[command(
        name = "gem",
        aliases = ["gem-embedding"],
        about = "GEM: Geodesic Embedding for RNA Motion in one cell space",
        long_about = "Geodesic Embedding for RNA Motion: a joint cell-feature embedding.\n\
                      Motion is the local velocity δ (the tangent);\n\
                      the lineage is the geodesic path it traces.\n\
                      Runs over the shared graph_embedding_util engine,\n\
                      which is modality-agnostic. Fed gene counts (spliced + unspliced) today;\n\
                      embeds any per-feature count.\n\
                      \n\
                      Per-gene β-sharing:\n\
                      each `{gene}/count/{spliced|unspliced}` row embeds as β_g.\n\
                      A gene's spliced and unspliced tracks thus share one identity.\n\
                      Two things are solved JOINTLY by default:\n\
                      cell identity θ → `{out}.cell_embedding.parquet` (raw),\n\
                      and the velocity increment δ → `{out}.velocity.parquet`.\n\
                      so θ is powered by both splice tracks rather than the spliced one alone.\n\
                      `--sequential-velocity` reverts to the older two-step fit:\n\
                      θ from the spliced edges, then δ from the unspliced with θ held fixed,\n\
                      which pins θ to the mature state for a cleaner δ readout.\n\
                      The nascent state is just θ+δ; ‖δ‖ is speed.\n\
                      Per-gene velocity is the in-model δ_g → `{out}.delta_feature_embedding.parquet`;\n\
                      it is written whenever the input carries unspliced rows. `--delta-l2 0`,\n\
                      the default, applies a mild ridge to keep it identified.\n\
                      The per-gene identity β_g is `{out}.beta_feature_embedding.parquet`,\n\
                      gene-keyed so a marker panel joins against it directly.\n\
                      \n\
                      `{out}.velocity_increment.parquet` is a DIAGNOSTIC, not the velocity:\n\
                      it is the raw per-cell Poisson increment δ_c,\n\
                      which a shrinkage-toward-origin common mode dominates: δ_c ≈ −0.5·θ,\n\
                      from fitting sparse unspliced counts absolutely.\n\
                      Use `{out}.velocity.parquet` for the velocity.\n\
                      \n\
                      With `--lineage-dag` it also shapes the embedding along a pseudobulk lineage.\n\
                      It then writes a per-cell pseudotime + fate backbone.\n\
                      That backbone is a prior for `senna lineage`, not a replacement.\n\
                      \n\
                      `{out}.gem.json` records that this prefix came from the EMBEDDING model,\n\
                      which is how `senna annotate-gem` and `senna lineage` pick their statistic.",
        after_long_help = "\
	Example:\n\
  senna gem out/rep1_wt_genes.zarr.zip -o out/gem\n\n\
  Multiple samples — pass them positionally, so shell globs work.\n\
  Each sample becomes a batch via its barcodes' `@batch` tag.\n\n\
  senna gem out/rep1_genes.zarr.zip out/rep2_genes.zarr.zip -o out/gem\n\
  senna gem out/*_genes.zarr.zip -o out/gem\n\n\
  The `--genes a,b` flag form still works, but not together with the positional one.")]
    Gem(GemArgs),

    #[command(
        name = "gem-encoder",
        // `gem-topic`: the cell latent IS a softmax simplex, so this is a
        // topic model over the two splice tracks — the name people reach for
        // when they come from `senna topic` rather than from `senna gem`.
        visible_aliases = ["gem-topic"],
        aliases = ["gem-enc"],
        about = "GEM-encoder: a masked generative model of the GEM",
        long_about = "GEM-encoder — the masked generative sibling of `senna gem`.\n\
                      \n\
                      Both fit the same geometry over the same spliced+unspliced counts,\n\
                      from opposite directions.\n\
                      `gem` is discriminative (NCE over cell-feature edges).\n\
                      This is generative and amortized:\n\
                      an encoder reads a cell\'s top-K GENES with BOTH splice tracks attached,\n\
                      pools each track over that context (not over the full gene space),\n\
                      and an embedded-topic decoder imputes whichever track was held out.\n\
                      \n\
                      The model runs the biology forward:\n\
                      u + delta -> s. Nascent pre-mRNA is transcribed first and matures into spliced mRNA,\n\
                      so the UNSPLICED embedding is the base rho and the spliced one is rho + delta.\n\
                      Delta is therefore the steady-state splice-ratio offset —\n\
                      log(splicing / degradation), not a splicing rate —\n\
                      because that is the combination that survives at steady state (s = (beta/gamma) u).\n\
                      A gene scores high either by splicing fast or by having stable mature mRNA,\n\
                      and this model cannot tell those apart.\n\
                      NOTE this is the OPPOSITE base from `senna gem`,\n\
                      whose delta shifts spliced -> unspliced;\n\
                      the two write same-named delta_feature_embedding.parquet files that are NOT comparable.\n\
                      `{out}.gem.json` records `delta_base`.\n\
                      \n\
                      Training masks a fraction of GENES with ONE draw shared by both tracks,\n\
                      and predicts both from ONE theta. That gives delta a monopoly:\n\
                      the only thing that can make the two tracks differ is delta itself.\n\
                      Hiding a whole track instead was tried and removed —\n\
                      it hands the encoder a competing LATENT delta, which it takes,\n\
                      and delta degenerates.\n\
                      \n\
                      VELOCITY is the cell-level delta = theta_nascent - theta_mature,\n\
                      each fitted POST HOC to its own track against the frozen dictionaries.\n\
                      Elliptical slice sampling, warm-started from the encoder,\n\
                      which also closes the amortization gap.\n\
                      The model has one latent by design,\n\
                      so it cannot express that difference while training;\n\
                      estimating delta first and reading the movement out of it keeps the two from competing.\n\
                      The per-axis population mean is removed before writing,\n\
                      and recorded in `{out}.gem.json` as `velocity_common_mode`.\n\
                      \n\
                      The latent is a softmax simplex — hence the `gem-topic` alias —\n\
                      and `{out}.latent.parquet` holds LOG THETA, so theta = exp(row),\n\
                      the same contract every senna topic-family run follows.\n\
                      Pick the loss with `--likelihood nb|multinomial`\n\
                      \n\
                      BATCH ADJUSTMENT IS ON BY DEFAULT,\n\
                      and you should check what your batches are:\n\
                      with several inputs and no `--batch-files`,\n\
                      each file's cells are tagged `@<sample>` and that tag becomes the batch —\n\
                      so on rep{1,2,3}_{wt,mut} the batches are the SIX samples,\n\
                      and the wt-vs-mut contrast goes out with the donor effects.\n\
                      Pass `--batch-files` with the labels you mean, or `--no-batch-adjust`.\n\
                      \n\
                      Pooling is a masked value-weighted sum per track, concatenated;\n\
                      the attention-slot variant was removed after it measured\n\
                      3.5x worse on between-cell variance,\n\
                      and went degenerate whenever a track was hidden.\n\
                      Ctrl-C stops training gracefully and still writes outputs,\n\
                      flagged as partial.",
        after_long_help = "\
	Example:\n\
  senna gem-encoder out/rep2_wt_genes.zarr.zip out/rep2_mut_genes.zarr.zip \\\n\
    -o out/gme -t 20 --device cuda\n\n\
  senna gem-encoder out/*_genes.zarr.zip -o out/gme --likelihood nb\n\n\
  Watch |delta| and the splice-ratio r in the log.\n\
  If delta collapses toward 0, or r is near 0, the velocity is not trustworthy.")]
    GemEncoder(GemEncoderArgs),

    #[command(
        name = "annotate-gem",
        aliases = ["annot-gem", "ann-gem"],
        about = "Marker-set cell-type annotation of a `senna gem` or `gem-encoder` run",
        long_about = "Annotate a gem-family run against a marker set.\n\
                      \n\
                      Reads the run's parquet outputs by prefix (`-f/--from`),\n\
                      plus a marker TSV (`gene<TAB>celltype`, `-m/--markers`).\n\
                      Then runs the shared term-ORA core:\n\
                      assign → distance-outlier QC → Leiden clustering,\n\
                      then cluster×term hypergeometric over-representation,\n\
                      permutation-calibrated.\n\
                      \n\
                      TWO SCORERS (`--mode`), and they are not two flavours of one statistic:\n\
                      they read different files and rest on different assumptions about the geometry.\n\
                      The default is read from `{from}.gem.json` rather than fixed,\n\
                      because the wrong one here does not error — it answers wrong.\n\
                      An embedding run (`senna gem`) → `projection`;\n\
                      a topic model (`senna gem-encoder` / `gem-topic`) → `enrichment`.\n\
                      A prefix that cannot say what produced it is reported, not guessed at.\n\
                      \n\
                      `projection` builds each type's centroid from its markers' CO-EMBEDDED feature vectors,\n\
                      then hands every cell to the nearest one.\n\
                      It reads `feature_embedding.parquet` plus `cell_embedding.parquet`.\n\
                      NOT the raw `beta_feature_embedding` or `delta_feature_embedding`,\n\
                      which are model parameters off the cell manifold. Its tracks:\n\
                      spliced:  /count/spliced rows   vs cell θ         → {out}.spliced.*\n\
                      velocity: /count/unspliced rows vs cell velocity  → {out}.velocity.*\n\
                      \n\
                      `enrichment` never forms a cell-gene inner product —\n\
                      on a topic model that is not a metric,\n\
                      since β depends only on gene-to-gene differences,\n\
                      and the absolute direction is a gauge the likelihood never pins.\n\
                      It asks per factor whether a type's panel is over-represented at the top of that factor's gene ranking,\n\
                      then carries the surviving factor×type edges to cells through θ.\n\
                      It reads `dictionary.parquet`, `latent.parquet`,\n\
                      `pb_latent.parquet` and `pb_gene.parquet` → {out}.enrichment.* Its tracks are `spliced` and `nascent`,\n\
                      NOT velocity: a displacement has no membership to carry a call through.\n\
                      `nascent` annotates the nascent PROGRAM — a state the cell is in,\n\
                      on the simplex —\n\
                      and reading it against `spliced` is the well-posed form of the question `velocity` asks.\n\
                      \n\
                      `--track both` (default) runs both of whichever pair applies;\n\
                      the second track is skipped with a warning when its inputs are absent.",
        after_long_help = "\
	Example:\n\
	senna gem --genes out/rep1_genes.zarr.zip -o out/gem\n\
  senna annotate-gem -f out/gem -m markers.tsv -o out/gem"
    )]
    AnnotateGem(AnnotateGemArgs),

    #[command(
        name = "lineage",
        aliases = ["trajectory", "traj"],
        about = "Velocity-oriented lineage + principal curves over a `senna gem` run",
        long_about = "Infer a velocity-oriented lineage over the embeddings from `senna gem`.\n\n\
            Reads a θ/δ pair by prefix (`-f/--from`), picked by `--theta-from`:\n\
            on an EMBEDDING run, cell_embedding.parquet + velocity.parquet (H space);\n\
            on a TOPIC run, latent.parquet + velocity_factor.parquet (the K-space simplex).\n\
            The topic default is deliberate:\n\
            `cell_embedding = θ·α` confines every cell to the convex hull of α's K rows,\n\
            so a diffuse softmax θ compresses the population toward that hull's centroid —\n\
            blobby for reasons no layout can undo.\n\
            `--latent-geometry` sets the metric (Hellinger on a simplex, else cosine).\n\
            Fits K k-means centroids on θ and an MST over them,\n\
            then TESTS the velocity direction of every candidate edge\n\
            (bootstrap CI + sign-flip permutation; an edge that cannot clear\n\
            --edge-alpha abstains rather than being handed a direction).\n\
            Maximum-weight branching turns those calls into a rooted FOREST —\n\
            contradictions cut, weak parents rewired —\n\
            so a dataset with disconnected structure yields several trees, not one forced tree.\n\
            Slingshot-style smooth principal curves are then fit per tree\n\
            → per-cell pseudotime + branch.\n\
            Cells on a tree too small to carry a curve get NaN pseudotime\n\
            (reported, and skipped by `senna dyn-assoc`).\n\
            `--no-edge-direction` keeps the geometric MST instead;\n\
            `--no-orient-velocity` ignores velocity entirely.\n\n\
            Root selection (priority order):\n\
            --root-node, --root-cell, --root-type (marker-grounded, needs --markers),\n\
            --root-from-gem (gem's velocity-DAG source),\n\
            else the velocity-flux source.\n\n\
            The low-coverage modalities are NOT embedded here;\n\
            this produces the lineage ordering that a separate confounder-adjusted test runs against.\n\n\
            Outputs (all `{out}`-prefixed parquet):\n\
            nodes, node_velocity,\n\
            edges (every candidate edge with its velocity_flux, CI, q and call),\n\
            trees (the selected branching), lineages, pseudotime,\n\
            cell_lineage_weights, lineage_pseudotime, curves;\n\
            with --markers also lineage_annot.* + trajectory_annotation;\n\
            with --layout phate (default) or umap also {cells,nodes,curves}_2d,\n\
            plus velocity_grid_2d (the gridded δ arrow field, when the run has δ).\n\
            The layout embeds θ alone by default, so position means identity,\n\
            and the arrow field carries the direction.\n\n\
            Reference:\n  \
            Street et al., \"Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics\",\n\
            BMC Genomics, 19:477, 2018.\n\
            https://doi.org/10.1186/s12864-018-4772-0",
        after_long_help = "\
	Example:\n\
	senna gem --genes out/rep1_genes.zarr.zip -o out/gem\n\
  senna lineage -f out/gem -o out/gem"
    )]
    Lineage(LineageArgs),

    #[command(
        name = "lineage-plot",
        aliases = ["plot-lineage", "trajectory-plot"],
        about = "Publication-style figure (PDF/PNG/SVG) of a `senna lineage` trajectory over its 2D embedding",
        long_about = "Render the outputs of `senna lineage --markers` into one annotated figure,\n\
                      with the default --layout phate: cells laid out on the PHATE embedding,\n\
                      coloured by coarse cell type (default) or pseudotime,\n\
                      with a trajectory backbone, velocity arrows and MST nodes overlaid.\n\
                      \n\
                      Reads by prefix (`-f/--from`): {from}.cells_2d.parquet (PHATE coords),\n\
                      {from}.lineage_annot.annot.parquet (per-cell coarse_label),\n\
                      {from}.curves_2d.parquet (principal curves),\n\
                      {from}.nodes_2d.parquet (MST nodes),\n\
                      {from}.trajectory_annotation.parquet (node role/cell_type),\n\
                      and {from}.pseudotime.parquet (for --color-by pseudotime).\n\
                      \n\
                      The cells are drawn as transparent raster layers per cell type,\n\
                      from a qualitative palette, with a legend — confident calls solid,\n\
                      mixed ones faded —\n\
                      or one continuous blue->red pseudotime layer (with a colourbar).\n\
                      \n\
                      The backbone is `--trajectory auto` by default:\n\
                      the Slingshot principal curves when the run has few lineages,\n\
                      otherwise the MST drawn ONCE, with stroke weight by traversal count.\n\
                      The curves all share the trunk,\n\
                      so past ~24 lineages they overplot into an opaque mat.\n\
                      Force it with `tree`, `curves` or `none`.\n\
                      Direction is ALWAYS shown as velocity arrows read off `velocity_flux`,\n\
                      independent of that choice, and only on edges whose velocity earned one.\n\
                      Nodes are dark overlays; the root is marked with a red star,\n\
                      and `--label-nodes` (default `per-type`) labels one node per called cell type,\n\
                      on its most-differentiated node.\n\
                      Uses the shared plot-utils rasterize -> SVG -> render pipeline;\n\
                      writes {out}.plot.pdf by default (--png / --svg add those formats, --no-pdf skips the PDF).\n\
                      The scatter is a raster layer,\n\
                      so the PDF is a hybrid (vector text over raster points at --dpi; raise --dpi to 300-600 for print).",
        after_long_help = "\
	Example:\n\
	senna lineage -f out/gem -o out/lin --markers markers.tsv\n\
	senna lineage-plot -f out/lin\n\
  senna lineage-plot -f out/lin -o out/lin_pt --color-by pseudotime"
    )]
    LineagePlot(LineagePlotArgs),

    #[command(
        name = "dyn-assoc",
        aliases = ["assoc", "temporal-assoc", "trend"],
        about = "Bayesian between-branch modality contrast along a `senna lineage`",
        long_about = "Test whether a modality (m6a/apa/atoi) diverges between lineage branches.\n\n\
            Downstream of `senna lineage` (like `annotate` is to `gem`).\n\
            Cells are pooled into pseudotime BINS,\n\
            and each branch L is tested against the rest with a binomial GLM:\n\
            logit(p_{b,g}) = α_b + β·1[g=L],\n\
            where b indexes the bin.\n\
            The per-bin baseline α_b conditions out pseudotime,\n\
            a matched null, à la tradeSeq patternTest / cocoa,\n\
            so β is the branch's pseudotime-adjusted log-odds excess.\n\
            Coverage (edited + unedited) is the binomial denominator,\n\
            so detection bias is conditioned out;\n\
            a shrinkage prior N(0, τ²) on β damps noisy calls,\n\
            stable across seeds, with no permutation machinery.\n\
            Reports the posterior mean effect, 90% credible interval,\n\
            and lfsr = min(P(β>0), P(β<0));\n\
            the within-branch trend GAM (--trend-method) runs alongside.\n\n\
            Each level writes three tables —\n\
            {out}.branch_contrast / _profile / _trend.parquet.\n\n\
            If the lineage was annotated (`senna lineage --markers`, which leaves a\n\
            {from}.lineage_annot.membership.tsv — or point `--celltype-annot` at any\n\
            `cell<TAB>cell_type` TSV), the same two tests are also reported per CELL TYPE —\n\
            cells sharing an annotated type are pooled across lineages\n\
            ({out}.celltype_contrast / _profile / _trend.parquet).\n\
            The between-cell-type contrast is the clean deliverable;\n\
            the within-cell-type trend is secondary\n\
            (pooling divergent lineages onto one pseudotime axis weakens the trend reading).\n\
            Skip with --no-celltype.\n\n\
            Not double-dipping:\n\
            branches come from gem θ + velocity, which never see the modality.\n\n\
            Output is tidy:\n\
            `site | gene | subunit | branch` (branch level) or\n\
            `site | gene | subunit | cell_type` (cell-type level —\n\
            no branch column, because a cell-type aggregate pools cells across branches),\n\
            then the values.\n\
            The Bayesian tables also carry `ess` and `mcse_lfsr`:\n\
            lfsr is a Monte-Carlo tail probability,\n\
            so a site near --fdr-alpha can cross it with the seed,\n\
            and mcse_lfsr is that error per site.\n\
            When |lfsr - alpha| is not comfortably above mcse_lfsr,\n\
            raise --posterior-samples rather than reading anything into the effect.\n\n\
            Reference:\n  \
            Van den Berge et al., \"Trajectory-based differential expression analysis for single-cell sequencing data\",\n\
            Nat Commun 11:1201, 2020.",
        after_long_help = "\
	Example:\n\
	senna lineage -f out/gem -o out/lin --markers markers.tsv\n\
  senna dyn-assoc -f out/lin -s out/rep1_wt_m6a_site.zarr.zip --modality m6a -o out/m6a_assoc"
    )]
    Assoc(AssocArgs),

    #[command(
        about = "2D layout of cells (tsne / umap / phate) over batch-corrected pseudobulks.",
        long_about = "Builds PBs by batch-corrected multi-level collapsing.\n\
                      PB-PB cosine similarity is computed on log1p-CPM gene vectors.\n\
                      The chosen method lays those out.\n\
                      Every cell is then projected via Nyström.\n\
                      \n\
                      Updates `manifest.layout.{cell_coords, pb_coords, pb_gene_mean}`.\n\
                      `senna plot --from ...` then picks the layout up automatically.\n\
                      \n\
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
        long_about = "`senna plot --from run.senna.json` reads the manifest.\n\
                      It takes cell_coords, topics, annotation, clusters,\n\
                      labels and palette from there. It renders a 300-dpi rasterized scatter,\n\
                      with vector labels.\n\
                      \n\
                      Auto-fills missing pieces:\n  \
                      • no `layout.cell_coords` → runs `senna layout umap` first.\n  \
                      • `--colour-by cluster` but no clusters → runs Leiden on the latent.\n\
                      \n\
                      --colour-by cluster (default) | annotation | topic | pb-id | pseudotime.\n\
                      The default flips to `annotation` once the manifest has one.\n\
                      `senna annotate-by-enrichment` is what populates it.\n\
                      Cells are then coloured and labelled by predicted cell type.\n\
                      \n\
                      Outputs {out}.plot.{svg,png,pdf}. PDF is the default;\n\
                      pass --svg or --png for those."
    )]
    Plot(PlotArgs),

    #[command(
        about = "Topic-model diagnostics:\n\
                 per-batch structure bars + gene × topic dictionary.",
        long_about = "Admixture-style stacked-bar structure plots, one per batch.\n\
                      Panel width is ∝ #cells. A gene × topic dictionary summary follows.\n\
                      It is a Hinton plot at ≤ 100 genes, with a heatmap above.\n\
                      \n\
                      Usage: senna plot-topic --from run.senna.json\n\
                      \n\
                      PDF only by default; pass --svg / --png to also emit those.\n\
                      Outputs land under {out}.plots/{struct,dict}/.",
        visible_alias = "pt"
    )]
    PlotTopic(PlotTopicArgs),

    #[command(
        about = "Watson/Crick mirrored genomic-activity ideograms (Strand-seq style).",
        long_about = "Per-chromosome gene activity, split by strand, per cell type.\n\
                      Forward/Watson genes form a filled pileup rising upward.\n\
                      Reverse/Crick genes mirror downward. Both share one chromosome axis.\n\
                      \n\
                      Usage: senna plot-strand --from run.senna.json --gtf gencode.gtf\n\
                      \n\
                      Activity defaults to a gene × cell-type matrix.\n\
                      It is derived from `senna annotate-by-enrichment` outputs.\n\
                      Override it with --activity.\n\
                      \n\
                      One figure is written per cell type, chromosomes stacked.\n\
                      An optional consensus figure joins them, under {out}.strand/.\n\
                      PDF only by default; pass --svg or --png.",
        visible_alias = "ps"
    )]
    PlotStrand(PlotStrandArgs),
}

#[derive(Subcommand, Debug)]
enum LayoutCmd {
    #[command(about = "PHATE diffusion embedding of pseudobulks (recommended default).")]
    Phate(LayoutPhateArgs),
    #[command(
        about = "t-SNE of pseudobulks on raw-gene similarity (random init).",
        long_about = "t-SNE layout of pseudobulks.\n\
                      \n\
                      Similarity is computed on raw genes.\n\
                      The embedding starts from a random initialization.\n\
                      Cells are then placed from their pseudobulk coordinates."
    )]
    Tsne(LayoutTsneArgs),
    #[command(
        about = "UMAP-style SGD of pseudobulks over the fuzzy kNN graph.",
        long_about = "UMAP-style layout of pseudobulks.\n\
                      \n\
                      A fuzzy kNN graph is built over the pseudobulks.\n\
                      Attractive and repulsive forces are then optimized by SGD.\n\
                      Cells are placed from their pseudobulk coordinates."
    )]
    Umap(LayoutUmapArgs),
    #[command(
        about = "Reingold-Tilford tree layout from a pseudotime run.",
        long_about = "Reads the principal graph and root node from `manifest.pseudotime`,\n\
                      written by `senna pseudotime`. It then produces a top-down tree layout.\n\
                      y is geodesic pseudotime; x is sibling order.\n\
                      \n\
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
        Commands::MaskedSbp(args) => {
            fit_masked_sbp_model(args)?;
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
        Commands::AnnotateOntology(args) => {
            annotate_ontology(args)?;
        }
        Commands::AnnotateByProjection(args) => {
            annotate_by_projection(args)?;
        }
        Commands::Deconvolve(args) => {
            deconvolve::run(args)?;
        }
        Commands::Predict(args) => {
            predict_model(args)?;
        }
        Commands::Probe(args) => {
            run_probe(args)?;
        }
        Commands::Update(args) => {
            run_update(args)?;
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
        Commands::Gem(ref args) => run_gem_embedding(args)?,
        Commands::GemEncoder(ref args) => run_gem_encoder(args)?,
        Commands::AnnotateGem(ref args) => run_annotate_gem(args)?,
        Commands::Lineage(ref args) => run_lineage(args)?,
        Commands::LineagePlot(ref args) => run_lineage_plot(args)?,
        Commands::Assoc(ref args) => run_assoc(args)?,
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
