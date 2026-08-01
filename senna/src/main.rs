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
mod counterfactual;
mod deconvolve;
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
mod probe;
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
    AnnotateOntologyArgs, AnnotateProjectionArgs,
};
use bge::{fit_bge, BgeArgs};
use clustering::*;
use deconvolve::DeconvolveArgs;
use embed_common::*;
use eval_topic::*;
use fne::{fit_fne, FneArgs};
use impute::{impute_model, ImputeArgs};
use joint_topic::*;
use masked_topic::*;
use postprocess::*;
use predict::{predict_model, PredictArgs};
use probe::{run_probe, ProbeArgs};
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
                  Input: sparse backends in `.zarr` or `.h5`.\n\
                  Convert from Matrix Market with `data-beans from-mtx`.\n\n\
                  Each step writes its outputs back to the run manifest\n\
                  `{prefix}.senna.json`.\n\
                  Downstream commands read data and batch files from it.\n\
                  Steps 3 and 5 still need their own --latent / --out.\n\n  \
                  1. Train embedding   senna topic | masked-topic | svd\n                       \
                                       senna joint-topic | joint-svd       (multi-modality)\n  \
                  2. Held-out inference senna predict                       (apply trained model)\n  \
                  3. Cluster cells     senna clustering --from run.senna.json --latent L --out O\n  \
                  4. Annotate cells    senna annotate-by-enrichment --from run.senna.json -m markers.tsv\n  \
                  5. Trajectory        senna pseudotime --from run.senna.json --out O\n  \
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
                      Stages:\n\
                      \x20 1. batch-aware pseudobulk collapsing\n\
                      \x20 2. encoder-decoder VAE via SGD\n\
                      \x20 3. per-cell topic inference\n\n\
                      Decoders are multinom, nb and nbmixture (the default).\n\
                      Combine them with a comma-separated --decoder.\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.model.json, {out}.senna.json (run manifest)."
    )]
    Topic(TopicArgs),

    #[command(
        name = "masked-topic",
        about = "Train a masked-imputation embedded topic model (foundation-style).",
        long_about = "Embedded topic model trained by masked-gene imputation.\n\
                      There is no ELBO, and no posterior collapse.\n\
                      Encoder and decoder share a per-gene symbol embedding\n\
                      ρ ∈ ℝ^{D×H}, after Dieng et al. 2020 (ETM).\n\
                      The encoder pools a per-cell top-K feature window by\n\
                      single-query attention.\n\n\
                      Training splits each cell's top-K genes into visible and masked.\n\
                      θ_n = softmax(encoder(visible)), deterministic and KL-free.\n\
                      The NB head imputes the held-out genes with\n\
                      μ = residual · ℓ · (θ·β).\n\
                      There β_kg = softmax_g(α_k · ρ_g), and φ_g is a per-gene\n\
                      dispersion.\n\
                      \n\
                      The masked objective prevents collapse, not a KL bottleneck.\n\
                      So it scales with more data.\n\
                      Inference is encoder-only.\n\n\
                      Writes the same artifacts as `topic`.\n\
                      It adds `{out}.feature_embedding.parquet` (ρ) and\n\
                      `{out}.dispersion.parquet`.",
        visible_aliases = ["mtm"],
        aliases = ["itopic", "indexed-topic", "etm"]
    )]
    MaskedTopic(MaskedTopicArgs),

    #[command(
        name = "masked-vae",
        about = "Train a masked-imputation Gaussian VAE (BERT-style, continuous latent).",
        long_about = "Masked-imputation VAE.\n\
                      It is the Gaussian-latent sibling of `masked-topic`.\n\
                      The pipeline is the same: PB-collapse training, a shared\n\
                      per-gene ρ embedding, an NB ETM head, encoder-only inference.\n\
                      \n\
                      The encoder differs: it emits a reparameterized Gaussian\n\
                      latent z, with no simplex softmax, regularized by a KL term.\n\
                      That is a true variational bottleneck.\n\
                      exp(z) drives the NB head's per-topic intensities,\n\
                      μ_g = ℓ·Σ_t exp(z_t)·β_{t,g}.\n\
                      \n\
                      So the masked objective and the KL together train an\n\
                      unconstrained continuous embedding.\n\
                      The masked decoder is reused unchanged.\n\
                      Held-out genes are imputed, and the masked objective —\n\
                      not the KL alone — keeps the latent from collapsing.\n\n\
                      Writes the same artifacts as `masked-topic`.\n\
                      The NB objective is the only one available.",
        visible_aliases = ["bert"]
    )]
    MaskedVae(MaskedTopicArgs),

    #[command(
        name = "masked-sbp",
        about = "Train a masked-imputation topic model with a stick-breaking-process simplex.",
        long_about = "Stick-breaking-process (SBP) sibling of `masked-topic`.\n\
                      The masked-imputation pipeline is the same: a shared per-gene\n\
                      ρ embedding, an NB ETM head, a deterministic KL-free\n\
                      objective, encoder-only inference.\n\
                      \n\
                      The encoder differs: it maps its logits through a\n\
                      stick-breaking simplex instead of a softmax,\n\
                      θ_k = v_k·∏_{j<k}(1−v_j) with v_k = σ(η_k).\n\
                      \n\
                      Topics are therefore no longer exchangeable.\n\
                      Early sticks carry more mass a priori.\n\
                      That gives an intrinsic ordering and a self-pruning tail:\n\
                      later topics shrink toward 0 unless the data needs them.\n\
                      It is a soft, differentiable way to over-provision K and prune.\n\n\
                      Writes the same artifacts as `masked-topic`.",
        visible_aliases = ["sbp"]
    )]
    MaskedSbp(MaskedTopicArgs),

    #[command(
        about = "Train an scVI-style Gaussian VAE (continuous factor model).",
        long_about = "Gaussian (scVI-style) VAE.\n\
                      It is the continuous-latent sibling of `topic`.\n\
                      The pipeline is the same: batch-aware pseudobulk collapse,\n\
                      then a dense VAE.\n\
                      \n\
                      The encoder emits an unconstrained Gaussian latent z,\n\
                      with no simplex projection.\n\
                      The NB decoder maps z → π = softmax_d(z·W) → μ = library·π.\n\
                      \n\
                      Outputs are continuous factors, cell × factor, plus\n\
                      gene × factor loadings.\n\
                      They are not topic proportions and a topic-gene dictionary.\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.safetensors,\n\
                      {out}.model.json, {out}.senna.json (run manifest)."
    )]
    Vae(VaeArgs),

    #[command(
        about = "Train Nyström SVD embedding.",
        long_about = "Three stages:\n\
                      \x20 1. batch-aware pseudobulk collapsing\n\
                      \x20 2. randomized SVD\n\
                      \x20 3. per-cell Nyström projection\n\n\
                      Writes {out}.{latent,dictionary}.parquet, {out}.senna.json."
    )]
    Svd(SvdArgs),

    #[command(
        about = "Train joint topic model across modalities (independent or delta decoder).",
        long_about = "Joint topic-model embedding over modalities sharing cells.\n\
                      Data files form a row-major (modality × batch) table.\n\
                      -m sets the modality-row count.\n\n\
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
        long_about = "Joint SVD over a stack of modalities sharing cells.\n\
                      Data files form a row-major (modality × batch) table.\n\
                      -m sets the modality-row count.\n\
                      Cells must be shared; features may differ.\n\n\
                      Writes {out}.latent.parquet, {out}.senna.json."
    )]
    JointSvd(JointSvdArgs),

    #[command(
        about = "Train graph-based embedding (count-NCE, modality-agnostic).",
        long_about = "Joint embedding of features and cells in one H-dim space.\n\
                      It uses discriminative count-NCE on a sketch-coarsened\n\
                      pseudobulk (cell, feature) bipartite graph.\n\
                      \n\
                      Each input file contributes its rows to a shared feature axis.\n\
                      Cell barcodes union across files.\n\
                      The method is modality-agnostic, so any number of count\n\
                      panels works: RNA, ATAC, protein and so on.\n\
                      Scoring is bilinear: `E_f · E_c + b_f + b_c`.\n\
                      \n\
                      Positives are drawn by a two-stage stratified sampler.\n\
                      Stage 1 picks a pseudobulk with q(p) ∝ pb_size(p)^alpha_pb.\n\
                      Stage 2 picks a feature within it, weighted by μ_pf.\n\
                      Negatives are drawn UNIFORMLY over the global pool of\n\
                      expressed features, so they are abundance-independent.\n\n\
                      Training runs in two phases.\n\
                      Phase 1 embeds features and pseudobulks, learning the\n\
                      gene side.\n\
                      Phase 2 freezes that and densely fits each cell embedding.\n\
                      Every cell is swept about once per epoch.\n\
                      The per-cell fit is separable, so it is embarrassingly parallel.\n\n\
                      Writes {out}.{cell_embedding,dictionary,feature_embedding,\n\
                      feature_bias,cell_bias}.parquet and {out}.senna.json.\n\
                      The H-space cell embedding Z is always\n\
                      {out}.cell_embedding.parquet.\n\
                      \n\
                      Unless --skip-etm, an ETM is resolved too.\n\
                      That adds {out}.{latent,topic_embedding}.parquet,\n\
                      with latent = log θ.",
        alias = "embed-graph",
        alias = "gbe"
    )]
    Bge(BgeArgs),

    #[command(
        about = "Latent feature model over a feature-feature edge list.",
        long_about = "Learns per-feature latent embeddings from an edge list.\n\
                      No expression data is involved.\n\
                      \n\
                      Input is a TSV/CSV of feature-feature edges.\n\
                      BioGRID, STRING, KEGG, synthetic-lethality and regulatory\n\
                      networks all fit.\n\
                      \n\
                      Embeddings E ∈ ℝ^{D×H} come from a continuous\n\
                      Miller-Griffiths-Jordan link-prediction model:\n  \n  \
                      s(i, j) = (E_i ⊙ γ) · E_j + b_i + b_j\n  \n\
                      Training is binary cross-entropy with degree^α negative\n\
                      sampling, the node2vec convention.\n\
                      The model is symmetric by construction.\n\n\
                      Writes {out}.feature_embedding.parquet.\n\
                      feature_bias, gamma, log_likelihood and senna.json ship too.\n\
                      \n\
                      The output shape matches the freeze loader behind\n\
                      `senna masked-topic --freeze-feature-embedding`.\n\
                      An `fne` run is therefore a direct gene-side input to\n\
                      downstream cell-side training."
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
                      It then trains a gene embedding ρ ∈ ℝ^{D×H} and a topic\n\
                      embedding α ∈ ℝ^{K×H} against the raw counts, by\n\
                      bipartite NCE.\n\
                      The cell embedding is derived from frozen θ as Z = θ·α:\n  \n  \
                      score(cell c, gene g) = (θ_c·α)·ρ_g + b_g\n  \n\
                      Writes {out}.{feature_embedding,cell_embedding,topic_embedding}\
                      .parquet, plus senna.json with kind=resolve-embedding-space.\n\
                      \n\
                      The result is a metric H-space where genes, topics and cells\n\
                      coexist.\n\
                      Downstream clustering and `senna annotate-by-enrichment` read it.\n\
                      H defaults to K, but may exceed it."
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
        about = "Drift probe: novelty verdict for held-out data vs a trained masked model.",
        long_about = "Read-only drift probe (the covered-vs-new gate). Scores query cells'\n\
                      per-cell predictive fit under a trained masked model\n\
                      (masked-topic/-vae/-sbp), calibrates a null from an in-distribution\n\
                      --calibration backend, and flags query cells below the null tail.\n\
                      Emits a batch-level covered/novel verdict.\n\n\
                      Usage: senna probe --model M --calibration ref.zarr query.zarr -o out\n\
                      Writes {out}.probe.tsv (per-cell fit + flag) and {out}.probe.json."
    )]
    Probe(ProbeArgs),

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
        name = "annotate-by-projection",
        visible_aliases = ["ann-by-proj", "annot-by-proj"],
        about = "Annotate cells via firm marker over-representation on the co-embedding.",
        long_about = "Embedding-grounded alternative to `annotate-by-enrichment` for runs with a\n\
                      co-embedded gene space (bge / fne / resolve-embedding-space). Pipeline:\n\
                      build each type's IDF-weighted marker centroid → Euclidean\n\
                      nearest-centroid per cell → distance-outlier QC → Leiden cluster →\n\
                      cluster × term hypergeometric over-representation, permutation-calibrated\n\
                      → per-cluster call broadcast to cells. Optional TreeBH ontology with\n\
                      --obo/--label-cl. Never re-reads raw counts (complementary to\n\
                      enrichment, which is raw-count-grounded).\n\n\
                      Usage: senna annotate-by-projection --from run.senna.json -m markers.tsv -o out\n\
                      Writes {out}.{argmax.tsv,membership.tsv,annot.parquet,cluster_term_*.parquet,\n\
                      null_calibration.tsv}; updates `manifest.annotate.*`."
    )]
    AnnotateByProjection(AnnotateProjectionArgs),

    #[command(
        name = "deconvolve",
        visible_aliases = ["deconv", "deconvolution"],
        about = "Deconvolve bulk samples into cell-type fractions + per-type expression.",
        long_about = "Projection-based hierarchical-Bayes bulk deconvolution built on a feature\n\
                      embedding (`senna bge --skip-etm`, exact; or `masked-topic`, approximate).\n\
                      Reconstructs each cell type's gene profile from the embedding, projects\n\
                      bulk samples into the shared latent, and runs a full Gibbs sampler\n\
                      (Gamma-Poisson fractions + multinomial gene split + elliptical-slice anchor\n\
                      updates that carry annotate-by-projection uncertainty).\n\n\
                      Usage: senna deconvolve --from run.senna.json -m markers.tsv --bulk bulk.parquet\n\
                      Writes {out}.{fractions,fractions_ci,abundance,residual}.tsv,\n\
                      {out}.{sample_embedding,anchors}.parquet, {out}.expression/*.parquet."
    )]
    Deconvolve(DeconvolveArgs),

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
                      • no `layout.cell_coords` → runs `senna layout umap` first.\n  \
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
    #[command(
        about = "t-SNE of pseudobulks on raw-gene similarity (random init).",
        long_about = "t-SNE layout of pseudobulks.\n\n\
                      Similarity is computed on raw genes, and the embedding\n\
                      starts from a random initialization.\n\
                      Cells are then placed from their pseudobulk coordinates."
    )]
    Tsne(LayoutTsneArgs),
    #[command(
        about = "UMAP-style SGD of pseudobulks over the fuzzy kNN graph.",
        long_about = "UMAP-style layout of pseudobulks.\n\n\
                      A fuzzy kNN graph is built over the pseudobulks.\n\
                      Attractive and repulsive forces are then optimized by SGD.\n\
                      Cells are placed from their pseudobulk coordinates."
    )]
    Umap(LayoutUmapArgs),
    #[command(
        about = "Reingold-Tilford tree layout from a pseudotime run.",
        long_about = "Reads the principal graph + root node from `manifest.pseudotime`\n\
                      (written by `senna pseudotime`), then produces a top-down tree\n\
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
