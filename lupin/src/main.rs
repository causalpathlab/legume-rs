//! `lupin`: Lexicon-Using Projection for Identity Naming.

mod annotate;
mod describe;

use anyhow::Result;
use clap::{Parser, Subcommand};
use gene_text::cli::{run_knn_graph, run_qc, KnnGraphCmd, QcCmd};
use senna::assoc::run::{run_assoc, AssocArgs};
use senna::lineage::args::LineageArgs;
// `lineage` itself takes resolved paths; these adapters turn `-f run.senna.json`
// into those and record the artifacts they produce.
use senna::lineage_manifest::{run_lineage_from_manifest, run_pseudotime_from_manifest};
use senna::lineage_plot::{run_lineage_plot, LineagePlotArgs};
use senna::pseudotime::PseudotimeArgs;

use annotate::{run_annotate, AnnotateCliArgs};
use describe::{run_describe, DescribeArgs};

#[derive(Parser)]
#[command(
    name = "lupin",
    version,
    about = "Lexicon-Using Projection for Identity Naming — text graphs, annotation, lineage, describe."
)]
struct Cli {
    #[arg(short, long, global = true, help = "Verbose logging")]
    verbose: bool,
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(
        name = "text-qc",
        about = "Inspect the word vocabulary and its frequency cuts, without encoding",
        long_about = "The vocabulary step of `word-graph` on its own,\n\
                      so the cuts can be inspected before paying for the model pass.\n\
                      \n\
                      Tokenise every description, drop stopwords and filler,\n\
                      then cut both tails of the document-frequency distribution by quantile.\n\
                      Prints the df histogram, the cuts and the words on each side of them,\n\
                      and writes {out}.vocab.tsv.\n\
                      Tune the stopword list and the quantiles here,\n\
                      then hand the file to `word-graph --vocab-file`.\n\
                      `word-graph` runs this step itself when no file is given."
    )]
    TextQc(QcCmd),
    #[command(
        name = "word-graph",
        alias = "vocab-graph",
        about = "Encode the descriptions and write the text graph: feature–word and feature–feature edges",
        long_about = "Runs the vocabulary step,\n\
                      then a BERT-family encoder from the Hugging Face Hub over every description,\n\
                      and writes the text graph.\n\
                      \n\
                      {out}.feature_word.edges.tsv maps each feature to the words of its text\n\
                      (weight = contextual cosine × TF-IDF).\n\
                      {out}.knn_graph.edges.tsv lists the nearest features by text similarity.\n\
                      Both are typed edge files for `senna fne --edges`.\n\
                      Also writes {out}.text_embedding.parquet (pooled, centred),\n\
                      {out}.vocab.tsv and {out}.feature_text.tsv."
    )]
    WordGraph(KnnGraphCmd),
    #[command(
        name = "annotate",
        about = "Cell-type annotation by enrichment, embedding projection, or auto-dispatch",
        long_about = "Unified annotation entry point.\n\
                      \n\
                      `--method enrichment` runs the senna topic/svd enrichment pipeline.\n\
                      `--method projection` runs senna co-embed projection.\n\
                      With `--feature-embedding` / `--cell-embedding` (or pinto parquets),\n\
                      it runs pinto-style ORA instead.\n\
                      `--method auto` (default) picks projection when embeddings resolve, else enrichment.\n\
                      Ontology-only follow-up: `--from` + `--obo` + `--label-cl` without markers."
    )]
    Annotate(AnnotateCliArgs),
    #[command(
        name = "lineage",
        about = "Geometry-first lineage and principal curves over a senna gem run"
    )]
    Lineage(LineageArgs),
    #[command(
        name = "lineage-plot",
        aliases = ["plot-lineage", "trajectory-plot"],
        about = "Publication-style figure of a senna lineage trajectory over its 2D embedding"
    )]
    LineagePlot(LineagePlotArgs),
    #[command(
        name = "dyn-assoc",
        about = "Bayesian between-branch modality contrast along a senna lineage"
    )]
    DynAssoc(AssocArgs),
    #[command(
        name = "pseudotime",
        about = "Monocle-style principal-graph pseudotime from a senna latent embedding"
    )]
    Pseudotime(PseudotimeArgs),
    #[command(
        name = "describe",
        about = "Short citation-checked sentence from annotate / lineage_annot evidence",
        long_about = "Builds structured evidence from `{from}.annot.parquet` (or argmax / lineage_annot).\n\
                      Optionally fishes per-cluster keywords from a `word-graph` prefix\n\
                      (`feature_word.edges` over each cluster's markers;\n\
                      `--feature-embedding` adds nearest-neighbour genes first).\n\
                      When `{from}.cluster_term_q.parquet` is present,\n\
                      a second FDR-significant contender is named if one exists.\n\
                      Writes `{out}.describe.json` and `{out}.describe.md`.\n\
                      \n\
                      The composer never invents labels: sentences are citation-checked.\n\
                      Default composer is a citation-checked template (candle decoder TBD)."
    )]
    Describe(DescribeArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(if cli.verbose { "debug" } else { "info" }),
    )
    .init();
    match cli.cmd {
        Commands::TextQc(c) => run_qc(&c),
        Commands::WordGraph(c) => run_knn_graph(&c),
        Commands::Annotate(c) => run_annotate(&c),
        Commands::Lineage(c) => run_lineage_from_manifest(&c),
        Commands::LineagePlot(c) => run_lineage_plot(&c),
        Commands::DynAssoc(c) => run_assoc(&c),
        Commands::Pseudotime(c) => run_pseudotime_from_manifest(&c),
        Commands::Describe(c) => run_describe(&c),
    }
}
