//! `gene-text`: the text side of the feature graph. A small Hugging Face
//! encoder reads gene, term and pathway descriptions; a document-frequency
//! vocabulary keeps the words worth a node; and every feature is linked to
//! the words of its own text, weighted by the encoder's contextual
//! similarity times TF-IDF. The edges feed `senna fne --edges`; the pooled
//! vectors are a text prior for the gene side.

mod edges;
mod encoder;
mod sources;
mod vocab;

use anyhow::Result;
use auxiliary_data::feature_names::FeatureNameKind;
use candle_core::Device;
use clap::{Args, Parser, Subcommand, ValueEnum};
use edges::{
    centred_unit, csls_top_k, feature_word_weights, mean_of, score_doc, top_k_cosine,
    write_typed_edges,
};
use encoder::{Encoder, ModelSpec, Pooling};
use log::info;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use sources::Corpus;
use std::io::Write;
use vocab::{tokenize, DfQcOpts, Occurrence, TokenizeOpts, Vocabulary};

#[derive(Parser)]
#[command(
    name = "gene-text",
    version,
    about = "Descriptions in, gene–word edges and text vectors out"
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
        about = "Inspect the word vocabulary and its frequency cuts, without encoding",
        long_about = "The vocabulary step of `knn-graph` on its own,\n\
                      so the cuts can be inspected before paying for the model pass:\n\
                      tokenise every description, drop stopwords and filler,\n\
                      cut both tails of the document-frequency distribution by quantile.\n\
                      Prints the df histogram, the cuts and the words on each side of them,\n\
                      and writes {out}.vocab.tsv.\n\
                      Tune the stopword list and the quantiles here,\n\
                      then hand the file to `knn-graph --vocab-file`.\n\
                      `knn-graph` runs this step itself when no file is given."
    )]
    Qc(QcCmd),
    #[command(
        alias = "knn",
        about = "Encode the descriptions and write the text graph: feature–word and feature–feature edges",
        long_about = "Runs the vocabulary step,\n\
                      then a BERT-family encoder from the Hugging Face Hub over every description,\n\
                      and writes the text graph:\n\
                      {out}.feature_word.edges.tsv, each feature to the words of its text\n\
                      (weight = contextual cosine × TF-IDF),\n\
                      and {out}.knn_graph.edges.tsv, the nearest features by text similarity.\n\
                      Both are typed edge files for `senna fne --edges`.\n\
                      Also writes {out}.text_embedding.parquet (pooled, centred),\n\
                      {out}.vocab.tsv and {out}.feature_text.tsv."
    )]
    KnnGraph(KnnGraphCmd),
}

#[derive(Args, Clone)]
struct SourceArgs {
    #[arg(
        long,
        value_delimiter = ',',
        help = "Generic `feature type name text` TSV(s) (what `senna fne --export-text` writes)"
    )]
    text: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "UniProt TSV(s): gene primary, protein names, function"
    )]
    uniprot_tsv: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "NCBI gene_info file(s): symbol, synonyms, full name"
    )]
    gene_info: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "OBO ontologies: term names and definitions"
    )]
    obo: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "GMT gene sets: set names and descriptions"
    )]
    gmt: Vec<String>,
    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for canonical gene-name matching"
    )]
    feature_name_delim: char,
    #[arg(
        long,
        default_value_t = false,
        help = "Keep gene names exactly as given"
    )]
    feature_name_exact: bool,
}

impl SourceArgs {
    fn corpus(&self) -> Result<Corpus> {
        let kind = if self.feature_name_exact {
            FeatureNameKind::Exact
        } else {
            FeatureNameKind::Gene {
                delim: self.feature_name_delim,
            }
        };
        let mut c = Corpus::new(kind);
        for p in &self.text {
            c.add_generic_tsv(p)?;
        }
        for p in &self.uniprot_tsv {
            c.add_uniprot_tsv(p)?;
        }
        for p in &self.gene_info {
            c.add_ncbi_gene_info(p)?;
        }
        for p in &self.obo {
            c.add_obo(p)?;
        }
        for p in &self.gmt {
            c.add_gmt(p)?;
        }
        let dropped = c.retain_with_text();
        if dropped > 0 {
            info!("corpus: {dropped} features had no text and were dropped");
        }
        anyhow::ensure!(
            !c.is_empty(),
            "no descriptions read; pass --text, --uniprot-tsv, --gene-info, --obo or --gmt"
        );
        c.log_summary();
        Ok(c)
    }
}

#[derive(Args, Clone)]
struct QcArgs {
    #[arg(long, default_value_t = 3, help = "Shortest word kept (characters)")]
    min_chars: usize,
    #[arg(long, default_value_t = false, help = "Snowball-stem the words")]
    stem: bool,
    #[arg(
        long,
        help = "Extra stopwords, one per line, on top of the NLTK list and the built-in filler"
    )]
    extra_stopwords: Option<String>,
    #[arg(
        long,
        default_value_t = 0.05,
        help = "Drop words at or below the df of this vocabulary quantile (0 = off)"
    )]
    df_lower_quantile: f64,
    #[arg(
        long,
        default_value_t = 0.995,
        help = "Drop words at or above the df of this vocabulary quantile (1 = off)"
    )]
    df_upper_quantile: f64,
    #[arg(
        long,
        default_value_t = 2,
        help = "Absolute floor on df, applied on top of the quantile"
    )]
    min_df: usize,
    #[arg(
        long,
        default_value_t = 1.0,
        help = "Absolute cap on df as a fraction of the documents (1 = off)"
    )]
    max_df_frac: f64,
}

impl QcArgs {
    fn tokenize_opts(&self) -> Result<TokenizeOpts> {
        TokenizeOpts::new(self.min_chars, self.stem, self.extra_stopwords.as_deref())
    }
    fn df_opts(&self) -> DfQcOpts {
        DfQcOpts {
            lower_quantile: self.df_lower_quantile,
            upper_quantile: self.df_upper_quantile,
            min_df: self.min_df,
            max_df_frac: self.max_df_frac,
        }
    }
}

#[derive(Args)]
struct QcCmd {
    #[command(flatten)]
    sources: SourceArgs,
    #[command(flatten)]
    qc: QcArgs,
    #[arg(
        short,
        long,
        required = true,
        help = "Output prefix ({out}.vocab.tsv, {out}.feature_text.tsv)"
    )]
    out: String,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum PoolingArg {
    Mean,
    Cls,
}

#[derive(Args)]
struct KnnGraphCmd {
    #[command(flatten)]
    sources: SourceArgs,
    #[command(flatten)]
    qc: QcArgs,
    #[arg(
        long,
        help = "Reuse a tuned {out}.vocab.tsv from `qc` instead of recomputing the cuts"
    )]
    vocab_file: Option<String>,
    #[arg(
        long,
        default_value = "sentence-transformers/all-MiniLM-L6-v2",
        help = "Hugging Face model id (any BERT-family encoder)"
    )]
    model: String,
    #[arg(long, default_value = "main", help = "Model revision")]
    revision: String,
    #[arg(
        long,
        help = "Local model directory (config.json, tokenizer.json, model.safetensors); skips the Hub"
    )]
    model_dir: Option<String>,
    #[arg(long, value_enum, default_value_t = PoolingArg::Mean, help = "Document pooling")]
    pooling: PoolingArg,
    #[arg(
        long,
        default_value_t = 256,
        help = "Longest text in tokens; longer texts are truncated"
    )]
    max_tokens: usize,
    #[arg(long, default_value_t = 32, help = "Texts per forward pass")]
    batch: usize,
    #[arg(long, value_enum, default_value_t = ComputeDevice::Cpu, help = "Compute device")]
    device: ComputeDevice,
    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,
    #[arg(long, default_value_t = 32, help = "Words kept per feature, by weight")]
    words_per_feature: usize,
    #[arg(
        long,
        default_value_t = 0,
        help = "Also link each feature to this many nearest vocabulary words its text lacks (CSLS); 0 = off"
    )]
    expand_k: usize,
    #[arg(
        long,
        default_value_t = 10,
        help = "Feature–feature text-similarity edges per feature; 0 = off"
    )]
    knn: usize,
    #[arg(short, long, required = true, help = "Output prefix")]
    out: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(if cli.verbose { "debug" } else { "info" }),
    )
    .init();
    match cli.cmd {
        Commands::Qc(c) => run_qc(&c),
        Commands::KnnGraph(c) => run_knn_graph(&c),
    }
}

fn write_corpus(corpus: &Corpus, path: &str) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "feature\ttype\tname\ttext")?;
    for d in corpus.docs() {
        writeln!(w, "{}\t{}\t{}\t{}", d.feature, d.ty, d.name, d.text)?;
    }
    w.flush()?;
    Ok(())
}

fn tokenize_corpus(corpus: &Corpus, opts: &TokenizeOpts) -> Vec<(String, Vec<Occurrence>)> {
    corpus
        .docs()
        .par_iter()
        .map(|d| {
            let s = d.sentence();
            let occ = tokenize(&s, opts);
            (s, occ)
        })
        .collect()
}

fn run_qc(c: &QcCmd) -> Result<()> {
    matrix_util::common_io::mkdir_parent(&c.out)?;
    let corpus = c.sources.corpus()?;
    let opts = c.qc.tokenize_opts()?;
    let docs = tokenize_corpus(&corpus, &opts);
    let occ: Vec<Vec<Occurrence>> = docs.into_iter().map(|(_, o)| o).collect();
    let vocab = Vocabulary::build(&occ, &c.qc.df_opts());
    vocab.write_tsv(&format!("{}.vocab.tsv", c.out))?;
    write_corpus(&corpus, &format!("{}.feature_text.tsv", c.out))?;
    info!(
        "wrote {}.vocab.tsv ({} words kept; rare cut df ≤ {}, common cut df ≥ {}) and {}.feature_text.tsv",
        c.out,
        vocab.kept.len(),
        vocab.lower_cut.map_or("off".to_string(), |v| v.to_string()),
        vocab.upper_cut.map_or("off".to_string(), |v| v.to_string()),
        c.out
    );
    Ok(())
}

fn run_knn_graph(c: &KnnGraphCmd) -> Result<()> {
    matrix_util::common_io::mkdir_parent(&c.out)?;
    let corpus = c.sources.corpus()?;
    let opts = c.qc.tokenize_opts()?;
    let docs = tokenize_corpus(&corpus, &opts);
    let vocab = match &c.vocab_file {
        Some(p) => Vocabulary::read_tsv(p, docs.len())?,
        None => {
            let occ: Vec<Vec<Occurrence>> = docs.iter().map(|(_, o)| o.clone()).collect();
            Vocabulary::build(&occ, &c.qc.df_opts())
        }
    };
    vocab.write_tsv(&format!("{}.vocab.tsv", c.out))?;
    write_corpus(&corpus, &format!("{}.feature_text.tsv", c.out))?;
    anyhow::ensure!(!vocab.kept.is_empty(), "the vocabulary is empty after QC");

    let device = match c.device {
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::new_cuda(c.device_no)?,
        ComputeDevice::Metal => Device::new_metal(c.device_no)?,
    };
    let spec = ModelSpec {
        model_id: c.model.clone(),
        revision: c.revision.clone(),
        local_dir: c.model_dir.as_deref().map(std::path::PathBuf::from),
    };
    let pooling = match c.pooling {
        PoolingArg::Mean => Pooling::Mean,
        PoolingArg::Cls => Pooling::Cls,
    };
    let enc = Encoder::load(&spec, c.max_tokens, pooling, device.clone())?;
    let h = enc.hidden();
    let n = docs.len();
    let v = vocab.kept.len();

    // Encode in batches: pooled vectors, per-feature word weights, and the
    // running mean vector of every vocabulary word.
    let mut pooled: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut feature_words: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
    let mut word_sum: Vec<Vec<f32>> = vec![vec![0f32; h]; v];
    let mut word_n: Vec<u32> = vec![0; v];
    let mut n_missed = 0usize;
    let bar = indicatif_bar(n as u64);
    for chunk in docs.chunks(c.batch.max(1)) {
        let texts: Vec<String> = chunk.iter().map(|(s, _)| s.clone()).collect();
        let encs = enc.encode_batch(&texts)?;
        for (e, (_, occ)) in encs.iter().zip(chunk) {
            let scores = score_doc(&e.pooled, &e.tokens, occ, &vocab);
            n_missed += occ
                .iter()
                .filter(|o| vocab.index.contains_key(&o.word))
                .count()
                - scores.iter().map(|s| s.count as usize).sum::<usize>();
            for s in &scores {
                for (a, x) in word_sum[s.word].iter_mut().zip(&s.vec) {
                    *a += x;
                }
                word_n[s.word] += 1;
            }
            feature_words.push(feature_word_weights(&scores, &vocab, c.words_per_feature));
            pooled.push(e.pooled.clone());
        }
        bar.inc(chunk.len() as u64);
    }
    bar.finish_and_clear();
    if n_missed > 0 {
        info!("{n_missed} word occurrences fell beyond --max-tokens and were not scored");
    }

    // text_embedding.parquet: pooled vectors, centred.
    let center = mean_of(&pooled, h);
    let names: Vec<Box<str>> = corpus.docs().iter().map(|d| d.feature.clone()).collect();
    let cols: Vec<Box<str>> = (0..h).map(|i| format!("t{i}").into_boxed_str()).collect();
    let mut flat = Vec::with_capacity(n * h);
    for p in &pooled {
        flat.extend(p.iter().zip(&center).map(|(x, m)| x - m));
    }
    let mat = nalgebra::DMatrix::<f32>::from_row_iterator(n, h, flat);
    mat.to_parquet_with_names(
        &format!("{}.text_embedding.parquet", c.out),
        (Some(&names), Some("feature")),
        Some(&cols),
    )?;
    write_named_types(&corpus, &format!("{}.feature_types.parquet", c.out))?;

    // feature_word.edges.tsv
    let path = format!("{}.feature_word.edges.tsv", c.out);
    let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let mut n_edges = 0usize;
    for (d, words) in corpus.docs().iter().zip(&feature_words) {
        n_edges += write_typed_edges(
            &mut w,
            words.iter().map(|(wi, wt)| {
                (
                    d.ty.as_ref(),
                    d.feature.as_ref(),
                    "word",
                    vocab.kept[*wi].as_ref(),
                    *wt,
                )
            }),
        )?;
    }
    info!("wrote {n_edges} feature–word edges to {path}");

    if c.expand_k > 0 || c.knn > 0 {
        let feat = centred_unit(&pooled, &center, &device)?;
        if c.expand_k > 0 {
            let seen: Vec<u32> = word_n.clone();
            let word_vecs: Vec<Vec<f32>> = word_sum
                .iter()
                .zip(&word_n)
                .map(|(s, &k)| s.iter().map(|x| x / k.max(1) as f32).collect())
                .collect();
            let words = centred_unit(&word_vecs, &center, &device)?;
            let hits = csls_top_k(&feat, &words, c.expand_k + c.words_per_feature, 10)?;
            let path = format!("{}.feature_word_expanded.edges.tsv", c.out);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let mut n_exp = 0usize;
            for ((d, own), hit) in corpus.docs().iter().zip(&feature_words).zip(&hits) {
                let taken: Vec<(usize, f32)> = hit
                    .iter()
                    .filter(|(j, _)| seen[*j] > 0 && !own.iter().any(|(o, _)| o == j))
                    .take(c.expand_k)
                    .map(|(j, s)| (*j, s.max(0.0)))
                    .collect();
                n_exp += write_typed_edges(
                    &mut w,
                    taken.iter().map(|(j, s)| {
                        (
                            d.ty.as_ref(),
                            d.feature.as_ref(),
                            "word",
                            vocab.kept[*j].as_ref(),
                            *s,
                        )
                    }),
                )?;
            }
            info!("wrote {n_exp} expanded feature–word edges (CSLS) to {path}");
        }
        if c.knn > 0 {
            let hits = top_k_cosine(&feat, &feat, c.knn, true)?;
            let path = format!("{}.knn_graph.edges.tsv", c.out);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let docs = corpus.docs();
            let mut n_knn = 0usize;
            for (i, hit) in hits.iter().enumerate() {
                // Endpoints ordered by (type, feature) so a gene–term pair
                // always lands in the one relation `gene:term`, whichever
                // side found the other, and `senna fne` folds the two
                // directions of a same-type pair into one edge.
                n_knn += write_typed_edges(
                    &mut w,
                    hit.iter().map(|(j, s)| {
                        let (a, b) = if (&docs[i].ty, &docs[i].feature)
                            <= (&docs[*j].ty, &docs[*j].feature)
                        {
                            (&docs[i], &docs[*j])
                        } else {
                            (&docs[*j], &docs[i])
                        };
                        (
                            a.ty.as_ref(),
                            a.feature.as_ref(),
                            b.ty.as_ref(),
                            b.feature.as_ref(),
                            s.max(0.0),
                        )
                    }),
                )?;
            }
            info!("wrote {n_knn} text-similarity edges to {path}");
        }
    }
    info!(
        "done: {n} features × {h} dims in {}.text_embedding.parquet; {} vocabulary words",
        c.out, v
    );
    Ok(())
}

fn write_named_types(corpus: &Corpus, path: &str) -> Result<()> {
    let names: Vec<Box<str>> = corpus.docs().iter().map(|d| d.feature.clone()).collect();
    let types: Vec<Box<str>> = corpus.docs().iter().map(|d| d.ty.clone()).collect();
    matrix_util::parquet::write_named_table(
        path,
        "feature",
        &names,
        &[(Box::from("type"), matrix_util::parquet::Column::Str(&types))],
    )
}

fn indicatif_bar(n: u64) -> indicatif::ProgressBar {
    let bar = indicatif::ProgressBar::new(n);
    bar.set_style(
        indicatif::ProgressStyle::with_template(
            "{msg} {bar:30} {pos}/{len} [{elapsed_precise}<{eta_precise}]",
        )
        .expect("template"),
    );
    bar.set_message("encoding");
    bar
}
