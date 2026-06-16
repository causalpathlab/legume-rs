//! `senna annotate-by-projection` — light marker-set cell-type annotation by
//! projecting cell types onto the **frozen feature embedding** of an
//! embedding run (bge / fne).
//!
//! A thin adapter over the shared, model-agnostic
//! [`graph_embedding_util::type_annotation::annotate_embeddings`]: it reads a
//! `run.senna.json` manifest, loads `outputs.feature_embedding` (gene × H)
//! and `outputs.latent` (cell × H — for bge/fne the latent *is* the cell
//! embedding), and hands them to the shared routine. Outputs
//! `{out}.{kind}_annot.{posterior,zscore,type_embedding}.parquet`.
//!
//! This is the *light, per-cell, clustering-free* complement to
//! [`super::by_enrichment::run`] (cluster-level marker enrichment with FDR). It is
//! restricted to embedding kinds: a topic run's `latent` is in topic space,
//! not the H-space its `feature_embedding` lives in.

use anyhow::{Context, Result};
use clap::Args;
use std::path::Path;

use graph_embedding_util::type_annotation::{annotate_embeddings, AnnotateProjConfig};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use crate::run_manifest::{derive_out_prefix, resolve, RunKind, RunManifest};

#[derive(Args, Debug)]
pub struct AnnotateProjectArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest from `senna bge` / `senna fne` / `senna resolve-embedding-space`"
    )]
    pub from: Box<str>,

    #[arg(
        short = 'm',
        long = "markers",
        required = true,
        help = "Marker TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix. Defaults to `--from` with `.senna.json`/`.json` stripped"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature for the per-cell posterior (lower → sharper)"
    )]
    pub temperature: f32,

    #[arg(
        long = "num-perm",
        default_value_t = 200,
        help = "Permutation draws per type for the null (0 = skip z-scores)"
    )]
    pub num_perm: usize,

    #[arg(long, default_value_t = 42, help = "RNG seed (permutation null)")]
    pub seed: u64,

    #[arg(
        long = "no-idf",
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,
}

pub fn run(args: &AnnotateProjectArgs) -> Result<()> {
    let (manifest, dir) = RunManifest::load(Path::new(args.from.as_ref()))?;
    anyhow::ensure!(
        matches!(
            manifest.kind,
            RunKind::Bge | RunKind::Fne | RunKind::ResolveEmbeddingSpace
        ),
        "annotate-by-projection needs an embedding run (bge / fne / resolve-embedding-space) whose \
         `latent` is an H-space cell embedding; got kind={}. For topic runs use \
         `senna annotate-by-enrichment` (cluster-level enrichment), or first run \
         `senna resolve-embedding-space` to lift the topic θ into an embedding.",
        manifest.kind
    );

    // Feature embedding ρ. `bge --resolve-etm` records it explicitly; a plain
    // `bge`/`fne` run (no ETM) writes it as `dictionary` (there `dictionary` IS
    // E_feat, not a topic simplex), so fall back to that. ETM runs always set
    // `feature_embedding`, so the fallback never mis-picks a β simplex.
    let feat_rel = manifest
        .outputs
        .feature_embedding
        .as_deref()
        .or(manifest.outputs.dictionary.as_deref())
        .context("manifest has neither outputs.feature_embedding nor outputs.dictionary")?;
    // Prefer the explicit H-space cell embedding when the run records one
    // (`bge --resolve-etm`, `resolve-embedding-space`): there `latent` is the
    // topic θ, NOT the embedding ρ lives in. Plain bge/fne set no
    // `cell_embedding`, and their `latent` IS the cell embedding — fall back.
    let cell_rel = manifest
        .outputs
        .cell_embedding
        .as_deref()
        .or(manifest.outputs.latent.as_deref())
        .context(
            "manifest has neither outputs.cell_embedding nor outputs.latent (cell embedding). \
             For topic-space annotation use `senna annotate-by-topic`.",
        )?;

    let feat_path = resolve(&dir, feat_rel).to_string_lossy().into_owned();
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell_path = resolve(&dir, cell_rel).to_string_lossy().into_owned();
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => derive_out_prefix(&args.from),
    };
    mkdir_parent(&out)?;

    let cfg = AnnotateProjConfig {
        temperature: args.temperature,
        n_perm: args.num_perm,
        seed: args.seed,
    };
    annotate_embeddings(
        &feat.mat,
        &feat.rows,
        &cell.mat,
        &cell.rows,
        &args.markers,
        &format!("{out}.{}_annot", manifest.kind),
        !args.no_idf,
        &cfg,
    )?;
    Ok(())
}
