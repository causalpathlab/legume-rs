//! Entry point for `senna gem` (alias `gem-embedding`).
//!
//! Genes-only joint embedding, plus optional co-measured modality tracks
//! (m6a, atoi, apa), over the shared `graph_embedding_util` engine via
//! `senna bge`'s driver ([`crate::bge::driver::fit_embed_family`]): the
//! bilinear score `e_feat·e_cell + b_feat + b_cell`, phase-1
//! multilevel-pseudobulk training + phase-2 analytical per-cell projection,
//! and the same output set `senna bge` writes.
//!
//! [`crate::gem::load::resolve_inputs`] classifies and sample-id-matches
//! every input file, [`crate::gem::load::load_gem_data`] loads them and
//! assigns the [`crate::gem::tracks::TrackPlan`], and
//! [`crate::gem::hvg::gem_hvg_row_weights`] pools that plan's rows per gene
//! for HVG projection weighting. The track plan itself is not yet wired
//! into the driver (`tracks: None` below) — a later task passes it through
//! and adds `{out}.feature_contrast.parquet`.

use crate::bge::driver::{fit_embed_family, EmbedPlan};
use crate::gem::args::GemArgs;
use crate::gem::hvg::gem_hvg_row_weights;
use crate::gem::load::{load_gem_data, resolve_inputs};
use matrix_util::common_io::mkdir_parent;

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    validate_args(args)?;

    let batch_files = crate::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

    let inputs = resolve_inputs(&args.genes, &args.modality_files, &args.genes_sample_strip)?;
    let (unified, plan) = load_gem_data(&inputs, batch_files, args.preload_data)?;
    let hvg_weights = gem_hvg_row_weights(&unified, &plan, &args.hvg, args.block_size)?;

    let data_files = inputs.files.clone();
    fit_embed_family(EmbedPlan {
        kind: crate::run_manifest::RunKind::Gem,
        knobs: args.knobs(),
        unified,
        data_files,
        multiome: None,
        hvg_weights,
        tracks: None,
        offset_l2: args.offset_l2,
        pb_reference: None,
        init_from: None,
        train_args: crate::run_manifest::record_train_args(args)?,
        after_fit: None,
    })
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.embedding_dim
    );
    Ok(())
}

#[cfg(test)]
#[path = "run/tests.rs"]
mod tests;
