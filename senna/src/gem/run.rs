//! Entry point for `senna gem` (alias `gem-embedding`).
//!
//! Genes-only joint embedding, plus optional co-measured modality tracks
//! (m6a, atoi, apa), over the shared `graph_embedding_util` engine via
//! `senna bge`'s driver ([`crate::bge::driver::fit_embed_family`]): the
//! bilinear score `e_feat·e_cell + b_feat + b_cell`, phase-1
//! multilevel-pseudobulk training + phase-2 analytical per-cell projection,
//! and the same output set `senna bge` writes, PLUS one contrast row per
//! gene-and-modality (`{out}.feature_contrast.parquet`) wherever a modality
//! carries both of its channels.
//!
//! [`crate::gem::load::resolve_inputs`] classifies and sample-id-matches
//! every input file, [`crate::gem::load::load_gem_data`] loads them and
//! assigns the [`crate::gem::tracks::TrackPlan`] (the row grammar's read of
//! which track and gene every row belongs to), and
//! [`crate::gem::hvg::gem_hvg_row_weights`] pools that plan's rows per gene
//! for HVG projection weighting. The plan is handed to the driver as
//! `FitConfig.tracks` (so phase 1 and phase 2 both train per-track) and,
//! after the fit, to [`crate::gem::contrast::write_contrast`] (so the
//! contrast table reads the same RAW loading the fit produced). A table
//! given with `--{freeze,init,lora}-feature-embedding` is resolved onto the
//! plan by [`crate::gem::preset::resolve_gem_preset`]; `--embedding-dim` is
//! settled against it, and `--offset-rank` against the settled H.

use crate::bge::driver::{fit_embed_family, EmbedPlan};
use crate::gem::args::GemArgs;
use crate::gem::contrast::write_contrast;
use crate::gem::hvg::gem_hvg_row_weights;
use crate::gem::load::{load_gem_data, resolve_inputs};
use legume_numeric::matrix::common_io::mkdir_parent;

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    validate_args(args)?;

    let batch_files = senna::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

    let inputs = resolve_inputs(&args.genes, &args.modality_files, &args.genes_sample_strip)?;
    let (unified, plan) = load_gem_data(&inputs, batch_files, args.preload_data)?;
    let hvg_weights = gem_hvg_row_weights(&unified, &plan, &args.hvg, args.block_size)?;

    let preset = crate::gem::preset::resolve_gem_preset(
        args.feature_embedding.resolve()?,
        &unified.feature_names,
        &plan,
    )?;
    let embedding_dim =
        crate::feature_preset::resolve_dim(args.embedding_dim, preset.base.as_ref())?;
    anyhow::ensure!(embedding_dim > 0, "--embedding-dim must be > 0");
    validate_offset_rank(args.offset_rank, embedding_dim)?;

    let data_files = inputs.files.clone();
    fit_embed_family(EmbedPlan {
        kind: senna::run_manifest::RunKind::Gem,
        knobs: args.knobs(embedding_dim),
        unified,
        data_files,
        multiome: None,
        hvg_weights,
        tracks: Some(plan.clone()),
        offset_l2: args.offset_l2,
        offset_rank: args.offset_rank,
        preset_features: preset.base,
        preset_offsets: preset.offsets,
        carried: preset.carried,
        pb_reference: None,
        init_from: None,
        train_args: senna::run_manifest::record_train_args(args)?,
        after_fit: Some(&|a| write_contrast(a, &plan)),
    })
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    args.collapse
        .reject_pb_reference(senna::run_manifest::RunKind::Gem)?;
    Ok(())
}

/// `--offset-rank` against the settled H, by the engine's own rule, with the
/// two flags named.
pub(crate) fn validate_offset_rank(rank: usize, h: usize) -> anyhow::Result<()> {
    graph_embedding_util::validate_offset_rank(rank, h)
        .map_err(|e| anyhow::anyhow!("--offset-rank {rank} against --embedding-dim {h}: {e}"))
}

#[cfg(test)]
#[path = "run/tests.rs"]
mod tests;
