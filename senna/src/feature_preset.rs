//! `--{freeze,init,lora}-feature-embedding <prefix>` for every model with a
//! gene table (`senna bge`, `senna simba`, `senna fne`): the gene rows of an
//! earlier run's feature table, matched onto the caller's gene axis, to pin,
//! to start from, or to anchor a low-rank residual to. The result is a
//! [`ge::PresetRows`] with ids into the given axis; the PBG commands lift it
//! to their node ids with `map_ids`.
//!
//! The source is any run whose prefix resolves through
//! [`crate::run_manifest::resolve_feature_loading`] — typically `senna fne`,
//! whose table also holds terms, words and cell types. Those rows are skipped
//! by the run's `feature_types.parquet` when it exists; a source without one
//! is taken to be all genes. Genes of this axis with no source row stay free.

use auxiliary_data::feature_types::{read_feature_types, GENE_TYPE};
use auxiliary_data::frozen_features::{load_frozen_feature_host, FrozenLoadArgs};
use graph_embedding_util as ge;
use graph_embedding_util::PresetMode;
use log::info;
use rustc_hash::FxHashSet;

pub(crate) fn load_preset_genes(
    prefix: &str,
    mode: PresetMode,
    feature_names: &[Box<str>],
    kind: &ge::FeatureNameKind,
) -> anyhow::Result<ge::PresetRows> {
    let flag = crate::feature_embedding_args::flag_name(mode);
    let (dictionary_path, _bias) = crate::run_manifest::resolve_feature_loading(prefix)
        .map_err(|e| anyhow::anyhow!("{flag} {prefix}: {e}"))?;
    let host = load_frozen_feature_host(FrozenLoadArgs {
        dictionary_path: &dictionary_path,
        bias_path: None,
        target_feature_names: feature_names,
        name_kind: kind.clone(),
    })?;

    // Which source rows are genes: the types table, when the run wrote one.
    let gene_src: Option<FxHashSet<usize>> = read_feature_types(prefix)?.map(|rows| {
        let gene_names: FxHashSet<Box<str>> = rows
            .into_iter()
            .filter(|(_, t)| t.as_ref() == GENE_TYPE)
            .map(|(n, _)| n)
            .collect();
        host.src_names
            .iter()
            .enumerate()
            .filter(|(_, n)| gene_names.contains(*n))
            .map(|(i, _)| i)
            .collect()
    });

    let h = host.e_feat.ncols();
    let mut ids: Vec<u32> = Vec::new();
    let mut rows: Vec<f32> = Vec::new();
    for (j, (&target, &src)) in host
        .keep_target_indices
        .iter()
        .zip(&host.keep_src_indices)
        .enumerate()
    {
        if gene_src.as_ref().is_some_and(|s| !s.contains(&src)) {
            continue;
        }
        ids.push(target as u32);
        rows.extend(host.e_feat.row(j).iter().copied());
    }
    anyhow::ensure!(
        !ids.is_empty(),
        "{flag} {prefix}: no gene of this feature axis has a row in {dictionary_path}"
    );
    mode.validate(h)?;
    info!(
        "Feature side from {dictionary_path} (H={h}): {} of {} features {}",
        ids.len(),
        feature_names.len(),
        mode.describe()
    );
    Ok(ge::PresetRows { ids, rows, mode })
}

/// `--embedding-dim` against the preset's width, the loader having refused an
/// empty match (so the division is exact).
pub(crate) fn resolve_dim(
    cli_embedding_dim: ge::EmbeddingDim,
    preset: Option<&ge::PresetRows>,
) -> anyhow::Result<usize> {
    cli_embedding_dim
        .resolve(preset.map(ge::PresetRows::width))?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--embedding-dim auto takes H from a given feature embedding; none was given"
            )
        })
}

#[cfg(test)]
mod tests;
