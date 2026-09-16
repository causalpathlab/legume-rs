//! `senna bge --{freeze,init}-feature-embedding <prefix>`: the gene rows of an
//! earlier run's feature table, matched onto this fit's feature axis, for phase
//! 1 to pin or to start from (see [`ge::fit::hier::PresetGenes`]).
//!
//! The source is any run whose prefix resolves through
//! [`crate::run_manifest::resolve_feature_loading`] — typically `senna fne`,
//! whose table also holds terms, words and cell types. Those rows are skipped
//! by the run's `feature_types.parquet` when it exists; a source without one
//! is taken to be all genes. Genes of this axis with no source row stay free.

use auxiliary_data::frozen_features::{load_frozen_feature_host, FrozenLoadArgs};
use graph_embedding_util as ge;
use log::info;
use matrix_util::parquet::read_parquet_string_columns_by_name;
use rustc_hash::FxHashSet;
use std::path::Path;

pub(crate) fn load_preset_genes(
    prefix: &str,
    freeze: bool,
    feature_names: &[Box<str>],
    kind: &ge::FeatureNameKind,
) -> anyhow::Result<ge::fit::hier::PresetGenes> {
    let flag = if freeze {
        "--freeze-feature-embedding"
    } else {
        "--init-feature-embedding"
    };
    let (dictionary_path, _bias) = crate::run_manifest::resolve_feature_loading(prefix)
        .map_err(|e| anyhow::anyhow!("{flag} {prefix}: {e}"))?;
    let host = load_frozen_feature_host(FrozenLoadArgs {
        dictionary_path: &dictionary_path,
        bias_path: None,
        target_feature_names: feature_names,
        name_kind: kind.clone(),
    })?;

    // Which source rows are genes: the types table, when the run wrote one.
    let types_path = format!("{prefix}.feature_types.parquet");
    let gene_src: Option<FxHashSet<usize>> = if Path::new(&types_path).exists() {
        let cols = read_parquet_string_columns_by_name(&types_path, &["feature", "type"])?;
        let by_name: FxHashSet<&str> = cols[0]
            .iter()
            .zip(&cols[1])
            .filter(|(_, t)| t.as_ref() == crate::fne::graph::GENE_TYPE)
            .map(|(n, _)| n.as_ref())
            .collect();
        Some(
            host.src_names
                .iter()
                .enumerate()
                .filter(|(_, n)| by_name.contains(n.as_ref()))
                .map(|(i, _)| i)
                .collect(),
        )
    } else {
        None
    };

    let h = host.e_feat.ncols();
    let mut gene: Vec<u32> = Vec::new();
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
        gene.push(target as u32);
        rows.extend(host.e_feat.row(j).iter().copied());
    }
    anyhow::ensure!(
        !gene.is_empty(),
        "{flag} {prefix}: no gene of this feature axis has a row in {dictionary_path}"
    );
    info!(
        "Feature side from {dictionary_path} (H={h}): {} of {} features {}; the rest train",
        gene.len(),
        feature_names.len(),
        if freeze { "pinned" } else { "warm-started" }
    );
    Ok(ge::fit::hier::PresetGenes { gene, rows, freeze })
}

#[cfg(test)]
mod tests;
