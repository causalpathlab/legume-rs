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
//!
//! Under a pinning mode the rows the match left unused — genes the data lacks
//! and every non-gene row — come back out as [`CarriedRows`]: appended
//! unchanged to the run's own ρ table once it is written, with
//! `feature_types.parquet` naming every row's type, so a run on a narrow
//! feature axis (a panel) still hands on the full table it was given.

use auxiliary_data::feature_types::{read_feature_types, FeatureType, GENE_TYPE};
use auxiliary_data::frozen_features::{load_frozen_feature_host, FrozenLoadArgs};
use graph_embedding_util as ge;
use graph_embedding_util::PresetMode;
use log::info;
use rustc_hash::FxHashSet;

pub(crate) use crate::carried_rows::CarriedRows;

/// The preset of a command's `--{freeze,init,lora}-feature-embedding`, when
/// one was given: the rows to pin or start from, and the rows to carry.
pub(crate) fn resolve_preset(
    resolved: Option<(&str, PresetMode)>,
    feature_names: &[Box<str>],
    kind: &ge::FeatureNameKind,
) -> anyhow::Result<(Option<ge::PresetRows>, Option<CarriedRows>)> {
    let Some((prefix, mode)) = resolved else {
        return Ok((None, None));
    };
    let (rows, carried) = load_preset_genes(prefix, mode, feature_names, kind)?;
    Ok((Some(rows), carried))
}

pub(crate) fn load_preset_genes(
    prefix: &str,
    mode: PresetMode,
    feature_names: &[Box<str>],
    kind: &ge::FeatureNameKind,
) -> anyhow::Result<(ge::PresetRows, Option<CarriedRows>)> {
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
    let src_types: Option<Vec<FeatureType>> = read_feature_types(prefix)?;
    let gene_src: Option<FxHashSet<usize>> = src_types.as_ref().map(|rows| {
        let gene_names: FxHashSet<&str> = rows
            .iter()
            .filter(|(_, t)| t.as_ref() == GENE_TYPE)
            .map(|(n, _)| n.as_ref())
            .collect();
        host.src_names
            .iter()
            .enumerate()
            .filter(|(_, n)| gene_names.contains(n.as_ref()))
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
    let carried = CarriedRows::from_unmatched(
        mode.pins(),
        flag,
        &host,
        feature_names,
        kind,
        src_types.as_deref().unwrap_or(&[]),
        &dictionary_path,
    )?;
    Ok((ge::PresetRows { ids, rows, mode }, carried))
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

/// For the engines' end-to-end tests: a source run wider than the data.
#[cfg(test)]
pub(crate) mod test_support {
    use matrix_util::traits::IoOps;
    use nalgebra::DMatrix;

    /// The extra rows [`widen`] adds: a gene the data lacks and a term.
    pub(crate) const EXTRA: [(&str, &str); 2] = [("EXTRA1", "gene"), ("GO:9999999", "term")];

    /// Write `{out}.feature_loading.parquet` = the ρ table at `src_rho_path`
    /// plus [`EXTRA`], with a types table over every row; returns the extra
    /// rows' values for the caller to look for in a run's output.
    pub(crate) fn widen(src_rho_path: &str, out: &str) -> DMatrix<f32> {
        let t = DMatrix::<f32>::from_parquet(src_rho_path).unwrap();
        let (n, h) = (t.mat.nrows(), t.mat.ncols());
        let extra =
            DMatrix::<f32>::from_fn(EXTRA.len(), h, |i, k| (i + 1) as f32 * 0.25 + k as f32);
        let mat = matrix_util::dmatrix_util::concatenate_vertical(&[t.mat, extra.clone()]).unwrap();
        let mut names = t.rows.clone();
        let mut types: Vec<Box<str>> = vec!["gene".into(); n];
        for (name, ty) in EXTRA {
            names.push(name.into());
            types.push(ty.into());
        }
        mat.to_parquet_with_names(
            &format!("{out}.feature_loading.parquet"),
            (Some(&names), Some("gene")),
            Some(&t.cols),
        )
        .unwrap();
        auxiliary_data::feature_types::write_feature_types(out, &names, &types).unwrap();
        extra
    }

    /// Assert the run at `out` wrote [`EXTRA`] after its own rows in `rho_path`,
    /// row for row equal to `extra`, and typed them in its types table.
    pub(crate) fn assert_carried(out: &str, rho_path: &str, own_rows: usize, extra: &DMatrix<f32>) {
        let t = DMatrix::<f32>::from_parquet(rho_path).unwrap();
        assert_eq!(t.mat.nrows(), own_rows + EXTRA.len(), "{rho_path}");
        for (i, (name, _)) in EXTRA.iter().enumerate() {
            assert_eq!(t.rows[own_rows + i].as_ref(), *name);
            assert_eq!(
                t.mat.row(own_rows + i),
                extra.row(i),
                "{name} is carried unchanged"
            );
        }
        let types = auxiliary_data::feature_types::read_feature_types(out)
            .unwrap()
            .expect("a types table over every row");
        assert_eq!(types.len(), own_rows + EXTRA.len());
        for (i, (name, ty)) in EXTRA.iter().enumerate() {
            assert_eq!(types[own_rows + i], ((*name).into(), (*ty).into()));
        }
    }
}
