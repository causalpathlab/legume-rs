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

use auxiliary_data::feature_types::{read_feature_types, write_feature_types, GENE_TYPE};
use auxiliary_data::frozen_features::{
    load_frozen_feature_host, FrozenFeatureHost, FrozenLoadArgs,
};
use graph_embedding_util as ge;
use graph_embedding_util::PresetMode;
use log::info;
use matrix_util::parquet::peek_parquet_field_names;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;
use rustc_hash::{FxHashMap, FxHashSet};

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
    let carried = if mode.pins() {
        CarriedRows::from_host(&host, feature_names, kind, prefix, &dictionary_path)?
    } else {
        info!(
            "{flag}: the rows of {dictionary_path} that matched nothing are not carried \
             through, since the trained rows leave the table's space"
        );
        None
    };
    Ok((ge::PresetRows { ids, rows, mode }, carried))
}

/// The rows of a given table that pin nothing on this run's feature axis,
/// kept to write back beside the trained table. Only meaningful when the
/// matched rows stay in the table's space (freeze, lora): then the trained
/// rows and these sit in one space, and the output is the full table.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CarriedRows {
    pub names: Vec<Box<str>>,
    /// One per row, from the source run's types table (`gene` without one).
    pub types: Vec<Box<str>>,
    /// `[n, H]`, in source order.
    pub rows: DMatrix<f32>,
    /// The file the rows came from, for the log.
    pub source: String,
}

impl CarriedRows {
    /// Every source row the match left unused whose name (raw, or canonical
    /// under `kind`) is not a feature of `target_names` either — a duplicate
    /// canonical row of a matched gene, or a word that also names a feature,
    /// would otherwise come out twice. `None` when nothing is left over.
    pub(crate) fn from_host(
        host: &FrozenFeatureHost,
        target_names: &[Box<str>],
        kind: &ge::FeatureNameKind,
        source_prefix: &str,
        dictionary_path: &str,
    ) -> anyhow::Result<Option<Self>> {
        let used: FxHashSet<usize> = host.keep_src_indices.iter().copied().collect();
        let taken: FxHashSet<Box<str>> = target_names
            .iter()
            .flat_map(|n| [n.clone(), kind.canonicalize(n)])
            .collect();
        let src_types: FxHashMap<Box<str>, Box<str>> = read_feature_types(source_prefix)?
            .map(|rows| rows.into_iter().collect())
            .unwrap_or_default();
        let idx: Vec<usize> = host
            .src_names
            .iter()
            .enumerate()
            .filter(|(i, n)| {
                !used.contains(i) && !taken.contains(*n) && !taken.contains(&kind.canonicalize(n))
            })
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            return Ok(None);
        }
        let names: Vec<Box<str>> = idx.iter().map(|&i| host.src_names[i].clone()).collect();
        let types: Vec<Box<str>> = names
            .iter()
            .map(|n| {
                src_types
                    .get(n)
                    .cloned()
                    .unwrap_or_else(|| GENE_TYPE.into())
            })
            .collect();
        let rows = host.src_e_feat.select_rows(idx.iter());
        Ok(Some(Self {
            names,
            types,
            rows,
            source: dictionary_path.to_string(),
        }))
    }

    /// Append these rows to the ρ table the run wrote at `rho_path`, keeping
    /// its row axis and column names, and write `{out_prefix}.feature_types.parquet`
    /// over every row: the run's own rows keep the types it wrote, or are all
    /// `gene` when it wrote none. A carried row whose name the run wrote
    /// itself (a term or cell type both graphs hold, say) is superseded by
    /// the run's own, trained row.
    pub(crate) fn append_to(&self, out_prefix: &str, rho_path: &str) -> anyhow::Result<()> {
        let fields = peek_parquet_field_names(rho_path)?;
        let row_axis = fields
            .first()
            .ok_or_else(|| anyhow::anyhow!("{rho_path}: no columns"))?;
        let table = DMatrix::<f32>::from_parquet(rho_path)?;
        let (n, h) = (table.mat.nrows(), table.mat.ncols());
        anyhow::ensure!(
            h == self.rows.ncols(),
            "{rho_path} is {h} wide but the carried rows of {} are {}",
            self.source,
            self.rows.ncols()
        );
        let own: FxHashSet<&str> = table.rows.iter().map(AsRef::as_ref).collect();
        let keep: Vec<usize> = (0..self.names.len())
            .filter(|&i| !own.contains(self.names[i].as_ref()))
            .collect();
        let superseded = self.names.len() - keep.len();
        if keep.is_empty() {
            info!(
                "Nothing of {} to carry through: the run wrote all {superseded} of its unmatched rows itself",
                self.source
            );
            return Ok(());
        }
        let own_types: Vec<Box<str>> = match read_feature_types(out_prefix)? {
            Some(rows) => {
                anyhow::ensure!(
                    rows.len() == n && rows.iter().zip(&table.rows).all(|((a, _), b)| a == b),
                    "{}: rows disagree with {rho_path}",
                    auxiliary_data::feature_types::feature_types_path(out_prefix)
                );
                rows.into_iter().map(|(_, t)| t).collect()
            }
            None => vec![GENE_TYPE.into(); n],
        };

        let m = keep.len();
        let mut mat = DMatrix::<f32>::zeros(n + m, h);
        mat.rows_mut(0, n).copy_from(&table.mat);
        mat.rows_mut(n, m)
            .copy_from(&self.rows.select_rows(keep.iter()));
        let mut names = table.rows;
        names.extend(keep.iter().map(|&i| self.names[i].clone()));
        mat.to_parquet_with_names(rho_path, (Some(&names), Some(row_axis)), Some(&table.cols))?;
        let mut types = own_types;
        types.extend(keep.iter().map(|&i| self.types[i].clone()));
        write_feature_types(out_prefix, &names, &types)?;
        let n_gene = keep
            .iter()
            .filter(|&&i| self.types[i].as_ref() == GENE_TYPE)
            .count();
        info!(
            "Carried {m} rows of {} through unchanged into {rho_path} ({n_gene} gene, {} other; \
             {superseded} superseded by the run's own rows); every row's type is in {}",
            self.source,
            m - n_gene,
            auxiliary_data::feature_types::feature_types_path(out_prefix)
        );
        Ok(())
    }
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
        let mut mat = DMatrix::<f32>::zeros(n + EXTRA.len(), h);
        mat.rows_mut(0, n).copy_from(&t.mat);
        mat.rows_mut(n, EXTRA.len()).copy_from(&extra);
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
