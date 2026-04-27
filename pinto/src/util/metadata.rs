//! JSON metadata output for pinto runs.
//!
//! Writes a `{prefix}.metadata.json` file containing:
//! - Run parameters
//! - Output file paths (relative to prefix)
//! - Data statistics (n_cells, n_edges, n_communities, etc.)
//! - Hierarchical level information

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PintoMetadata {
    pub command: String,
    pub version: String,
    pub timestamp: String,
    pub prefix: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_files: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_file: Option<String>,

    pub n_cells: usize,
    pub n_genes: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_edges: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_communities: Option<usize>,

    pub outputs: OutputFiles,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<Vec<LevelInfo>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OutputFiles {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_pairs: Option<String>,

    /// Bare coordinate column basenames (without the `left_`/`right_`
    /// prefix) that the writer emitted alongside `coord_pairs`. The
    /// reader uses these in fixed order — `[x, y]` — instead of the
    /// fragile auto-discovery over `left_*` schema fields.
    /// `None` for older runs that pre-date this field; readers fall
    /// back to auto-discovery in that case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_columns: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub propensity: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_community: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub gene_community: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub scores: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_effects: Option<String>,

    /// Cosine dictionary-merge artifacts: the merge tree and its consensus
    /// cut. Absent when no collapses pass `--merge-cut` (in that case the
    /// draft is the final partition). The merged consensus partition itself
    /// is published under the bare prefix
    /// (`{prefix}.{propensity,link_community,gene_community}.parquet`), so this
    /// struct only points at the auxiliary tree + cut files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dict_merge: Option<DictMergeFiles>,

    /// JSON sidecar from `pinto lr-activity`. Optional; written only when
    /// the lr-activity subcommand is run against this prefix and emits
    /// per-significant-pair edge participation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_activity: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DictMergeFiles {
    /// Full agglomerative merge tree (one row per merge step).
    pub merges: String,
    /// Per-fine-community consensus label produced by the cut.
    pub cut: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LevelInfo {
    pub tag: String,
    pub level_index: usize,
    pub propensity: String,
    pub link_community: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gene_community: Option<String>,
    /// `Some(true)` if the propensity parquet at this level carries an
    /// `entropy` column (post-Phase-1 runs). `None` for older runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entropy_present: Option<bool>,
}

impl PintoMetadata {
    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let meta: PintoMetadata = serde_json::from_str(&json)?;
        Ok(meta)
    }
}

fn now_secs() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

/// Build a `LevelInfo` for a per-cascade-level set of outputs at
/// `{prefix}.L{level_index}.*`.
pub fn lc_level_info(prefix: &str, level_index: usize) -> LevelInfo {
    let tag = format!("L{level_index}");
    LevelInfo {
        tag: tag.clone(),
        level_index,
        propensity: format!("{prefix}.{tag}.propensity.parquet"),
        link_community: format!("{prefix}.{tag}.link_community.parquet"),
        gene_community: Some(format!("{prefix}.{tag}.gene_community.parquet")),
        entropy_present: Some(true),
    }
}

/// Inputs shared by every metadata builder (`lc` / `dsvd`).
///
/// Bundled so call sites stay readable and the builders dodge
/// `clippy::too_many_arguments`.
pub struct RunInputs<'a> {
    pub prefix: &'a str,
    pub data_files: &'a [Box<str>],
    pub coord_file: Option<&'a str>,
    pub coord_columns: &'a [Box<str>],
    pub n_cells: usize,
    pub n_genes: usize,
    pub n_edges: usize,
    /// Number of communities (lc) / clusters (dsvd) — same K dim either way.
    pub k: usize,
}

/// Helper to create metadata for `pinto lc` runs.
///
/// `cascade_level_indices` is the list of `l` values for which
/// `{prefix}.L{l}.*` files were actually written by the cascade
/// (skipped levels are absent — the cascade drops levels with too few
/// super-edges, so indices need not be contiguous and may not start at 0).
/// `merge_present` is `true` when the dictionary-merge step produced a
/// consensus collapse and its tree + cut files were written.
pub fn create_lc_metadata(
    inputs: &RunInputs<'_>,
    merge_present: bool,
    cascade_level_indices: &[usize],
) -> PintoMetadata {
    let prefix = inputs.prefix;
    let dict_merge = merge_present.then(|| DictMergeFiles {
        merges: format!("{prefix}.dict_merges.parquet"),
        cut: format!("{prefix}.dict_merges.cut.parquet"),
    });

    let mut levels: Vec<LevelInfo> = cascade_level_indices
        .iter()
        .map(|&l| lc_level_info(prefix, l))
        .collect();
    let tail_index = cascade_level_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    levels.push(LevelInfo {
        tag: "final".to_string(),
        level_index: tail_index,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: format!("{prefix}.link_community.parquet"),
        gene_community: Some(format!("{prefix}.gene_community.parquet")),
        entropy_present: Some(true),
    });

    PintoMetadata {
        command: "lc".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(inputs.data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: inputs.coord_file.map(|s| s.to_string()),
        n_cells: inputs.n_cells,
        n_genes: inputs.n_genes,
        n_edges: Some(inputs.n_edges),
        n_communities: Some(inputs.k),
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            gene_community: Some(format!("{prefix}.gene_community.parquet")),
            scores: Some(format!("{prefix}.scores.parquet")),
            batch_effects: None,
            dict_merge,
            lr_activity: None,
        },
        levels: Some(levels),
    }
}

fn coord_columns_field(cols: &[Box<str>]) -> Option<Vec<String>> {
    if cols.is_empty() {
        None
    } else {
        Some(cols.iter().map(|s| s.to_string()).collect())
    }
}

/// Helper for `pinto dsvd` runs. Only one "final" level is produced;
/// the cascade does not run.
pub fn create_dsvd_metadata(inputs: &RunInputs<'_>) -> PintoMetadata {
    let prefix = inputs.prefix;
    let levels = vec![LevelInfo {
        tag: "final".to_string(),
        level_index: 0,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: format!("{prefix}.link_community.parquet"),
        gene_community: Some(format!("{prefix}.gene_community.parquet")),
        entropy_present: Some(true),
    }];

    PintoMetadata {
        command: "dsvd".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(inputs.data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: inputs.coord_file.map(|s| s.to_string()),
        n_cells: inputs.n_cells,
        n_genes: inputs.n_genes,
        n_edges: Some(inputs.n_edges),
        n_communities: Some(inputs.k),
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: None,
            gene_community: Some(format!("{prefix}.gene_community.parquet")),
            scores: None,
            batch_effects: Some(format!("{prefix}.delta.parquet")),
            dict_merge: None,
            lr_activity: None,
        },
        levels: Some(levels),
    }
}

/// Helper for the standalone `pinto prop` command. Inputs are precomputed
/// latent + coord-pair files, not raw expression, so `data_files` is
/// optional.
pub fn create_prop_metadata(
    prefix: &str,
    expr_files: Option<&[Box<str>]>,
    coord_pair_file: Option<&str>,
    n_vertices: usize,
    n_clusters: usize,
) -> PintoMetadata {
    let levels = vec![LevelInfo {
        tag: "final".to_string(),
        level_index: 0,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: format!("{prefix}.edge_cluster.parquet"),
        gene_community: None,
        entropy_present: Some(true),
    }];

    PintoMetadata {
        command: "prop".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: expr_files.map(|fs| fs.iter().map(|s| s.to_string()).collect()),
        coord_file: coord_pair_file.map(|s| s.to_string()),
        n_cells: n_vertices,
        n_genes: 0,
        n_edges: None,
        n_communities: Some(n_clusters),
        outputs: OutputFiles {
            coord_pairs: coord_pair_file.map(|s| s.to_string()),
            coord_columns: None,
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: None,
            gene_community: None,
            scores: None,
            batch_effects: None,
            dict_merge: None,
            lr_activity: None,
        },
        levels: Some(levels),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_roundtrip_lc() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("run").to_string_lossy().to_string();
        let data_files: Vec<Box<str>> = vec!["a.h5".into(), "b.h5".into()];
        let coord_cols: Vec<Box<str>> =
            vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        let meta = create_lc_metadata(
            &RunInputs {
                prefix: &prefix,
                data_files: &data_files,
                coord_file: Some("a.tsv,b.tsv"),
                coord_columns: &coord_cols,
                n_cells: 1234,
                n_genes: 18000,
                n_edges: 55555,
                k: 12,
            },
            true,
            &[0, 1, 2],
        );
        let path = dir.path().join("run.metadata.json");
        meta.write(&path).unwrap();
        let back = PintoMetadata::read(&path).unwrap();
        assert_eq!(back.command, "lc");
        assert_eq!(back.n_cells, 1234);
        assert_eq!(back.n_communities, Some(12));
        let levels = back.levels.expect("levels");
        // 3 cascade levels + final = 4 (final carries the merged consensus)
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0].tag, "L0");
        assert_eq!(levels[3].tag, "final");
        assert_eq!(levels[3].entropy_present, Some(true));
        assert!(back.outputs.dict_merge.is_some());
        assert!(back.outputs.lr_activity.is_none());
        assert_eq!(
            back.outputs.coord_columns.as_deref(),
            Some(
                &[
                    "pxl_row_in_fullres".to_string(),
                    "pxl_col_in_fullres".to_string()
                ][..]
            )
        );
    }

    #[test]
    fn metadata_roundtrip_lc_merge_no_collapse() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("run").to_string_lossy().to_string();
        let data_files: Vec<Box<str>> = vec!["a.h5".into()];
        let meta = create_lc_metadata(
            &RunInputs {
                prefix: &prefix,
                data_files: &data_files,
                coord_file: None,
                coord_columns: &[],
                n_cells: 100,
                n_genes: 200,
                n_edges: 300,
                k: 8,
            },
            false,
            &[],
        );
        let path = dir.path().join("run.metadata.json");
        meta.write(&path).unwrap();
        let back = PintoMetadata::read(&path).unwrap();
        let levels = back.levels.expect("levels");
        // 0 cascade levels + final = 1
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].tag, "final");
        assert!(back.outputs.dict_merge.is_none());
    }
}
