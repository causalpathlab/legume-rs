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

    #[serde(skip_serializing_if = "Option::is_none")]
    pub propensity: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_community: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub gene_topic: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub scores: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_effects: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub bhc: Option<BhcFiles>,

    /// JSON sidecar from `pinto lr-activity`. Optional; written only when
    /// the lr-activity subcommand is run against this prefix and emits
    /// per-significant-pair edge participation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_activity: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BhcFiles {
    pub merges: String,
    pub cut: String,
    pub propensity: String,
    pub link_community: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LevelInfo {
    pub tag: String,
    pub level_index: usize,
    pub propensity: String,
    pub link_community: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gene_topic: Option<String>,
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
        gene_topic: Some(format!("{prefix}.{tag}.gene_topic.parquet")),
        entropy_present: Some(true),
    }
}

/// Helper to create metadata for `pinto lc` runs.
///
/// `cascade_level_indices` is the list of `l` values for which
/// `{prefix}.L{l}.*` files were actually written by the cascade
/// (skipped levels are absent — the cascade drops levels with too few
/// super-edges, so indices need not be contiguous and may not start at 0).
/// `bhc_present` is `true` when the BHC consensus path wrote
/// `{prefix}.bhc.*` files.
#[allow(clippy::too_many_arguments)]
pub fn create_lc_metadata(
    prefix: &str,
    data_files: &[Box<str>],
    coord_file: Option<&str>,
    n_cells: usize,
    n_genes: usize,
    n_edges: usize,
    k: usize,
    bhc_present: bool,
    cascade_level_indices: &[usize],
) -> PintoMetadata {
    let bhc = bhc_present.then(|| BhcFiles {
        merges: format!("{prefix}.bhc.merges.parquet"),
        cut: format!("{prefix}.bhc.cut.parquet"),
        propensity: format!("{prefix}.bhc.propensity.parquet"),
        link_community: format!("{prefix}.bhc.link_community.parquet"),
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
        gene_topic: Some(format!("{prefix}.gene_topic.parquet")),
        entropy_present: Some(true),
    });
    if bhc_present {
        levels.push(LevelInfo {
            tag: "bhc".to_string(),
            level_index: tail_index + 1,
            propensity: format!("{prefix}.bhc.propensity.parquet"),
            link_community: format!("{prefix}.bhc.link_community.parquet"),
            gene_topic: Some(format!("{prefix}.bhc.gene_topic.parquet")),
            entropy_present: Some(true),
        });
    }

    PintoMetadata {
        command: "lc".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: coord_file.map(|s| s.to_string()),
        n_cells,
        n_genes,
        n_edges: Some(n_edges),
        n_communities: Some(k),
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            gene_topic: Some(format!("{prefix}.gene_topic.parquet")),
            scores: Some(format!("{prefix}.scores.parquet")),
            batch_effects: None,
            bhc,
            lr_activity: None,
        },
        levels: Some(levels),
    }
}

/// Helper for `pinto dsvd` runs. Only one "final" level is produced;
/// the cascade does not run.
pub fn create_dsvd_metadata(
    prefix: &str,
    data_files: &[Box<str>],
    coord_file: Option<&str>,
    n_cells: usize,
    n_genes: usize,
    n_edges: usize,
    n_clusters: usize,
) -> PintoMetadata {
    let levels = vec![LevelInfo {
        tag: "final".to_string(),
        level_index: 0,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: format!("{prefix}.link_community.parquet"),
        gene_topic: Some(format!("{prefix}.gene_topic.parquet")),
        entropy_present: Some(true),
    }];

    PintoMetadata {
        command: "dsvd".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: coord_file.map(|s| s.to_string()),
        n_cells,
        n_genes,
        n_edges: Some(n_edges),
        n_communities: Some(n_clusters),
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: None,
            gene_topic: Some(format!("{prefix}.gene_topic.parquet")),
            scores: None,
            batch_effects: Some(format!("{prefix}.delta.parquet")),
            bhc: None,
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
        gene_topic: None,
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
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: None,
            gene_topic: None,
            scores: None,
            batch_effects: None,
            bhc: None,
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
        let meta = create_lc_metadata(
            &prefix,
            &data_files,
            Some("a.tsv,b.tsv"),
            1234,
            18000,
            55555,
            12,
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
        // 3 cascade levels + final + bhc = 5
        assert_eq!(levels.len(), 5);
        assert_eq!(levels[0].tag, "L0");
        assert_eq!(levels[3].tag, "final");
        assert_eq!(levels[4].tag, "bhc");
        assert_eq!(levels[3].entropy_present, Some(true));
        assert!(back.outputs.bhc.is_some());
        assert!(back.outputs.lr_activity.is_none());
    }

    #[test]
    fn metadata_roundtrip_lc_no_bhc() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("run").to_string_lossy().to_string();
        let data_files: Vec<Box<str>> = vec!["a.h5".into()];
        let meta = create_lc_metadata(&prefix, &data_files, None, 100, 200, 300, 8, false, &[]);
        let path = dir.path().join("run.metadata.json");
        meta.write(&path).unwrap();
        let back = PintoMetadata::read(&path).unwrap();
        let levels = back.levels.expect("levels");
        // 0 cascade levels + final = 1
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].tag, "final");
        assert!(back.outputs.bhc.is_none());
    }
}
