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

#[derive(Serialize, Deserialize, Debug, Clone)]
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
}

impl PintoMetadata {
    #[allow(dead_code)]
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

/// Helper to create metadata for link community runs
#[allow(dead_code)]
pub fn create_lc_metadata(
    prefix: &str,
    data_files: &[Box<str>],
    coord_file: Option<&str>,
    n_cells: usize,
    n_genes: usize,
    n_edges: usize,
    k: usize,
    has_bhc: bool,
    level_tags: Vec<String>,
) -> PintoMetadata {
    let coord_pairs_path = format!("{}.coord_pairs.parquet", prefix);
    let propensity_path = format!("{}.propensity.parquet", prefix);
    let link_community_path = format!("{}.link_community.parquet", prefix);
    let gene_topic_path = format!("{}.gene_topic.parquet", prefix);
    let scores_path = format!("{}.scores.parquet", prefix);

    let bhc = if has_bhc {
        Some(BhcFiles {
            merges: format!("{}.bhc.merges.parquet", prefix),
            cut: format!("{}.bhc.cut.parquet", prefix),
            propensity: format!("{}.bhc.propensity.parquet", prefix),
            link_community: format!("{}.bhc.link_community.parquet", prefix),
        })
    } else {
        None
    };

    let levels = if !level_tags.is_empty() {
        Some(
            level_tags
                .into_iter()
                .enumerate()
                .map(|(i, tag)| LevelInfo {
                    tag: tag.clone(),
                    level_index: i,
                    propensity: format!("{}.{}.propensity.parquet", prefix, tag),
                    link_community: format!("{}.{}.link_community.parquet", prefix, tag),
                    gene_topic: Some(format!("{}.{}.gene_topic.parquet", prefix, tag)),
                })
                .collect(),
        )
    } else {
        None
    };

    PintoMetadata {
        command: "lc".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string(),
        prefix: prefix.to_string(),
        data_files: Some(data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: coord_file.map(|s| s.to_string()),
        n_cells,
        n_genes,
        n_edges: Some(n_edges),
        n_communities: Some(k),
        outputs: OutputFiles {
            coord_pairs: Some(coord_pairs_path),
            propensity: Some(propensity_path),
            link_community: Some(link_community_path),
            gene_topic: Some(gene_topic_path),
            scores: Some(scores_path),
            batch_effects: None,
            bhc,
        },
        levels,
    }
}
