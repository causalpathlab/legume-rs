//! Read/write a senna-format run manifest (`{prefix}.senna.json`) so
//! chickpea outputs feed straight into `senna layout` / `senna plot`
//! / `senna annotate` via their `--from` flag.
//!
//! Schema mirrors `senna::run_manifest::RunManifest` (kept narrow:
//! data + output paths only). All paths are stored relative to the
//! manifest file's directory so a run dir can be moved.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub version: u32,
    // "svd" → senna's non-topic-family layout/plot path (signed continuous loadings).
    pub kind: String,
    pub prefix: String,
    #[serde(default)]
    pub data: RunData,
    #[serde(default)]
    pub outputs: RunOutputs,
    #[serde(default)]
    pub layout: serde_json::Value,
    #[serde(default)]
    pub cluster: RunCluster,
    #[serde(default)]
    pub annotate: RunAnnotate,
    #[serde(default)]
    pub pseudotime: serde_json::Value,
    #[serde(default)]
    pub defaults: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunData {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunOutputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunCluster {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clusters: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunAnnotate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_celltype_q: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_celltype_es: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub markers: Option<String>,
}

/// Path to write `{prefix}.senna.json`.
pub fn manifest_path(prefix: &str) -> String {
    format!("{prefix}.senna.json")
}

/// Make `path` relative to `manifest_dir` if it lives under it; otherwise
/// return the absolute path. Mirrors `senna::run_manifest::rel_to_manifest`.
pub fn rel_to_manifest(manifest_dir: &Path, path: &str) -> String {
    let abs = PathBuf::from(path);
    let abs = if abs.is_absolute() {
        abs
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&abs))
            .unwrap_or(abs)
    };
    let manifest_abs = manifest_dir
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.to_path_buf());
    let written_abs = abs.canonicalize().unwrap_or(abs);
    match written_abs.strip_prefix(&manifest_abs) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => written_abs.to_string_lossy().into_owned(),
    }
}

/// Resolve a manifest-relative path against `manifest_dir`.
pub fn resolve(manifest_dir: &Path, rel: &str) -> PathBuf {
    let p = Path::new(rel);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        manifest_dir.join(p)
    }
}

pub fn save(manifest: &RunManifest, path: &str) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load(path: &str) -> anyhow::Result<(RunManifest, PathBuf)> {
    let s = std::fs::read_to_string(path)?;
    let m: RunManifest = serde_json::from_str(&s)?;
    let dir = Path::new(path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    Ok((m, dir))
}
