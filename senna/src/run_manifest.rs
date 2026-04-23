// Several constructors / savers here are consumed by the `senna topic`
// / `layout` wiring landing in later passes — silence dead-code until then.
#![allow(dead_code)]

//! Run manifest — the single JSON artifact that ties a senna run
//! together across subcommands.
//!
//! Shape: `senna topic` / `itopic` / `joint-topic` write a fresh
//! manifest at the end of training. `senna layout` reads it, produces 2D
//! coords, and updates the `layout{}` section in place. `senna plot` (and
//! future postprocess commands) read the fully-enriched manifest and
//! work with zero further flags. CLI flags on those commands stay
//! available and win over manifest values when both are supplied.
//!
//! The schema is deliberately narrow — data paths + output artifact
//! paths + a couple of UI defaults. Training hyperparameters are
//! intentionally *not* serialized: the manifest is a run descriptor,
//! not a config language. If you want to re-run with the same settings,
//! that's what shell history and Makefiles are for.
//!
//! All path values are resolved relative to the manifest file's own
//! directory, so a run directory can be moved or copied without
//! breaking downstream reads.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Schema version. Bump only on breaking renames or semantic changes.
/// Readers accept any version and log a warning for newer-than-known.
pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub version: u32,
    /// Subcommand that produced the run: `"topic" | "itopic" | "joint-topic"`.
    pub kind: String,
    /// The `--out` prefix the training command was run with.
    pub prefix: String,
    #[serde(default)]
    pub data: RunData,
    #[serde(default)]
    pub outputs: RunOutputs,
    #[serde(default)]
    pub layout: RunLayout,
    #[serde(default)]
    pub annotate: RunAnnotate,
    #[serde(default)]
    pub defaults: RunDefaults,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunData {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_null: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunOutputs {
    /// `{out}.latent.parquet`: cell × K matrix. For topic runs this is
    /// log-softmax topic proportions; for SVD runs it's component
    /// scores. Consumers that argmax (e.g. `senna plot --colour-by
    /// topic`) should check `kind` before assuming topic semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latent: Option<String>,
    /// `{out}.dictionary.parquet`: gene × K loadings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<String>,
    /// Optional `group_id<TAB>display_name` TSV for `senna plot` labels.
    /// User-populated; no senna subcommand writes this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_labels: Option<String>,
    /// `{out}.cell_proj.parquet` — cell × `proj_dim` random projection
    /// computed during training. Cached so `senna layout` can re-derive
    /// PB structure (via RSVD + multi-level collapse on the projection)
    /// without touching raw data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_proj: Option<String>,
    /// `{out}.safetensors` — trained VAE weights (topic / itopic /
    /// joint-topic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `{out}.metadata.json` — topic-model metadata for `senna
    /// eval-topic` (topic / itopic / joint-topic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// `{out}.pb_gene.parquet` — G × P pseudobulk gene aggregates at the
    /// finest collapse level. Consumed by `senna annotate` to build a
    /// permutation null without touching the raw zarr.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_gene: Option<String>,
    /// `{out}.pb_latent.parquet` — P × K PB-level mean topic proportions
    /// (topic kinds) or mean SVD component scores (svd kinds). For topic
    /// kinds this is derived from the encoder forward on the finest
    /// collapse; for SVD it's `proj_kn.transpose()` at the finest level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_latent: Option<String>,
    /// `{out}.dictionary_empirical.parquet` — G × K empirical β at full
    /// gene resolution: row-scaled by NB Fisher-info weights and column-
    /// normalized to the topic simplex. Avoids the lossy expand-from-coarse
    /// approximation in `dictionary` (which ships at the feature-coarsened
    /// resolution and is interpolated back). `senna annotate` prefers this
    /// when present; falls back to `dictionary` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary_empirical: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunLayout {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_coords: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_coords: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pb_gene_mean: Option<String>,
}

/// Paths to artifacts produced by `senna annotate` — the bipartite-enrichment
/// annotation pass. Populated by annotate, not by training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunAnnotate {
    /// `{annotate_out}.annotation.parquet` — N × C cell posterior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation: Option<String>,
    /// `{annotate_out}.argmax.tsv` — per-cell label + max probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argmax: Option<String>,
    /// `{annotate_out}.topic_celltype_q.parquet` — K × C FDR-sparse
    /// softmax-normalized Q matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic_celltype_q: Option<String>,
    /// `{annotate_out}.topic_celltype_es.parquet` — K × C raw + restandardized
    /// ES diagnostic matrix (long-format: k, c, es, es_std, p, q).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic_celltype_es: Option<String>,
    /// Input marker-gene TSV path (provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub markers: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunDefaults {
    /// Default `--colour-by` for `senna plot`: `"topic" | "cluster" | "pb-id"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colour_by: Option<String>,
    /// Default `--palette` for `senna plot`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette: Option<String>,
}

impl RunManifest {
    pub fn new(kind: &str, prefix: &str) -> Self {
        Self {
            version: MANIFEST_VERSION,
            kind: kind.into(),
            prefix: prefix.into(),
            data: RunData::default(),
            outputs: RunOutputs::default(),
            layout: RunLayout::default(),
            annotate: RunAnnotate::default(),
            defaults: RunDefaults::default(),
        }
    }

    /// Read the manifest and return it together with its parent
    /// directory (used to resolve the relative paths inside).
    pub fn load(path: &Path) -> anyhow::Result<(Self, PathBuf)> {
        let raw = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        let m: Self = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse {}: {e}", path.display()))?;
        if m.version > MANIFEST_VERSION {
            log::warn!(
                "manifest {} is v{} but this binary supports up to v{MANIFEST_VERSION}; \
                 proceeding (unknown fields will be ignored)",
                path.display(),
                m.version
            );
        }
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok((m, dir))
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let s = serde_json::to_string_pretty(self)?;
        fs::write(path, s).map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;
        log::info!("wrote {}", path.display());
        Ok(())
    }
}

/// Resolve a path listed in a manifest against the manifest's own
/// directory. Absolute paths pass through unchanged. Convert the result
/// to a `String` via `.to_string_lossy().into_owned()` at call sites
/// that already hold string-typed paths.
#[must_use]
pub fn resolve(manifest_dir: &Path, rel: &str) -> PathBuf {
    let p = Path::new(rel);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        manifest_dir.join(p)
    }
}

/// Default manifest filename given a run `--out` prefix.
#[must_use]
pub fn default_path(prefix: &str) -> String {
    format!("{prefix}.senna.json")
}

/// Per-run description assembled by each training subcommand, handed to
/// `write_run_manifest` which owns the `RunManifest` / `save` plumbing.
pub struct RunDescription<'a> {
    /// One of `"topic" | "itopic" | "joint-topic" | "svd" | "joint-svd"`.
    pub kind: &'a str,
    /// The `--out` prefix; used both for the manifest filename and as
    /// the `prefix` field inside.
    pub prefix: &'a str,
    pub data_input: &'a [String],
    pub data_batch: &'a [String],
    pub data_input_null: &'a [String],
    /// Suffix after `{basename}.` for the dictionary parquet, e.g.
    /// `"dictionary.parquet"` or (joint-topic) `"base_dictionary.parquet"`.
    /// `None` to omit — SVD runs still produce one, topic runs always do.
    pub dictionary_suffix: Option<&'a str>,
    /// True if the run emits `{basename}.safetensors` +
    /// `{basename}.metadata.json` (topic + itopic; not joint-topic, not
    /// SVD).
    pub has_model: bool,
    /// True if the run emits `{basename}.cell_proj.parquet` — the
    /// cached per-cell random projection layout reuses. All training
    /// subcommands that produce PBs (topic, itopic, joint-topic, svd,
    /// joint-svd) should set this.
    pub has_cell_proj: bool,
    /// Suffix after `{basename}.` for the PB-level gene aggregates parquet,
    /// e.g. `"pb_gene.parquet"`. `None` to omit.
    pub pb_gene_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the PB-level latent parquet, e.g.
    /// `"pb_latent.parquet"`. `None` to omit.
    pub pb_latent_suffix: Option<&'a str>,
    /// Suffix after `{basename}.` for the empirical NB-Fisher-weighted
    /// dictionary parquet, e.g. `"dictionary_empirical.parquet"`. `None`
    /// to omit.
    pub dictionary_empirical_suffix: Option<&'a str>,
    /// Default `--colour-by` for downstream plot / layout.
    pub default_colour_by: &'a str,
}

/// Write `{prefix}.senna.json` describing the run that just finished.
///
/// All artifact paths inside the manifest are stored as *basenames*
/// (e.g. `"run1.latent.parquet"`) so they resolve correctly relative to
/// the manifest's own directory — even when the run directory is moved
/// after writing.
pub fn write_run_manifest(desc: &RunDescription<'_>) -> anyhow::Result<()> {
    let basename = Path::new(desc.prefix)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| desc.prefix.to_string());

    let mut m = RunManifest::new(desc.kind, desc.prefix);
    m.data.input = desc.data_input.to_vec();
    m.data.input_null = desc.data_input_null.to_vec();
    m.data.batch = desc.data_batch.to_vec();

    m.outputs.latent = Some(format!("{basename}.latent.parquet"));
    if let Some(suf) = desc.dictionary_suffix {
        m.outputs.dictionary = Some(format!("{basename}.{suf}"));
    }
    if desc.has_model {
        m.outputs.model = Some(format!("{basename}.safetensors"));
        m.outputs.metadata = Some(format!("{basename}.metadata.json"));
    }
    if desc.has_cell_proj {
        m.outputs.cell_proj = Some(format!("{basename}.cell_proj.parquet"));
    }
    if let Some(suf) = desc.pb_gene_suffix {
        m.outputs.pb_gene = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.pb_latent_suffix {
        m.outputs.pb_latent = Some(format!("{basename}.{suf}"));
    }
    if let Some(suf) = desc.dictionary_empirical_suffix {
        m.outputs.dictionary_empirical = Some(format!("{basename}.{suf}"));
    }
    m.defaults.colour_by = Some(desc.default_colour_by.into());

    let path = default_path(desc.prefix);
    m.save(Path::new(&path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut m = RunManifest::new("topic", "/tmp/run1");
        m.data.input = vec!["a.zarr".into(), "b.zarr".into()];
        m.outputs.latent = Some("run1.latent.parquet".into());
        m.layout.cell_coords = Some("run1.cell_coords.parquet".into());
        m.defaults.colour_by = Some("topic".into());
        let json = serde_json::to_string(&m).unwrap();
        let back: RunManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind, "topic");
        assert_eq!(back.data.input.len(), 2);
        assert_eq!(back.outputs.latent.as_deref(), Some("run1.latent.parquet"));
        assert_eq!(
            back.layout.cell_coords.as_deref(),
            Some("run1.cell_coords.parquet")
        );
    }

    #[test]
    fn resolve_respects_absolute_and_relative() {
        let dir = Path::new("/tmp/runs");
        assert_eq!(
            resolve(dir, "x.parquet"),
            PathBuf::from("/tmp/runs/x.parquet")
        );
        assert_eq!(
            resolve(dir, "/abs/y.parquet"),
            PathBuf::from("/abs/y.parquet")
        );
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let raw = r#"{"version":1,"kind":"topic","prefix":"r","extra_future_field":42}"#;
        let m: RunManifest = serde_json::from_str(raw).unwrap();
        assert_eq!(m.prefix, "r");
    }
}
