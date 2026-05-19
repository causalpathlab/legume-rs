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

use matrix_util::traits::IoOps;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

type Mat = nalgebra::DMatrix<f32>;

/// `(cell_to_pb_per_level [finest-last], cell_names)` — the raw payload
/// loaded from a `cell_to_pb.parquet`, ready to be aligned to a
/// caller's data axis.
pub type InheritedPartition = (Vec<Vec<usize>>, Vec<Box<str>>);

/// Read `{prefix}.cell_to_pb.parquet` (N × num_levels f32, cell-name
/// rows, `level_0..level_{L-1}` columns) into the same
/// [`InheritedPartition`] shape that
/// [`InheritedFromManifest::load_cell_to_pb`] returns. Exposed so
/// callers that don't hold a full `InheritedFromManifest` (e.g.
/// `senna layout`) can reuse the loader.
pub fn load_cell_to_pb_raw(path: &str) -> anyhow::Result<InheritedPartition> {
    let mat_with_names = Mat::from_parquet_with_row_names(path, Some(0))?;
    let cell_names_src = mat_with_names.rows;
    let mat = mat_with_names.mat;
    let n_src = mat.nrows();
    let num_levels = mat.ncols();
    anyhow::ensure!(
        num_levels >= 1,
        "{}: cell_to_pb parquet has 0 data columns",
        path
    );
    anyhow::ensure!(
        cell_names_src.len() == n_src,
        "{}: row-name count {} != matrix rows {}",
        path,
        cell_names_src.len(),
        n_src
    );
    // Parquet columns are level_0..level_{L-1} (finest-first); emit
    // finest-last to match `PreparedData.collapsed_levels`.
    let mut cell_to_pb_per_level: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    for lvl in (0..num_levels).rev() {
        let mut col: Vec<usize> = Vec::with_capacity(n_src);
        for i in 0..n_src {
            col.push(mat[(i, lvl)] as usize);
        }
        cell_to_pb_per_level.push(col);
    }
    log::info!(
        "--from: loaded inherited cell_to_pb {} (num_levels={}, N_src={})",
        path,
        num_levels,
        n_src,
    );
    Ok((cell_to_pb_per_level, cell_names_src))
}

/// Schema version. Bump only on breaking renames or semantic changes.
/// Readers accept any version and log a warning for newer-than-known.
pub const MANIFEST_VERSION: u32 = 1;

/// Subcommand that produced the run. Serde-encoded as kebab-case strings
/// (`"topic"`, `"itopic"`, `"joint-topic"`, `"svd"`, `"joint-svd"`,
/// `"bge"`, `"fne"`) so the JSON wire format is stable across renames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunKind {
    Topic,
    Itopic,
    JointTopic,
    Svd,
    JointSvd,
    Bge,
    Fne,
}

impl RunKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RunKind::Topic => "topic",
            RunKind::Itopic => "itopic",
            RunKind::JointTopic => "joint-topic",
            RunKind::Svd => "svd",
            RunKind::JointSvd => "joint-svd",
            RunKind::Bge => "bge",
            RunKind::Fne => "fne",
        }
    }

    /// Topic-family kinds (topic / itopic / joint-topic) — produce a
    /// probability-simplex β. SVD-family kinds produce signed loadings.
    #[must_use]
    pub fn is_topic_family(self) -> bool {
        matches!(self, RunKind::Topic | RunKind::Itopic | RunKind::JointTopic)
    }
}

impl std::fmt::Display for RunKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub version: u32,
    pub kind: RunKind,
    /// The `--out` prefix the training command was run with.
    pub prefix: String,
    #[serde(default)]
    pub data: RunData,
    #[serde(default)]
    pub outputs: RunOutputs,
    #[serde(default)]
    pub layout: RunLayout,
    #[serde(default)]
    pub cluster: RunCluster,
    #[serde(default)]
    pub annotate: RunAnnotate,
    #[serde(default)]
    pub pseudotime: RunPseudotime,
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
    /// `{out}.model.json` — topic-model metadata for `senna
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
    /// `{out}.feature_embedding.parquet` — D × H learned per-gene embedding
    /// ρ (indexed-topic only). Shared between encoder and decoder under the
    /// ETM factorization β = log_softmax_d(α · ρᵀ); each gene's row is its
    /// learned coordinate in the topic-model embedding space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_embedding: Option<String>,
    /// `{out}.cell_to_pb.parquet` — N × num_levels u32 matrix of the
    /// post-refinement cell→pseudobulk membership per coarsening level
    /// (finest-last to match `collapsed_levels`). Cached so a downstream
    /// `senna {topic, itopic, ce-topic} --from` chain can skip the
    /// expensive HNSW + binary-sort + DC-SBM refinement step and feed
    /// the precomputed partition straight into the per-PB Gamma fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_to_pb: Option<String>,
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

/// Paths to artifacts produced by `senna cluster`. Populated by `senna
/// cluster` when invoked with `--from <manifest>`; otherwise empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunCluster {
    /// `{cluster_out}.clusters.parquet` — cells × 1 cluster id (NaN for unassigned).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clusters: Option<String>,
}

/// Paths to artifacts produced by `senna annotate` — the cluster-based
/// marker enrichment annotation pass. Populated by annotate, not by training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunAnnotate {
    /// `{annotate_out}.annotation.parquet` — N × C cell posterior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation: Option<String>,
    /// `{annotate_out}.argmax.tsv` — per-cell label + max probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argmax: Option<String>,
    /// `{annotate_out}.cluster_celltype_q.parquet` — nClusters × C FDR-sparse
    /// softmax-normalized Q matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_celltype_q: Option<String>,
    /// `{annotate_out}.cluster_celltype_es.parquet` — nClusters × C raw ES.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_celltype_es: Option<String>,
    /// `{annotate_out}.cluster_expression.parquet` — G × nClusters NB-Fisher-
    /// adjusted per-cluster mean expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_expression: Option<String>,
    /// Input marker-gene TSV path (provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub markers: Option<String>,
}

/// Paths to artifacts produced by `senna pseudotime`. Populated when the
/// command is invoked with `--from <manifest>`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunPseudotime {
    /// `{pt_out}.pseudotime.parquet` — cells × 1 scalar pseudotime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pseudotime: Option<String>,
    /// `{pt_out}.principal_graph.nodes.parquet` — K × D centroid
    /// coordinates in the latent space the graph was fit on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_latent: Option<String>,
    /// `{pt_out}.principal_graph.nodes_2d.parquet` — K × 2 centroid
    /// coordinates in the 2D layout space (only written when
    /// `layout.cell_coords` is present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes_2d: Option<String>,
    /// `{pt_out}.principal_graph.edges.parquet` — E × 3 (from, to, weight).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<String>,
    /// Root principal-graph node id used when computing pseudotime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_node: Option<usize>,
    /// `{pt_out}.tree_layout.cell_coords.parquet` — N × 2 cell positions
    /// in a Reingold-Tilford tree layout (x = sibling slot, y = geodesic
    /// pseudotime). Used by `senna plot --colour-by pseudotime` to render
    /// a Monocle-2-style tree plot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_cell_coords: Option<String>,
    /// `{pt_out}.tree_layout.nodes_2d.parquet` — K × 2 principal-graph
    /// node positions in the same tree layout as `tree_cell_coords`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_nodes_2d: Option<String>,
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
    pub fn new(kind: RunKind, prefix: &str) -> Self {
        Self {
            version: MANIFEST_VERSION,
            kind,
            prefix: prefix.into(),
            data: RunData::default(),
            outputs: RunOutputs::default(),
            layout: RunLayout::default(),
            cluster: RunCluster::default(),
            annotate: RunAnnotate::default(),
            pseudotime: RunPseudotime::default(),
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

/// Inverse of [`resolve`]: turn a freshly-written artifact path into the
/// manifest-relative form for storage. Canonicalizes both sides so the
/// strip works across symlinks and `./`-anchored relative paths, and
/// silently keeps the absolute form when the artifact lives outside the
/// manifest dir (e.g. `-o /tmp/foo` while the manifest is in `~/work/`).
#[must_use]
pub fn rel_to_manifest(manifest_dir: &Path, written_path: &str) -> String {
    let abs = PathBuf::from(written_path);
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

/// Resolved chain-from inputs that a downstream training subcommand
/// (`senna {itopic, cell-embedded-topic}`) inherits via `--from`.
/// All paths are already resolved against the manifest's directory so
/// the caller can hand them straight to data loaders.
pub struct InheritedFromManifest {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Vec<Box<str>>,
    /// `--init-feature-embedding` / `--freeze-feature-embedding` prefix.
    /// Resolved to the manifest's `prefix` after directory resolution;
    /// the spec resolver downstream will probe `{prefix}.dictionary.parquet`
    /// (bge/fne layout) or `{prefix}.feature_embedding.parquet`
    /// (topic-family layout).
    pub feature_embedding_prefix: Box<str>,
    /// Path to the source run's `{prefix}.cell_to_pb.parquet`
    /// (`outputs.cell_to_pb`), resolved against the manifest dir.
    /// When present, downstream trainers can use the post-refinement
    /// cell→pb membership and skip the BBKNN + Poisson DC-SBM step.
    /// `None` when the source manifest has no `cell_to_pb` output.
    pub cell_to_pb_path: Option<Box<str>>,
    /// Source manifest kind — useful for logging.
    pub source_kind: RunKind,
}

impl InheritedFromManifest {
    /// Pick the effective input file list: explicit CLI wins; otherwise
    /// the manifest's inherited list. Bails if neither is non-empty so
    /// the data loader gets a clear "no inputs" error instead of an
    /// empty `Vec`.
    pub fn resolve_data(
        inherited: Option<&Self>,
        cli: &[Box<str>],
    ) -> anyhow::Result<Vec<Box<str>>> {
        let out = if !cli.is_empty() {
            cli.to_vec()
        } else {
            inherited.map(|i| i.data_files.clone()).unwrap_or_default()
        };
        anyhow::ensure!(
            !out.is_empty(),
            "no input data files — pass at least one positional .zarr/.h5 path or use --from"
        );
        Ok(out)
    }

    /// Pick the effective batch file list: explicit CLI wins; otherwise
    /// the manifest's inherited list when non-empty.
    pub fn resolve_batch(
        inherited: Option<&Self>,
        cli: Option<&[Box<str>]>,
    ) -> Option<Vec<Box<str>>> {
        match (cli, inherited) {
            (Some(b), _) => Some(b.to_vec()),
            (None, Some(i)) if !i.batch_files.is_empty() => Some(i.batch_files.clone()),
            _ => None,
        }
    }

    /// Read the inherited `cell_to_pb.parquet` into raw
    /// `(cell_to_pb_per_level [finest-last][N_src], cell_names_src)`.
    /// `None` when the source manifest had no `cell_to_pb` output.
    /// Caller is expected to call [`Self::align_cell_to_pb_to_cells`]
    /// against its own `data_vec.column_names()` before feeding the
    /// partition into the collapse.
    pub fn load_cell_to_pb(&self) -> anyhow::Result<Option<InheritedPartition>> {
        let Some(path) = self.cell_to_pb_path.as_deref() else {
            return Ok(None);
        };
        Ok(Some(load_cell_to_pb_raw(path)?))
    }

    /// Align a loaded partition to `data_cell_names`: short-circuit
    /// when orders already match; otherwise reorder by cell name and
    /// bail (with a preview of misses) if any data cell is absent
    /// from the source. Each level's inner `Vec` ends up with
    /// `data_cell_names.len()` entries.
    pub fn align_cell_to_pb_to_cells(
        cell_to_pb_per_level_src: Vec<Vec<usize>>,
        cell_names_src: &[Box<str>],
        data_cell_names: &[Box<str>],
    ) -> anyhow::Result<Vec<Vec<usize>>> {
        if cell_names_src == data_cell_names {
            log::info!("--from: cell-name order matches data axis (no cell_to_pb reorder)");
            return Ok(cell_to_pb_per_level_src);
        }
        let src_index: rustc_hash::FxHashMap<&str, usize> = cell_names_src
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i))
            .collect();
        let mut missing: Vec<&str> = Vec::new();
        let mut perm: Vec<usize> = Vec::with_capacity(data_cell_names.len());
        for name in data_cell_names {
            match src_index.get(name.as_ref()) {
                Some(&i) => perm.push(i),
                None => missing.push(name.as_ref()),
            }
        }
        if !missing.is_empty() {
            let preview: Vec<&str> = missing.iter().copied().take(5).collect();
            anyhow::bail!(
                "--from: {} of {} data cells are absent from the inherited cell_to_pb \
                 (e.g. {:?}); the source run was trained on a different cell set",
                missing.len(),
                data_cell_names.len(),
                preview,
            );
        }
        let n_data = data_cell_names.len();
        let mut out: Vec<Vec<usize>> = Vec::with_capacity(cell_to_pb_per_level_src.len());
        for lvl in cell_to_pb_per_level_src {
            let mut col: Vec<usize> = Vec::with_capacity(n_data);
            for &src_i in &perm {
                col.push(lvl[src_i]);
            }
            out.push(col);
        }
        log::info!(
            "--from: reordered inherited cell_to_pb by cell name ({}→{} cells aligned)",
            cell_names_src.len(),
            n_data,
        );
        Ok(out)
    }
}

/// Load a `senna.json` manifest and extract the fields a downstream
/// trainer would inherit. Bails for source kinds that don't write a
/// feature embedding (SVD / joint-SVD); accepts bge, fne, and the
/// topic-family kinds.
pub fn inherit_from(manifest_path: &str) -> anyhow::Result<InheritedFromManifest> {
    let (m, dir) = RunManifest::load(Path::new(manifest_path))?;
    match m.kind {
        RunKind::Bge | RunKind::Fne | RunKind::Topic | RunKind::Itopic | RunKind::JointTopic => {}
        RunKind::Svd | RunKind::JointSvd => anyhow::bail!(
            "--from manifest kind '{}' has no feature embedding to inherit; \
             use a bge / fne / topic-family run as the source",
            m.kind
        ),
    }
    let to_box = |s: &str| -> Box<str> { resolve(&dir, s).to_string_lossy().into_owned().into() };
    let data_files: Vec<Box<str>> = m.data.input.iter().map(|s| to_box(s)).collect();
    let batch_files: Vec<Box<str>> = m.data.batch.iter().map(|s| to_box(s)).collect();
    let feature_embedding_prefix: Box<str> = to_box(&m.prefix);
    let cell_to_pb_path: Option<Box<str>> = m.outputs.cell_to_pb.as_deref().map(to_box);
    Ok(InheritedFromManifest {
        data_files,
        batch_files,
        feature_embedding_prefix,
        cell_to_pb_path,
        source_kind: m.kind,
    })
}

/// Per-run description assembled by each training subcommand, handed to
/// `write_run_manifest` which owns the `RunManifest` / `save` plumbing.
pub struct RunDescription<'a> {
    pub kind: RunKind,
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
    /// `{basename}.model.json` (topic + itopic; not joint-topic, not
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
    /// Suffix after `{basename}.` for the per-gene feature embedding ρ
    /// parquet (indexed-topic only), e.g. `"feature_embedding.parquet"`.
    /// `None` to omit.
    pub feature_embedding_suffix: Option<&'a str>,
    /// Default `--colour-by` for downstream plot / layout.
    pub default_colour_by: &'a str,
    /// True if the run emits `{basename}.latent.parquet` (per-cell K-dim
    /// latent). All cell-based subcommands set this; `fne` (feature-only)
    /// sets this `false`.
    pub has_latent: bool,
    /// True if the run emits `{basename}.cell_to_pb.parquet` — the
    /// post-refinement cell→pseudobulk membership per coarsening level.
    /// Set by topic-family fits that ran `collapse_columns_multilevel_*`
    /// so a downstream `--from` chain can skip the refinement step.
    pub has_cell_to_pb: bool,
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

    if desc.has_latent {
        m.outputs.latent = Some(format!("{basename}.latent.parquet"));
    }
    if let Some(suf) = desc.dictionary_suffix {
        m.outputs.dictionary = Some(format!("{basename}.{suf}"));
    }
    if desc.has_model {
        m.outputs.model = Some(format!("{basename}.safetensors"));
        m.outputs.metadata = Some(format!("{basename}.model.json"));
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
    if let Some(suf) = desc.feature_embedding_suffix {
        m.outputs.feature_embedding = Some(format!("{basename}.{suf}"));
    }
    if desc.has_cell_to_pb {
        m.outputs.cell_to_pb = Some(format!("{basename}.cell_to_pb.parquet"));
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
        let mut m = RunManifest::new(RunKind::Topic, "/tmp/run1");
        m.data.input = vec!["a.zarr".into(), "b.zarr".into()];
        m.outputs.latent = Some("run1.latent.parquet".into());
        m.layout.cell_coords = Some("run1.cell_coords.parquet".into());
        m.defaults.colour_by = Some("topic".into());
        let json = serde_json::to_string(&m).unwrap();
        let back: RunManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind, RunKind::Topic);
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
