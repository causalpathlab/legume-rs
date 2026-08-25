//! JSON metadata output for pinto runs.
//!
//! Writes a `{prefix}.pinto.json` file containing:
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

    /// Present only when the input's feature axis carried
    /// `{gene}/count/{spliced,unspliced}` rows. Absent means "no channels",
    /// which is not the same as "no genes identified" — see
    /// [`SpliceTrackInfo::n_delta_identified`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub splice: Option<SpliceTrackInfo>,

    pub outputs: OutputFiles,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<Vec<LevelInfo>>,
}

/// The base track for `delta` — see [`SpliceTrackInfo::delta_base`].
pub const DELTA_BASE_SPLICED: &str = "spliced";

/// What a splice-channelized input's two tracks can pin.
///
/// `n_genes` on the parent counts GENES; a channelized matrix has two rows per
/// gene and every gene-side output is on the gene axis, so `n_rows` is the only
/// place the matrix's own shape survives.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpliceTrackInfo {
    /// Matrix rows behind the parent's `n_genes`.
    pub n_rows: usize,
    /// Genes carrying counts on BOTH tracks — the only ones for which a
    /// nascent-minus-mature contrast is identified at all. With no spliced
    /// counts only `beta + delta` is pinned; with no unspliced counts `delta`
    /// enters no likelihood term and would come straight from the prior.
    pub n_delta_identified: usize,
    /// Nascent share of the total library. The second half of the go/no-go: a
    /// negligible share means the contrast is nominally identified and
    /// practically empty.
    pub nascent_count_fraction: f64,
    /// Which track the deviation `delta` is measured FROM. Always `"spliced"`
    /// here: `unspliced = beta + delta`, matching `senna gem`.
    ///
    /// Recorded because the sign is NOT a convention the whole workspace shares.
    /// `senna gem-encoder` uses the opposite base (`spliced = rho + delta`), so
    /// two `delta_feature_embedding.parquet` files are comparable only after
    /// reading this field. Without it the tables look interchangeable and are not.
    pub delta_base: String,
    /// Whether `delta_feature_embedding.parquet` was fit against the LIVE cell
    /// embedding (a refresh) or the cold pseudobulk SVD basis.
    ///
    /// Three states, and they are different findings:
    /// - **absent** — no delta table was written at all (`--gate-mode learned`
    ///   runs no sampler; a spliced-only input identifies no delta). A consumer
    ///   should not go looking for the file.
    /// - `false` — written, but fit against the cold pseudobulk SVD basis
    ///   (`--selection-refresh-epochs 0`, or an early stop before the first
    ///   refresh). A diagnostic, NOT a dictionary: it is in a different frame
    ///   from `feature_embedding.parquet` and must not be added to it.
    /// - `true` — written against the live cell embedding, so the two share a
    ///   frame and delta may be added.
    ///
    /// A consumer that adds delta to the gene embedding MUST check this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_from_refresh: Option<bool>,
    /// Median unspliced counts per identified gene, and the pseudobulk bins they
    /// are spread over. `None` under `--gate-mode learned`, which runs no sampler.
    ///
    /// [`Self::n_delta_identified`] is STRUCTURAL — at least one count on each
    /// track — and passing it says nothing about whether the contrast can be
    /// estimated. These two say that. Below ~1 count per bin the Poisson has
    /// nothing to fit, however high the identified fraction looks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_median_counts: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_counts_per_pseudobulk: Option<f32>,
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

    /// Per-batch contact-association score table from
    /// `pinto lr-activity --edge-scores-only`. Optional; written only when
    /// that mode is run against this prefix. Descriptive phenotypes, not a
    /// test, so it is a separate slot from `lr_activity`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_scores: Option<String>,

    /// `pinto cage` cell embedding `[N × D]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_embedding: Option<String>,

    /// `pinto cage` per-cell bias `[N]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_bias: Option<String>,

    /// `pinto cage` per-gene per-dim posterior mean EFFECTIVE loading
    /// `E[z·β]` `[G × D]` — `pip` multiplied in already, so do NOT gate it
    /// again. Not consumed by training; shipped for downstream use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_posterior_mean: Option<String>,

    /// `pinto cage` feature (gene) embedding `[G × D]` — same shared
    /// D-dim space as `cell_embedding`. Cosine similarity between
    /// feature rows is directly interpretable. The trained effects as-is:
    /// the selection gated the gradient, so it is already expressed here
    /// and must not be re-applied on top.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_embedding: Option<String>,

    /// `pinto cage` per-gene bias `[G]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gene_bias: Option<String>,

    /// Legacy slot: hard cluster labels `[N × 1]` from a run that
    /// clustered CELLS directly. No subcommand writes it now — cage
    /// clusters cell PAIRS and publishes `propensity` /
    /// `link_community` / `gene_community` like `lc` and `dsvd`. Kept so
    /// manifests written by older runs still round-trip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clusters: Option<String>,
}

/// What the dictionary merge actually did, when it ran. `None` when it was
/// skipped or produced no collapses, which is also when `DictMergeFiles` has no
/// paths to point at.
#[derive(Clone, Copy, Debug)]
pub struct DictMergeSummary {
    pub min_nnz: usize,
    pub genes_scored: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DictMergeFiles {
    /// Full agglomerative merge tree (one row per merge step).
    pub merges: String,
    /// Per-fine-community consensus label produced by the cut.
    pub cut: String,
    /// Detection cutoff the merge scored on. Chosen from the data when
    /// `--merge-min-nnz` is unset, so a run is not reproducible from its
    /// outputs without it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_nnz: Option<usize>,
    /// How many genes cleared that cutoff.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genes_scored: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LevelInfo {
    pub tag: String,
    pub level_index: usize,
    pub propensity: String,
    /// Per-edge community parquet. Every subcommand that clusters cell
    /// pairs (lc / dsvd / cage) writes one; `None` is kept for
    /// runs that produce no per-edge table at all, which plot's
    /// `discover_levels` falls back from gracefully.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_community: Option<String>,
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
        link_community: Some(format!("{prefix}.{tag}.link_community.parquet")),
        gene_community: Some(format!("{prefix}.{tag}.gene_community.parquet")),
        entropy_present: Some(true),
    }
}

/// Build the `final` `LevelInfo` every subcommand that publishes a propensity
/// ends on: `lc`'s tail level, and `dsvd`'s / `cage`'s only level. All three
/// write the same three parquets at the bare prefix with an `entropy` column,
/// so the shape lives here once rather than as three literals that have to be
/// kept in step. The sibling of [`lc_level_info`], which does the same job for
/// the `L*` cascade levels.
pub fn final_level_info(prefix: &str, level_index: usize) -> LevelInfo {
    LevelInfo {
        tag: "final".to_string(),
        level_index,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: Some(format!("{prefix}.link_community.parquet")),
        gene_community: Some(format!("{prefix}.gene_community.parquet")),
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
    merge: Option<DictMergeSummary>,
    splice: Option<SpliceTrackInfo>,
    cascade_level_indices: &[usize],
) -> PintoMetadata {
    let prefix = inputs.prefix;
    let dict_merge = merge.map(|m| DictMergeFiles {
        merges: format!("{prefix}.dict_merges.parquet"),
        cut: format!("{prefix}.dict_merges.cut.parquet"),
        min_nnz: Some(m.min_nnz),
        genes_scored: Some(m.genes_scored),
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
    levels.push(final_level_info(prefix, tail_index));

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
        splice,
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
            lr_scores: None,
            cell_embedding: None,
            cell_bias: None,
            feature_posterior_mean: None,
            feature_embedding: None,
            gene_bias: None,
            clusters: None,
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
    let levels = vec![final_level_info(prefix, 0)];

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
        splice: None,
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
            lr_scores: None,
            cell_embedding: None,
            cell_bias: None,
            feature_posterior_mean: None,
            feature_embedding: None,
            gene_bias: None,
            clusters: None,
        },
        levels: Some(levels),
    }
}

/// Helper for `pinto cage` runs. One `final` level, in the same shape
/// `lc` / `dsvd` publish: cage projects every cell pair onto its frozen gene
/// embedding, clusters those pairs into link communities, and derives cell
/// propensity from incident-edge fractions — so the level's slots point at
/// the same three parquets, `entropy` included.
///
/// `has_batch_effects` is `true` when the run had ≥2 batches and
/// `{prefix}.delta.parquet` was written. `inputs.k` is the number of edge
/// clusters, which is what `n_communities` reports — not the embedding
/// width, which is a different quantity and has no slot here.
pub fn create_cage_metadata(
    inputs: &RunInputs<'_>,
    has_batch_effects: bool,
    splice: Option<SpliceTrackInfo>,
) -> PintoMetadata {
    let prefix = inputs.prefix;

    let levels = vec![final_level_info(prefix, 0)];

    PintoMetadata {
        command: "cage".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(inputs.data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: inputs.coord_file.map(|s| s.to_string()),
        n_cells: inputs.n_cells,
        n_genes: inputs.n_genes,
        n_edges: Some(inputs.n_edges),
        n_communities: Some(inputs.k),
        splice,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            gene_community: Some(format!("{prefix}.gene_community.parquet")),
            scores: Some(format!("{prefix}.scores.parquet")),
            batch_effects: has_batch_effects.then(|| format!("{prefix}.delta.parquet")),
            dict_merge: None,
            lr_activity: None,
            lr_scores: None,
            cell_embedding: Some(format!("{prefix}.cell_embedding.parquet")),
            cell_bias: Some(format!("{prefix}.cell_bias.parquet")),
            feature_posterior_mean: Some(format!("{prefix}.feature_posterior_mean.parquet")),
            feature_embedding: Some(format!("{prefix}.feature_embedding.parquet")),
            gene_bias: Some(format!("{prefix}.gene_bias.parquet")),
            clusters: None,
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
        link_community: Some(format!("{prefix}.link_community.parquet")),
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
        splice: None,
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
            lr_scores: None,
            cell_embedding: None,
            cell_bias: None,
            feature_posterior_mean: None,
            feature_embedding: None,
            gene_bias: None,
            clusters: None,
        },
        levels: Some(levels),
    }
}
