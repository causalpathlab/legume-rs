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

    /// Which graph the pairs came from. Absent on manifests written before
    /// this was recorded, and on runs that build no graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphParams>,

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

/// The k's that decided the cell-pair graph.
///
/// `n_edges` says how many pairs a run produced; this says why. Without it the
/// only way to tell a spatial-only run from an augmented one is to open
/// `coord_pairs.parquet` and notice that `edge_kind` is missing — the column is
/// written only when a union happened. Recording the inputs makes a run
/// reproducible from its manifest alone.
///
/// To replay: `knn_base` is `-k` when the run had `coord_file`, and
/// `--knn-expr` when it did not — those are the same graph either way, since
/// without coordinates the expression graph IS the base graph. `knn_expr` is
/// the augmentation, and is meaningful only alongside a `coord_file`.
///
/// Every field is `#[serde(default)]`: this block gains fields over time, and
/// a manifest written by an older build must stay readable rather than
/// failing the WHOLE file — `PintoMetadata::backfill_output` swallows a read
/// error at debug level, so a hard failure here is silent.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct GraphParams {
    /// Neighbours per cell in the base graph: spatial under `--coord`,
    /// expression without it.
    ///
    /// `0` only ever means "an older writer omitted this" — no run builds a
    /// 0-NN graph, and `resolve_knn` refuses one.
    pub knn_base: usize,
    /// Neighbours per cell in the expression graph unioned in. `0` means none,
    /// which is the default under `--coord` and always the case without it.
    /// So `knn_expr > 0` is the test for "this run was augmented" — there is
    /// no second field saying so, because a stored copy could contradict this
    /// one under `#[serde(default)]`.
    ///
    /// This is what was ASKED FOR, not what the union yielded: a scoped search
    /// whose every component is smaller than k contributes no edges and still
    /// reports a positive k here. Read `n_edges` against an unaugmented
    /// control to learn what the union actually added.
    pub knn_expr: usize,
    /// `--knn-expr-scope`, as its CLI spelling. Absent when no expression
    /// search ran — recording it there would assert a scoping decision the run
    /// never made.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knn_expr_scope: Option<String>,
    /// `--reciprocal`: mutual-KNN instead of union-KNN. Applies to every graph
    /// the run builds and changes the edge set, so two runs differing only in
    /// this would otherwise emit identical blocks with different `n_edges`.
    pub reciprocal: bool,
}

impl From<&crate::util::input::ResolvedKnn> for GraphParams {
    /// Built from the RESOLVED spec, not the raw flags, so the manifest records
    /// what the run did rather than what was typed. Those differ on every
    /// defaulted run, which is most of them.
    fn from(knn: &crate::util::input::ResolvedKnn) -> Self {
        use clap::ValueEnum;
        Self {
            knn_base: knn.base,
            knn_expr: knn.augment,
            // Only recorded when a search actually ran. Writing it
            // unconditionally would assert a scoping decision the run never
            // made, which is the whole failure this struct exists to prevent.
            knn_expr_scope: (knn.augment > 0).then(|| {
                knn.scope
                    .to_possible_value()
                    .expect("KnnExprScope has no skipped variants")
                    .get_name()
                    .to_string()
            }),
            reciprocal: knn.reciprocal,
        }
    }
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

    /// Trained PB (finest-level super-cell) embedding table. cage's
    /// trained unit is the PB, not the cell; `cell_embedding` above is a
    /// propensity-weighted readout, not a trained table.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pb_embedding: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pb_bias: Option<String>,

    /// Cell -> finest-level PB id map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_pb: Option<String>,

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
    /// Read a run's manifest, set one output slot, write it back.
    ///
    /// Both `lr-activity` modes back-fill the upstream manifest this way;
    /// keeping it in one place stops the two call sites from drifting on
    /// error handling (one used to swallow write failures silently).
    pub fn backfill_output(path: &std::path::Path, set: impl FnOnce(&mut OutputFiles)) {
        match Self::read(path) {
            Ok(mut meta) => {
                set(&mut meta.outputs);
                if let Err(e) = meta.write(path) {
                    log::warn!("could not update {}: {e}", path.display());
                }
            }
            Err(e) => {
                log::debug!("no upstream manifest at {} to update: {e}", path.display());
            }
        }
    }

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
    /// Which graph produced `n_edges`. Not optional: every builder taking
    /// `RunInputs` builds a graph. `pinto prop` re-cuts an existing pair table
    /// and has none, but it does not go through here.
    pub graph: GraphParams,
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
        graph: Some(inputs.graph.clone()),
        splice,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            gene_community: Some(format!("{prefix}.gene_community.parquet")),
            scores: Some(format!("{prefix}.scores.parquet")),
            dict_merge,
            ..Default::default()
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
        graph: Some(inputs.graph.clone()),
        splice: None,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            gene_community: Some(format!("{prefix}.gene_community.parquet")),
            batch_effects: Some(format!("{prefix}.delta.parquet")),
            ..Default::default()
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
        graph: Some(inputs.graph.clone()),
        splice,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            gene_community: Some(format!("{prefix}.gene_community.parquet")),
            scores: Some(format!("{prefix}.scores.parquet")),
            batch_effects: has_batch_effects.then(|| format!("{prefix}.delta.parquet")),
            cell_embedding: Some(format!("{prefix}.cell_embedding.parquet")),
            pb_embedding: Some(format!("{prefix}.pb_embedding.parquet")),
            pb_bias: Some(format!("{prefix}.pb_bias.parquet")),
            cell_pb: Some(format!("{prefix}.cell_pb.parquet")),
            feature_posterior_mean: Some(format!("{prefix}.feature_posterior_mean.parquet")),
            feature_embedding: Some(format!("{prefix}.feature_embedding.parquet")),
            gene_bias: Some(format!("{prefix}.gene_bias.parquet")),
            ..Default::default()
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
        graph: None,
        n_communities: Some(n_clusters),
        splice: None,
        outputs: OutputFiles {
            coord_pairs: coord_pair_file.map(|s| s.to_string()),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            ..Default::default()
        },
        levels: Some(levels),
    }
}
