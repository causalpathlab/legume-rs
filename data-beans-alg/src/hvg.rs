//! Highly-variable-gene (HVG) selection.
//!
//! Two surfaces:
//! - [`select_hvg`] / [`select_hvg_with_indices`] / [`select_hvg_by_stats`] —
//!   dense-matrix scoring via NB dispersion trend (`σ²(μ) = μ + φ(μ)·μ²`),
//!   ranking genes by excess dispersion above the trend.
//! - [`select_hvg_streaming`] — sparse streaming wrapper that computes
//!   per-gene `(mean, variance)` of raw counts directly from CSC chunks
//!   (lock-free, per-thread accumulators merged at the end), then
//!   delegates to [`select_hvg_by_stats`].
//!
//! Used by senna and chickpea to reweight the random-projection basis so
//! the sketch geometry reflects variable biology rather than housekeeping
//! signal.

use crate::nb_dispersion::DispersionTrend;
use crate::sparse_streaming::streaming_sparse_running_stats;
use clap::Args;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans::utilities::name_matching::GeneIndex;
use log::info;
use matrix_util::common_io::read_name_list;
use matrix_util::traits::RunningStatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[cfg(test)]
mod tests;

/// Select top N highly variable genes by NB excess dispersion.
///
/// # Arguments
/// * `mat` - Expression matrix (rows = samples, columns = genes)
/// * `n_genes` - Number of HVGs to select
///
/// # Returns
/// Matrix with only the selected HVG columns.
pub fn select_hvg(mat: &DMatrix<f32>, n_genes: usize) -> DMatrix<f32> {
    select_hvg_with_indices(mat, n_genes).0
}

/// Select HVGs and return both the subset matrix and the selected indices.
pub fn select_hvg_with_indices(mat: &DMatrix<f32>, n_genes: usize) -> (DMatrix<f32>, Vec<usize>) {
    let (n_samples, n_genes_total) = (mat.nrows(), mat.ncols());

    if n_genes >= n_genes_total {
        let indices: Vec<usize> = (0..n_genes_total).collect();
        return (mat.clone(), indices);
    }

    let (means, vars): (Vec<f32>, Vec<f32>) = (0..n_genes_total)
        .into_par_iter()
        .map(|j| {
            let col = mat.column(j);
            let mean: f32 = col.iter().sum::<f32>() / n_samples as f32;
            let var: f32 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n_samples as f32;
            (mean, var)
        })
        .unzip();

    let hvg_indices = select_hvg_by_stats(&means, &vars, n_genes);

    let mut hvg_mat = DMatrix::zeros(n_samples, hvg_indices.len());
    for (new_j, &old_j) in hvg_indices.iter().enumerate() {
        for i in 0..n_samples {
            hvg_mat[(i, new_j)] = mat[(i, old_j)];
        }
    }

    (hvg_mat, hvg_indices)
}

/// Select top-N HVG indices from pre-computed per-gene means and variances.
/// Returns indices sorted ascending.
pub fn select_hvg_by_stats(means: &[f32], vars: &[f32], n_genes: usize) -> Vec<usize> {
    assert_eq!(means.len(), vars.len());
    let n_genes_total = means.len();
    if n_genes >= n_genes_total {
        return (0..n_genes_total).collect();
    }

    let trend = DispersionTrend::fit(means, vars);
    let mut ranked: Vec<(usize, f32)> = means
        .iter()
        .zip(vars.iter())
        .enumerate()
        .map(|(j, (&mu, &v))| (j, trend.excess(mu, v)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut hvg_indices: Vec<usize> = ranked.iter().take(n_genes).map(|(idx, _)| *idx).collect();
    hvg_indices.sort_unstable();
    hvg_indices
}

/// Shared CLI args for HVG gating of the random projection.
#[derive(Args, Debug, Clone)]
pub struct HvgCliArgs {
    #[arg(
        long = "n-hvg",
        default_value_t = 5000,
        help = "Keep top N highly variable genes (0 disables HVG)",
        long_help = "Select top N genes via binned residual-variance\n\
                     (scanpy/Seurat-style). Collapsing and batch-effect\n\
                     estimation still see all genes. 0 disables HVG.\n\
                     \n\
                     What the selection DOES depends on the command: in\n\
                     `senna` it weights the random projection / pb sketch only\n\
                     (every gene is still trained); in `pinto` and `faba gem`\n\
                     it hard-subsets the trained gene axis."
    )]
    pub n_hvg: usize,

    #[arg(
        long,
        help = "Pre-computed HVG list (replaces --n-hvg selection)",
        long_help = "Use exactly these features instead of selecting HVGs.\n\
                     Takes precedence over --n-hvg. Accepts .txt / .tsv /\n\
                     .csv / .parquet (optionally gzipped); see\n\
                     --must-train-features for the file format."
    )]
    pub feature_list_file: Option<Box<str>>,

    #[arg(
        long = "must-train-features",
        value_name = "FILE",
        help = "Keep these features in the HVG selection regardless of the cut",
        long_help = "Force-include list: UNIONed into the --n-hvg selection (unlike\n\
                     --feature-list-file, which REPLACES it), and also exempt from the\n\
                     --feature-null-fdr drop where that flag exists.\n\
                     \n\
                     WHAT THIS BUYS YOU DEPENDS ON THE COMMAND, because the HVG\n\
                     selection means different things:\n\
                     \n\
                     • `pinto`, `faba gem` — the selection HARD-SUBSETS the trained\n\
                       gene axis, so a feature that misses the cut is not fit at all\n\
                       (it only gets a post-hoc PROJECTED embedding). Naming it here\n\
                       is what puts it in the model. This is the intended use.\n\
                     \n\
                     • `senna` (topic / svd / vae / bge) — HVG only WEIGHTS the random\n\
                       projection used for pseudobulk sketching; every feature is\n\
                       trained either way. Naming it here raises its projection weight\n\
                       and nothing more. It will NOT change whether a gene is fit, so\n\
                       it is not a fix for weak marker embeddings here.\n\
                     \n\
                     Format is inferred from the extension: .txt / .tsv / .csv /\n\
                     .parquet, optionally gzipped. One name per row; a gene-like header\n\
                     (`gene`, `feature`, `symbol`, …) picks the column, else the first\n\
                     column is used. EVERY OTHER COLUMN IS IGNORED, so a curated\n\
                     `gene<TAB>celltype` marker table can be passed as-is.\n\
                     \n\
                     Names are matched leniently (case-insensitive, symbol ↔\n\
                     `ENSG…_SYMBOL` either way); unmatched names are logged, not fatal.\n\
                     A no-op when nothing would drop a feature anyway (--n-hvg 0, and\n\
                     --feature-null-fdr 0 on the subcommands that have it)."
    )]
    pub must_train_features: Option<Box<str>>,
}

/// A force-include list of feature names loaded from `--must-train-features`:
/// features that must enter the fit whether or not they make the HVG cut.
#[derive(Clone)]
pub struct MustTrainFeatures {
    names: Vec<Box<str>>,
    source: Box<str>,
}

impl MustTrainFeatures {
    /// Load the list from any of the supported formats. See the
    /// `--must-train-features` `long_help` for the accepted layouts.
    pub fn load(file_path: &str) -> anyhow::Result<Self> {
        Self::load_union(std::slice::from_ref(&file_path))
    }

    /// Load the **union** of several name lists as one force-train set.
    ///
    /// The motivating pair is an explicit `--must-train-features` list plus the `--markers`
    /// panel a downstream `annotate` will score on: those genes have to be on the trained
    /// axis for their calls to mean anything, and requiring the user to name the same file
    /// twice is exactly the kind of step that gets forgotten. Empty input is not an error —
    /// it just yields an empty set.
    pub fn load_union(file_paths: &[&str]) -> anyhow::Result<Self> {
        let mut names: Vec<Box<str>> = Vec::new();
        for path in file_paths {
            let read = read_name_list(path)?;
            info!("force-train: {} name(s) read from {path}", read.len());
            names.extend(read);
        }
        names.sort_unstable();
        names.dedup();
        Ok(Self {
            names,
            source: file_paths.join(" + ").into(),
        })
    }

    /// Union of **already-loaded** lists, so a caller holding both an explicit force-train
    /// list and a marker panel can merge them without re-reading either file.
    #[must_use]
    pub fn union(parts: &[&Self]) -> Self {
        let mut names: Vec<Box<str>> = parts.iter().flat_map(|p| p.names.iter().cloned()).collect();
        names.sort_unstable();
        names.dedup();
        let source = parts
            .iter()
            .map(|p| p.source.as_ref())
            .collect::<Vec<_>>()
            .join(" + ");
        Self {
            names,
            source: source.into_boxed_str(),
        }
    }

    /// Resolve the list against a vocabulary (feature names, or the interned gene
    /// keys behind them) and return the matching indices, ascending and deduped.
    ///
    /// Matching is lenient — see [`GeneIndex`] — so a `CD8A` panel resolves against
    /// an `ENSG00000153563_CD8A` vocabulary (and vice versa). A name that matches
    /// nothing is reported rather than fatal: marker panels routinely carry genes
    /// that a given assay never captured.
    #[must_use]
    pub fn resolve(&self, vocab: &[Box<str>]) -> Vec<usize> {
        self.resolve_with(&GeneIndex::build(vocab))
    }

    /// [`Self::resolve`] against an **already-built** index. Building a [`GeneIndex`] lowercases
    /// and hash-indexes the whole gene vocabulary — tens of thousands of allocations — so a
    /// caller resolving two lists against the same vocabulary should build it once and call
    /// this twice, rather than paying for the index on each list.
    #[must_use]
    pub fn resolve_with(&self, index: &GeneIndex) -> Vec<usize> {
        let mut hits: Vec<usize> = Vec::with_capacity(self.names.len());
        let mut misses: Vec<&str> = Vec::new();
        for name in &self.names {
            match index.match_gene(name) {
                Some(i) => hits.push(i),
                None => misses.push(name.as_ref()),
            }
        }
        hits.sort_unstable();
        hits.dedup();

        info!(
            "force-train: {} / {} name(s) from {} matched the data",
            hits.len(),
            self.names.len(),
            self.source
        );
        if !misses.is_empty() {
            let preview: Vec<&str> = misses.iter().take(10).copied().collect();
            log::warn!(
                "force-train: {} name(s) not found in the data and ignored: {:?}{}",
                misses.len(),
                preview,
                if misses.len() > preview.len() {
                    " …"
                } else {
                    ""
                }
            );
        }
        hits
    }

    /// [`Self::resolve`] without the match/miss summary, for a second resolution
    /// against the same vocabulary (the caller already reported it once, and
    /// re-reporting reads as a second, different list).
    #[must_use]
    pub fn resolve_quiet(&self, vocab: &[Box<str>]) -> Vec<usize> {
        self.resolve_quiet_with(&GeneIndex::build(vocab))
    }

    /// [`Self::resolve_quiet`] against an already-built index — see [`Self::resolve_with`].
    #[must_use]
    pub fn resolve_quiet_with(&self, index: &GeneIndex) -> Vec<usize> {
        let mut hits: Vec<usize> = self
            .names
            .iter()
            .filter_map(|name| index.match_gene(name))
            .collect();
        hits.sort_unstable();
        hits.dedup();
        hits
    }
}

/// Load the `--must-train-features` list, but only when a feature selection is
/// actually running.
///
/// With selection off every feature is trained already, so the list has nothing to
/// promote — silently doing nothing would look like the flag worked, so say so.
pub fn load_must_train(
    must_train_file: Option<&str>,
    selection_on: bool,
) -> anyhow::Result<Option<MustTrainFeatures>> {
    load_must_train_union(must_train_file.as_slice(), selection_on)
}

/// [`load_must_train`] over the **union** of several lists — an explicit force-train list
/// plus, say, the marker panel the run will later be annotated with. Same no-op guard: with
/// selection off, nothing can be promoted, so say so rather than appear to work.
pub fn load_must_train_union(
    paths: &[&str],
    selection_on: bool,
) -> anyhow::Result<Option<MustTrainFeatures>> {
    if paths.is_empty() {
        return Ok(None);
    }
    if !selection_on {
        log::warn!(
            "force-train list ({}) is a no-op: feature selection is off \
             (--n-hvg 0 / no feature list), so every feature is trained anyway.",
            paths.join(" + ")
        );
        return Ok(None);
    }
    MustTrainFeatures::load_union(paths).map(Some)
}

/// Merge `extra` into `selected` in place (both become ascending + deduped).
/// Returns how many indices were genuinely added.
pub fn union_indices(selected: &mut Vec<usize>, extra: &[usize]) -> usize {
    let before = selected.len();
    selected.extend_from_slice(extra);
    selected.sort_unstable();
    selected.dedup();
    selected.len() - before
}

/// HVG selection result used by SVD / topic / indexed-topic / joint-*
/// pipelines to subset or weight the feature axis.
#[derive(Clone)]
pub struct HvgSelection {
    pub selected_indices: Vec<usize>,
    pub selected_names: Vec<Box<str>>,
    #[allow(dead_code)]
    pub index_map: FxHashMap<usize, usize>,
}

impl HvgSelection {
    /// Per-feature weight vector suitable for `project_columns_weighted`:
    /// 1.0 at selected indices, 0.0 elsewhere.
    #[must_use]
    pub fn row_weights(&self, n_total: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; n_total];
        for &i in &self.selected_indices {
            if i < n_total {
                w[i] = 1.0;
            }
        }
        w
    }
}

/// Stream cells through the sparse backend to compute per-gene mean and
/// variance of raw expression, then select the top `n_features` HVGs via
/// the shared NB-trend scoring routine.
///
/// If `feature_list_file` is supplied it takes precedence and the HVG
/// computation is skipped entirely.
///
/// `must_train` is then UNIONed into whichever selection ran, so a curated panel
/// survives even when it does not make the variance cut.
pub fn select_hvg_streaming(
    data_vec: &SparseIoVec,
    max_features: Option<usize>,
    feature_list_file: Option<&str>,
    must_train: Option<&MustTrainFeatures>,
    block_size: Option<usize>,
) -> anyhow::Result<HvgSelection> {
    let feature_names = data_vec.row_names()?;

    let mut selected_indices = if let Some(path) = feature_list_file {
        load_feature_list_from_file(path, &feature_names)?
    } else {
        let n_features = max_features
            .ok_or_else(|| anyhow::anyhow!("max_features or feature_list_file must be provided"))?;
        if n_features == 0 {
            return Err(anyhow::anyhow!("max_features must be >= 1"));
        }

        let stat = streaming_sparse_running_stats(data_vec, block_size, "HVG")?;
        let selected = select_hvg_by_stats(&stat.mean(), &stat.variance(), n_features);

        info!(
            "Selected {} / {} highly variable features (NB dispersion-trend excess)",
            selected.len(),
            feature_names.len()
        );
        selected
    };

    if let Some(must_train) = must_train {
        let forced = must_train.resolve(&feature_names);
        let added = union_indices(&mut selected_indices, &forced);
        info!(
            "--must-train-features: {added} feature(s) force-added on top of the selection \
             ({} of the {} matched were already selected); {} features kept in total",
            forced.len() - added,
            forced.len(),
            selected_indices.len()
        );
    }

    selected_indices.sort_unstable();
    Ok(build_selection(selected_indices, &feature_names))
}

/// Resolve an explicit `--feature-list-file` against the data's feature names.
/// Same lenient matching as the force-include list, so the two flags agree on
/// what a name means.
fn load_feature_list_from_file(
    file_path: &str,
    all_feature_names: &[Box<str>],
) -> anyhow::Result<Vec<usize>> {
    let names_from_file = read_name_list(file_path)?;

    let index = GeneIndex::build(all_feature_names);
    let mut selected_indices: Vec<usize> = Vec::new();
    let mut not_found = 0usize;
    for name in &names_from_file {
        match index.match_gene(name) {
            Some(idx) => selected_indices.push(idx),
            None => not_found += 1,
        }
    }
    if selected_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No features from file matched data. File: {file_path}"
        ));
    }
    if not_found > 0 {
        log::warn!("{not_found} features from {file_path} not found in the data");
    }
    selected_indices.sort_unstable();
    selected_indices.dedup();

    info!(
        "Loaded {} features from {}",
        selected_indices.len(),
        file_path
    );

    Ok(selected_indices)
}

fn build_selection(selected_indices: Vec<usize>, feature_names: &[Box<str>]) -> HvgSelection {
    let selected_names: Vec<Box<str>> = selected_indices
        .iter()
        .map(|&i| feature_names[i].clone())
        .collect();
    let index_map: FxHashMap<usize, usize> = selected_indices
        .iter()
        .enumerate()
        .map(|(new_i, &old_i)| (old_i, new_i))
        .collect();
    HvgSelection {
        selected_indices,
        selected_names,
        index_map,
    }
}
