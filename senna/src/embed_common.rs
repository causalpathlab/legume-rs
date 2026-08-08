#![allow(dead_code)]

pub use log::info;
pub use std::sync::{Arc, Mutex};

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;

pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use data_beans::qc_lib::{QcArgs, QcConfig};
pub use data_beans::sparse_data_visitors::*;
pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_stack::*;
pub use data_beans::sparse_io_vector::*;

pub use candle_util::{candle_core, candle_nn};

pub use clap::{Args, Parser, Subcommand, ValueEnum};

pub use matrix_param::io::ParamIo;
pub use matrix_param::traits::{Inference, TwoStatParam};
pub use matrix_util::common_io::{mkdir_parent, remove_file};
pub use matrix_util::dmatrix_rsvd::nystrom_basis;
pub use matrix_util::traits::*;

pub use matrix_util::common_io::file_ext;
pub use matrix_util::dmatrix_util::concatenate_horizontal;

pub use data_beans_alg::collapse_data::*;
pub use data_beans_alg::feature_coarsening::*;
pub use data_beans_alg::feature_coarsening_multilevel::{
    compute_multilevel_feature_coarsening, refine_multilevel_feature_coarsening, FeatureKnnContext,
    MultilevelRefineParams,
};
pub use data_beans_alg::random_projection::*;

/// Build `{prefix}0..{prefix}{k-1}` axis-id column names — the explicit
/// "this column is topic/cluster N" convention used by every K-dim
/// writer in this crate (and pinto's `C{c}` analogue). A reader can
/// recover the integer ID from the column name alone, surviving column
/// reordering, schema audits, and partial subsetting.
#[must_use]
pub fn axis_id_names(prefix: &str, k: usize) -> Vec<Box<str>> {
    (0..k)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

/// Inverse of [`axis_id_names`]. Accepts the explicit `{prefix}{c}` form
/// and the legacy bare-integer fallback (matrix-util's default column
/// names) so older parquets still load.
#[must_use]
pub fn parse_axis_id(name: &str, prefix: &str) -> Option<i64> {
    if let Some(rest) = name.strip_prefix(prefix) {
        if let Ok(c) = rest.parse::<i64>() {
            return Some(c);
        }
    }
    name.parse::<i64>().ok()
}

/// Map every column to its axis ID via [`parse_axis_id`]. Returns `None`
/// if any column doesn't carry an ID — caller can then fall back to a
/// positional check.
#[must_use]
pub fn try_parse_axis_ids(cols: &[Box<str>], prefix: &str) -> Option<Vec<i64>> {
    cols.iter().map(|c| parse_axis_id(c, prefix)).collect()
}

/// Clap-declared defaults for an `Args` struct — see
/// [`matrix_util::clap_defaults`]. Re-exported because senna's arg structs name
/// it by path in `#[serde(default = "...")]`.
pub use matrix_util::clap_defaults::clap_defaults;

/// Shared compute device enum for candle-based models
#[derive(ValueEnum, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl ComputeDevice {
    pub fn to_device(&self, device_no: usize) -> anyhow::Result<candle_core::Device> {
        Ok(match self {
            ComputeDevice::Cpu => candle_core::Device::Cpu,
            ComputeDevice::Cuda => candle_core::Device::new_cuda(device_no)?,
            ComputeDevice::Metal => candle_core::Device::new_metal(device_no)?,
        })
    }
}

/// Batch adjustment method
#[derive(ValueEnum, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum AdjMethod {
    Batch,
    Residual,
}

impl AdjMethod {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            AdjMethod::Batch => "batch",
            AdjMethod::Residual => "residual",
        }
    }
}

/// Shared CNV detection CLI args (used by SVD, topic, masked-topic).
/// Providing `--gff` or `--cnv-ground-truth` turns on the per-sample HMM CNV
/// model from `cnv::per_sample`.
#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "matrix_util::clap_defaults::clap_defaults")]
pub struct CnvArgs {
    #[arg(long, help = "GFF/GTF annotation for CNV detection.")]
    pub gff: Option<Box<str>>,

    #[arg(
        long,
        help = "CNV ground-truth TSV (alternative to --gff; from `data-beans simulate`).",
        hide = true
    )]
    pub cnv_ground_truth: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of CN states (3 = del/neutral/gain; 5/6 = inferCNV i6-style).",
        hide = true
    )]
    pub cnv_states: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "If ≥3, BIC-select K ∈ [3..max] via kmeans on the marginal signal.",
        hide = true
    )]
    pub cnv_gmm_k_max: usize,
}

/// Training score tracker for topic models
pub struct TrainScores {
    pub llik: Vec<f32>,
    pub kl: Vec<f32>,
}

impl TrainScores {
    pub fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let mat = Mat::from_columns(&[
            DVec::from_vec(self.llik.clone()),
            DVec::from_vec(self.kl.clone()),
        ]);

        let score_types = vec![
            "log_likelihood".to_string().into_boxed_str(),
            "kl_divergence".to_string().into_boxed_str(),
        ];

        let epochs: Vec<Box<str>> = (0..mat.nrows())
            .map(|x| (x + 1).to_string().into_boxed_str())
            .collect();

        mat.to_parquet_with_names(
            file_path,
            (Some(&epochs), Some("epoch")),
            Some(&score_types),
        )
    }
}

/// Read a matrix from parquet or delimited text file
pub fn read_mat(file_path: &str) -> anyhow::Result<MatWithNames<Mat>> {
    Ok(match file_ext(file_path)?.as_ref() {
        "parquet" => Mat::from_parquet(file_path)?,
        _ => Mat::read_data(file_path, &['\t', ','], None, Some(0), None, None)?,
    })
}

/// Apply topic smoothing in log-space: exp → mix with uniform → log.
pub fn smooth_topics(
    log_z_nk: candle_core::Tensor,
    alpha: f64,
) -> candle_core::Result<candle_core::Tensor> {
    if alpha > 0.0 {
        let kk = log_z_nk.dim(1)? as f64;
        ((log_z_nk.exp()? * (1.0 - alpha))? + alpha / kk)?.log()
    } else {
        Ok(log_z_nk)
    }
}

/// Output container for bulk data aligned to a reference gene list
pub struct BulkDataOut {
    pub genes: Vec<Box<str>>,
    pub samples: Vec<Box<str>>,
    pub data: Mat,
}

/// Pick the naming rule that bridges a reference gene axis and one or more
/// other axes.
///
/// The signature usually lives on only one axis — a bare-symbol reference
/// carries no delimiter, so by itself it sniffs as `Exact` and no bridge is
/// built. Detect per axis and adopt whichever is informative; canonicalizing
/// under `Gene` is a no-op for names lacking the delimiter, so adopting the
/// informative side is safe for both.
#[must_use]
pub fn reconcile_name_kind(
    reference: &[Box<str>],
    others: &[&[Box<str>]],
) -> auxiliary_data::feature_names::FeatureNameKind {
    use auxiliary_data::feature_names::FeatureNameKind;
    let ref_kind = FeatureNameKind::auto_detect(reference);
    if !ref_kind.is_exact() {
        return ref_kind;
    }
    others
        .iter()
        .map(|axis| FeatureNameKind::auto_detect(axis))
        .find(|k| !k.is_exact())
        .unwrap_or(ref_kind)
}

/// Read bulk data files and align rows to the given gene list.
///
/// Names are reconciled through the shared canonicalizer
/// ([`auxiliary_data::feature_names::FeatureNameKind`]) rather than by string
/// equality, so a bulk file naming genes `ENSG00000105329_TGFB1` aligns to a
/// reference naming them `TGFB1` (and vice versa) with no pre-editing.
///
/// The rule is detected over the UNION of both axes, not the reference alone:
/// the naming signature usually lives on only one side (a bare-symbol reference
/// carries no `_`, so by itself it sniffs as `Exact` and the bridge is never
/// built). Locus-style and mixed axes ride the same path.
pub fn read_bulk_data_aligned(
    bulk_data_files: &[Box<str>],
    genes: &[Box<str>],
) -> anyhow::Result<BulkDataOut> {
    use dashmap::DashMap as HashMap;

    // Read every bulk file up front so the naming rule can see both axes.
    let loaded: Vec<MatWithNames<Mat>> = bulk_data_files
        .iter()
        .map(|f| read_mat(f.as_ref()))
        .collect::<anyhow::Result<_>>()?;

    // Detect PER AXIS and keep whichever is informative. Sniffing the pooled
    // names does not work: `auto_detect` needs a >=50% majority, and a
    // bare-symbol reference pooled with an `ENSG…_SYM` bulk leaves the gene-like
    // share under half, so the pair sniffs as `Exact` and never bridges.
    // Canonicalizing under `Gene` is a no-op for names lacking the delimiter, so
    // adopting the informative side is safe for both.
    let bulk_axes: Vec<&[Box<str>]> = loaded.iter().map(|m| m.rows.as_slice()).collect();
    let name_kind = reconcile_name_kind(genes, &bulk_axes);

    // First writer wins, matching the positional-scan semantics elsewhere.
    let gene_to_position: HashMap<Box<str>, usize> = HashMap::new();
    for (i, g) in genes.iter().enumerate() {
        gene_to_position
            .entry(name_kind.canonicalize(g))
            .or_insert(i);
    }

    let ngenes = genes.len();
    info!("use {ngenes} genes as common features (name rule: {name_kind:?})");

    let mut samples = vec![];
    let mut bulk_data_vec = vec![];

    for (bulk_file, m) in bulk_data_files.iter().zip(loaded) {
        let MatWithNames {
            rows: raw_genes,
            cols: raw_samples,
            mat: raw_ds,
        } = m;

        let ncols = raw_samples.len();

        let mut padded_ds = Mat::zeros(ngenes, ncols);
        let mut matched = 0usize;
        for (i, g) in raw_genes.iter().enumerate() {
            if let Some(r) = gene_to_position.get(&name_kind.canonicalize(g)) {
                // ADD rather than overwrite: canonicalization is many-to-one
                // (several bulk rows can collapse onto one reference gene), and
                // these are counts, so the contributions sum.
                let mut dst = padded_ds.row_mut(*r.value());
                dst += &raw_ds.row(i);
                matched += 1;
            }
        }
        // Unmatched rows are silently zero-filled, so a naming convention the
        // canonicalizer cannot bridge would yield an all-zero bulk and a
        // confident, meaningless downstream answer. Say so instead.
        let frac = matched as f64 / raw_genes.len().max(1) as f64;
        anyhow::ensure!(
            matched > 0,
            "{bulk_file}: none of its {} gene names align to the reference, even after \
             canonicalization ({name_kind:?}). Bulk rows look like `{}`, the reference like \
             `{}`.",
            raw_genes.len(),
            raw_genes.first().map_or("", |g| g.as_ref()),
            genes.first().map_or("", |g| g.as_ref())
        );
        if frac < 0.5 {
            log::warn!(
                "{bulk_file}: only {matched}/{} bulk genes ({:.1}%) align to the reference; the \
                 rest are zero-filled and contribute nothing",
                raw_genes.len(),
                100.0 * frac
            );
        } else {
            info!(
                "{bulk_file}: aligned {matched}/{} genes to the reference",
                raw_genes.len()
            );
        }

        samples.extend(raw_samples);
        bulk_data_vec.push(padded_ds);
    }
    let bulk_data = concatenate_horizontal(&bulk_data_vec)?;

    info!(
        "Read bulk data {} genes x {} samples",
        ngenes,
        samples.len()
    );
    Ok(BulkDataOut {
        genes: genes.to_vec(),
        samples,
        data: bulk_data,
    })
}

// `clip_grads_and_step` lived here as a hand-rolled copy of the global-L2 clip.
// It has been removed in favour of the single `candle_util::vae::clip_grads_and_step`,
// which additionally skips the step when the gradient norm is non-finite
// (this copy laundered one `Inf` gradient into all-`NaN` parameters via
// `Inf * 0`). `senna joint-topic` — its only caller — now imports that one.

/// Per-cell topic proportions `θ [N, K]` (rows sum to 1) from the latent a
/// masked run stores in `{out}.latent.parquet`.
///
/// The simplex heads store `log θ`, so this is `exp`. The Gaussian
/// (`masked-vae`) head stores a raw unconstrained `z` and reaches the decoder
/// through `log_softmax(z)` (see
/// `candle_util::vae::masked_topic::decoder_log_theta`), so its proportions are
/// `softmax(z)`. Plain `exp(z)` — what the θ consumers used to do for every
/// head alike — is not a proportion at all: unnormalized, and unbounded above.
///
/// Anything needing θ from a masked latent goes through here. The raw latent is
/// what gets written to disk and is *not* interchangeable with this.
#[must_use]
pub fn latent_to_theta(z_nk: &Mat, head: candle_util::vae::masked_topic::LatentHead) -> Mat {
    use candle_util::vae::masked_topic::LatentHead;
    match head {
        LatentHead::Softmax | LatentHead::StickBreaking => z_nk.map(f32::exp),
        LatentHead::Gaussian => {
            let mut theta = z_nk.clone();
            softmax_rows_inplace(&mut theta);
            theta
        }
    }
}

/// In-place numerically-stable per-row softmax on a host matrix (`[N, K]` →
/// each row `softmax`ed over K): subtract the row max, `exp`, then divide by
/// the row sum. A degenerate row — all `-inf`, or carrying a `NaN` — has no
/// valid softmax, so it is zeroed rather than left holding `NaN`.
///
/// Shared by [`latent_to_theta`]'s Gaussian arm and any caller that must map a
/// latent onto the simplex without a `LatentHead` in hand (e.g. `impute`, which
/// reads a bare parquet). Applied to a genuine `log θ` latent, the `exp`
/// recovers `θ` and the subsequent renormalization is a no-op (rows already
/// sum to 1).
pub fn softmax_rows_inplace(m: &mut Mat) {
    for mut row in m.row_iter_mut() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        if max.is_finite() {
            row.iter_mut().for_each(|x| {
                *x = (*x - max).exp();
                sum += *x;
            });
        }
        // `sum` is finite and > 0 for any row with a finite max and no NaN; the
        // `else` zeros an all-`-inf` row (max not finite) or a NaN-poisoned one
        // (sum is NaN) — see doc.
        if sum.is_finite() && sum > 0.0 {
            row.iter_mut().for_each(|x| *x /= sum);
        } else {
            row.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}
