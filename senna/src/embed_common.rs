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

/// [`try_parse_axis_ids`] with the positional fallback every plot uses: a
/// table whose columns carry no `{prefix}{c}` IDs (an embedding's `h0..`,
/// or arbitrary names) is numbered `0..n` in column order.
#[must_use]
pub fn axis_ids_or_positions(cols: &[Box<str>], prefix: &str) -> Vec<i64> {
    try_parse_axis_ids(cols, prefix).unwrap_or_else(|| (0..cols.len() as i64).collect())
}

/// Clap-declared defaults for an `Args` struct — see
/// [`matrix_util::clap_defaults`]. Re-exported because senna's arg structs name
/// it by path in `#[serde(default = "...")]`.
pub use matrix_util::clap_defaults::clap_defaults;

/// Posterior-mean PB matrix `[D, n_pb]`, preferring the batch-adjusted
/// estimate when available. Anchor selection and ambient-profile
/// estimation both want the cleanest cell-type signal — the batch-
/// adjusted posterior strips per-batch effects out of the mean.
pub fn preferred_posterior_mean(collapsed: &CollapsedOut) -> &Mat {
    collapsed.mu_adjusted.as_ref().map_or_else(
        || collapsed.mu_observed.posterior_mean(),
        matrix_param::traits::Inference::posterior_mean,
    )
}

/// Posterior log-mean PB matrix `[D, n_pb]`, preferring batch-adjusted.
/// For Gamma(α, β), returns E[log X] = ψ(α) - log(β).
pub fn preferred_posterior_log_mean(collapsed: &CollapsedOut) -> &Mat {
    collapsed.mu_adjusted.as_ref().map_or_else(
        || collapsed.mu_observed.posterior_log_mean(),
        matrix_param::traits::Inference::posterior_log_mean,
    )
}

/// Shared compute device enum for candle-based models
#[derive(ValueEnum, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Cuda => "cuda",
            ComputeDevice::Metal => "metal",
        })
    }
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

/// NCE training objective for a feature/cell embedding (maps to
/// [`graph_embedding_util::loss::NceObjective`]). Shared by `bge` and `gem`,
/// which train the same engine and must name its losses the same way.
#[derive(ValueEnum, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum NceObjectiveArg {
    /// Per-pair logistic (SGNS): each (positive, negative) pair decided
    /// independently — bge's historical loss, byte-identical runs.
    Logistic,
    /// Sampled-softmax / InfoNCE: the negatives compete with the positive in one
    /// softmax; sharpens separation on dense count data (gem's default).
    Softmax,
}

impl NceObjectiveArg {
    #[must_use]
    pub fn to_ge(&self) -> graph_embedding_util::loss::NceObjective {
        match self {
            NceObjectiveArg::Logistic => graph_embedding_util::loss::NceObjective::Logistic,
            NceObjectiveArg::Softmax => graph_embedding_util::loss::NceObjective::Softmax,
        }
    }
}

/// How far up the collapse tree a gene-pair negative's partner cell is drawn
/// from (CLI surface for [`graph_embedding_util::HopWeights`]).
#[derive(ValueEnum, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum GeneHopsArg {
    /// Equal mass on every hop, sisters through the root.
    Uniform,
    /// More mass on nearby hops: sisters most, the root least.
    Near,
    /// More mass on distant hops: the root most, sisters least.
    Far,
    /// Sisters only — the hardest negatives; a group with no sister escalates.
    Sisters,
    /// Root only — a plain draw over every other group, no tree matching.
    Root,
}

impl GeneHopsArg {
    #[must_use]
    pub fn to_ge(&self) -> graph_embedding_util::HopWeights {
        use graph_embedding_util::HopWeights as H;
        match self {
            GeneHopsArg::Uniform => H::Uniform,
            GeneHopsArg::Near => H::Near,
            GeneHopsArg::Far => H::Far,
            GeneHopsArg::Sisters => H::Sisters,
            GeneHopsArg::Root => H::Root,
        }
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
#[serde(default = "crate::embed_common::clap_defaults")]
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

/// Delimiters a dense bulk table may use.
pub const BULK_DELIMS: [char; 2] = ['\t', ','];

/// Whether the first line of a text table is a header.
///
/// `auto` decides by type (a non-numeric field after column 0 cannot be a
/// count row). Its one blind spot is a header of all-numeric sample IDs,
/// which is what `yes` is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum HeaderArg {
    #[default]
    Auto,
    Yes,
    No,
}

/// How a dense bulk table is read: which axis is genes, and whether its first
/// line is a header. Both default to deciding from the file.
#[derive(Clone, Copy, Debug, Default)]
pub struct BulkTableOpts {
    /// `None` decides from the overlap with the reference genes.
    pub orientation: Option<Orientation>,
    pub header: HeaderArg,
}

/// The two flags that say how a bulk table is laid out. One definition,
/// flattened into every subcommand that reads one, so the help text and the
/// defaults cannot drift between them.
#[derive(Args, Clone, Copy, Debug, Default)]
pub struct BulkTableArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = OrientationArg::Auto,
        help = "Which axis of the bulk table is genes: auto|genes-by-samples|samples-by-genes",
        long_help = "Which axis of the bulk table carries the genes.\n\
                     `auto` scores both axes against the reference genes (the\n\
                     model's gene axis) and takes the decisive one. It refuses\n\
                     when neither matches, or both do.\n\
                     Pass the orientation to settle an ambiguous file."
    )]
    pub bulk_orientation: OrientationArg,

    #[arg(
        long,
        value_enum,
        default_value_t = HeaderArg::Auto,
        help = "Whether a bulk text table's first line is a header: auto|yes|no",
        long_help = "Whether the first line of a bulk text table is a header.\n\
                     `auto` decides by type: a line with a non-numeric field after\n\
                     column 0 cannot be counts. Its blind spot is a header of\n\
                     all-numeric sample IDs, which reads as data; pass `yes` then.\n\
                     The decision and the line are logged. Parquet is unaffected."
    )]
    pub bulk_header: HeaderArg,
}

impl BulkTableArgs {
    #[must_use]
    pub fn opts(&self) -> BulkTableOpts {
        BulkTableOpts {
            orientation: self.bulk_orientation.forced(),
            header: self.bulk_header,
        }
    }
}

/// Read a labeled dense table (parquet, or tab/comma text) with its row and
/// column names.
///
/// Text differs from [`read_mat`] in one way: the header line is DETECTED
/// (or forced by `header`). `read_mat` reads text headerless, which is right
/// for its callers (latent and cluster tables written by this tool) and wrong
/// for a bulk count table from outside, whose first line names the samples.
/// Fed that, `read_mat` parsed the names as counts and died in the parser.
/// Column 0 is the row-name column for text.
///
/// Parquet finds its name column by TYPE: the first string column, wherever
/// it sits. A table with no string column has no names to align on and is
/// refused rather than read with a data column stringified as names.
pub fn read_labeled_mat(file_path: &str, header: HeaderArg) -> anyhow::Result<MatWithNames<Mat>> {
    use matrix_util::common_io::{detect_header_row_numeric, first_line_fields};
    Ok(match file_ext(file_path)?.as_ref() {
        "parquet" => {
            let idx = matrix_util::parquet::first_string_column(file_path)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "{file_path}: no string column to take row names from (every column is \
                     numeric), so there is no name column to align genes on. Write the \
                     table with its gene names as a column."
                )
            })?;
            if idx != 0 {
                info!("{file_path}: row names come from column {idx}, the first string column");
            }
            Mat::from_parquet_with_row_names(file_path, Some(idx))?
        }
        _ => {
            let header_row = match header {
                HeaderArg::Auto => detect_header_row_numeric(file_path, &BULK_DELIMS),
                HeaderArg::Yes => {
                    info!("{file_path}: first line taken as a header (--bulk-header yes)");
                    Some(0)
                }
                HeaderArg::No => {
                    // Reading a non-numeric line as counts would die in the
                    // parser; refuse here, with the fields, instead.
                    if detect_header_row_numeric(file_path, &BULK_DELIMS).is_some() {
                        let fields = first_line_fields(file_path, &BULK_DELIMS)?;
                        anyhow::bail!(
                            "{file_path}: --bulk-header no, but the first line has non-numeric \
                             fields and cannot be read as counts: {fields:?}"
                        );
                    }
                    None
                }
            };
            if header_row.is_none() {
                log::warn!(
                    "{file_path}: read without a header; samples are named by position. Add a \
                     header line, or pass --bulk-header yes if the first line is one."
                );
            }
            Mat::read_data(file_path, &BULK_DELIMS, header_row, Some(0), None, None)?
        }
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

/// Which axis of a dense table carries the genes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Rows are genes, columns are samples: the contract every reader wants.
    GenesBySamples,
    /// Rows are samples, columns are genes: the table is turned on read.
    SamplesByGenes,
}

/// CLI form of [`Orientation`]: `auto` decides from the model's gene axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum OrientationArg {
    #[default]
    Auto,
    GenesBySamples,
    SamplesByGenes,
}

impl OrientationArg {
    /// The forced orientation, or `None` for `auto`.
    #[must_use]
    pub fn forced(self) -> Option<Orientation> {
        match self {
            Self::Auto => None,
            Self::GenesBySamples => Some(Orientation::GenesBySamples),
            Self::SamplesByGenes => Some(Orientation::SamplesByGenes),
        }
    }
}

/// How many labels on `axis` are the model's genes, under the naming rule
/// that bridges THAT axis to the model.
///
/// Scored per axis on purpose. `reconcile_name_kind` adopts whichever side is
/// informative, so one rule chosen off the sample IDs could depress the gene
/// axis's count and flip the answer.
fn genes_matched(axis: &[Box<str>], model_genes: &[Box<str>]) -> usize {
    // Lowercased after canonicalizing, the same key `build_gene_remap_with`
    // aligns on, so the axis this picks is the axis the remap will match.
    let kind = reconcile_name_kind(model_genes, &[axis]);
    let key = |g: &str| kind.canonicalize(g).to_lowercase();
    let model: std::collections::HashSet<String> = model_genes.iter().map(|g| key(g)).collect();
    axis.iter().filter(|l| model.contains(&key(l))).count()
}

/// Decide which axis of a dense table is the gene axis.
///
/// This is measured, not guessed: gene names match the model's gene axis
/// after canonicalization and sample IDs do not, so the two label sets are
/// scored against `model_genes` and the winner has to be decisive (the other
/// axis under a tenth of it). Anything less is reported, with the labels of
/// both axes and of the model, rather than resolved:
/// - nothing matches: a naming failure or the wrong file;
/// - both match: sample IDs that are themselves gene symbols. Pass
///   `--bulk-orientation` to say which is which.
///
/// `forced` short-circuits the evidence.
pub fn resolve_orientation(
    rows: &[Box<str>],
    cols: &[Box<str>],
    model_genes: &[Box<str>],
    forced: Option<Orientation>,
) -> anyhow::Result<Orientation> {
    if let Some(o) = forced {
        info!("bulk orientation forced to {o:?}");
        return Ok(o);
    }
    let r = genes_matched(rows, model_genes);
    let c = genes_matched(cols, model_genes);
    let n = model_genes.len();
    let preview = |v: &[Box<str>]| -> String {
        v.iter()
            .take(5)
            .map(AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(", ")
    };
    anyhow::ensure!(
        r + c > 0,
        "neither axis of the bulk table matches the model's {n} genes, even after \
         canonicalization. Rows look like [{}], columns like [{}], the model's genes like \
         [{}]. Check the gene naming, or that this is the right file.",
        preview(rows),
        preview(cols),
        preview(model_genes)
    );
    let decisive = |win: usize, lose: usize| lose * 10 < win;
    if decisive(r, c) {
        info!(
            "bulk table is genes × samples: matched {r} of the model's {n} genes on the row \
             axis ({c} on the column axis)"
        );
        Ok(Orientation::GenesBySamples)
    } else if decisive(c, r) {
        info!(
            "bulk table is samples × genes; transposing: matched {c} of the model's {n} genes \
             on the column axis ({r} on the row axis)"
        );
        Ok(Orientation::SamplesByGenes)
    } else {
        anyhow::bail!(
            "both axes of the bulk table look like genes: rows matched {r} and columns \
             matched {c} of the model's {n} genes (rows [{}], columns [{}]). Say which is \
             which with --bulk-orientation genes-by-samples or samples-by-genes.",
            preview(rows),
            preview(cols)
        )
    }
}

/// Put genes on the rows: a no-op for `GenesBySamples`, one transpose otherwise.
#[must_use]
pub fn oriented(m: MatWithNames<Mat>, o: Orientation) -> MatWithNames<Mat> {
    match o {
        Orientation::GenesBySamples => m,
        Orientation::SamplesByGenes => MatWithNames {
            rows: m.cols,
            cols: m.rows,
            mat: m.mat.transpose(),
        },
    }
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
///
/// Each file is read with its header detected ([`read_labeled_mat`]) and
/// turned so genes are on the rows ([`resolve_orientation`]), so a samples ×
/// genes table aligns instead of failing as "no gene names match".
pub fn read_bulk_data_aligned(
    bulk_data_files: &[Box<str>],
    genes: &[Box<str>],
    opts: &BulkTableOpts,
) -> anyhow::Result<BulkDataOut> {
    use dashmap::DashMap as HashMap;

    // Read every bulk file up front so the naming rule can see both axes.
    let loaded: Vec<MatWithNames<Mat>> = bulk_data_files
        .iter()
        .map(|f| {
            let m = read_labeled_mat(f.as_ref(), opts.header)?;
            let o = resolve_orientation(&m.rows, &m.cols, genes, opts.orientation)
                .map_err(|e| anyhow::anyhow!("{f}: {e}"))?;
            Ok(oriented(m, o))
        })
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

/// How concentrated a `[N, K]` proportion matrix is: `(mean effective topics,
/// mean max weight)`. Effective topics is `exp(H(θ_row))` — `K` for a flat row,
/// `1` for a one-hot one — so the pair reads directly: a mean near 1 with a max
/// near 1 is a latent that has collapsed every cell onto a single topic.
///
/// Rows are renormalized before the entropy so a caller may pass unnormalized
/// weights; a zero row contributes `(1, 0)`.
#[must_use]
pub fn latent_sharpness(theta_nk: &Mat) -> (f32, f32) {
    let n = theta_nk.nrows();
    if n == 0 {
        return (f32::NAN, f32::NAN);
    }
    let (mut eff_sum, mut max_sum) = (0f64, 0f64);
    for row in theta_nk.row_iter() {
        // `f32::max` DISCARDS a NaN operand, so `NaN.max(1e-12)` is 1e-12 and one
        // non-finite entry would turn the mean into `+inf` while still printing a
        // plausible max-θ beside it. This is the diagnostic that distinguishes a
        // collapsed latent from a diverged one, so a diverged one must say NaN.
        let raw: f32 = row.iter().sum();
        if !raw.is_finite() {
            return (f32::NAN, f32::NAN);
        }
        let z: f32 = raw.max(1e-12);
        let mut h = 0f64;
        let mut m = 0f32;
        for &v in row.iter() {
            let p = v / z;
            if p > 0.0 {
                h -= f64::from(p) * f64::from(p).ln();
            }
            m = m.max(p);
        }
        eff_sum += h.exp();
        max_sum += f64::from(m);
    }
    ((eff_sum / n as f64) as f32, (max_sum / n as f64) as f32)
}

/// Row-wise L2 normalization in place: Euclidean distance on the result
/// equals cosine distance on the input. A ~zero row is left unchanged —
/// normalizing it would blow it up to an arbitrary unit direction (and the
/// retrieval core reads an all-zero row as "no evidence").
pub fn l2_normalize_rows_inplace(m: &mut Mat) {
    for mut row in m.row_iter_mut() {
        let norm = row.norm();
        if norm > 1e-9 {
            row /= norm;
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

#[cfg(test)]
#[path = "embed_common_tests.rs"]
mod embed_common_tests;
