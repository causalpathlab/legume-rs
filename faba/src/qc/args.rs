//! CLI surface for `faba qc` and `faba qc-report`.

use crate::common::*;

/// Site-level thresholds shared by `faba qc` (as filters) and `faba qc-report`
/// (as the criteria it sweeps). Every one of these is computed from columns
/// the producers write to `m6a_sites.parquet` / `atoi_sites.parquet`, plus
/// the per-site cell count read off the `_site` matrices, so a cut can be
/// revisited without touching a BAM.
///
/// Thresholds are marginal, and the p-value one in particular is a plain
/// marginal p-value, NOT a q-value. faba does no multiplicity correction:
/// Benjamini-Hochberg needs independence or positive regression dependence,
/// and neighbouring candidate C's are covered by the SAME reads. A read
/// converted at one site is evidence against the unconverted neighbour, so the
/// dependence is not even reliably positive. Under arbitrary dependence the
/// valid procedure is Benjamini-Yekutieli, whose ~ln(m) penalty at the tens of
/// thousands of putative sites of one library means "BY and call almost nothing"
/// or "stop claiming FDR control". The p-values are also non-uniform under H0, because
/// a site only exists once it clears `--min-conversion`. `faba qc-report`
/// shows what each cutoff keeps; how to calibrate it is left to the user.
#[derive(Args, Debug, Clone, serde::Serialize)]
pub struct SiteFilterArgs {
    #[arg(
        long = "site-max-pv",
        default_value_t = 0.05,
        help = "Keep a site only if its p-value is <= this (marginal, no FDR); 1 disables"
    )]
    pub site_max_pv: f32,

    #[arg(
        long = "site-min-log-odds",
        default_value_t = 1e-4,
        help = "m6A only: keep a site only if the raw log odds ratio (signal vs control) is >= this",
        long_help = "m6A only: keep a site only if the RAW cross-product log odds ratio\n\
                     ln((a_w*u_m)/(u_w*a_m)) of signal over control is at least this.\n\
                     A control that never converts reads +inf and passes; a variant that\n\
                     converts equally in both arms reads exactly 0 and is dropped at any\n\
                     positive floor. The default asks only that the signal arm out-convert\n\
                     the control (1e-4 sits below the smallest odds ratio above 1 an integer\n\
                     table can express). Computed from the count columns, not from the\n\
                     Haldane-corrected `log_odds` column, which is for reporting.\n\
                     A-to-I has no control arm and ignores it."
    )]
    pub site_min_log_odds: f32,

    #[arg(
        long = "site-min-coverage",
        default_value_t = 3,
        help = "Keep a site only if total coverage (signal + control reads) is >= this"
    )]
    pub site_min_coverage: u64,

    #[arg(
        long = "site-min-converted",
        default_value_t = 1,
        help = "Keep a site only if it has >= this many converted signal reads"
    )]
    pub site_min_converted: u64,

    #[arg(
        long = "site-min-edit-ratio",
        default_value_t = 0.0,
        help = "Keep a site only if converted / coverage (signal arm) is >= this"
    )]
    pub site_min_edit_ratio: f32,

    #[arg(
        long = "site-max-edit-ratio",
        default_value_t = 1.0,
        help = "Keep a site only if converted / coverage (signal arm) is <= this (e.g. 0.95 to drop variant-like sites)"
    )]
    pub site_max_edit_ratio: f32,

    #[arg(
        long = "site-min-fold",
        default_value_t = 1.0,
        help = "m6A only: keep a site only if signal edit rate / control edit rate is >= this (a clean control passes)"
    )]
    pub site_min_fold: f32,

    #[arg(
        long = "site-min-cells",
        default_value_t = 10,
        help = "Keep a site only if >= this many kept cells carry a converted read at it (summed over batches)",
        long_help = "Keep a site only if at least this many kept cells carry a converted read at it,\n\
                     summed over batches, read off the `_site` matrices. This is the single-cell\n\
                     reproducibility control (the scDART-seq 'seen in >= 10 cells' rule)."
    )]
    pub site_min_cells: usize,
}

#[derive(Args, Debug, serde::Serialize)]
pub struct QcArgs {
    #[arg(help = "A faba output directory (from `faba all` or the standalone producers)")]
    pub input_dir: Box<str>,

    #[arg(
        short = 'o',
        long = "output",
        required = true,
        help = "Output directory for the filtered fileset (must not already contain files)"
    )]
    pub output: Box<str>,

    #[arg(
        long = "no-zip",
        default_value_t = false,
        help = "Write .zarr directories instead of .zarr.zip"
    )]
    pub no_zip: bool,

    ///////////////////////////////////////
    // Feature / cell axes (data-beans) //
    ///////////////////////////////////////
    #[arg(
        short = 'r',
        long = "row-nnz-cutoff",
        default_value_t = 0,
        help = "Drop a feature row with fewer than this many non-zero kept cells (0 = off); site rows use --site-min-cells instead"
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        short = 'c',
        long = "column-nnz-cutoff",
        default_value_t = 0,
        help = "Drop a cell with fewer than this many non-zero genes in its `_count` matrix (0 = off)"
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long = "auto-cutoff",
        default_value_t = false,
        help = "Derive an nnz cutoff per axis by BIC-guarded 2-means on log(1+nnz) where none was given (as `data-beans squeeze --auto-cutoff`)"
    )]
    pub auto_cutoff: bool,

    #[arg(
        long = "show-histogram",
        default_value_t = false,
        help = "Print the nnz histograms behind the cutoffs"
    )]
    pub show_histogram: bool,

    #[arg(
        long = "no-cell-qc",
        default_value_t = false,
        help = "Skip the MAD-outlier cell QC on the `_count` matrix (keep every cell that clears the nnz cutoff)"
    )]
    pub no_cell_qc: bool,

    #[arg(
        long = "qc-mads",
        default_value_t = 5.0,
        help = "MAD band for the cell outlier drops (detected genes and total counts)"
    )]
    pub qc_mads: f32,

    #[arg(
        long = "qc-min-cell-nnz",
        default_value_t = 2,
        help = "Near-empty floor on detected genes per cell for the MAD QC"
    )]
    pub qc_min_cell_nnz: usize,

    #[arg(
        long = "block-size",
        help = "Column block size for the streaming stat passes"
    )]
    pub block_size: Option<usize>,

    #[command(flatten)]
    pub site: SiteFilterArgs,
}

#[derive(Args, Debug, serde::Serialize)]
pub struct QcReportArgs {
    #[arg(help = "A faba output directory (from `faba all` or the standalone producers)")]
    pub input_dir: Box<str>,

    #[arg(
        short = 'o',
        long = "output",
        required = true,
        help = "Output prefix: writes {prefix}.qc_report.parquet"
    )]
    pub output: Box<str>,

    #[arg(long = "width", default_value_t = 50, help = "Width of the ASCII bars")]
    pub width: usize,

    #[arg(
        long = "quiet",
        default_value_t = false,
        help = "Skip the ASCII chart on stderr"
    )]
    pub quiet: bool,

    #[arg(
        long = "block-size",
        help = "Column block size for the streaming stat passes"
    )]
    pub block_size: Option<usize>,
}
