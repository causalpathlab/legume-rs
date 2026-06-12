//! Cell-axis quality control (library-only): robust (MAD) outlier detection
//! plus the near-empty floor. Built on the streaming stat collectors in the
//! sibling `qc` module and consumed by senna / pinto — NOT by the data-beans
//! CLI — so it lives in its own module that the binary never compiles (keeps
//! `qc` fully bin-used and avoids dead-code in the bin target).

use crate::qc::{collect_column_stat_across_vec, collect_row_stat_across_vec};
use crate::sparse_io_stack::SparseIoStack;
use crate::sparse_io_vector::SparseIoVec;
use log::warn;
use matrix_util::common_io::write_lines;
use matrix_util::traits::RunningStatOps;
use regex::Regex;

// ---------------------------------------------------------------------------
// Cell-axis quality control: robust (MAD) outlier detection + near-empty floor.
//
// This MAY DROP CELLS (columns): callers that enable QC end up with fewer
// cells in the working set and/or in the per-cell outputs than the input had,
// so downstream consumers must not assume a 1:1 positional mapping to the
// input barcodes — join by cell name. (Feature/row QC, off by default, may
// likewise drop genes/rows.)
//
// Shared by senna and pinto. Two-tier policy (see `compute_qc`):
//   * near-empty cells (`nnz < min_cell_nnz`) are kept in training but
//     dropped from the *output* (gem-style; see `faba gem --min-cell-nnz`);
//   * non-near-empty MAD outliers are dropped from *training* via
//     `SparseIoVec::mask_columns` (so they leave the outputs too).
// ---------------------------------------------------------------------------

/// Which side(s) of the robust band count as outliers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// Flag values far *below* the median (e.g. low counts / few genes).
    Lower,
    /// Flag values far *above* the median (e.g. high mito fraction).
    Upper,
    /// Flag both extremes.
    Both,
}

/// Configuration for cell QC. Plain struct (no clap) so non-clap crates
/// can construct it directly; the clap surface is [`QcArgs`].
#[derive(Clone, Debug)]
pub struct QcConfig {
    /// MAD multiplier for the robust band (larger = more permissive).
    pub n_mads: f32,
    /// Near-empty floor: cells with fewer than this many detected features
    /// (nnz across all rows) are masked at output, not dropped from
    /// training. 0 disables the floor.
    pub min_cell_nnz: usize,
    /// Hard floor on total counts per cell, in addition to MAD. 0 disables.
    pub min_counts_per_cell: f32,
    /// Regex over row (feature) names selecting mitochondrial genes; enables
    /// the per-cell mito-fraction metric. `None` disables it.
    pub mito_pattern: Option<String>,
    /// Optional hard max mitochondrial fraction (0..1).
    pub mito_max_frac: Option<f32>,
    /// Regex selecting ribosomal genes (enables ribo-fraction metric).
    pub ribo_pattern: Option<String>,
    /// Optional hard max ribosomal fraction (0..1).
    pub ribo_max_frac: Option<f32>,
    /// Per-cell metrics that drive MAD outlier flagging.
    pub mad_on_n_genes: bool,
    pub mad_on_counts: bool,
    pub mad_on_mito: bool,
    /// Feature-axis QC (off by default): drop genes expressed in fewer than
    /// this many cells. 0 disables.
    pub feature_min_cells: usize,
    /// Master switch for the MAD train-drop tier. When `false`, only the
    /// near-empty floor (output mask) is computed — used by inference
    /// (predict/impute) so query cells are never silently dropped.
    pub drop_outliers: bool,
    /// Automatic cell calling: pick the per-cell nnz cutoff by 2-means on
    /// `log(1+nnz)` (ambient↔real boundary) and **train-drop** every cell below
    /// it, floored at `min_cell_nnz`. So called-out ambient is removed up front
    /// (via the caller's `mask_columns`), not just output-masked — it never
    /// shapes the model. No-op when `drop_outliers` is false (inference).
    pub auto_cell_cutoff: bool,
    /// Print the per-cell nnz histogram + the suggested/applied cutoff (the
    /// same ASCII summary as `data-beans squeeze --show-histogram`).
    pub qc_histogram: bool,
}

impl Default for QcConfig {
    fn default() -> Self {
        Self {
            n_mads: 5.0,
            min_cell_nnz: 2,
            min_counts_per_cell: 0.0,
            mito_pattern: None,
            mito_max_frac: None,
            ribo_pattern: None,
            ribo_max_frac: None,
            mad_on_n_genes: true,
            mad_on_counts: true,
            mad_on_mito: false,
            feature_min_cells: 0,
            drop_outliers: true,
            auto_cell_cutoff: false,
            qc_histogram: false,
        }
    }
}

/// Median of a non-empty slice via `O(n)` quickselect (reorders in place).
fn median_in_place(xs: &mut [f32]) -> f32 {
    let mid = xs.len() / 2;
    xs.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    xs[mid]
}

/// Robust outlier keep-mask: `keep[i] = false` when `values[i]` falls
/// outside `median ± n_mads · MAD · 1.4826` on the requested `tail`.
/// Count-like metrics should pass `log1p = true` so the band is symmetric
/// on the multiplicative scale. Consolidates the median+MAD idiom used in
/// `cnv::hmm`.
/// `consider` (when `Some`) restricts the median/MAD band to the cells it
/// marks `true` — so e.g. near-empty cells don't contaminate the robust
/// center — while the keep decision is still returned for every cell.
pub fn robust_outlier_keep(
    values: &[f32],
    n_mads: f32,
    tail: Tail,
    log1p: bool,
    consider: Option<&[bool]>,
) -> Vec<bool> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let xform = |v: f32| if log1p { v.max(0.0).ln_1p() } else { v };

    // Band statistics computed over the considered subset only.
    let mut xs: Vec<f32> = values
        .iter()
        .enumerate()
        .filter(|(i, _)| consider.is_none_or(|c| c[*i]))
        .map(|(_, &v)| xform(v))
        .filter(|v| v.is_finite())
        .collect();
    if xs.is_empty() {
        return vec![true; n];
    }
    // O(n) median via quickselect (no full sort).
    let median = median_in_place(&mut xs);

    let mut dev: Vec<f32> = xs.iter().map(|&v| (v - median).abs()).collect();
    let mad = (median_in_place(&mut dev) * 1.4826).max(1e-8);

    let lo = median - n_mads * mad;
    let hi = median + n_mads * mad;
    values
        .iter()
        .map(|&raw| {
            let v = xform(raw);
            if !v.is_finite() {
                return false;
            }
            match tail {
                Tail::Lower => v >= lo,
                Tail::Upper => v <= hi,
                Tail::Both => v >= lo && v <= hi,
            }
        })
        .collect()
}

/// Resolve a regex over row (feature) names into matching row indices, in
/// the same compact-row order as `SparseIoVec::row_names()`. Feeds
/// `collect_column_stat_across_vec(.., Some(&rows), ..)` for subset sums.
pub fn resolve_rows_by_regex(row_names: &[Box<str>], pattern: &str) -> anyhow::Result<Vec<usize>> {
    let re = Regex::new(pattern)?;
    Ok(row_names
        .iter()
        .enumerate()
        .filter(|(_, name)| re.is_match(name))
        .map(|(i, _)| i)
        .collect())
}

/// Per-cell / per-feature QC outcome.
pub struct QcReport {
    /// Cells to retain in training (len = num_columns). `false` only for
    /// non-near-empty MAD outliers — drives `SparseIoVec::mask_columns`.
    pub train_keep: Vec<bool>,
    /// Near-empty cells (len = num_columns): kept in training, masked at
    /// output via [`QcReport::output_keep_idx`].
    pub near_empty: Vec<bool>,
    /// Features to retain (len = num_rows). All `true` unless
    /// `feature_min_cells > 0`.
    pub feature_keep: Vec<bool>,
    pub n_genes: Vec<f32>,
    pub total_counts: Vec<f32>,
    pub mito_frac: Option<Vec<f32>>,
    pub ribo_frac: Option<Vec<f32>>,
    pub n_cells_dropped: usize,
    pub n_features_dropped: usize,
}

impl QcReport {
    /// Indices, in *post-`mask_columns`* column order, of the cells to emit
    /// at output: surviving (training-kept) cells that are not near-empty.
    /// Apply with `Mat::select_rows` on per-cell output matrices and filter
    /// the barcode slice by the same indices.
    pub fn output_keep_idx(&self) -> Vec<usize> {
        let mut idx = Vec::new();
        let mut new_pos = 0usize;
        for c in 0..self.train_keep.len() {
            if !self.train_keep[c] {
                continue; // dropped from training entirely
            }
            if !self.near_empty[c] {
                idx.push(new_pos); // c survives at compact index new_pos
            }
            new_pos += 1;
        }
        idx
    }

    /// Emit-keep indices in the **original** (unmasked) column order: cells
    /// that are neither MAD-dropped nor near-empty. For paths that keep every
    /// cell in training and instead `select_rows` the per-cell outputs
    /// directly (e.g. bge's `UnifiedData`, which has no `mask_columns`).
    pub fn emit_idx_unmasked(&self) -> Vec<usize> {
        (0..self.train_keep.len())
            .filter(|&c| self.train_keep[c] && !self.near_empty[c])
            .collect()
    }
}

/// Compute the two-tier cell-QC report. `block_size` controls the
/// streaming stat passes (`None` = default chunking).
pub fn compute_qc(
    data: &SparseIoVec,
    cfg: &QcConfig,
    block_size: Option<usize>,
) -> anyhow::Result<QcReport> {
    let row_names = data.row_names()?;

    // Per-cell totals across all rows (nnz = n_genes, tot = total_counts).
    // Use the running-stat accessors directly to skip the unused mean/std.
    let col_stat = collect_column_stat_across_vec(data, None, block_size)?;
    let n_genes = col_stat.count_positives();
    let total_counts = col_stat.sum();

    // Optional mito / ribo fractions via row-name regex subsets.
    let frac_for = |pattern: &Option<String>| -> anyhow::Result<Option<Vec<f32>>> {
        let Some(pat) = pattern else {
            return Ok(None);
        };
        let rows = resolve_rows_by_regex(&row_names, pat)?;
        if rows.is_empty() {
            warn!(
                "QC: pattern `{}` matched no features — metric disabled",
                pat
            );
            return Ok(None);
        }
        let sub_tot = collect_column_stat_across_vec(data, Some(&rows), block_size)?.sum();
        let frac = sub_tot
            .iter()
            .zip(total_counts.iter())
            .map(|(&s, &t)| if t > 0.0 { s / t } else { 0.0 })
            .collect::<Vec<f32>>();
        Ok(Some(frac))
    };
    let mito_frac = frac_for(&cfg.mito_pattern)?;
    let ribo_frac = frac_for(&cfg.ribo_pattern)?;

    // Feature axis stat (only when feature QC is enabled).
    let feature_n_cells = if cfg.feature_min_cells > 0 {
        Some(collect_row_stat_across_vec(data, block_size)?.count_positives())
    } else {
        None
    };

    Ok(qc_from_metrics(
        QcMetrics {
            n_genes,
            total_counts,
            mito_frac,
            ribo_frac,
            feature_n_cells,
            n_rows: data.num_rows(),
        },
        cfg,
    ))
}

/// Modality-agnostic cell QC for a [`SparseIoStack`]: per-cell `n_genes` /
/// `total_counts` are **summed across all member modalities** (mirrors
/// `faba gem`'s "a cell rich in any one modality is kept"). Mito/ribo and
/// feature-axis QC are skipped on stacks (row names are per-modality).
pub fn compute_qc_stack(
    stack: &SparseIoStack,
    cfg: &QcConfig,
    block_size: Option<usize>,
) -> anyhow::Result<QcReport> {
    let n_cols = stack.num_columns()?;
    let mut n_genes = vec![0f32; n_cols];
    let mut total_counts = vec![0f32; n_cols];
    for member in stack.stack.iter() {
        let cs = collect_column_stat_across_vec(member, None, block_size)?;
        let ng = cs.count_positives();
        let ct = cs.sum();
        for c in 0..n_cols {
            n_genes[c] += ng[c];
            total_counts[c] += ct[c];
        }
    }
    if cfg.mito_pattern.is_some() || cfg.ribo_pattern.is_some() || cfg.feature_min_cells > 0 {
        warn!("QC: mito/ribo/feature thresholds are ignored for stacked (multi-modal) data");
    }
    Ok(qc_from_metrics(
        QcMetrics {
            n_genes,
            total_counts,
            mito_frac: None,
            ribo_frac: None,
            feature_n_cells: None,
            n_rows: 0, // feature axis not masked on stacks
        },
        cfg,
    ))
}

/// Pre-computed per-cell / per-feature metrics fed to [`qc_from_metrics`].
/// Lets the single-modality and stacked QC paths share one decision rule.
struct QcMetrics {
    n_genes: Vec<f32>,
    total_counts: Vec<f32>,
    mito_frac: Option<Vec<f32>>,
    ribo_frac: Option<Vec<f32>>,
    /// `None` when feature-axis QC is disabled.
    feature_n_cells: Option<Vec<f32>>,
    n_rows: usize,
}

/// Apply the two-tier QC decision (near-empty floor + MAD outliers) to
/// pre-computed metrics. Shared by [`compute_qc`] and [`compute_qc_stack`].
fn qc_from_metrics(m: QcMetrics, cfg: &QcConfig) -> QcReport {
    let QcMetrics {
        n_genes,
        total_counts,
        mito_frac,
        ribo_frac,
        feature_n_cells,
        n_rows,
    } = m;
    let n_cols = n_genes.len();

    // Tier 1: near-empty floor.
    let near_empty: Vec<bool> = n_genes
        .iter()
        .map(|&g| (g as usize) < cfg.min_cell_nnz)
        .collect();

    // Tier 2: MAD outliers among non-near-empty cells. The band statistics
    // are fit over the non-near-empty cells only, so near-empty cells can't
    // contaminate the robust center (matters when a large fraction of cells
    // are near-empty, e.g. multimodal Union matrices).
    let not_near_empty: Vec<bool> = near_empty.iter().map(|&e| !e).collect();
    let consider = Some(not_near_empty.as_slice());
    let mut outlier = vec![false; n_cols];
    if cfg.drop_outliers {
        let mut bands: Vec<Vec<bool>> = Vec::new();
        if cfg.mad_on_n_genes {
            bands.push(robust_outlier_keep(
                &n_genes,
                cfg.n_mads,
                Tail::Lower,
                true,
                consider,
            ));
        }
        if cfg.mad_on_counts {
            bands.push(robust_outlier_keep(
                &total_counts,
                cfg.n_mads,
                Tail::Lower,
                true,
                consider,
            ));
        }
        if cfg.mad_on_mito {
            if let Some(mf) = mito_frac.as_ref() {
                bands.push(robust_outlier_keep(
                    mf,
                    cfg.n_mads,
                    Tail::Upper,
                    false,
                    consider,
                ));
            }
        }
        for c in 0..n_cols {
            if near_empty[c] {
                continue; // floor takes precedence; never a train-drop
            }
            let mut fail = total_counts[c] < cfg.min_counts_per_cell;
            if let (Some(mf), Some(cap)) = (mito_frac.as_ref(), cfg.mito_max_frac) {
                fail |= mf[c] > cap;
            }
            if let (Some(rf), Some(cap)) = (ribo_frac.as_ref(), cfg.ribo_max_frac) {
                fail |= rf[c] > cap;
            }
            for band in bands.iter() {
                if !band[c] {
                    fail = true;
                    break;
                }
            }
            outlier[c] = fail;
        }

        // Automatic cell calling: 2-means on n_genes picks the ambient↔real
        // boundary; every cell below it (including near-empty ones) is
        // train-dropped, so the caller's `mask_columns(train_keep)` removes it
        // up front. Floored at `min_cell_nnz`. The histogram + cutoff are
        // optionally printed (same ASCII summary as `squeeze --show-histogram`).
        if cfg.auto_cell_cutoff || cfg.qc_histogram {
            let suggested = crate::qc::suggest_nnz_cutoff(&n_genes);
            let floor = if cfg.auto_cell_cutoff {
                cfg.min_cell_nnz.max(suggested.unwrap_or(cfg.min_cell_nnz))
            } else {
                cfg.min_cell_nnz
            };
            crate::qc::print_nnz_summary("Cell", &n_genes, floor, suggested);
            if cfg.auto_cell_cutoff {
                for (c, &g) in n_genes.iter().enumerate() {
                    if (g as usize) < floor {
                        outlier[c] = true;
                    }
                }
            }
        }
    }

    let mut train_keep: Vec<bool> = outlier.iter().map(|&o| !o).collect();
    let mut n_cells_dropped = outlier.iter().filter(|&&o| o).count();

    // Guardrail: never produce an empty matrix.
    if n_cols > 0 && n_cells_dropped >= n_cols {
        warn!(
            "QC would drop all {} cells — keeping all (check thresholds)",
            n_cols
        );
        train_keep = vec![true; n_cols];
        n_cells_dropped = 0;
    }

    // Feature axis (only when feature_n_cells was computed).
    let (feature_keep, n_features_dropped) = match feature_n_cells {
        Some(n_cells_expr) => {
            let keep: Vec<bool> = n_cells_expr
                .iter()
                .map(|&c| (c as usize) >= cfg.feature_min_cells)
                .collect();
            let dropped = keep.iter().filter(|&&k| !k).count();
            if !keep.is_empty() && dropped >= keep.len() {
                warn!("QC would drop all features — keeping all");
                (vec![true; keep.len()], 0)
            } else {
                (keep, dropped)
            }
        }
        None => (vec![true; n_rows], 0),
    };

    QcReport {
        train_keep,
        near_empty,
        feature_keep,
        n_genes,
        total_counts,
        mito_frac,
        ribo_frac,
        n_cells_dropped,
        n_features_dropped,
    }
}

/// Filter a per-cell `Vec<T>` in lockstep with a cell keep-mask. Used to
/// keep batch labels / coordinates aligned after `mask_columns`.
pub fn filter_by_keep<T: Clone>(items: &[T], keep: &[bool]) -> Vec<T> {
    items
        .iter()
        .zip(keep.iter())
        .filter(|&(_, &k)| k)
        .map(|(x, _)| x.clone())
        .collect()
}

/// Write a per-cell QC table (TSV): name, n_genes, total_counts,
/// [mito_frac], [ribo_frac], near_empty (0/1), train_keep (0/1).
pub fn write_qc_report(
    path: &str,
    cell_names: &[Box<str>],
    report: &QcReport,
) -> anyhow::Result<()> {
    use std::fmt::Write as _;
    let n = report.train_keep.len();
    anyhow::ensure!(
        cell_names.len() == n,
        "write_qc_report: {} names != {} cells",
        cell_names.len(),
        n
    );

    let mut header = String::from("#cell\tn_genes\ttotal_counts");
    if report.mito_frac.is_some() {
        header.push_str("\tmito_frac");
    }
    if report.ribo_frac.is_some() {
        header.push_str("\tribo_frac");
    }
    header.push_str("\tnear_empty\ttrain_keep");

    let mut lines: Vec<Box<str>> = Vec::with_capacity(n + 1);
    lines.push(header.into_boxed_str());
    for c in 0..n {
        let mut line = String::new();
        let _ = write!(
            line,
            "{}\t{}\t{}",
            cell_names[c], report.n_genes[c], report.total_counts[c]
        );
        if let Some(mf) = report.mito_frac.as_ref() {
            let _ = write!(line, "\t{}", mf[c]);
        }
        if let Some(rf) = report.ribo_frac.as_ref() {
            let _ = write!(line, "\t{}", rf[c]);
        }
        let _ = write!(
            line,
            "\t{}\t{}",
            report.near_empty[c] as u8, report.train_keep[c] as u8
        );
        lines.push(line.into_boxed_str());
    }
    write_lines(&lines, path)
}

/// Clap surface for cell QC, shared by senna and pinto subcommands.
///
/// **Cell QC is ON by default and CAN DROP CELLS (columns).** Low-quality
/// cells are removed: near-empty cells are omitted from the per-cell outputs,
/// and robust (MAD) outlier cells are excluded from training entirely. As a
/// result, the per-cell output files (`*.latent.parquet`, `*.cell_proj.parquet`,
/// `*.cell_to_pb.parquet`, `*.cell_embedding.parquet`, pinto propensity, etc.)
/// **may contain fewer rows (cells) than the input** — do not assume a 1:1,
/// positional correspondence with the input barcodes; always join by the cell
/// name/barcode column. Pass `--no-qc` to disable and keep every cell, or
/// `--qc-report <path>` to dump the per-cell keep/drop flags. (Feature/row QC
/// is OFF unless `--qc-feature-min-cells` is set.)
#[derive(clap::Args, Debug, Clone)]
pub struct QcArgs {
    /// Disable cell QC entirely (keep every input cell).
    #[arg(
        long = "no-qc",
        default_value_t = false,
        long_help = "Disable cell quality control entirely and keep every input cell.\n\
                     \n\
                     By DEFAULT (without this flag) cell QC is ON and DROPS cells\n\
                     (columns): near-empty cells (fewer than --qc-min-cell-nnz\n\
                     detected features) are omitted from the per-cell outputs, and\n\
                     robust MAD outliers are excluded from training, so output\n\
                     parquet files may have FEWER rows than the input. Join outputs\n\
                     by the cell/barcode name column, not by position. Use\n\
                     --qc-report to see what was dropped."
    )]
    pub no_qc: bool,

    /// MAD multiplier for the robust outlier band; smaller = drops more cells.
    #[arg(long = "qc-mads", default_value_t = 5.0)]
    pub qc_mads: f32,

    #[arg(
        long = "qc-min-cell-nnz",
        default_value_t = 2,
        help = "Near-empty floor: cells with fewer detected features than this are",
        long_help = "Near-empty floor: cells with fewer detected features than this\n\
                     are dropped from the per-cell outputs (still kept in training)."
    )]
    pub qc_min_cell_nnz: usize,

    #[arg(
        long = "qc-min-counts",
        default_value_t = 0.0,
        help = "Hard floor on total counts per cell — cells below it are dropped",
        long_help = "Hard floor on total counts per cell — cells below it are dropped\n\
                     from training (0 disables)."
    )]
    pub qc_min_counts: f32,

    #[arg(
        long = "qc-mito-pattern",
        help = "Regex over feature names selecting mitochondrial genes (enables the",
        long_help = "Regex over feature names selecting mitochondrial genes (enables\n\
                     the mito-fraction outlier metric), e.g. `(?i)^MT-`."
    )]
    pub qc_mito_pattern: Option<String>,

    /// Hard max mitochondrial fraction (0..1).
    #[arg(long = "qc-mito-max-frac")]
    pub qc_mito_max_frac: Option<f32>,

    /// Regex over feature names selecting ribosomal genes.
    #[arg(long = "qc-ribo-pattern")]
    pub qc_ribo_pattern: Option<String>,

    /// Hard max ribosomal fraction (0..1).
    #[arg(long = "qc-ribo-max-frac")]
    pub qc_ribo_max_frac: Option<f32>,

    #[arg(
        long = "qc-feature-min-cells",
        default_value_t = 0,
        help = "Feature/row QC (off by default): DROP genes (rows) expressed in fewer",
        long_help = "Feature/row QC (off by default): DROP genes (rows) expressed in\n\
                     fewer than this many cells. Not applied by `bge` (cell-only\n\
                     QC there)."
    )]
    pub qc_feature_min_cells: usize,

    #[arg(
        long = "qc-report",
        help = "Write a per-cell QC table (.tsv) of metrics + near_empty/train_keep",
        long_help = "Write a per-cell QC table (.tsv) of metrics +\n\
                     near_empty/train_keep flags, so you can see exactly which\n\
                     cells were dropped."
    )]
    pub qc_report: Option<Box<str>>,

    #[arg(
        long = "no-qc-auto-cutoff",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Disable automatic cell calling (then --qc-min-cell-nnz alone is the floor)",
        long_help = "Automatic cell calling (ON by default; pass `--no-qc-auto-cutoff` to\n\
                     disable). Picks the per-cell nnz cutoff by 2-means clustering of\n\
                     log(1+nnz) — the ambient↔real-cell boundary, same routine as\n\
                     `data-beans squeeze` — train-drops every cell below it (floored at\n\
                     --qc-min-cell-nnz), and prints the nnz histogram. Because the caller\n\
                     subsets the backend to `train_keep` (mask_columns), the called-out\n\
                     ambient droplets are removed up front and never shape the model.\n\
                     Disabled → --qc-min-cell-nnz is the only floor. (Note: 2-means always\n\
                     splits, so on already-filtered inputs it may still trim a low tail;\n\
                     disable it then, or set --qc-min-cell-nnz.)"
    )]
    pub qc_auto_cutoff: bool,

    #[arg(
        long = "qc-histogram",
        default_value_t = false,
        help = "Print the per-cell nnz histogram even when auto-cutoff is disabled",
        long_help = "Print an ASCII histogram of the per-cell nnz distribution with the\n\
                     suggested (2-means) and applied cutoffs marked — the same summary as\n\
                     `data-beans squeeze --show-histogram`. Auto-cutoff already prints it;\n\
                     this forces it under `--no-qc-auto-cutoff` (e.g. to pick\n\
                     --qc-min-cell-nnz by hand)."
    )]
    pub qc_histogram: bool,
}

impl QcArgs {
    /// QC config, or `None` under `--no-qc`.
    pub fn to_config(&self) -> Option<QcConfig> {
        (!self.no_qc).then(|| QcConfig {
            n_mads: self.qc_mads,
            min_cell_nnz: self.qc_min_cell_nnz,
            min_counts_per_cell: self.qc_min_counts,
            mito_pattern: self.qc_mito_pattern.clone(),
            mito_max_frac: self.qc_mito_max_frac,
            ribo_pattern: self.qc_ribo_pattern.clone(),
            ribo_max_frac: self.qc_ribo_max_frac,
            mad_on_n_genes: true,
            mad_on_counts: true,
            mad_on_mito: self.qc_mito_pattern.is_some(),
            feature_min_cells: self.qc_feature_min_cells,
            drop_outliers: true,
            auto_cell_cutoff: self.qc_auto_cutoff,
            qc_histogram: self.qc_histogram,
        })
    }
}

#[cfg(test)]
mod qc_tests {
    use super::*;

    #[test]
    fn robust_lower_flags_low_outlier() {
        let v = vec![100.0, 110.0, 90.0, 105.0, 95.0, 1.0];
        let keep = robust_outlier_keep(&v, 3.0, Tail::Lower, true, None);
        assert!(!keep[5], "the value 1.0 should be a lower outlier");
        assert!(keep[..5].iter().all(|&k| k), "the bulk should be kept");
        // upper tail must not flag a low value
        let keep_up = robust_outlier_keep(&v, 3.0, Tail::Upper, true, None);
        assert!(keep_up[5], "lower outlier kept under Tail::Upper");
    }

    #[test]
    fn robust_uniform_keeps_all() {
        let v = vec![7.0; 20];
        let keep = robust_outlier_keep(&v, 5.0, Tail::Both, true, None);
        assert!(keep.iter().all(|&k| k));
    }

    #[test]
    fn auto_cutoff_train_drops_ambient() {
        // Bimodal nnz: 3 ambient (~2) + 3 real (~100). With auto cell calling on
        // (MAD tiers off so the auto floor is the only decider), the 2-means
        // cutoff lands between the modes and the ambient cells are train-dropped.
        let n_genes = vec![2.0, 2.0, 2.0, 100.0, 100.0, 100.0];
        let cfg = QcConfig {
            auto_cell_cutoff: true,
            qc_histogram: false,
            drop_outliers: true,
            mad_on_n_genes: false,
            mad_on_counts: false,
            mad_on_mito: false,
            min_cell_nnz: 0,
            ..QcConfig::default()
        };
        let report = qc_from_metrics(
            QcMetrics {
                n_genes: n_genes.clone(),
                total_counts: n_genes,
                mito_frac: None,
                ribo_frac: None,
                feature_n_cells: None,
                n_rows: 0,
            },
            &cfg,
        );
        assert_eq!(
            report.train_keep,
            vec![false, false, false, true, true, true]
        );
        assert_eq!(report.n_cells_dropped, 3);
    }

    #[test]
    fn robust_consider_excludes_contaminants_from_band() {
        // A real cluster around 100 plus a large block of near-empty 0s that
        // would drag a naive median/MAD down. With `consider` masking the
        // zeros out of the band, a genuine low real cell (40) is still flagged.
        let mut v = vec![0.0; 12];
        v.extend([100.0, 102.0, 98.0, 101.0, 99.0, 40.0]);
        let mut consider = vec![false; 12];
        consider.extend([true; 6]);
        let keep = robust_outlier_keep(&v, 2.0, Tail::Lower, true, Some(&consider));
        assert!(
            !keep[17],
            "40 is a lower outlier of the real cluster (~100)"
        );
        // Without `consider`, the zeros dominate the median and 40 survives.
        let keep_naive = robust_outlier_keep(&v, 2.0, Tail::Lower, true, None);
        assert!(
            keep_naive[17],
            "naive band (contaminated by zeros) keeps 40"
        );
    }

    #[test]
    fn output_keep_idx_skips_dropped_and_near_empty() {
        // cells: 0 keep, 1 near-empty (kept in training), 2 MAD-drop, 3 keep
        let report = QcReport {
            train_keep: vec![true, true, false, true],
            near_empty: vec![false, true, false, false],
            feature_keep: vec![],
            n_genes: vec![],
            total_counts: vec![],
            mito_frac: None,
            ribo_frac: None,
            n_cells_dropped: 1,
            n_features_dropped: 0,
        };
        // post-mask order: cell0->0, cell1->1, cell3->2 (cell2 dropped)
        // emit non-near-empty survivors: 0 and 2
        assert_eq!(report.output_keep_idx(), vec![0, 2]);
    }
}
