//! `faba qc-report`: what each `faba qc` knob keeps, measured.
//!
//! For every criterion, the report sweeps a grid of thresholds with every
//! other criterion off and records what survives, so the table answers "relax
//! to here, keep this many". The grids are dense at the permissive end, because
//! that is where the trade-off is decided. A −log10(p) histogram of every
//! putative site comes first, so the distribution the `max_pv` sweep cuts
//! through is visible in its own right. No error rate is estimated: how a
//! marginal p-value or any other column should be calibrated is left to the
//! user, with the table as the evidence.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use data_beans::aux::feature_rows::{parse_feature_row, APA, COUNT};
use rustc_hash::FxHashMap;

use crate::common::*;
use crate::editing::io::write_record_batch;

use super::args::{QcReportArgs, SiteFilterArgs};
use super::layout::{scan_input_dir, SITE_MODALITIES};
use super::matrix::{open_matrix, row_nnz_sum};
use super::sites::{accumulate_site_cells, read_site_table, SiteTable};

#[derive(Debug, Clone)]
pub struct ReportRow {
    pub modality: Box<str>,
    pub criterion: Box<str>,
    /// What survives: `site`, `gene` (a count/APA unit) or `cell`.
    pub unit: &'static str,
    pub threshold: f64,
    pub n_kept: u64,
    pub n_genes: u64,
}

const PV_GRID: &[f64] = &[
    1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.02, 0.01, 1e-3,
    1e-4,
];
const LOG_ODDS_GRID: &[f64] = &[
    f64::NEG_INFINITY,
    -1.0,
    -0.5,
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
];
const COVERAGE_GRID: &[f64] = &[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0,
];
const CONVERTED_GRID: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0];
const EDIT_RATIO_GRID: &[f64] = &[
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
];
const FOLD_GRID: &[f64] = &[1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0];
const CELLS_GRID: &[f64] = &[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0,
];
const COUNTS_GRID: &[f64] = &[
    1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
];
const GENES_PER_CELL_GRID: &[f64] = &[
    1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0,
];

/// Criterion name of the −log10(p) histogram rows: `threshold` is the bin's
/// lower edge, `n_kept` the sites in `[edge, edge + HIST_BIN)`, the last bin
/// open-ended at `HIST_MAX`.
pub const HIST_CRITERION: &str = "neglog10_pv_hist";
const HIST_BIN: f64 = 0.5;
const HIST_MAX: f64 = 10.0;

/// Histogram of −log10(p) over all putative sites, so the distribution the
/// `max_pv` sweep cuts through is visible in its own right.
fn histogram_neglog10(t: &SiteTable, out: &mut Vec<ReportRow>) {
    let n_bins = (HIST_MAX / HIST_BIN) as usize + 1;
    let mut counts = vec![0u64; n_bins];
    for &p in &t.pv {
        let s = if p <= 0.0 {
            HIST_MAX
        } else {
            (-(p as f64).log10()).clamp(0.0, HIST_MAX)
        };
        counts[((s / HIST_BIN) as usize).min(n_bins - 1)] += 1;
    }
    for (b, &n) in counts.iter().enumerate() {
        out.push(ReportRow {
            modality: t.modality.clone(),
            criterion: HIST_CRITERION.into(),
            unit: "site",
            threshold: b as f64 * HIST_BIN,
            n_kept: n,
            n_genes: 0,
        });
    }
}

fn sweep_sites(t: &SiteTable, n_cells: Option<&[usize]>, out: &mut Vec<ReportRow>) {
    histogram_neglog10(t, out);
    let base = SiteFilterArgs::permissive();
    let mut gene_seen = vec![false; t.n_genes];

    let mut sweep = |criterion: &str, grid: &[f64], make: &dyn Fn(f64) -> SiteFilterArgs| {
        for &x in grid {
            let f = make(x);
            let mut n_kept = 0u64;
            let mut n_genes = 0u64;
            gene_seen.iter_mut().for_each(|g| *g = false);
            for i in 0..t.len() {
                if f.reason(t, i, n_cells.map(|c| c[i])).is_none() {
                    n_kept += 1;
                    let g = t.gene_id[i] as usize;
                    if !gene_seen[g] {
                        gene_seen[g] = true;
                        n_genes += 1;
                    }
                }
            }
            out.push(ReportRow {
                modality: t.modality.clone(),
                criterion: criterion.into(),
                unit: "site",
                threshold: x,
                n_kept,
                n_genes,
            });
        }
    };

    sweep("max_pv", PV_GRID, &|x| SiteFilterArgs {
        site_max_pv: x as f32,
        ..base.clone()
    });
    if t.has_control() {
        sweep("min_log_odds", LOG_ODDS_GRID, &|x| SiteFilterArgs {
            site_min_log_odds: x as f32,
            ..base.clone()
        });
        sweep("min_fold", FOLD_GRID, &|x| SiteFilterArgs {
            site_min_fold: x as f32,
            ..base.clone()
        });
    }
    sweep("min_coverage", COVERAGE_GRID, &|x| SiteFilterArgs {
        site_min_coverage: x as u64,
        ..base.clone()
    });
    sweep("min_converted", CONVERTED_GRID, &|x| SiteFilterArgs {
        site_min_converted: x as u64,
        ..base.clone()
    });
    sweep("min_edit_ratio", EDIT_RATIO_GRID, &|x| SiteFilterArgs {
        site_min_edit_ratio: x as f32,
        ..base.clone()
    });
    if n_cells.is_some() {
        sweep("min_cells", CELLS_GRID, &|x| SiteFilterArgs {
            site_min_cells: x as usize,
            ..base.clone()
        });
    }
}

/// Per-unit `(nnz, sum)` pooled over a modality's channel rows and over
/// batches, for the `count` and `apa` matrices whose unit is the gene.
fn sweep_units(
    modality: &str,
    units: &FxHashMap<Box<str>, (usize, f64)>,
    cell_nnz: &[usize],
    out: &mut Vec<ReportRow>,
) {
    let mut push = |criterion: &str, unit: &'static str, x: f64, n: u64| {
        out.push(ReportRow {
            modality: modality.into(),
            criterion: criterion.into(),
            unit,
            threshold: x,
            n_kept: n,
            n_genes: if unit == "gene" { n } else { 0 },
        });
    };
    for &x in CELLS_GRID {
        let n = units.values().filter(|(nnz, _)| *nnz >= x as usize).count() as u64;
        push("min_cells", "gene", x, n);
    }
    for &x in COUNTS_GRID {
        let n = units.values().filter(|(_, sum)| *sum >= x).count() as u64;
        push("min_counts", "gene", x, n);
    }
    if modality == COUNT {
        for &x in GENES_PER_CELL_GRID {
            let n = cell_nnz.iter().filter(|&&v| v >= x as usize).count() as u64;
            push("min_genes_per_cell", "cell", x, n);
        }
    }
}

fn write_parquet(rows: &[ReportRow], path: &str) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("modality", DataType::Utf8, false),
        Field::new("criterion", DataType::Utf8, false),
        Field::new("unit", DataType::Utf8, false),
        Field::new("threshold", DataType::Float64, false),
        Field::new("n_kept", DataType::UInt64, false),
        Field::new("n_genes", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.modality.as_ref()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|r| r.criterion.as_ref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.unit).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.threshold).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter().map(|r| r.n_kept).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                rows.iter().map(|r| r.n_genes).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    write_record_batch(&batch, path)
}

fn fmt_threshold(x: f64) -> String {
    if x == f64::NEG_INFINITY {
        "-inf".into()
    } else if x.fract() == 0.0 && x.abs() < 1e6 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// One ASCII panel per (modality, criterion), a bar per threshold scaled to
/// the panel's largest count, in the style of `faba metagene`.
pub fn print_ascii(rows: &[ReportRow], width: usize) {
    let mut groups: Vec<(Box<str>, Box<str>)> = Vec::new();
    for r in rows {
        let g = (r.modality.clone(), r.criterion.clone());
        if !groups.contains(&g) {
            groups.push(g);
        }
    }
    for (modality, criterion) in groups {
        let panel: Vec<&ReportRow> = rows
            .iter()
            .filter(|r| r.modality == modality && r.criterion == criterion)
            .collect();
        let unit = panel.first().map(|r| r.unit).unwrap_or("site");
        let max = panel.iter().map(|r| r.n_kept).max().unwrap_or(0).max(1) as f64;
        let is_hist = &*criterion == HIST_CRITERION;
        if is_hist {
            eprintln!(
                "\n== {modality} : -log10(p) histogram  (sites per bin of {HIST_BIN}; last bin >= {HIST_MAX})"
            );
        } else {
            eprintln!("\n== {modality} : {criterion}  (kept {unit}s)");
        }
        for r in panel {
            let n = (width as f64 * r.n_kept as f64 / max).round() as usize;
            let tail = if unit == "site" && !is_hist {
                format!(" {} sites, {} genes", r.n_kept, r.n_genes)
            } else {
                format!(" {} {}s", r.n_kept, unit)
            };
            let label = if is_hist {
                if r.threshold >= HIST_MAX {
                    format!(">={}", fmt_threshold(r.threshold))
                } else {
                    format!(
                        "{}-{}",
                        fmt_threshold(r.threshold),
                        fmt_threshold(r.threshold + HIST_BIN)
                    )
                }
            } else {
                fmt_threshold(r.threshold)
            };
            eprintln!(
                "{:>8} |{}{}|{}",
                label,
                "*".repeat(n),
                " ".repeat(width.saturating_sub(n)),
                tail
            );
        }
    }
}

pub fn run_qc_report(args: &QcReportArgs) -> anyhow::Result<()> {
    let layout = scan_input_dir(&args.input_dir)?;
    let mut rows: Vec<ReportRow> = Vec::new();

    // Editing sites: parquet columns + cells per site from the `_site` matrices.
    let mut tables: FxHashMap<Box<str>, SiteTable> = FxHashMap::default();
    for (modality, path) in &layout.site_tables {
        tables.insert(modality.clone(), read_site_table(path, modality)?);
    }
    let mut site_cells: FxHashMap<Box<str>, FxHashMap<Box<str>, usize>> = FxHashMap::default();
    for m in &layout.matrices {
        let Some(modality) = m.site_modality() else {
            continue;
        };
        let Some(t) = tables.get(modality) else {
            continue;
        };
        let data = open_matrix(&m.path)?;
        let row_names = data.row_names()?;
        let (row_nnz, _) = row_nnz_sum(data.as_ref(), args.block_size)?;
        accumulate_site_cells(
            &row_names,
            &row_nnz,
            t.converted_channel(),
            site_cells.entry(modality.into()).or_default(),
        );
    }
    for modality in SITE_MODALITIES {
        let Some(t) = tables.get(*modality) else {
            continue;
        };
        if t.is_empty() {
            log::warn!("{modality}: site table is empty; nothing to sweep");
            continue;
        }
        info!("{modality}: sweeping {} putative sites", t.len());
        let n_cells = site_cells.get(*modality).map(|acc| t.cells_per_site(acc));
        sweep_sites(t, n_cells.as_deref(), &mut rows);
    }

    // Gene units: `count` and `apa` matrices, pooled over channels and batches.
    for modality in [COUNT, APA] {
        let mut units: FxHashMap<Box<str>, (usize, f64)> = FxHashMap::default();
        let mut cell_nnz: Vec<usize> = Vec::new();
        let mut seen = false;
        for m in layout.matrices.iter().filter(|m| &*m.kind == modality) {
            seen = true;
            let data = open_matrix(&m.path)?;
            let row_names = data.row_names()?;
            let (nnz, sum) = row_nnz_sum(data.as_ref(), args.block_size)?;
            for (name, (&n, &s)) in row_names.iter().zip(nnz.iter().zip(&sum)) {
                let unit: Box<str> = parse_feature_row(name)
                    .map(|f| f.unit())
                    .unwrap_or_else(|| name.clone());
                let e = units.entry(unit).or_insert((0, 0.0));
                e.0 = e.0.max(n);
                e.1 += s;
            }
            if modality == COUNT {
                let ncol = data.num_columns().unwrap_or(0);
                cell_nnz.extend((0..ncol).map(|j| data.column_nnz(j).unwrap_or(0) as usize));
            }
        }
        if seen {
            info!("{modality}: sweeping {} units", units.len());
            sweep_units(modality, &units, &cell_nnz, &mut rows);
        }
    }

    let path = format!("{}.qc_report.parquet", args.output);
    write_parquet(&rows, &path)?;
    info!("wrote {} rows to {path}", rows.len());
    if !args.quiet {
        print_ascii(&rows, args.width);
    }
    Ok(())
}
