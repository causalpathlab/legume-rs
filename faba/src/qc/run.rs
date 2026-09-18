//! `faba qc`: read a faba output directory, decide cells, features and sites,
//! and write a new, filtered fileset. Never in place.
//!
//! Order of operations, per batch:
//! 1. cells are decided once, on `{batch}_count`, and the same keep set is
//!    applied to every matrix of the batch so modalities stay column-aligned;
//! 2. sites are decided on the parquet columns plus the per-site kept-cell
//!    count read off the `_site` matrices (pooled over batches);
//! 3. `_site` matrices keep both channels of every kept site;
//! 4. gene-level `{batch}_m6a` / `{batch}_atoi` are RE-POOLED from the filtered
//!    site matrix, so they cannot disagree with the site cut;
//! 5. every other matrix takes the cell keep set and `--row-nnz-cutoff`;
//! 6. every other file is copied through.

use std::sync::Arc;

use crate::common::*;
use data_beans::qc_lib::{compute_qc, write_qc_report, QcConfig};
use data_beans::sparse_io_vector::SparseIoVec;
use rustc_hash::{FxHashMap, FxHashSet};

use super::args::QcArgs;
use super::layout::{file_name, scan_input_dir, InputLayout, MatrixFile, SITE_MODALITIES};
use super::matrix::{
    open_matrix, row_nnz_over_columns, select_columns, shape, write_subset, Backend, OutSpec,
    Written,
};
use super::repool::repool_gene_level;
use super::sites::{
    accumulate_site_cells, read_site_table, site_matrix_rows, write_site_tables, SiteTable,
};

struct SummaryRow {
    file: Box<str>,
    before: (usize, usize, usize),
    after: (usize, usize, usize),
}

/// Decide the cells of one batch on its `_count` matrix, through the
/// data-beans cell QC in one pass: a near-empty floor (the larger of
/// `--column-nnz-cutoff` and `--qc-min-cell-nnz`), the 2-means suggestion
/// under `--auto-cutoff`, and the MAD-outlier drops unless `--no-cell-qc`.
/// Writes the per-cell verdicts and the kept barcodes beside the outputs.
fn decide_cells(
    m: &MatrixFile,
    args: &QcArgs,
    out_dir: &str,
) -> anyhow::Result<FxHashSet<Box<str>>> {
    let data = open_matrix(&m.path)?;
    let names = data.column_names()?;
    let mut vec = SparseIoVec::new();
    vec.push(Arc::from(data), None)?;
    let floor = if args.no_cell_qc {
        1
    } else {
        args.qc_min_cell_nnz
    };
    let cfg = QcConfig {
        n_mads: args.qc_mads,
        min_cell_nnz: args.column_nnz_cutoff.max(floor),
        mad_on_n_genes: !args.no_cell_qc,
        mad_on_counts: !args.no_cell_qc,
        auto_cell_cutoff: args.auto_cutoff,
        qc_histogram: args.show_histogram,
        ..QcConfig::default()
    };
    let report = compute_qc(&vec, &cfg, args.block_size)?;
    write_qc_report(
        &format!("{}/{}_cell_qc_report.tsv", out_dir, m.batch),
        &names,
        &report,
    )?;

    let idx = report.emit_idx_unmasked();
    anyhow::ensure!(
        !idx.is_empty(),
        "{}: no cell survives the cell QC; relax --column-nnz-cutoff / --qc-min-cell-nnz or pass --no-cell-qc",
        m.batch
    );
    let lines: Vec<Box<str>> = idx.iter().map(|&c| names[c].clone()).collect();
    write_lines(&lines, &format!("{}/{}_cells.tsv.gz", out_dir, m.batch))?;
    info!(
        "{}: kept {} of {} cells (near-empty floor {}, MAD QC {})",
        m.batch,
        lines.len(),
        names.len(),
        cfg.min_cell_nnz,
        if args.no_cell_qc { "off" } else { "on" }
    );
    Ok(lines.into_iter().collect())
}

/// `data-beans squeeze`'s rule for a feature-axis nnz cutoff: an explicit
/// non-zero value wins, else the 2-means suggestion under `--auto-cutoff`,
/// else nothing. Empty rows always drop.
fn row_cutoff(args: &QcArgs, nnz: &[usize], label: &str) -> usize {
    let nnz_f: Vec<f32> = nnz.iter().map(|&x| x as f32).collect();
    let suggested = (args.auto_cutoff || args.show_histogram)
        .then(|| suggest_nnz_cutoff(&nnz_f))
        .flatten();
    let cutoff = if args.row_nnz_cutoff > 0 {
        args.row_nnz_cutoff
    } else if args.auto_cutoff {
        suggested.unwrap_or(0)
    } else {
        0
    };
    if args.show_histogram {
        print_nnz_summary(label, "nnz", &nnz_f, cutoff, suggested);
    } else if args.auto_cutoff {
        info!("{label}: row nnz cutoff {cutoff} (2-means suggestion {suggested:?})");
    }
    cutoff.max(1)
}

fn out_spec<'a>(m: &'a MatrixFile, out_dir: &'a str, stem: &'a str, no_zip: bool) -> OutSpec<'a> {
    OutSpec {
        out_dir,
        stem,
        backend: &m.backend,
        zip: m.zipped && !no_zip,
    }
}

fn record(
    summary: &mut Vec<SummaryRow>,
    stem: &str,
    before: (usize, usize, usize),
    w: Option<Written>,
) {
    match w {
        Some(w) => {
            info!(
                "{stem}: {}x{} ({} nnz) -> {}x{} ({} nnz)",
                before.0, before.1, before.2, w.nrow, w.ncol, w.nnz
            );
            summary.push(SummaryRow {
                file: file_name(&w.target),
                before,
                after: (w.nrow, w.ncol, w.nnz),
            });
        }
        None => {
            log::warn!("{stem}: nothing survives the cut; not written");
            summary.push(SummaryRow {
                file: format!("{stem} (not written)").into(),
                before,
                after: (0, 0, 0),
            });
        }
    }
}

/// A matrix opened once with its axis names and shape.
struct Opened {
    data: Backend,
    row_names: Vec<Box<str>>,
    col_names: Vec<Box<str>>,
    before: (usize, usize, usize),
}

fn open_with_names(m: &MatrixFile) -> anyhow::Result<Opened> {
    let data = open_matrix(&m.path)?;
    let row_names = data.row_names()?;
    let col_names = data.column_names()?;
    let before = shape(data.as_ref());
    Ok(Opened {
        data,
        row_names,
        col_names,
        before,
    })
}

/// A `_site` matrix opened in step 2 and written in steps 3-4.
struct SiteMatrix {
    m: MatrixFile,
    opened: Opened,
    cols: Vec<usize>,
}

pub fn run_qc(args: &QcArgs) -> anyhow::Result<()> {
    let out_dir = args.output.as_ref();
    if std::path::Path::new(out_dir).exists() && std::fs::read_dir(out_dir)?.next().is_some() {
        anyhow::bail!("output directory {out_dir} already contains files; choose an empty one");
    }
    std::fs::create_dir_all(out_dir)?;

    let layout: InputLayout = scan_input_dir(&args.input_dir)?;
    let batches = layout.batches();
    info!(
        "{}: {} matrices over {} batches, {} site tables, {} other files",
        args.input_dir,
        layout.matrices.len(),
        batches.len(),
        layout.site_tables.len(),
        layout.other_files.len()
    );
    let mut summary: Vec<SummaryRow> = Vec::new();

    // 1. cells, per batch, on `_count`.
    let mut cells: FxHashMap<Box<str>, FxHashSet<Box<str>>> = FxHashMap::default();
    for b in &batches {
        match layout.matrix(b, "count") {
            Some(m) => {
                cells.insert(b.clone(), decide_cells(m, args, out_dir)?);
            }
            None => log::warn!(
                "{b}: no `_count` matrix; keeping every non-empty cell of its other matrices"
            ),
        }
    }

    // 2. sites: parquet columns first, then cells per site from the `_site`
    //    matrices (pooled over batches).
    let mut tables: FxHashMap<Box<str>, SiteTable> = FxHashMap::default();
    for (modality, path) in &layout.site_tables {
        tables.insert(modality.clone(), read_site_table(path, modality)?);
    }
    let mut site_matrices: Vec<SiteMatrix> = Vec::new();
    let mut site_cells: FxHashMap<Box<str>, FxHashMap<Box<str>, usize>> = FxHashMap::default();
    for m in &layout.matrices {
        let Some(modality) = m.site_modality() else {
            continue;
        };
        let Some(t) = tables.get(modality) else {
            log::warn!(
                "{}: no {modality}_sites.parquet; the matrix is not written",
                m.stem()
            );
            continue;
        };
        let opened = open_with_names(m)?;
        let cols = select_columns(opened.data.as_ref(), &opened.col_names, cells.get(&m.batch));
        let row_nnz = row_nnz_over_columns(opened.data.as_ref(), &cols)?;
        accumulate_site_cells(
            &opened.row_names,
            &row_nnz,
            t.converted_channel(),
            site_cells.entry(modality.into()).or_default(),
        );
        site_matrices.push(SiteMatrix {
            m: m.clone(),
            opened,
            cols,
        });
    }
    let mut kept_sites: FxHashMap<Box<str>, FxHashSet<Box<str>>> = FxHashMap::default();
    for modality in SITE_MODALITIES {
        let Some(t) = tables.get(*modality) else {
            continue;
        };
        let n_cells = site_cells.get(*modality).map(|acc| t.cells_per_site(acc));
        let reasons = args.site.reasons(t, n_cells.as_deref());
        let kept: FxHashSet<Box<str>> = t
            .key
            .iter()
            .zip(&reasons)
            .filter(|(_, r)| r.is_none())
            .map(|(k, _)| k.clone())
            .collect();
        let (n_kept, n_dropped) = write_site_tables(
            t,
            &reasons,
            &format!("{out_dir}/{modality}_sites.parquet"),
            &format!("{out_dir}/{modality}_sites_dropped.parquet"),
        )?;
        let mut by_reason: FxHashMap<&str, usize> = FxHashMap::default();
        for r in reasons.iter().flatten() {
            *by_reason.entry(r.label()).or_insert(0) += 1;
        }
        let mut by_reason: Vec<_> = by_reason.into_iter().collect();
        by_reason.sort();
        info!(
            "{modality}: kept {n_kept} sites, dropped {n_dropped} {by_reason:?}; cells per site {}",
            if n_cells.is_some() {
                "from the _site matrices"
            } else {
                "unavailable (no _site matrix), cells check skipped"
            }
        );
        kept_sites.insert((*modality).into(), kept);
    }

    // 3 + 4. site matrices, then the re-pooled gene-level matrices.
    let mut repooled: FxHashSet<(Box<str>, Box<str>)> = FxHashSet::default();
    for sm in &site_matrices {
        let modality = sm.m.site_modality().unwrap_or_default();
        let rows = site_matrix_rows(&sm.opened.row_names, kept_sites.get(modality));
        let stem = sm.m.stem();
        let w = write_subset(
            sm.opened.data.as_ref(),
            &sm.cols,
            &rows,
            &sm.opened.row_names,
            &sm.opened.col_names,
            &out_spec(&sm.m, out_dir, &stem, args.no_zip),
        )?;
        record(&mut summary, &stem, sm.opened.before, w);

        let gene_stem = format!("{}_{}", sm.m.batch, modality);
        let w = repool_gene_level(
            sm.opened.data.as_ref(),
            &sm.cols,
            &rows,
            &sm.opened.row_names,
            &sm.opened.col_names,
            args.row_nnz_cutoff,
            &out_spec(&sm.m, out_dir, &gene_stem, args.no_zip),
        )?;
        let before = layout
            .matrix(&sm.m.batch, modality)
            .and_then(|gm| open_matrix(&gm.path).ok())
            .map(|d| shape(d.as_ref()))
            .unwrap_or((0, 0, 0));
        record(&mut summary, &gene_stem, before, w);
        repooled.insert((sm.m.batch.clone(), modality.into()));
    }

    // 5. everything else: cells + row nnz.
    for m in &layout.matrices {
        if m.site_modality().is_some() || repooled.contains(&(m.batch.clone(), m.kind.clone())) {
            continue;
        }
        let opened = open_with_names(m)?;
        let cols = select_columns(opened.data.as_ref(), &opened.col_names, cells.get(&m.batch));
        let row_nnz = row_nnz_over_columns(opened.data.as_ref(), &cols)?;
        let stem = m.stem();
        let cutoff = row_cutoff(args, &row_nnz, &format!("{stem} rows"));
        let rows: Vec<usize> = (0..row_nnz.len())
            .filter(|&r| row_nnz[r] >= cutoff)
            .collect();
        let w = write_subset(
            opened.data.as_ref(),
            &cols,
            &rows,
            &opened.row_names,
            &opened.col_names,
            &out_spec(m, out_dir, &stem, args.no_zip),
        )?;
        record(&mut summary, &stem, opened.before, w);
    }

    // 6. copy-through, except the per-batch cell lists `qc` rewrote.
    for f in &layout.other_files {
        let name = file_name(f);
        if name
            .strip_suffix("_cells.tsv.gz")
            .is_some_and(|b| cells.contains_key(b))
        {
            continue;
        }
        std::fs::copy(f.as_ref(), format!("{out_dir}/{name}"))?;
    }

    let mut lines: Vec<Box<str>> = vec![
        "#file\trows_before\tcols_before\tnnz_before\trows_after\tcols_after\tnnz_after".into(),
    ];
    for s in &summary {
        lines.push(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                s.file, s.before.0, s.before.1, s.before.2, s.after.0, s.after.1, s.after.2
            )
            .into(),
        );
    }
    write_lines(&lines, &format!("{out_dir}/qc_summary.tsv"))?;
    info!("done: {out_dir}");
    Ok(())
}
