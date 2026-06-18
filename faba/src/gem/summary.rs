//! `faba gem-summary` — group any count matrix by cell type and report
//! per-feature statistics, the tidy "gene × cell-type, per modality" summary.
//!
//! Decoupled from `faba gem-annotate`: annotation is run once (signatures,
//! permutation null, clustering); summarizing is a cheap, re-runnable step you
//! repeat across measures (m6a_ratio, m6a_mixture, converted, genes,
//! atoi_ratio, …) against the *same* labels. Labels come from a membership
//! source — annotate's `*.annot.parquet` or any 2-column `cell<TAB>label` TSV
//! (so labels from any tool work, no gem manifest required).
//!
//! Each `--matrix` group is grouped by cell type and reduced to per-(feature,
//! group) `nnz/tot/mu/sig` via the exact same path as `data-beans stat -s row
//! -g` (reused, not reimplemented). One `{out}.summary_{label}.parquet` per
//! group, long format `(name, group, nnz, tot, mu, sig)` where
//! `name = {gene}/{modality}/{detail}` and `group` is the cell-type label.
//! `mu = tot / cells-in-group` (incl. uncovered zeros); `tot/nnz` is the mean
//! over covered cells — both recoverable from one output, serving ratios and
//! counts alike.

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use log::info;
use std::fs::File;
use std::sync::Arc;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;

use data_beans::hdf5_io::resolve_backend_file;
use data_beans::qc::collect_stratified_row_stat_across_vec;
use data_beans::sparse_io::{open_sparse_matrix, COLUMN_SEP};
use data_beans::sparse_io_vector::{ColumnAlignment, SparseIoVec};

use matrix_util::common_io::{basename, mkdir_parent};
use matrix_util::membership::Membership;
use matrix_util::parquet::peek_parquet_field_names;
use matrix_util::sparse_stat::save_grouped_stats_parquet;

use super::sample_id::{file_sample_id, longest_common_underscore_suffix};

#[derive(Args, Debug)]
pub struct GemSummaryArgs {
    #[arg(
        long,
        short = 'l',
        help = "Cell→cell-type labels: a gem-annotate `*.annot.parquet` or a \
                2-column `cell<TAB>label` TSV/CSV"
    )]
    pub membership: Box<str>,

    #[arg(
        long = "matrix",
        required = true,
        help = "Matrix to summarize per cell type; repeat for each measure",
        long_help = "Matrix to summarize, grouped by cell type. Repeat the flag \
                     for each measure (one `summary_{label}.parquet` per flag). \
                     Comma-separate files within one flag to STACK replicates \
                     into a single summary. Optional `label=` prefix sets the \
                     output label (default: first file's basename). E.g.\n  \
                     --matrix rep1_m6a_ratio.zarr.zip,rep2_m6a_ratio.zarr.zip\n  \
                     --matrix m6a_mix=rep1_m6a_mixture.zarr.zip"
    )]
    pub matrix: Vec<Box<str>>,

    #[arg(
        long = "label-column",
        default_value = "coarse_label",
        help = "Column to read when --membership is a parquet (e.g. coarse_label, fine_label)"
    )]
    pub label_column: Box<str>,

    #[arg(
        long,
        short = 'd',
        default_value_t = '@',
        help = "Delimiter to extract the base barcode for matching (handles `@batch` suffixes)"
    )]
    pub delimiter: char,

    #[arg(
        long = "sample-strip",
        default_value = "",
        help = "Per-file suffix to strip when deriving the @sample tag (e.g. `_m6a_ratio`)",
        long_help = "When the membership is `@sample`-tagged (a multi-sample `faba gem` run), \
                     each matrix file's barcodes are tagged `{barcode}@{sample_id}` to match — \
                     following the same convention as `faba gem`. The sample id is the file's \
                     basename with this suffix removed; with ≥2 files per --matrix flag it is \
                     auto-derived as their longest common `_`-suffix, so an explicit value is \
                     only needed for single-file flags (e.g. `--sample-strip _m6a_ratio` turns \
                     `rep1_wt_m6a_ratio` into `rep1_wt`)."
    )]
    pub sample_strip: Box<str>,

    #[arg(
        long = "no-sample-tag",
        help = "Disable @sample barcode tagging even when the membership is tagged (match bare barcodes)"
    )]
    pub no_sample_tag: bool,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (default: alongside the membership file)"
    )]
    pub out: Option<Box<str>>,
}

pub fn run_gem_summary(args: &GemSummaryArgs) -> Result<()> {
    let out = args
        .out
        .as_deref()
        .map(str::to_owned)
        .unwrap_or_else(|| default_out(&args.membership));
    mkdir_parent(&out)?;

    let raw = load_membership(&args.membership, &args.label_column)?;
    info!(
        "{} membership entries, {} groups: {:?}",
        raw.len(),
        raw.unique_groups().len(),
        raw.unique_groups()
    );

    // If the membership is `@sample`-tagged (multi-sample `faba gem` run), tag
    // each matrix file's barcodes the same way so columns match exactly — see
    // `super::sample_id`. Bare (single-sample) memberships skip tagging.
    let membership_tagged = raw.iter().any(|(k, _)| k.contains(COLUMN_SEP));
    let tag = membership_tagged && !args.no_sample_tag;
    info!(
        "membership {} @sample-tagged → barcode tagging {}",
        if membership_tagged { "is" } else { "is not" },
        if tag { "ON" } else { "OFF" }
    );

    // Tagged columns must equal tagged keys, so match exact-only (no base-key /
    // prefix fallback — that would re-introduce cross-sample ambiguity for
    // QC-dropped matrix cells, and the prefix scan is O(n²)). Untagged keeps the
    // base-key (`--delimiter`) + prefix matching `data-beans stat` uses.
    let membership = if tag {
        Membership::from_pairs(raw.iter().map(|(k, v)| (k.clone(), v.clone())), false)
    } else {
        raw.with_delimiter(args.delimiter)
    };

    for spec in &args.matrix {
        let (label, files) = parse_group(spec)?;
        summarize_group(&label, &files, &membership, &out, tag, &args.sample_strip)
            .with_context(|| format!("summarizing matrix group '{label}'"))?;
    }
    Ok(())
}

/// Group one (possibly stacked) matrix by cell type and write
/// `{out}.summary_{label}.parquet`. Mirrors the `data-beans stat -s row -g`
/// grouped path; when `tag` is set, barcodes are `@sample`-tagged (the
/// `faba gem` convention) before matching so multi-sample replicates match
/// the tagged membership exactly instead of by ambiguous base-key.
fn summarize_group(
    label: &str,
    files: &[Box<str>],
    membership: &Membership,
    out: &str,
    tag: bool,
    sample_strip: &str,
) -> Result<()> {
    let data = if tag {
        // Per-file `@sample` tag under Union column alignment. Strip: explicit
        // override, else the group's longest common `_`-suffix (≥2 files).
        let basenames: Vec<Box<str>> = files.iter().map(|f| basename(f)).collect::<Result<_>>()?;
        let strip: Box<str> = if sample_strip.is_empty() {
            longest_common_underscore_suffix(&basenames)
        } else {
            sample_strip.into()
        };
        let mut data = SparseIoVec::new().with_column_alignment(ColumnAlignment::Union)?;
        let mut sample_ids: Vec<Box<str>> = Vec::with_capacity(files.len());
        for f in files {
            let (backend, file) = resolve_backend_file(f, None)?;
            let this_data = open_sparse_matrix(&file, &backend)?;
            let sid = file_sample_id(f, &strip)?;
            data.push_with_barcode_suffix(Arc::from(this_data), None, Some(&sid))?;
            sample_ids.push(sid);
        }
        info!("[{label}] @sample tags (strip {strip:?}): {sample_ids:?}");
        data
    } else {
        // Plain stack (mirrors `data-beans stat`): attach the file basename to
        // disambiguate columns when more than one file is given.
        let attach_data_name = files.len() > 1;
        let mut data = SparseIoVec::new();
        for f in files {
            let (backend, file) = resolve_backend_file(f, None)?;
            let this_data = open_sparse_matrix(&file, &backend)?;
            let data_name = attach_data_name.then(|| basename(&file)).transpose()?;
            data.push(Arc::from(this_data), data_name)?;
        }
        data
    };

    let cols = data.column_names()?;
    let (column_membership, stats) = membership.match_keys(&cols);
    info!(
        "[{label}] column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
        stats.exact,
        stats.base_key,
        stats.prefix,
        stats.total_matched(),
        stats.total()
    );
    if column_membership.is_empty() {
        bail!(
            "[{label}] no matrix columns matched the membership file — \
             check barcodes / --delimiter (data sample: {:?})",
            cols.iter().take(3).collect::<Vec<_>>()
        );
    }

    let (group_names, group_stats) =
        collect_stratified_row_stat_across_vec(&data, &column_membership, None)?;

    let out_path = format!("{out}.summary_{label}.parquet");
    save_grouped_stats_parquet(&out_path, &data.row_names()?, &group_names, &group_stats)?;
    info!("[{label}] wrote {out_path} ({} groups)", group_names.len());
    Ok(())
}

/// Build a [`Membership`] from either annotate's `*.annot.parquet` (read the
/// `cell` + `label_column` string columns) or a 2-column `cell<TAB>label`
/// TSV/CSV (`Membership::from_file`, the same reader `data-beans stat` uses).
fn load_membership(path: &str, label_column: &str) -> Result<Membership> {
    if path.ends_with(".parquet") {
        let fields = peek_parquet_field_names(path)?;
        let col_idx = |name: &str| {
            fields
                .iter()
                .position(|f| f.as_ref() == name)
                .ok_or_else(|| anyhow!("missing '{name}' column in {path}"))
        };
        let cell_idx = col_idx("cell")?;
        let label_idx = col_idx(label_column)?;

        let file = File::open(path).with_context(|| format!("opening {path}"))?;
        let reader = SerializedFileReader::new(file)?;
        let mut pairs: Vec<(Box<str>, Box<str>)> = Vec::new();
        for record in reader.get_row_iter(None)? {
            let row = record?;
            let cell: Box<str> = row.get_string(cell_idx)?.clone().into_boxed_str();
            let label: Box<str> = row.get_string(label_idx)?.clone().into_boxed_str();
            pairs.push((cell, label));
        }
        info!("loaded {} membership entries from {path}", pairs.len());
        Ok(Membership::from_pairs(pairs, true))
    } else {
        Membership::from_file(path, 0, 1, true)
    }
}

/// Parse one `--matrix` spec into `(label, files)`. A leading `label=` (where
/// the left side looks like a bare label — no `/` or `.`) sets the output
/// label; otherwise the label is the first file's basename. Files are
/// comma-separated and stacked.
fn parse_group(spec: &str) -> Result<(Box<str>, Vec<Box<str>>)> {
    let (label, files_str) = match spec.split_once('=') {
        Some((lhs, rhs)) if !lhs.is_empty() && !lhs.contains(['/', '.']) => {
            (Some(lhs.to_string().into_boxed_str()), rhs)
        }
        _ => (None, spec),
    };
    let files: Vec<Box<str>> = files_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string().into_boxed_str())
        .collect();
    if files.is_empty() {
        bail!("no files in --matrix '{spec}'");
    }
    let label = match label {
        Some(l) => l,
        None => basename(&files[0])?,
    };
    Ok((label, files))
}

/// Default output prefix when `--out` is omitted: the membership path with a
/// recognized label-file suffix trimmed, so
/// `…/run.gem_annot.annot.parquet` → `…/run.gem_annot`.
fn default_out(membership: &str) -> String {
    for sfx in [
        ".annot.parquet",
        ".membership.tsv",
        ".tsv.gz",
        ".csv.gz",
        ".parquet",
        ".tsv",
        ".csv",
    ] {
        if let Some(s) = membership.strip_suffix(sfx) {
            return s.to_string();
        }
    }
    membership.to_string()
}
