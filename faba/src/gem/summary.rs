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
//! Data files are passed positionally (like `data-beans stat`); multiple files
//! stack into one matrix. The cells are grouped by cell type and reduced to
//! per-(feature, group) `nnz/tot/mu/sig` via the exact same path as `data-beans
//! stat -s row -g` (reused, not reimplemented).
//!
//! What it adds over `data-beans stat`: it is **modality-aware**. Each
//! `{gene}/{modality}/{detail}` row name is split into separate `gene`,
//! `modality`, and `component` columns (rather than one opaque `name`), so the
//! output is the tidy "gene × cell-type, per modality" table you can pivot /
//! filter by modality with no string surgery — and join across modalities later.
//! It also matches multi-sample `@sample`-tagged cells, which `data-beans stat`
//! cannot disambiguate. One `{out}.summary.parquet`, long format
//! `(gene, modality, component, group, nnz, tot, mu, sig)` where `group` is the
//! cell-type label. `mu = tot / cells-in-group` (incl. uncovered zeros);
//! `tot/nnz` is the mean over covered cells — both recoverable from one output,
//! serving ratios and counts alike. Rows that aren't `{gene}/{modality}/{detail}`
//! (e.g. a bare gene-count matrix) fall back to `gene = name`, empty modality.

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
use matrix_util::sparse_stat::save_grouped_stats_parquet_cols;

use rustc_hash::FxHashSet;

use super::feature_table::parse_feature_name;
use super::sample_id::{longest_underscore_prefix_in, strip_sample_id};

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
        required = true,
        value_delimiter = ',',
        help = "Data backend file(s) to summarize per cell type (.zarr/.h5), positional like `data-beans stat`",
        long_help = "Sparse count/ratio data backend file(s) to group by cell type — \
                     a features × cells matrix whose rows are `{gene}/{modality}/{detail}` \
                     (the per-cell measures emitted by `faba genes` / `dartseq` / `atoi` / \
                     `apa`, e.g. `*_m6a_ratio`, `*_m6a_mixture`, `*_genes`), NOT the gem \
                     embedding parquets. Passed positionally, exactly like `data-beans \
                     stat`. Multiple files (space- or comma-separated) STACK into one \
                     matrix (e.g. replicates), producing a single \
                     `{out}.summary.parquet`. To summarize several distinct measures, run \
                     once per measure (with a different `-o`). E.g.\n  \
                     faba gem-summary -l run.gem_annot.annot.parquet rep1_m6a_ratio.zarr.zip\n  \
                     faba gem-summary -l run.gem_annot.annot.parquet rep1.zarr.zip rep2.zarr.zip"
    )]
    pub data_files: Vec<Box<str>>,

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
        help = "Override @sample inference by stripping this suffix from each file basename",
        long_help = "When the membership is `@sample`-tagged (a multi-sample `faba gem` run), \
                     each matrix file's barcodes are tagged `{barcode}@{sample_id}` to match — \
                     the `faba gem` convention. By DEFAULT the sample id is inferred as the \
                     membership sample id that is the longest `_`-aligned PREFIX of the file's \
                     basename (`rep1_mut_apa_mixture` → `rep1_mut`), so no flag is normally \
                     needed. Set this only to override that, deriving the id by suffix removal \
                     instead (e.g. `--sample-strip _m6a_ratio` turns `rep1_wt_m6a_ratio` into \
                     `rep1_wt`)."
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
    // The membership's distinct `@sample` ids (the suffix after `@`). Each matrix
    // file's `@sample` tag is inferred as the longest of these that is a
    // `_`-aligned prefix of its basename — exactly how `faba gem` matches a
    // satellite file to a genes sample (`infer_satellite_strip`). Robust to the
    // file's measure suffix (`_apa_mixture`, `_genes`, …), unlike a common-suffix
    // strip across heterogeneous files. (Computed before `raw` is consumed below.)
    let membership_sample_ids: FxHashSet<Box<str>> = if tag {
        // Dedup the `@sample` suffixes as borrowed `&str` first, then own only the
        // handful of distinct ids — not one allocation per (potentially 100k+)
        // membership cell.
        raw.iter()
            .filter_map(|(k, _)| k.rsplit_once(COLUMN_SEP).map(|(_, sid)| sid))
            .collect::<FxHashSet<&str>>()
            .into_iter()
            .map(Box::from)
            .collect()
    } else {
        FxHashSet::default()
    };

    let membership = if tag {
        Membership::from_pairs(raw.iter().map(|(k, v)| (k.clone(), v.clone())), false)
    } else {
        raw.with_delimiter(args.delimiter)
    };

    // All data files stack into ONE matrix (replicates), like `data-beans stat`,
    // and write a single `{out}.summary.parquet`. Run once per measure (with a
    // different `-o`) to summarize several distinct measures.
    summarize_group(
        &args.data_files,
        &membership,
        &membership_sample_ids,
        &out,
        tag,
        &args.sample_strip,
    )
    .context("summarizing matrix")?;
    Ok(())
}

/// Group one (possibly stacked) matrix by cell type and write
/// `{out}.summary.parquet`. Mirrors the `data-beans stat -s row -g` grouped
/// path; when `tag` is set, barcodes are `@sample`-tagged (the `faba gem`
/// convention) before matching so multi-sample replicates match the tagged
/// membership exactly instead of by ambiguous base-key.
fn summarize_group(
    files: &[Box<str>],
    membership: &Membership,
    membership_sample_ids: &FxHashSet<Box<str>>,
    out: &str,
    tag: bool,
    sample_strip: &str,
) -> Result<()> {
    let data = if tag {
        // Per-file `@sample` tag under Union column alignment. The tag is the
        // membership sample id that is the longest `_`-aligned PREFIX of the
        // file's basename (`rep1_mut_apa_mixture` → `rep1_mut`) — the same
        // basename→sample match `faba gem` uses for satellites. An explicit
        // `--sample-strip` overrides via suffix removal.
        let mut data = SparseIoVec::new().with_column_alignment(ColumnAlignment::Union)?;
        let mut sample_ids: Vec<Box<str>> = Vec::with_capacity(files.len());
        for f in files {
            let bn = basename(f)?;
            let sid: Box<str> = if !sample_strip.is_empty() {
                strip_sample_id(&bn, sample_strip)
            } else {
                longest_underscore_prefix_in(&bn, membership_sample_ids).ok_or_else(|| {
                    let mut ids: Vec<&str> =
                        membership_sample_ids.iter().map(|s| s.as_ref()).collect();
                    ids.sort_unstable();
                    anyhow!(
                        "can't infer the @sample tag for '{bn}': no membership sample id \
                         {ids:?} is a `_`-aligned prefix of it. Pass --sample-strip, or check that \
                         the file belongs to a sample present in the membership."
                    )
                })?
            };
            let (backend, file) = resolve_backend_file(f, None)?;
            let this_data = open_sparse_matrix(&file, &backend)?;
            data.push_with_barcode_suffix(Arc::from(this_data), None, Some(&sid))?;
            sample_ids.push(sid);
        }
        info!("@sample tags: {sample_ids:?}");
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
        "column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
        stats.exact,
        stats.base_key,
        stats.prefix,
        stats.total_matched(),
        stats.total()
    );
    if column_membership.is_empty() {
        bail!(
            "no matrix columns matched the membership file — \
             check barcodes / --delimiter (data sample: {:?})",
            cols.iter().take(3).collect::<Vec<_>>()
        );
    }

    let (group_names, group_stats) =
        collect_stratified_row_stat_across_vec(&data, &column_membership, None)?;

    // Modality-aware output: split each `{gene}/{modality}/{detail}` row name into
    // gene / modality / component columns (this is what makes gem-summary differ
    // from `data-beans stat`, which keeps the row name opaque). Rows that don't
    // follow the convention (e.g. a plain gene-count matrix with bare gene names)
    // fall back to gene = the whole name, modality/component empty.
    let row_names = data.row_names()?;
    let (mut genes, mut modalities, mut components) = (
        Vec::with_capacity(row_names.len()),
        Vec::with_capacity(row_names.len()),
        Vec::with_capacity(row_names.len()),
    );
    for name in &row_names {
        match parse_feature_name(name) {
            Some(k) => {
                genes.push(k.gene);
                modalities.push(k.modality);
                components.push(k.detail);
            }
            None => {
                genes.push(name.clone());
                modalities.push("".into());
                components.push("".into());
            }
        }
    }

    let out_path = format!("{out}.summary.parquet");
    save_grouped_stats_parquet_cols(
        &out_path,
        &[
            ("gene", &genes),
            ("modality", &modalities),
            ("component", &components),
        ],
        &group_names,
        &group_stats,
    )?;
    info!("wrote {out_path} ({} groups)", group_names.len());
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
