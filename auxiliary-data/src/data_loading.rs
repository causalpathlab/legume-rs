use std::sync::Arc;

use log::info;

use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::common_io::{self, basename, read_lines};

use crate::feature_names::FeatureNameKind;
use data_beans::sparse_io_vector::RowAlignment;

/// Arguments for loading multiple sparse data files with shared row names.
#[derive(Default)]
pub struct ReadSharedRowsArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub preload: bool,
    /// Cross-file row-name canonicalization rule. `None` = auto-detect
    /// via [`FeatureNameKind::auto_detect`] once row names are in hand.
    /// `Some(kind)` skips detection and uses the caller's choice.
    /// Default = `None` (auto).
    pub feature_kind: Option<FeatureNameKind>,
    /// How to align row names across input files. Default
    /// [`RowAlignment::Intersect`] preserves single-modality semantics
    /// (keep only common rows). Switch to [`RowAlignment::Union`] for
    /// multi-modal load (e.g. paired RNA + ATAC) where features are
    /// disjoint and the modalities are glued via cells, not features.
    pub row_alignment: RowAlignment,
}

/// Sparse data with per-cell batch labels.
pub struct SparseDataWithBatch {
    pub data: SparseIoVec,
    pub batch: Vec<Box<str>>,
}

/// Load multiple sparse data files, verify shared row names, and auto-detect batch.
///
/// Batch assignment priority:
/// 1. Explicit batch files (one label per cell per file)
/// 2. Embedded `@`-separated batch info in column names (e.g., `barcode@donor`)
/// 3. File name as batch label (one batch per input file)
pub fn read_data_on_shared_rows(args: ReadSharedRowsArgs) -> anyhow::Result<SparseDataWithBatch> {
    // to avoid duplicate barcodes in the column names
    let attach_data_name = args.data_files.len() > 1;

    // Open every backend first (preserving order). For LocusOverlap we
    // need to peek all row names before installing the canonicalizer.
    type OpenedBackend = Box<dyn data_beans::sparse_io::SparseIo<IndexIter = Vec<usize>>>;
    let mut opened: Vec<(Box<str>, OpenedBackend)> = Vec::with_capacity(args.data_files.len());
    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);
        let mut data = try_open_or_convert(data_file)?;
        if args.preload {
            data.preload_columns()?;
        }
        opened.push((data_file.clone(), data));
    }

    let mut data_vec = SparseIoVec::new()
        .with_row_alignment(args.row_alignment)
        .expect("with_row_alignment on empty SparseIoVec");

    use crate::feature_names::FeatureNameKind;

    // Peek row names once if auto-detect needs them, or if the
    // caller-specified kind needs a global cross-name pass.
    let needs_names = args.feature_kind.is_none()
        || args
            .feature_kind
            .as_ref()
            .is_some_and(|k| k.needs_global_pass());
    let all_names: Option<Vec<Box<str>>> = if needs_names {
        let mut acc: Vec<Box<str>> = Vec::new();
        for (_, d) in opened.iter() {
            acc.extend(d.row_names()?);
        }
        Some(acc)
    } else {
        None
    };

    let resolved_kind: FeatureNameKind = match args.feature_kind.clone() {
        Some(k) => k,
        None => {
            let names = all_names.as_ref().expect("peeked when auto");
            let k = FeatureNameKind::auto_detect(names);
            info!(
                "Row alignment: auto-detected feature name kind → {:?} ({} rows)",
                k,
                names.len()
            );
            k
        }
    };

    // Install canonicalizer. Three paths:
    //   * Mixed                            → per-name dispatcher (needs names).
    //   * Locus { merge_overlapping: true } → overlap cluster map (needs names).
    //   * Anything else                    → pure per-name closure from `canonicalize`.
    match &resolved_kind {
        FeatureNameKind::Mixed => {
            let names = all_names.as_ref().expect("peeked for Mixed").clone();
            info!(
                "Row alignment: building MIXED-kind canonical map over {} names \
                 across {} file(s)",
                names.len(),
                opened.len()
            );
            let canon = crate::feature_names::build_mixed_kind_canonicalizer(&names);
            data_vec = data_vec
                .with_row_canonicalizer(move |name| canon(name))
                .expect("with_row_canonicalizer on empty SparseIoVec");
        }
        FeatureNameKind::Locus {
            merge_overlapping: true,
        } => {
            let names = all_names
                .as_ref()
                .expect("peeked for Locus merge_overlapping")
                .clone();
            info!(
                "Row alignment: building locus-overlap canonical map over {} names \
                 across {} file(s)",
                names.len(),
                opened.len()
            );
            let canon = crate::feature_names::build_locus_overlap_canonicalizer(&names);
            data_vec = data_vec
                .with_row_canonicalizer(move |name| canon(name))
                .expect("with_row_canonicalizer on empty SparseIoVec");
        }
        kind => {
            if let Some(canon) = kind.clone().into_canonicalizer() {
                info!(
                    "Row alignment: applying {:?} canonicalizer across {} file(s)",
                    kind,
                    opened.len()
                );
                // SAFETY: data_vec is empty; with_row_canonicalizer only errors
                // if backends were already pushed.
                data_vec = data_vec
                    .with_row_canonicalizer(move |name| canon(name))
                    .expect("with_row_canonicalizer on empty SparseIoVec");
            }
        }
    }
    if args.row_alignment == RowAlignment::Union {
        info!(
            "Row alignment: UNION across {} file(s) — features from each \
             backend retained, non-observing backends pad with zero",
            opened.len()
        );
    }
    for (data_file, data) in opened.into_iter() {
        let data_name = attach_data_name.then(|| basename(&data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    // SparseIoVec already aligns rows to the intersection of row names
    // across all backends; warn if any backend introduced new rows that
    // had to be dropped.
    let intersection_size = data_vec.num_rows();
    for j in 0..data_vec.len() {
        let backend_rows = data_vec[j].num_rows().unwrap_or(0);
        if backend_rows != intersection_size {
            info!(
                "Backend {} has {} rows; using {} shared rows for fitting",
                j, backend_rows, intersection_size
            );
        }
    }

    // check batch membership
    let mut batch_membership = Vec::with_capacity(data_vec.len());

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else {
        let column_counts = data_vec.num_columns_by_data()?;
        let column_names = data_vec.column_names()?;
        let mut col_start = 0usize;

        for (file_idx, &ncols) in column_counts.iter().enumerate() {
            let data_file = args.data_files[file_idx].clone();
            let (_dir, file_base, _ext) = common_io::dir_base_ext(&data_file)?;

            let col_end = col_start + ncols;
            let file_columns = &column_names[col_start..col_end];

            let appended_suffix =
                attach_data_name.then(|| format!("@{}", file_base).into_boxed_str());
            let (tags, used_embedded) = infer_batch_from_columns(
                file_columns,
                file_base.as_ref(),
                appended_suffix.as_deref(),
            );
            if used_embedded {
                info!(
                    "File {}: using embedded batch from column names (file '{}')",
                    file_idx, file_base
                );
            } else {
                info!(
                    "File {}: using file name '{}' as batch",
                    file_idx, file_base
                );
            }
            batch_membership.extend(tags);
            col_start = col_end;
        }
    }

    if batch_membership.len() != data_vec.num_columns() {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()
        ));
    }

    Ok(SparseDataWithBatch {
        data: data_vec,
        batch: batch_membership,
    })
}

/// Infer per-cell batch labels for one file's column names.
///
/// `appended_suffix` is the `@{file_base}` barcode disambiguator that
/// `SparseIoVec::push` tacks onto every column when multiple files are
/// loaded; it must be stripped *before* searching for a real embedded
/// `@batch` tag, otherwise `rsplit('@')` picks up the basename and every
/// cell in a file collapses to one wrong batch label.
///
/// Returns `(tags, used_embedded)` where `used_embedded=true` means the
/// raw column names already contained an `@batch` tag.
fn infer_batch_from_columns(
    file_columns: &[Box<str>],
    file_base: &str,
    appended_suffix: Option<&str>,
) -> (Vec<Box<str>>, bool) {
    fn raw_of<'a>(name: &'a str, suffix: Option<&str>) -> &'a str {
        match suffix {
            Some(sfx) => name.strip_suffix(sfx).unwrap_or(name),
            None => name,
        }
    }

    let has_embedded_batch = file_columns
        .first()
        .is_some_and(|name| raw_of(name.as_ref(), appended_suffix).contains('@'));

    if has_embedded_batch {
        let tags = file_columns
            .iter()
            .map(|col_name| {
                let raw = raw_of(col_name.as_ref(), appended_suffix);
                let embedded = raw.rsplit('@').next().unwrap_or(raw);
                embedded.to_string().into_boxed_str()
            })
            .collect();
        (tags, true)
    } else {
        let fallback: Box<str> = file_base.to_string().into_boxed_str();
        (vec![fallback; file_columns.len()], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols(v: &[&str]) -> Vec<Box<str>> {
        v.iter()
            .map(|s| (*s).to_string().into_boxed_str())
            .collect()
    }

    #[test]
    fn embedded_donor_survives_push_suffix() {
        // Simulates `SparseIoVec::push` appending `@mix` to each column name
        // when multiple files are loaded. Raw names are `ACGT-1@donorA`,
        // `ACGT-2@donorB`.
        let names = cols(&[
            "ACGT-1@donorA@mix",
            "ACGT-2@donorB@mix",
            "ACGT-3@donorA@mix",
            "ACGT-4@donorB@mix",
        ]);
        let (tags, used_embedded) = infer_batch_from_columns(&names, "mix", Some("@mix"));
        assert!(used_embedded);
        assert_eq!(
            tags.iter().map(|b| b.as_ref()).collect::<Vec<_>>(),
            vec!["donorA", "donorB", "donorA", "donorB"]
        );
    }

    #[test]
    fn no_embedded_batch_falls_back_to_file_base() {
        // Barcodes without any embedded `@`.
        let names = cols(&["AAAA@s1", "CCCC@s1"]);
        let (tags, used_embedded) = infer_batch_from_columns(&names, "s1", Some("@s1"));
        assert!(!used_embedded);
        assert_eq!(
            tags.iter().map(|b| b.as_ref()).collect::<Vec<_>>(),
            vec!["s1", "s1"]
        );
    }

    #[test]
    fn single_file_embedded_batch() {
        // Single file → no `@file_base` suffix was appended.
        let names = cols(&["ACGT-1@donorA", "ACGT-2@donorB"]);
        let (tags, used_embedded) = infer_batch_from_columns(&names, "only", None);
        assert!(used_embedded);
        assert_eq!(
            tags.iter().map(|b| b.as_ref()).collect::<Vec<_>>(),
            vec!["donorA", "donorB"]
        );
    }

    #[test]
    fn single_file_no_embedded_batch() {
        let names = cols(&["AAAA", "CCCC"]);
        let (tags, used_embedded) = infer_batch_from_columns(&names, "only", None);
        assert!(!used_embedded);
        assert_eq!(
            tags.iter().map(|b| b.as_ref()).collect::<Vec<_>>(),
            vec!["only", "only"]
        );
    }

    #[test]
    fn empty_file_columns() {
        let names: Vec<Box<str>> = vec![];
        let (tags, used_embedded) = infer_batch_from_columns(&names, "x", Some("@x"));
        assert!(!used_embedded);
        assert!(tags.is_empty());
    }
}
