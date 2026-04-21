//! Shared parsing and alignment utilities for `gene<TAB>celltype` marker
//! files. Used by the `annotate` subcommand (to build the gene×celltype
//! membership matrix used for anchor margin scoring) and by the
//! training-time anchor-based topic-model prior (to label data-driven
//! anchor pseudobulks against user-provided markers).

use crate::embed_common::Mat;
use matrix_util::common_io::{read_lines_of_words_delim, ReadLinesOut};
use rustc_hash::FxHashSet as HashSet;

/// Flexible gene-name matching — delegates to the shared implementation in
/// `data_beans` so all consumers (annotate, anchor prior, interactive marker
/// augmentation) agree on symbol / alias / case normalization.
pub use data_beans::utilities::name_matching::flexible_name_match as flexible_gene_match;

/// The annotation-matrix form expected by `annotate`: a 0/1 membership
/// matrix aligned to the dictionary gene order, plus the sorted list of
/// celltype names that index the columns.
pub(crate) struct AnnotInfo {
    pub membership_ga: Mat,
    pub annot_names: Vec<Box<str>>,
}

/// Aligned marker table for the anchor-based topic-model prior. The
/// `membership_gc` matrix holds TF-IDF weights `w_g = ln(C / c_g)` for
/// each (gene, celltype) pair; non-markers and genes shared by all
/// celltypes are 0. Celltypes index its columns.
pub(crate) struct MarkerInfo {
    pub celltypes: Vec<Box<str>>,
    pub membership_gc: Mat,
}

/// Parse a gene/celltype TSV or CSV into `(gene, celltype)` pairs. Blank and
/// single-token lines are silently skipped. The delimiter can be tab, comma,
/// or space — whichever splits the first line into at least two tokens.
pub(crate) fn read_marker_gene_info(file_path: &str) -> anyhow::Result<Vec<(Box<str>, Box<str>)>> {
    let ReadLinesOut { lines, header: _ } = read_lines_of_words_delim(file_path, &['\t', ','], -1)?;

    Ok(lines
        .into_iter()
        .filter_map(|words| {
            if words.len() < 2 {
                None
            } else {
                Some((words[0].clone(), words[1].clone()))
            }
        })
        .collect())
}

/// Reweight binary membership in place: `w_g = ln(C / c_g)` where `c_g` is
/// the number of celltypes claiming gene `g`. Genes shared by all celltypes
/// receive weight 0, which removes them from downstream claim scoring.
/// Returns `ln(C)` (the maximum possible weight) for logging.
fn apply_idf_weights(mat: &mut Mat) -> f32 {
    let n_genes = mat.nrows();
    let n_ct = mat.ncols();
    let c_total = n_ct as f32;
    for g in 0..n_genes {
        let c_g = mat.row(g).iter().filter(|&&v| v > 0.0).count() as f32;
        if c_g == 0.0 {
            continue;
        }
        let w = (c_total / c_g).ln();
        for c in 0..n_ct {
            if mat[(g, c)] > 0.0 {
                mat[(g, c)] = w;
            }
        }
    }
    c_total.ln()
}

/// Build the `AnnotInfo` form (membership matrix + celltype names) from a
/// marker TSV and the dictionary's gene-name order. Genes are matched to
/// `row_names` via `flexible_gene_match`; unmatched markers are logged and
/// dropped.
pub(crate) fn build_annotation_matrix(
    marker_gene_path: &str,
    row_names: &[Box<str>],
) -> anyhow::Result<AnnotInfo> {
    let marker_pairs = read_marker_gene_info(marker_gene_path)?;

    if marker_pairs.is_empty() {
        return Err(anyhow::anyhow!("empty/invalid marker gene information"));
    }

    let mut annot_set: HashSet<Box<str>> = HashSet::default();
    for (_, cell_type) in &marker_pairs {
        let normalized = cell_type.replace(' ', "_");
        annot_set.insert(normalized.into_boxed_str());
    }
    let mut annot_names: Vec<Box<str>> = annot_set.into_iter().collect();
    annot_names.sort();

    let n_genes = row_names.len();
    let n_annots = annot_names.len();
    let mut membership = Mat::zeros(n_genes, n_annots);

    let mut matched = 0;
    let mut unmatched = Vec::new();

    for (gene, cell_type) in &marker_pairs {
        let normalized_type = cell_type.replace(' ', "_");
        let annot_idx = annot_names
            .iter()
            .position(|n| n.as_ref() == normalized_type)
            .unwrap();

        if let Some(gene_idx) = row_names
            .iter()
            .position(|dict_gene| flexible_gene_match(gene, dict_gene))
        {
            membership[(gene_idx, annot_idx)] = 1.0;
            matched += 1;
        } else {
            unmatched.push(gene.clone());
        }
    }

    if !unmatched.is_empty() && unmatched.len() <= 10 {
        log::info!("Unmatched marker genes: {unmatched:?}");
    } else if !unmatched.is_empty() {
        log::info!("{} marker genes not found in dictionary", unmatched.len());
    }

    let max_idf = apply_idf_weights(&mut membership);
    log::info!(
        "Matched {}/{} marker genes to {} cell types (IDF max ln(C) = {:.3})",
        matched,
        marker_pairs.len(),
        n_annots,
        max_idf,
    );

    Ok(AnnotInfo {
        membership_ga: membership,
        annot_names,
    })
}

/// Load a marker TSV and align it to the dictionary in the richer form used
/// by the anchor-based prior. Keeps the raw `(gene, celltype)` pairs so the
/// caller can iterate markers per celltype at scoring time without another
/// file read.
pub(crate) fn load_marker_info(
    marker_gene_path: &str,
    row_names: &[Box<str>],
) -> anyhow::Result<MarkerInfo> {
    let pairs = read_marker_gene_info(marker_gene_path)?;
    if pairs.is_empty() {
        return Err(anyhow::anyhow!("empty/invalid marker gene information"));
    }

    let mut celltype_set: HashSet<Box<str>> = HashSet::default();
    for (_, ct) in &pairs {
        celltype_set.insert(ct.replace(' ', "_").into_boxed_str());
    }
    let mut celltypes: Vec<Box<str>> = celltype_set.into_iter().collect();
    celltypes.sort();

    let n_genes = row_names.len();
    let n_ct = celltypes.len();
    let mut membership_gc = Mat::zeros(n_genes, n_ct);

    let mut matched = 0usize;
    for (gene, ct) in &pairs {
        let normalized = ct.replace(' ', "_");
        let c = celltypes
            .iter()
            .position(|n| n.as_ref() == normalized)
            .unwrap();
        if let Some(g) = row_names
            .iter()
            .position(|dict_gene| flexible_gene_match(gene, dict_gene))
        {
            membership_gc[(g, c)] = 1.0;
            matched += 1;
        }
    }

    let max_idf = apply_idf_weights(&mut membership_gc);
    log::info!(
        "MarkerInfo: matched {}/{} markers across {} celltypes (TF-IDF max ln(C) = {:.3})",
        matched,
        pairs.len(),
        n_ct,
        max_idf,
    );

    Ok(MarkerInfo {
        celltypes,
        membership_gc,
    })
}
