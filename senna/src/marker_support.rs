//! Shared parsing and alignment utilities for `gene<TAB>celltype` marker
//! files. Used by the `annotate` subcommand to build the membership
//! matrix for vMF assignment.

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
        let normalized = cell_type.replace(" ", "_");
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
        let normalized_type = cell_type.replace(" ", "_");
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
        log::info!("Unmatched marker genes: {:?}", unmatched);
    } else if !unmatched.is_empty() {
        log::info!("{} marker genes not found in dictionary", unmatched.len());
    }

    log::info!(
        "Matched {}/{} marker genes to {} cell types",
        matched,
        marker_pairs.len(),
        n_annots
    );

    Ok(AnnotInfo {
        membership_ga: membership,
        annot_names,
    })
}

