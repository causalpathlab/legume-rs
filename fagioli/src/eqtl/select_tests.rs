//! Tests for top-K selection: the ranking is per gene, the retained set is
//! per variant.

use super::*;
use crate::eqtl::reader::{GeneTable, QtlEntry};

/// Build a table from `(gene, variant, celltype, beta, se)` rows. Names are
/// synthesized, so tests only speak in indices.
fn table(rows: &[(u32, u32, u32, f32, f32)]) -> QtlData {
    let n_genes = rows.iter().map(|r| r.0).max().unwrap_or(0) as usize + 1;
    let n_variants = rows.iter().map(|r| r.1).max().unwrap_or(0) as usize + 1;
    let n_ct = rows.iter().map(|r| r.2).max().unwrap_or(0) as usize + 1;

    let mut tables: Vec<GeneTable> = (0..n_genes).map(|_| GeneTable::default()).collect();
    for &(g, v, k, beta, se) in rows {
        tables[g as usize].entries.push(QtlEntry {
            variant: v,
            celltype: k,
            beta,
            se,
            pip_weight: None,
        });
    }
    for t in &mut tables {
        t.entries.sort_unstable_by_key(|e| (e.variant, e.celltype));
    }

    QtlData {
        genes: (0..n_genes).map(|g| Box::from(format!("G{g}"))).collect(),
        celltypes: (0..n_ct).map(|k| Box::from(format!("C{k}"))).collect(),
        variants: (0..n_variants)
            .map(|v| Box::from(format!("1:{v}")))
            .collect(),
        tables,
        n_files_read: 1,
        n_files_skipped: 0,
        n_rows_dropped: 0,
        n_rows_duplicate: 0,
        n_genes_dropped: 0,
    }
}

fn selected_variants(sel: &Selection) -> Vec<u32> {
    let mut v: Vec<u32> = sel.pairs.iter().map(|p| p.variant).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Variant 0 is consistently associated across both cell types; variant 1
/// is stronger in cell type 0 alone. Ranking once per gene keeps variant 0.
/// Ranking within each cell type would rank variant 1 first in cell type 0,
/// pull it into the union, and put a variant selected for its
/// cell-type-specific strength into the very contrast the model has to
/// learn.
#[test]
fn top_k_ranks_once_per_gene_not_per_celltype() {
    let data = table(&[
        // gene 0, variant 0: chi2 = 9 in both cell types.
        (0, 0, 0, 0.3, 0.1),
        (0, 0, 1, 0.3, 0.1),
        // gene 0, variant 1: chi2 = 16 in cell type 0, 0 in cell type 1.
        (0, 1, 0, 0.4, 0.1),
        (0, 1, 1, 0.0, 0.1),
    ]);

    let sel = select_top_variants(&data, 1);
    assert_eq!(selected_variants(&sel), vec![0]);
    assert_eq!(sel.n_selected_variants, 1);
    // The unselected variant's pair is gone.
    assert_eq!(sel.pairs.len(), 1);
    assert_eq!(sel.n_pairs_dropped, 1);
}

/// A variant selected for one gene must keep the rows of every other gene
/// it was tested against: those pairs are the gene-swap negative pool.
#[test]
fn a_selected_variant_keeps_all_the_genes_it_was_tested_against() {
    let data = table(&[
        // gene 0 picks variant 0 (its only variant).
        (0, 0, 0, 0.5, 0.1),
        (0, 0, 1, 0.5, 0.1),
        // gene 1 was also tested against variant 0, but ranks variant 1 first.
        (1, 0, 0, 0.01, 0.1),
        (1, 0, 1, 0.01, 0.1),
        (1, 1, 0, 0.9, 0.1),
        (1, 1, 1, 0.9, 0.1),
    ]);

    let sel = select_top_variants(&data, 1);
    assert_eq!(selected_variants(&sel), vec![0, 1]);

    // Variant 0 is retained for gene 1 even though gene 1 never chose it.
    assert!(sel
        .pairs
        .iter()
        .any(|p| p.gene == 1 && p.variant == 0 && p.obs.len() == 2));
    // Variant 1 belongs to gene 1 only, so gene 0 gains no pair for it.
    assert!(!sel.pairs.iter().any(|p| p.gene == 0 && p.variant == 1));
}

/// The rank is a precision-weighted mean, so a huge chi-square carried by
/// one imprecise cell type loses to a moderate one seen precisely.
#[test]
fn ranking_weights_cell_types_by_precision() {
    let data = table(&[
        // variant 0: chi2 100 at se 1 (weight 1), chi2 1 at se 0.1 (weight 100).
        (0, 0, 0, 10.0, 1.0),
        (0, 0, 1, 0.1, 0.1),
        // variant 1: chi2 3 in both, both precise.
        (0, 1, 0, 0.173_205, 0.1),
        (0, 1, 1, 0.173_205, 0.1),
    ]);

    let sel = select_top_variants(&data, 1);
    assert_eq!(selected_variants(&sel), vec![1]);
}

#[test]
fn top_k_larger_than_the_gene_keeps_everything() {
    let data = table(&[
        (0, 0, 0, 0.5, 0.1),
        (0, 1, 0, 0.4, 0.1),
        (0, 2, 0, 0.3, 0.1),
    ]);

    let sel = select_top_variants(&data, 5);
    assert_eq!(selected_variants(&sel), vec![0, 1, 2]);
    assert_eq!(sel.n_pairs_dropped, 0);
    assert_eq!(sel.n_rows, 3);
}
