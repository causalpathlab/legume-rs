//! `FeatureNameKind::reconcile` — the rule that turns one auto-detected kind
//! per input file into the kind the loader installs for all of them.
//!
//! Detecting over the pooled names is what broke `senna update
//! --use-pb-reference`: a raw `ENSG_SYM` cohort pooled with a carried
//! reference already on the bare-symbol axis sniffed as `Exact` (the
//! gene-like share fell under half), so the two spellings of every gene
//! became two rows. Per-file detection sees Gene on one side and Exact on
//! the other, and Gene is a no-op on bare symbols, so adopting it is safe.

use super::FeatureNameKind;

const GENE: FeatureNameKind = FeatureNameKind::Gene { delim: '_' };
const LOCUS: FeatureNameKind = FeatureNameKind::Locus {
    merge_overlapping: true,
};

#[test]
fn all_exact_stays_exact() {
    assert_eq!(
        FeatureNameKind::reconcile(&[FeatureNameKind::Exact, FeatureNameKind::Exact]),
        FeatureNameKind::Exact
    );
}

#[test]
fn a_bare_symbol_file_adopts_the_gene_rule_of_its_neighbour() {
    // The pb_reference case: raw cohort + canonical reference.
    assert_eq!(
        FeatureNameKind::reconcile(&[GENE, FeatureNameKind::Exact]),
        GENE
    );
    assert_eq!(
        FeatureNameKind::reconcile(&[FeatureNameKind::Exact, GENE]),
        GENE
    );
}

#[test]
fn agreeing_files_keep_their_kind() {
    assert_eq!(FeatureNameKind::reconcile(&[GENE, GENE]), GENE);
    assert_eq!(FeatureNameKind::reconcile(&[LOCUS, LOCUS]), LOCUS);
}

#[test]
fn genes_and_loci_across_files_dispatch_per_name() {
    assert_eq!(
        FeatureNameKind::reconcile(&[GENE, LOCUS]),
        FeatureNameKind::Mixed
    );
    // Order and an Exact bystander do not change the answer.
    assert_eq!(
        FeatureNameKind::reconcile(&[FeatureNameKind::Exact, LOCUS, GENE]),
        FeatureNameKind::Mixed
    );
}

#[test]
fn mixed_anywhere_is_mixed() {
    assert_eq!(
        FeatureNameKind::reconcile(&[FeatureNameKind::Mixed, GENE]),
        FeatureNameKind::Mixed
    );
    assert_eq!(
        FeatureNameKind::reconcile(&[FeatureNameKind::Exact, FeatureNameKind::Mixed]),
        FeatureNameKind::Mixed
    );
}

#[test]
fn no_files_is_exact() {
    assert_eq!(FeatureNameKind::reconcile(&[]), FeatureNameKind::Exact);
}
