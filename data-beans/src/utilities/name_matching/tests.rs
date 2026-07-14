use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

#[test]
fn gene_index_resolves_hgnc_renames_both_ways() {
    // A matrix on the CURRENT HGNC release; a marker panel on an older one. Without the
    // alias tier every one of these silently drops out of the panel.
    let dict = names(&["H4C3", "H2BC21", "H1-0", "MARCHF2", "SEPTIN7", "MTARC1", "CD8A"]);
    let idx = GeneIndex::build(&dict);
    assert_eq!(idx.match_gene("HIST1H4C"), Some(0));
    assert_eq!(idx.match_gene("HIST2H2BE"), Some(1));
    assert_eq!(idx.match_gene("H1F0"), Some(2));
    assert_eq!(idx.match_gene("MARCH2"), Some(3), "rule-based MARCH family");
    assert_eq!(idx.match_gene("SEPT7"), Some(4), "rule-based SEPT family");
    assert_eq!(
        idx.match_gene("MARC1"),
        Some(5),
        "MARC1 is MTARC1, not a MARCH-family member"
    );

    // ...and the reverse: an old-release matrix, a current panel.
    let old = names(&["HIST1H4C", "MARCH2", "SEPT7", "CD8A"]);
    let idx_old = GeneIndex::build(&old);
    assert_eq!(idx_old.match_gene("H4C3"), Some(0));
    assert_eq!(idx_old.match_gene("MARCHF2"), Some(1));
    assert_eq!(idx_old.match_gene("SEPTIN7"), Some(2));

    // The alias tier resolves through the faba `{gene}/count/{track}` row keys and the
    // `ENSG…_SYMBOL` form too, since it runs on the extracted symbol.
    let faba = names(&["ENSG00000197061_H4C3/count/spliced", "CD8A/count/spliced"]);
    let idx_faba = GeneIndex::build(&faba);
    assert_eq!(idx_faba.match_gene("HIST1H4C"), Some(0));

    // A gene with no alias must not be coerced into one.
    assert_eq!(idx.match_gene("ZZZ9"), None);
}

#[test]
fn gene_index_tiers_and_idf() {
    let dict = names(&["ENSG00000153563_CD8A", "MS4A1", "A_FOO", "FOO"]);
    let idx = GeneIndex::build(&dict);

    // exact (case-insensitive)
    assert_eq!(idx.match_gene("ms4a1"), Some(1));
    // symbol = last `_`-segment
    assert_eq!(idx.match_gene("CD8A"), Some(0));
    // exact preferred over an earlier-indexed suffix match (the refinement)
    assert_eq!(idx.match_gene("FOO"), Some(3));
    // unmatched
    assert_eq!(idx.match_gene("ZZZ9"), None);

    // bare ENSG query resolves to the `ENSG…_SYMBOL` row (and the combined
    // form resolves too) — HGNC / ENSG / ENSG_HGNC reconcile both ways.
    assert_eq!(idx.match_gene("ENSG00000153563"), Some(0));
    assert_eq!(idx.match_gene("ENSG00000153563_CD8A"), Some(0));

    // dict keyed by bare ENSG: a combined query still resolves via the
    // leading-id tier.
    let dict2 = names(&["ENSG00000153563", "MS4A1"]);
    let idx2 = GeneIndex::build(&dict2);
    assert_eq!(idx2.match_gene("ENSG00000153563_CD8A"), Some(0));
    assert_eq!(idx2.match_gene("ensg00000153563"), Some(0));

    // IDF: ubiquitous (df == C) → 0; exclusive → ln(C)
    assert_eq!(idf_weight(4, 4), 0.0);
    assert!(idf_weight(4, 1) > 1.38 && idf_weight(4, 1) < 1.39);
}
