use super::*;

#[test]
fn site_id_uses_the_shared_apa_modality_token() {
    // The emitted token must be `feature_name::APA`, not a hand-spelled variant:
    // consumers select rows by splitting on it (`assoc`, `gem_encoder::load`).
    assert_eq!(
        site_id("ENSG1_TP53", "3").as_ref(),
        "ENSG1_TP53/apa/3",
        "mixture component row"
    );
    assert_eq!(
        site_id("ENSG1_TP53", "chr17:7668402").as_ref(),
        "ENSG1_TP53/apa/chr17:7668402",
        "simple-mode site row"
    );
}

#[test]
fn site_gene_inverts_site_id() {
    for subunit in ["0", "12", "chr1:100"] {
        let id = site_id("ENSG1_TP53", subunit);
        assert_eq!(site_gene(&id), "ENSG1_TP53");
    }
}

#[test]
fn site_gene_passes_through_an_id_with_no_infix() {
    // The callers that split treat a bare gene as itself rather than dropping it.
    assert_eq!(site_gene("ENSG1_TP53"), "ENSG1_TP53");
}

#[test]
fn a_gene_containing_the_infix_splits_at_the_first_one() {
    // `split_once` is deliberate: gene keys are `{id}_{symbol}` and never contain
    // `/`, so the first infix is the real boundary.
    assert_eq!(site_gene("G/apa/1/apa/2"), "G");
}
