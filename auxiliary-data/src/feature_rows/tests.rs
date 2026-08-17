//! Round-trip + edge-case tests for the canonical feature-name convention.

use super::*;

#[test]
fn gene_level_row_round_trips() {
    let row = feature_row("ENSG00000141510_TP53", M6A, METHYLATED, None);
    assert_eq!(row.as_ref(), "ENSG00000141510_TP53/m6a/methylated");
    let parsed = parse_feature_row(&row).unwrap();
    assert_eq!(parsed.gene, "ENSG00000141510_TP53");
    assert_eq!(parsed.modality, M6A);
    assert_eq!(parsed.channel, METHYLATED);
    assert_eq!(parsed.subunit, None);
    assert_eq!(parsed.unit().as_ref(), "ENSG00000141510_TP53");
}

#[test]
fn subunit_row_round_trips_and_keeps_the_gene_recoverable() {
    // site subunit (no EM) — channel is the trailing field
    let site = feature_row("GENE", M6A, UNMETHYLATED, Some("chr17:7668402-7687550"));
    assert_eq!(site.as_ref(), "GENE/m6a/chr17:7668402-7687550/unmethylated");
    let p = parse_feature_row(&site).unwrap();
    assert_eq!(p.subunit, Some("chr17:7668402-7687550"));
    assert_eq!(p.channel, UNMETHYLATED);
    assert_eq!(p.unit().as_ref(), "GENE/m6a/chr17:7668402-7687550");
    assert_eq!(p.unit().split('/').next().unwrap(), "GENE");

    // component subunit (EM)
    let comp = feature_row("GENE", ATOI, EDITED, Some("3"));
    assert_eq!(comp.as_ref(), "GENE/atoi/3/edited");
    let p = parse_feature_row(&comp).unwrap();
    assert_eq!(p.modality, ATOI);
    assert_eq!(p.channel, EDITED);
    assert_eq!(p.unit().as_ref(), "GENE/atoi/3");
}

#[test]
fn a_units_two_channels_share_a_contiguous_prefix() {
    // The unit is the prefix; only the trailing channel differs.
    let m = feature_row("GENE", M6A, METHYLATED, Some("0"));
    let u = feature_row("GENE", M6A, UNMETHYLATED, Some("0"));
    assert_eq!(m.as_ref(), "GENE/m6a/0/methylated");
    assert_eq!(u.as_ref(), "GENE/m6a/0/unmethylated");
    let (pm, pu) = (
        parse_feature_row(&m).unwrap(),
        parse_feature_row(&u).unwrap(),
    );
    assert_eq!(pm.unit(), pu.unit());
    assert_eq!(pm.unit().as_ref(), "GENE/m6a/0");
}

#[test]
fn count_and_other_modalities_use_the_same_shape() {
    assert_eq!(
        feature_row("GENE", COUNT, SPLICED, None).as_ref(),
        "GENE/count/spliced"
    );
    assert_eq!(
        feature_row("GENE", APA, PROXIMAL, Some("chr1:100-200")).as_ref(),
        "GENE/apa/chr1:100-200/proximal"
    );
    // BAF's unit is the locus, not a gene, and it takes no subunit — the two
    // channels hang directly off the coordinate.
    assert_eq!(
        feature_row("chr1:200", BAF, ALT, None).as_ref(),
        "chr1:200/baf/alt"
    );
    assert_eq!(
        feature_row("chr1:200", BAF, DEPTH, None).as_ref(),
        "chr1:200/baf/depth"
    );
}

/// A BAF row still round-trips through the generic parser: three fields, so the
/// locus lands in the `gene` slot and the parsed unit is the bare locus.
#[test]
fn baf_rows_parse_with_the_locus_as_the_unit() {
    let parsed = parse_feature_row("chr1:200/baf/alt").expect("3-field row parses");
    assert_eq!(parsed.gene, "chr1:200");
    assert_eq!(parsed.modality, BAF);
    assert_eq!(parsed.channel, ALT);
    assert_eq!(parsed.subunit, None);
    assert_eq!(parsed.unit().as_ref(), "chr1:200");

    // Both channels of a locus share the unit, which is what lets a consumer
    // pair them for the ratio.
    let depth = parse_feature_row("chr1:200/baf/depth").expect("3-field row parses");
    assert_eq!(depth.unit(), parsed.unit());
}

#[test]
fn rows_outside_three_or_four_fields_are_rejected() {
    assert!(parse_feature_row("GENE").is_none());
    assert!(parse_feature_row("GENE/m6a").is_none());
    assert!(parse_feature_row("GENE/m6a/methylated").is_some());
    assert!(parse_feature_row("GENE/m6a/0/methylated").is_some());
    assert!(parse_feature_row("GENE/m6a/0/x/methylated").is_none());
}
