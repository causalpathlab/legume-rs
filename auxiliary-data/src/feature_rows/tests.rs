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

/////////////////////////////////////
// gene-count splitter + interner  //
/////////////////////////////////////

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

#[test]
fn splits_count_rows_into_gene_and_track() {
    assert_eq!(
        split_count_row("ENSG001_GENE1/count/spliced"),
        Some(("ENSG001_GENE1", false))
    );
    assert_eq!(
        split_count_row("ENSG001_GENE1/count/unspliced"),
        Some(("ENSG001_GENE1", true))
    );
}

/// A row that is not a gene-level count row must be REJECTED, not silently
/// absorbed as a spliced one.
///
/// The old `rsplit_once("/count/")` could not tell the two apart — both fell to
/// the same branch — so `GENE1/m6a/methylated` became a mature gene literally
/// named `GENE1/m6a/methylated`, and a per-site row became a mature row of the
/// right gene. Neither errored.
#[test]
fn non_count_rows_are_rejected_not_silently_called_spliced() {
    // wrong modality
    assert_eq!(split_count_row("ENSG001_GENE1/m6a/methylated"), None);
    // right modality, wrong channel: `total` IS spliced + unspliced, so taking
    // it as a third track would count the gene twice
    assert_eq!(split_count_row("ENSG001_GENE1/count/total"), None);
    // sub-gene resolution: a site row is not pairable at gene resolution
    assert_eq!(
        split_count_row("ENSG001_GENE1/count/chr1:100/spliced"),
        None
    );
    // not a feature row at all
    assert_eq!(split_count_row("weird_name"), None);
}

/// A gene's two rows must intern to ONE gene id under either policy — that
/// pairing is the whole point of the map.
#[test]
fn both_tracks_of_a_gene_share_one_id() {
    let rows = names(&[
        "A/count/spliced",
        "B/count/unspliced",
        "A/count/unspliced",
        "B/count/spliced",
        "C/count/spliced",
    ]);
    for policy in [UnparsedRowPolicy::OwnGene, UnparsedRowPolicy::Reject] {
        let map = intern_count_rows(&rows, policy);
        assert_eq!(map.n_genes(), 3, "{policy:?}");
        assert_eq!(
            map.gene_names.as_slice(),
            names(&["A", "B", "C"]).as_slice()
        );
        assert_eq!(map.row_to_gene, vec![0, 1, 0, 1, 2]);
        assert_eq!(
            map.row_is_nascent,
            vec![false, true, true, false, false],
            "nascent flags must follow the /count/unspliced suffix"
        );
        assert!(map.unparsed.is_empty());
        assert_eq!(map.n_nascent_rows(), 2);
    }
}

/// The two policies differ ONLY on a row the splitter rejects, and the
/// difference is the whole reason both exist.
///
/// Break it by giving `Reject` the `OwnGene` body and the id table stops being
/// a gene axis: `n_genes` grows by one per stray row, and a pooling consumer
/// silently gains a phantom gene it will happily fit.
#[test]
fn the_policies_differ_only_on_an_unparsable_row() {
    let rows = names(&["A/count/spliced", "A/count/total", "A/count/unspliced"]);

    let keep = intern_count_rows(&rows, UnparsedRowPolicy::OwnGene);
    assert_eq!(keep.n_genes(), 2, "the total row gets its own id");
    assert_eq!(keep.gene_names[1].as_ref(), "A/count/total");
    assert_ne!(
        keep.row_to_gene[1], keep.row_to_gene[0],
        "a `total` row summed into the spliced track double-counts the gene"
    );

    let strict = intern_count_rows(&rows, UnparsedRowPolicy::Reject);
    assert_eq!(strict.n_genes(), 1, "only gene A is on the axis");
    assert_eq!(strict.row_to_gene[1], NO_GENE);
    assert_eq!(strict.unparsed, vec![1]);

    // Common to both: every row keeps its index, and an unpairable row is never
    // flagged nascent, so it can never reach a velocity contrast.
    for map in [&keep, &strict] {
        assert_eq!(map.n_rows(), rows.len());
        assert_eq!(map.row_is_nascent.len(), rows.len());
        assert!(!map.row_is_nascent[1]);
    }
}

/// A spliced-only matrix is a legitimate input; it simply pins no
/// nascent-minus-mature contrast, and callers key their identifiability report
/// on that rather than on an error.
#[test]
fn a_spliced_only_matrix_interns_cleanly_with_no_nascent_rows() {
    let rows = names(&["A/count/spliced", "B/count/spliced"]);
    let map = intern_count_rows(&rows, UnparsedRowPolicy::Reject);
    assert_eq!(map.n_genes(), 2);
    assert_eq!(map.n_nascent_rows(), 0);
    assert!(map.unparsed.is_empty());
}

/// Real gene symbols contain slashes, and standard human references ship at
/// least one. Such a gene's count rows have FOUR fields, so the old positional
/// rule read them as a sub-gene row of a DIFFERENT gene:
/// `{id}_GENE1` / `GENE1B` / `count` / `spliced`. Nothing errored — the gene
/// simply stopped existing and its two channel rows stopped pairing.
///
/// Break the fix by restoring the pure field-count match and both asserts fail:
/// `gene` loses `/BTR` and `modality` comes back as `BTR`.
#[test]
fn a_unit_may_contain_slashes_because_gene_symbols_do() {
    let row = feature_row("ENSG001_GENE1/GENE1B", COUNT, SPLICED, None);
    assert_eq!(row.as_ref(), "ENSG001_GENE1/GENE1B/count/spliced");

    let parsed = parse_feature_row(&row).unwrap();
    assert_eq!(parsed.gene, "ENSG001_GENE1/GENE1B");
    assert_eq!(parsed.modality, COUNT);
    assert_eq!(parsed.channel, SPLICED);
    assert_eq!(parsed.subunit, None);

    // and it must still pair across tracks
    assert_eq!(
        split_count_row("ENSG001_GENE1/GENE1B/count/unspliced"),
        Some(("ENSG001_GENE1/GENE1B", true))
    );

    // the sub-gene form of the same gene keeps its subunit
    let site = feature_row("A/B", M6A, METHYLATED, Some("chr1:100"));
    let parsed = parse_feature_row(&site).unwrap();
    assert_eq!(parsed.gene, "A/B");
    assert_eq!(parsed.modality, M6A);
    assert_eq!(parsed.subunit, Some("chr1:100"));
    assert_eq!(parsed.unit().as_ref(), "A/B/m6a/chr1:100");
}

/// The modality vocabulary LOCATES the split; it does not validate it. A
/// producer that still emits an inline modality token outside the constant list
/// must keep parsing exactly as it did, or this change would break the readers
/// it was supposed to leave alone.
#[test]
fn an_unknown_modality_falls_back_to_the_positional_rule() {
    let three = parse_feature_row("GENE/pileup/forward").unwrap();
    assert_eq!(
        (three.gene, three.modality, three.channel),
        ("GENE", "pileup", "forward")
    );
    assert_eq!(three.subunit, None);

    let four = parse_feature_row("GENE/pileup/7/forward").unwrap();
    assert_eq!(
        (four.gene, four.modality, four.channel),
        ("GENE", "pileup", "forward")
    );
    assert_eq!(four.subunit, Some("7"));

    // Five fields with no known modality is still junk.
    assert!(parse_feature_row("GENE/pileup/7/x/forward").is_none());
    // ...and so is a five-field row whose only known token sits too far left.
    assert!(parse_feature_row("GENE/count/7/x/spliced").is_none());
}
