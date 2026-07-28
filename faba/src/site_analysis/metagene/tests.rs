use super::*;

//////////////
// Fixtures  //
//////////////

const CHR: &str = "chr1";

fn rec(gene: &str, feature_type: FeatureType, start: i64, stop: i64, strand: Strand) -> GffRecord {
    GffRecord {
        seqname: CHR.into(),
        feature_type,
        start,
        stop,
        strand,
        gene_id: GeneId::Ensembl(gene.into()),
        gene_name: GeneSymbol::Symbol(gene.into()),
        gene_type: GeneType::CodingGene,
    }
}

fn non_coding_rec(gene: &str, start: i64, stop: i64, strand: Strand) -> GffRecord {
    GffRecord {
        gene_type: GeneType::LincRNA,
        ..rec(gene, FeatureType::Gene, start, stop, strand)
    }
}

/// A merged feature built straight from intervals, for index-level tests.
fn feature(strand: Strand, intervals: &[(i64, i64)]) -> MergedFeature {
    MergedFeature {
        seqname: CHR.into(),
        strand,
        intervals: intervals.to_vec(),
        total_len: intervals.iter().map(|&(s, e)| e - s + 1).sum(),
    }
}

/// A two-exon coding gene: CDS exons split by an intron at 1101..1999, a UTR
/// record beyond each end, and the codons that decide which end is which.
/// On the reverse strand the codons swap, so the SAME coordinates describe a
/// gene whose 5' end is the high one.
fn two_exon_gene(gene: &str, strand: Strand) -> Vec<GffRecord> {
    let (start_codon, stop_codon) = match strand {
        Strand::Forward => ((1000, 1002), (2098, 2100)),
        Strand::Backward => ((2098, 2100), (1000, 1002)),
    };
    vec![
        rec(gene, FeatureType::CDS, 1000, 1100, strand),
        rec(gene, FeatureType::CDS, 2000, 2100, strand),
        rec(
            gene,
            FeatureType::StartCodon,
            start_codon.0,
            start_codon.1,
            strand,
        ),
        rec(
            gene,
            FeatureType::StopCodon,
            stop_codon.0,
            stop_codon.1,
            strand,
        ),
        rec(gene, FeatureType::UTR, 900, 999, strand),
        rec(gene, FeatureType::UTR, 2101, 2300, strand),
    ]
}

/// Sites carry 0-based positions; every coordinate in these tests is the
/// 1-based GFF one.
fn site(pos_1based: i64) -> GenomicSite {
    GenomicSite {
        chr: CHR.into(),
        position: pos_1based - 1,
        strand: Strand::Forward,
    }
}

/// `[5'UTR, CDS, 3'UTR, ncRNA]` site totals.
fn totals(hist: &GeneFeatureHistogram) -> [usize; 4] {
    [
        hist.five_prime.iter().sum(),
        hist.cds.iter().sum(),
        hist.three_prime.iter().sum(),
        hist.non_coding.iter().sum(),
    ]
}

/// Where one site lands under the default: protein-coding genes only.
fn classify(records: &[GffRecord], pos_1based: i64) -> [usize; 4] {
    classify_with_non_coding(records, pos_1based, false)
}

fn classify_with_non_coding(
    records: &[GffRecord],
    pos_1based: i64,
    include_non_coding: bool,
) -> [usize; 4] {
    let grid = MetageneGrid::from_records(records, 60, include_non_coding).expect("grid");
    totals(&count_metagene(&[site(pos_1based)], &grid))
}

/////////////////////////
// Defect 2: find scan  //
/////////////////////////

#[test]
fn find_reaches_past_a_shorter_later_interval() {
    // Sorted by start: [(100, 5000), (200, 300)]. Walking back from the
    // rightmost start <= 1000 hits (200, 300) first; `start < position` is no
    // reason to stop, because (100, 5000) still contains 1000.
    let idx = FeatureIndex::from_features(&[
        feature(Strand::Forward, &[(100, 5000)]),
        feature(Strand::Forward, &[(200, 300)]),
    ]);

    let hit = idx.find(CHR, 1000).expect("1000 lies inside (100, 5000)");
    assert_eq!((hit.start, hit.stop), (100, 5000));

    // The short interval still wins where it actually applies (rightmost
    // containing start), and nothing is invented past the far end.
    let hit = idx.find(CHR, 250).expect("250 lies inside (200, 300)");
    assert_eq!((hit.start, hit.stop), (200, 300));
    assert_eq!(idx.find(CHR, 5001), None);
    assert_eq!(idx.find(CHR, 99), None);
    assert_eq!(idx.find("chrX", 1000), None);
}

#[test]
fn find_scans_back_over_several_non_containing_intervals() {
    // Three short intervals stack up after the long one; the backward scan has
    // to cross all of them before it reaches the interval that contains 4000.
    let idx = FeatureIndex::from_features(&[
        feature(Strand::Forward, &[(100, 5000)]),
        feature(Strand::Forward, &[(200, 300)]),
        feature(Strand::Forward, &[(400, 500)]),
        feature(Strand::Forward, &[(600, 700)]),
    ]);
    let hit = idx.find(CHR, 4000).expect("4000 lies inside (100, 5000)");
    assert_eq!((hit.start, hit.stop), (100, 5000));

    // …and the max-stop bound still lets it give up: nothing covers 5500.
    assert_eq!(idx.find(CHR, 5500), None);
}

//////////////////////////////////
// Defect 1: real vs span model  //
//////////////////////////////////

#[test]
fn intronic_site_between_cds_exons_is_not_called_cds() {
    for strand in [Strand::Forward, Strand::Backward] {
        let records = two_exon_gene("G", strand);
        // 1500 sits in the intron between CDS exons 1000..1100 and 2000..2100.
        // A min..max CDS span reaches 1000..2100 and would claim it.
        assert_eq!(classify(&records, 1500), [0, 0, 0, 0], "{strand:?}");
        // The exons themselves are still CDS.
        assert_eq!(classify(&records, 1050), [0, 1, 0, 0], "{strand:?}");
        assert_eq!(classify(&records, 2050), [0, 1, 0, 0], "{strand:?}");
    }
}

#[test]
fn three_prime_site_under_the_union_cds_span_is_three_prime() {
    // Isoform A stops at 1100 and its 3'UTR runs 1101..1300. Isoform B has a
    // further CDS exon at 1400..1600, so the min..max CDS span is 1000..1600 —
    // it covers the whole of A's 3'UTR, and since sites are tested
    // 5'UTR -> CDS -> 3'UTR that span used to take those sites.
    let records = vec![
        rec("G", FeatureType::CDS, 1000, 1100, Strand::Forward),
        rec("G", FeatureType::CDS, 1400, 1600, Strand::Forward),
        rec("G", FeatureType::StartCodon, 1000, 1002, Strand::Forward),
        rec("G", FeatureType::StopCodon, 1098, 1100, Strand::Forward),
        rec("G", FeatureType::UTR, 900, 999, Strand::Forward),
        rec("G", FeatureType::UTR, 1101, 1300, Strand::Forward),
    ];

    assert_eq!(classify(&records, 1200), [0, 0, 1, 0]);
    // The real CDS exons on either side of it are untouched.
    assert_eq!(classify(&records, 1050), [0, 1, 0, 0]);
    assert_eq!(classify(&records, 1500), [0, 1, 0, 0]);
    // …and so is the 5' end.
    assert_eq!(classify(&records, 950), [1, 0, 0, 0]);
}

#[test]
fn generic_utr_records_are_split_by_codon_distance() {
    // Forward: the low-coordinate UTR is the 5' one. Reverse: the high one is.
    assert_eq!(
        classify(&two_exon_gene("G", Strand::Forward), 950),
        [1, 0, 0, 0]
    );
    assert_eq!(
        classify(&two_exon_gene("G", Strand::Forward), 2200),
        [0, 0, 1, 0]
    );
    assert_eq!(
        classify(&two_exon_gene("G", Strand::Backward), 950),
        [0, 0, 1, 0]
    );
    assert_eq!(
        classify(&two_exon_gene("G", Strand::Backward), 2200),
        [1, 0, 0, 0]
    );
}

#[test]
fn isoform_records_merge_instead_of_spanning() {
    // Two isoforms' CDS exons overlap, a third abuts the merged stretch, and a
    // fourth is separated by an intron: 1000..1300 and 2000..2100, never
    // 1000..2100.
    let records = vec![
        rec("G", FeatureType::CDS, 1000, 1100, Strand::Forward),
        rec("G", FeatureType::CDS, 1050, 1200, Strand::Forward),
        rec("G", FeatureType::CDS, 1201, 1300, Strand::Forward),
        rec("G", FeatureType::CDS, 2000, 2100, Strand::Forward),
    ];
    let tracks = build_feature_tracks(&records, false).expect("tracks");
    assert_eq!(tracks.cds.len(), 1);
    assert_eq!(tracks.cds[0].intervals, vec![(1000, 1300), (2000, 2100)]);
    assert_eq!(tracks.cds[0].total_len, 301 + 101);
}

#[test]
fn non_coding_genes_keep_whole_gene_boundaries() {
    let records = vec![non_coding_rec("NC", 5000, 6000, Strand::Forward)];
    assert_eq!(classify_with_non_coding(&records, 5500, true), [0, 0, 0, 1]);
    assert_eq!(classify_with_non_coding(&records, 4999, true), [0, 0, 0, 0]);
}

#[test]
fn non_coding_genes_are_dropped_by_default() {
    let records = vec![non_coding_rec("NC", 5000, 6000, Strand::Forward)];
    // The site is inside the gene, and still contributes nothing: without
    // --include-non-coding the ncRNA track is not built at all.
    assert_eq!(classify(&records, 5500), [0, 0, 0, 0]);

    let grid = MetageneGrid::from_records(&records, 60, false).expect("grid");
    assert!(grid.non_coding.is_empty());
    // Zero-wide, so `to_tsv` writes no ncRNA rows rather than 60 zeros.
    assert_eq!(grid.nbins[3], 0);

    // …and the flag is what decides that, not the absence of such genes:
    // asked-for-but-empty still gets its full-width track.
    let on = MetageneGrid::from_records(&records, 60, true).expect("grid");
    assert_eq!(on.nbins[3], 60);
}

#[test]
fn a_coding_gene_keeps_its_sites_when_a_non_coding_gene_overlaps_it() {
    // The ncRNA gene spans the whole coding one, so every coding site is also
    // inside it. Turning the track ON must not let it claim any of them —
    // without an overlap in the fixture the assertion could not fail either way.
    let mut records = two_exon_gene("G", Strand::Forward);
    records.push(non_coding_rec("NC", 900, 2300, Strand::Forward));

    for pos in [950, 1050, 2050, 2200] {
        assert_eq!(
            classify(&records, pos),
            classify_with_non_coding(&records, pos, true),
            "position {pos} moved when the ncRNA track was enabled"
        );
    }
    // The intron belongs to no coding feature, so there the ncRNA gene DOES
    // take the site once its track exists — the flag is doing something.
    assert_eq!(classify(&records, 1500), [0, 0, 0, 0]);
    assert_eq!(classify_with_non_coding(&records, 1500, true), [0, 0, 0, 1]);
}

///////////////////////////////
// Spliced relative position  //
///////////////////////////////

#[test]
fn relative_position_runs_along_the_spliced_feature() {
    // Two exons, 100 bp each: the intron 200..299 must consume no coordinate.
    let idx = FeatureIndex::from_features(&[feature(Strand::Forward, &[(100, 199), (300, 399)])]);
    let rel = |pos| idx.find(CHR, pos).expect("hit").relative_pos(pos);

    assert_eq!(rel(100), 0);
    assert_eq!(rel(199), 99);
    // First base of exon 2 is base 100 of the transcript, not base 200.
    assert_eq!(rel(300), 100);
    assert_eq!(rel(399), 199);
}

#[test]
fn reverse_strand_relative_position_runs_five_to_three() {
    // Same intervals, reverse strand: the 5' end is the HIGHEST coordinate.
    let idx = FeatureIndex::from_features(&[feature(Strand::Backward, &[(100, 199), (300, 399)])]);
    let rel = |pos| idx.find(CHR, pos).expect("hit").relative_pos(pos);

    assert_eq!(rel(399), 0);
    assert_eq!(rel(300), 99);
    assert_eq!(rel(199), 100);
    assert_eq!(rel(100), 199);

    // …and the first transcribed base bins at the front of the track.
    let hit = idx.find(CHR, 399).expect("hit");
    assert_eq!(hit.bin(399, 10), 0);
    assert_eq!(idx.find(CHR, 100).expect("hit").bin(100, 10), 9);
}

#[test]
fn bins_are_a_fraction_of_the_spliced_length() {
    let idx = FeatureIndex::from_features(&[feature(Strand::Forward, &[(100, 199), (300, 399)])]);
    let bin = |pos| idx.find(CHR, pos).expect("hit").bin(pos, 10);

    assert_eq!(bin(100), 0);
    // Base 100 of 200 is the middle of the transcript, though it is 3/4 of the
    // way across the genomic span.
    assert_eq!(bin(300), 5);
    assert_eq!(bin(399), 9);
}
