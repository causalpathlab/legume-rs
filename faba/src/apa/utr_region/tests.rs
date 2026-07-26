use super::*;
use genomic_data::gff::GeneType;

////////////////////////
// Spliced coordinate  //
////////////////////////

/// Exon lengths 100 / 50 / 50 = spliced length 200, over a genomic reach of
/// 450. Every assertion below is against the 200, which is the whole point.
const EXONS: [(i64, i64); 3] = [(100, 199), (300, 349), (500, 549)];
const SPLICED_LEN: usize = 200;

fn three_exon(strand: Strand) -> UtrRegion {
    UtrRegion {
        chr: "chr1".into(),
        start: EXONS[0].0,
        end: EXONS[2].1,
        strand,
        name: "TEST".into(),
        utr_length: SPLICED_LEN,
        exons: EXONS.to_vec(),
    }
}

/// Every base the exons cover, in genomic order.
fn exonic_positions() -> Vec<i64> {
    EXONS.iter().flat_map(|&(s, e)| s..=e).collect()
}

#[test]
fn spliced_offset_walks_the_exons_on_the_forward_strand() {
    let utr = three_exon(Strand::Forward);
    // Offset 1 at the lowest coordinate; each exon boundary is seamless.
    assert_eq!(utr.spliced_offset(100), Some(1));
    assert_eq!(utr.spliced_offset(199), Some(100));
    assert_eq!(utr.spliced_offset(300), Some(101));
    assert_eq!(utr.spliced_offset(349), Some(150));
    assert_eq!(utr.spliced_offset(500), Some(151));
    assert_eq!(utr.spliced_offset(549), Some(200));
}

#[test]
fn spliced_offset_counts_from_the_high_end_on_the_reverse_strand() {
    let utr = three_exon(Strand::Backward);
    // A reverse-strand transcript reads 5'->3' as the coordinate falls, so
    // UTR position 1 is the HIGHEST genomic base, not the lowest.
    assert_eq!(utr.spliced_offset(549), Some(1));
    assert_eq!(utr.spliced_offset(500), Some(50));
    assert_eq!(utr.spliced_offset(349), Some(51));
    assert_eq!(utr.spliced_offset(300), Some(100));
    assert_eq!(utr.spliced_offset(199), Some(101));
    assert_eq!(utr.spliced_offset(100), Some(200));
}

#[test]
fn genomic_from_spliced_inverts_spliced_offset_at_every_exonic_base() {
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        for pos in exonic_positions() {
            let offset = utr
                .spliced_offset(pos)
                .unwrap_or_else(|| panic!("{} is exonic", pos));
            assert_eq!(
                utr.genomic_from_spliced(offset),
                Some(pos),
                "round trip broke at {}",
                pos
            );
        }
        // And the other way: every offset lands on a base that maps back to it.
        for offset in 1..=SPLICED_LEN as i64 {
            let pos = utr.genomic_from_spliced(offset).expect("offset in range");
            assert_eq!(utr.spliced_offset(pos), Some(offset));
        }
    }
}

#[test]
fn an_intron_has_no_spliced_offset() {
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        // Introns 200..299 and 350..499, plus the flanks: none of these bases
        // is part of the 3'UTR, so none of them has a position within it.
        for pos in [99, 200, 250, 299, 350, 499, 550] {
            assert_eq!(utr.spliced_offset(pos), None, "{} is not exonic", pos);
        }
    }
}

#[test]
fn genomic_from_spliced_rejects_offsets_outside_the_utr() {
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        assert_eq!(utr.genomic_from_spliced(0), None);
        assert_eq!(utr.genomic_from_spliced(-1), None);
        assert_eq!(utr.genomic_from_spliced(SPLICED_LEN as i64 + 1), None);
    }
}

/////////////////////
// Read placement  //
/////////////////////

#[test]
fn overlap_spliced_charges_a_read_only_for_the_exons_it_covers() {
    let utr = three_exon(Strand::Forward);
    // 150..320 reaches 171 genomic bases but only 50 + 21 = 71 spliced ones.
    let (x_rel, len) = utr.overlap_spliced(150, 320).expect("touches two exons");
    assert_eq!(x_rel, 51);
    assert_eq!(len, 71);
    assert_ne!(len, 320 - 150 + 1, "genomic reach must not be the length");
}

#[test]
fn overlap_spliced_orients_a_spanning_read_from_the_high_end_on_the_reverse_strand() {
    let utr = three_exon(Strand::Backward);
    let (x_rel, len) = utr.overlap_spliced(150, 320).expect("touches two exons");
    // The read's 5'-most base is now its highest, 320.
    assert_eq!(Some(x_rel), utr.spliced_offset(320));
    assert_eq!(len, 71);
    // The 3' end of a spliced read is x + l - 1 — the identity `fragment.rs`
    // reports a junction read's pA position with.
    assert_eq!(utr.genomic_from_spliced(x_rel + len - 1), Some(150));
}

#[test]
fn overlap_spliced_drops_a_read_that_lies_in_an_intron() {
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        assert_eq!(utr.overlap_spliced(220, 249), None, "intron 200..299");
        assert_eq!(utr.overlap_spliced(400, 450), None, "intron 350..499");
        assert_eq!(utr.overlap_spliced(600, 700), None, "past the UTR");
    }
}

#[test]
fn alpha_to_genomic_range_maps_alpha_through_the_exons() {
    let utr = three_exon(Strand::Forward);
    // Alpha 101 is the first base of exon 2; a linear span map would have put
    // it at 100 + 101 = 201, inside the intron.
    let (start, stop) = utr.alpha_to_genomic_range(101.0, 10.0);
    assert_eq!((start, stop), (290, 310));

    let utr = three_exon(Strand::Backward);
    // Reverse: alpha 101 counts down from 549 and lands on 199.
    let (start, stop) = utr.alpha_to_genomic_range(101.0, 10.0);
    assert_eq!((start, stop), (189, 209));
}

#[test]
fn alpha_to_genomic_range_clamps_alpha_that_drifted_off_the_utr() {
    let utr = three_exon(Strand::Forward);
    // EM can nudge alpha a hair past either end; a site is worth keeping at the
    // boundary rather than losing to an out-of-range map.
    assert_eq!(utr.alpha_to_genomic_range(0.0, 0.0).0, 100);
    assert_eq!(
        utr.alpha_to_genomic_range(SPLICED_LEN as f64 + 5.0, 0.0).0,
        549
    );
}

/////////////////////////
// GFF-built regions   //
/////////////////////////

fn gff_rec(
    seqname: &str,
    gene: &str,
    feature_type: FeatureType,
    start: i64,
    stop: i64,
    strand: Strand,
) -> GffRecord {
    GffRecord {
        seqname: seqname.into(),
        feature_type,
        start,
        stop,
        strand,
        gene_id: GeneId::Ensembl(gene.into()),
        gene_name: GeneSymbol::Symbol(gene.into()),
        gene_type: GeneType::CodingGene,
    }
}

/// A two-CDS-exon coding gene whose 3' end carries TWO UTR exons split by an
/// intron, and a 5'UTR record on the other side of the start codon. Only the
/// generic `UTR` feature is used, so the codon-distance rule has to do the
/// sorting — which is what GENCODE actually forces.
fn spliced_utr_gene(strand: Strand) -> Vec<GffRecord> {
    let (start_codon, stop_codon) = match strand {
        Strand::Forward => ((1000, 1002), (2098, 2100)),
        Strand::Backward => ((2098, 2100), (1000, 1002)),
    };
    vec![
        gff_rec("chr1", "ENSGX", FeatureType::CDS, 1000, 1100, strand),
        gff_rec("chr1", "ENSGX", FeatureType::CDS, 2000, 2100, strand),
        gff_rec(
            "chr1",
            "ENSGX",
            FeatureType::StartCodon,
            start_codon.0,
            start_codon.1,
            strand,
        ),
        gff_rec(
            "chr1",
            "ENSGX",
            FeatureType::StopCodon,
            stop_codon.0,
            stop_codon.1,
            strand,
        ),
        gff_rec("chr1", "ENSGX", FeatureType::UTR, 700, 800, strand),
        gff_rec("chr1", "ENSGX", FeatureType::UTR, 900, 999, strand),
        gff_rec("chr1", "ENSGX", FeatureType::UTR, 2101, 2300, strand),
        gff_rec("chr1", "ENSGX", FeatureType::UTR, 2500, 2600, strand),
    ]
}

#[test]
fn a_gff_region_is_the_utr_exons_not_the_span_over_them() {
    let regions = build_utr_regions_from_gff(&spliced_utr_gene(Strand::Forward)).unwrap();
    assert_eq!(regions.len(), 1);
    let utr = &regions[0];

    assert_eq!(utr.exons, vec![(2101, 2300), (2500, 2600)]);
    // 200 + 101 spliced, against a 500bp span — the span is what the union
    // gene model used to report as the UTR length.
    assert_eq!(utr.utr_length, 301);
    assert_eq!(utr.end - utr.start + 1, 500);
    assert_eq!(utr.name.as_ref(), "ENSGX_ENSGX");

    // No exon may reach back into CDS (1000..1100, 2000..2100).
    for &(s, e) in utr.exons.iter() {
        assert!(s > 2100, "exon {}..{} overlaps CDS", s, e);
    }
}

#[test]
fn a_reverse_strand_gff_region_takes_the_utr_below_the_stop_codon() {
    let regions = build_utr_regions_from_gff(&spliced_utr_gene(Strand::Backward)).unwrap();
    assert_eq!(regions.len(), 1);
    let utr = &regions[0];

    // Stop codon at 1000..1002, so the 3'UTR is the LOW side of the gene.
    assert_eq!(utr.exons, vec![(700, 800), (900, 999)]);
    assert_eq!(utr.utr_length, 201);
    assert_eq!(utr.strand, Strand::Backward);
    // Its 5'-most base is the highest one.
    assert_eq!(utr.spliced_offset(999), Some(1));
}

#[test]
fn abutting_utr_records_merge_into_one_exon() {
    // Two records meeting base-to-base are one uninterrupted stretch; leaving
    // them split would put a seam where the transcript has none.
    let mut records = spliced_utr_gene(Strand::Forward);
    records.push(gff_rec(
        "chr1",
        "ENSGX",
        FeatureType::UTR,
        2301,
        2400,
        Strand::Forward,
    ));
    let regions = build_utr_regions_from_gff(&records).unwrap();
    assert_eq!(regions[0].exons, vec![(2101, 2400), (2500, 2600)]);
    assert_eq!(regions[0].utr_length, 401);
}

#[test]
fn par_copies_on_chrx_and_chry_stay_separate_regions() {
    // `parse_ensembl_id` drops the `_PAR_Y` suffix, so both copies arrive under
    // one gene id. Pooling on the id alone would splice them into one 400bp UTR
    // sitting on whichever chromosome was seen first.
    let records = vec![
        gff_rec(
            "chrX",
            "ENSGPAR",
            FeatureType::ThreePrimeUTR,
            1000,
            1199,
            Strand::Forward,
        ),
        gff_rec(
            "chrY",
            "ENSGPAR",
            FeatureType::ThreePrimeUTR,
            5000,
            5199,
            Strand::Forward,
        ),
    ];
    let regions = build_utr_regions_from_gff(&records).unwrap();

    assert_eq!(regions.len(), 2, "one region per chromosome");
    for region in regions.iter() {
        assert_eq!(region.exons.len(), 1);
        assert_eq!(region.utr_length, 200);
    }
    assert_eq!(regions[0].chr.as_ref(), "chrX");
    assert_eq!(regions[0].exons, vec![(1000, 1199)]);
    assert_eq!(regions[1].chr.as_ref(), "chrY");
    assert_eq!(regions[1].exons, vec![(5000, 5199)]);

    // The two would otherwise share a name, hence a `site_id` row.
    assert_ne!(regions[0].name, regions[1].name);
}

///////////////////
// The BED path  //
///////////////////

fn write_bed(lines: &[&str]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::with_suffix(".bed").unwrap();
    for line in lines {
        writeln!(f, "{}", line).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn a_bed_region_keeps_the_lengths_and_offsets_it_always_had() {
    let f = write_bed(&["chr1\t1000\t2000\t+", "chr1\t1000\t2000\t-"]);
    let regions = load_utr_regions_from_bed(f.path().to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 2);

    for region in regions.iter() {
        // BED is half-open, so the block is `end - start` long — unchanged.
        assert_eq!(region.utr_length, 1000);
        // One unspliced block, in the 1-based inclusive frame `exons` is in.
        assert_eq!(region.exons, vec![(1001, 2000)]);
        assert_eq!(region.to_bed().start, 1000);
        assert_eq!(region.to_bed().stop, 2000);
    }

    // A read at htslib 0-based [1200, 1300) maps exactly where the old
    // span arithmetic put it: `x = ref_start - bed_start + 1` on the forward
    // strand and `x = bed_end - ref_end + 1` on the reverse, length 100 either way.
    assert_eq!(regions[0].strand, Strand::Forward);
    assert_eq!(regions[0].overlap_spliced(1201, 1300), Some((201, 100)));
    assert_eq!(regions[1].strand, Strand::Backward);
    assert_eq!(regions[1].overlap_spliced(1201, 1300), Some((701, 100)));
}

////////////////////////////////
// End-to-end read extraction  //
////////////////////////////////

/// Write a coordinate-sorted, indexed one-contig BAM. Each read is
/// `(0-based pos, CIGAR, query length)`.
fn write_bam(dir: &std::path::Path, reads: &[(i64, &str, usize)]) -> String {
    use rust_htslib::bam::{self, header::HeaderRecord, record::CigarString, Header, Record};

    let path = dir.join("reads.bam");
    let path_str = path.to_str().unwrap().to_string();

    let mut header = Header::new();
    let mut hd = HeaderRecord::new(b"HD");
    hd.push_tag(b"VN", "1.6").push_tag(b"SO", "coordinate");
    header.push_record(&hd);
    let mut sq = HeaderRecord::new(b"SQ");
    sq.push_tag(b"SN", "chr1").push_tag(b"LN", 10_000);
    header.push_record(&sq);

    {
        let mut writer = bam::Writer::from_path(&path, &header, bam::Format::Bam).unwrap();
        for (i, (pos, cigar, qlen)) in reads.iter().enumerate() {
            let mut rec = Record::new();
            let cigar = CigarString::try_from(*cigar).unwrap();
            let seq = vec![b'A'; *qlen];
            let qual = vec![40u8; *qlen];
            rec.set(format!("read{}", i).as_bytes(), Some(&cigar), &seq, &qual);
            rec.set_tid(0);
            rec.set_pos(*pos);
            rec.set_mapq(60);
            // `Record::new` starts out unmapped, which `extract_fragments_cached`
            // (rightly) drops before it ever reaches the coordinate map.
            rec.unset_unmapped();
            rec.push_aux(b"CB", rust_htslib::bam::record::Aux::String("CELL1"))
                .unwrap();
            rec.push_aux(
                b"UB",
                rust_htslib::bam::record::Aux::String(match i {
                    0 => "UMIA",
                    1 => "UMIB",
                    _ => "UMIC",
                }),
            )
            .unwrap();
            writer.write(&rec).unwrap();
        }
    }

    bam::index::build(&path, None, bam::index::Type::Bai, 1).unwrap();
    path_str
}

fn extract(bam_path: &str, utr: &UtrRegion) -> Vec<crate::apa::fragment::FragmentRecord> {
    let polya = crate::apa::fragment::PolyAFilterParams {
        min_tail: 10,
        max_non_at: 2,
        internal_prime_window: 10,
        internal_prime_count: 8,
    };
    let mut cache = crate::data::bam_io::BamReaderCache::new();
    crate::apa::fragment::extract_fragments_cached(
        &mut cache, bam_path, utr, b"CB", b"UB", &polya, 0,
    )
    .unwrap()
}

#[test]
fn an_intronic_read_yields_no_fragment_and_a_spanning_one_reports_spliced_length() {
    let dir = tempfile::tempdir().unwrap();
    let bam = write_bam(
        dir.path(),
        &[
            // Wholly inside exon 1: 1-based 110..119.
            (109, "10M", 10),
            // Spans the UTR's first intron: 1-based 150..199, gap, 300..320.
            (149, "50M100N21M", 71),
            // Wholly inside the intron 200..299: 1-based 220..249.
            (219, "30M", 30),
        ],
    );

    let utr = three_exon(Strand::Forward);
    let frags = extract(&bam, &utr);

    // The intronic read is fetched by the bounding-box window and must then be
    // dropped: it touches no 3'UTR base, so it has no 3'UTR position.
    assert_eq!(
        frags.len(),
        2,
        "the intronic read must not become a fragment"
    );

    let mut by_x: Vec<(f32, f32)> = frags.iter().map(|f| (f.x, f.l)).collect();
    by_x.sort_by(|a, b| a.0.total_cmp(&b.0));

    assert_eq!(by_x[0], (11.0, 10.0), "exonic read");
    // 171 genomic bases, 71 spliced: the intron the read jumped is not length
    // the model may spend on a poly-A position.
    assert_eq!(by_x[1], (51.0, 71.0), "intron-spanning read");
}

#[test]
fn a_reverse_strand_read_is_measured_from_the_high_end_of_the_spliced_utr() {
    let dir = tempfile::tempdir().unwrap();
    let bam = write_bam(dir.path(), &[(149, "50M100N21M", 71)]);

    let utr = three_exon(Strand::Backward);
    let frags = extract(&bam, &utr);

    assert_eq!(frags.len(), 1);
    // 5'-most covered base is now 320, whose spliced offset is 80.
    assert_eq!(frags[0].x, 80.0);
    assert_eq!(frags[0].l, 71.0);
}

/// The reported poly(A) point must lie inside the range reported beside it.
///
/// `genomic_alpha` and `genomic_start`/`genomic_stop` land in the same parquet
/// row, and were computed two different ways: the range through the exons, the
/// point by a linear `start + alpha`. On a spliced UTR those disagree — the
/// point could sit in an intron, outside its own range — so both now go through
/// `alpha_to_genomic`, and this pins that they cannot drift apart again.
#[test]
fn the_reported_point_lies_within_the_reported_range() {
    for strand in [Strand::Forward, Strand::Backward] {
        // Two exons either side of a 400 bp intron.
        let utr = UtrRegion {
            chr: "chr1".into(),
            start: 1000,
            end: 1699,
            strand,
            name: "G".into(),
            utr_length: 300,
            exons: vec![(1000, 1149), (1550, 1699)],
        };
        let in_exon = |p: i64| utr.exons.iter().any(|&(s, e)| p >= s && p <= e);
        for alpha in 1..=utr.utr_length as i64 {
            let point = utr.alpha_to_genomic(alpha as f64);
            let (lo, hi) = utr.alpha_to_genomic_range(alpha as f64, 25.0);
            assert!(
                point >= lo && point <= hi,
                "{strand:?} alpha={alpha}: point {point} outside its own range {lo}..{hi}"
            );
            assert!(
                in_exon(point),
                "{strand:?} alpha={alpha}: point {point} landed in the intron"
            );
        }
    }
}
