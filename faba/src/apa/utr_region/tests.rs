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

/// The one-block call: the read's outer span, which is what placement used to
/// be given for every read regardless of how it was spliced.
fn span(utr: &UtrRegion, start: i64, stop: i64) -> Option<SplicedCover> {
    utr.overlap_spliced_blocks([(start, stop)])
}

#[test]
fn a_single_block_read_is_charged_only_for_the_exons_it_covers() {
    let utr = three_exon(Strand::Forward);
    // 150..320 reaches 171 genomic bases but only 50 + 21 = 71 spliced ones.
    let cover = span(&utr, 150, 320).expect("touches two exons");
    assert_eq!(cover.x_rel, 51);
    assert_eq!(cover.len, 71);
    assert_ne!(
        cover.len,
        320 - 150 + 1,
        "genomic reach must not be the length"
    );
    // One block covers an unbroken run, so the 3' end is where x + l - 1 lands.
    assert_eq!(cover.three_prime_rel, cover.x_rel + cover.len - 1);
}

#[test]
fn a_single_block_read_is_oriented_from_the_high_end_on_the_reverse_strand() {
    let utr = three_exon(Strand::Backward);
    let cover = span(&utr, 150, 320).expect("touches two exons");
    // The read's 5'-most base is now its highest, 320.
    assert_eq!(Some(cover.x_rel), utr.spliced_offset(320));
    assert_eq!(cover.len, 71);
    assert_eq!(cover.three_prime_rel, cover.x_rel + cover.len - 1);
    assert_eq!(utr.genomic_from_spliced(cover.three_prime_rel), Some(150));
}

#[test]
fn a_read_that_lies_in_an_intron_is_dropped() {
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        assert_eq!(span(&utr, 220, 249), None, "intron 200..299");
        assert_eq!(span(&utr, 400, 450), None, "intron 350..499");
        assert_eq!(span(&utr, 600, 700), None, "past the UTR");
    }
}

/////////////////////////////
// Blocks, not outer spans  //
/////////////////////////////

/// The read the outer span got wrong: 1-based blocks 150..179 and 320..340, so
/// its gap 180..319 swallows 20 exonic bases (180..199) that it never aligned
/// to. Both blocks and span start at 150 and end at 340, so only the length —
/// the quantity APA spends on a poly-A position — can tell them apart.
const GAPPED_BLOCKS: [(i64, i64); 2] = [(150, 179), (320, 340)];

#[test]
fn a_gap_over_exonic_bases_is_not_charged_to_the_read() {
    let utr = three_exon(Strand::Forward);
    let cover = utr
        .overlap_spliced_blocks(GAPPED_BLOCKS)
        .expect("both blocks are exonic");

    // 30 + 21 aligned bases, against the 91 the outer span 150..340 claims.
    assert_eq!(cover.len, 51);
    assert_eq!(
        span(&utr, 150, 340).unwrap().len,
        91,
        "what the span claims"
    );

    assert_eq!(cover.x_rel, 51, "5'-most base is still 150");
    // The gap breaks the covered run, so the 3' end is past x + l - 1 and only
    // the block that reached it can say where it is.
    assert_eq!(cover.three_prime_rel, 141);
    assert_eq!(utr.genomic_from_spliced(cover.three_prime_rel), Some(340));
    assert!(cover.three_prime_rel > cover.x_rel + cover.len - 1);
}

#[test]
fn a_gapped_read_is_oriented_from_its_highest_base_on_the_reverse_strand() {
    let utr = three_exon(Strand::Backward);
    let cover = utr
        .overlap_spliced_blocks(GAPPED_BLOCKS)
        .expect("both blocks are exonic");

    assert_eq!(cover.len, 51);
    // Transcript 5' is now the read's highest base, 340, not its lowest.
    assert_eq!(cover.x_rel, 60);
    assert_eq!(Some(cover.x_rel), utr.spliced_offset(340));
    assert_eq!(cover.three_prime_rel, 150);
    assert_eq!(utr.genomic_from_spliced(cover.three_prime_rel), Some(150));
}

#[test]
fn a_gap_that_matches_an_annotated_intron_still_gives_the_span_answer() {
    // The common case, and the one the span was already right about: blocks
    // 150..199 and 300..320 skip exactly the intron 200..299.
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        let cover = utr
            .overlap_spliced_blocks([(150, 199), (300, 320)])
            .expect("both blocks are exonic");
        assert_eq!(Some(cover), span(&utr, 150, 320), "{strand:?} regressed");
        assert_eq!(cover.len, 71);
        // Nothing punched a hole, so the pA identity is the old one.
        assert_eq!(cover.three_prime_rel, cover.x_rel + cover.len - 1);
    }
}

#[test]
fn a_block_inside_an_intron_contributes_nothing() {
    for strand in [Strand::Forward, Strand::Backward] {
        let utr = three_exon(strand);
        // 220..249 is intronic, so the answer is the exonic block's alone.
        assert_eq!(
            utr.overlap_spliced_blocks([(150, 179), (220, 249)]),
            span(&utr, 150, 179),
        );
        // With every block in an intron there is no 3'UTR position to report.
        assert_eq!(
            utr.overlap_spliced_blocks([(220, 249), (400, 450)]),
            None,
            "{strand:?}"
        );
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
    let forward = span(&regions[0], 1201, 1300).unwrap();
    assert_eq!((forward.x_rel, forward.len), (201, 100));
    assert_eq!(regions[1].strand, Strand::Backward);
    let backward = span(&regions[1], 1201, 1300).unwrap();
    assert_eq!((backward.x_rel, backward.len), (701, 100));
}

////////////////////////////////
// End-to-end read extraction  //
////////////////////////////////

/// The query bases a CIGAR implies: `A` under a soft clip, `G` everywhere else.
///
/// The clip is the poly-A tail `extract_fragments_cached` calls a junction read
/// on, so spelling one in the CIGAR is enough to get one. Keeping the aligned
/// bases off `A` is what stops the internal-priming check from vetoing it.
fn query_bases(cigar: &rust_htslib::bam::record::CigarString) -> Vec<u8> {
    use rust_htslib::bam::record::Cigar;
    cigar
        .iter()
        .flat_map(|op| match *op {
            Cigar::SoftClip(len) => vec![b'A'; len as usize],
            Cigar::Match(len) | Cigar::Ins(len) | Cigar::Equal(len) | Cigar::Diff(len) => {
                vec![b'G'; len as usize]
            }
            _ => Vec::new(),
        })
        .collect()
}

/// Write a coordinate-sorted, indexed one-contig BAM. Each read is
/// `(0-based pos, CIGAR)`; the query is derived from the CIGAR.
fn write_bam(dir: &std::path::Path, reads: &[(i64, &str)]) -> String {
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
        for (i, (pos, cigar)) in reads.iter().enumerate() {
            let mut rec = Record::new();
            let cigar = CigarString::try_from(*cigar).unwrap();
            let seq = query_bases(&cigar);
            let qual = vec![40u8; seq.len()];
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
            (109, "10M"),
            // Spans the UTR's first intron: 1-based 150..199, gap, 300..320.
            (149, "50M100N21M"),
            // Wholly inside the intron 200..299: 1-based 220..249.
            (219, "30M"),
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
    let bam = write_bam(dir.path(), &[(149, "50M100N21M")]);

    let utr = three_exon(Strand::Backward);
    let frags = extract(&bam, &utr);

    assert_eq!(frags.len(), 1);
    // 5'-most covered base is now 320, whose spliced offset is 80.
    assert_eq!(frags[0].x, 80.0);
    assert_eq!(frags[0].l, 71.0);
}

/// The read whose `N` gap does not line up with the UTR's intron.
///
/// `30M140N21M` at 0-based 149 aligns 1-based 150..179 and 320..340 and skips
/// 180..319 — which includes exon 1's last 20 bases. Its outer span is the
/// 150..340 of the test above, so placing reads by span cannot see the
/// difference and charges the read the 20 bases it never aligned to.
#[test]
fn a_gap_that_misses_the_annotated_intron_is_charged_by_block_not_by_span() {
    let dir = tempfile::tempdir().unwrap();
    let bam = write_bam(dir.path(), &[(149, "30M140N21M")]);

    let utr = three_exon(Strand::Forward);
    let frags = extract(&bam, &utr);

    assert_eq!(frags.len(), 1);
    assert_eq!(frags[0].x, 51.0);
    // 30 + 21 aligned bases. The outer span 150..340 would have said 91.
    assert_eq!(frags[0].l, 51.0);
    assert_eq!(
        span(&utr, 150, 340).unwrap().len,
        91,
        "what the span claims"
    );
}

/// A junction read's pA site is where its blocks END, not where its charged
/// length runs out.
#[test]
fn a_gapped_junction_read_reports_the_pa_site_its_last_block_reached() {
    let dir = tempfile::tempdir().unwrap();
    // The same read, plus a 12bp poly-A soft clip past its last aligned base.
    let bam = write_bam(dir.path(), &[(149, "30M140N21M12S")]);

    let utr = three_exon(Strand::Forward);
    let frags = extract(&bam, &utr);

    assert_eq!(frags.len(), 1);
    assert!(frags[0].is_junction);
    assert_eq!(frags[0].r, 12.0);
    // Spliced offset of 340, the read's 3'-most aligned base. The gap broke the
    // covered run, so x + l - 1 = 101 stops 40 bases short of it — and offset
    // 101 is genomic 300, a base this read never touched.
    assert_eq!(frags[0].pa_site, Some(141.0));
    assert_eq!(frags[0].x + frags[0].l - 1.0, 101.0);
    assert_eq!(utr.genomic_from_spliced(141), Some(340));
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

//////////////////////////////////
// Aligned blocks off the CIGAR //
//////////////////////////////////

/// A mapped record carrying `cigar` at 0-based `pos`, query bases and all.
fn record_with_cigar(pos: i64, cigar: &str) -> rust_htslib::bam::Record {
    use rust_htslib::bam::{record::CigarString, Record};

    let cigar = CigarString::try_from(cigar).unwrap();
    let seq = query_bases(&cigar);
    let qual = vec![40u8; seq.len()];
    let mut rec = Record::new();
    rec.set(b"read", Some(&cigar), &seq, &qual);
    rec.set_tid(0);
    rec.set_pos(pos);
    rec.set_mapq(60);
    rec.unset_unmapped();
    rec
}

/// The blocks fragment extraction walks must be htslib's blocks, op for op.
///
/// `extract_fragments_cached` dropped `rec.aligned_blocks()`: it unpacks the CIGAR a second time.
/// Borrowing the CIGAR the poly-A checks unpacked is only safe while both walks agree everywhere.
/// `D` splits a read just as `N` does, and `I` closes a block without moving the reference.
/// Clips and pads emit nothing and move nothing.
#[test]
fn aligned_blocks_off_a_borrowed_cigar_match_htslibs_own() {
    use crate::data::poly_a_utils::aligned_blocks;
    use rust_htslib::bam::ext::BamRecordExtensions;

    let cases = [
        // Plain match, and the operations that are pure query-side padding.
        "100M",
        "10S80M10S",
        "5H95M",
        "5H10S80M10S5H",
        // Leading and trailing clips only on one side.
        "12S88M",
        "88M12S",
        // Insertions: no reference movement, but htslib still breaks the block.
        "40M5I55M",
        "10S40M5I50M10S",
        // Deletions split exactly like skips do — the easiest divergence to write.
        "40M5D55M",
        "40M100N60M",
        "40M5D55M200N30M",
        // Multiple skips, and a read that both clips and skips repeatedly.
        "30M100N30M200N40M",
        "10S30M100N30M200N30M10S",
        // `=`/`X` are match operations and must emit blocks like `M`.
        "50=50X",
        "20=5X10N30=15X",
        // Degenerate edges: a read that aligns nothing, and a leading skip.
        "100S",
        "50N100M",
    ];

    for cigar in cases {
        for pos in [0i64, 1, 149, 1_000_000] {
            let rec = record_with_cigar(pos, cigar);
            let expected: Vec<[i64; 2]> = rec.aligned_blocks().collect();
            let actual: Vec<[i64; 2]> = aligned_blocks(&rec.cigar(), rec.pos()).collect();
            assert_eq!(actual, expected, "{cigar} at pos {pos}");
        }
    }
}

/// The `D`/`N` split is the one htslib behaviour a hand-written walk tends to drop.
#[test]
fn a_deletion_splits_a_block_the_same_way_a_skip_does() {
    use crate::data::poly_a_utils::aligned_blocks;

    let rec = record_with_cigar(100, "40M5D55M");
    let blocks: Vec<[i64; 2]> = aligned_blocks(&rec.cigar(), rec.pos()).collect();
    // The 5 deleted reference bases 140..144 belong to no block.
    assert_eq!(blocks, vec![[100, 140], [145, 200]]);

    // An insertion moves nothing, so the two blocks abut at the same coordinate.
    let rec = record_with_cigar(100, "40M5I55M");
    let blocks: Vec<[i64; 2]> = aligned_blocks(&rec.cigar(), rec.pos()).collect();
    assert_eq!(blocks, vec![[100, 140], [140, 195]]);
}
