use super::*;
use genomic_data::transcript::build_transcript_models;

//////////////
// Fixtures  //
//////////////

const CHR: &str = "chr1";

fn rec(
    tx: &str,
    gene: &str,
    feature_type: FeatureType,
    start: i64,
    stop: i64,
    strand: Strand,
) -> GffRecord {
    GffRecord {
        seqname: CHR.into(),
        feature_type,
        start,
        stop,
        strand,
        gene_id: GeneId::Ensembl(gene.into()),
        gene_name: GeneSymbol::Symbol(gene.into()),
        gene_type: GeneType::CodingGene,
        transcript_id: TranscriptId::Ensembl(tx.into()),
    }
}

fn non_coding_rec(gene: &str, start: i64, stop: i64, strand: Strand) -> GffRecord {
    GffRecord {
        gene_type: GeneType::LincRNA,
        transcript_id: TranscriptId::Missing,
        ..rec("none", gene, FeatureType::Gene, start, stop, strand)
    }
}

/// A two-exon coding transcript: exons 1000..1199 and 1500..1699, CDS
/// 1100..1599 with a separate stop codon, leaving a 100 nt 5'UTR and a 97 nt
/// 3'UTR on the forward strand.
fn two_exon_tx(tx: &str, gene: &str, strand: Strand) -> Vec<GffRecord> {
    let stop_codon = match strand {
        Strand::Forward => (1600, 1602),
        Strand::Backward => (1097, 1099),
    };
    vec![
        rec(tx, gene, FeatureType::Exon, 1000, 1199, strand),
        rec(tx, gene, FeatureType::Exon, 1500, 1699, strand),
        rec(tx, gene, FeatureType::CDS, 1100, 1199, strand),
        rec(tx, gene, FeatureType::CDS, 1500, 1599, strand),
        rec(
            tx,
            gene,
            FeatureType::StopCodon,
            stop_codon.0,
            stop_codon.1,
            strand,
        ),
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

/// Build the index the way `run_metagene` does, then place sites on it.
fn place(records: &[GffRecord], positions: &[i64]) -> Vec<SiteAssignment> {
    let models = elect_longest_isoform(build_transcript_models(records));
    let nc = non_coding_bodies(records);
    let index = RegionIndex::build(&models, &nc);
    let sites: Vec<GenomicSite> = positions.iter().map(|&p| site(p)).collect();
    assign_sites(&sites, &index).0
}

/// `[5'UTR, CDS, 3'UTR, ncRNA]` totals for one site.
fn classify(records: &[GffRecord], pos_1based: i64) -> [usize; 4] {
    let mut out = [0usize; 4];
    for a in place(records, &[pos_1based]) {
        out[a.region] += 1;
    }
    out
}

/////////////////////////
// The interval index   //
/////////////////////////

fn feature(strand: Strand, intervals: &[(i64, i64)]) -> TranscriptModel {
    let total: i64 = intervals.iter().map(|&(s, e)| e - s + 1).sum();
    TranscriptModel {
        gene_id: GeneId::Ensembl("G".into()),
        gene_name: GeneSymbol::Symbol("G".into()),
        transcript_id: TranscriptId::Ensembl("T".into()),
        seqname: CHR.into(),
        strand,
        utr5: vec![],
        cds: intervals.to_vec(),
        utr3: vec![],
        utr5_size: 0,
        cds_size: total,
        utr3_size: 0,
    }
}

fn hits(idx: &RegionIndex, pos: i64) -> Vec<IndexedInterval> {
    let mut v = Vec::new();
    idx.find_all(CHR, pos, &mut v);
    v
}

#[test]
fn find_reaches_past_a_shorter_later_interval() {
    // Sorted by start: [(100, 5000), (200, 300)]. Walking back from the
    // rightmost start <= 1000 hits (200, 300) first; `start < position` is no
    // reason to stop, because (100, 5000) still contains 1000.
    let idx = RegionIndex::build(
        &[
            feature(Strand::Forward, &[(100, 5000)]),
            feature(Strand::Forward, &[(200, 300)]),
        ],
        &[],
    );

    let h = hits(&idx, 1000);
    assert_eq!(h.len(), 1);
    assert_eq!((h[0].start, h[0].stop), (100, 5000));

    assert_eq!(hits(&idx, 5001).len(), 0);
    assert_eq!(hits(&idx, 99).len(), 0);
    let mut v = Vec::new();
    idx.find_all("chrX", 1000, &mut v);
    assert!(v.is_empty());
}

#[test]
fn find_scans_back_over_several_non_containing_intervals() {
    let idx = RegionIndex::build(
        &[
            feature(Strand::Forward, &[(100, 5000)]),
            feature(Strand::Forward, &[(200, 300)]),
            feature(Strand::Forward, &[(400, 500)]),
            feature(Strand::Forward, &[(600, 700)]),
        ],
        &[],
    );
    let h = hits(&idx, 4000);
    assert_eq!(h.len(), 1);
    assert_eq!((h[0].start, h[0].stop), (100, 5000));

    // …and the max-stop bound still lets it give up: nothing covers 5500.
    assert_eq!(hits(&idx, 5500).len(), 0);
}

#[test]
fn overlapping_transcripts_each_take_the_site() {
    // MetaPlotR emits one row per site and transcript. A site inside two
    // elected transcripts must be counted in both, not just the first found.
    let idx = RegionIndex::build(
        &[
            feature(Strand::Forward, &[(100, 5000)]),
            feature(Strand::Forward, &[(900, 1100)]),
        ],
        &[],
    );
    assert_eq!(hits(&idx, 1000).len(), 2);
}

///////////////////////////////
// Spliced relative position  //
///////////////////////////////

#[test]
fn relative_position_runs_along_the_spliced_feature() {
    // Two exons, 100 bp each: the intron 200..299 must consume no coordinate.
    let idx = RegionIndex::build(&[feature(Strand::Forward, &[(100, 199), (300, 399)])], &[]);
    let rel = |pos| hits(&idx, pos)[0].relative_pos(pos);

    assert_eq!(rel(100), 0);
    assert_eq!(rel(199), 99);
    // First base of exon 2 is base 100 of the transcript, not base 200.
    assert_eq!(rel(300), 100);
    assert_eq!(rel(399), 199);
}

#[test]
fn reverse_strand_relative_position_runs_five_to_three() {
    let idx = RegionIndex::build(&[feature(Strand::Backward, &[(100, 199), (300, 399)])], &[]);
    let rel = |pos| hits(&idx, pos)[0].relative_pos(pos);

    assert_eq!(rel(399), 0);
    assert_eq!(rel(300), 99);
    assert_eq!(rel(199), 100);
    assert_eq!(rel(100), 199);

    assert_eq!(hits(&idx, 399)[0].place(0, 399).bin(10), 0);
    assert_eq!(hits(&idx, 100)[0].place(0, 100).bin(10), 9);
}

#[test]
fn bins_are_a_fraction_of_the_spliced_length() {
    let idx = RegionIndex::build(&[feature(Strand::Forward, &[(100, 199), (300, 399)])], &[]);
    let bin = |pos| hits(&idx, pos)[0].place(0, pos).bin(10);

    assert_eq!(bin(100), 0);
    // Base 100 of 200 is the middle of the transcript, though it is 3/4 of the
    // way across the genomic span.
    assert_eq!(bin(300), 5);
    assert_eq!(bin(399), 9);
}

//////////////////////////
// Region assignment     //
//////////////////////////

#[test]
fn regions_are_disjoint_so_no_priority_order_is_needed() {
    // Under the gene-union model a site was tested 5'UTR -> CDS -> 3'UTR and an
    // oversized CDS could claim a 3'UTR site. Within one transcript the three
    // regions cannot overlap, so each site has exactly one home.
    for strand in [Strand::Forward, Strand::Backward] {
        let records = two_exon_tx("ENST1", "G", strand);
        for pos in [1050, 1150, 1550, 1650] {
            let total: usize = classify(&records, pos).iter().sum();
            assert!(total <= 1, "position {pos} landed twice on {strand:?}");
        }
    }
}

#[test]
fn intronic_site_between_cds_exons_is_not_called_cds() {
    for strand in [Strand::Forward, Strand::Backward] {
        let records = two_exon_tx("ENST1", "G", strand);
        // 1300 sits in the intron between exons 1000..1199 and 1500..1699.
        assert_eq!(classify(&records, 1300), [0, 0, 0, 0], "{strand:?}");
        // The coding exons themselves are still CDS.
        assert_eq!(classify(&records, 1150), [0, 1, 0, 0], "{strand:?}");
        assert_eq!(classify(&records, 1550), [0, 1, 0, 0], "{strand:?}");
    }
}

#[test]
fn stop_codon_bins_as_the_last_cds_bin_not_the_first_utr3_bin() {
    // The whole reason the coding extent is widened over the stop codon.
    let records = two_exon_tx("ENST1", "G", Strand::Forward);
    assert_eq!(classify(&records, 1601), [0, 1, 0, 0]);

    let a = place(&records, &[1601]);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].region, CDS);
    assert_eq!(
        a[0].bin(10),
        9,
        "the stop codon is at the 3' end of the CDS"
    );
}

#[test]
fn rel_location_spans_zero_to_three() {
    let records = two_exon_tx("ENST1", "G", Strand::Forward);
    let at = |p: i64| place(&records, &[p])[0].rel_location();

    // First 5'UTR base is 0.0, first CDS base 1.0, first 3'UTR base 2.0.
    assert!((at(1000) - 0.0).abs() < 1e-9);
    assert!((at(1100) - 1.0).abs() < 1e-9);
    assert!((at(1603) - 2.0).abs() < 1e-9);
    // …and the last 3'UTR base stays below 3.
    assert!(at(1699) < 3.0 && at(1699) > 2.9);
}

#[test]
fn reverse_strand_rel_location_also_runs_five_to_three() {
    let records = two_exon_tx("ENST1", "G", Strand::Backward);
    let at = |p: i64| place(&records, &[p])[0].rel_location();
    // On the reverse strand the 5'UTR is the HIGH end, and every region reads
    // 5'->3' as the genomic coordinate DECREASES. So the transcript starts at
    // 1699 and the 3'UTR's first base is its highest coordinate, 1096 — the
    // low end, 1000, is where the transcript ENDS.
    assert!((at(1699) - 0.0).abs() < 1e-9);
    assert!((at(1096) - 2.0).abs() < 1e-9);
    assert!(at(1000) > 2.9 && at(1000) < 3.0);
}

//////////////////////////
// Isoform election      //
//////////////////////////

#[test]
fn single_isoform_genes_are_kept() {
    // MetaPlotR's published `dist[duplicated(gene_name), ]` drops these.
    let records = two_exon_tx("ENST1", "G", Strand::Forward);
    assert_eq!(classify(&records, 1150), [0, 1, 0, 0]);
}

#[test]
fn a_site_only_on_the_losing_isoform_is_dropped_under_longest() {
    // ENST2 is longer and does not reach 1750; ENST1 does. Electing the longest
    // isoform means the site at 1750 has nowhere to go.
    let mut records = two_exon_tx("ENST1", "G", Strand::Forward);
    records.push(rec(
        "ENST1",
        "G",
        FeatureType::Exon,
        1700,
        1800,
        Strand::Forward,
    ));
    records.extend(vec![
        rec("ENST2", "G", FeatureType::Exon, 1000, 2500, Strand::Forward),
        rec("ENST2", "G", FeatureType::CDS, 1100, 1599, Strand::Forward),
        rec(
            "ENST2",
            "G",
            FeatureType::StopCodon,
            1600,
            1602,
            Strand::Forward,
        ),
    ]);

    // ENST2 (1501 nt) beats ENST1, and covers 1750 itself.
    let a = place(&records, &[1750]);
    assert_eq!(a.len(), 1);
    assert_eq!(
        a[0].model,
        Some(0),
        "only one transcript survives the election, so index 0"
    );
}

//////////////////////////
// Scale factors        //
//////////////////////////

#[test]
fn scale_factors_are_site_weighted_not_transcript_weighted() {
    // Two transcripts with very different 3'UTRs. The one with the SMALL 3'UTR
    // carries nine sites, the one with the large 3'UTR carries one. MetaPlotR
    // takes its medians over the site table, so the median must follow the
    // nine-site transcript, not sit between the two transcripts.
    let models = vec![sized_model(100, 1000, 100), sized_model(100, 1000, 9000)];
    let mk = |model: u32| SiteAssignment {
        site: 0,
        model: Some(model),
        region: UTR3,
        rel: 0,
        total_len: 1,
    };
    let mut assignments: Vec<SiteAssignment> = (0..9).map(|_| mk(0)).collect();
    assignments.push(mk(1));

    let sf = scale_factors(&assignments, &models).expect("scale factors");
    assert_eq!(
        sf.median()[UTR3],
        100.0,
        "a transcript-weighted median would be 4550"
    );
    assert!((sf.utr3_sf - 0.1).abs() < 1e-9);
}

#[test]
fn scale_factors_need_a_coding_assignment() {
    let models = vec![sized_model(100, 0, 100)];
    let assignments = vec![SiteAssignment {
        site: 0,
        model: Some(0),
        region: UTR3,
        rel: 0,
        total_len: 1,
    }];
    assert!(scale_factors(&assignments, &models).is_err());
}

//////////////////////////
// Bin allocation       //
//////////////////////////

#[test]
fn bins_follow_the_medians_and_always_sum_to_n() {
    // A CDS twice either UTR: half the bins, a quarter each.
    assert_eq!(allocate_bins(200, &[100, 200, 100]), [50, 100, 50]);
    // Non-dividing totals still sum exactly.
    for n in [7usize, 57, 199, 200, 1000] {
        for m in [[155, 1026, 1720], [1, 1, 1], [3, 100, 7]] {
            let got = allocate_bins(n, &m);
            assert_eq!(got.iter().sum::<usize>(), n, "n={n} m={m:?} got={got:?}");
        }
    }
}

#[test]
fn a_region_with_no_length_gets_no_bins() {
    assert_eq!(allocate_bins(100, &[0, 100, 100]), [0, 50, 50]);
}

//////////////////////////
// Output                //
//////////////////////////

/// A model carrying nothing but its three region sizes, for the median tests.
fn sized_model(utr5: i64, cds: i64, utr3: i64) -> TranscriptModel {
    TranscriptModel {
        utr5_size: utr5,
        cds_size: cds,
        utr3_size: utr3,
        ..feature(Strand::Forward, &[(1, 1)])
    }
}

fn histogram(counts: [Vec<usize>; 4], sf5: f64, sf3: f64) -> GeneFeatureHistogram {
    GeneFeatureHistogram {
        counts,
        scale: ScaleFactors {
            twice_median: [310, 2052, 3440],
            utr5_sf: sf5,
            utr3_sf: sf3,
        },
    }
}

#[test]
fn bin_edges_reproduce_metaplotr_rescaling() {
    let h = histogram([vec![0; 2], vec![0; 2], vec![0; 2], vec![]], 0.15, 1.68);
    // The CDS keeps width 1 and spans 1..2; each UTR is scaled to its median
    // size relative to the CDS and butts against it.
    assert!((h.bin_edges(UTR5, 0).0 - 0.85).abs() < 1e-9);
    assert!((h.bin_edges(UTR5, 1).1 - 1.0).abs() < 1e-9);
    assert!((h.bin_edges(CDS, 0).0 - 1.0).abs() < 1e-9);
    assert!((h.bin_edges(CDS, 1).1 - 2.0).abs() < 1e-9);
    assert!((h.bin_edges(UTR3, 0).0 - 2.0).abs() < 1e-9);
    assert!((h.bin_edges(UTR3, 1).1 - 3.68).abs() < 1e-9);
}

#[test]
fn tsv_keeps_its_first_three_columns_and_integrates_to_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mg.tsv");
    let path = path.to_str().unwrap();

    let h = histogram([vec![1, 3], vec![10, 20], vec![5, 1], vec![]], 0.15, 1.68);
    h.to_tsv(path).expect("write");

    let text = std::fs::read_to_string(path).expect("read");
    let mut lines = text.lines();
    assert_eq!(
        lines.next().unwrap(),
        "#feature\tgenomic_bin\tcount\tbin_start\tbin_end\tfrac\tdensity",
        "the first three columns are an output contract"
    );

    // density integrates to 1 over the coding regions.
    let mut integral = 0.0;
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let (lo, hi): (f64, f64) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        integral += f[6].parse::<f64>().unwrap() * (hi - lo);
    }
    assert!(
        (integral - 1.0).abs() < 1e-4,
        "density integrated to {integral}"
    );
}

//////////////////////////
// Non-coding track     //
//////////////////////////

#[test]
fn the_grid_sizes_all_four_tracks_together() {
    // All four widths come from one constructor. They were split once before,
    // with the ncRNA width set by its own statement, and this is the unit that
    // can observe them agreeing.
    let scale = ScaleFactors {
        twice_median: [200, 400, 200],
        utr5_sf: 0.5,
        utr3_sf: 0.5,
    };

    let off = BinGrid::new(100, &scale, false).0;
    assert_eq!(off, [25, 50, 25, 0], "no ncRNA track unless asked for");
    assert_eq!(off[..3].iter().sum::<usize>(), 100);

    let on = BinGrid::new(100, &scale, true).0;
    assert_eq!(
        on,
        [25, 50, 25, 100],
        "ncRNA gets the whole budget, own axis"
    );
    // Asking for the track and getting it is the ONLY difference: the three
    // coding widths must not move when the flag flips.
    assert_eq!(off[..3], on[..3]);
}

#[test]
fn a_non_coding_gene_is_not_a_coding_transcript() {
    let records = vec![non_coding_rec("NC", 5000, 6000, Strand::Forward)];
    assert!(build_transcript_models(&records).is_empty());
}

#[test]
fn non_coding_genes_keep_whole_gene_boundaries() {
    let records = vec![non_coding_rec("NC", 5000, 6000, Strand::Forward)];
    assert_eq!(classify(&records, 5500), [0, 0, 0, 1]);
    assert_eq!(classify(&records, 4999), [0, 0, 0, 0]);
}
