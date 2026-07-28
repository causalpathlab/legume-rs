use super::*;

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

/// A two-exon coding transcript.
///
/// Exons 1000..1199 and 1500..1699 (400 nt spliced). The coding sequence runs
/// 1100..1599 with its stop codon written separately, GENCODE-style, at the 3'
/// end. On the forward strand that leaves a 100 nt 5'UTR and a 100 nt 3'UTR.
fn two_exon_tx(tx: &str, gene: &str, strand: Strand) -> Vec<GffRecord> {
    // Stop codon sits at the 3' end of the CDS, which flips with the strand.
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

fn only(models: Vec<TranscriptModel>) -> TranscriptModel {
    assert_eq!(models.len(), 1, "expected exactly one model");
    models.into_iter().next().unwrap()
}

////////////////////////
// Region composition  //
////////////////////////

#[test]
fn regions_split_on_the_coding_extent_and_are_spliced() {
    let m = only(build_transcript_models(&two_exon_tx(
        "ENST1",
        "G",
        Strand::Forward,
    )));

    // 1000..1099 is the 5'UTR; the stop codon extends the CDS to 1602, so the
    // 3'UTR starts at 1603 and runs to the end of exon 2.
    assert_eq!(m.utr5, vec![(1000, 1099)]);
    assert_eq!(m.cds, vec![(1100, 1199), (1500, 1602)]);
    assert_eq!(m.utr3, vec![(1603, 1699)]);

    // Spliced: the 1200..1499 intron consumes nothing.
    assert_eq!(m.utr5_size, 100);
    assert_eq!(m.cds_size, 100 + 103);
    assert_eq!(m.utr3_size, 97);
    assert_eq!(m.trx_len(), 100 + 203 + 97);
}

#[test]
fn the_five_and_three_prime_flanks_swap_on_the_reverse_strand() {
    let m = only(build_transcript_models(&two_exon_tx(
        "ENST1",
        "G",
        Strand::Backward,
    )));

    // Same coordinates, opposite reading direction: the HIGH end is now 5'.
    // The stop codon sits at 1097..1099 — the transcript's 3' end — so it
    // widens the coding extent DOWNWARDS, and the 5' flank is untouched.
    assert_eq!(m.utr5, vec![(1600, 1699)]);
    assert_eq!(m.cds, vec![(1097, 1199), (1500, 1599)]);
    assert_eq!(m.utr3, vec![(1000, 1096)]);
    assert_eq!(m.utr5_size, 100);
    assert_eq!(m.utr3_size, 97);
}

#[test]
fn stop_codon_is_part_of_the_cds() {
    // GENCODE excludes the stop codon from CDS; UCSC's tables (MetaPlotR's
    // input) fold it into cdsEnd. Without the widening these three bases would
    // sit at the very START of the 3'UTR, which is the bin an m6A metagene is
    // read for.
    let with_stop = only(build_transcript_models(&two_exon_tx(
        "ENST1",
        "G",
        Strand::Forward,
    )));

    let no_stop: Vec<GffRecord> = two_exon_tx("ENST1", "G", Strand::Forward)
        .into_iter()
        .filter(|r| r.feature_type != FeatureType::StopCodon)
        .collect();
    let without = only(build_transcript_models(&no_stop));

    assert_eq!(with_stop.cds_size - without.cds_size, 3);
    assert_eq!(without.utr3_size - with_stop.utr3_size, 3);
}

#[test]
fn regions_are_disjoint_and_sum_to_the_spliced_length() {
    for strand in [Strand::Forward, Strand::Backward] {
        let m = only(build_transcript_models(&two_exon_tx("ENST1", "G", strand)));
        let mut all: Vec<(i64, i64)> = Vec::new();
        all.extend(&m.utr5);
        all.extend(&m.cds);
        all.extend(&m.utr3);
        all.sort_unstable();
        for w in all.windows(2) {
            assert!(w[0].1 < w[1].0, "regions overlap on {strand:?}: {w:?}");
        }
        let covered: i64 = all.iter().map(|&(s, e)| e - s + 1).sum();
        assert_eq!(covered, m.trx_len(), "{strand:?}");
    }
}

#[test]
fn isoforms_do_not_merge_with_each_other() {
    // The whole point of the module: two isoforms of one gene stay separate,
    // where the gene-union model would fuse them into one longer 3'UTR.
    let mut records = two_exon_tx("ENST1", "G", Strand::Forward);
    records.extend(rec_span("ENST2", "G", Strand::Forward));

    let models = build_transcript_models(&records);
    assert_eq!(models.len(), 2);
    assert!(models
        .iter()
        .all(|m| m.gene_id == GeneId::Ensembl("G".into())));
}

/// A second, longer isoform of the same gene: one exon reaching further 3'.
fn rec_span(tx: &str, gene: &str, strand: Strand) -> Vec<GffRecord> {
    vec![
        rec(tx, gene, FeatureType::Exon, 1000, 2500, strand),
        rec(tx, gene, FeatureType::CDS, 1100, 1599, strand),
        rec(tx, gene, FeatureType::StopCodon, 1600, 1602, strand),
    ]
}

#[test]
fn a_transcript_without_cds_is_dropped() {
    let records = vec![
        rec("ENST1", "G", FeatureType::Exon, 1000, 1199, Strand::Forward),
        rec("ENST1", "G", FeatureType::Exon, 1500, 1699, Strand::Forward),
    ];
    assert!(build_transcript_models(&records).is_empty());
}

#[test]
fn gene_level_rows_carry_no_transcript_and_are_ignored() {
    let mut records = two_exon_tx("ENST1", "G", Strand::Forward);
    records.push(GffRecord {
        transcript_id: TranscriptId::Missing,
        ..rec("unused", "G", FeatureType::Gene, 900, 2000, Strand::Forward)
    });
    assert_eq!(build_transcript_models(&records).len(), 1);
}

////////////////////////
// Longest-isoform     //
////////////////////////

#[test]
fn the_longest_isoform_wins() {
    let mut records = two_exon_tx("ENST1", "G", Strand::Forward); // 400 nt
    records.extend(rec_span("ENST2", "G", Strand::Forward)); // 1501 nt

    let elected = elect_longest_isoform(build_transcript_models(&records));
    assert_eq!(elected.len(), 1);
    assert_eq!(
        elected[0].transcript_id,
        TranscriptId::Ensembl("ENST2".into())
    );
}

#[test]
fn single_isoform_genes_are_kept() {
    // MetaPlotR's published `dist[duplicated(gene_name), ]` drops these
    // entirely. We follow its stated intent instead, and this pins that.
    let records = two_exon_tx("ENST1", "G", Strand::Forward);
    let elected = elect_longest_isoform(build_transcript_models(&records));
    assert_eq!(elected.len(), 1);
    assert_eq!(
        elected[0].transcript_id,
        TranscriptId::Ensembl("ENST1".into())
    );
}

#[test]
fn ties_break_on_transcript_id_so_the_election_is_reproducible() {
    // Two isoforms of identical spliced length. Record order out of the parser
    // is not stable (`par_bridge`), so the winner must not depend on it.
    let mut a = two_exon_tx("ENST_B", "G", Strand::Forward);
    let b = two_exon_tx("ENST_A", "G", Strand::Forward);
    a.extend(b.clone());

    let forward = elect_longest_isoform(build_transcript_models(&a));
    let mut reversed = a.clone();
    reversed.reverse();
    let backward = elect_longest_isoform(build_transcript_models(&reversed));

    assert_eq!(forward.len(), 1);
    assert_eq!(
        forward[0].transcript_id,
        TranscriptId::Ensembl("ENST_A".into()),
        "the lexicographically smaller id wins"
    );
    assert_eq!(forward[0].transcript_id, backward[0].transcript_id);
}

#[test]
fn each_gene_keeps_its_own_winner() {
    let mut records = two_exon_tx("ENST1", "G1", Strand::Forward);
    records.extend(rec_span("ENST2", "G1", Strand::Forward));
    records.extend(two_exon_tx("ENST3", "G2", Strand::Backward));

    let elected = elect_longest_isoform(build_transcript_models(&records));
    assert_eq!(elected.len(), 2);
    let mut ids: Vec<String> = elected
        .iter()
        .map(|m| m.transcript_id.to_string())
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["ENST2".to_string(), "ENST3".to_string()]);
}
