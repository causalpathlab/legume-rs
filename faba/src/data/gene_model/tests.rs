use super::*;
use genomic_data::gff::GeneId;

/// Two exons of 100 bp either side of an 800 bp intron.
fn model() -> SplicedGenes {
    let mut exons = FxHashMap::default();
    exons.insert(
        GeneId::Ensembl("G".into()),
        vec![(1000, 1100), (1900, 2000)],
    );
    SplicedGenes { exons }
}

fn g() -> GeneId {
    GeneId::Ensembl("G".into())
}

/// The whole point: an intron must not consume transcript coordinate.
///
/// Genomically the second exon starts 900 bp after the first, but in the
/// transcript it starts at 100 — the 800 bp intron is not in the mRNA.
#[test]
fn introns_consume_no_transcript_coordinate() {
    let m = model();
    assert_eq!(m.rel_pos(&g(), 1000, Strand::Forward), Some(0));
    assert_eq!(m.rel_pos(&g(), 1099, Strand::Forward), Some(99));
    // First base of exon 2: genomic offset would be 900, spliced is 100.
    assert_eq!(m.rel_pos(&g(), 1900, Strand::Forward), Some(100));
    assert_eq!(m.rel_pos(&g(), 1999, Strand::Forward), Some(199));
}

/// On the reverse strand the transcript reads from the highest coordinate down,
/// so offset 0 is the LAST genomic base.
#[test]
fn reverse_strand_counts_from_the_other_end() {
    let m = model();
    assert_eq!(m.rel_pos(&g(), 1999, Strand::Backward), Some(0));
    assert_eq!(m.rel_pos(&g(), 1900, Strand::Backward), Some(99));
    assert_eq!(m.rel_pos(&g(), 1099, Strand::Backward), Some(100));
    assert_eq!(m.rel_pos(&g(), 1000, Strand::Backward), Some(199));
}

/// Every exonic base maps to a distinct offset, and the offsets are exactly
/// `0..total` — no gaps, no collisions, on either strand.
#[test]
fn exonic_bases_tile_the_transcript_exactly() {
    let m = model();
    for strand in [Strand::Forward, Strand::Backward] {
        let mut seen: Vec<i64> = (1000..1100)
            .chain(1900..2000)
            .map(|p| m.rel_pos(&g(), p, strand).expect("exonic"))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..200).collect::<Vec<_>>(), "{strand:?}");
    }
}

/// A base that is not in the transcript has no transcript coordinate. Returning
/// the nearest exon edge instead would be indistinguishable from a real value.
#[test]
fn intronic_and_outside_positions_have_no_coordinate() {
    let m = model();
    assert_eq!(m.rel_pos(&g(), 1500, Strand::Forward), None, "intronic");
    assert_eq!(
        m.rel_pos(&g(), 1100, Strand::Forward),
        None,
        "first intron base"
    );
    assert_eq!(
        m.rel_pos(&g(), 1899, Strand::Forward),
        None,
        "last intron base"
    );
    assert_eq!(
        m.rel_pos(&g(), 999, Strand::Forward),
        None,
        "before the gene"
    );
    assert_eq!(
        m.rel_pos(&g(), 2000, Strand::Forward),
        None,
        "past the gene"
    );
    assert_eq!(
        m.rel_pos(&GeneId::Ensembl("ABSENT".into()), 1000, Strand::Forward),
        None,
        "gene with no exon model"
    );
}
