use super::*;
use crate::snp::SnpGenotype;

fn site(chr: &str, pos: i64) -> SnpSite {
    SnpSite {
        chr: chr.into(),
        pos,
        ref_allele: b'A',
        alt_allele: b'G',
        rsid: None,
        counts: crate::data::dna::DnaBaseCount::new(),
        genotype: SnpGenotype::Het,
        gq: 99.0,
    }
}

fn gene_sites(entries: &[(&str, &[(&str, i64)])]) -> DashMap<GeneId, Vec<SnpSite>> {
    let map = DashMap::<GeneId, Vec<SnpSite>>::default();
    for (gene, loci) in entries {
        map.insert(
            GeneId::Ensembl((*gene).into()),
            loci.iter().map(|(c, p)| site(c, *p)).collect(),
        );
    }
    map
}

fn loci_of(map: &DashMap<GeneId, Vec<SnpSite>>) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = map
        .iter()
        .flat_map(|e| {
            e.value()
                .iter()
                .map(|s| (s.chr.to_string(), s.pos))
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort();
    out
}

/// The whole point of the owner assignment: a variant inside two overlapping
/// genes must survive exactly once. Rows are named for the locus, so leaving it
/// under both genes would fetch its reads twice and write both tallies to the
/// same row name.
#[test]
fn a_locus_in_two_genes_is_kept_once() {
    let map = gene_sites(&[
        ("GENE_A", &[("chr1", 100), ("chr1", 200)]),
        ("GENE_B", &[("chr1", 200), ("chr1", 300)]),
    ]);

    let dropped = dedup_loci_by_owner_gene(&map);

    assert_eq!(dropped, 1, "chr1:200 was carried by two genes");
    assert_eq!(
        loci_of(&map),
        vec![
            ("chr1".to_string(), 100),
            ("chr1".to_string(), 200),
            ("chr1".to_string(), 300)
        ],
        "every distinct locus survives, and only once"
    );
}

/// The shared locus goes to the lexicographically smallest gene id, so the
/// choice does not depend on GFF order or on which thread got there first.
#[test]
fn the_owner_is_the_smallest_gene_id_not_the_first_seen() {
    let map = gene_sites(&[
        ("ZZZ_LAST", &[("chr1", 200)]),
        ("AAA_FIRST", &[("chr1", 200)]),
    ]);

    dedup_loci_by_owner_gene(&map);

    assert_eq!(
        map.get(&GeneId::Ensembl("AAA_FIRST".into()))
            .map(|v| v.len()),
        Some(1),
        "smallest gene id owns the locus"
    );
    assert!(
        map.get(&GeneId::Ensembl("ZZZ_LAST".into())).is_none(),
        "a gene left with no loci is dropped from the map entirely"
    );
}

/// Same coordinate on a different contig is a different locus.
#[test]
fn the_same_position_on_two_contigs_is_two_loci() {
    let map = gene_sites(&[("GENE_A", &[("chr1", 200)]), ("GENE_B", &[("chr2", 200)])]);

    let dropped = dedup_loci_by_owner_gene(&map);

    assert_eq!(dropped, 0);
    assert_eq!(loci_of(&map).len(), 2);
}

/// Nothing shared, nothing touched.
#[test]
fn disjoint_genes_are_left_alone() {
    let map = gene_sites(&[
        ("GENE_A", &[("chr1", 100), ("chr1", 150)]),
        ("GENE_B", &[("chr1", 300)]),
    ]);

    let dropped = dedup_loci_by_owner_gene(&map);

    assert_eq!(dropped, 0);
    assert_eq!(map.len(), 2);
    assert_eq!(loci_of(&map).len(), 3);
}
