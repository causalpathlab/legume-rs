use crate::data::dna::{Dna, DnaBaseCount};
use crate::data::dna_stat_map::{CountMode, DnaBaseFreqMap, HashMap};
use genomic_data::sam::CellBarcode;

fn cb(s: &str) -> CellBarcode {
    CellBarcode::Barcode(s.into())
}

/// Build a `PerCell` map by hand: two cells, both covering pos 100, one also
/// covering pos 200. Bypasses the BAM reader so the pooling logic is the only
/// thing under test.
fn per_cell_fixture() -> DnaBaseFreqMap<'static> {
    let mut map = DnaBaseFreqMap::new_with_cell_barcode("CB", None);
    let CountMode::PerCell { counts, .. } = &mut map.mode else {
        unreachable!("new_with_cell_barcode must build a PerCell map");
    };

    let mut cell_a = DnaBaseCount::new();
    cell_a.add(Some(&Dna::C), 7);
    cell_a.add(Some(&Dna::T), 3);
    let mut cell_b = DnaBaseCount::new();
    cell_b.add(Some(&Dna::C), 4);
    cell_b.add(Some(&Dna::T), 1);

    let mut at_100: HashMap<CellBarcode, DnaBaseCount> = HashMap::default();
    at_100.insert(cb("AAA-1"), cell_a);
    at_100.insert(cb("BBB-1"), cell_b.clone());
    counts.insert(100, at_100);

    let mut at_200: HashMap<CellBarcode, DnaBaseCount> = HashMap::default();
    at_200.insert(cb("BBB-1"), cell_b);
    counts.insert(200, at_200);

    map
}

/// Regression: discovery pools over cells even when the scan is stratified.
/// `marginal_frequency_map` returns `None` for `PerCell`, which turned the
/// cell-membership discovery path into a hard "failed to count wt freq" error.
#[test]
fn pooled_frequency_map_sums_over_cells_in_per_cell_mode() {
    let map = per_cell_fixture();
    assert!(
        map.marginal_frequency_map().is_none(),
        "PerCell has no marginal map -- that is exactly why pooling is needed"
    );

    let pooled = map.pooled_frequency_map();
    assert_eq!(pooled.len(), 2);

    let p100 = pooled.get(&100).expect("pos 100 pooled");
    assert_eq!(p100.get(Some(&Dna::C)), 11); // 7 + 4
    assert_eq!(p100.get(Some(&Dna::T)), 4); //  3 + 1
    assert_eq!(p100.total(), 15);

    let p200 = pooled.get(&200).expect("pos 200 pooled");
    assert_eq!(p200.get(Some(&Dna::C)), 4);
    assert_eq!(p200.total(), 5);
}

/// In `Marginal` mode pooling is the identity, and borrows rather than clones.
#[test]
fn pooled_frequency_map_borrows_the_marginal_map_unchanged() {
    let mut map = DnaBaseFreqMap::new();
    let CountMode::Marginal { counts, .. } = &mut map.mode else {
        unreachable!("new must build a Marginal map");
    };
    let mut at_100 = DnaBaseCount::new();
    at_100.add(Some(&Dna::A), 9);
    counts.insert(100, at_100);

    let pooled = map.pooled_frequency_map();
    assert!(matches!(pooled, std::borrow::Cow::Borrowed(_)));
    assert_eq!(pooled.get(&100).expect("pos 100").get(Some(&Dna::A)), 9);
    assert_eq!(
        pooled.as_ref(),
        map.marginal_frequency_map().expect("marginal map")
    );
}

/// The filter must bite in `Marginal` mode, which is where bulk site discovery
/// runs — `CountMode::PerCell`'s `cell_membership` does not exist there, so
/// without this a de-diluted discovery pass would silently count every cell.
#[test]
fn keep_cells_is_settable_on_a_marginal_map() {
    let mut map = DnaBaseFreqMap::new();
    assert!(map.marginal_frequency_map().is_some(), "starts Marginal");
    let mut keep = rustc_hash::FxHashSet::default();
    keep.insert(CellBarcode::Barcode("AAA-1".into()));
    map.set_keep_cells("CB", Some(std::sync::Arc::new(keep)));
    // Still Marginal: the filter must not silently switch the map to per-cell
    // accounting, which would change its memory profile.
    assert!(map.marginal_frequency_map().is_some(), "stays Marginal");
}

/// One map is reused across BAMs in the discovery loop, re-restricted per
/// library. A set left over from the previous library would silently filter this
/// one against barcodes that mean nothing in it, so clearing has to actually clear.
#[test]
fn clearing_the_cell_filter_really_drops_it() {
    let mut map = DnaBaseFreqMap::new();
    let mut keep = rustc_hash::FxHashSet::default();
    keep.insert(CellBarcode::Barcode("AAA-1".into()));
    map.set_keep_cells("CB", Some(std::sync::Arc::new(keep)));
    assert!(map.keep_cells.is_some());
    map.set_keep_cells("CB", None);
    assert!(
        map.keep_cells.is_none(),
        "a stale keep-set would filter the next library against the wrong barcodes"
    );
}

/// A UMI is only unique WITHIN a cell, so two cells may legitimately carry the
/// same UMI sequence for different molecules. Collapsing on the UMI alone
/// deletes one of them.
///
/// `docs/profiling-methods.md` states the contract: reads sharing a UMI collapse
/// to one observation **per cell per gene**. `Marginal` -- the DEFAULT discovery
/// path for m6A, A-to-I and every SNP pass -- deduped on the bare UMI hash, so
/// it silently dropped real coverage on exactly the high-expression genes that
/// clear the coverage floors and produce calls. Five other dedup sites in the
/// crate key on `(cell, UMI)`, two of them under comments claiming they match
/// this one.
#[test]
fn marginal_dedup_keeps_the_same_umi_from_different_cells() {
    use rust_htslib::bam::record::{Aux, Record};

    let read = |cell: &str, umi: &str| {
        let mut rec = Record::new();
        let cigar = rust_htslib::bam::record::CigarString::try_from("4M").unwrap();
        rec.set(b"r", Some(&cigar), b"ACGT", &[40u8; 4]);
        rec.set_tid(0);
        rec.set_pos(100);
        rec.set_mapq(60);
        rec.unset_unmapped();
        rec.push_aux(b"CB", Aux::String(cell)).unwrap();
        rec.push_aux(b"UB", Aux::String(umi)).unwrap();
        rec
    };

    let mut map = DnaBaseFreqMap::new();
    map.set_umi_tag("UB", "CB");
    // Same UMI string, two different cells: two molecules, not one.
    map.add_bam_record(&read("CELL_A", "SAMEUMI"));
    map.add_bam_record(&read("CELL_B", "SAMEUMI"));

    let counts = map.marginal_frequency_map().expect("Marginal map");
    let depth: usize = counts.get(&100).map(|c| c.total()).unwrap_or(0);
    assert_eq!(
        depth, 2,
        "two cells sharing a UMI are two molecules; deduping on the UMI alone \
         drops one and understates coverage"
    );

    // ...and the same cell repeating a UMI is still one molecule.
    let mut same = DnaBaseFreqMap::new();
    same.set_umi_tag("UB", "CB");
    same.add_bam_record(&read("CELL_A", "SAMEUMI"));
    same.add_bam_record(&read("CELL_A", "SAMEUMI"));
    let counts = same.marginal_frequency_map().expect("Marginal map");
    assert_eq!(
        counts.get(&100).map(|c| c.total()).unwrap_or(0),
        1,
        "a PCR duplicate within one cell must still collapse"
    );
}
