use genomic_data::bed::*;
use genomic_data::gff::*;
use genomic_data::sam::*;

use rust_htslib::bam::{self, record::Aux, Read};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

thread_local! {
    /// Per-worker cell-barcode interner. Each rayon worker thread gets
    /// its own; lookups are O(1) FxHashMap probes and repeat barcodes
    /// hand back a cloned `Arc<str>` (refcount bump, no malloc). Across
    /// a chr1 APA run this cuts `Box<str>` allocations from ~30M to
    /// roughly the cell count (~5–10k per worker).
    static BARCODE_INTERNER: RefCell<FxHashMap<Box<[u8]>, Arc<str>>> =
        RefCell::new(FxHashMap::default());
}

fn intern_barcode(bytes: &[u8]) -> Arc<str> {
    BARCODE_INTERNER.with(|cell| {
        let mut map = cell.borrow_mut();
        if let Some(v) = map.get(bytes) {
            return v.clone();
        }
        // Allocate only on first sight of this barcode in this worker.
        let s: Arc<str> = Arc::from(std::str::from_utf8(bytes).unwrap_or(""));
        map.insert(Box::from(bytes), s.clone());
        s
    })
}

/// Hash a UMI byte slice to u64 — see `UmiBarcode::Hash`.
#[inline]
pub fn hash_umi_bytes(umi: &[u8]) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    umi.hash(&mut h);
    h.finish()
}

/// Per-thread cache of opened `bam::IndexedReader`s keyed by file path.
///
/// Opening a BAM + loading its `.bai` index is expensive (hundreds of ms on
/// large files); the previous code paid this cost once per gene per BAM
/// inside the rayon par_iter. With `try_for_each_init(|| BamReaderCache::new(),
/// ...)` each worker thread pays it once per BAM total and reuses the reader
/// across every gene scheduled to it.
///
/// `bam::IndexedReader` is `!Sync` (holds a raw htslib handle), so the cache
/// must stay thread-local — never share it across threads.
#[derive(Default)]
pub struct BamReaderCache {
    readers: FxHashMap<Box<str>, bam::IndexedReader>,
}

impl BamReaderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_open(&mut self, bam_file_path: &str) -> anyhow::Result<&mut bam::IndexedReader> {
        if !self.readers.contains_key(bam_file_path) {
            let index_file = bam_file_path.to_string() + ".bai";
            let reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;
            self.readers.insert(bam_file_path.into(), reader);
        }
        Ok(self.readers.get_mut(bam_file_path).expect("just inserted"))
    }
}

/// Extract cell barcode from a BAM record's auxiliary tag.
///
/// Returns an `Arc<str>` from a per-thread interner — repeat barcodes are
/// shared rather than re-allocated, and downstream `.clone()`s become
/// refcount bumps instead of heap allocations.
pub fn extract_cell_barcode(record: &bam::Record, tag: &[u8]) -> CellBarcode {
    match record.aux(tag) {
        Ok(Aux::String(barcode)) => CellBarcode::Barcode(intern_barcode(barcode.as_bytes())),
        _ => CellBarcode::Missing,
    }
}

/// Extract UMI from a BAM record's auxiliary tag as a hash. UMIs are only
/// used for dedup, never displayed; hashing avoids the per-read string
/// allocation.
pub fn extract_umi(record: &bam::Record, tag: &[u8]) -> UmiBarcode {
    match record.aux(tag) {
        Ok(Aux::String(umi)) => UmiBarcode::Hash(hash_umi_bytes(umi.as_bytes())),
        _ => UmiBarcode::Missing,
    }
}

/// Check if a BAM record's gene barcode matches the expected gene ID.
/// Compares raw `&str` from the aux tag to avoid allocating a `GeneId`.
pub fn matches_gene(
    record: &bam::Record,
    tag: &str,
    expected: &GeneId,
    include_missing: bool,
) -> bool {
    match record.aux(tag.as_bytes()) {
        Ok(Aux::String(id)) => match (parse_ensembl_id(id), expected) {
            (Some(parsed), GeneId::Ensembl(exp)) => parsed == exp.as_ref(),
            _ => include_missing,
        },
        _ => include_missing,
    }
}

/// Stream non-duplicate BAM records in a BED region.
///
/// The visitor borrows a **single reusable `bam::Record` buffer** — htslib's
/// `read(&mut rec)` overwrites it in place each iteration, so the visitor
/// must not stash `rec` past its call. This avoids the per-record
/// `Record::new()` allocation that `bam_reader.records()` would do, which
/// dominates the allocator on heavy UTRs (10⁵+ reads × 16 threads).
pub fn for_each_record_in_region(
    bam_file_path: &str,
    bed: &Bed,
    visitor: impl FnMut(&str, &bam::Record),
) -> anyhow::Result<()> {
    let mut cache = BamReaderCache::new();
    for_each_record_in_region_cached(&mut cache, bam_file_path, bed, visitor)
}

/// Like [`for_each_record_in_region`] but reuses an `IndexedReader` from
/// `cache` so repeated calls (one per gene/region) don't re-open the BAM.
pub fn for_each_record_in_region_cached(
    cache: &mut BamReaderCache,
    bam_file_path: &str,
    bed: &Bed,
    mut visitor: impl FnMut(&str, &bam::Record),
) -> anyhow::Result<()> {
    let chr = bed.chr.as_ref();
    let region = (chr, bed.start, bed.stop);

    let bam_reader = cache.get_or_open(bam_file_path)?;
    bam_reader.fetch(region)?;

    let mut rec = bam::Record::new();
    while let Some(res) = bam_reader.read(&mut rec) {
        if res.is_err() {
            continue;
        }
        if !rec.is_duplicate() {
            visitor(chr, &rec);
        }
    }

    Ok(())
}

/// Stream non-duplicate BAM records matching a gene barcode. See
/// [`for_each_record_in_region`] for the buffer-reuse contract.
pub fn for_each_record_in_gene(
    bam_file_path: &str,
    gff_record: &GffRecord,
    gene_barcode_tag: &str,
    include_missing_barcode: bool,
    visitor: impl FnMut(&bam::Record),
) -> anyhow::Result<()> {
    let mut cache = BamReaderCache::new();
    for_each_record_in_gene_cached(
        &mut cache,
        bam_file_path,
        gff_record,
        gene_barcode_tag,
        include_missing_barcode,
        visitor,
    )
}

/// Like [`for_each_record_in_gene`] but reuses an `IndexedReader` from `cache`.
pub fn for_each_record_in_gene_cached(
    cache: &mut BamReaderCache,
    bam_file_path: &str,
    gff_record: &GffRecord,
    gene_barcode_tag: &str,
    include_missing_barcode: bool,
    mut visitor: impl FnMut(&bam::Record),
) -> anyhow::Result<()> {
    let region = (
        gff_record.seqname.as_ref(),
        gff_record.start,
        gff_record.stop,
    );

    let bam_reader = cache.get_or_open(bam_file_path)?;
    bam_reader.fetch(region)?;

    let gene_id = &gff_record.gene_id;

    let mut rec = bam::Record::new();
    while let Some(res) = bam_reader.read(&mut rec) {
        if res.is_err() {
            continue;
        }
        if !rec.is_duplicate()
            && matches_gene(&rec, gene_barcode_tag, gene_id, include_missing_barcode)
        {
            visitor(&rec);
        }
    }

    Ok(())
}
