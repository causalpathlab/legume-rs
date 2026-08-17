use crate::common::*;
use crate::data::bam_io;
use rust_htslib::bam::{self, ext::BamRecordExtensions};

use rustc_hash::FxHashMap as HashMap;

pub struct SplicedUnsplicedTriplets {
    pub spliced: Vec<(CellBarcode, Box<str>, f32)>,
    pub unspliced: Vec<(CellBarcode, Box<str>, f32)>,
}

/// Format gene key as `"{gene_id}_{gene_symbol}"`.
///
/// Feature naming convention: `{gene_key}/{modality}/{detail}`
/// e.g. `ENSG00001234_BRCA2/count/spliced`
pub fn format_gene_key(rec: &GffRecord) -> Box<str> {
    match &rec.gene_name {
        GeneSymbol::Symbol(sym) if !sym.is_empty() => {
            format!("{}_{}", rec.gene_id, sym).into_boxed_str()
        }
        _ => rec.gene_id.to_string().into_boxed_str(),
    }
}

/// BAM tags and the read-admission threshold, shared by both gene counters.
///
/// Grouped rather than passed as loose arguments so every counting entry point
/// takes the same admission policy by construction: `faba genes` and the gene QC
/// pass behind each modality build one of these and cannot drift apart in which
/// tag they read or which alignments they trust.
#[derive(Clone, Copy)]
pub struct CountReadOpts<'a> {
    pub cell_barcode_tag: &'a str,
    pub gene_barcode_tag: &'a str,
    /// UMI tag for per-gene dedup; `None` counts reads instead of molecules.
    pub umi_tag: Option<&'a [u8]>,
    pub min_mapping_quality: u8,
}

/// Alignment-level admission for gene counting.
///
/// Deliberately narrower than [`bam_io::passes_alignment_filters`], which the
/// pileup modalities use: that predicate also requires `is_proper_pair()` for
/// paired records, which would drop legitimate reads from paired-end bulk input,
/// and `faba genes` quantifies bulk as well as single-cell libraries. The
/// duplicate flag is already screened by the `for_each_record_in_gene*`
/// iterators, so it is not repeated here.
///
/// The `is_unmapped` clause guards the `--min-mapping-quality 0` escape hatch: a
/// region fetch can return a placed-but-unmapped record, which has no aligned
/// blocks, and [`SpliceAwareReadCounter::classify`] reads "no intronic block" as
/// spliced. Without this it would be counted as a spliced molecule.
///
/// Call this BEFORE UMI dedup: a rejected read must not claim its molecule's
/// `(cell, UMI)` slot, or it would shadow a later well-aligned read of the same
/// molecule and undercount the gene.
#[inline]
fn passes_count_filters(bam_record: &bam::Record, min_mapping_quality: u8) -> bool {
    !bam_record.is_unmapped() && bam_io::passes_mapping_filters(bam_record, min_mapping_quality)
}

pub fn count_read_per_gene(
    cache: &mut bam_io::BamReaderCache,
    bam_file: &str,
    rec: &GffRecord,
    opts: CountReadOpts<'_>,
) -> anyhow::Result<Vec<(CellBarcode, Box<str>, f32)>> {
    if rec.gene_id == GeneId::Missing {
        return Ok(vec![]);
    }

    let gene_name = format_gene_key(rec);
    let row_name: Box<str> = format!("{}/count/total", gene_name).into();
    let mut read_counter = ReadCounter::new(opts);

    bam_io::for_each_record_in_gene_cached(
        cache,
        bam_file,
        rec,
        opts.gene_barcode_tag,
        false,
        |bam_record| {
            read_counter.count(bam_record);
        },
    )?;

    Ok(read_counter
        .to_vec()
        .into_iter()
        .map(|(cb, x)| (cb, row_name.clone(), x as f32))
        .collect())
}

pub fn count_read_per_gene_splice(
    cache: &mut bam_io::BamReaderCache,
    bam_file: &str,
    rec: &GffRecord,
    exon_intervals: &HashMap<GeneId, Vec<(i64, i64)>>,
    opts: CountReadOpts<'_>,
) -> anyhow::Result<SplicedUnsplicedTriplets> {
    if rec.gene_id == GeneId::Missing {
        return Ok(SplicedUnsplicedTriplets {
            spliced: vec![],
            unspliced: vec![],
        });
    }

    let exons = match exon_intervals.get(&rec.gene_id) {
        Some(e) if !e.is_empty() => e.as_slice(),
        _ => {
            // No exon annotations for this gene — skip entirely
            return Ok(SplicedUnsplicedTriplets {
                spliced: vec![],
                unspliced: vec![],
            });
        }
    };

    let gene_name = format_gene_key(rec);
    let spliced_name: Box<str> = format!("{}/count/spliced", gene_name).into();
    let unspliced_name: Box<str> = format!("{}/count/unspliced", gene_name).into();
    let mut counter = SpliceAwareReadCounter::new(opts, exons);

    bam_io::for_each_record_in_gene_cached(
        cache,
        bam_file,
        rec,
        opts.gene_barcode_tag,
        false,
        |bam_record| {
            counter.classify_and_count(bam_record);
        },
    )?;

    let spliced = counter
        .spliced
        .into_iter()
        .map(|(cb, x)| (cb, spliced_name.clone(), x as f32))
        .collect();

    let unspliced = counter
        .unspliced
        .into_iter()
        .map(|(cb, x)| (cb, unspliced_name.clone(), x as f32))
        .collect();

    Ok(SplicedUnsplicedTriplets { spliced, unspliced })
}

/// Per-gene UMI deduplication shared by both read counters: tracks the
/// `(cell, UMI-hash)` pairs already counted so each molecule is counted once.
/// `tag == None` disables dedup (every read is counted); reads with no UMI tag
/// are never treated as duplicates. Mirrors the per-(cell,UMI) collapse in
/// `DnaStatMap` used by the atoi/dartseq paths.
struct UmiDedup<'a> {
    tag: Option<&'a [u8]>,
    seen: rustc_hash::FxHashSet<(CellBarcode, u64)>,
}

impl<'a> UmiDedup<'a> {
    fn new(tag: Option<&'a [u8]>) -> Self {
        Self {
            tag,
            seen: rustc_hash::FxHashSet::default(),
        }
    }

    /// `true` if this record's `(cell, UMI)` was already counted for the
    /// current gene and should be skipped.
    fn is_duplicate(&mut self, bam_record: &bam::Record, cell_barcode: &CellBarcode) -> bool {
        let Some(tag) = self.tag else {
            return false;
        };
        let UmiBarcode::Hash(h) = bam_io::extract_umi(bam_record, tag) else {
            return false;
        };
        !self.seen.insert((cell_barcode.clone(), h))
    }
}

struct ReadCounter<'a> {
    cell_to_count: HashMap<CellBarcode, usize>,
    opts: CountReadOpts<'a>,
    dedup: UmiDedup<'a>,
}

impl<'a> ReadCounter<'a> {
    fn new(opts: CountReadOpts<'a>) -> Self {
        Self {
            cell_to_count: HashMap::default(),
            dedup: UmiDedup::new(opts.umi_tag),
            opts,
        }
    }

    fn to_vec(&self) -> Vec<(CellBarcode, usize)> {
        self.cell_to_count
            .iter()
            .map(|(cb, x)| (cb.clone(), *x))
            .collect()
    }

    fn count(&mut self, bam_record: &bam::Record) {
        if !passes_count_filters(bam_record, self.opts.min_mapping_quality) {
            return;
        }
        let cell_barcode =
            bam_io::extract_cell_barcode(bam_record, self.opts.cell_barcode_tag.as_bytes());
        if self.dedup.is_duplicate(bam_record, &cell_barcode) {
            return;
        }
        *self.cell_to_count.entry(cell_barcode).or_default() += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceClass {
    Spliced,
    Unspliced,
}

struct SpliceAwareReadCounter<'a> {
    spliced: HashMap<CellBarcode, usize>,
    unspliced: HashMap<CellBarcode, usize>,
    opts: CountReadOpts<'a>,
    exons: &'a [(i64, i64)], // sorted, merged, 0-based half-open
    dedup: UmiDedup<'a>,
}

impl<'a> SpliceAwareReadCounter<'a> {
    fn new(opts: CountReadOpts<'a>, exons: &'a [(i64, i64)]) -> Self {
        Self {
            spliced: HashMap::default(),
            unspliced: HashMap::default(),
            exons,
            dedup: UmiDedup::new(opts.umi_tag),
            opts,
        }
    }

    fn classify_and_count(&mut self, bam_record: &bam::Record) {
        // Before dedup — here a rejected read would also freeze the molecule's
        // spliced/unspliced class from an alignment we chose not to trust.
        if !passes_count_filters(bam_record, self.opts.min_mapping_quality) {
            return;
        }

        let cell_barcode =
            bam_io::extract_cell_barcode(bam_record, self.opts.cell_barcode_tag.as_bytes());

        // Skip reads without a valid cell barcode
        if cell_barcode == CellBarcode::Missing {
            return;
        }

        // UMI dedup keyed by (cell, UMI): the first read of a molecule in this
        // gene is counted (and fixes its spliced/unspliced class); later reads
        // of the same molecule are dropped.
        if self.dedup.is_duplicate(bam_record, &cell_barcode) {
            return;
        }

        match self.classify(bam_record) {
            SpliceClass::Spliced => {
                *self.spliced.entry(cell_barcode).or_default() += 1;
            }
            SpliceClass::Unspliced => {
                *self.unspliced.entry(cell_barcode).or_default() += 1;
            }
        }
    }

    /// Classify a read as spliced or unspliced.
    ///
    /// - **Unspliced**: any aligned block has intronic extent > 0
    /// - **Spliced**: everything else (exon-only reads lumped into spliced,
    ///   following the alevin-fry S+A convention)
    fn classify(&self, bam_record: &bam::Record) -> SpliceClass {
        // Consume the block iterator directly: collecting first cost one
        // malloc/free per counted read, and the loop only ever scans forward.
        for [b_start, b_end] in bam_record.aligned_blocks() {
            if self.intronic_extent(b_start, b_end) > 0 {
                return SpliceClass::Unspliced;
            }
        }

        SpliceClass::Spliced
    }

    /// Returns the number of base pairs in `[b_start, b_end)` that fall
    /// outside all exon intervals.
    fn intronic_extent(&self, b_start: i64, b_end: i64) -> i64 {
        let mut covered = 0i64;
        for &(e_start, e_end) in self.exons {
            if e_start >= b_end {
                break; // exons are sorted
            }
            if e_end <= b_start {
                continue;
            }
            let overlap_start = b_start.max(e_start);
            let overlap_end = b_end.min(e_end);
            covered += overlap_end - overlap_start;
        }
        (b_end - b_start) - covered
    }
}

#[cfg(test)]
mod tests;
