use crate::common::*;
use crate::data::bam_io;
use rust_htslib::bam::{self, ext::BamRecordExtensions};

use fnv::FnvHashMap as HashMap;

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

pub fn count_read_per_gene(
    bam_file: &str,
    rec: &GffRecord,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
) -> anyhow::Result<Vec<(CellBarcode, Box<str>, f32)>> {
    if rec.gene_id == GeneId::Missing {
        return Ok(vec![]);
    }

    let gene_name = format_gene_key(rec);
    let row_name: Box<str> = format!("{}/count/total", gene_name).into();
    let mut read_counter = ReadCounter::new(cell_barcode_tag);

    bam_io::for_each_record_in_gene(bam_file, rec, gene_barcode_tag, false, |bam_record| {
        read_counter.count(bam_record);
    })?;

    Ok(read_counter
        .to_vec()
        .into_iter()
        .map(|(cb, x)| (cb, row_name.clone(), x as f32))
        .collect())
}

pub fn count_read_per_gene_splice(
    bam_file: &str,
    rec: &GffRecord,
    exon_intervals: &HashMap<GeneId, Vec<(i64, i64)>>,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
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
    let mut counter = SpliceAwareReadCounter::new(cell_barcode_tag, exons);

    bam_io::for_each_record_in_gene(bam_file, rec, gene_barcode_tag, false, |bam_record| {
        counter.classify_and_count(&bam_record);
    })?;

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

pub struct ReadCounter<'a> {
    cell_to_count: HashMap<CellBarcode, usize>,
    cell_barcode_tag: &'a str,
}

impl<'a> ReadCounter<'a> {
    pub fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            cell_to_count: HashMap::default(),
            cell_barcode_tag,
        }
    }

    pub fn to_vec(&self) -> Vec<(CellBarcode, usize)> {
        self.cell_to_count
            .iter()
            .map(|(cb, x)| (cb.clone(), *x))
            .collect()
    }

    pub fn count(&mut self, bam_record: bam::Record) {
        let cell_barcode =
            bam_io::extract_cell_barcode(&bam_record, self.cell_barcode_tag.as_bytes());
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
    cell_barcode_tag: &'a str,
    exons: &'a [(i64, i64)], // sorted, merged, 0-based half-open
}

impl<'a> SpliceAwareReadCounter<'a> {
    fn new(cell_barcode_tag: &'a str, exons: &'a [(i64, i64)]) -> Self {
        Self {
            spliced: HashMap::default(),
            unspliced: HashMap::default(),
            cell_barcode_tag,
            exons,
        }
    }

    fn classify_and_count(&mut self, bam_record: &bam::Record) {
        let cell_barcode =
            bam_io::extract_cell_barcode(bam_record, self.cell_barcode_tag.as_bytes());

        // Skip reads without a valid cell barcode
        if cell_barcode == CellBarcode::Missing {
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
        let blocks: Vec<[i64; 2]> = bam_record.aligned_blocks().collect();

        for &[b_start, b_end] in &blocks {
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
