use crate::common::*;
use crate::data::bam_io;
use crate::data::util_htslib::*;
use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Cigar};

use fnv::FnvHashMap as HashMap;

#[derive(Args, Debug)]
pub struct GeneCountArgs {
    /// `.bam` files to quantify
    #[arg(value_delimiter = ',', required = true)]
    bam_files: Vec<Box<str>>,

    /// Gene annotation (`GFF`) file
    #[arg(short = 'g', long = "gff", required = true)]
    gff_file: Box<str>,

    /// (10x) cell/sample barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "CB")]
    cell_barcode_tag: Box<str>,

    /// gene barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "GX")]
    gene_barcode_tag: Box<str>,

    /// bam record type (gene, transcript, exon, utr)
    #[arg(long, default_value = "gene")]
    record_type: Box<str>,

    /// gene type (protein_coding, pseudogene, lncRNA)
    #[arg(long, default_value = "protein_coding")]
    gene_type: Box<str>,

    /// number of non-zero cutoff for rows/genes
    #[arg(short, long, default_value_t = 10)]
    row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns/cells
    #[arg(short, long, default_value_t = 10)]
    column_nnz_cutoff: usize,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// output header for `data-beans` files
    #[arg(short, long, required = true)]
    output: Box<str>,

    /// Count spliced and unspliced reads separately (for RNA velocity)
    #[arg(long, default_value_t = false)]
    splice: bool,

    /// Buffer zone (bp) around exon boundaries for splice classification
    #[arg(long, default_value_t = 3)]
    intron_buffer: i64,
}

/// simply count the occurence of gene and cell barcode
pub fn run_gene_count(args: &GeneCountArgs) -> anyhow::Result<()> {
    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need bam files"));
    }

    for x in args.bam_files.iter() {
        check_bam_index(x, None)?;
    }

    info!("data files:");
    for x in args.bam_files.iter() {
        info!("{}", x);
    }

    let backend = args.backend.clone();
    info!("parsing GFF file: {}", args.gff_file);

    if args.splice {
        run_splice_aware(args, &backend)
    } else {
        run_simple(args, &backend)
    }
}

fn run_simple(args: &GeneCountArgs, backend: &SparseIoBackend) -> anyhow::Result<()> {
    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;
    gff_map.subset(args.gene_type.clone().into());
    info!("found {} features", gff_map.len());

    if gff_map.is_empty() {
        info!("found no feature in {}", args.gff_file.as_ref());
        return Ok(());
    }

    let njobs = gff_map.len() as u64;
    info!("Combining reads over {} blocks", njobs);

    let gene_level_stats = gff_map
        .records()
        .par_iter()
        .progress_count(njobs)
        .map(|x| count_read_per_gene(args, x))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let output = args.output.clone();
    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    format_data_triplets(gene_level_stats)
        .to_backend(&backend_file)?
        .qc(SqueezeCutoffs {
            row: args.row_nnz_cutoff,
            column: args.column_nnz_cutoff,
        })?;

    Ok(())
}

fn run_splice_aware(args: &GeneCountArgs, backend: &SparseIoBackend) -> anyhow::Result<()> {
    let all_records = read_gff_record_vec(args.gff_file.as_ref())?;

    // Build gene map and exon intervals from the same parse
    let gene_map = build_gene_map(&all_records, Some(&FeatureType::Gene))?;
    let exon_map = build_exon_intervals(&all_records);

    let mut gff_map = GffRecordMap::from_map(gene_map);
    gff_map.subset(args.gene_type.clone().into());
    info!("found {} features", gff_map.len());

    if gff_map.is_empty() {
        info!("found no feature in {}", args.gff_file.as_ref());
        return Ok(());
    }

    // Convert DashMap exon intervals to FnvHashMap for fast per-gene lookup
    let exon_intervals: HashMap<GeneId, Vec<(i64, i64)>> = exon_map.into_iter().collect();

    let njobs = gff_map.len() as u64;
    info!(
        "Combining reads (splice-aware) over {} blocks, intron_buffer={}",
        njobs, args.intron_buffer
    );

    // Genes with no exon annotations are skipped
    let records = gff_map.records();
    let results: Vec<SplicedUnsplicedTriplets> = records
        .par_iter()
        .progress_count(njobs)
        .map(|rec| count_read_per_gene_splice(args, rec, &exon_intervals))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut spliced_triplets = Vec::new();
    let mut unspliced_triplets = Vec::new();
    for r in results {
        spliced_triplets.extend(r.spliced);
        unspliced_triplets.extend(r.unspliced);
    }

    info!(
        "spliced triplets: {}, unspliced triplets: {}",
        spliced_triplets.len(),
        unspliced_triplets.len()
    );

    let output = args.output.clone();
    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    // Write spliced matrix
    let spliced_file = match backend {
        SparseIoBackend::HDF5 => format!("{}_spliced.h5", &output),
        SparseIoBackend::Zarr => format!("{}_spliced.zarr", &output),
    };
    info!("writing spliced counts to {}", spliced_file);
    format_data_triplets(spliced_triplets)
        .to_backend(&spliced_file)?
        .qc(cutoffs.clone())?;

    // Write unspliced matrix
    let unspliced_file = match backend {
        SparseIoBackend::HDF5 => format!("{}_unspliced.h5", &output),
        SparseIoBackend::Zarr => format!("{}_unspliced.zarr", &output),
    };
    info!("writing unspliced counts to {}", unspliced_file);
    format_data_triplets(unspliced_triplets)
        .to_backend(&unspliced_file)?
        .qc(cutoffs)?;

    Ok(())
}

struct SplicedUnsplicedTriplets {
    spliced: Vec<(CellBarcode, GeneId, f32)>,
    unspliced: Vec<(CellBarcode, GeneId, f32)>,
}

fn count_read_per_gene(
    args: &GeneCountArgs,
    rec: &GffRecord,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, f32)>> {
    let gene_id = &rec.gene_id;

    if gene_id == &GeneId::Missing {
        return Ok(vec![]);
    }

    let mut read_counter = ReadCounter::new(&args.cell_barcode_tag);

    for file in &args.bam_files {
        bam_io::for_each_record_in_gene(file, rec, &args.gene_barcode_tag, false, |bam_record| {
            read_counter.count(bam_record);
        })?;
    }

    Ok(read_counter
        .to_vec()
        .into_iter()
        .map(|(cb, x)| (cb, gene_id.clone(), x as f32))
        .collect())
}

fn count_read_per_gene_splice(
    args: &GeneCountArgs,
    rec: &GffRecord,
    exon_intervals: &HashMap<GeneId, Vec<(i64, i64)>>,
) -> anyhow::Result<SplicedUnsplicedTriplets> {
    let gene_id = &rec.gene_id;

    if gene_id == &GeneId::Missing {
        return Ok(SplicedUnsplicedTriplets {
            spliced: vec![],
            unspliced: vec![],
        });
    }

    let exons = match exon_intervals.get(gene_id) {
        Some(e) if !e.is_empty() => e.as_slice(),
        _ => {
            // No exon annotations for this gene — skip entirely
            return Ok(SplicedUnsplicedTriplets {
                spliced: vec![],
                unspliced: vec![],
            });
        }
    };

    let mut counter =
        SpliceAwareReadCounter::new(&args.cell_barcode_tag, exons, args.intron_buffer);

    for file in &args.bam_files {
        bam_io::for_each_record_in_gene(file, rec, &args.gene_barcode_tag, false, |bam_record| {
            counter.classify_and_count(&bam_record);
        })?;
    }

    let spliced = counter
        .spliced
        .into_iter()
        .map(|(cb, x)| (cb, gene_id.clone(), x as f32))
        .collect();

    let unspliced = counter
        .unspliced
        .into_iter()
        .map(|(cb, x)| (cb, gene_id.clone(), x as f32))
        .collect();

    Ok(SplicedUnsplicedTriplets { spliced, unspliced })
}

struct ReadCounter<'a> {
    cell_to_count: HashMap<CellBarcode, usize>,
    cell_barcode_tag: &'a str,
}

impl<'a> ReadCounter<'a> {
    fn new(cell_barcode_tag: &'a str) -> Self {
        Self {
            cell_to_count: HashMap::default(),
            cell_barcode_tag,
        }
    }

    fn to_vec(&self) -> Vec<(CellBarcode, usize)> {
        self.cell_to_count
            .iter()
            .map(|(cb, x)| (cb.clone(), *x))
            .collect()
    }

    fn count(&mut self, bam_record: bam::Record) {
        let cell_barcode =
            bam_io::extract_cell_barcode(&bam_record, self.cell_barcode_tag.as_bytes());
        *self.cell_to_count.entry(cell_barcode).or_default() += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceClass {
    Spliced,
    Unspliced,
    Ambiguous,
}

struct SpliceAwareReadCounter<'a> {
    spliced: HashMap<CellBarcode, usize>,
    unspliced: HashMap<CellBarcode, usize>,
    cell_barcode_tag: &'a str,
    exons: &'a [(i64, i64)], // sorted, merged, 0-based half-open
    intron_buffer: i64,
}

impl<'a> SpliceAwareReadCounter<'a> {
    fn new(cell_barcode_tag: &'a str, exons: &'a [(i64, i64)], intron_buffer: i64) -> Self {
        Self {
            spliced: HashMap::default(),
            unspliced: HashMap::default(),
            cell_barcode_tag,
            exons,
            intron_buffer,
        }
    }

    fn classify_and_count(&mut self, bam_record: &bam::Record) {
        let class = self.classify(bam_record);
        if class == SpliceClass::Ambiguous {
            return;
        }

        let cell_barcode =
            bam_io::extract_cell_barcode(bam_record, self.cell_barcode_tag.as_bytes());

        match class {
            SpliceClass::Spliced => {
                *self.spliced.entry(cell_barcode).or_default() += 1;
            }
            SpliceClass::Unspliced => {
                *self.unspliced.entry(cell_barcode).or_default() += 1;
            }
            SpliceClass::Ambiguous => unreachable!(),
        }
    }

    fn classify(&self, bam_record: &bam::Record) -> SpliceClass {
        let has_splice_junction = bam_record
            .cigar()
            .iter()
            .any(|op| matches!(op, Cigar::RefSkip(_)));

        // Get aligned blocks: 0-based half-open [start, end)
        let blocks: Vec<[i64; 2]> = bam_record.aligned_blocks().collect();

        if blocks.is_empty() {
            return SpliceClass::Ambiguous;
        }

        let mut any_intronic = false;
        let mut all_exonic = true;

        for &[b_start, b_end] in &blocks {
            let intronic_bp = self.intronic_extent(b_start, b_end);
            if intronic_bp >= self.intron_buffer {
                any_intronic = true;
                all_exonic = false;
            } else if intronic_bp > 0 {
                // Within buffer zone — not fully exonic but not definitively intronic
                all_exonic = false;
            }
        }

        if any_intronic {
            return SpliceClass::Unspliced;
        }

        if has_splice_junction && all_exonic {
            return SpliceClass::Spliced;
        }

        // Multi-exon read without N in CIGAR but all blocks exonic and
        // spanning multiple exons → spliced
        if all_exonic && self.spans_multiple_exons(&blocks) {
            return SpliceClass::Spliced;
        }

        SpliceClass::Ambiguous
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

    /// Check if the aligned blocks span multiple distinct exons.
    fn spans_multiple_exons(&self, blocks: &[[i64; 2]]) -> bool {
        let mut exons_hit = 0usize;
        let read_start = blocks.first().map_or(0, |b| b[0]);
        let read_end = blocks.last().map_or(0, |b| b[1]);

        for &(e_start, e_end) in self.exons {
            if e_start >= read_end {
                break;
            }
            if e_end <= read_start {
                continue;
            }
            exons_hit += 1;
            if exons_hit >= 2 {
                return true;
            }
        }
        false
    }
}
