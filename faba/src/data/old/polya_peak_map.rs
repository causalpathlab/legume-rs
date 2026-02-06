// #![allow(dead_code)]

use genomic_data::bed::Bed;
use genomic_data::gff::GffRecord;
use genomic_data::sam::{CellBarcode, Strand};
use crate::data::visitors_htslib::*;

use coitrees::{COITree, Interval, IntervalTree};
use fnv::FnvHashMap as HashMap;
use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

/// Helper struct to store peak region information
#[derive(Clone, Debug)]
pub struct PeakRegion {
    pub start: i64,
    pub end: i64,
    pub peak_pos: i64, // The actual endpoint position
    pub depth: usize,
    pub strand: Strand,
    pub misprime: bool,
}

/// Metadata stored in COITree intervals for polyA peaks
#[derive(Clone, Debug, Copy, Default)]
pub struct PolyAPeakData {
    pub peak_pos: i64, // The actual endpoint position
    pub depth: usize,
    pub strand: Strand,
    pub misprime: bool,
}

/// Collector for polyA read endpoints, following the ReadCoverage pattern
pub struct PolyAPeakCollector<'a> {
    // Track read endpoints: contig -> position -> strand -> (cell_barcode -> count)
    ends: HashMap<Box<str>, HashMap<i64, HashMap<Strand, HashMap<CellBarcode, usize>>>>,

    // Track mis-primed reads: contig -> position -> strand -> (cell_barcode -> count)
    misprime_site: HashMap<Box<str>, HashMap<i64, HashMap<Strand, HashMap<CellBarcode, usize>>>>,

    // Configuration
    cell_barcode_tag: &'a str,
    umi_tag: &'a str,
    misprime_a_count: usize, // Minimum A/T count to flag mis-priming
    misprime_in: usize,      // Window size to check for poly-A/T
}

impl VisitWithBamOps for PolyAPeakCollector<'_> {}

impl<'a> PolyAPeakCollector<'a> {
    /// Create a new PolyAPeakCollector
    pub fn new(
        cell_barcode_tag: &'a str,
        umi_tag: &'a str,
        misprime_a_count: usize,
        misprime_in: usize,
    ) -> Self {
        Self {
            ends: HashMap::default(),
            misprime_site: HashMap::default(),
            cell_barcode_tag,
            umi_tag,
            misprime_a_count,
            misprime_in,
        }
    }

    /// Update collector with a BAM record
    pub fn update(&mut self, chr: &str, bam_record: bam::Record) {
        // 1. Extract cell barcode
        let cell_barcode = match bam_record.aux(self.cell_barcode_tag.as_bytes()) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };

        // 2. Skip duplicates (already handled by visitor, but double-check)
        if bam_record.is_duplicate() {
            return;
        }

        // 3. Determine strand
        let strand = if bam_record.is_reverse() {
            Strand::Backward
        } else {
            Strand::Forward
        };

        // 4. Get endpoint position
        let pos = if bam_record.is_reverse() {
            bam_record.pos() as i64 // 0-based start for reverse
        } else {
            (bam_record.reference_end() - 1) as i64 // 0-based end - 1 for forward
        };

        // 5. Check for mis-priming
        let seq = bam_record.seq().as_bytes();
        let misprime = check_misprime(&seq, bam_record.is_reverse(), self.misprime_in, self.misprime_a_count);

        // 6. Update ends count
        let pos_map = self
            .ends
            .entry(chr.into())
            .or_default()
            .entry(pos)
            .or_default()
            .entry(strand.clone())
            .or_default();
        *pos_map.entry(cell_barcode.clone()).or_insert(0) += 1;

        // 7. Update misprime count if detected
        if misprime {
            let misprime_map = self
                .misprime_site
                .entry(chr.into())
                .or_default()
                .entry(pos)
                .or_default()
                .entry(strand)
                .or_default();
            *misprime_map.entry(cell_barcode).or_insert(0) += 1;
        }
    }

    /// Update collector with a BAM record from gene visitor (4-parameter version)
    pub fn update_by_gene(&mut self, gff_record: &GffRecord, _gene_barcode_tag: &str, bam_record: bam::Record) {
        // Use the chromosome from GffRecord
        let chr = gff_record.seqname.as_ref();
        self.update(chr, bam_record);
    }

    /// Create peak regions from collected endpoints
    pub fn create_regions(
        &self,
        depth_threshold: usize,
        region_size: i64,
    ) -> HashMap<Box<str>, HashMap<i64, PeakRegion>> {
        let mut regions: HashMap<Box<str>, HashMap<i64, PeakRegion>> = HashMap::default();

        // Iterate through all contigs
        for (contig, pos_map) in &self.ends {
            let contig_regions = regions.entry(contig.clone()).or_default();

            // Iterate through all positions
            for (pos, strand_map) in pos_map {
                for (strand, cell_counts) in strand_map {
                    // Sum counts across all cells/barcodes
                    let total_count: usize = cell_counts.values().sum();

                    if total_count >= depth_threshold {
                        // Calculate misprime fraction
                        let misprime_count: usize = self
                            .misprime_site
                            .get(contig)
                            .and_then(|pm| pm.get(pos))
                            .and_then(|sm| sm.get(strand))
                            .map(|cm| cm.values().sum())
                            .unwrap_or(0);

                        let misprime = misprime_count >= total_count / 2;

                        // Create region based on strand
                        let (start, end) = match strand {
                            Strand::Forward => ((*pos - region_size).max(0), *pos),
                            Strand::Backward => (*pos, *pos + region_size),
                        };

                        contig_regions.insert(
                            start,
                            PeakRegion {
                                start,
                                end,
                                peak_pos: *pos,
                                depth: total_count,
                                strand: strand.clone(),
                                misprime,
                            },
                        );
                    }
                }
            }
        }

        regions
    }

    /// Resolve overlapping regions, keeping the one with highest depth
    pub fn resolve_overlaps(
        regions: HashMap<Box<str>, HashMap<i64, PeakRegion>>,
    ) -> HashMap<Box<str>, HashMap<i64, PeakRegion>> {
        let mut resolved: HashMap<Box<str>, HashMap<i64, PeakRegion>> = HashMap::default();

        for (contig, region_map) in regions {
            let contig_resolved = resolved.entry(contig).or_default();

            // Process separately by strand
            for strand in [Strand::Forward, Strand::Backward] {
                // Collect regions for this strand
                let mut strand_regions: Vec<_> = region_map
                    .iter()
                    .filter(|(_, r)| r.strand == strand)
                    .map(|(start, region)| (*start, region.clone()))
                    .collect();

                // Sort by start position
                strand_regions.sort_by_key(|(start, _)| *start);

                // Keep best non-overlapping regions
                for i in 0..strand_regions.len() {
                    let (start, region) = &strand_regions[i];
                    let mut is_best = true;

                    // Check upstream overlaps
                    for j in (0..i).rev() {
                        let (_, other) = &strand_regions[j];
                        if other.end < *start {
                            break; // No more overlaps upstream
                        }
                        if other.depth > region.depth {
                            is_best = false;
                            break;
                        }
                    }

                    // Check downstream overlaps
                    if is_best {
                        for j in (i + 1)..strand_regions.len() {
                            let (other_start, other) = &strand_regions[j];
                            if *other_start > region.end {
                                break; // No more overlaps downstream
                            }
                            if other.depth >= region.depth {
                                is_best = false;
                                break;
                            }
                        }
                    }

                    if is_best {
                        contig_resolved.insert(*start, region.clone());
                    }
                }
            }
        }

        resolved
    }

    /// Convert to COITree structures (all peaks)
    pub fn to_coitrees(
        &self,
        depth_threshold: usize,
        region_size: i64,
    ) -> HashMap<Box<str>, COITree<PolyAPeakData, u32>> {
        let mut trees: HashMap<Box<str>, Vec<Interval<PolyAPeakData>>> = HashMap::default();

        // Iterate through all contigs
        for (contig, pos_map) in &self.ends {
            let intervals = trees.entry(contig.clone()).or_default();

            // Iterate through all positions
            for (pos, strand_map) in pos_map {
                for (strand, cell_counts) in strand_map {
                    // Sum counts across all cells/barcodes
                    let total_count: usize = cell_counts.values().sum();

                    if total_count >= depth_threshold {
                        // Calculate misprime fraction
                        let misprime_count: usize = self
                            .misprime_site
                            .get(contig)
                            .and_then(|pm| pm.get(pos))
                            .and_then(|sm| sm.get(strand))
                            .map(|cm| cm.values().sum())
                            .unwrap_or(0);

                        let misprime = misprime_count >= total_count / 2;

                        // Create interval based on strand
                        let (start, stop) = match strand {
                            Strand::Forward => ((*pos - region_size).max(0), *pos),
                            Strand::Backward => (*pos, *pos + region_size),
                        };

                        intervals.push(Interval::new(
                            start as i32,
                            stop as i32,
                            PolyAPeakData {
                                peak_pos: *pos,
                                depth: total_count,
                                strand: strand.clone(),
                                misprime,
                            },
                        ));
                    }
                }
            }
        }

        // Build COITrees from intervals
        trees
            .into_iter()
            .map(|(contig, intervals)| (contig, COITree::new(&intervals)))
            .collect()
    }

    /// Convert to COITree structures with overlap resolution
    pub fn to_coitrees_resolved(
        &self,
        depth_threshold: usize,
        region_size: i64,
    ) -> HashMap<Box<str>, COITree<PolyAPeakData, u32>> {
        // First create regions
        let regions = self.create_regions(depth_threshold, region_size);

        // Then resolve overlaps
        let resolved = Self::resolve_overlaps(regions);

        // Convert to COITree
        let mut trees: HashMap<Box<str>, Vec<Interval<PolyAPeakData>>> = HashMap::default();

        for (contig, region_map) in resolved {
            let intervals = trees.entry(contig).or_default();

            for (_, region) in region_map {
                intervals.push(Interval::new(
                    region.start as i32,
                    region.end as i32,
                    PolyAPeakData {
                        peak_pos: region.peak_pos,
                        depth: region.depth,
                        strand: region.strand,
                        misprime: region.misprime,
                    },
                ));
            }
        }

        trees
            .into_iter()
            .map(|(contig, intervals)| (contig, COITree::new(&intervals)))
            .collect()
    }
}

/// Check if a read is mis-primed based on poly-A/T content at read ends
fn check_misprime(seq: &[u8], is_reverse: bool, misprime_in: usize, misprime_a_count: usize) -> bool {
    let window = if is_reverse {
        &seq[..misprime_in.min(seq.len())]
    } else {
        &seq[seq.len().saturating_sub(misprime_in)..]
    };

    let target = if is_reverse { b'T' } else { b'A' };
    let count = window.iter().filter(|&&b| b == target).count();
    count >= misprime_a_count
}

/// Main processing function: process BAM file and return COITree structures
pub fn process_polya_ends_to_peaks(
    bam_file: &str,
    depth_threshold: usize,
    region_size: i64,
    cell_barcode_tag: &str,
    umi_tag: &str,
    misprime_a_count: usize,
    misprime_in: usize,
    resolve_overlaps: bool,
) -> anyhow::Result<HashMap<Box<str>, COITree<PolyAPeakData, u32>>> {
    use crate::data::util_htslib::create_bam_jobs;

    log::info!("Processing polyA ends from {}", bam_file);

    // 1. Create BAM processing jobs (similar to run_read_depth)
    let jobs = create_bam_jobs(bam_file, None, None)?;
    let njobs = jobs.len() as u64;
    log::info!("Processing {} genomic regions", njobs);

    // 2. Create collector and process BAM file
    let mut collector = PolyAPeakCollector::new(cell_barcode_tag, umi_tag, misprime_a_count, misprime_in);

    // Process each region
    for (chr, start, stop) in &jobs {
        let bed = Bed {
            chr: chr.clone(),
            start: *start,
            stop: *stop,
        };

        // Visit BAM file and collect endpoints
        collector.visit_bam_by_region(bam_file, &bed, &PolyAPeakCollector::update)?;
    }

    log::info!("Creating COITree structures...");
    // 3. Convert to COITree (with or without overlap resolution)
    let trees = if resolve_overlaps {
        collector.to_coitrees_resolved(depth_threshold, region_size)
    } else {
        collector.to_coitrees(depth_threshold, region_size)
    };

    // 4. Print statistics
    let total_peaks: usize = trees.values().map(|tree| tree.len() as usize).sum();
    log::info!(
        "Found {} polyA peaks across {} contigs",
        total_peaks,
        trees.len()
    );

    Ok(trees)
}

// /// Optional: Helper function to process polyA peaks within specific gene regions
// pub fn process_polya_in_gene(
//     bam_file: &str,
//     gff_record: &GffRecord,
//     gene_barcode_tag: &str,
//     depth_threshold: usize,
//     region_size: i64,
//     cell_barcode_tag: &str,
//     umi_tag: &str,
//     misprime_a_count: usize,
//     misprime_in: usize,
// ) -> anyhow::Result<Vec<Interval<PolyAPeakData>>> {
//     let mut collector = PolyAPeakCollector::new(cell_barcode_tag, umi_tag, misprime_a_count, misprime_in);

//     // Process BAM file within gene region
//     collector.visit_bam_by_gene(bam_file, gff_record, gene_barcode_tag, &PolyAPeakCollector::update_by_gene)?;

//     // Convert to intervals for this gene
//     let trees = collector.to_coitrees_resolved(depth_threshold, region_size);

//     // Extract intervals for the gene's chromosome by querying the entire range
//     let mut intervals = Vec::new();
//     if let Some(tree) = trees.get(&gff_record.seqname) {
//         tree.query(gff_record.start as i32, gff_record.stop as i32, |interval| {
//             intervals.push(Interval::new(interval.first, interval.last, *interval.metadata));
//         });
//     }

//     Ok(intervals)
// }
