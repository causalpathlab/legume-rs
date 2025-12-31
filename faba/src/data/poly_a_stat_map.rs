#![allow(dead_code)]

use crate::data::dna_stat_traits::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;
use fnv::FnvHashMap as HashMap;
use rust_htslib::bam::record::Cigar;
use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

pub struct PolyASiteMap {
    position_to_count_with_cell: Option<HashMap<i64, HashMap<CellBarcode, usize>>>,
    position_to_count: Option<HashMap<i64, usize>>,
    internal_prime_to_count: HashMap<i64, usize>,
    qc_args: PolyASiteArgs,
    cell_barcode_tag: Option<Vec<u8>>,
}

#[derive(Clone)]
pub struct PolyASiteArgs {
    pub min_tail_length: usize,
    pub max_non_a_or_t_bases: usize,
    pub internal_prime_in: usize,
    pub internal_prime_a_or_t_count: usize,
}

impl PolyASiteMap {
    pub fn new_with_cell_barcode(qc_args: PolyASiteArgs, cell_barcode_tag: &str) -> Self {
        Self {
            position_to_count_with_cell: Some(HashMap::default()),
            position_to_count: None,
            internal_prime_to_count: HashMap::default(),
            qc_args: qc_args.clone(),
            cell_barcode_tag: Some(cell_barcode_tag.as_bytes().to_vec()),
        }
    }

    pub fn new(qc_args: PolyASiteArgs) -> Self {
        Self {
            position_to_count_with_cell: None,
            position_to_count: Some(HashMap::default()),
            internal_prime_to_count: HashMap::default(),
            qc_args: qc_args.clone(),
            cell_barcode_tag: None,
        }
    }
}

impl VisitWithBamOps for PolyASiteMap {}

impl DnaStatMap for PolyASiteMap {
    /// Update frequency maps
    ///
    /// Parse CIGAR string and get potential poly-A cleavage sites
    /// - mark cleavage sites
    /// - mark internally primed sites (later filter out)
    /// while filtering out reads by:
    /// - soft-clipped length
    /// - need to consider #A's or #T's within the soft-clipped
    ///
    /// We will keep track of "S" here:
    ///
    /// FORWARD
    /// NNNNNNNNNNNAAAAAAAAAAAAAAA [read]
    /// |||||||||||............... [match]
    /// NNNNNNNNNNNS-------------- [reference sequence]
    ///
    /// or
    ///
    /// BACKWARD
    /// TTTTTTTTTTTTTTTTNNNNNNNNNN [read]
    /// ................|||||||||| [match]
    /// ---------------SNNNNNNNNNN [reference sequence]
    ///
    fn add_bam_record(&mut self, bam_record: bam::Record) {
        // Get the soft-clipped poly-A/T length from CIGAR
        let cigar = bam_record.cigar();
        let poly_length = if bam_record.is_reverse() {
            // REVERSE: poly-T at the beginning (first CIGAR op)
            match cigar.first() {
                Some(Cigar::SoftClip(len)) => *len as i64,
                _ => 0,
            }
        } else {
            // FORWARD: poly-A at the end (last CIGAR op)
            match cigar.last() {
                Some(Cigar::SoftClip(len)) => *len as i64,
                _ => 0,
            }
        };

        // Filter by minimum poly-A/T length (based on CIGAR soft-clip)
        if poly_length < self.qc_args.min_tail_length as i64 {
            return;
        }

        // Count A (forward) or T (reverse) bases in the soft-clipped
        // region to filter by maximum non-A/non-T bases allowed in the tail
        let num_a_or_t = count_a_or_t_bases_in_tail(&bam_record);
        let non_a_or_t_count = poly_length as usize - num_a_or_t;
        if non_a_or_t_count > self.qc_args.max_non_a_or_t_bases {
            return;
        }

        // Check for internal priming in the aligned sequence
        let is_internally_primed = check_internal_prime(
            &bam_record,
            self.qc_args.internal_prime_in,
            self.qc_args.internal_prime_a_or_t_count,
        );

        // Calculate genomic position for poly-A site
        let poly_a_position = if bam_record.is_reverse() {
            // Reverse: poly-T ends at reference_start()
            bam_record.reference_start()
        } else {
            // Forward: poly-A starts at reference_end()
            bam_record.reference_end()
        };

        if is_internally_primed {
            // Track internal priming count by position
            *self
                .internal_prime_to_count
                .entry(poly_a_position)
                .or_default() += 1;
        } else {
            // Records with cell barcode information
            if let (Some(tag), Some(freq_map)) = (
                &self.cell_barcode_tag,
                &mut self.position_to_count_with_cell,
            ) {
                let cell_barcode = match bam_record.aux(tag) {
                    Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
                    _ => CellBarcode::Missing,
                };

                let cell_map = freq_map.entry(poly_a_position).or_default();
                *cell_map.entry(cell_barcode).or_default() += 1;
            }

            // just keeping the marginal frequencies
            if let Some(map) = &mut self.position_to_count {
                *map.entry(poly_a_position).or_default() += 1;
            }
        }
    }
    fn sorted_positions(&self) -> Vec<i64> {
        let mut positions: Vec<i64> = if let Some(map) = &self.position_to_count_with_cell {
            map.keys().copied().collect()
        } else if let Some(map) = &self.position_to_count {
            map.keys().copied().collect()
        } else {
            Vec::new()
        };
        positions.sort_unstable();
        positions
    }
}

impl PolyASiteMap {
    /// Get cell-level counts at a specific position
    /// Returns None if position doesn't exist or cell barcode tracking is not enabled
    pub fn get_cell_counts_at(&self, position: i64) -> Option<&HashMap<CellBarcode, usize>> {
        self.position_to_count_with_cell
            .as_ref()
            .and_then(|map| map.get(&position))
    }

    /// Get total count at a specific position (aggregated across all cells)
    /// Returns 0 if position doesn't exist
    pub fn get_total_count_at(&self, position: i64) -> usize {
        if let Some(map) = &self.position_to_count_with_cell {
            map.get(&position)
                .map(|cell_map| cell_map.values().sum())
                .unwrap_or(0)
        } else if let Some(map) = &self.position_to_count {
            map.get(&position).copied().unwrap_or(0)
        } else {
            0
        }
    }

    /// Get all positions with their total counts
    /// Returns a vector of (position, count) tuples sorted by position
    pub fn positions_with_counts(&self) -> Vec<(i64, usize)> {
        let mut result: Vec<(i64, usize)> = if let Some(map) = &self.position_to_count_with_cell {
            map.iter()
                .map(|(pos, cell_map)| (*pos, cell_map.values().sum()))
                .collect()
        } else if let Some(map) = &self.position_to_count {
            map.iter().map(|(pos, count)| (*pos, *count)).collect()
        } else {
            Vec::new()
        };
        result.sort_by_key(|(pos, _)| *pos);
        result
    }
}

/// Get the query alignment boundaries from CIGAR string
/// Returns (start, end) positions in the query sequence (0-based, half-open)
fn get_query_alignment_bounds(cigar: &[Cigar], seq_len: usize) -> (usize, usize) {
    let mut start = 0;
    let mut end = seq_len;

    // Calculate aligned start (skip leading soft/hard clips)
    for op in cigar.iter() {
        match op {
            Cigar::SoftClip(len) | Cigar::HardClip(len) => start += *len as usize,
            _ => break,
        }
    }

    // Calculate aligned end (skip trailing soft/hard clips)
    for op in cigar.iter().rev() {
        match op {
            Cigar::SoftClip(len) | Cigar::HardClip(len) => end -= *len as usize,
            _ => break,
        }
    }

    (start, end)
}

/// Count the number of A (forward) or T (reverse) bases in the soft-clipped region
/// Returns the count of A/T bases
fn count_a_or_t_bases_in_tail(bam_record: &bam::Record) -> usize {
    let cigar = bam_record.cigar();
    let is_reverse = bam_record.is_reverse();

    // Get soft-clip length from CIGAR
    let soft_clip_len = if is_reverse {
        // REVERSE: poly-T at the beginning (first CIGAR op)
        match cigar.first() {
            Some(Cigar::SoftClip(len)) => *len as usize,
            _ => return 0,
        }
    } else {
        // FORWARD: poly-A at the end (last CIGAR op)
        match cigar.last() {
            Some(Cigar::SoftClip(len)) => *len as usize,
            _ => return 0,
        }
    };

    let seq = bam_record.seq().as_bytes();
    let soft_clip_len = soft_clip_len.min(seq.len());

    let soft_clip_seq = if is_reverse {
        // REVERSE: poly-T at the beginning (5' end)
        &seq[..soft_clip_len]
    } else {
        // FORWARD: poly-A at the end (3' end)
        &seq[seq.len().saturating_sub(soft_clip_len)..]
    };

    let target = if is_reverse { b'T' } else { b'A' };
    soft_clip_seq
        .iter()
        .filter(|&&b| b == target || b == target.to_ascii_lowercase())
        .count()
}

/// Check if a read is internally-primed based on poly-A/T content at
/// the end of the aligned sequence (checks for genomic A-rich or T-rich regions)
///
/// * bam_record: The BAM record to check
/// * misprime_in: Number of bases at the end of alignment to check
/// * misprime_a_count: Minimum number of A/T bases to consider internal priming
fn check_internal_prime(
    bam_record: &bam::Record,
    misprime_in: usize,
    misprime_a_count: usize,
) -> bool {
    // Extract aligned sequence (excluding soft-clipped regions)
    let seq = bam_record.seq().as_bytes();
    let cigar = bam_record.cigar();
    let (aligned_start, aligned_end) = get_query_alignment_bounds(&cigar, seq.len());
    let aligned_seq = &seq[aligned_start..aligned_end];

    // For forward reads: check the 3' end of the aligned sequence (before poly-A tail)
    // For reverse reads: check the 5' end of the aligned sequence (before poly-T tail)
    let is_reverse = bam_record.is_reverse();
    let window = if is_reverse {
        &aligned_seq[..misprime_in.min(aligned_seq.len())]
    } else {
        &aligned_seq[aligned_seq.len().saturating_sub(misprime_in)..]
    };

    let target = if is_reverse { b'T' } else { b'A' };
    let count = window.into_iter().filter(|&&b| b == target).count();
    count >= misprime_a_count
}
