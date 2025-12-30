use crate::data::dna_stat_traits::DnaStatMap;
use crate::data::sam::CellBarcode;
use crate::data::visitors_htslib::VisitWithBamOps;

use fnv::FnvHashMap as HashMap;
use rust_htslib::bam::ext::BamRecordExtensions;
use rust_htslib::bam::record::Cigar;
use rust_htslib::bam::{self, record::Aux};

pub struct PolyASiteSifter<'a> {
    forward_polya_sites: HashMap<CellBarcode, HashMap<Box<str>, HashMap<u64, usize>>>,
    backward_polya_sites: HashMap<CellBarcode, HashMap<Box<str>, HashMap<u64, usize>>>,
    forward_internal_prime_count: HashMap<Box<str>, HashMap<u64, usize>>,
    backward_internal_prime_count: HashMap<Box<str>, HashMap<u64, usize>>,
    internal_prime_in: usize,
    internal_prime_a_or_t_count: usize,
    min_tail_length: usize,
    max_non_a_or_t_bases: usize,
    cell_barcode_tag: &'a str,
}

impl VisitWithBamOps for PolyASiteSifter<'_> {}

impl DnaStatMap for PolyASiteSifter<'_> {
    fn add_bam_record(&mut self, bam_record: bam::Record) {
        unimplemented!("");
    }
    fn sorted_positions(&self) -> Vec<i64> {
        unimplemented!("");
    }
}

impl<'a> PolyASiteSifter<'a> {
    /// Create a new PolyACoverage analyzer
    ///
    /// # Parameters
    ///
    /// * `cell_barcode_tag` - BAM tag name for cell barcode (e.g., "CB")
    /// * `internal_prime_in` - Number of bases at the end of aligned sequence to check for internal priming
    /// * `internal_prime_a_or_t_count` - Minimum number of A (forward) or T (reverse) bases required to flag as internal priming
    /// * `min_tail_length` - Minimum poly-A/T length required (based on CIGAR soft-clip length)
    /// * `max_non_a_or_t_bases` - Maximum number of non-A (forward) or non-T (reverse) bases allowed in the soft-clipped region
    ///
    /// # Returns
    ///
    /// A new PolyACoverage instance that tracks poly-A/T coverage and detects internal priming
    pub fn new(
        cell_barcode_tag: &'a str,
        internal_prime_in: usize,
        internal_prime_a_or_t_count: usize,
        min_tail_length: usize,
        max_non_a_or_t_bases: usize,
    ) -> Self {
        Self {
            forward_polya_sites: HashMap::default(),
            backward_polya_sites: HashMap::default(),
            forward_internal_prime_count: HashMap::default(),
            backward_internal_prime_count: HashMap::default(),
            internal_prime_in,
            internal_prime_a_or_t_count,
            min_tail_length,
            max_non_a_or_t_bases,
            cell_barcode_tag,
        }
    }

    // pub fn collect_from_bam(
    //     &mut self,
    //     bam_file_path: &str,
    //     bed: &crate::data::bed::Bed,
    // ) -> anyhow::Result<()> {
    //     // self.visit_bam_by_region(bam_file_path, bed, &Self::update)
    // }

    // pub update_by_gene(

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
    pub fn update(&mut self, chr: &str, bam_record: bam::Record) {
        let cell_barcode = match bam_record.aux(self.cell_barcode_tag.as_bytes()) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };

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
        if poly_length < self.min_tail_length as i64 {
            return;
        }

        // Count A (forward) or T (reverse) bases in the soft-clipped
        // region to filter by maximum non-A/non-T bases allowed in
        // the tail
        let num_a_or_t = count_a_or_t_bases_in_tail(&bam_record);
        let non_a_or_t_count = poly_length as usize - num_a_or_t;
        if non_a_or_t_count > self.max_non_a_or_t_bases {
            return;
        }

        // Check for internal priming in the aligned sequence
        let is_internally_primed = check_internal_prime(
            &bam_record,
            self.internal_prime_in,
            self.internal_prime_a_or_t_count,
        );

        // Calculate genomic position for poly-A site
        let poly_a_position = if bam_record.is_reverse() {
            // Reverse: poly-T ends at reference_start()
            bam_record.reference_start() as u64
        } else {
            // Forward: poly-A starts at reference_end()
            bam_record.reference_end() as u64
        };

        if is_internally_primed {
            // Track internal priming count by position
            let internal_prime_count = if bam_record.is_reverse() {
                &mut self.backward_internal_prime_count
            } else {
                &mut self.forward_internal_prime_count
            };

            let chr_pos_count = internal_prime_count.entry(chr.into()).or_default();
            *chr_pos_count.entry(poly_a_position).or_default() += 1;
        } else {
            // Track poly-A site position by cell barcode and chromosome
            let polya_sites = if bam_record.is_reverse() {
                &mut self.backward_polya_sites
            } else {
                &mut self.forward_polya_sites
            };

            let chr_to_sites = polya_sites.entry(cell_barcode).or_default();
            let pos_count = chr_to_sites.entry(chr.into()).or_default();
            *pos_count.entry(poly_a_position).or_default() += 1;
        }
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
