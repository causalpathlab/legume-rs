#![allow(dead_code)]

use rust_htslib::bam;
use rust_htslib::bam::ext::BamRecordExtensions;
use rust_htslib::bam::record::Cigar;

/// Get the query alignment boundaries from CIGAR string
/// Returns (start, end) positions in the query sequence (0-based, half-open)
pub fn get_query_alignment_bounds(cigar: &[Cigar], seq_len: usize) -> (usize, usize) {
    let mut start = 0;
    let mut end = seq_len;

    // Calculate aligned start (skip leading soft clips)
    // Note: HardClip bases are NOT in the query sequence, so we only handle SoftClip
    for op in cigar.iter() {
        match op {
            Cigar::SoftClip(len) => start += *len as usize,
            Cigar::HardClip(_) => {}
            _ => break,
        }
    }

    // Calculate aligned end (skip trailing soft clips)
    for op in cigar.iter().rev() {
        match op {
            Cigar::SoftClip(len) => end = end.saturating_sub(*len as usize),
            Cigar::HardClip(_) => {}
            _ => break,
        }
    }

    (start.min(end), end)
}

/// Count the number of A (forward) or T (reverse) bases in the soft-clipped region
/// Returns the count of A/T bases
pub fn count_a_or_t_bases_in_tail(bam_record: &bam::Record) -> usize {
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
pub fn check_internal_prime(
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
    let count = window.iter().filter(|&&b| b == target).count();
    count >= misprime_a_count
}

/// Get the soft-clip (poly-A/T tail) length from a BAM record's CIGAR.
/// Returns the length and whether it's at the forward (3') or reverse (5') end.
pub fn get_polya_tail_length(bam_record: &bam::Record) -> i64 {
    let cigar = bam_record.cigar();
    if bam_record.is_reverse() {
        match cigar.first() {
            Some(Cigar::SoftClip(len)) => *len as i64,
            _ => 0,
        }
    } else {
        match cigar.last() {
            Some(Cigar::SoftClip(len)) => *len as i64,
            _ => 0,
        }
    }
}

/// Calculate the genomic position of a poly-A cleavage site from a BAM record.
pub fn get_polya_position(bam_record: &bam::Record) -> i64 {
    if bam_record.is_reverse() {
        bam_record.reference_start()
    } else {
        bam_record.reference_end()
    }
}
