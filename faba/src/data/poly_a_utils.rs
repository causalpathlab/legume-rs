use rust_htslib::bam;
use rust_htslib::bam::record::Cigar;

///////////////////////////
// Unpack the CIGAR once //
///////////////////////////
// `bam::Record::cigar()` decodes the packed ops into a fresh `Vec<Cigar>` on
// every call, and `cigar_cached()` only answers after an explicit
// `cache_cigar()` on a `&mut Record`, which a visitor holding `&Record` cannot
// do. So every helper that re-derived the CIGAR from the record cost one
// malloc+free per read, on every read of every region. The `_from_cigar` forms
// below borrow a CIGAR the caller unpacked once; the record-taking forms stay
// for callers that hold no CIGAR.

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
/// A/T bases in the soft-clipped tail, read off a CIGAR the caller already unpacked.
pub fn count_a_or_t_bases_in_tail_from_cigar(cigar: &[Cigar], bam_record: &bam::Record) -> usize {
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

/// Is this read internally primed?
///
/// Checks poly-A/T content at the end of the aligned sequence, which catches a
/// tail templated by a genomic A-rich or T-rich stretch rather than transcribed.
/// Takes a CIGAR the caller already unpacked, because `Record::cigar()`
/// allocates on every call.
///
/// * cigar: the record's unpacked CIGAR
/// * bam_record: The BAM record to check
/// * misprime_in: Number of bases at the end of alignment to check
/// * misprime_a_count: Minimum number of A/T bases to consider internal priming
pub fn check_internal_prime_from_cigar(
    cigar: &[Cigar],
    bam_record: &bam::Record,
    misprime_in: usize,
    misprime_a_count: usize,
) -> bool {
    // Extract aligned sequence (excluding soft-clipped regions)
    let seq = bam_record.seq().as_bytes();
    let (aligned_start, aligned_end) = get_query_alignment_bounds(cigar, seq.len());
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

/// Get the soft-clip (poly-A/T tail) length from an already-unpacked CIGAR.
/// The tail sits at the CIGAR's 5' end on a reverse read and its 3' end on a forward one.
pub fn get_polya_tail_length(cigar: &[Cigar], is_reverse: bool) -> i64 {
    if is_reverse {
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

///////////////////////////
// Reference-side blocks //
///////////////////////////

/// The read's aligned blocks in htslib's frame: 0-based, half-open `[start, stop)`, ascending.
///
/// Same walk as `BamRecordExtensions::aligned_blocks`, op for op.
/// `M`/`=`/`X` emit a block and advance the reference.
/// `D` and `N` advance it without emitting, so a deletion splits the read just as a splice does.
/// `I`/`S`/`H`/`P` move nothing, and htslib still closes the block at them.
/// So `10M5I10M` reports two abutting blocks, not one.
///
/// The one difference is the allocation.
/// htslib's version calls `Record::cigar()` internally and clones the ops per read.
pub fn aligned_blocks(
    cigar: &[Cigar],
    reference_start: i64,
) -> impl Iterator<Item = [i64; 2]> + '_ {
    let mut pos = reference_start;
    cigar.iter().filter_map(move |op| match *op {
        Cigar::Match(len) | Cigar::Equal(len) | Cigar::Diff(len) => {
            let start = pos;
            pos += len as i64;
            Some([start, pos])
        }
        Cigar::Del(len) | Cigar::RefSkip(len) => {
            pos += len as i64;
            None
        }
        _ => None,
    })
}
