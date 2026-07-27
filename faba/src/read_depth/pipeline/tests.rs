use crate::data::util_htslib::contig_blocks;
use crate::read_depth::pipeline::{bin_aligned_block_size, bin_edges};

/// A contig length that is a multiple of no bin size under test, so the final
/// bin is genuinely short and the "except the last one" clause is exercised.
const CONTIG_LEN: i64 = 10_500_123;

/// The bins `faba depth` would emit for one contig, in order: block the contig
/// exactly as [`crate::data::util_htslib::create_bam_jobs`] does, then tile each
/// block the way the counting closure does.
fn bins(resolution_kb: f32, block_size_mb: usize) -> (usize, Vec<(usize, usize)>) {
    let segment_size = (resolution_kb * 1000.0) as usize;
    let block_size = bin_aligned_block_size(block_size_mb * 1_000_000, segment_size);

    let bins = contig_blocks(CONTIG_LEN, block_size as i64, 0)
        .into_iter()
        .flat_map(|(lb, ub)| bin_edges(lb as usize, ub as usize, segment_size).collect::<Vec<_>>())
        .collect();

    (segment_size, bins)
}

/// Bins are the feature vocabulary, and a short bin is indistinguishable from a
/// full one once it is named `{chr}:{start}-{end}` -- it just reports ~1/3 the
/// depth. Tiling from each block's own `lb` produced one such bin per block
/// whenever the block was not a whole number of bins.
#[test]
fn every_bin_is_full_width_except_the_last_of_a_contig() {
    // Deliberately non-dividing pairs first: 1 Mb / 3 kb = 333.33 bins,
    // 3 Mb / 7 kb = 428.57, 1 Mb / 250 kb divides but 1 Mb / 300 kb does not.
    for (resolution_kb, block_size_mb) in [
        (3.0f32, 1usize),
        (7.0, 3),
        (300.0, 1),
        (0.5, 2),
        (10.0, 1),
        (250.0, 1),
    ] {
        let (segment_size, bins) = bins(resolution_kb, block_size_mb);
        let label = format!("-r {} -b {}", resolution_kb, block_size_mb);
        assert!(!bins.is_empty(), "{label}: no bins");

        let mut expected_lb = 0usize;
        for (i, &(lb, ub)) in bins.iter().enumerate() {
            assert_eq!(lb, expected_lb, "{label}: bins must be contiguous");
            assert_eq!(
                lb % segment_size,
                0,
                "{label}: bin {lb} is off the genome-absolute grid"
            );

            let width = ub - lb;
            if i + 1 == bins.len() {
                // The one truncation everybody keeps: the contig ends here.
                assert!(width > 0 && width <= segment_size, "{label}: last bin");
            } else {
                assert_eq!(width, segment_size, "{label}: short bin at {lb}");
            }
            expected_lb = ub;
        }

        assert_eq!(
            expected_lb, CONTIG_LEN as usize,
            "{label}: bins must cover the contig"
        );
    }
}

/// `--block-size-mb` is a parallelism knob. If it moved the bin edges, two runs
/// of the same BAM at different block sizes would produce feature vocabularies
/// that cannot be compared or concatenated.
#[test]
fn the_bin_grid_does_not_depend_on_the_block_size() {
    let (_, reference) = bins(3.0, 1);
    for block_size_mb in [2usize, 5, 7, 64] {
        let (_, other) = bins(3.0, block_size_mb);
        assert_eq!(
            reference, other,
            "-b {block_size_mb} moved the bin grid at -r 3"
        );
    }
}

/// A block smaller than a bin would otherwise emit nothing but truncated bins.
#[test]
fn a_block_is_always_at_least_one_whole_bin() {
    assert_eq!(bin_aligned_block_size(1_000_000, 3_000), 1_002_000);
    assert_eq!(bin_aligned_block_size(1_000_000, 250_000), 1_000_000);
    assert_eq!(bin_aligned_block_size(0, 3_000), 3_000);
    assert_eq!(bin_aligned_block_size(1_000, 3_000), 3_000);
}
