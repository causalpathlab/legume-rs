//! Per-cell-type read-depth track from BAM, for the Miami bottom track.
//!
//! Streams reads in the gene region (reusing faba's cached BAM reader),
//! maps each read's cell barcode to a cell type, and accumulates per-base
//! coverage of the read's **aligned blocks** (so introns are not counted —
//! same convention as the splice-aware counter) into the SAME [`BinEdges`]
//! grid the top track uses. Multiple BAMs are pooled into one set of
//! per-cell-type bins.

use super::bin::BinEdges;
use crate::data::bam_io;
use crate::data::cell_membership::CellMembership;
use genomic_data::bed::Bed;
use rust_htslib::bam::ext::BamRecordExtensions;
use rustc_hash::FxHashMap;

/// Group label used for the single track when no membership is supplied.
pub const ALL_CELLS: &str = "";

/// Add the bp-overlap of one aligned block `[bs, be)` (0-based half-open,
/// genomic) to each bin it touches. Pure (no BAM) so it is unit-testable.
pub fn accumulate_block(bins: &mut [f64], edges: &BinEdges, bs: i64, be: i64) {
    let lo = bs.max(edges.min_pos);
    let hi = be.min(edges.max_pos + 1);
    if lo >= hi {
        return;
    }
    let span = edges.span();
    let nbins = edges.num_bins as u64;
    // Genomic interval covered by bin `c` (matches the TSV bin convention
    // in pileup.rs::write_pileup_tsv); last bin extends to max_pos+1.
    let bin_lo = |c: u64| edges.min_pos + (c * span / nbins) as i64;
    let bin_hi = |c: u64| {
        if c + 1 == nbins {
            edges.max_pos + 1
        } else {
            edges.min_pos + ((c + 1) * span / nbins) as i64
        }
    };
    let c0 = edges.col_of(lo) as u64;
    let c1 = edges.col_of(hi - 1) as u64;
    for c in c0..=c1 {
        let overlap = hi.min(bin_hi(c)) - lo.max(bin_lo(c));
        if overlap > 0 {
            bins[c as usize] += overlap as f64;
        }
    }
}

/// Per-cell-type binned read depth over `region`. Returns `group ->
/// binned coverage` on the shared `edges` grid. With `membership = None`,
/// all reads land in the single [`ALL_CELLS`] group.
pub fn read_depth_binned(
    bam_files: &[Box<str>],
    region: &Bed,
    edges: &BinEdges,
    cell_barcode_tag: &str,
    membership: Option<&CellMembership>,
) -> anyhow::Result<FxHashMap<Box<str>, Vec<f64>>> {
    let mut by_group: FxHashMap<Box<str>, Vec<f64>> = FxHashMap::default();
    let tag = cell_barcode_tag.as_bytes();

    for bam in bam_files {
        bam_io::for_each_record_in_region(bam, region, |_chr, rec| {
            let group: Box<str> = match membership {
                None => ALL_CELLS.into(),
                Some(m) => {
                    let bc = bam_io::extract_cell_barcode(rec, tag);
                    match m.matches_barcode(&bc) {
                        Some(ct) => ct,
                        None => return, // unbarcoded / not in membership
                    }
                }
            };
            let bins = by_group
                .entry(group)
                .or_insert_with(|| vec![0.0; edges.num_bins]);
            for [bs, be] in rec.aligned_blocks() {
                accumulate_block(bins, edges, bs, be);
            }
        })?;
    }

    if let Some(m) = membership {
        let (matched, total) = m.match_stats();
        log::info!(
            "read-depth: matched {}/{} barcoded reads to {} cell type(s)",
            matched,
            total,
            by_group.len()
        );
    }

    Ok(by_group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_bp_overlap_splits_across_bins() {
        // span = max-min = 100, 10 bins -> 10bp bins ([0,10),[10,20),...).
        let edges = BinEdges::new(0, 100, 10);
        let mut bins = vec![0.0; 10];
        // block [5, 25): 5bp in bin0, 10bp in bin1, 5bp in bin2.
        accumulate_block(&mut bins, &edges, 5, 25);
        assert!((bins[0] - 5.0).abs() < 1e-9, "{:?}", bins);
        assert!((bins[1] - 10.0).abs() < 1e-9, "{:?}", bins);
        assert!((bins[2] - 5.0).abs() < 1e-9, "{:?}", bins);
        // total bp == clipped block length, regardless of bin boundaries
        assert!((bins.iter().sum::<f64>() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn block_clipped_to_extent() {
        let edges = BinEdges::new(0, 99, 10);
        let mut bins = vec![0.0; 10];
        // block extends past both ends; only [0,100) of it counts -> 100bp.
        accumulate_block(&mut bins, &edges, -50, 500);
        assert!((bins.iter().sum::<f64>() - 100.0).abs() < 1e-9);
        // fully outside -> nothing
        let mut bins2 = vec![0.0; 10];
        accumulate_block(&mut bins2, &edges, 500, 600);
        assert_eq!(bins2.iter().sum::<f64>(), 0.0);
    }
}
