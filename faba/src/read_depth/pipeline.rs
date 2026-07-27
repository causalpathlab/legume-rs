use crate::common::*;
use crate::data::util_htslib::*;
use crate::read_depth::coverage::ReadCoverageCollector;
use crate::read_depth::run::ReadDepthArgs;

use coitrees::IntervalTree;
use genomic_data::bed::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Called cells per BAM file, keyed by BAM path exactly like
/// [`crate::quant::GeneCountQc::cells_by_batch`] -- a cell set is only meaningful
/// within the library that produced it.
pub type ValidCellsByBam = FxHashMap<Box<str>, FxHashSet<CellBarcode>>;

pub fn run_read_depth_pipeline(args: &ReadDepthArgs) -> anyhow::Result<()> {
    // `faba depth` calls no cells of its own, so the only standalone source of a
    // keep-set is `faba genes` output the user points at.
    let from_flag = match args.valid_cells_file.as_deref() {
        Some(dir) => Some(crate::quant::load_valid_cells_dir(dir, &args.bam_files)?),
        None => None,
    };

    run_read_depth_pipeline_with_cells(args, from_flag.as_ref())
}

/// [`run_read_depth_pipeline`] with the cell sets supplied directly, for callers
/// that already called cells in memory (the `faba all` pipeline) and should not
/// have to round-trip them through `{batch}_cells.tsv.gz`.
pub fn run_read_depth_pipeline_with_cells(
    args: &ReadDepthArgs,
    valid_cells: Option<&ValidCellsByBam>,
) -> anyhow::Result<()> {
    let batch_names = uniq_batch_names(&args.bam_files)?;
    std::fs::create_dir_all(args.output.as_ref())?;

    let backend = args.backend.clone();
    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    let segment_size = (args.resolution_kb * 1000.0) as usize;
    let block_size = bin_aligned_block_size(args.block_size_mb * 1_000_000, segment_size);

    for (bam_file, batch_name) in args.bam_files.iter().zip(&batch_names) {
        // Borrowed for the whole BAM and handed to every block job; each job's
        // collector drops non-cells before it ever stores an interval.
        let keep_cells = valid_cells.and_then(|m| m.get(bam_file));
        match keep_cells {
            Some(cells) => info!("{}: restricted to {} cells", batch_name, cells.len()),
            None => info!("{}: every observed barcode is kept", batch_name),
        }

        // build a coitree for each chromosome, each cell barcode, and
        // each coitree can keep track of coverage
        let jobs = create_bam_jobs(bam_file, Some(block_size), Some(0))?;
        let njobs = jobs.len() as u64;
        info!("Combining reads for {} over {} blocks", batch_name, njobs);

        let segment_stats = jobs
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .map(
                |(chr, lb, ub)| -> anyhow::Result<Vec<(CellBarcode, Box<str>, f32)>> {
                    let bed = Bed {
                        chr: chr.clone(),
                        start: *lb,
                        stop: *ub,
                    };

                    let mut read_coverage = ReadCoverageCollector::new(&args.cell_barcode_tag);
                    read_coverage.set_keep_cells(keep_cells);
                    read_coverage.collect_from_bam(bam_file, &bed)?;

                    let coverage_interval_tree = read_coverage.to_coitrees();

                    // define segments as specified by the resolution parameter
                    let start = *lb as usize;
                    let stop = *ub as usize;

                    // now count them all
                    let mut ret = vec![];
                    for (cb, chr_tree) in coverage_interval_tree {
                        for (chr, tree) in chr_tree {
                            for (lb, ub) in bin_edges(start, stop, segment_size) {
                                // `ub` is an exclusive segment end; the tree is
                                // end-inclusive, so query up to `ub - 1` or the
                                // segment reaches one base into its neighbour.
                                let nn = tree.query_count(lb as i32, ub as i32 - 1);
                                if nn > 0 {
                                    let feature = format!("{}:{}-{}", chr, lb, ub);
                                    ret.push((cb.clone(), feature.into_boxed_str(), nn as f32));
                                }
                            }
                        }
                    }

                    Ok(ret)
                },
            )
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        info!(
            "constructing backend data with {} segments",
            segment_stats.len()
        );

        // `{batch}_depth`, not a bare `{batch}`. Every other modality names its
        // matrix `{batch}_{modality}`, and inside a shared `faba all` output
        // directory an unsuffixed file is ambiguous -- it reads like the
        // batch's primary matrix rather than one modality among several.
        let name = format!("{batch_name}_depth");
        let out = crate::quant::BackendOutputPath::new(&args.output, &name, &backend, args.zip);

        format_data_triplets(segment_stats)
            .to_backend(&out.write_path)?
            .qc(cutoffs.clone())?;

        out.finalize()?;
    }

    info!("done");
    Ok(())
}

//////////////////
// The bin grid //
//////////////////

/// Round a block size UP to a whole number of `segment_size` bins.
///
/// [`create_bam_jobs`] starts every block at a multiple of the block size, so a
/// block that is a whole number of bins puts every block boundary on a bin edge.
/// The grid is then genome-absolute -- bins at `k * segment_size` from position 0
/// of each contig -- and no bin straddles a block, so blocks need no cross-block
/// aggregation and stay independently countable.
///
/// This matters because `--block-size-mb` is a PARALLELISM knob. Tiling from each
/// block's own `lb` instead made the feature vocabulary depend on it: `-r 3 -b 1`
/// ended all ~3000 blocks with a 1 kb bin named exactly like a 3 kb one, so two
/// runs at different block sizes were not comparable.
fn bin_aligned_block_size(block_size: usize, segment_size: usize) -> usize {
    // At least one whole bin, or a block would emit nothing but truncated bins.
    block_size.div_ceil(segment_size).max(1) * segment_size
}

/// Bin edges `[lb, ub)` tiling `[start, stop)` at `segment_size`.
///
/// The last bin is truncated at `stop`. With a bin-aligned block size the only
/// `stop` that is not already a bin edge is the end of the contig, so the sole
/// short bin is the last one of a chromosome -- expected, and kept, as in
/// HMMcopy, QDNAseq and Ginkgo.
fn bin_edges(
    start: usize,
    stop: usize,
    segment_size: usize,
) -> impl Iterator<Item = (usize, usize)> {
    (start..stop)
        .step_by(segment_size)
        .map(move |lb| (lb, (lb + segment_size).min(stop)))
}

#[cfg(test)]
mod tests;
