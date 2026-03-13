use crate::common::*;
use crate::data::util_htslib::*;
use crate::read_depth::coverage::ReadCoverageCollector;
use crate::run_read_depth::ReadDepthArgs;

use coitrees::IntervalTree;
use genomic_data::bed::*;

pub fn run_read_depth_pipeline(args: &ReadDepthArgs) -> anyhow::Result<()> {
    let batch_names = uniq_batch_names(&args.bam_files)?;
    std::fs::create_dir_all(args.output.as_ref())?;

    let backend = args.backend.clone();
    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    for (bam_file, batch_name) in args.bam_files.iter().zip(&batch_names) {
        // build a coitree for each chromosome, each cell barcode, and
        // each coitree can keep track of coverage
        let jobs = create_bam_jobs(bam_file, Some(args.block_size_mb * 1_000_000), Some(0))?;
        let njobs = jobs.len() as u64;
        info!("Combining reads for {} over {} blocks", batch_name, njobs);

        let segment_stats = jobs
            .par_iter()
            .progress_count(njobs)
            .map(
                |(chr, lb, ub)| -> anyhow::Result<Vec<(CellBarcode, Box<str>, f32)>> {
                    let start = *lb;
                    let stop = *ub;

                    let bed = Bed {
                        chr: chr.clone(),
                        start,
                        stop,
                    };

                    let mut read_coverage = ReadCoverageCollector::new(&args.cell_barcode_tag);
                    read_coverage.collect_from_bam(bam_file, &bed)?;

                    let coverage_interval_tree = read_coverage.to_coitrees();

                    // define segments as specified by the resolution parameter
                    let start = *lb as usize;
                    let stop = *ub as usize;
                    let segment_size = (args.resolution_kb * 1000.0) as usize;

                    // now count them all
                    let mut ret = vec![];
                    for (cb, chr_tree) in coverage_interval_tree {
                        for (chr, tree) in chr_tree {
                            for lb in (start..stop).step_by(segment_size) {
                                let ub = (lb + segment_size).min(stop);
                                let nn = tree.query_count(lb as i32, ub as i32);
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

        let backend_file = match &backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &args.output, batch_name),
        };

        format_data_triplets(segment_stats)
            .to_backend(&backend_file)?
            .qc(cutoffs.clone())?;
    }

    info!("done");
    Ok(())
}
