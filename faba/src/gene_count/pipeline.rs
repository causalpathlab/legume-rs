use crate::common::*;
use crate::gene_count::splice::*;
use crate::run_gene_count::GeneCountArgs;
use matrix_util::traits::RunningStatOps;

use fnv::FnvHashMap as HashMap;

pub fn run_simple(
    args: &GeneCountArgs,
    backend: &SparseIoBackend,
    batch_names: &[Box<str>],
) -> anyhow::Result<()> {
    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;
    gff_map.subset(args.gene_type.clone().into());
    info!("found {} features", gff_map.len());

    if gff_map.is_empty() {
        info!("found no feature in {}", args.gff_file.as_ref());
        return Ok(());
    }

    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    let records = gff_map.records();

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        let njobs = records.len() as u64;
        info!("Combining reads for {} over {} genes", batch_name, njobs);

        let gene_level_stats: Vec<(CellBarcode, Box<str>, f32)> = records
            .par_iter()
            .progress_count(njobs)
            .map(|rec| {
                count_read_per_gene(
                    bam_file,
                    rec,
                    &args.cell_barcode_tag,
                    &args.gene_barcode_tag,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        let backend_file = match backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &args.output, batch_name),
        };

        format_data_triplets(gene_level_stats)
            .to_backend(&backend_file)?
            .qc(cutoffs.clone())?;
    }

    Ok(())
}

pub fn run_splice_aware(
    args: &GeneCountArgs,
    backend: &SparseIoBackend,
    batch_names: &[Box<str>],
) -> anyhow::Result<()> {
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
    let records = gff_map.records();

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        let njobs = records.len() as u64;
        info!(
            "Combining reads (splice-aware) for {} over {} genes, intron_buffer={}",
            batch_name, njobs, args.intron_buffer
        );

        let results: Vec<SplicedUnsplicedTriplets> = records
            .par_iter()
            .progress_count(njobs)
            .map(|rec| {
                count_read_per_gene_splice(
                    bam_file,
                    rec,
                    &exon_intervals,
                    &args.cell_barcode_tag,
                    &args.gene_barcode_tag,
                    args.intron_buffer,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut spliced_triplets = Vec::new();
        let mut unspliced_triplets = Vec::new();
        for r in results {
            spliced_triplets.extend(r.spliced);
            unspliced_triplets.extend(r.unspliced);
        }

        info!(
            "{}: spliced triplets: {}, unspliced triplets: {}",
            batch_name,
            spliced_triplets.len(),
            unspliced_triplets.len()
        );

        // Collect union of cells and genes across both matrices
        let UnionNames {
            col_names,
            cell_to_index,
            row_names,
            feature_to_index,
        } = collect_union_names(&spliced_triplets, &unspliced_triplets);

        info!(
            "union: {} genes x {} cells",
            row_names.len(),
            col_names.len()
        );

        // Write spliced matrix with shared names
        let spliced_file = match backend {
            SparseIoBackend::HDF5 => format!("{}/{}_spliced.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}_spliced.zarr", &args.output, batch_name),
        };
        info!("writing spliced counts to {}", spliced_file);
        let spliced_data = format_data_triplets_shared(
            spliced_triplets,
            &feature_to_index,
            &cell_to_index,
            row_names.clone(),
            col_names.clone(),
        )
        .to_backend(&spliced_file)?;

        // Write unspliced matrix with shared names
        let unspliced_file = match backend {
            SparseIoBackend::HDF5 => format!("{}/{}_unspliced.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}_unspliced.zarr", &args.output, batch_name),
        };
        info!("writing unspliced counts to {}", unspliced_file);
        let unspliced_data = format_data_triplets_shared(
            unspliced_triplets,
            &feature_to_index,
            &cell_to_index,
            row_names,
            col_names,
        )
        .to_backend(&unspliced_file)?;

        // Union QC: keep a row/col if it passes the cutoff in either matrix
        info!("union Q/C across spliced and unspliced matrices");
        let block_size = 100;

        let spliced_row = collect_row_stat(spliced_data.as_ref(), block_size)?;
        let unspliced_row = collect_row_stat(unspliced_data.as_ref(), block_size)?;
        let spliced_col = collect_column_stat(spliced_data.as_ref(), block_size)?;
        let unspliced_col = collect_column_stat(unspliced_data.as_ref(), block_size)?;

        let s_row_nnz = spliced_row.count_positives();
        let u_row_nnz = unspliced_row.count_positives();
        let s_col_nnz = spliced_col.count_positives();
        let u_col_nnz = unspliced_col.count_positives();

        let row_cutoff = args.row_nnz_cutoff;
        let col_cutoff = args.column_nnz_cutoff;

        let row_idx: Vec<usize> = (0..s_row_nnz.len())
            .filter(|&i| {
                (s_row_nnz[i] as usize) >= row_cutoff || (u_row_nnz[i] as usize) >= row_cutoff
            })
            .collect();

        let col_idx: Vec<usize> = (0..s_col_nnz.len())
            .filter(|&i| {
                (s_col_nnz[i] as usize) >= col_cutoff || (u_col_nnz[i] as usize) >= col_cutoff
            })
            .collect();

        info!(
            "after union Q/C: {} genes x {} cells",
            row_idx.len(),
            col_idx.len()
        );

        let col_ref = if col_idx.len() < s_col_nnz.len() {
            Some(&col_idx)
        } else {
            None
        };
        let row_ref = if row_idx.len() < s_row_nnz.len() {
            Some(&row_idx)
        } else {
            None
        };

        let mut spliced_data = open_sparse_matrix(&spliced_file, backend)?;
        spliced_data.subset_columns_rows(col_ref, row_ref)?;

        let mut unspliced_data = open_sparse_matrix(&unspliced_file, backend)?;
        unspliced_data.subset_columns_rows(col_ref, row_ref)?;
    }

    Ok(())
}
