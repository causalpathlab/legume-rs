use crate::common::*;
use crate::gene_count::splice::*;
use crate::run_gene_count::GeneCountArgs;

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
    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

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

        // Build total counts (spliced + unspliced) for QC
        // Gene key: strip "/count/spliced" or "/count/unspliced" suffix, use "/count/total"
        let total_triplets: Vec<(CellBarcode, Box<str>, f32)> = {
            let mut totals: fnv::FnvHashMap<(CellBarcode, Box<str>), f32> =
                fnv::FnvHashMap::default();
            for (cb, feat, val) in spliced_triplets.iter().chain(unspliced_triplets.iter()) {
                let gene_key = feat
                    .rfind("/count/")
                    .map(|pos| &feat[..pos])
                    .unwrap_or(feat.as_ref());
                let total_key: Box<str> = format!("{}/count/total", gene_key).into();
                *totals.entry((cb.clone(), total_key)).or_default() += val;
            }
            totals.into_iter().map(|((c, k), v)| (c, k, v)).collect()
        };

        // QC on total counts
        let total_file = match backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &args.output, batch_name),
        };
        info!("writing total counts to {}", total_file);
        format_data_triplets(total_triplets)
            .to_backend(&total_file)?
            .qc(cutoffs.clone())?;

        // Read back surviving genes and cells from total counts QC
        let total_data = open_sparse_matrix(&total_file, backend)?;
        let qc_row_names = total_data.row_names()?;
        let qc_col_names = total_data.column_names()?;
        info!(
            "after Q/C on total counts: {} genes x {} cells",
            qc_row_names.len(),
            qc_col_names.len()
        );

        // Build index maps for the QC-passing names
        // Map total row names back to spliced/unspliced feature names
        let qc_gene_keys: fnv::FnvHashSet<Box<str>> = qc_row_names
            .iter()
            .filter_map(|name| {
                name.rfind("/count/")
                    .map(|pos| name[..pos].to_string().into_boxed_str())
            })
            .collect();

        let qc_cells: fnv::FnvHashSet<CellBarcode> = qc_col_names
            .iter()
            .map(|name| CellBarcode::Barcode(name.clone()))
            .collect();

        // Filter spliced/unspliced triplets to QC-passing genes and cells
        let filter_triplets = |triplets: Vec<(CellBarcode, Box<str>, f32)>| -> Vec<_> {
            triplets
                .into_iter()
                .filter(|(cb, feat, _)| {
                    let gene_key = feat
                        .rfind("/count/")
                        .map(|pos| feat[..pos].to_string().into_boxed_str())
                        .unwrap_or_else(|| feat.clone());
                    qc_gene_keys.contains(&gene_key) && qc_cells.contains(cb)
                })
                .collect()
        };

        let spliced_triplets = filter_triplets(spliced_triplets);
        let unspliced_triplets = filter_triplets(unspliced_triplets);

        // Use shared names from QC-passing set for consistent dimensions
        let UnionNames {
            col_names,
            cell_to_index,
            row_names,
            feature_to_index,
        } = collect_union_names(&spliced_triplets, &unspliced_triplets);

        info!(
            "writing spliced/unspliced: {} genes x {} cells",
            row_names.len(),
            col_names.len()
        );

        // Write spliced matrix
        let spliced_file = match backend {
            SparseIoBackend::HDF5 => format!("{}/{}_spliced.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}_spliced.zarr", &args.output, batch_name),
        };
        format_data_triplets_shared(
            spliced_triplets,
            &feature_to_index,
            &cell_to_index,
            row_names.clone(),
            col_names.clone(),
        )
        .to_backend(&spliced_file)?;
        info!("wrote spliced counts to {}", spliced_file);

        // Write unspliced matrix
        let unspliced_file = match backend {
            SparseIoBackend::HDF5 => format!("{}/{}_unspliced.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}_unspliced.zarr", &args.output, batch_name),
        };
        format_data_triplets_shared(
            unspliced_triplets,
            &feature_to_index,
            &cell_to_index,
            row_names,
            col_names,
        )
        .to_backend(&unspliced_file)?;
        info!("wrote unspliced counts to {}", unspliced_file);
    }

    Ok(())
}
