use crate::common::*;
use crate::gene_count::splice::*;
use crate::pipeline_util::extract_gene_key;
use crate::run_gene_count::GeneCountArgs;

use rustc_hash::FxHashMap as HashMap;

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

        let out = crate::pipeline_util::BackendOutputPath::new(
            &args.output,
            batch_name,
            backend,
            args.zip,
        );

        format_data_triplets(gene_level_stats)
            .to_backend(&out.write_path)?
            .qc(cutoffs.clone())?;

        out.finalize()?;
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

    // Convert DashMap exon intervals to FxHashMap for fast per-gene lookup
    let exon_intervals: HashMap<GeneId, Vec<(i64, i64)>> = exon_map.into_iter().collect();
    let records = gff_map.records();
    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        let njobs = records.len() as u64;
        info!(
            "Combining reads (splice-aware) for {} over {} genes",
            batch_name, njobs
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

        // Build total counts (spliced + unspliced) for QC.
        // Parallel fold/reduce: each rayon worker accumulates a local
        // (cell, gene_key) → val map; we then merge the shards and
        // materialize the "{gene_key}/count/total" feature names once.
        let total_triplets: Vec<(CellBarcode, Box<str>, f32)> = {
            use rayon::prelude::*;
            type Shard<'a> = rustc_hash::FxHashMap<(CellBarcode, &'a str), f32>;

            let totals: Shard = spliced_triplets
                .par_iter()
                .chain(unspliced_triplets.par_iter())
                .fold(Shard::default, |mut acc, (cb, feat, val)| {
                    let gene_key = extract_gene_key(feat);
                    *acc.entry((cb.clone(), gene_key)).or_default() += *val;
                    acc
                })
                .reduce(Shard::default, |mut a, mut b| {
                    if a.len() < b.len() {
                        std::mem::swap(&mut a, &mut b);
                    }
                    for (k, v) in b {
                        *a.entry(k).or_default() += v;
                    }
                    a
                });

            // Memoize gene_key → "{gene_key}/count/total" once (serial; cheap
            // relative to the fold above).
            let mut total_key_cache: HashMap<&str, Box<str>> = HashMap::default();
            for (_, gk) in totals.keys() {
                total_key_cache
                    .entry(gk)
                    .or_insert_with(|| format!("{}/count/total", gk).into());
            }
            totals
                .into_iter()
                .map(|((c, gk), v)| (c, total_key_cache[gk].clone(), v))
                .collect()
        };

        // QC on total counts
        let total_out = crate::pipeline_util::BackendOutputPath::new(
            &args.output,
            batch_name,
            backend,
            args.zip,
        );
        let total_file = total_out.write_path.clone();
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
        let qc_gene_keys: rustc_hash::FxHashSet<Box<str>> = qc_row_names
            .iter()
            .map(|name| extract_gene_key(name).into())
            .collect();

        let qc_cells: rustc_hash::FxHashSet<CellBarcode> = qc_col_names
            .iter()
            .map(|name| CellBarcode::Barcode(name.clone()))
            .collect();

        // Filter spliced/unspliced triplets to QC-passing genes and cells.
        let filter_triplets = |triplets: Vec<(CellBarcode, Box<str>, f32)>| -> Vec<_> {
            use rayon::prelude::*;
            triplets
                .into_par_iter()
                .filter(|(cb, feat, _)| {
                    let gene_key: &str = extract_gene_key(feat);
                    qc_gene_keys.contains(gene_key) && qc_cells.contains(cb)
                })
                .collect()
        };

        let (spliced_triplets, unspliced_triplets) = rayon::join(
            || filter_triplets(spliced_triplets),
            || filter_triplets(unspliced_triplets),
        );

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

        // Write spliced + unspliced matrices in parallel (independent I/O;
        // rayon::join shares the existing worker pool, so the to_backend
        // internals don't oversubscribe).
        let spliced_out = crate::pipeline_util::BackendOutputPath::new(
            &args.output,
            &format!("{}_spliced", batch_name),
            backend,
            args.zip,
        );
        let unspliced_out = crate::pipeline_util::BackendOutputPath::new(
            &args.output,
            &format!("{}_unspliced", batch_name),
            backend,
            args.zip,
        );

        let (spliced_res, unspliced_res) = rayon::join(
            || {
                format_data_triplets_shared(
                    spliced_triplets,
                    &feature_to_index,
                    &cell_to_index,
                    row_names.clone(),
                    col_names.clone(),
                )
                .to_backend(&spliced_out.write_path)
            },
            || {
                format_data_triplets_shared(
                    unspliced_triplets,
                    &feature_to_index,
                    &cell_to_index,
                    row_names.clone(),
                    col_names.clone(),
                )
                .to_backend(&unspliced_out.write_path)
            },
        );
        spliced_res?;
        unspliced_res?;
        info!("wrote spliced counts to {}", spliced_out.target_path);
        info!("wrote unspliced counts to {}", unspliced_out.target_path);

        // Finalize archives (zip the .zarr staging dirs if applicable).
        // total_out finalized last so the post-QC backend handle has been
        // dropped by the time we zip.
        spliced_out.finalize()?;
        unspliced_out.finalize()?;
        drop(total_data);
        total_out.finalize()?;
    }

    Ok(())
}
