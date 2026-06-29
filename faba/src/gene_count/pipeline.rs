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
    // Count + cell-call on ALL biotypes (Cell Ranger-faithful, matching the
    // `faba all` pipeline); the biotype subset is applied only to the quantified
    // output + pooled gene ids below, never to cell-calling.
    let gff_map = GffRecordMap::from(args.gff_file.as_ref())?;
    info!("found {} features", gff_map.len());

    if gff_map.is_empty() {
        info!("found no feature in {}", args.gff_file.as_ref());
        return Ok(());
    }

    let records = gff_map.records();
    let gene_key_to_id: HashMap<Box<str>, GeneId> = records
        .iter()
        .map(|rec| (format_gene_key(rec), rec.gene_id.clone()))
        .collect();
    let mut all_gene_ids: rustc_hash::FxHashSet<GeneId> = rustc_hash::FxHashSet::default();
    let cell_call = args.cell_qc.params();
    let umi_tag = args.umi_dedup_tag();
    let mito_keys = crate::pipeline_util::mito_gene_keys(&records, &args.mito_chr);
    info!(
        "{} mitochondrial gene(s) on {} ({})",
        mito_keys.len(),
        args.mito_chr,
        if args.keep_mito {
            "kept in matrix"
        } else {
            "excluded from matrix"
        }
    );

    // Biotype + mito quantification gate: count/cell-call on all biotypes, restrict
    // only the quantified output + pooled gene ids. Mirrors `faba all`'s quantify_gene.
    let selected_gene_keys: Option<rustc_hash::FxHashSet<Box<str>>> = if args.gene_type.is_empty() {
        None
    } else {
        let target: genomic_data::gff::GeneType = args.gene_type.clone().into();
        Some(
            records
                .iter()
                .filter(|r| r.gene_type == target)
                .map(format_gene_key)
                .collect(),
        )
    };
    let quantify_gene = |gk: &str| -> bool {
        (args.keep_mito || !mito_keys.contains(gk))
            && selected_gene_keys.as_ref().is_none_or(|s| s.contains(gk))
    };

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        let njobs = records.len() as u64;
        info!("Combining reads for {} over {} genes", batch_name, njobs);

        let gene_level_stats: Vec<(CellBarcode, Box<str>, f32)> = records
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .map(|rec| {
                count_read_per_gene(
                    bam_file,
                    rec,
                    &args.cell_barcode_tag,
                    &args.gene_barcode_tag,
                    umi_tag,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        // Cell calling (barcode QC) + gene nnz filter (no splice split here, so
        // the unspliced track is empty).
        let (passing_genes, passing_cells) = crate::pipeline_util::qc_passing_keys(
            &gene_level_stats,
            &[],
            args.row_nnz_cutoff,
            0,
            args.column_nnz_cutoff,
            &cell_call,
        );
        info!(
            "{}: {} genes x {} cells passed QC (cell-filter {:?})",
            batch_name,
            passing_genes.len(),
            passing_cells.len(),
            cell_call.filter
        );

        // Mitochondrial QC: report per-cell MT fraction (from full pre-filter
        // counts), optionally drop high-MT cells, and exclude MT genes from the
        // matrix unless --keep-mito.
        let mt_stats =
            crate::pipeline_util::mito_cell_stats(&[&gene_level_stats], &passing_cells, &mito_keys);
        crate::pipeline_util::write_mt_qc(&args.output, batch_name, &mt_stats)?;
        let passing_cells = crate::pipeline_util::apply_mito_filter(
            passing_cells,
            &mt_stats,
            args.max_mito_frac,
            args.no_mito_cell_qc,
        );
        let passing_genes: rustc_hash::FxHashSet<Box<str>> = passing_genes
            .into_iter()
            .filter(|gk| quantify_gene(gk))
            .collect();

        let gene_level_stats: Vec<(CellBarcode, Box<str>, f32)> = gene_level_stats
            .into_par_iter()
            .filter(|(cb, feat, _)| {
                passing_genes.contains(extract_gene_key(feat)) && passing_cells.contains(cb)
            })
            .collect();

        crate::pipeline_util::write_qc_cells(&args.output, batch_name, &passing_cells)?;
        for gk in &passing_genes {
            if let Some(id) = gene_key_to_id.get(gk) {
                all_gene_ids.insert(id.clone());
            }
        }

        let out = crate::pipeline_util::BackendOutputPath::new(
            &args.output,
            batch_name,
            backend,
            args.zip,
        );

        format_data_triplets(gene_level_stats).to_backend(&out.write_path)?;

        out.finalize()?;
    }

    crate::pipeline_util::write_qc_genes(&args.output, &all_gene_ids)?;

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

    // Count + cell-call on ALL biotypes (Cell Ranger-faithful, matching the
    // `faba all` pipeline); the biotype subset is applied only to the quantified
    // output + pooled gene ids below, never to cell-calling.
    let gff_map = GffRecordMap::from_map(gene_map);
    info!("found {} features", gff_map.len());

    if gff_map.is_empty() {
        info!("found no feature in {}", args.gff_file.as_ref());
        return Ok(());
    }

    // Convert DashMap exon intervals to FxHashMap for fast per-gene lookup
    let exon_intervals: HashMap<GeneId, Vec<(i64, i64)>> = exon_map.into_iter().collect();
    let records = gff_map.records();
    // gene_key → GeneId, to write the pooled retained-gene artifact after the loop.
    let gene_key_to_id: HashMap<Box<str>, GeneId> = records
        .iter()
        .map(|rec| (format_gene_key(rec), rec.gene_id.clone()))
        .collect();
    let mut all_gene_ids: rustc_hash::FxHashSet<GeneId> = rustc_hash::FxHashSet::default();
    let umi_tag = args.umi_dedup_tag();
    let mito_keys = crate::pipeline_util::mito_gene_keys(&records, &args.mito_chr);
    info!(
        "{} mitochondrial gene(s) on {} ({})",
        mito_keys.len(),
        args.mito_chr,
        if args.keep_mito {
            "kept in matrix"
        } else {
            "excluded from matrix"
        }
    );

    // Biotype + mito quantification gate: count/cell-call on all biotypes, restrict
    // only the quantified output + pooled gene ids. Mirrors `faba all`'s quantify_gene.
    let selected_gene_keys: Option<rustc_hash::FxHashSet<Box<str>>> = if args.gene_type.is_empty() {
        None
    } else {
        let target: genomic_data::gff::GeneType = args.gene_type.clone().into();
        Some(
            records
                .iter()
                .filter(|r| r.gene_type == target)
                .map(format_gene_key)
                .collect(),
        )
    };
    let quantify_gene = |gk: &str| -> bool {
        (args.keep_mito || !mito_keys.contains(gk))
            && selected_gene_keys.as_ref().is_none_or(|s| s.contains(gk))
    };

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        let njobs = records.len() as u64;
        info!(
            "Combining reads (splice-aware) for {} over {} genes",
            batch_name, njobs
        );

        let results: Vec<SplicedUnsplicedTriplets> = records
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .map(|rec| {
                count_read_per_gene_splice(
                    bam_file,
                    rec,
                    &exon_intervals,
                    &args.cell_barcode_tag,
                    &args.gene_barcode_tag,
                    umi_tag,
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

        // Cell calling (barcode QC) + gene nnz filter — shared with the modality
        // QC path so `faba genes` retains the same cells the other modalities do.
        let cell_call = args.cell_qc.params();
        let (passing_genes, passing_cells) = crate::pipeline_util::qc_passing_keys(
            &spliced_triplets,
            &unspliced_triplets,
            args.row_nnz_cutoff,
            0,
            args.column_nnz_cutoff,
            &cell_call,
        );
        info!(
            "{}: {} genes x {} cells passed QC (cell-filter {:?})",
            batch_name,
            passing_genes.len(),
            passing_cells.len(),
            cell_call.filter
        );

        // Mitochondrial QC: report per-cell MT fraction (from full pre-filter
        // counts), optionally drop high-MT cells, and exclude MT genes from the
        // matrix unless --keep-mito.
        let mt_stats = crate::pipeline_util::mito_cell_stats(
            &[&spliced_triplets, &unspliced_triplets],
            &passing_cells,
            &mito_keys,
        );
        crate::pipeline_util::write_mt_qc(&args.output, batch_name, &mt_stats)?;
        let passing_cells = crate::pipeline_util::apply_mito_filter(
            passing_cells,
            &mt_stats,
            args.max_mito_frac,
            args.no_mito_cell_qc,
        );
        let passing_genes: rustc_hash::FxHashSet<Box<str>> = passing_genes
            .into_iter()
            .filter(|gk| quantify_gene(gk))
            .collect();

        let keep =
            |triplets: Vec<(CellBarcode, Box<str>, f32)>| -> Vec<(CellBarcode, Box<str>, f32)> {
                triplets
                    .into_par_iter()
                    .filter(|(cb, feat, _)| {
                        passing_genes.contains(extract_gene_key(feat)) && passing_cells.contains(cb)
                    })
                    .collect()
            };
        let (spliced_triplets, unspliced_triplets) =
            rayon::join(|| keep(spliced_triplets), || keep(unspliced_triplets));

        // Passable QC artifacts: per-batch cells now, pooled gene ids after the loop.
        crate::pipeline_util::write_qc_cells(&args.output, batch_name, &passing_cells)?;
        for gk in &passing_genes {
            if let Some(id) = gene_key_to_id.get(gk) {
                all_gene_ids.insert(id.clone());
            }
        }

        // Total counts (spliced + unspliced) — the primary `{batch}` matrix.
        // Parallel fold/reduce: each rayon worker accumulates a local
        // (cell, gene_key) → val map; merge the shards and materialize the
        // "{gene_key}/count/total" feature names once.
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

        let total_out = crate::pipeline_util::BackendOutputPath::new(
            &args.output,
            batch_name,
            backend,
            args.zip,
        );
        format_data_triplets(total_triplets).to_backend(&total_out.write_path)?;
        info!("wrote total counts to {}", total_out.target_path);

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
        total_out.finalize()?;
    }

    // Pooled retained genes (shared vocabulary) for --valid-genes reuse.
    crate::pipeline_util::write_qc_genes(&args.output, &all_gene_ids)?;

    Ok(())
}
