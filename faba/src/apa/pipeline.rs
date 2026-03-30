use crate::apa::cell_assign::*;
use crate::apa::em::*;
use crate::apa::fragment::*;
use crate::apa::likelihood::*;
use crate::apa::site_discovery::*;
use crate::apa::utr_region::*;
use crate::common::*;
use crate::data::poly_a_stat_map::PolyASiteMap;
use crate::run_apa::CountApaArgs;

use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use genomic_data::bed::BedWithGene;
use genomic_data::gff::{build_union_gene_model, read_gff_record_vec, GeneId, GffRecordMap};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::sync::{Arc, Mutex};

pub fn run_simple(args: &CountApaArgs) -> anyhow::Result<()> {
    let gff_file = args
        .gff_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--gff is required for simple mode"))?;

    info!("parsing GFF file: {}", gff_file);
    let mut gff_map = GffRecordMap::from(gff_file)?;

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    info!("found {} genes", gff_map.len());
    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    // FIRST PASS: identify poly-A sites
    let gene_sites = find_all_polya_sites(&gff_map, args)?;
    if gene_sites.is_empty() {
        info!("no poly-A sites found");
        return Ok(());
    }

    // Apply A-to-I mask if provided
    if let Some(ref mask_file) = args.atoi_mask_file {
        use crate::editing::io::load_atoi_mask_from_parquet;
        use crate::editing::mask::filter_polya_by_mask;

        info!("Loading A-to-I mask from {}", mask_file);
        let atoi_mask = load_atoi_mask_from_parquet(mask_file.as_ref())?;
        info!("Loaded A-to-I mask with {} positions", atoi_mask.len());

        let n_before: usize = gene_sites.iter().map(|x| x.value().len()).sum();
        filter_polya_by_mask(&gene_sites, &atoi_mask, &gff_map);
        let n_after: usize = gene_sites.iter().map(|x| x.value().len()).sum();
        info!(
            "A-to-I masking: {} → {} poly-A sites ({} removed)",
            n_before,
            n_after,
            n_before - n_after
        );

        if gene_sites.is_empty() {
            info!("no poly-A sites remaining after A-to-I masking");
            return Ok(());
        }
    }

    // Apply SNP mask if provided
    if let Some(ref mask_file) = args.snp_mask_file {
        use crate::editing::mask::filter_polya_by_mask;
        use crate::snp::io::load_snp_mask_from_parquet;

        info!("Loading SNP mask from {}", mask_file);
        let snp_mask = load_snp_mask_from_parquet(mask_file.as_ref())?;

        let n_before: usize = gene_sites.iter().map(|x| x.value().len()).sum();
        filter_polya_by_mask(&gene_sites, &snp_mask, &gff_map);
        let n_after: usize = gene_sites.iter().map(|x| x.value().len()).sum();
        info!(
            "SNP masking: {} → {} poly-A sites ({} removed)",
            n_before,
            n_after,
            n_before - n_after
        );

        if gene_sites.is_empty() {
            info!("no poly-A sites remaining after SNP masking");
            return Ok(());
        }
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} poly-A sites", ndata);

    // SECOND PASS: collect cell-level counts at sites
    let site_key = |x: &BedWithGene, gff_map: &GffRecordMap| -> Box<str> {
        let gene_part = gff_map
            .get(&x.gene)
            .map(|gff| format!("{}_{}", gff.gene_id, gff.gene_name))
            .unwrap_or_else(|| format!("{}", x.gene));
        format!("{}/pA/{}:{}", gene_part, x.chr, x.start).into_boxed_str()
    };

    let cutoffs = args.qc_cutoffs();
    let batch_names = uniq_batch_names(&args.bam_files)?;

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names.iter()) {
        process_simple_bam(
            bam_file,
            batch_name,
            &gene_sites,
            args,
            &gff_map,
            &site_key,
            &cutoffs,
        )?;
    }

    info!("done");
    Ok(())
}

/// Bin a position based on resolution
#[inline]
fn bin_position(position: i64, resolution: usize) -> (i64, i64) {
    let start = ((position as usize) / resolution * resolution) as i64;
    let stop = start + resolution as i64;
    (start, stop)
}

fn find_all_polya_sites(
    gff_map: &GffRecordMap,
    args: &CountApaArgs,
) -> anyhow::Result<DashMap<GeneId, Vec<i64>>> {
    let njobs = gff_map.len();
    info!("Searching poly-A sites over {} genes", njobs);

    let arc_gene_sites = Arc::new(DashMap::<GeneId, Vec<i64>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_polya_sites_in_gene(rec, args, arc_gene_sites.clone())
        })?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

fn find_polya_sites_in_gene(
    gff_record: &GffRecord,
    args: &CountApaArgs,
    arc_gene_sites: Arc<DashMap<GeneId, Vec<i64>>>,
) -> anyhow::Result<()> {
    let mut polya_map = PolyASiteMap::new(args.polya_site_args());

    for bam_file in &args.bam_files {
        polya_map.update_from_gene(
            bam_file,
            gff_record,
            &args.gene_barcode_tag,
            args.include_missing_barcode,
        )?;
    }

    let filtered_positions: Vec<i64> = polya_map
        .positions_with_counts()
        .into_iter()
        .filter(|(_, count)| *count >= args.min_coverage)
        .map(|(pos, _)| pos)
        .collect();

    if !filtered_positions.is_empty() {
        arc_gene_sites.insert(gff_record.gene_id.clone(), filtered_positions);
    }

    Ok(())
}

fn process_simple_bam(
    bam_file: &str,
    batch_name: &str,
    gene_sites: &DashMap<GeneId, Vec<i64>>,
    args: &CountApaArgs,
    gff_map: &GffRecordMap,
    site_key: &(impl Fn(&BedWithGene, &GffRecordMap) -> Box<str> + Send + Sync),
    cutoffs: &SqueezeCutoffs,
) -> anyhow::Result<()> {
    info!(
        "collecting cell-level data over {} sites from {} ...",
        gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
        bam_file
    );

    let stats = gather_polya_stats(gene_sites, args, gff_map, bam_file)?;
    info!("collected {} cell-level poly-A counts", stats.len());

    let site_data_file = args.backend_file_path(batch_name);
    let triplets = summarize_simple_stats(&stats, |bed| site_key(bed, gff_map));
    let data = triplets.to_backend(&site_data_file)?;
    data.qc(cutoffs.clone())?;
    info!("created data backend: {}", &site_data_file);

    Ok(())
}

fn gather_polya_stats(
    gene_sites: &DashMap<GeneId, Vec<i64>>,
    args: &CountApaArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_count(gene_sites.len() as u64)
        .try_for_each(|gs| -> anyhow::Result<()> {
            let gene = gs.key();
            let sites = gs.value();

            if let Some(gff) = gff_map.get(gene) {
                let stats = collect_gene_stats(args, bam_file, &gff, gene, sites)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_stats(
    args: &CountApaArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    gene_id: &GeneId,
    positions: &[i64],
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    if positions.is_empty() {
        return Ok(Vec::new());
    }

    const PADDING: i64 = 100;

    // Read the gene region ONCE for all sites instead of once per site
    let mut polya_map =
        PolyASiteMap::new_with_cell_barcode(args.polya_site_args(), &args.cell_barcode_tag);

    let min_pos = positions.iter().copied().min().unwrap();
    let max_pos = positions.iter().copied().max().unwrap();

    let mut gff = gff_record.clone();
    gff.start = (min_pos - PADDING).max(0);
    gff.stop = max_pos + PADDING;

    polya_map.update_from_gene(
        bam_file,
        &gff,
        &args.gene_barcode_tag,
        args.include_missing_barcode,
    )?;

    let chr = gff_record.seqname.as_ref();
    let strand = &gff_record.strand;
    let mut all_stats = Vec::new();

    for &position in positions {
        let Some(cell_counts) = polya_map.get_cell_counts_at(position) else {
            continue;
        };

        let (start, stop) = bin_position(position, args.resolution_bp);

        let stats = cell_counts
            .iter()
            .filter(|(cb, _)| args.include_missing_barcode || *cb != &CellBarcode::Missing)
            .map(|(cell_barcode, count)| {
                (
                    cell_barcode.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start,
                        stop,
                        gene: gene_id.clone(),
                        strand: *strand,
                    },
                    *count,
                )
            });
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

fn summarize_simple_stats<F, T>(
    stats: &[(CellBarcode, BedWithGene, usize)],
    feature_key_func: F,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
{
    let combined_data = stats
        .par_iter()
        .fold(
            rustc_hash::FxHashMap::<(CellBarcode, T), usize>::default,
            |mut acc, (cb, bed, count)| {
                let key = (cb.clone(), feature_key_func(bed));
                *acc.entry(key).or_default() += count;
                acc
            },
        )
        .reduce(rustc_hash::FxHashMap::default, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
            a
        });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, v as f32))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}

// ─────────────────────────────────────────────────────────
// Mixture mode (SCAPE EM, migrated from run_apa_mix)
// ─────────────────────────────────────────────────────────

pub fn run_mixture(args: &CountApaArgs) -> anyhow::Result<()> {
    // Get primary BAM basename for output file naming
    let batch_names = uniq_batch_names(&args.bam_files)?;
    let primary_batch_name = batch_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("no BAM files provided"))?;

    let mut utrs = load_utrs(args)?;

    // Filter UTRs to expressed genes if available
    if let Some(ref valid_gene_ids) = args.valid_gene_ids {
        let before = utrs.len();
        let valid_prefixes: Vec<String> =
            valid_gene_ids.iter().map(|gid| gid.to_string()).collect();
        utrs.retain(|utr| {
            // UTR name format: "GENE_ID_SYMBOL" or "GENE_ID"
            valid_prefixes
                .iter()
                .any(|prefix| utr.name.starts_with(prefix.as_str()))
        });
        info!(
            "filtered UTRs to expressed genes: {} -> {}",
            before,
            utrs.len()
        );
    }

    if utrs.is_empty() {
        info!("no UTR regions found");
        return Ok(());
    }

    let pre_sites = if let Some(ref pre_path) = args.pre_sites {
        info!("loading pre-identified sites from {}", pre_path);
        let sites = load_pre_sites(pre_path)?;
        info!("loaded sites for {} UTRs", sites.len());
        Some(sites)
    } else {
        None
    };

    let njobs = utrs.len();
    info!("processing {} UTRs...", njobs);

    let results: Vec<(Vec<CellSiteCount>, Vec<ApaSiteAnnotation>)> = utrs
        .par_iter()
        .progress_count(njobs as u64)
        .map(|utr| process_utr(utr, &args.bam_files, pre_sites.as_ref(), args))
        .collect::<anyhow::Result<Vec<_>>>()?;

    info!("processed {} UTRs", utrs.len());

    let total_counts: usize = results.iter().map(|(c, _)| c.len()).sum();
    let total_annots: usize = results.iter().map(|(_, a)| a.len()).sum();

    let mut all_counts = Vec::with_capacity(total_counts);
    let mut all_annotations = Vec::with_capacity(total_annots);
    for (counts, annots) in results {
        all_counts.extend(counts);
        all_annotations.extend(annots);
    }

    info!("collected {} cell-site counts", all_counts.len());

    // Filter to QC-passing cells from gene count step
    if let Some(ref valid_cells) = args.valid_cell_barcodes {
        let before = all_counts.len();
        all_counts.retain(|c| valid_cells.contains(&c.cell_barcode));
        info!(
            "filtered to QC-passing cells: {} -> {} counts",
            before,
            all_counts.len()
        );
    }

    if all_counts.is_empty() {
        info!("no counts to output");
        return Ok(());
    }

    // Compute PDUI before consuming all_counts
    if args.compute_pdui {
        compute_and_write_pdui(
            &all_counts,
            &all_annotations,
            &utrs,
            args,
            primary_batch_name,
        )?;
    }

    // Rows=sites, cols=cells
    let triplets_data: Vec<(CellBarcode, Box<str>, f32)> = all_counts
        .into_iter()
        .map(|c| (c.cell_barcode, c.site_id, c.count as f32))
        .collect();

    let triplets = format_data_triplets(triplets_data);
    let output_name = format!("{}_apa", primary_batch_name);
    let output_file = args.backend_file_path(&output_name);
    let data = triplets.to_backend(&output_file)?;
    data.qc(SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    })?;
    info!("created output: {}", &output_file);

    // Write APA site annotation Parquet
    if !all_annotations.is_empty() {
        let parquet_path = format!("{}/apa_components.parquet", &args.output);
        write_apa_annotations(&all_annotations, &parquet_path)?;
        info!(
            "wrote {} site annotations to {}",
            all_annotations.len(),
            parquet_path
        );
    }

    info!("done");
    Ok(())
}

/// Load UTR regions from --gff or --utr-bed.
fn load_utrs(args: &CountApaArgs) -> anyhow::Result<Vec<UtrRegion>> {
    if let Some(ref gff_file) = args.gff_file {
        info!("parsing GFF/GTF file: {}", gff_file);
        let records = read_gff_record_vec(gff_file)?;
        info!("read {} GFF records", records.len());

        let model = build_union_gene_model(&records)?;
        info!(
            "found {} 3'-UTR regions in gene model",
            model.three_prime_utr.len()
        );

        let mut utrs = Vec::new();
        for entry in model.three_prime_utr.iter() {
            let gene_id = entry.key();
            let rec = entry.value();
            let utr_length = (rec.stop - rec.start) as usize;
            if utr_length < args.min_utr_length {
                continue;
            }
            let name: Box<str> = match &rec.gene_name {
                genomic_data::gff::GeneSymbol::Symbol(s) => format!("{}_{}", gene_id, s).into(),
                genomic_data::gff::GeneSymbol::Missing => format!("{}", gene_id).into(),
            };
            utrs.push(UtrRegion {
                chr: rec.seqname.clone(),
                start: rec.start,
                end: rec.stop,
                strand: rec.strand,
                name,
                utr_length,
            });
        }

        info!("kept {} 3'-UTRs (>= {}bp)", utrs.len(), args.min_utr_length);
        Ok(utrs)
    } else if let Some(ref utr_bed) = args.utr_bed {
        info!("loading UTR regions from {}", utr_bed);
        let utrs = load_utr_regions_from_bed(utr_bed)?;
        info!("loaded {} UTR regions", utrs.len());
        Ok(utrs)
    } else {
        Err(anyhow::anyhow!(
            "must provide either --gff or --utr-bed for mixture mode"
        ))
    }
}

/// Load pre-identified sites from a BED file.
fn load_pre_sites(path: &str) -> anyhow::Result<rustc_hash::FxHashMap<Box<str>, Vec<f32>>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut sites: rustc_hash::FxHashMap<Box<str>, Vec<f32>> = rustc_hash::FxHashMap::default();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 4 {
            let name = fields[3];
            if let Some((utr_name, pos_str)) = name.rsplit_once('@') {
                if let Ok(pos) = pos_str.parse::<f32>() {
                    sites.entry(utr_name.into()).or_default().push(pos);
                }
            }
        }
    }

    Ok(sites)
}

/// Process a single UTR: extract fragments, discover/load sites, run EM, assign cells.
fn process_utr(
    utr: &UtrRegion,
    bam_files: &[Box<str>],
    pre_sites: Option<&rustc_hash::FxHashMap<Box<str>, Vec<f32>>>,
    args: &CountApaArgs,
) -> anyhow::Result<(Vec<CellSiteCount>, Vec<ApaSiteAnnotation>)> {
    let mut all_fragments = Vec::new();
    for bam_file in bam_files {
        let polya_params = PolyAFilterParams {
            min_tail: args.polya_min_tail_length,
            max_non_at: args.polya_max_non_a_or_t,
            internal_prime_window: args.polya_internal_prime_window,
            internal_prime_count: args.polya_internal_prime_count,
        };
        let frags = extract_fragments(
            bam_file,
            utr,
            args.cell_barcode_tag.as_bytes(),
            args.umi_tag.as_bytes(),
            &polya_params,
        )?;
        all_fragments.extend(frags);
    }

    log::debug!(
        "UTR {} ({}:{}-{}, L={}): {} fragments extracted",
        utr.name,
        utr.chr,
        utr.start,
        utr.end,
        utr.utr_length,
        all_fragments.len()
    );

    if all_fragments.len() < args.min_fragments {
        return Ok((Vec::new(), Vec::new()));
    }

    let candidate_sites = if let Some(pre) = pre_sites {
        pre.get(&utr.name).cloned().unwrap_or_default()
    } else {
        let raw_sites = discover_sites_from_junctions(&all_fragments, args.min_coverage);
        if !raw_sites.is_empty() {
            merge_nearby_sites(&raw_sites, &all_fragments, args.merge_distance)
        } else {
            let bandwidth = 100.0;
            let coverage_sites =
                discover_sites_from_coverage(&all_fragments, utr.utr_length as f32, bandwidth);
            merge_nearby_sites(&coverage_sites, &all_fragments, args.merge_distance)
        }
    };

    let n_junction = all_fragments.iter().filter(|f| f.is_junction).count();
    log::debug!(
        "UTR {}: {} candidate sites, {} junction reads",
        utr.name,
        candidate_sites.len(),
        n_junction
    );

    if candidate_sites.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let lik_params = args.lik_params();
    let (theta_lik_matrix, theta_grid) =
        precompute_theta_lik_matrix(&all_fragments, utr.utr_length as f32, &lik_params);

    let default_beta = (args.min_beta + args.max_beta) / 2.0;
    let beta_arr: Vec<f32> = vec![default_beta; candidate_sites.len()];

    let em_params = args.em_params();

    // Rank candidate sites by fragment coverage (descending) for BIC model selection
    let merge_dist = args.merge_distance;
    let site_order = {
        let mut scored: Vec<(usize, usize)> = candidate_sites
            .iter()
            .enumerate()
            .map(|(i, &site)| {
                let count = all_fragments
                    .iter()
                    .filter(|f| {
                        f.pa_site
                            .map(|pa| (pa - site).abs() < merge_dist)
                            .unwrap_or(false)
                    })
                    .count();
                (i, count)
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().map(|(i, _)| i).collect::<Vec<_>>()
    };

    let site_data = SiteModelData {
        alpha_arr: &candidate_sites,
        beta_arr: &beta_arr,
        theta_lik_matrix: &theta_lik_matrix,
        theta_grid: &theta_grid,
        utr_length: utr.utr_length as f32,
        max_polya: lik_params.max_polya,
    };
    let em_result = select_sites_by_bic(&site_data, &em_params, &site_order);

    log::debug!(
        "UTR {}: BIC selected {} sites (from {} candidates), BIC={:.1}, LL={:.1}, {} iters",
        utr.name,
        em_result.alphas.len(),
        candidate_sites.len(),
        em_result.bic,
        em_result.log_lik,
        em_result.n_iter,
    );

    let (cell_counts, annotations) = assign_fragments_to_sites(&all_fragments, &em_result, utr);

    Ok((cell_counts, annotations))
}

/// Compute PDUI for genes with exactly 2 active pA sites and write a sparse matrix.
fn compute_and_write_pdui(
    all_counts: &[CellSiteCount],
    all_annotations: &[ApaSiteAnnotation],
    utrs: &[UtrRegion],
    args: &CountApaArgs,
    primary_batch_name: &str,
) -> anyhow::Result<()> {
    use crate::apa::pdui::compute_pdui;
    use rustc_hash::FxHashMap;

    info!("Computing PDUI...");

    // Build a strand lookup from UTRs by gene name
    let strand_by_gene: FxHashMap<Box<str>, genomic_data::sam::Strand> =
        utrs.iter().map(|u| (u.name.clone(), u.strand)).collect();

    // Group annotations and counts by gene_name
    let mut annots_by_gene: FxHashMap<Box<str>, Vec<ApaSiteAnnotation>> = FxHashMap::default();
    for a in all_annotations {
        annots_by_gene
            .entry(a.gene_name.clone())
            .or_default()
            .push(a.clone());
    }

    let mut counts_by_gene: FxHashMap<Box<str>, Vec<&CellSiteCount>> = FxHashMap::default();
    for c in all_counts {
        // Extract gene name from site_id (format: "GENE/pA/k")
        let gene_name = c
            .site_id
            .find("/pA/")
            .map(|pos| &c.site_id[..pos])
            .unwrap_or(c.site_id.as_ref());

        counts_by_gene.entry(gene_name.into()).or_default().push(c);
    }

    let mut pdui_triplets: Vec<(CellBarcode, Box<str>, f32)> = Vec::new();
    let mut n_pdui_genes = 0;

    for (gene_name, annots) in &annots_by_gene {
        if annots.len() != 2 {
            continue;
        }

        let strand = match strand_by_gene.get(gene_name) {
            Some(s) => *s,
            None => continue,
        };

        let gene_counts: Vec<CellSiteCount> = counts_by_gene
            .get(gene_name)
            .map(|cs| {
                cs.iter()
                    .map(|c| CellSiteCount {
                        cell_barcode: c.cell_barcode.clone(),
                        site_id: c.site_id.clone(),
                        count: c.count,
                    })
                    .collect()
            })
            .unwrap_or_default();

        if let Some(pdui_result) = compute_pdui(&gene_counts, annots, strand) {
            n_pdui_genes += 1;
            for (cb, pdui_val) in &pdui_result.cell_pdui {
                pdui_triplets.push((cb.clone(), gene_name.clone(), *pdui_val));
            }
        }
    }

    info!(
        "PDUI: computed for {} genes, {} cell-gene values",
        n_pdui_genes,
        pdui_triplets.len()
    );

    if pdui_triplets.is_empty() {
        info!("No PDUI values to output");
        return Ok(());
    }

    let triplets = format_data_triplets(pdui_triplets);
    let output_name = format!("{}_pdui", primary_batch_name);
    let output_file = args.backend_file_path(&output_name);
    let data = triplets.to_backend(&output_file)?;
    data.qc(SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    })?;
    info!("PDUI: created {}", &output_file);

    Ok(())
}

/// Write APA site annotations to a Parquet file.
fn write_apa_annotations(annotations: &[ApaSiteAnnotation], path: &str) -> anyhow::Result<()> {
    let mut site_ids = Vec::with_capacity(annotations.len());
    let mut gene_names = Vec::with_capacity(annotations.len());
    let mut chrs = Vec::with_capacity(annotations.len());
    let mut genomic_alphas = Vec::with_capacity(annotations.len());
    let mut betas = Vec::with_capacity(annotations.len());
    let mut genomic_starts = Vec::with_capacity(annotations.len());
    let mut genomic_stops = Vec::with_capacity(annotations.len());
    let mut pi_weights = Vec::with_capacity(annotations.len());
    let mut expected_tails = Vec::with_capacity(annotations.len());

    for a in annotations {
        site_ids.push(a.site_id.as_ref());
        gene_names.push(a.gene_name.as_ref());
        chrs.push(a.chr.as_ref());
        genomic_alphas.push(a.genomic_alpha);
        betas.push(a.beta);
        genomic_starts.push(a.genomic_start);
        genomic_stops.push(a.genomic_stop);
        pi_weights.push(a.pi_weight);
        expected_tails.push(a.expected_tail_length);
    }

    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("site_id", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("gene_name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("chr", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("genomic_alpha", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("beta", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new("genomic_start", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("genomic_stop", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("pi_weight", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new(
            "expected_tail_length",
            arrow::datatypes::DataType::Float32,
            false,
        ),
    ]);

    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![
            Arc::new(StringArray::from(site_ids)) as ArrayRef,
            Arc::new(StringArray::from(gene_names)) as ArrayRef,
            Arc::new(StringArray::from(chrs)) as ArrayRef,
            Arc::new(Int64Array::from(genomic_alphas)) as ArrayRef,
            Arc::new(Float32Array::from(betas)) as ArrayRef,
            Arc::new(Int64Array::from(genomic_starts)) as ArrayRef,
            Arc::new(Int64Array::from(genomic_stops)) as ArrayRef,
            Arc::new(Float32Array::from(pi_weights)) as ArrayRef,
            Arc::new(Float32Array::from(expected_tails)) as ArrayRef,
        ],
    )?;

    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
