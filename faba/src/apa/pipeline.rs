use crate::apa::cell_assign::*;
use crate::apa::em::*;
use crate::apa::fragment::*;
use crate::apa::likelihood::*;
use crate::apa::site_discovery::*;
use crate::apa::utr_region::*;
use crate::common::*;
use crate::data::poly_a_stat_map::PolyASiteMap;
use crate::run_apa::CountApaArgs;

use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray, UInt32Array};
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
        .progress_with(new_progress_bar(njobs as u64))
        .try_for_each_init(
            crate::data::bam_io::BamReaderCache::new,
            |cache, rec| -> anyhow::Result<()> {
                find_polya_sites_in_gene(cache, rec, args, arc_gene_sites.clone())
            },
        )?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

fn find_polya_sites_in_gene(
    cache: &mut crate::data::bam_io::BamReaderCache,
    gff_record: &GffRecord,
    args: &CountApaArgs,
    arc_gene_sites: Arc<DashMap<GeneId, Vec<i64>>>,
) -> anyhow::Result<()> {
    let mut polya_map = PolyASiteMap::new(args.polya_site_args());

    for bam_file in &args.bam_files {
        polya_map.update_from_gene_cached(
            cache,
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

    let out = args.backend_output_path(batch_name);
    let triplets = summarize_simple_stats(&stats, |bed| site_key(bed, gff_map));
    let data = triplets.to_backend(&out.write_path)?;
    data.qc(cutoffs.clone())?;
    info!("created data backend: {}", &out.target_path);

    drop(data);
    out.finalize()?;
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
        .progress_with(new_progress_bar(gene_sites.len() as u64))
        .try_for_each_init(
            crate::data::bam_io::BamReaderCache::new,
            |cache, gs| -> anyhow::Result<()> {
                let gene = gs.key();
                let sites = gs.value();

                if let Some(gff) = gff_map.get(gene) {
                    let stats = collect_gene_stats(cache, args, bam_file, &gff, gene, sites)?;
                    arc_ret.lock().expect("lock").extend(stats);
                }
                Ok(())
            },
        )?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_stats(
    cache: &mut crate::data::bam_io::BamReaderCache,
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

    polya_map.update_from_gene_cached(
        cache,
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
    let mut utrs = load_utrs(args)?;

    // Filter UTRs to expressed genes if available
    if let Some(ref valid_gene_ids) = args.valid_gene_ids {
        let before = utrs.len();
        // UTR name format: "GENE_ID_SYMBOL" or "GENE_ID" (see load_utrs).
        // Match on the gene_id token before the first '_' against the expressed set.
        let valid_ids: rustc_hash::FxHashSet<Box<str>> = valid_gene_ids
            .iter()
            .map(|gid| gid.to_string().into_boxed_str())
            .collect();
        utrs.retain(|utr| {
            let gid = utr
                .name
                .split_once('_')
                .map(|(g, _)| g)
                .unwrap_or(&utr.name);
            valid_ids.contains(gid)
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

    // Big-first scheduling (LPT): a few highly-expressed UTRs carry ~100x the
    // fragments and dominate wall-clock. Start the heaviest first so they overlap
    // the long tail of light UTRs instead of trailing it on one idle-starved
    // thread. Output is unaffected — rows are reordered to a sorted vocabulary
    // downstream, so the per-UTR schedule order does not change results.
    utrs.sort_by_key(|u| std::cmp::Reverse(u.utr_length));

    // ATOI + SNP position masks, pooled once into one (chr, genomic_pos) set. The
    // mixture path drops candidate poly-A sites coinciding with an edit/variant;
    // run_simple did this but run_mixture historically skipped it, so the pipeline
    // computed both masks and silently ignored them.
    let site_mask = load_polya_site_mask(args)?;

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

    // Per-phase wall-clock summed across worker threads: BAM fragment extraction
    // (I/O) vs site-selection/EM (compute). Diagnoses where APA spends its time.
    use std::sync::atomic::{AtomicU64, Ordering};
    let extract_ns = AtomicU64::new(0);
    let bic_ns = AtomicU64::new(0);
    let discover_ns = AtomicU64::new(0);
    let assign_ns = AtomicU64::new(0);

    let results: Vec<(Vec<CellSiteCount>, Vec<ApaSiteAnnotation>)> = utrs
        .par_iter()
        .progress_with(new_progress_bar(njobs as u64))
        .map_init(crate::data::bam_io::BamReaderCache::new, |cache, utr| {
            process_utr(
                cache,
                utr,
                &args.bam_files,
                pre_sites.as_ref(),
                site_mask.as_ref(),
                &extract_ns,
                &bic_ns,
                &discover_ns,
                &assign_ns,
                args,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    info!("processed {} UTRs", utrs.len());
    info!(
        "APA timing (summed over threads): extraction {:.1}s, discovery {:.1}s, \
         fast-assign {:.1}s, site-selection/EM {:.1}s",
        extract_ns.load(Ordering::Relaxed) as f64 / 1e9,
        discover_ns.load(Ordering::Relaxed) as f64 / 1e9,
        assign_ns.load(Ordering::Relaxed) as f64 / 1e9,
        bic_ns.load(Ordering::Relaxed) as f64 / 1e9,
    );

    let total_counts: usize = results.iter().map(|(c, _)| c.len()).sum();
    let total_annots: usize = results.iter().map(|(_, a)| a.len()).sum();

    let mut all_counts = Vec::with_capacity(total_counts);
    let mut all_annotations = Vec::with_capacity(total_annots);
    for (counts, annots) in results {
        all_counts.extend(counts);
        all_annotations.extend(annots);
    }

    info!("collected {} cell-site counts", all_counts.len());

    // Filter to QC-passing cells from gene count step. Each count carries its
    // batch index, so it is checked against that batch's own called cell set
    // (per-library knee); batches without a passed set are left unfiltered.
    // `cells_by_batch` is keyed by BAM file path (see `GeneCountQc`), NOT batch
    // name — basenames collide across 10x libraries — so look it up by the path,
    // matching the conversion pipeline (`editing::pipeline`).
    if let Some(ref valid_cells) = args.valid_cell_barcodes {
        // Resolve each batch index to its cell set once (one entry per BAM), so
        // the per-count retain is a Vec index + barcode lookup instead of hashing
        // the BAM-path key for every one of the (potentially millions of) counts.
        let by_batch: Vec<_> = args.bam_files.iter().map(|f| valid_cells.get(f)).collect();
        let before = all_counts.len();
        all_counts
            .retain(|c| by_batch[c.batch as usize].is_none_or(|set| set.contains(&c.cell_barcode)));
        info!(
            "filtered to QC-passing cells: {} -> {} counts",
            before,
            all_counts.len()
        );
    }

    // Optionally drop genes that resolved to a single pA site: a lone site
    // carries no relative usage signal (PDUI is undefined, the count is the
    // gene total). A gene's active-site count is its annotation count.
    if args.drop_single_component {
        use rustc_hash::FxHashMap;
        let mut sites_per_gene: FxHashMap<Box<str>, usize> = FxHashMap::default();
        for a in &all_annotations {
            *sites_per_gene.entry(a.gene_name.clone()).or_insert(0) += 1;
        }
        let keep_gene = |g: &str| sites_per_gene.get(g).copied().unwrap_or(0) >= 2;
        let (before_c, before_a) = (all_counts.len(), all_annotations.len());
        all_annotations.retain(|a| keep_gene(&a.gene_name));
        all_counts.retain(|c| {
            let gene = c
                .site_id
                .split_once("/pA/")
                .map(|(g, _)| g)
                .unwrap_or(&c.site_id);
            keep_gene(gene)
        });
        info!(
            "drop-single-component: {} -> {} counts, {} -> {} sites",
            before_c,
            all_counts.len(),
            before_a,
            all_annotations.len()
        );
    }

    if all_counts.is_empty() {
        info!("no counts to output");
        return Ok(());
    }

    // Compute PDUI before consuming all_counts (per batch, see below)
    if args.compute_pdui {
        compute_and_write_pdui(&all_counts, &all_annotations, &utrs, args)?;
    }

    // The SCAPE fit is shared (pooled across BAMs), but each replicate gets
    // its own `{batch}_apa_mixture` matrix. Rows (GENE/pA/component) are a
    // shared vocabulary, so reorder to a sorted union for stackability.
    let batch_names = uniq_batch_names(&args.bam_files)?;
    let mut by_batch: rustc_hash::FxHashMap<u32, Vec<(CellBarcode, Box<str>, f32)>> =
        rustc_hash::FxHashMap::default();
    for c in all_counts {
        by_batch
            .entry(c.batch)
            .or_default()
            .push((c.cell_barcode, c.site_id, c.count as f32));
    }

    let mut all_rows = rustc_hash::FxHashSet::<Box<str>>::default();
    let mut out_files: Vec<crate::pipeline_util::BackendOutputPath> = Vec::new();
    // The per-cell component matrix is opt-in (`--mixture`); the SCAPE fit above
    // already ran because PDUI needs it, so this only gates the extra output.
    if args.write_mixture {
        for (batch_idx, batch_name) in batch_names.iter().enumerate() {
            let Some(trip) = by_batch.remove(&(batch_idx as u32)) else {
                continue;
            };
            if trip.is_empty() {
                continue;
            }
            let out = args.backend_output_path(&format!("{}_apa_mixture", batch_name));
            let data = format_data_triplets(trip).to_backend(&out.write_path)?;
            data.qc(SqueezeCutoffs {
                row: args.row_nnz_cutoff,
                column: args.column_nnz_cutoff,
            })?;
            all_rows.extend(data.row_names()?);
            info!("created output: {}", &out.target_path);
            drop(data);
            out_files.push(out);
        }
    }

    let mut rows_sorted: Vec<_> = all_rows.into_iter().collect();
    rows_sorted.sort();
    for out in &out_files {
        open_sparse_matrix(&out.write_path, &args.backend)?.reorder_rows(&rows_sorted)?;
    }
    for out in out_files {
        out.finalize()?;
    }

    // Write APA site annotation Parquet (shared definitions, single file)
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
        let mut records = read_gff_record_vec(gff_file)?;
        info!("read {} GFF records", records.len());
        // Honor --gene-type on the mixture path (run_simple subsets the GffRecordMap;
        // mixture builds UTRs straight from records, so filter here). The pipeline
        // leaves this None and inherits the biotype subset via valid_gene_ids.
        if let Some(ref gt) = args.gene_type {
            let before = records.len();
            records.retain(|r| r.gene_type == *gt);
            info!(
                "gene-type filter: {} -> {} GFF records",
                before,
                records.len()
            );
        }

        let model = build_union_gene_model(&records)?;
        info!(
            "found {} 3'-UTR regions in gene model",
            model.three_prime_utr.len()
        );

        let mut utrs = Vec::new();
        for entry in model.three_prime_utr.iter() {
            let gene_id = entry.key();
            let rec = entry.value();
            // GFF coords are 1-based inclusive; nucleotide span is stop - start + 1.
            let utr_length = (rec.stop - rec.start + 1) as usize;
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

/// ATOI/SNP position masks pooled by chromosome: `chr -> {genomic_pos}`. Keying
/// by chr lets `process_utr` do one borrowed lookup per UTR and an i64-only
/// membership test per candidate (no per-candidate chr-string clone or hash).
type PolyaSiteMask = rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<i64>>;

/// Load and pool the ATOI + SNP position masks referenced by `CountApaArgs`.
/// Returns `None` when neither is set. The mixture path checks discovered
/// candidate poly-A sites against this union and drops any that coincide with an
/// A-to-I edit or SNP — parity with run_simple.
fn load_polya_site_mask(args: &CountApaArgs) -> anyhow::Result<Option<PolyaSiteMask>> {
    let mut mask: PolyaSiteMask = rustc_hash::FxHashMap::default();
    if let Some(ref f) = args.atoi_mask_file {
        let m = crate::editing::io::load_atoi_mask_from_parquet(f.as_ref())?;
        info!("APA: loaded {} ATOI mask positions from {}", m.len(), f);
        for (chr, pos) in m {
            mask.entry(chr).or_default().insert(pos);
        }
    }
    if let Some(ref f) = args.snp_mask_file {
        let m = crate::snp::io::load_snp_mask_from_parquet(f.as_ref())?;
        info!("APA: loaded {} SNP mask positions from {}", m.len(), f);
        for (chr, pos) in m {
            mask.entry(chr).or_default().insert(pos);
        }
    }
    Ok(if mask.is_empty() { None } else { Some(mask) })
}

/// Fast 2-site PDUI path: treat a UTR as effectively single-site (no PDUI)
/// unless the runner-up cluster carries at least `1/this` of the dominant
/// cluster's reads. Guards against calling PDUI on a spurious minor peak.
const MIN_RUNNERUP_MASS_RATIO: usize = 10;

/// Process a single UTR: extract fragments, discover/load sites, run EM, assign cells.
#[allow(clippy::too_many_arguments)]
fn process_utr(
    cache: &mut crate::data::bam_io::BamReaderCache,
    utr: &UtrRegion,
    bam_files: &[Box<str>],
    pre_sites: Option<&rustc_hash::FxHashMap<Box<str>, Vec<f32>>>,
    site_mask: Option<&PolyaSiteMask>,
    extract_ns: &std::sync::atomic::AtomicU64,
    bic_ns: &std::sync::atomic::AtomicU64,
    discover_ns: &std::sync::atomic::AtomicU64,
    assign_ns: &std::sync::atomic::AtomicU64,
    args: &CountApaArgs,
) -> anyhow::Result<(Vec<CellSiteCount>, Vec<ApaSiteAnnotation>)> {
    let mut all_fragments = Vec::new();
    // Parallel to `all_fragments`: the batch (replicate) each fragment came
    // from. Sites are fit on the pooled fragments, but counts are emitted
    // per batch, so each fragment must remember its origin.
    let mut frag_batch: Vec<u32> = Vec::new();
    let t_extract = std::time::Instant::now();
    for (batch_idx, bam_file) in bam_files.iter().enumerate() {
        let polya_params = PolyAFilterParams {
            min_tail: args.polya_min_tail_length,
            max_non_at: args.polya_max_non_a_or_t,
            internal_prime_window: args.polya_internal_prime_window,
            internal_prime_count: args.polya_internal_prime_count,
        };
        let frags = extract_fragments_cached(
            cache,
            bam_file,
            utr,
            args.cell_barcode_tag.as_bytes(),
            args.umi_tag.as_bytes(),
            &polya_params,
            args.min_mapping_quality,
        )?;
        frag_batch.extend(std::iter::repeat_n(batch_idx as u32, frags.len()));
        all_fragments.extend(frags);
    }
    extract_ns.fetch_add(
        t_extract.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );

    // When UMI dedup is disabled, give each fragment a unique UMI hash so the
    // (cell, component) HashSet in cell_assign sees them as distinct reads
    // rather than collapsing all UmiBarcode::Missing into one observation.
    if args.no_umi_dedup {
        for (i, frag) in all_fragments.iter_mut().enumerate() {
            frag.umi = genomic_data::sam::UmiBarcode::Hash(i as u64);
        }
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

    // Fast PDUI path (default; no --mixture): find pA clusters by recursive mass
    // bisection straight from read positions — no KDE / merge / EM — then take
    // the two highest-mass clusters and hard-assign reads to the nearest.
    let fast_pdui = args.compute_pdui && !args.write_mixture && !args.apa_em_pdui;
    if fast_pdui {
        let t_discover = std::time::Instant::now();
        let mut positions: Vec<f32> = all_fragments
            .iter()
            .map(|f| f.pa_site.unwrap_or(f.x + f.l))
            .collect();
        positions.sort_unstable_by(f32::total_cmp);
        let mut clusters = crate::apa::site_discovery::discover_sites_bisect(
            &positions,
            args.merge_distance,
            args.min_coverage,
        );
        // Drop clusters coinciding with an A-to-I edit / SNP (parity with the EM path).
        if let Some(masked) = site_mask.and_then(|m| m.get(&*utr.chr)) {
            clusters.retain(|&(alpha, _)| {
                !masked.contains(&utr.alpha_to_genomic_range(alpha as f64, 0.0).0)
            });
        }
        discover_ns.fetch_add(
            t_discover.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if clusters.len() < 2 {
            return Ok((Vec::new(), Vec::new())); // single-site → no PDUI
        }
        clusters.sort_unstable_by(|a, b| b.1.cmp(&a.1)); // by read count, desc
                                                         // Require the runner-up to be a non-trivial fraction of the dominant peak.
        if clusters[1].1 * MIN_RUNNERUP_MASS_RATIO < clusters[0].1 {
            return Ok((Vec::new(), Vec::new()));
        }
        let t_assign = std::time::Instant::now();
        let sites = [clusters[0].0, clusters[1].0];
        let beta = (args.min_beta + args.max_beta) / 2.0;
        let out = crate::apa::cell_assign::assign_fragments_two_site_fast(
            &all_fragments,
            &frag_batch,
            sites,
            beta,
            utr,
        );
        assign_ns.fetch_add(
            t_assign.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        return Ok(out);
    }

    let t_discover = std::time::Instant::now();
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

    // Drop candidate sites that coincide with an A-to-I edit or SNP. Candidate
    // positions are UTR-relative alpha; map each to genomic via the same transform
    // the EM uses, then test (chr, pos) membership. Parity with run_simple.
    let candidate_sites: Vec<f32> = match site_mask.and_then(|m| m.get(&*utr.chr)) {
        Some(masked_pos) => {
            let before = candidate_sites.len();
            let kept: Vec<f32> = candidate_sites
                .into_iter()
                .filter(|&alpha| {
                    let g = utr.alpha_to_genomic_range(alpha as f64, 0.0).0;
                    !masked_pos.contains(&g)
                })
                .collect();
            if kept.len() != before {
                log::debug!(
                    "UTR {}: masked candidate sites {} -> {}",
                    utr.name,
                    before,
                    kept.len()
                );
            }
            kept
        }
        None => candidate_sites,
    };
    discover_ns.fetch_add(
        t_discover.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );

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

    // Cluster fragments before EM: the SCAPE likelihood depends only on
    // (x, l, r, is_junction), so fragments sharing this tuple are
    // mathematically identical and we can run inference on the
    // representatives with multiplicity weights. Heavy UTRs typically
    // collapse 5-50× — every downstream phase (theta_lik_matrix,
    // all_site_lls, EM iterations) scales with M = # clusters, not N.
    let bins = crate::apa::fragment::ClusterBins::default();
    let (clusters, cluster_idx) = crate::apa::fragment::cluster_fragments(&all_fragments, bins);
    let cluster_counts: Vec<f32> = clusters.iter().map(|c| c.count as f32).collect();
    let n_for_bic = all_fragments.len();
    log::debug!(
        "UTR {}: clustered {} → {} ({:.1}×)",
        utr.name,
        all_fragments.len(),
        clusters.len(),
        all_fragments.len() as f32 / clusters.len().max(1) as f32,
    );

    let (theta_lik_matrix, theta_grid) =
        precompute_theta_lik_matrix(&clusters, utr.utr_length as f32, &lik_params);

    let default_beta = (args.min_beta + args.max_beta) / 2.0;
    let beta_arr: Vec<f32> = vec![default_beta; candidate_sites.len()];

    let em_params = args.em_params();

    // Rank candidate sites by nearby fragment mass (descending) for greedy BIC
    // model selection. Score on the clustered representatives (count = multiplicity,
    // pa_site preserved within a 5bp bin) via a sorted-candidate sweep — O(M log K)
    // instead of O(K*N) over raw fragments. site_order only sets the greedy add
    // order, so the small binning is immaterial.
    let merge_dist = args.merge_distance;
    let site_order = {
        let mut sorted_cands: Vec<(f32, usize)> = candidate_sites
            .iter()
            .copied()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        sorted_cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut counts = vec![0u32; candidate_sites.len()];
        for cl in &clusters {
            if let Some(pa) = cl.pa_site {
                // Candidates with |pa - site| < merge_dist (open interval, matching
                // the original strict comparison).
                let lo = sorted_cands.partition_point(|&(s, _)| s <= pa - merge_dist);
                let hi = sorted_cands.partition_point(|&(s, _)| s < pa + merge_dist);
                for &(_, idx) in &sorted_cands[lo..hi] {
                    counts[idx] += cl.count;
                }
            }
        }
        let mut order: Vec<usize> = (0..candidate_sites.len()).collect();
        order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
        order
    };

    let site_data = SiteModelData {
        alpha_arr: &candidate_sites,
        beta_arr: &beta_arr,
        theta_lik_matrix: &theta_lik_matrix,
        theta_grid: &theta_grid,
        cluster_counts: &cluster_counts,
        n_for_bic,
        utr_length: utr.utr_length as f32,
        max_polya: lik_params.max_polya,
    };
    let t_bic = std::time::Instant::now();
    let em_result = select_sites_by_bic(&site_data, &em_params, &site_order);
    bic_ns.fetch_add(
        t_bic.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );

    log::debug!(
        "UTR {}: BIC selected {} sites (from {} candidates), BIC={:.1}, LL={:.1}, {} iters",
        utr.name,
        em_result.alphas.len(),
        candidate_sites.len(),
        em_result.bic,
        em_result.log_lik,
        em_result.n_iter,
    );

    let (cell_counts, annotations) =
        assign_fragments_to_sites(&all_fragments, &cluster_idx, &frag_batch, &em_result, utr);

    Ok((cell_counts, annotations))
}

/// Compute PDUI (per batch) for genes with exactly 2 active pA sites and
/// write one `{batch}_apa_pdui` sparse matrix per replicate. The 2-site
/// definitions are shared (pooled fit); only the per-cell counts split by
/// batch, so the matrices share a gene (row) vocabulary.
fn compute_and_write_pdui(
    all_counts: &[CellSiteCount],
    all_annotations: &[ApaSiteAnnotation],
    utrs: &[UtrRegion],
    args: &CountApaArgs,
) -> anyhow::Result<()> {
    use crate::apa::pdui::compute_pdui;
    use rustc_hash::FxHashMap;

    info!("Computing PDUI (per batch)...");

    // Build a strand lookup from UTRs by gene name
    let strand_by_gene: FxHashMap<Box<str>, genomic_data::sam::Strand> =
        utrs.iter().map(|u| (u.name.clone(), u.strand)).collect();

    // Group annotations by gene_name (shared 2-site definitions)
    let mut annots_by_gene: FxHashMap<Box<str>, Vec<ApaSiteAnnotation>> = FxHashMap::default();
    for a in all_annotations {
        annots_by_gene
            .entry(a.gene_name.clone())
            .or_default()
            .push(a.clone());
    }

    // Group counts by (batch, gene)
    let mut counts_by_batch_gene: FxHashMap<(u32, Box<str>), Vec<&CellSiteCount>> =
        FxHashMap::default();
    for c in all_counts {
        let gene_name = c
            .site_id
            .find("/pA/")
            .map(|pos| &c.site_id[..pos])
            .unwrap_or(c.site_id.as_ref());
        counts_by_batch_gene
            .entry((c.batch, gene_name.into()))
            .or_default()
            .push(c);
    }

    let batch_names = uniq_batch_names(&args.bam_files)?;
    let mut all_rows = rustc_hash::FxHashSet::<Box<str>>::default();
    let mut out_files: Vec<crate::pipeline_util::BackendOutputPath> = Vec::new();

    use crate::pipeline_util::push_channel_row;
    use faba::feature_name::{APA, DISTAL, PROXIMAL};
    for (batch_idx, batch_name) in batch_names.iter().enumerate() {
        let b = batch_idx as u32;
        let mut apa_triplets: Vec<(CellBarcode, Box<str>, f32)> = Vec::new();
        let mut n_pdui_genes = 0;

        for (gene_name, annots) in &annots_by_gene {
            if annots.len() != 2 {
                continue;
            }
            let strand = match strand_by_gene.get(gene_name) {
                Some(s) => *s,
                None => continue,
            };
            let gene_counts: &[&CellSiteCount] = counts_by_batch_gene
                .get(&(b, gene_name.clone()))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if gene_counts.is_empty() {
                continue;
            }
            if let Some(pdui_result) = compute_pdui(gene_counts, annots, strand) {
                n_pdui_genes += 1;
                // Emit proximal/distal COUNTS as channel rows (aggregated per
                // gene = the 2-site UTR): {gene}/apa/{proximal,distal}.
                for (cb, prox, dist) in &pdui_result.cell_counts {
                    push_channel_row(&mut apa_triplets, cb, gene_name, APA, PROXIMAL, *prox);
                    push_channel_row(&mut apa_triplets, cb, gene_name, APA, DISTAL, *dist);
                }
            }
        }

        if apa_triplets.is_empty() {
            continue;
        }
        info!(
            "APA[{}]: {} genes, {} cell-gene channel values",
            batch_name,
            n_pdui_genes,
            apa_triplets.len()
        );
        let out = args.backend_output_path(&format!("{}_apa", batch_name));
        let data = format_data_triplets(apa_triplets).to_backend(&out.write_path)?;
        data.qc(SqueezeCutoffs {
            row: args.row_nnz_cutoff,
            column: args.column_nnz_cutoff,
        })?;
        all_rows.extend(data.row_names()?);
        info!("PDUI: created {}", &out.target_path);
        drop(data);
        out_files.push(out);
    }

    if out_files.is_empty() {
        info!("No PDUI values to output");
        return Ok(());
    }

    let mut rows_sorted: Vec<_> = all_rows.into_iter().collect();
    rows_sorted.sort();
    for out in &out_files {
        open_sparse_matrix(&out.write_path, &args.backend)?.reorder_rows(&rows_sorted)?;
    }
    for out in out_files {
        out.finalize()?;
    }
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
    let mut utr_lengths: Vec<u32> = Vec::with_capacity(annotations.len());

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
        utr_lengths.push(a.utr_length);
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
        arrow::datatypes::Field::new("utr_length", arrow::datatypes::DataType::UInt32, false),
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
            Arc::new(UInt32Array::from(utr_lengths)) as ArrayRef,
        ],
    )?;

    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
