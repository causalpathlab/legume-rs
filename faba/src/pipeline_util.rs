pub mod mass_enrichment;

use crate::common::*;
use crate::data::conversion::*;
use crate::data::util_htslib::*;
use crate::gene_count::splice::{count_read_per_gene_splice, format_gene_key};

use dashmap::DashMap as HashMap;
use data_beans::zarr_io::finalize_zarr_output;
use genomic_data::gff::GffRecordMap;
use genomic_data::sam::CellBarcode;
use rustc_hash::FxHashMap;

/// Backend output filenames for a given `(output_dir, batch_name)` pair.
///
/// `write_path` is what we hand to the SparseIo backend (a `.zarr` directory
/// or `.h5` file). `target_path` is what the user actually wants on disk,
/// either the same as `write_path` (for HDF5 or zarr-without-zip) or its
/// `.zarr.zip` archive. After the backend has been fully written, call
/// `finalize_backend_output(&write_path, &target_path)` to zip up Zarr
/// outputs when applicable (no-op otherwise).
pub struct BackendOutputPath {
    pub write_path: Box<str>,
    pub target_path: Box<str>,
}

impl BackendOutputPath {
    pub fn new(output_dir: &str, name: &str, backend: &SparseIoBackend, zip: bool) -> Self {
        match backend {
            SparseIoBackend::HDF5 => {
                let p: Box<str> = format!("{}/{}.h5", output_dir, name).into_boxed_str();
                Self {
                    write_path: p.clone(),
                    target_path: p,
                }
            }
            SparseIoBackend::Zarr if zip => Self {
                write_path: format!("{}/{}.zarr", output_dir, name).into_boxed_str(),
                target_path: format!("{}/{}.zarr.zip", output_dir, name).into_boxed_str(),
            },
            SparseIoBackend::Zarr => {
                let p: Box<str> = format!("{}/{}.zarr", output_dir, name).into_boxed_str();
                Self {
                    write_path: p.clone(),
                    target_path: p,
                }
            }
        }
    }

    /// Zip the staging `.zarr` directory into the `.zarr.zip` target (no-op for
    /// HDF5 and for zarr without zip). Call once the backend has been written
    /// and all open handles dropped.
    pub fn finalize(&self) -> anyhow::Result<()> {
        finalize_zarr_output(&self.write_path, &self.target_path)
    }
}

/// Check BAM indices for all files
pub fn check_all_bam_indices(bam_files: &[Box<str>]) -> anyhow::Result<()> {
    for bam_file in bam_files {
        info!("checking .bai file for {}...", bam_file);
        check_bam_index(bam_file, None)?;
    }
    Ok(())
}

/// Build gene key as `{gene_id}_{symbol}` for feature naming.
///
/// Feature naming convention: `{gene_key}/{modality}/{detail}`
pub fn create_gene_key_function(
    gff_map: &GffRecordMap,
) -> impl Fn(&BedWithGene) -> Box<str> + Send + Sync + '_ {
    |x: &BedWithGene| -> Box<str> {
        gff_map
            .get(&x.gene)
            .map(|gff| format!("{}_{}", gff.gene_id, gff.gene_name))
            .unwrap_or_else(|| format!("{}", x.gene))
            .into_boxed_str()
    }
}

/// Push a channel-last row = `count` into `triplets`, skipping zeros to keep the
/// matrix sparse. `subunit = None` emits a gene-level `{gene}/{modality}/{channel}`
/// row; `Some(s)` emits a sub-gene `{gene}/{modality}/{s}/{channel}` row (site or
/// component). The single place the channel-count producers (editing, APA) spell
/// the [`feature_row`] convention.
///
/// [`feature_row`]: faba::feature_name::feature_row
pub fn push_channel_row(
    triplets: &mut Vec<(CellBarcode, Box<str>, f32)>,
    cb: &CellBarcode,
    gene: &str,
    modality: &str,
    channel: &str,
    subunit: Option<&str>,
    count: usize,
) {
    if count > 0 {
        triplets.push((
            cb.clone(),
            faba::feature_name::feature_row(gene, modality, channel, subunit),
            count as f32,
        ));
    }
}

/// Aggregate conversion stats to **gene level** and emit two channel rows per
/// gene into one matrix, in the channel-last convention
/// ([`crate::feature_name`]):
///
/// ```text
/// {gene}/{modality}/{pos_channel} = Σ_sites converted    (e.g. methylated / edited)
/// {gene}/{modality}/{neg_channel} = Σ_sites unconverted  (e.g. unmethylated / unedited)
/// ```
///
/// All of a gene's sites are pooled per cell (both channels ride in
/// [`ConversionData`]); zero counts are skipped to keep the matrix sparse. This
/// is the gene-per-channel `(positive, coverage)` form the co-embedding consumes.
pub fn summarize_stats_two_channel<F>(
    stats: &[(CellBarcode, BedWithGene, ConversionData)],
    gene_key_func: F,
    modality: &str,
    pos_channel: &str,
    neg_channel: &str,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> Box<str> + Send + Sync,
{
    // Pool sites → gene per cell (sums converted + unconverted across sites).
    let combined: HashMap<(CellBarcode, Box<str>), ConversionData> = HashMap::default();
    stats.par_iter().for_each(|(cb, bed, dat)| {
        let key = (cb.clone(), gene_key_func(bed));
        combined.entry(key).or_default().add_assign(dat);
    });

    // Two channel rows per (cell, gene); drop zeros (sparse).
    let mut triplets: Vec<(CellBarcode, Box<str>, f32)> = Vec::with_capacity(combined.len() * 2);
    for ((cb, gene), dat) in combined {
        push_channel_row(
            &mut triplets,
            &cb,
            &gene,
            modality,
            pos_channel,
            None,
            dat.converted,
        );
        push_channel_row(
            &mut triplets,
            &cb,
            &gene,
            modality,
            neg_channel,
            None,
            dat.unconverted,
        );
    }
    format_data_triplets(triplets)
}

/// Aggregate conversion stats to **per-site** resolution and emit two channel
/// rows per site into one matrix, using the single-base site as the subunit
/// ([`crate::feature_name`]):
///
/// ```text
/// {gene}/{modality}/{chr}:{pos}/{pos_channel} = converted    (methylated / edited)
/// {gene}/{modality}/{chr}:{pos}/{neg_channel} = unconverted  (unmethylated / unedited)
/// ```
///
/// m6A and A-to-I sites are single base pairs, so the subunit is just the
/// 0-based position `{chr}:{start}` — not a `start-stop` interval. This is the
/// finer-grained sibling of [`summarize_stats_two_channel`], which pools every
/// site into its gene; here distinct sites stay separate rows. Zero counts are
/// skipped to keep the matrix sparse.
///
/// `min_cells` applies a **unit-aware** feature QC: a site is kept only if it is
/// detected in at least `min_cells` cells, and when kept BOTH of its channel
/// rows are emitted together (never de-paired — mirrors the way Cell Ranger
/// keeps all of a gene's features together rather than filtering channels
/// independently). `0` or `1` disables the filter. The gene axis is already
/// filtered upstream by `gene_min_cells` (see [`passing_genes_from_stats`]); the
/// per-site axis is new feature space that nothing else QCs. Note `stats` only
/// carries cells with a converted read at the site, so this counts
/// signal-bearing cells.
pub fn summarize_stats_per_site<F>(
    stats: &[(CellBarcode, BedWithGene, ConversionData)],
    gene_key_func: F,
    modality: &str,
    pos_channel: &str,
    neg_channel: &str,
    min_cells: usize,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> Box<str> + Send + Sync,
{
    // Pool per (cell, gene, site). Distinct sites within a gene stay separate;
    // repeated observations of the same (cell, site) sum.
    let combined: HashMap<(CellBarcode, Box<str>, Box<str>), ConversionData> = HashMap::default();
    stats.par_iter().for_each(|(cb, bed, dat)| {
        let gene = gene_key_func(bed);
        let site: Box<str> = format!("{}:{}", bed.chr, bed.start).into();
        let key = (cb.clone(), gene, site);
        combined.entry(key).or_default().add_assign(dat);
    });

    // Drain into a Vec so we can count cells per site, then filter sequentially.
    let entries: Vec<_> = combined.into_iter().collect();

    // Unit-aware min-cells: cells detected per (gene, site). Borrow keys from
    // `entries` to avoid clones; only built when the filter is active.
    let cells_per_site: rustc_hash::FxHashMap<(&str, &str), usize> = if min_cells > 1 {
        let mut m: rustc_hash::FxHashMap<(&str, &str), usize> = rustc_hash::FxHashMap::default();
        for ((_, gene, site), _) in &entries {
            *m.entry((gene.as_ref(), site.as_ref())).or_insert(0) += 1;
        }
        m
    } else {
        rustc_hash::FxHashMap::default()
    };
    let keep = |gene: &str, site: &str| -> bool {
        min_cells <= 1
            || cells_per_site
                .get(&(gene, site))
                .is_some_and(|&n| n >= min_cells)
    };

    // Two channel rows per kept (cell, gene, site); drop zeros (sparse).
    let mut triplets: Vec<(CellBarcode, Box<str>, f32)> = Vec::with_capacity(entries.len() * 2);
    for ((cb, gene, site), dat) in &entries {
        if !keep(gene, site) {
            continue;
        }
        push_channel_row(
            &mut triplets,
            cb,
            gene,
            modality,
            pos_channel,
            Some(site.as_ref()),
            dat.converted,
        );
        push_channel_row(
            &mut triplets,
            cb,
            gene,
            modality,
            neg_channel,
            Some(site.as_ref()),
            dat.unconverted,
        );
    }
    format_data_triplets(triplets)
}

///////////////////
// Gene count QC //
///////////////////

/// Gene-count QC result. Genes are pooled across batches (a shared feature
/// vocabulary), but retained cells are kept **per batch** — each BAM is a
/// separate library with its own cell-calling knee, and the same barcode string
/// in two libraries denotes different cells, so they must not be unioned.
pub struct GeneCountQc {
    pub gene_ids: rustc_hash::FxHashSet<GeneId>,
    /// Per-library valid cells, keyed by **BAM file path** (stable and
    /// order-independent). Basenames collide across 10x libraries
    /// (`possorted_genome_bam.bam`), so a positional batch name would mis-key
    /// between the QC pass and the quant pass when their BAM orderings differ.
    pub cells_by_batch: rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>>,
    /// Persisted per-batch gene-count matrix path (target on disk), keyed by
    /// **BAM file path** like `cells_by_batch`. Empty when QC did not write a
    /// matrix (e.g. the `--valid-cells` reuse path). Consumers (mass enrichment)
    /// load these instead of re-scanning the BAMs.
    pub matrix_by_batch: rustc_hash::FxHashMap<Box<str>, Box<str>>,
}

/// How [`run_gene_count_qc`] persists the gene-count matrix it already builds for
/// QC. Only the *matrix* is optional — the QC artifacts always land in
/// [`GeneQcRequest::output_dir`], so a run can never drop cells without leaving
/// the `{batch}_mt_qc.tsv.gz` record of why.
pub struct GeneMatrixSink<'a> {
    pub backend: &'a SparseIoBackend,
    pub zip: bool,
}

/// Extract gene_key from a feature name like `"GENE_SYM/count/spliced"` → `"GENE_SYM"`.
#[inline]
pub fn extract_gene_key(feat: &str) -> &str {
    feat.rfind("/count/")
        .map(|pos| &feat[..pos])
        .unwrap_or(feat)
}

/// One batch's QC primitives, computed in a single pass over its triplets:
///
/// - `gene_stats`: per gene_key, the `(nnz, total)` = (# cells expressing it,
///   summed counts) on the **total** (spliced + unspliced) track. `total` is
///   only populated when [`batch_qc`] is asked for it. Owned keys so callers
///   can accumulate them across batches (pooled gene QC) before thresholding.
/// - `passing_cells`: the **spliced-only** cell call (Cell Ranger-faithful).
pub struct BatchQc {
    pub gene_stats: FxHashMap<Box<str>, (usize, f64)>,
    pub passing_cells: rustc_hash::FxHashSet<CellBarcode>,
}

/// Compute one batch's [`BatchQc`]. The two QC layers are deliberately scoped
/// differently:
///
/// - **Gene stats** count spliced + unspliced together, so a gene whose mass is
///   entirely intronic (zero spliced, non-zero unspliced) is not dropped.
/// - **Cell calling** stays spliced-only — Cell Ranger calls cells from exonic
///   gene-expression UMIs, so the nnz floor and the cell-calling policy
///   (OrdMag/EmptyDrops/…) both see only the spliced track.
///
/// `want_total` populates the per-gene summed counts (skip the extra pass when
/// no min-count gate is set). Pass an empty `unspliced` slice for
/// non-splice-aware callers (gene stats then cover the single `spliced` track).
pub fn batch_qc(
    spliced: &[(CellBarcode, Box<str>, f32)],
    unspliced: &[(CellBarcode, Box<str>, f32)],
    want_total: bool,
    cell_min_genes: usize,
    cell_call: &crate::cell_qc::CellCallParams,
) -> BatchQc {
    // Unique (cell, gene_key) pairs on the spliced track. Hashed once and then
    // reused for both QC layers: cell QC reads it as-is, gene QC folds the
    // unspliced keys on top.
    let spliced_pairs: rustc_hash::FxHashSet<(&CellBarcode, &str)> = spliced
        .par_iter()
        .map(|(cb, feat, _)| (cb, extract_gene_key(feat)))
        .collect();

    // Cell nnz (spliced only).
    let mut cell_nnz: FxHashMap<&CellBarcode, usize> = FxHashMap::default();
    for &(cb, _) in &spliced_pairs {
        *cell_nnz.entry(cb).or_default() += 1;
    }

    // Gene nnz over spliced + unspliced: reuse the spliced pair-set and fold in
    // the unspliced keys instead of re-hashing spliced.
    let mut gene_cell_pairs = spliced_pairs;
    gene_cell_pairs.extend(
        unspliced
            .iter()
            .map(|(cb, feat, _)| (cb, extract_gene_key(feat))),
    );

    let mut gene_nnz: FxHashMap<&str, usize> = FxHashMap::default();
    for &(_, gk) in &gene_cell_pairs {
        *gene_nnz.entry(gk).or_default() += 1;
    }

    // Total counts per gene (spliced + unspliced), only when requested.
    let gene_total: FxHashMap<&str, f64> = if want_total {
        let mut m: FxHashMap<&str, f64> = FxHashMap::default();
        for (_, feat, v) in spliced.iter().chain(unspliced.iter()) {
            *m.entry(extract_gene_key(feat)).or_default() += *v as f64;
        }
        m
    } else {
        FxHashMap::default()
    };

    // Own the gene keys so the stats can be summed across batches.
    let gene_stats: FxHashMap<Box<str>, (usize, f64)> = gene_nnz
        .into_iter()
        .map(|(gk, nnz)| {
            let total = if want_total {
                gene_total.get(gk).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            (Box::from(gk), (nnz, total))
        })
        .collect();

    // The `cell_min_genes` nnz floor always applies. Beyond it, the cell-calling
    // policy (OrdMag/EmptyDrops/min-counts) decides which barcodes are real
    // cells; `Nnz` keeps the raw superset (today's behaviour).
    let nnz_cells: rustc_hash::FxHashSet<CellBarcode> = cell_nnz
        .into_iter()
        .filter(|(_, n)| *n >= cell_min_genes)
        .map(|(cb, _)| cb.clone())
        .collect();

    let passing_cells: rustc_hash::FxHashSet<CellBarcode> =
        if cell_call.filter == crate::cell_qc::CellFilter::Nnz {
            nnz_cells
        } else {
            // Spliced-only QC: no unspliced contribution to the cell call.
            let counts = crate::cell_qc::CellCounts::from_triplets(spliced, &[]);
            crate::cell_qc::call_cells(&counts, cell_call)
                .into_iter()
                .filter(|cb| nnz_cells.contains(cb))
                .collect()
        };

    BatchQc {
        gene_stats,
        passing_cells,
    }
}

/// Apply the gene thresholds to per-gene `(nnz, total)` stats. The same predicate
/// serves a single batch (one [`qc_one_batch`] result) or stats pooled across
/// batches. A `gene_min_counts` of 0 disables the total-count threshold.
pub fn passing_genes_from_stats(
    gene_stats: &FxHashMap<Box<str>, (usize, f64)>,
    gene_min_cells: usize,
    gene_min_counts: usize,
) -> rustc_hash::FxHashSet<Box<str>> {
    let min_counts = gene_min_counts as f64;
    gene_stats
        .iter()
        .filter(|(_, (nnz, _))| *nnz >= gene_min_cells)
        .filter(|(_, (_, total))| gene_min_counts == 0 || *total >= min_counts)
        .map(|(gk, _)| gk.clone())
        .collect()
}

/// Fold one batch's [`BatchQc::gene_stats`] into a running pooled map, so the
/// gene filter can be applied once on the whole dataset.
///
/// Summing per-batch `(nnz, total)` is exact: each cell belongs to exactly one
/// library, so per-batch cell counts add up to the dataset-wide count per gene.
/// Barcode *strings* can repeat across libraries (see [`GeneCountQc`]), but this
/// sums per-gene counts rather than unioning barcodes, so the collisions are
/// irrelevant here.
pub fn accumulate_gene_stats(
    pooled: &mut FxHashMap<Box<str>, (usize, f64)>,
    batch_stats: FxHashMap<Box<str>, (usize, f64)>,
) {
    for (gene_key, (nnz, total)) in batch_stats {
        let e = pooled.entry(gene_key).or_default();
        e.0 += nnz;
        e.1 += total;
    }
}

/// Splice-aware gene counting + QC over every batch: count → [`qc_one_batch`] →
/// per-batch matrix → pooled gene vocabulary.
///
/// **The one gene-counting loop.** `faba all` (via `run_gene_counting_step`) and
/// the shared modality QC behind `dartseq` / `atoi` / `apa` are both just calls to
/// this with a different [`GeneQcRequest`]. They used to be two near-identical
/// copies of this loop, and the mito cell filter was added to one and forgotten in
/// the other — the copies *were* the defect, so there is now only one.
/// (`faba genes` keeps its own writers: it emits a different set of matrices.)
pub fn run_gene_count_qc(gff_file: &str, req: &GeneQcRequest) -> anyhow::Result<GeneCountQc> {
    info!("=== Gene expression QC ===");

    let bam_files = req.bam_files;
    let persist = req.persist.as_ref();
    let batch_names = uniq_batch_names(bam_files)?;

    let all_records = read_gff_record_vec(gff_file)?;
    let gene_map = build_gene_map(&all_records, Some(&FeatureType::Gene))?;
    let exon_map = build_exon_intervals(&all_records);
    let exon_intervals: FxHashMap<GeneId, Vec<(i64, i64)>> = exon_map.into_iter().collect();

    let gff_map = GffRecordMap::from_map(gene_map);
    info!("Loaded {} genes for expression QC", gff_map.len());

    let records = gff_map.records();

    // Build gene_key → GeneId mapping for reverse lookup after QC
    let gene_key_to_id: FxHashMap<Box<str>, GeneId> = records
        .iter()
        .map(|rec| (format_gene_key(rec), rec.gene_id.clone()))
        .collect();

    // Resolved once, outside the batch loop: the gate is a property of the
    // annotation, not of a batch.
    let gate = GeneGate::new(&records, req.gene_type, req.mito.clone());

    let mut expressed_gene_ids: rustc_hash::FxHashSet<GeneId> = rustc_hash::FxHashSet::default();
    let mut cells_by_batch: rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>> =
        rustc_hash::FxHashMap::default();
    let mut matrix_by_batch: rustc_hash::FxHashMap<Box<str>, Box<str>> =
        rustc_hash::FxHashMap::default();
    // Gene stats pooled across batches → thresholded once on the whole dataset
    // (partition-invariant) rather than per-batch-then-union. See
    // [`accumulate_gene_stats`] for why summing per-batch counts is exact.
    let mut pooled_gene_stats: FxHashMap<Box<str>, (usize, f64)> = FxHashMap::default();

    for (bam_file, batch_name) in bam_files.iter().zip(batch_names.iter()) {
        let njobs = records.len() as u64;
        info!(
            "Counting genes (splice-aware) in {} ({} genes)",
            bam_file, njobs
        );

        let results: Vec<_> = records
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .map(|rec| {
                count_read_per_gene_splice(
                    bam_file,
                    rec,
                    &exon_intervals,
                    req.cell_barcode_tag,
                    req.gene_barcode_tag,
                    req.umi_tag,
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
            "{} spliced, {} unspliced triplets",
            spliced_triplets.len(),
            unspliced_triplets.len()
        );

        // Cell call + mito cell filter + this batch's QC artifacts, in the one
        // shared place (see `qc_one_batch`).
        let bq = qc_one_batch(
            &spliced_triplets,
            &unspliced_triplets,
            &gate,
            req.gene_min_counts,
            req.cell_min_genes,
            &req.cell_call,
            Some(QcArtifacts {
                dir: req.output_dir,
                batch_name,
            }),
        )?;

        // This batch's matrix carries only what we quantify: count-QC runs on every
        // gene, then the gate narrows to the selected biotype and drops mito genes
        // (unless --keep-mito). The shared vocabulary is pooled across batches and
        // thresholded after the loop.
        let passing_genes: rustc_hash::FxHashSet<Box<str>> =
            passing_genes_from_stats(&bq.gene_stats, req.gene_min_cells, req.gene_min_counts)
                .into_iter()
                .filter(|gk| gate.quantify(gk))
                .collect();
        accumulate_gene_stats(&mut pooled_gene_stats, bq.gene_stats);

        info!(
            "{}: {} genes, {} cells passed QC",
            batch_name,
            passing_genes.len(),
            bq.passing_cells.len()
        );

        // Persist the gene-count matrix we already built (no extra BAM scan), so
        // counts are reported in every mode and mass enrichment can reuse them.
        if let Some(sink) = persist {
            let keep = |t: Vec<(CellBarcode, Box<str>, f32)>| -> Vec<_> {
                t.into_par_iter()
                    .filter(|(cb, feat, _)| {
                        passing_genes.contains(extract_gene_key(feat))
                            && bq.passing_cells.contains(cb)
                    })
                    .collect::<Vec<_>>()
            };
            let spliced = keep(spliced_triplets);
            let unspliced = keep(unspliced_triplets);

            let UnionNames {
                col_names,
                cell_to_index,
                row_names,
                feature_to_index,
            } = collect_union_names(&spliced, &unspliced);

            let out = BackendOutputPath::new(
                req.output_dir,
                &format!("{}_genes", batch_name),
                sink.backend,
                sink.zip,
            );
            let merged: Vec<_> = spliced.into_iter().chain(unspliced).collect();
            format_data_triplets_shared(
                merged,
                &feature_to_index,
                &cell_to_index,
                row_names,
                col_names,
            )
            .to_backend(&out.write_path)?;
            out.finalize()?;
            info!(
                "{}: wrote spliced + unspliced to {}",
                batch_name, out.target_path
            );
            matrix_by_batch.insert(bam_file.clone(), out.target_path);
        }

        // Keyed by BAM file path (stable + order-independent across passes).
        cells_by_batch.insert(bam_file.clone(), bq.passing_cells);
    }

    // Threshold the pooled gene stats once → the shared vocabulary for
    // --valid-genes reuse and downstream modality masking. Same gate as the
    // per-batch matrices, so the frozen set ATOI/APA/m6A inherit carries the
    // biotype subset and the mito exclusion.
    let passing_genes =
        passing_genes_from_stats(&pooled_gene_stats, req.gene_min_cells, req.gene_min_counts);
    for gene_key in passing_genes.iter().filter(|gk| gate.quantify(gk)) {
        if let Some(gene_id) = gene_key_to_id.get(gene_key) {
            expressed_gene_ids.insert(gene_id.clone());
        }
    }
    write_qc_genes(req.output_dir, &expressed_gene_ids)?;

    let total_cells: usize = cells_by_batch.values().map(|s| s.len()).sum();
    info!(
        "Gene QC summary: {} genes, {} cells passed across {} BAM files",
        expressed_gene_ids.len(),
        total_cells,
        bam_files.len()
    );

    Ok(GeneCountQc {
        gene_ids: expressed_gene_ids,
        cells_by_batch,
        matrix_by_batch,
    })
}

/// Resolve a `--no-umi-dedup` / `--umi-tag` flag pair to the byte tag used for
/// deduplication, or `None` when dedup is disabled. Shared by every subcommand
/// that counts genes so the two flags resolve identically everywhere.
pub fn resolve_umi_tag(no_umi_dedup: bool, umi_tag: &str) -> Option<&[u8]> {
    if no_umi_dedup {
        None
    } else {
        Some(umi_tag.as_bytes())
    }
}

/// Inputs for [`resolve_gene_qc`]. Each modality runner builds one from its own
/// args struct (the field names differ — e.g. apa's `gff_file` is optional and
/// m6a counts over `wt_bam_files`), then the resolution logic is shared.
pub struct GeneQcRequest<'a> {
    pub bam_files: &'a [Box<str>],
    pub cell_barcode_tag: &'a str,
    pub gene_barcode_tag: &'a str,
    /// UMI BAM tag for dedup during QC counting; `None` counts reads.
    pub umi_tag: Option<&'a [u8]>,
    /// GFF for the recompute path; `None` skips recompute (reuse can still run).
    pub gff_file: Option<&'a str>,
    /// Where the QC artifacts land (`{batch}_cells.tsv.gz`, `{batch}_mt_qc.tsv.gz`,
    /// `genes_kept.tsv.gz`). Always written; see [`GeneMatrixSink`] for the
    /// separate, optional matrix write.
    pub output_dir: &'a str,
    /// Biotype to quantify; `""` keeps all. Narrows only what is *quantified* —
    /// cell calling always sees every biotype. The modality runners pass `""`:
    /// they apply their own `--gene-type` by subsetting the gff for site
    /// discovery, and their QC counts every biotype by design.
    pub gene_type: &'a str,
    pub gene_min_cells: usize,
    pub gene_min_counts: usize,
    pub cell_min_genes: usize,
    pub cell_call: crate::cell_qc::CellCallParams,
    /// Mitochondrial QC policy: the per-cell MT filter applied by [`qc_one_batch`]
    /// and the MT-gene exclusion applied to the quantified gene set.
    pub mito: MitoQcParams,
    pub valid_cells_file: Option<&'a str>,
    pub valid_genes_file: Option<&'a str>,
    pub skip_gene_qc: bool,
    /// When set, the QC pass persists the gene-count matrix it builds (so counts
    /// are reported in every mode, and mass enrichment can reuse it). `None`
    /// skips the write (e.g. the `--valid-cells` reuse path builds no matrix).
    pub persist: Option<GeneMatrixSink<'a>>,
}

/// Resolve a modality's gene-expression QC: reuse a passed per-batch cell set
/// (`--valid-cells` + optional `--valid-genes`) written by `faba genes`, or
/// recompute it in memory (per-batch cell calling). Returns `None` when QC is
/// skipped. **Convention:** an empty `gene_ids` means "no gene-level filter"
/// (e.g. `--valid-cells` without `--valid-genes`) — callers must treat it as
/// "keep all genes", not "filter to nothing". Does NOT mutate any gff/UTR
/// structure; the caller applies the gene set however it filters.
pub fn resolve_gene_qc(req: &GeneQcRequest) -> anyhow::Result<Option<GeneCountQc>> {
    if let Some(dir) = req.valid_cells_file {
        let cells_by_batch = load_valid_cells_dir(dir, req.bam_files)?;
        let gene_ids = match req.valid_genes_file {
            Some(gf) => load_valid_genes(gf)?,
            None => rustc_hash::FxHashSet::default(),
        };
        Ok(Some(GeneCountQc {
            gene_ids,
            cells_by_batch,
            // Reuse path (`--valid-cells`) recomputes no matrix.
            matrix_by_batch: rustc_hash::FxHashMap::default(),
        }))
    } else if !req.skip_gene_qc {
        match req.gff_file {
            Some(gff_file) => Ok(Some(run_gene_count_qc(gff_file, req)?)),
            None => Ok(None),
        }
    } else {
        Ok(None)
    }
}

/// [`resolve_gene_qc`] plus the gff-map retain that `run_atoi` / `run_m6a` need.
/// Retains `gff_map` to the QC-passing genes when a gene filter is present (a
/// non-empty `gene_ids`); an empty set leaves the gff untouched (keep all).
pub fn resolve_modality_gene_qc(
    gff_map: &mut GffRecordMap,
    req: &GeneQcRequest,
) -> anyhow::Result<Option<GeneCountQc>> {
    let qc = resolve_gene_qc(req)?;
    if let Some(ref qc) = qc {
        if !qc.gene_ids.is_empty() {
            gff_map.retain_by_ids(&qc.gene_ids);
            info!("After gene QC: {} genes retained", gff_map.len());
        }
    }
    Ok(qc)
}

/// Write a batch's retained cell barcodes to `{dir}/{batch}_cells.tsv.gz`
/// (one barcode per line) — the passable artifact consumed by
/// [`load_valid_cells_dir`].
pub fn write_qc_cells(
    dir: &str,
    batch_name: &str,
    cells: &rustc_hash::FxHashSet<CellBarcode>,
) -> anyhow::Result<()> {
    let mut lines: Vec<Box<str>> = cells
        .iter()
        .map(|c| c.to_string().into_boxed_str())
        .collect();
    lines.sort();
    let path = format!("{}/{}_cells.tsv.gz", dir, batch_name);
    write_lines(&lines, &path)?;
    info!("wrote {} retained cells to {}", lines.len(), path);
    Ok(())
}

/// Write the retained gene ids to `{dir}/genes_kept.tsv.gz` (one id per line) —
/// the passable artifact consumed by [`load_valid_genes`]. Genes are a shared
/// vocabulary, so this is a single pooled file (not per batch).
pub fn write_qc_genes(dir: &str, gene_ids: &rustc_hash::FxHashSet<GeneId>) -> anyhow::Result<()> {
    let mut lines: Vec<Box<str>> = gene_ids
        .iter()
        .map(|g| g.to_string().into_boxed_str())
        .collect();
    lines.sort();
    let path = format!("{}/genes_kept.tsv.gz", dir);
    write_lines(&lines, &path)?;
    info!("wrote {} retained genes to {}", lines.len(), path);
    Ok(())
}

///////////////////////////
// Mitochondrial gene QC //
///////////////////////////

/// Chromosomes treated as mitochondrial unless `--mito-chr` says otherwise. The
/// one source of truth: both the clap default and [`MitoQcParams::default`] read it.
pub const MITO_CHR_DEFAULT: &str = "chrM,chrMT,MT,M";

/// Shared CLI knobs for mitochondrial QC, flattened into every subcommand that
/// does gene-count QC (`genes`, `apa`, `atoi`, `dartseq`, `all`) so the policy
/// is spelled the same way everywhere. Mirrors [`crate::cell_qc::CellQcArgs`];
/// resolve to the clap-free [`MitoQcParams`] with [`MitoQcArgs::params`].
#[derive(clap::Args, Debug, Clone, serde::Serialize)]
pub struct MitoQcArgs {
    /// Mitochondrial chromosome name(s), comma-separated
    #[arg(
        long = "mito-chr",
        default_value = MITO_CHR_DEFAULT,
        help = "Mitochondrial chromosome name(s) (comma-separated)",
        long_help = "Genes on these chromosomes are treated as mitochondrial:\n\
                     excluded from the quantified gene set (unless --keep-mito) and\n\
                     summarized in the per-cell MT-fraction QC. Matched\n\
                     case-insensitively against the GFF seqname."
    )]
    pub mito_chr: Box<str>,

    /// Keep mitochondrial genes in the quantified gene set (default: exclude)
    #[arg(
        long = "keep-mito",
        default_value_t = false,
        help = "Keep mitochondrial genes in the quantified gene set",
        long_help = "By default mitochondrial genes are dropped from what is\n\
                     quantified (their per-cell MT fraction is still reported as\n\
                     QC). Use this flag to retain them."
    )]
    pub keep_mito: bool,

    /// Max mitochondrial fraction per cell (0 = data-driven elbow cutoff)
    #[arg(
        long = "max-mito-frac",
        default_value_t = 0.0,
        help = "Max MT fraction per cell: >0 = fixed cutoff; 0 = elbow cutoff",
        long_help = "Cells whose mitochondrial UMI fraction exceeds the cutoff are\n\
                     removed during QC. A value > 0 is a fixed cutoff; the default 0\n\
                     uses a data-driven elbow cutoff on the MT% distribution (drops\n\
                     the high-MT burst tail). See --no-mito-cell-qc to disable."
    )]
    pub max_mito_frac: f64,

    /// Disable mitochondrial cell QC (report MT% only, drop no cells)
    #[arg(
        long = "no-mito-cell-qc",
        default_value_t = false,
        help = "Disable MT cell QC (report MT% only, drop no cells)",
        long_help = "Report per-cell MT% but drop no cells. Mitochondrial genes are\n\
                     still excluded from the quantified gene set unless --keep-mito."
    )]
    pub no_mito_cell_qc: bool,
}

impl Default for MitoQcArgs {
    fn default() -> Self {
        let d = MitoQcParams::default();
        Self {
            mito_chr: d.mito_chr,
            keep_mito: d.keep_mito,
            max_mito_frac: d.max_mito_frac,
            no_mito_cell_qc: d.no_mito_cell_qc,
        }
    }
}

impl MitoQcArgs {
    /// Resolve to [`MitoQcParams`].
    pub fn params(&self) -> MitoQcParams {
        MitoQcParams {
            mito_chr: self.mito_chr.clone(),
            keep_mito: self.keep_mito,
            max_mito_frac: self.max_mito_frac,
            no_mito_cell_qc: self.no_mito_cell_qc,
        }
    }
}

/// The resolved mitochondrial QC policy — the clap-free form of [`MitoQcArgs`],
/// carried in [`GeneQcRequest`] and consumed by [`qc_one_batch`].
#[derive(Debug, Clone)]
pub struct MitoQcParams {
    pub mito_chr: Box<str>,
    /// Keep MT genes in the quantified gene set (the QC metric is reported either way).
    pub keep_mito: bool,
    /// Fixed per-cell MT-fraction cutoff; 0 uses the data-driven elbow.
    pub max_mito_frac: f64,
    /// Report the MT fraction but drop no cells.
    pub no_mito_cell_qc: bool,
}

impl Default for MitoQcParams {
    fn default() -> Self {
        Self {
            mito_chr: MITO_CHR_DEFAULT.into(),
            keep_mito: false,
            max_mito_frac: 0.0,
            no_mito_cell_qc: false,
        }
    }
}

/// Which genes a run **quantifies**, resolved once against the annotation.
///
/// This is the *only* gate: a gene survives iff it is not mito-excluded AND matches
/// the selected biotype (when one is set). Keeping the mito key set next to the
/// policy that uses it means the two cannot be resolved from different
/// `--mito-chr` values, and it gives the three writers one object to consult
/// instead of three copies of the same predicate.
///
/// Cell calling never consults this — see [`qc_one_batch`] for why the QC layer
/// must stay blind to it.
pub struct GeneGate {
    mito: MitoQcParams,
    mito_keys: rustc_hash::FxHashSet<Box<str>>,
    /// Biotype subset; `None` quantifies every biotype.
    selected: Option<rustc_hash::FxHashSet<Box<str>>>,
}

impl GeneGate {
    /// Resolve the gate against the annotation, announcing both halves once.
    /// An empty `gene_type` keeps all biotypes (the modality subcommands, which
    /// expose no `--gene-type`, always pass one).
    pub fn new(records: &[GffRecord], gene_type: &str, mito: MitoQcParams) -> Self {
        let mito_keys = mito_gene_keys(records, &mito.mito_chr);
        info!(
            "{} mitochondrial gene(s) on {} ({})",
            mito_keys.len(),
            mito.mito_chr,
            if mito.keep_mito {
                "kept in matrix"
            } else {
                "excluded from matrix"
            }
        );

        let selected = if gene_type.is_empty() {
            info!("Gene biotype filter: OFF (all biotypes quantified)");
            None
        } else {
            let target: genomic_data::gff::GeneType = Box::<str>::from(gene_type).into();
            let keys: rustc_hash::FxHashSet<Box<str>> = records
                .iter()
                .filter(|r| r.gene_type == target)
                .map(format_gene_key)
                .collect();
            info!(
                "Gene biotype filter: quantifying {} '{}' genes (QC keeps all biotypes)",
                keys.len(),
                gene_type
            );
            Some(keys)
        };

        Self::from_keys(mito, mito_keys, selected)
    }

    /// Build the gate from an already-resolved MT key set, for callers that have
    /// the keys but not the annotation records.
    pub fn from_keys(
        mito: MitoQcParams,
        mito_keys: rustc_hash::FxHashSet<Box<str>>,
        selected: Option<rustc_hash::FxHashSet<Box<str>>>,
    ) -> Self {
        Self {
            mito,
            mito_keys,
            selected,
        }
    }

    /// The mitochondrial QC policy (cutoffs) this gate was built from.
    pub fn mito(&self) -> &MitoQcParams {
        &self.mito
    }

    /// The `gene_key`s on the mitochondrial chromosome(s) — the MT-fraction numerator.
    pub fn mito_keys(&self) -> &rustc_hash::FxHashSet<Box<str>> {
        &self.mito_keys
    }

    /// Whether a gene survives to quantification.
    pub fn quantify(&self, gene_key: &str) -> bool {
        (self.mito.keep_mito || !self.mito_keys.contains(gene_key))
            && self.selected.as_ref().is_none_or(|s| s.contains(gene_key))
    }
}

/// `gene_key`s whose gene sits on a mitochondrial chromosome. `mito_chr_spec`
/// is a comma-separated list of chromosome names matched case-insensitively
/// against `GffRecord::seqname` (e.g. `"chrM,MT"`).
pub fn mito_gene_keys(
    records: &[GffRecord],
    mito_chr_spec: &str,
) -> rustc_hash::FxHashSet<Box<str>> {
    let mito: rustc_hash::FxHashSet<String> = mito_chr_spec
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    records
        .iter()
        .filter(|r| mito.contains(&r.seqname.as_ref().to_ascii_lowercase()))
        .map(format_gene_key)
        .collect()
}

/// Accumulate per-cell `(mito_umi, total_umi)` over the passing cells from one
/// or more triplet sets (e.g. spliced + unspliced). `total` sums every gene;
/// `mito` sums only genes in `mito_keys`. Used for the MT-fraction QC metric.
///
/// Deliberately a serial fold: this decides a QC cutoff, and a rayon reduce would
/// sum the `f32`s in a work-stealing-dependent order, so a cell sitting exactly on
/// the threshold could fall either way between runs of the same data.
pub fn mito_cell_stats(
    triplet_sets: &[&[(CellBarcode, Box<str>, f32)]],
    passing_cells: &rustc_hash::FxHashSet<CellBarcode>,
    mito_keys: &rustc_hash::FxHashSet<Box<str>>,
) -> FxHashMap<CellBarcode, (f32, f32)> {
    // No MT genes in the annotation → every gene-key hash below is wasted work
    // (the fractions come out flat and no cell is ever dropped).
    let has_mito = !mito_keys.is_empty();
    let mut stats: FxHashMap<CellBarcode, (f32, f32)> = FxHashMap::default();
    for set in triplet_sets {
        for (cb, feat, val) in set.iter() {
            if !passing_cells.contains(cb) {
                continue;
            }
            // `get_mut` first: the barcode is an `Arc`, so `entry` would clone
            // (an atomic bump) on every triplet rather than once per cell.
            let e = match stats.get_mut(cb) {
                Some(e) => e,
                None => stats.entry(cb.clone()).or_insert((0.0, 0.0)),
            };
            e.1 += *val; // total
            if has_mito && mito_keys.contains(extract_gene_key(feat)) {
                e.0 += *val; // mito
            }
        }
    }
    stats
}

/// Write `{dir}/{batch}_mt_qc.tsv.gz`: `barcode  total_umi  mt_umi  mt_frac`,
/// one row per cell (sorted by barcode).
pub fn write_mt_qc(
    dir: &str,
    batch_name: &str,
    stats: &FxHashMap<CellBarcode, (f32, f32)>,
) -> anyhow::Result<()> {
    let mut rows: Vec<(&CellBarcode, f32, f32)> =
        stats.iter().map(|(cb, (m, t))| (cb, *m, *t)).collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    let mut lines: Vec<Box<str>> = Vec::with_capacity(rows.len() + 1);
    lines.push("barcode\ttotal_umi\tmt_umi\tmt_frac".into());
    for (cb, mt, tot) in rows {
        let frac = if tot > 0.0 { mt / tot } else { 0.0 };
        lines.push(format!("{}\t{}\t{}\t{:.6}", cb, tot, mt, frac).into());
    }
    let path = format!("{}/{}_mt_qc.tsv.gz", dir, batch_name);
    write_lines(&lines, &path)?;
    info!("wrote MT QC ({} cells) to {}", lines.len() - 1, path);
    Ok(())
}

/// Elbow threshold on an **ascending-sorted** MT-fraction distribution: the rank
/// of maximum perpendicular distance from the chord joining the first and last
/// point (the classic elbow / knee, same construction as
/// `matrix_util::archetypal::elbow_index`). Cells strictly above the returned
/// fraction are the high-MT "burst" tail.
///
/// Returns `None` when there is no usable signal — too few cells, a ~flat
/// distribution (e.g. no mitochondrial genes), or an elbow in the lower half
/// (which would cut a *majority* of cells: no clear minority burst population, so
/// don't filter). This is the "don't over-cut" guard.
pub fn mito_elbow_cutoff(sorted_fracs: &[f64]) -> Option<f64> {
    let n = sorted_fracs.len();
    if n < 50 {
        return None;
    }
    let (ymin, ymax) = (sorted_fracs[0], sorted_fracs[n - 1]);
    let span = ymax - ymin;
    if span <= 1e-9 {
        return None; // flat distribution (e.g. no mito genes)
    }
    // Normalised coords: x = rank/(n-1) ∈ [0,1], y = (frac-ymin)/span ∈ [0,1].
    // Chord runs (0,0)→(1,1), so the perpendicular distance is ∝ |x − y|.
    let xn = (n - 1) as f64;
    let (mut best_i, mut best_d) = (0usize, f64::NEG_INFINITY);
    for (i, &f) in sorted_fracs.iter().enumerate() {
        let x = i as f64 / xn;
        let y = (f - ymin) / span;
        let d = (x - y).abs();
        if d > best_d {
            best_d = d;
            best_i = i;
        }
    }
    // Over-filtering guard: the burst tail must be a minority.
    if best_i < n / 2 {
        return None;
    }
    Some(sorted_fracs[best_i])
}

/// Log the MT-fraction distribution across `passing_cells`, then drop high-MT
/// ("burst") cells. The cutoff is, in order: report-only if `disable`; the fixed
/// `max_frac` if `max_frac > 0`; otherwise a data-driven [`mito_elbow_cutoff`]
/// (the default). Returns the retained cell set. (MT genes are excluded from the
/// feature set separately by the caller.)
pub fn apply_mito_filter(
    passing_cells: rustc_hash::FxHashSet<CellBarcode>,
    stats: &FxHashMap<CellBarcode, (f32, f32)>,
    max_frac: f64,
    disable: bool,
) -> rustc_hash::FxHashSet<CellBarcode> {
    let frac_of = |cb: &CellBarcode| -> f64 {
        match stats.get(cb) {
            Some((m, t)) if *t > 0.0 => *m as f64 / *t as f64,
            _ => 0.0,
        }
    };
    let mut fracs: Vec<f64> = passing_cells.iter().map(&frac_of).collect();
    if fracs.is_empty() {
        return passing_cells;
    }
    fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = fracs[fracs.len() / 2];
    let mean = fracs.iter().sum::<f64>() / fracs.len() as f64;
    info!(
        "MT fraction over {} cells: median {:.3}, mean {:.3}, max {:.3}",
        fracs.len(),
        median,
        mean,
        fracs[fracs.len() - 1]
    );
    if disable {
        return passing_cells; // report-only
    }
    // Resolve the cutoff: explicit fixed threshold, else data-driven elbow.
    let (cutoff, kind) = if max_frac > 0.0 {
        (Some(max_frac), "fixed")
    } else {
        (mito_elbow_cutoff(&fracs), "elbow")
    };
    let Some(cutoff) = cutoff else {
        info!("MT QC: no clear high-MT burst population; no cells dropped");
        return passing_cells;
    };
    let kept: rustc_hash::FxHashSet<CellBarcode> = passing_cells
        .into_iter()
        .filter(|cb| frac_of(cb) <= cutoff)
        .collect();
    info!(
        "MT QC: dropped {} cells with MT fraction > {:.3} ({})",
        fracs.len() - kept.len(),
        cutoff,
        kind
    );
    kept
}

/// Where one batch's QC artifacts land: `{dir}/{batch}_mt_qc.tsv.gz` (the per-cell
/// MT table) and `{dir}/{batch}_cells.tsv.gz` (the passable cell set consumed by
/// `--valid-cells`). `None` runs the QC without writing anything.
pub struct QcArtifacts<'a> {
    pub dir: &'a str,
    pub batch_name: &'a str,
}

/// QC one batch end to end: gene/cell QC ([`batch_qc`]) → per-cell MT metric
/// ([`mito_cell_stats`]) → mito cell filter ([`apply_mito_filter`]) → the batch's
/// two QC artifacts.
///
/// **This is the single place a batch's passing cell set is decided.** Every path
/// that calls cells routes through it — the `faba all` loop, both `faba genes`
/// writers, and [`run_gene_count_qc`] (the shared QC behind `dartseq` / `atoi` /
/// `apa`) — so the cell sets cannot drift apart.
///
/// Two invariants are load-bearing, and the reason this is one function:
///
/// 1. **Cell calling stays mito-blind.** It sees every gene and biotype, so the
///    frozen cell set stays Cell Ranger-faithful. Only what is *quantified* gets
///    narrowed afterwards, via [`GeneGate::quantify`].
/// 2. **The MT fraction is computed on the FULL pre-filter counts** (spliced +
///    unspliced, *before* any biotype/mito **gene** gate) over the already-called
///    cells. Gate the genes first and the `mt_frac` denominator changes, which
///    silently moves both the fixed and the elbow cutoff.
///
/// Returns the batch's [`BatchQc`] with `passing_cells` already mito-filtered;
/// `gene_stats` is untouched (thresholding it — per batch or pooled — remains the
/// caller's business).
pub fn qc_one_batch(
    spliced: &[(CellBarcode, Box<str>, f32)],
    unspliced: &[(CellBarcode, Box<str>, f32)],
    gate: &GeneGate,
    gene_min_counts: usize,
    cell_min_genes: usize,
    cell_call: &crate::cell_qc::CellCallParams,
    out: Option<QcArtifacts>,
) -> anyhow::Result<BatchQc> {
    let bq = batch_qc(
        spliced,
        unspliced,
        gene_min_counts > 0,
        cell_min_genes,
        cell_call,
    );

    let mt_stats = mito_cell_stats(&[spliced, unspliced], &bq.passing_cells, gate.mito_keys());
    let mito = gate.mito();
    if let Some(ref out) = out {
        write_mt_qc(out.dir, out.batch_name, &mt_stats)?;
    }
    let passing_cells = apply_mito_filter(
        bq.passing_cells,
        &mt_stats,
        mito.max_mito_frac,
        mito.no_mito_cell_qc,
    );
    if let Some(ref out) = out {
        write_qc_cells(out.dir, out.batch_name, &passing_cells)?;
    }

    Ok(BatchQc {
        gene_stats: bq.gene_stats,
        passing_cells,
    })
}

/// Load a per-batch valid-cell set written by `faba genes` (one
/// `{batch}_cells.tsv.gz` per batch, one barcode per line) from `dir`. Missing
/// per-batch files are warned and skipped (that batch goes unfiltered).
pub fn load_valid_cells_dir(
    dir: &str,
    bam_files: &[Box<str>],
) -> anyhow::Result<rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>>> {
    // Files are named by batch (`{batch}_cells.tsv.gz`), but the in-memory map is
    // keyed by BAM file path so lookups are stable regardless of BAM ordering.
    let batch_names = uniq_batch_names(bam_files)?;
    let mut out: rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>> =
        rustc_hash::FxHashMap::default();
    for (bam_file, batch) in bam_files.iter().zip(batch_names.iter()) {
        let path = format!("{}/{}_cells.tsv.gz", dir, batch);
        if !std::path::Path::new(&path).exists() {
            log::warn!(
                "--valid-cells: no file for batch '{}' ({}); not filtered",
                batch,
                path
            );
            continue;
        }
        let cells: rustc_hash::FxHashSet<CellBarcode> = read_lines(&path)?
            .into_iter()
            .filter(|s| s.as_ref() != ".")
            .map(|s| CellBarcode::Barcode(std::sync::Arc::from(s.as_ref())))
            .collect();
        info!(
            "--valid-cells: loaded {} cells for batch '{}'",
            cells.len(),
            batch
        );
        out.insert(bam_file.clone(), cells);
    }
    Ok(out)
}

/// Load a retained-gene set written by `faba genes` (`{batch}_genes_kept.tsv.gz`,
/// one gene id per line). Genes are shared across batches, so a single file is read.
pub fn load_valid_genes(path: &str) -> anyhow::Result<rustc_hash::FxHashSet<GeneId>> {
    let genes: rustc_hash::FxHashSet<GeneId> = read_lines(path)?
        .into_iter()
        .filter(|s| s.as_ref() != ".")
        .map(|s| GeneId::Ensembl(s.as_ref().into()))
        .collect();
    info!("--valid-genes: loaded {} genes from {}", genes.len(), path);
    Ok(genes)
}

#[cfg(test)]
mod tests;
