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

/// Push a channel-last row `{unit}/{modality}/{channel}` = `count` into
/// `triplets`, skipping zeros to keep the matrix sparse. The single place the
/// channel-count producers (editing, APA) spell the [`feature_row`] convention.
///
/// [`feature_row`]: faba::feature_name::feature_row
pub fn push_channel_row(
    triplets: &mut Vec<(CellBarcode, Box<str>, f32)>,
    cb: &CellBarcode,
    unit: &str,
    modality: &str,
    channel: &str,
    count: usize,
) {
    if count > 0 {
        triplets.push((
            cb.clone(),
            faba::feature_name::feature_row(unit, modality, channel, None),
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
        push_channel_row(&mut triplets, &cb, &gene, modality, pos_channel, dat.converted);
        push_channel_row(&mut triplets, &cb, &gene, modality, neg_channel, dat.unconverted);
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
/// serves a single batch ([`qc_passing_keys`]) or stats pooled across batches.
/// A `gene_min_counts` of 0 disables the total-count threshold.
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

/// Per-batch in-memory QC: thresholds one batch's gene stats and returns its
/// cell call. Used by the `faba genes` writers, which filter each batch's matrix
/// against its own passing set. The pooled-vocabulary paths instead accumulate
/// [`batch_qc`] stats across batches and call [`passing_genes_from_stats`] once.
pub fn qc_passing_keys(
    spliced: &[(CellBarcode, Box<str>, f32)],
    unspliced: &[(CellBarcode, Box<str>, f32)],
    gene_min_cells: usize,
    gene_min_counts: usize,
    cell_min_genes: usize,
    cell_call: &crate::cell_qc::CellCallParams,
) -> (
    rustc_hash::FxHashSet<Box<str>>,
    rustc_hash::FxHashSet<CellBarcode>,
) {
    let bq = batch_qc(
        spliced,
        unspliced,
        gene_min_counts > 0,
        cell_min_genes,
        cell_call,
    );
    let passing_genes = passing_genes_from_stats(&bq.gene_stats, gene_min_cells, gene_min_counts);
    (passing_genes, bq.passing_cells)
}

/// Run in-memory gene count QC: count reads per gene (splice-aware),
/// filter genes by min_cells and cells by min_genes.
/// Returns passing gene IDs and cell barcodes (no disk output).
#[allow(clippy::too_many_arguments)]
pub fn run_gene_count_qc(
    gff_file: &str,
    bam_files: &[Box<str>],
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
    umi_tag: Option<&[u8]>,
    gene_min_cells: usize,
    gene_min_counts: usize,
    cell_min_genes: usize,
    cell_call: &crate::cell_qc::CellCallParams,
) -> anyhow::Result<GeneCountQc> {
    info!("=== Gene expression QC ===");

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

    let mut expressed_gene_ids: rustc_hash::FxHashSet<GeneId> = rustc_hash::FxHashSet::default();
    let mut cells_by_batch: rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>> =
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
                    cell_barcode_tag,
                    gene_barcode_tag,
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
            "{} spliced, {} unspliced triplets",
            spliced_triplets.len(),
            unspliced_triplets.len()
        );

        let bq = batch_qc(
            &spliced_triplets,
            &unspliced_triplets,
            gene_min_counts > 0,
            cell_min_genes,
            cell_call,
        );

        info!("{}: {} cells passed QC", batch_name, bq.passing_cells.len());

        accumulate_gene_stats(&mut pooled_gene_stats, bq.gene_stats);

        // Keyed by BAM file path (stable + order-independent across passes).
        cells_by_batch.insert(bam_file.clone(), bq.passing_cells);
    }

    // Threshold the pooled gene stats once → shared gene vocabulary.
    let passing_genes =
        passing_genes_from_stats(&pooled_gene_stats, gene_min_cells, gene_min_counts);
    for gene_key in &passing_genes {
        if let Some(gene_id) = gene_key_to_id.get(gene_key) {
            expressed_gene_ids.insert(gene_id.clone());
        }
    }

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
    pub gene_min_cells: usize,
    pub gene_min_counts: usize,
    pub cell_min_genes: usize,
    pub cell_call: crate::cell_qc::CellCallParams,
    pub valid_cells_file: Option<&'a str>,
    pub valid_genes_file: Option<&'a str>,
    pub skip_gene_qc: bool,
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
        }))
    } else if !req.skip_gene_qc {
        match req.gff_file {
            Some(gff_file) => Ok(Some(run_gene_count_qc(
                gff_file,
                req.bam_files,
                req.cell_barcode_tag,
                req.gene_barcode_tag,
                req.umi_tag,
                req.gene_min_cells,
                req.gene_min_counts,
                req.cell_min_genes,
                &req.cell_call,
            )?)),
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
pub fn mito_cell_stats(
    triplet_sets: &[&[(CellBarcode, Box<str>, f32)]],
    passing_cells: &rustc_hash::FxHashSet<CellBarcode>,
    mito_keys: &rustc_hash::FxHashSet<Box<str>>,
) -> FxHashMap<CellBarcode, (f32, f32)> {
    let mut stats: FxHashMap<CellBarcode, (f32, f32)> = FxHashMap::default();
    for set in triplet_sets {
        for (cb, feat, val) in set.iter() {
            if !passing_cells.contains(cb) {
                continue;
            }
            let e = stats.entry(cb.clone()).or_insert((0.0, 0.0));
            e.1 += *val; // total
            if mito_keys.contains(extract_gene_key(feat)) {
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
