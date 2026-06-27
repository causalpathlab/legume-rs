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

pub fn summarize_stats<F, V, T>(
    stats: &[(CellBarcode, BedWithGene, ConversionData)],
    feature_key_func: F,
    value_func: V,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
    V: Fn(&ConversionData) -> f32 + Send + Sync,
{
    let combined_data: HashMap<(CellBarcode, T), ConversionData> = HashMap::default();

    stats.par_iter().for_each(|(cb, k, dat)| {
        let key = (cb.clone(), feature_key_func(k));
        combined_data.entry(key).or_default().add_assign(dat);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, value_func(&v)))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}

// ========== Gene count QC ==========

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

/// In-memory QC: count nnz per gene and per cell from combined triplets,
/// return sets of passing gene_keys and cell barcodes. A `gene_min_counts`
/// of 0 disables the total-count threshold.
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
    // Collect unique (cell, gene_key) pairs using references (no per-triplet allocation)
    let cell_gene_pairs: rustc_hash::FxHashSet<(&CellBarcode, &str)> = spliced
        .par_iter()
        .chain(unspliced.par_iter())
        .map(|(cb, feat, _)| (cb, extract_gene_key(feat)))
        .collect();

    // Single pass: count nnz per gene and per cell
    let mut gene_nnz: FxHashMap<&str, usize> = FxHashMap::default();
    let mut cell_nnz: FxHashMap<&CellBarcode, usize> = FxHashMap::default();
    for &(cb, gk) in &cell_gene_pairs {
        *gene_nnz.entry(gk).or_default() += 1;
        *cell_nnz.entry(cb).or_default() += 1;
    }

    // Total counts per gene (sum over all triplets, not just unique cells)
    let gene_total: FxHashMap<&str, f64> = if gene_min_counts > 0 {
        let mut m: FxHashMap<&str, f64> = FxHashMap::default();
        for (_, feat, v) in spliced.iter().chain(unspliced.iter()) {
            *m.entry(extract_gene_key(feat)).or_default() += *v as f64;
        }
        m
    } else {
        FxHashMap::default()
    };

    let min_counts = gene_min_counts as f64;
    let passing_genes: rustc_hash::FxHashSet<Box<str>> = gene_nnz
        .into_iter()
        .filter(|(_, n)| *n >= gene_min_cells)
        .filter(|(gk, _)| {
            gene_min_counts == 0 || gene_total.get(gk).copied().unwrap_or(0.0) >= min_counts
        })
        .map(|(gk, _)| Box::from(gk))
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
            let counts = crate::cell_qc::CellCounts::from_triplets(spliced, unspliced);
            crate::cell_qc::call_cells(&counts, cell_call)
                .into_iter()
                .filter(|cb| nnz_cells.contains(cb))
                .collect()
        };

    (passing_genes, passing_cells)
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

        let (passing_genes, passing_cells) = qc_passing_keys(
            &spliced_triplets,
            &unspliced_triplets,
            gene_min_cells,
            gene_min_counts,
            cell_min_genes,
            cell_call,
        );

        info!(
            "{}: {} genes, {} cells passed QC",
            batch_name,
            passing_genes.len(),
            passing_cells.len()
        );

        // Map passing gene keys back to GeneIds (pooled across batches)
        for gene_key in &passing_genes {
            if let Some(gene_id) = gene_key_to_id.get(gene_key) {
                expressed_gene_ids.insert(gene_id.clone());
            }
        }

        // Keyed by BAM file path (stable + order-independent across passes).
        cells_by_batch.insert(bam_file.clone(), passing_cells);
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

/// Inputs for [`resolve_gene_qc`]. Each modality runner builds one from its own
/// args struct (the field names differ — e.g. apa's `gff_file` is optional and
/// m6a counts over `wt_bam_files`), then the resolution logic is shared.
pub struct GeneQcRequest<'a> {
    pub bam_files: &'a [Box<str>],
    pub cell_barcode_tag: &'a str,
    pub gene_barcode_tag: &'a str,
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
