use crate::common::*;
use crate::data::conversion::*;
use crate::data::util_htslib::*;
use crate::gene_count::splice::{count_read_per_gene_splice, format_gene_key};

use dashmap::DashMap as HashMap;
use fnv::FnvHashMap;
use genomic_data::gff::GffRecordMap;
use genomic_data::sam::CellBarcode;

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

pub struct GeneCountQc {
    pub gene_ids: fnv::FnvHashSet<GeneId>,
    pub cell_barcodes: fnv::FnvHashSet<CellBarcode>,
}

/// Extract gene_key from a feature name like `"GENE_SYM/count/spliced"` → `"GENE_SYM"`.
#[inline]
pub fn extract_gene_key(feat: &str) -> &str {
    feat.rfind("/count/")
        .map(|pos| &feat[..pos])
        .unwrap_or(feat)
}

/// In-memory QC: count nnz per gene and per cell from combined triplets,
/// return sets of passing gene_keys and cell barcodes.
pub fn qc_passing_keys(
    spliced: &[(CellBarcode, Box<str>, f32)],
    unspliced: &[(CellBarcode, Box<str>, f32)],
    gene_min_cells: usize,
    cell_min_genes: usize,
) -> (fnv::FnvHashSet<Box<str>>, fnv::FnvHashSet<CellBarcode>) {
    // Collect unique (cell, gene_key) pairs with nonzero total
    let cell_gene_pairs: fnv::FnvHashSet<(CellBarcode, Box<str>)> = spliced
        .par_iter()
        .chain(unspliced.par_iter())
        .map(|(cb, feat, _)| {
            let gk: Box<str> = extract_gene_key(feat).into();
            (cb.clone(), gk)
        })
        .collect();

    // Count nnz per gene (how many distinct cells)
    let gene_nnz: FnvHashMap<Box<str>, usize> =
        cell_gene_pairs
            .iter()
            .fold(FnvHashMap::default(), |mut acc, (_, gk)| {
                *acc.entry(gk.clone()).or_default() += 1;
                acc
            });

    // Count nnz per cell (how many distinct genes)
    let cell_nnz: FnvHashMap<CellBarcode, usize> =
        cell_gene_pairs
            .iter()
            .fold(FnvHashMap::default(), |mut acc, (cb, _)| {
                *acc.entry(cb.clone()).or_default() += 1;
                acc
            });

    let passing_genes: fnv::FnvHashSet<Box<str>> = gene_nnz
        .into_iter()
        .filter(|(_, n)| *n >= gene_min_cells)
        .map(|(gk, _)| gk)
        .collect();

    let passing_cells: fnv::FnvHashSet<CellBarcode> = cell_nnz
        .into_iter()
        .filter(|(_, n)| *n >= cell_min_genes)
        .map(|(cb, _)| cb)
        .collect();

    (passing_genes, passing_cells)
}

/// Run in-memory gene count QC: count reads per gene (splice-aware),
/// filter genes by min_cells and cells by min_genes.
/// Returns passing gene IDs and cell barcodes (no disk output).
pub fn run_gene_count_qc(
    gff_file: &str,
    bam_files: &[Box<str>],
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
    gene_min_cells: usize,
    cell_min_genes: usize,
) -> anyhow::Result<GeneCountQc> {
    info!("=== Gene expression QC ===");

    let all_records = read_gff_record_vec(gff_file)?;
    let gene_map = build_gene_map(&all_records, Some(&FeatureType::Gene))?;
    let exon_map = build_exon_intervals(&all_records);
    let exon_intervals: FnvHashMap<GeneId, Vec<(i64, i64)>> = exon_map.into_iter().collect();

    let gff_map = GffRecordMap::from_map(gene_map);
    info!("Loaded {} genes for expression QC", gff_map.len());

    let records = gff_map.records();

    // Build gene_key → GeneId mapping for reverse lookup after QC
    let gene_key_to_id: FnvHashMap<Box<str>, GeneId> = records
        .iter()
        .map(|rec| (format_gene_key(rec), rec.gene_id.clone()))
        .collect();

    let mut expressed_gene_ids: fnv::FnvHashSet<GeneId> = fnv::FnvHashSet::default();
    let mut valid_cell_barcodes: fnv::FnvHashSet<CellBarcode> = fnv::FnvHashSet::default();

    for bam_file in bam_files {
        let njobs = records.len() as u64;
        info!(
            "Counting genes (splice-aware) in {} ({} genes)",
            bam_file, njobs
        );

        let results: Vec<_> = records
            .par_iter()
            .progress_count(njobs)
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
            cell_min_genes,
        );

        info!(
            "{} genes, {} cells passed QC",
            passing_genes.len(),
            passing_cells.len()
        );

        // Map passing gene keys back to GeneIds
        for gene_key in &passing_genes {
            if let Some(gene_id) = gene_key_to_id.get(gene_key) {
                expressed_gene_ids.insert(gene_id.clone());
            }
        }

        valid_cell_barcodes.extend(passing_cells);
    }

    info!(
        "Gene QC summary: {} genes, {} cells passed across {} BAM files",
        expressed_gene_ids.len(),
        valid_cell_barcodes.len(),
        bam_files.len()
    );

    Ok(GeneCountQc {
        gene_ids: expressed_gene_ids,
        cell_barcodes: valid_cell_barcodes,
    })
}
