//! Shared gene counting utilities
//!
//! This module provides common gene counting functionality used by
//! both run_gene_count and cell clustering.

use crate::common::*;
use crate::data::bam_io;
use genomic_data::gff::{GeneId, GffRecord, GffRecordMap};

use rust_htslib::bam::{self, record::Aux, Read};
use rustc_hash::FxHashMap;

/// Count reads per cell for a single gene
///
/// # Arguments
/// * `bam_files` - BAM files to count reads from
/// * `rec` - GFF record for the gene
/// * `cell_barcode_tag` - BAM tag for cell barcode (e.g., "CB")
/// * `gene_barcode_tag` - BAM tag for gene ID (e.g., "GX")
/// * `min_mapq` - Minimum mapping quality (default: 20)
///
/// # Returns
/// * Vector of (CellBarcode, GeneId, count) tuples
pub fn count_reads_per_gene(
    bam_files: &[Box<str>],
    rec: &GffRecord,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
    min_mapq: u8,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, f32)>> {
    let gene_id = &rec.gene_id;

    if gene_id == &GeneId::Missing {
        return Ok(vec![]);
    }

    let mut cell_counts: FxHashMap<CellBarcode, usize> = FxHashMap::default();

    for bam_file in bam_files {
        count_reads_in_bam_for_gene(
            bam_file,
            rec,
            cell_barcode_tag,
            gene_barcode_tag,
            min_mapq,
            &mut cell_counts,
        )?;
    }

    Ok(cell_counts
        .into_iter()
        .map(|(cb, count)| (cb, gene_id.clone(), count as f32))
        .collect())
}

/// Count reads in a single BAM file for a specific gene
fn count_reads_in_bam_for_gene(
    bam_file: &str,
    rec: &GffRecord,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
    min_mapq: u8,
    cell_counts: &mut FxHashMap<CellBarcode, usize>,
) -> anyhow::Result<()> {
    let mut reader = bam::IndexedReader::from_path(bam_file)?;
    let chr = rec.seqname.as_ref();
    let start = (rec.start - 1).max(0);
    let stop = rec.stop;

    // Fetch reads in the gene region
    let tid = reader.header().tid(chr.as_bytes());
    if tid.is_none() {
        return Ok(()); // Chromosome not in BAM
    }

    reader.fetch((tid.unwrap(), start, stop))?;

    for result in reader.records() {
        let record = result?;

        // Quality filters (consistent with ATOI/APA/DART)

        // 1. Mapping quality
        if record.mapq() < min_mapq {
            continue;
        }

        // 2. Skip PCR duplicates
        if record.is_duplicate() {
            continue;
        }

        // 3. Skip secondary/supplementary alignments (multi-mappers)
        if record.is_secondary() || record.is_supplementary() {
            continue;
        }

        // 4. For paired-end reads, skip if not properly paired
        if record.is_paired() && !record.is_proper_pair() {
            continue;
        }

        // Check if this read belongs to this gene (str comparison, no allocation)
        if !bam_io::matches_gene(&record, gene_barcode_tag, &rec.gene_id, false) {
            continue;
        }

        // Get cell barcode
        let cell_barcode = match record.aux(cell_barcode_tag.as_bytes()) {
            Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
            _ => CellBarcode::Missing,
        };

        *cell_counts.entry(cell_barcode).or_default() += 1;
    }

    Ok(())
}

/// Collect gene counts from all genes in a GFF map
///
/// # Arguments
/// * `bam_files` - BAM files to count reads from
/// * `gff_map` - Gene annotation map
/// * `cell_barcode_tag` - BAM tag for cell barcode
/// * `gene_barcode_tag` - BAM tag for gene ID
/// * `min_mapq` - Minimum mapping quality (default: 20)
///
/// # Returns
/// * Vector of all (CellBarcode, GeneId, count) tuples
pub fn collect_all_gene_counts(
    bam_files: &[Box<str>],
    gff_map: &GffRecordMap,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
    min_mapq: u8,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, f32)>> {
    let njobs = gff_map.len() as u64;
    info!(
        "Collecting gene counts over {} genes (MAPQ ≥ {})",
        njobs, min_mapq
    );

    let results: Vec<_> = gff_map
        .records()
        .par_iter()
        .progress_count(njobs)
        .map(|rec| {
            count_reads_per_gene(bam_files, rec, cell_barcode_tag, gene_barcode_tag, min_mapq)
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(results)
}
