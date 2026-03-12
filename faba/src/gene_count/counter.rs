//! Shared gene counting utilities
//!
//! This module provides common gene counting functionality used by
//! both run_gene_count and cell clustering.

use crate::common::*;
use genomic_data::gff::{parse_ensembl_id, GeneId, GffRecord, GffRecordMap};

use fnv::FnvHashMap;
use rust_htslib::bam::{self, record::Aux, Read};

/// Count reads per cell for a single gene
///
/// # Arguments
/// * `bam_files` - BAM files to count reads from
/// * `rec` - GFF record for the gene
/// * `cell_barcode_tag` - BAM tag for cell barcode (e.g., "CB")
/// * `gene_barcode_tag` - BAM tag for gene ID (e.g., "GX")
///
/// # Returns
/// * Vector of (CellBarcode, GeneId, count) tuples
pub fn count_reads_per_gene(
    bam_files: &[Box<str>],
    rec: &GffRecord,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, f32)>> {
    let gene_id = &rec.gene_id;

    if gene_id == &GeneId::Missing {
        return Ok(vec![]);
    }

    let mut cell_counts: FnvHashMap<CellBarcode, usize> = FnvHashMap::default();

    for bam_file in bam_files {
        count_reads_in_bam_for_gene(
            bam_file,
            rec,
            cell_barcode_tag,
            gene_barcode_tag,
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
    cell_counts: &mut FnvHashMap<CellBarcode, usize>,
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

        // Check if this read belongs to this gene
        let read_gene_id = match record.aux(gene_barcode_tag.as_bytes()) {
            Ok(Aux::String(id)) => match parse_ensembl_id(id) {
                Some(id) => GeneId::Ensembl(id.into()),
                _ => continue,
            },
            _ => continue,
        };

        if read_gene_id != rec.gene_id {
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
///
/// # Returns
/// * Vector of all (CellBarcode, GeneId, count) tuples
pub fn collect_all_gene_counts(
    bam_files: &[Box<str>],
    gff_map: &GffRecordMap,
    cell_barcode_tag: &str,
    gene_barcode_tag: &str,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, f32)>> {
    let njobs = gff_map.len() as u64;
    info!("Collecting gene counts over {} genes", njobs);

    let results: Vec<_> = gff_map
        .records()
        .par_iter()
        .progress_count(njobs)
        .map(|rec| count_reads_per_gene(bam_files, rec, cell_barcode_tag, gene_barcode_tag))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(results)
}
