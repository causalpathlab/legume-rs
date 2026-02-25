use anyhow::Result;
use log::info;

use genomic_data::gff::{read_gff_record_vec, FeatureType, GeneId, GeneSymbol, GffRecord};
use genomic_data::sam::Strand;

use crate::simulation::{Gene, GeneAnnotations};
use crate::util::chr_eq;

/// Convert GffRecord to Gene (calculate TSS from start/end based on strand)
fn gff_record_to_gene(rec: &GffRecord) -> Gene {
    let tss = match rec.strand {
        Strand::Forward => rec.start as u64,
        Strand::Backward => rec.stop as u64,
    };

    let gene_name = match &rec.gene_name {
        GeneSymbol::Symbol(s) => Some(s.clone()),
        GeneSymbol::Missing => None,
    };

    Gene {
        gene_id: rec.gene_id.clone(),
        gene_name,
        chromosome: rec.seqname.clone(),
        tss,
        strand: rec.strand,
    }
}

/// Load gene annotations from GTF/GFF file using faba's parser
pub fn load_gtf(
    gtf_path: &str,
    chromosome_filter: Option<&str>,
    start_filter: Option<u64>,
    end_filter: Option<u64>,
    cis_window: u64,
) -> Result<GeneAnnotations> {
    info!("Loading gene annotations from GTF: {}", gtf_path);

    // Use faba's parser to read GFF records
    let all_records = read_gff_record_vec(gtf_path)?;

    // Filter to "gene" features only
    let gene_records: Vec<&GffRecord> = all_records
        .iter()
        .filter(|rec| rec.feature_type == FeatureType::Gene)
        .collect();

    info!("Found {} gene records before filtering", gene_records.len());

    // Apply filters and convert to Gene structs
    let mut genes = Vec::new();
    let mut seen_genes = std::collections::HashSet::new();

    for rec in gene_records {
        // Apply chromosome filter
        if let Some(chr_filter) = chromosome_filter {
            if !chr_eq(rec.seqname.as_ref(), chr_filter) {
                continue;
            }
        }

        // Calculate TSS
        let tss = match rec.strand {
            Strand::Forward => rec.start as u64,
            Strand::Backward => rec.stop as u64,
        };

        // Apply position filters
        if let Some(start_pos) = start_filter {
            if tss < start_pos {
                continue;
            }
        }
        if let Some(end_pos) = end_filter {
            if tss > end_pos {
                continue;
            }
        }

        // Skip duplicates
        if let GeneId::Ensembl(ref id) = rec.gene_id {
            if seen_genes.contains(id) {
                continue;
            }
            seen_genes.insert(id.clone());
        }

        genes.push(gff_record_to_gene(rec));
    }

    info!("Loaded {} genes from GTF (after filtering)", genes.len());

    Ok(GeneAnnotations { genes, cis_window })
}

/// Load gene annotations from a BED file.
///
/// Expected columns: `chr start end gene_id [gene_name [strand]]`.
/// TSS is inferred from start/end based on strand (default: Forward -> start).
pub fn load_bed_annotations(
    bed_path: &str,
    chromosome_filter: Option<&str>,
    cis_window: u64,
) -> Result<GeneAnnotations> {
    info!("Loading gene annotations from BED: {}", bed_path);

    let parsed =
        matrix_util::common_io::read_lines_of_words_delim(bed_path, &['\t', ',', ' '], -1)?;

    let mut genes = Vec::new();
    let mut seen_genes = std::collections::HashSet::new();

    for words in &parsed.lines {
        if words.len() < 4 {
            continue;
        }

        let chr = &words[0];

        // Apply chromosome filter
        if let Some(chr_filter) = chromosome_filter {
            if !chr_eq(chr.as_ref(), chr_filter) {
                continue;
            }
        }

        let start: u64 = words[1].parse().unwrap_or(0);
        let end: u64 = words[2].parse().unwrap_or(0);
        let gene_id_str = &words[3];

        // Skip duplicates
        if seen_genes.contains(gene_id_str) {
            continue;
        }
        seen_genes.insert(gene_id_str.clone());

        let gene_name = words.get(4).cloned();

        let strand = if let Some(s) = words.get(5) {
            match s.as_ref() {
                "-" => Strand::Backward,
                _ => Strand::Forward,
            }
        } else {
            Strand::Forward
        };

        let tss = match strand {
            Strand::Forward => start,
            Strand::Backward => end,
        };

        genes.push(Gene {
            gene_id: GeneId::Ensembl(gene_id_str.clone()),
            gene_name,
            chromosome: chr.clone(),
            tss,
            strand,
        });
    }

    info!("Loaded {} genes from BED (after filtering)", genes.len());

    Ok(GeneAnnotations { genes, cis_window })
}
