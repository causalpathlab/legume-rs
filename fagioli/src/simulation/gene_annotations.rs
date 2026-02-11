use anyhow::Result;
use log::info;
use rand::Rng;
use rand::SeedableRng;

use genomic_data::gff::{read_gff_record_vec, FeatureType, GeneId, GeneSymbol, GffRecord};
use genomic_data::sam::Strand;

/// Gene annotation simplified for eQTL simulation
#[derive(Debug, Clone)]
pub struct Gene {
    pub gene_id: GeneId,
    pub gene_name: Option<Box<str>>,
    pub chromosome: Box<str>,
    pub tss: u64, // Transcription start site
    pub strand: Strand,
}

/// Collection of gene annotations with cis window utilities
#[derive(Debug, Clone)]
pub struct GeneAnnotations {
    pub genes: Vec<Gene>,
    pub cis_window: u64, // Default: 1,000,000 bp (1Mb)
}

impl GeneAnnotations {
    /// Get cis-regulatory window for a gene (TSS ± cis_window)
    pub fn cis_region(&self, gene_idx: usize) -> (u64, u64) {
        let gene = &self.genes[gene_idx];
        let start = gene.tss.saturating_sub(self.cis_window);
        let end = gene.tss + self.cis_window;
        (start, end)
    }

    /// Get SNP indices within cis window for a gene
    pub fn cis_snp_indices(
        &self,
        gene_idx: usize,
        snp_positions: &[u64],
        snp_chromosomes: &[Box<str>],
    ) -> Vec<usize> {
        let gene = &self.genes[gene_idx];
        let (start, end) = self.cis_region(gene_idx);

        snp_positions
            .iter()
            .enumerate()
            .filter(|(idx, &pos)| {
                snp_chromosomes[*idx] == gene.chromosome && pos >= start && pos <= end
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter genes to a specific genomic region
    pub fn filter_to_region(&self, chromosome: &str, start: u64, end: u64) -> GeneAnnotations {
        let filtered_genes: Vec<Gene> = self
            .genes
            .iter()
            .filter(|g| g.chromosome.as_ref() == chromosome && g.tss >= start && g.tss <= end)
            .cloned()
            .collect();

        GeneAnnotations {
            genes: filtered_genes,
            cis_window: self.cis_window,
        }
    }
}

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
            if rec.seqname.as_ref() != chr_filter {
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

/// Simulate gene annotations for testing (random positions in region)
pub fn simulate_gene_annotations(
    num_genes: usize,
    chromosome: &str,
    region_start: u64,
    region_end: u64,
    cis_window: u64,
    seed: u64,
) -> GeneAnnotations {
    info!(
        "Simulating {} gene annotations on {}:{}-{}",
        num_genes, chromosome, region_start, region_end
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut genes = Vec::new();

    for g in 0..num_genes {
        let tss = rng.random_range(region_start..region_end);
        let strand = if rng.random_bool(0.5) {
            Strand::Forward
        } else {
            Strand::Backward
        };

        genes.push(Gene {
            gene_id: GeneId::Ensembl(Box::from(format!("ENSG{:011}", g))),
            gene_name: Some(Box::from(format!("Gene{}", g))),
            chromosome: Box::from(chromosome),
            tss,
            strand,
        });
    }

    GeneAnnotations { genes, cis_window }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_genes() {
        let genes = simulate_gene_annotations(100, "22", 20_000_000, 30_000_000, 1_000_000, 42);

        assert_eq!(genes.genes.len(), 100);
        assert_eq!(genes.cis_window, 1_000_000);

        // Check genes are in region
        for gene in &genes.genes {
            assert_eq!(gene.chromosome.as_ref(), "22");
            assert!(gene.tss >= 20_000_000);
            assert!(gene.tss <= 30_000_000);
            // Check gene ID format
            matches!(gene.gene_id, GeneId::Ensembl(_));
        }
    }

    #[test]
    fn test_cis_region() {
        let mut genes = simulate_gene_annotations(1, "22", 25_000_000, 25_000_001, 1_000_000, 42);
        // Force TSS to exactly 25M
        genes.genes[0].tss = 25_000_000;

        let (start, end) = genes.cis_region(0);

        assert_eq!(start, 24_000_000);
        assert_eq!(end, 26_000_000);
    }
}
