pub use genomic_data::coordinates::{Gene, GeneAnnotations};

use log::info;
use rand::Rng;
use rand::SeedableRng;

use genomic_data::gff::GeneId;
use genomic_data::sam::Strand;

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
