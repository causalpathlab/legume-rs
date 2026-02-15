use anyhow::Result;
use nalgebra::DMatrix;

/// Genotype matrix with metadata
#[derive(Debug, Clone)]
pub struct GenotypeMatrix {
    pub genotypes: DMatrix<f32>, // N individuals x M SNPs (values: 0, 1, 2)
    pub individual_ids: Vec<Box<str>>,
    pub snp_ids: Vec<Box<str>>,
    pub chromosomes: Vec<Box<str>>,
    pub positions: Vec<u64>,
    pub allele1: Vec<Box<str>>,
    pub allele2: Vec<Box<str>>,
}

impl GenotypeMatrix {
    pub fn num_individuals(&self) -> usize {
        self.genotypes.nrows()
    }

    pub fn num_snps(&self) -> usize {
        self.genotypes.ncols()
    }
}

/// Genomic region filter for efficient SNP loading
#[derive(Debug, Clone)]
pub struct GenomicRegion {
    pub chromosome: Option<String>,
    pub left_bound: Option<u64>,
    pub right_bound: Option<u64>,
}

impl GenomicRegion {
    pub fn new(
        chromosome: Option<String>,
        left_bound: Option<u64>,
        right_bound: Option<u64>,
    ) -> Self {
        Self {
            chromosome,
            left_bound,
            right_bound,
        }
    }

    /// Check if a SNP is within this region
    pub fn contains(&self, chr: &str, pos: u64) -> bool {
        // Check chromosome
        if let Some(ref target_chr) = self.chromosome {
            if chr != target_chr {
                return false;
            }
        }

        // Check left bound
        if let Some(left) = self.left_bound {
            if pos < left {
                return false;
            }
        }

        // Check right bound
        if let Some(right) = self.right_bound {
            if pos > right {
                return false;
            }
        }

        true
    }
}

/// Trait for reading genotype data from various formats
///
/// Implementations exist for:
/// - PLINK BED/BIM/FAM format
/// - VCF format (future)
/// - BGEN format (future)
pub trait GenotypeReader {
    /// Read genotype matrix with optional filters
    fn read(
        &mut self,
        max_individuals: Option<usize>,
        region: Option<GenomicRegion>,
    ) -> Result<GenotypeMatrix>;

    /// Get total number of individuals (before filtering)
    fn num_individuals(&mut self) -> Result<usize>;

    /// Get total number of SNPs (before filtering)
    fn num_snps(&mut self) -> Result<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genomic_region_contains() {
        let region = GenomicRegion::new(Some("chr1".to_string()), Some(1000), Some(2000));

        assert!(region.contains("chr1", 1500));
        assert!(!region.contains("chr2", 1500));
        assert!(!region.contains("chr1", 500));
        assert!(!region.contains("chr1", 2500));
    }

    #[test]
    fn test_genomic_region_chr_only() {
        let region = GenomicRegion::new(Some("chr1".to_string()), None, None);

        assert!(region.contains("chr1", 100));
        assert!(region.contains("chr1", 999999));
        assert!(!region.contains("chr2", 100));
    }

    #[test]
    fn test_genomic_region_position_only() {
        let region = GenomicRegion::new(None, Some(1000), Some(2000));

        assert!(region.contains("chr1", 1500));
        assert!(region.contains("chr2", 1500));
        assert!(!region.contains("chr1", 500));
    }
}
