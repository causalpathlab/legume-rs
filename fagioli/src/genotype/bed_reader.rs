use anyhow::Result;
use genomic_data::plink::PlinkBed;
use log::info;
use nalgebra::DMatrix;

use super::genotype_reader::{GenomicRegion, GenotypeMatrix, GenotypeReader};

/// PLINK BED format reader
pub struct BedReader {
    bed: PlinkBed,
}

impl BedReader {
    /// Create a new BED reader from file prefix
    pub fn new(bed_prefix: &str) -> Result<Self> {
        let bed = PlinkBed::new(bed_prefix)?;
        Ok(Self { bed })
    }
}

impl GenotypeReader for BedReader {
    fn read(
        &mut self,
        max_individuals: Option<usize>,
        region: Option<GenomicRegion>,
    ) -> Result<GenotypeMatrix> {
        let num_individuals_total = self.bed.iid_count();
        let total_snps = self.bed.sid_count();

        info!(
            "Total individuals: {}, SNPs: {}",
            num_individuals_total, total_snps
        );

        // Determine individuals to read
        let num_individuals = max_individuals
            .map(|max| max.min(num_individuals_total))
            .unwrap_or(num_individuals_total);

        // Filter SNPs by genomic region if specified
        let snp_indices: Option<Vec<usize>> = if let Some(ref r) = region {
            info!(
                "Filtering SNPs: chr={:?}, left={:?}, right={:?}",
                r.chromosome, r.left_bound, r.right_bound
            );

            let indices: Vec<usize> = (0..total_snps)
                .filter(|&idx| {
                    let chr = &self.bed.chromosome[idx];
                    let pos = self.bed.bp_position[idx] as u64;
                    r.contains(chr, pos)
                })
                .collect();

            info!("Selected {} SNPs within region", indices.len());

            if indices.is_empty() {
                anyhow::bail!("No SNPs found in specified genomic region");
            }

            Some(indices)
        } else {
            None
        };

        // Read genotype data — returns DMatrix<f32> directly
        info!("Reading genotype data...");
        let iid_range = if num_individuals < num_individuals_total {
            Some(0..num_individuals)
        } else {
            None
        };
        let val = self.bed.read_f32(iid_range)?;

        // Filter columns if needed
        let genotypes = if let Some(ref indices) = snp_indices {
            let n = val.nrows();
            let m = indices.len();
            DMatrix::from_fn(n, m, |i, j| val[(i, indices[j])])
        } else {
            val
        };

        // Apply mean imputation for missing values (NaN)
        let (n, m) = (genotypes.nrows(), genotypes.ncols());
        let mut genotypes = genotypes;
        info!("Applying mean imputation...");
        impute_missing_with_mean(&mut genotypes);

        // Extract metadata
        let individual_ids: Vec<Box<str>> = self
            .bed
            .iid
            .iter()
            .take(num_individuals)
            .map(|id| Box::from(id.as_str()))
            .collect();

        let (snp_ids, chromosomes, positions) = if let Some(ref indices) = snp_indices {
            let snp_ids = indices
                .iter()
                .map(|&i| Box::from(self.bed.sid[i].as_str()))
                .collect();
            let chromosomes = indices
                .iter()
                .map(|&i| Box::from(self.bed.chromosome[i].as_str()))
                .collect();
            let positions = indices
                .iter()
                .map(|&i| self.bed.bp_position[i] as u64)
                .collect();
            (snp_ids, chromosomes, positions)
        } else {
            let snp_ids = self
                .bed
                .sid
                .iter()
                .map(|id| Box::from(id.as_str()))
                .collect();
            let chromosomes = self
                .bed
                .chromosome
                .iter()
                .map(|chr| Box::from(chr.as_str()))
                .collect();
            let positions = self.bed.bp_position.iter().map(|&pos| pos as u64).collect();
            (snp_ids, chromosomes, positions)
        };

        info!("Successfully loaded: {} individuals × {} SNPs", n, m);

        Ok(GenotypeMatrix {
            genotypes,
            individual_ids,
            snp_ids,
            chromosomes,
            positions,
        })
    }

    fn num_individuals(&mut self) -> Result<usize> {
        Ok(self.bed.iid_count())
    }

    fn num_snps(&mut self) -> Result<usize> {
        Ok(self.bed.sid_count())
    }
}

/// Apply mean imputation to missing genotype values (NaN)
fn impute_missing_with_mean(genotypes: &mut DMatrix<f32>) {
    let num_snps = genotypes.ncols();
    let mut num_imputed = 0;

    for col_idx in 0..num_snps {
        let mut col = genotypes.column_mut(col_idx);

        // Calculate mean of non-missing values
        let non_missing: Vec<f32> = col.iter().copied().filter(|x| x.is_finite()).collect();

        if non_missing.is_empty() {
            // All values missing - impute with 0
            col.iter_mut().filter(|x| !x.is_finite()).for_each(|x| {
                *x = 0.0;
                num_imputed += 1;
            });
            continue;
        }

        let mean = non_missing.iter().sum::<f32>() / non_missing.len() as f32;

        // Replace NaN with mean
        col.iter_mut().filter(|x| !x.is_finite()).for_each(|x| {
            *x = mean;
            num_imputed += 1;
        });
    }

    if num_imputed > 0 {
        info!("Imputed {} missing genotype values", num_imputed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_imputation() {
        let mut mat = DMatrix::from_row_slice(3, 2, &[1.0, f32::NAN, 2.0, 1.0, f32::NAN, 2.0]);

        impute_missing_with_mean(&mut mat);

        // Column 0: mean of [1.0, 2.0] = 1.5
        assert!((mat[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((mat[(1, 0)] - 2.0).abs() < 1e-6);
        assert!((mat[(2, 0)] - 1.5).abs() < 1e-6);

        // Column 1: mean of [1.0, 2.0] = 1.5
        assert!((mat[(0, 1)] - 1.5).abs() < 1e-6);
        assert!((mat[(1, 1)] - 1.0).abs() < 1e-6);
        assert!((mat[(2, 1)] - 2.0).abs() < 1e-6);
    }
}
