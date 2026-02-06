use anyhow::{Context, Result};
use bed_reader::{Bed, ReadOptions};
use log::info;
use nalgebra::DMatrix;

use super::genotype_reader::{GenotypeMatrix, GenotypeReader, GenomicRegion};

/// PLINK BED format reader
pub struct BedReader {
    bed: Bed,
}

impl BedReader {
    /// Create a new BED reader from file prefix
    pub fn new(bed_prefix: &str) -> Result<Self> {
        let bed_path = format!("{}.bed", bed_prefix);
        let bed = Bed::new(&bed_path)
            .context(format!("Failed to open BED file: {}", bed_path))?;

        Ok(Self { bed })
    }
}

impl GenotypeReader for BedReader {
    fn read(
        &mut self,
        max_individuals: Option<usize>,
        region: Option<GenomicRegion>,
    ) -> Result<GenotypeMatrix> {
        // Get all metadata upfront to avoid multiple mutable borrows
        let iid = self.bed.iid()?.to_vec();
        let sid = self.bed.sid()?.to_vec();
        let chromosome = self.bed.chromosome()?.to_vec();
        let bp_position = self.bed.bp_position()?.to_vec();

        let num_individuals_total = iid.len();
        let total_snps = sid.len();

        info!("Total individuals: {}, SNPs: {}", num_individuals_total, total_snps);

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
                    let chr = &chromosome[idx];
                    let pos = bp_position[idx] as u64;
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

        // Build read options and read data
        info!("Reading genotype data...");
        let val = if num_individuals < num_individuals_total {
            ReadOptions::builder()
                .iid_index(0..num_individuals)
                .f32()
                .read(&mut self.bed)?
        } else {
            ReadOptions::builder()
                .f32()
                .read(&mut self.bed)?
        };

        // Filter columns if needed
        let val_filtered = if let Some(ref indices) = snp_indices {
            // Extract selected columns
            let (n, _) = val.dim();
            let m = indices.len();
            let mut filtered = ndarray::Array2::zeros((n, m));
            for (j_new, &j_old) in indices.iter().enumerate() {
                for i in 0..n {
                    filtered[[i, j_new]] = val[[i, j_old]];
                }
            }
            filtered
        } else {
            val
        };

        // Convert ndarray to DMatrix
        let (n, m) = val_filtered.dim();
        let mut genotypes = DMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                genotypes[(i, j)] = val_filtered[[i, j]];
            }
        }

        // Apply mean imputation for missing values (NaN)
        info!("Applying mean imputation...");
        impute_missing_with_mean(&mut genotypes);

        // Extract metadata
        let individual_ids: Vec<Box<str>> = iid
            .iter()
            .take(num_individuals)
            .map(|id| Box::from(id.as_str()))
            .collect();

        let (snp_ids, chromosomes, positions) = if let Some(ref indices) = snp_indices {
            // Filtered SNPs
            let snp_ids = indices.iter().map(|&i| Box::from(sid[i].as_str())).collect();
            let chromosomes = indices.iter().map(|&i| Box::from(chromosome[i].as_str())).collect();
            let positions = indices.iter().map(|&i| bp_position[i] as u64).collect();
            (snp_ids, chromosomes, positions)
        } else {
            // All SNPs
            let snp_ids = sid.iter().map(|id| Box::from(id.as_str())).collect();
            let chromosomes = chromosome.iter().map(|chr| Box::from(chr.as_str())).collect();
            let positions = bp_position.iter().map(|&pos| pos as u64).collect();
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
        Ok(self.bed.iid()?.len())
    }

    fn num_snps(&mut self) -> Result<usize> {
        Ok(self.bed.sid()?.len())
    }
}

/// Apply mean imputation to missing genotype values (NaN)
fn impute_missing_with_mean(genotypes: &mut DMatrix<f32>) {
    let num_snps = genotypes.ncols();
    let mut num_imputed = 0;

    for col_idx in 0..num_snps {
        let mut col = genotypes.column_mut(col_idx);

        // Calculate mean of non-missing values
        let non_missing: Vec<f32> = col.iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        if non_missing.is_empty() {
            // All values missing - impute with 0
            col.iter_mut()
                .filter(|x| !x.is_finite())
                .for_each(|x| {
                    *x = 0.0;
                    num_imputed += 1;
                });
            continue;
        }

        let mean = non_missing.iter().sum::<f32>() / non_missing.len() as f32;

        // Replace NaN with mean
        col.iter_mut()
            .filter(|x| !x.is_finite())
            .for_each(|x| {
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
        let mut mat = DMatrix::from_row_slice(3, 2, &[
            1.0, f32::NAN,
            2.0, 1.0,
            f32::NAN, 2.0,
        ]);

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
