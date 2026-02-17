use anyhow::Result;
use log::info;
use matrix_util::common_io::open_buf_writer;
use nalgebra::DMatrix;
use std::io::Write;

use super::ld_block::LdBlock;
use super::ld_score::LdScoreRecord;
use super::marginal_ols::SumstatRecord;
use crate::simulation::CellTypeGeneticEffects;

/// Streaming writer for summary statistics
pub struct SumstatWriter {
    writer: Box<dyn Write>,
    snp_ids: Vec<Box<str>>,
    chromosomes: Vec<Box<str>>,
    positions: Vec<u64>,
    num_individuals: usize,
}

impl SumstatWriter {
    pub fn new(
        path: &str,
        snp_ids: &[Box<str>],
        chromosomes: &[Box<str>],
        positions: &[u64],
        num_individuals: usize,
    ) -> Result<Self> {
        let mut writer = open_buf_writer(path)?;
        writeln!(
            writer,
            "trait_idx\tsnp_idx\tsnp_id\tchr\tpos\tn\tbeta\tse\tz\tpvalue"
        )?;

        Ok(Self {
            writer,
            snp_ids: snp_ids.to_vec(),
            chromosomes: chromosomes.to_vec(),
            positions: positions.to_vec(),
            num_individuals,
        })
    }

    pub fn write_block(&mut self, records: &[SumstatRecord]) -> Result<()> {
        for rec in records {
            writeln!(
                self.writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}",
                rec.trait_idx,
                rec.snp_idx,
                self.snp_ids[rec.snp_idx],
                self.chromosomes[rec.snp_idx],
                self.positions[rec.snp_idx],
                self.num_individuals,
                rec.beta,
                rec.se,
                rec.z,
                rec.pvalue,
            )?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Streaming writer for LD scores
pub struct LdScoreWriter {
    writer: Box<dyn Write>,
    snp_ids: Vec<Box<str>>,
    chromosomes: Vec<Box<str>>,
    positions: Vec<u64>,
}

impl LdScoreWriter {
    pub fn new(
        path: &str,
        snp_ids: &[Box<str>],
        chromosomes: &[Box<str>],
        positions: &[u64],
    ) -> Result<Self> {
        let mut writer = open_buf_writer(path)?;
        writeln!(writer, "snp_idx\tsnp_id\tchr\tpos\tl2\tnum_snps")?;

        Ok(Self {
            writer,
            snp_ids: snp_ids.to_vec(),
            chromosomes: chromosomes.to_vec(),
            positions: positions.to_vec(),
        })
    }

    pub fn write_block(&mut self, records: &[LdScoreRecord]) -> Result<()> {
        for rec in records {
            writeln!(
                self.writer,
                "{}\t{}\t{}\t{}\t{:.4}\t{}",
                rec.snp_idx,
                self.snp_ids[rec.snp_idx],
                self.chromosomes[rec.snp_idx],
                self.positions[rec.snp_idx],
                rec.l2,
                rec.num_snps_in_block,
            )?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Write LD block definitions
pub fn write_ld_blocks(path: &str, blocks: &[LdBlock]) -> Result<()> {
    let mut writer = open_buf_writer(path)?;
    writeln!(writer, "block_idx\tchr\tbp_start\tbp_end\tnum_snps")?;

    for block in blocks {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            block.block_idx,
            block.chr,
            block.bp_start,
            block.bp_end,
            block.num_snps(),
        )?;
    }

    writer.flush()?;
    info!("Wrote {} LD blocks to {}", blocks.len(), path);
    Ok(())
}

/// Write ground truth causal effects
pub fn write_ground_truth(
    path: &str,
    block_effects: &[(usize, CellTypeGeneticEffects)],
    blocks: &[LdBlock],
    snp_ids: &[Box<str>],
    chromosomes: &[Box<str>],
    positions: &[u64],
) -> Result<()> {
    let mut writer = open_buf_writer(path)?;
    writeln!(
        writer,
        "block_idx\teffect_type\ttrait_idx\tsnp_idx\tsnp_id\tchr\tpos\teffect_size"
    )?;

    for (block_idx, effects) in block_effects {
        let block = &blocks[*block_idx];
        let offset = block.snp_start;

        // Shared effects
        for (j, &local_idx) in effects.shared_causal_indices.iter().enumerate() {
            let global_idx = offset + local_idx;
            for trait_idx in 0..effects.num_cell_types {
                let effect = effects.shared_effect_sizes[(trait_idx, j)];
                if effect.abs() > 1e-10 {
                    writeln!(
                        writer,
                        "{}\tshared\t{}\t{}\t{}\t{}\t{}\t{:.6}",
                        block_idx,
                        trait_idx,
                        global_idx,
                        snp_ids[global_idx],
                        chromosomes[global_idx],
                        positions[global_idx],
                        effect,
                    )?;
                }
            }
        }

        // Independent effects
        for (trait_idx, trait_indices) in effects.independent_causal_indices.iter().enumerate() {
            for (j, &local_idx) in trait_indices.iter().enumerate() {
                let global_idx = offset + local_idx;
                let effect = effects.independent_effect_sizes[(trait_idx, j)];
                if effect.abs() > 1e-10 {
                    writeln!(
                        writer,
                        "{}\tindependent\t{}\t{}\t{}\t{}\t{}\t{:.6}",
                        block_idx,
                        trait_idx,
                        global_idx,
                        snp_ids[global_idx],
                        chromosomes[global_idx],
                        positions[global_idx],
                        effect,
                    )?;
                }
            }
        }
    }

    writer.flush()?;
    info!("Wrote ground truth to {}", path);
    Ok(())
}

/// Write confounder matrix
pub fn write_confounders(path: &str, confounders: &DMatrix<f32>) -> Result<()> {
    if confounders.ncols() == 0 {
        return Ok(());
    }

    let mut writer = open_buf_writer(path)?;

    // Header
    let header: Vec<String> = (0..confounders.ncols())
        .map(|j| format!("confounder_{}", j))
        .collect();
    writeln!(writer, "{}", header.join("\t"))?;

    // Data rows
    for i in 0..confounders.nrows() {
        let row: Vec<String> = (0..confounders.ncols())
            .map(|j| format!("{:.6}", confounders[(i, j)]))
            .collect();
        writeln!(writer, "{}", row.join("\t"))?;
    }

    writer.flush()?;
    info!(
        "Wrote confounder matrix ({} x {}) to {}",
        confounders.nrows(),
        confounders.ncols(),
        path
    );
    Ok(())
}
