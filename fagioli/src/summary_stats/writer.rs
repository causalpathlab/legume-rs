use anyhow::Result;
use log::info;
use matrix_util::common_io::open_buf_writer;
use nalgebra::DMatrix;
use rust_htslib::bgzf;
use rust_htslib::tpool::ThreadPool;
use std::io::Write;

use super::ld_block::LdBlock;
use super::ld_score::LdScoreRecord;
use super::marginal_ols::SumstatRecord;
use crate::simulation::CellTypeGeneticEffects;

/// Open a BGZF writer with htslib thread pool for parallel compression.
fn open_bgzf_writer(path: &str, tpool: Option<&ThreadPool>) -> Result<bgzf::Writer> {
    let mut writer = bgzf::Writer::from_path(path)?;
    if let Some(tp) = tpool {
        writer.set_thread_pool(tp)?;
    }
    Ok(writer)
}

/// Streaming writer for summary statistics (BGZF-compressed, tabix-compatible)
pub struct SumstatWriter {
    writer: bgzf::Writer,
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
        tpool: Option<&ThreadPool>,
    ) -> Result<Self> {
        let mut writer = open_bgzf_writer(path, tpool)?;
        writeln!(
            writer,
            "#chr\tstart\tend\tsnp_id\ttrait_idx\tn\tbeta\tse\tz\tpvalue"
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
            let pos = self.positions[rec.snp_idx];
            writeln!(
                self.writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}",
                self.chromosomes[rec.snp_idx],
                pos.saturating_sub(1),
                pos,
                self.snp_ids[rec.snp_idx],
                rec.trait_idx,
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

/// Streaming writer for LD scores (BGZF-compressed, tabix-compatible)
pub struct LdScoreWriter {
    writer: bgzf::Writer,
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
        tpool: Option<&ThreadPool>,
    ) -> Result<Self> {
        let mut writer = open_bgzf_writer(path, tpool)?;
        writeln!(writer, "#chr\tstart\tend\tsnp_id\tl2\tnum_snps")?;

        Ok(Self {
            writer,
            snp_ids: snp_ids.to_vec(),
            chromosomes: chromosomes.to_vec(),
            positions: positions.to_vec(),
        })
    }

    pub fn write_block(&mut self, records: &[LdScoreRecord]) -> Result<()> {
        for rec in records {
            let pos = self.positions[rec.snp_idx];
            writeln!(
                self.writer,
                "{}\t{}\t{}\t{}\t{:.4}\t{}",
                self.chromosomes[rec.snp_idx],
                pos.saturating_sub(1),
                pos,
                self.snp_ids[rec.snp_idx],
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

/// Write LD block definitions (BGZF-compressed, tabix-compatible)
pub fn write_ld_blocks(path: &str, blocks: &[LdBlock], tpool: Option<&ThreadPool>) -> Result<()> {
    let mut writer = open_bgzf_writer(path, tpool)?;
    writeln!(writer, "#chr\tstart\tend\tblock_idx\tnum_snps")?;

    for block in blocks {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            block.chr,
            block.bp_start.saturating_sub(1),
            block.bp_end,
            block.block_idx,
            block.num_snps(),
        )?;
    }

    writer.flush()?;
    info!("Wrote {} LD blocks to {}", blocks.len(), path);
    Ok(())
}

/// Write ground truth causal effects (BGZF-compressed, tabix-compatible)
pub fn write_ground_truth(
    path: &str,
    block_effects: &[(usize, CellTypeGeneticEffects)],
    blocks: &[LdBlock],
    snp_ids: &[Box<str>],
    chromosomes: &[Box<str>],
    positions: &[u64],
    tpool: Option<&ThreadPool>,
) -> Result<()> {
    // Collect all records, then sort by position for tabix compatibility
    struct GtRecord {
        global_idx: usize,
        block_idx: usize,
        effect_type: &'static str,
        trait_idx: usize,
        effect: f32,
    }

    let mut records: Vec<GtRecord> = Vec::new();

    for (block_idx, effects) in block_effects {
        let block = &blocks[*block_idx];
        let offset = block.snp_start;

        for (j, &local_idx) in effects.shared_causal_indices.iter().enumerate() {
            let global_idx = offset + local_idx;
            for trait_idx in 0..effects.num_cell_types {
                let effect = effects.shared_effect_sizes[(trait_idx, j)];
                if effect.abs() > 1e-10 {
                    records.push(GtRecord {
                        global_idx,
                        block_idx: *block_idx,
                        effect_type: "shared",
                        trait_idx,
                        effect,
                    });
                }
            }
        }

        for (trait_idx, trait_indices) in effects.independent_causal_indices.iter().enumerate() {
            for (j, &local_idx) in trait_indices.iter().enumerate() {
                let global_idx = offset + local_idx;
                let effect = effects.independent_effect_sizes[(trait_idx, j)];
                if effect.abs() > 1e-10 {
                    records.push(GtRecord {
                        global_idx,
                        block_idx: *block_idx,
                        effect_type: "independent",
                        trait_idx,
                        effect,
                    });
                }
            }
        }
    }

    // Sort by position for tabix compatibility
    records.sort_by_key(|r| positions[r.global_idx]);

    let mut writer = open_bgzf_writer(path, tpool)?;
    writeln!(
        writer,
        "#chr\tstart\tend\tsnp_id\tblock_idx\teffect_type\ttrait_idx\teffect_size"
    )?;

    for rec in &records {
        let pos = positions[rec.global_idx];
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
            chromosomes[rec.global_idx],
            pos.saturating_sub(1),
            pos,
            snp_ids[rec.global_idx],
            rec.block_idx,
            rec.effect_type,
            rec.trait_idx,
            rec.effect,
        )?;
    }

    writer.flush()?;
    info!("Wrote ground truth to {}", path);
    Ok(())
}

/// Write confounder matrix (plain gzip, not BED-format)
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
