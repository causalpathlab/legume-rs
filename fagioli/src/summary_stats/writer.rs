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
use crate::simulation::MediationGeneEffects;

/// Open a BGZF writer with htslib thread pool for parallel compression.
fn open_bgzf_writer(path: &str, tpool: Option<&ThreadPool>) -> Result<bgzf::Writer> {
    let mut writer = bgzf::Writer::from_path(path)?;
    if let Some(tp) = tpool {
        writer.set_thread_pool(tp)?;
    }
    Ok(writer)
}

/// Shared SNP metadata for constructing streaming writers.
pub struct SnpWriterParams<'a> {
    pub path: &'a str,
    pub snp_ids: &'a [Box<str>],
    pub chromosomes: &'a [Box<str>],
    pub positions: &'a [u64],
    pub allele1: &'a [Box<str>],
    pub allele2: &'a [Box<str>],
    pub num_individuals: usize,
    pub tpool: Option<&'a ThreadPool>,
}

/// Streaming writer for summary statistics (BGZF-compressed, tabix-compatible)
pub struct SumstatWriter {
    writer: bgzf::Writer,
    snp_ids: Vec<Box<str>>,
    chromosomes: Vec<Box<str>>,
    positions: Vec<u64>,
    allele1: Vec<Box<str>>,
    allele2: Vec<Box<str>>,
    num_individuals: usize,
}

impl SumstatWriter {
    pub fn new(params: &SnpWriterParams) -> Result<Self> {
        let mut writer = open_bgzf_writer(params.path, params.tpool)?;
        writeln!(
            writer,
            "#chr\tstart\tend\tsnp_id\ta1\ta2\ttrait_idx\tn\tbeta\tse\tz\tpvalue"
        )?;

        Ok(Self {
            writer,
            snp_ids: params.snp_ids.to_vec(),
            chromosomes: params.chromosomes.to_vec(),
            positions: params.positions.to_vec(),
            allele1: params.allele1.to_vec(),
            allele2: params.allele2.to_vec(),
            num_individuals: params.num_individuals,
        })
    }

    pub fn write_block(&mut self, records: &[SumstatRecord]) -> Result<()> {
        for rec in records {
            let pos = self.positions[rec.snp_idx];
            writeln!(
                self.writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}",
                self.chromosomes[rec.snp_idx],
                pos.saturating_sub(1),
                pos,
                self.snp_ids[rec.snp_idx],
                self.allele1[rec.snp_idx],
                self.allele2[rec.snp_idx],
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

// ── Mediation-specific writers ──────────────────────────────────────────────

/// Streaming writer for eQTL summary statistics (per gene, cis-SNPs only)
pub struct EqtlSumstatWriter {
    writer: bgzf::Writer,
    snp_ids: Vec<Box<str>>,
    chromosomes: Vec<Box<str>>,
    positions: Vec<u64>,
    allele1: Vec<Box<str>>,
    allele2: Vec<Box<str>>,
    num_individuals: usize,
}

impl EqtlSumstatWriter {
    pub fn new(params: &SnpWriterParams) -> Result<Self> {
        let mut writer = open_bgzf_writer(params.path, params.tpool)?;
        writeln!(
            writer,
            "#chr\tstart\tend\tsnp_id\ta1\ta2\tgene_idx\tgene_id\tn\tbeta\tse\tz\tpvalue"
        )?;

        Ok(Self {
            writer,
            snp_ids: params.snp_ids.to_vec(),
            chromosomes: params.chromosomes.to_vec(),
            positions: params.positions.to_vec(),
            allele1: params.allele1.to_vec(),
            allele2: params.allele2.to_vec(),
            num_individuals: params.num_individuals,
        })
    }

    /// Write eQTL summary stats for one gene's cis-SNPs.
    /// `records` come from `compute_block_sumstats` with trait_idx=0.
    pub fn write_gene_block(
        &mut self,
        records: &[SumstatRecord],
        gene_idx: usize,
        gene_id: &str,
    ) -> Result<()> {
        for rec in records {
            let pos = self.positions[rec.snp_idx];
            writeln!(
                self.writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}",
                self.chromosomes[rec.snp_idx],
                pos.saturating_sub(1),
                pos,
                self.snp_ids[rec.snp_idx],
                self.allele1[rec.snp_idx],
                self.allele2[rec.snp_idx],
                gene_idx,
                gene_id,
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

/// Write mediation ground truth: causal eQTL SNPs with their effect chain
pub fn write_mediation_ground_truth(
    path: &str,
    effects: &[MediationGeneEffects],
    gene_ids: &[Box<str>],
    snp_ids: &[Box<str>],
    chromosomes: &[Box<str>],
    positions: &[u64],
    tpool: Option<&ThreadPool>,
) -> Result<()> {
    use crate::simulation::GeneRole;

    struct GtRecord {
        snp_idx: usize,
        gene_idx: usize,
        role: GeneRole,
        is_observed: bool,
        alpha: f32,
        beta: f32,
        theta: f32,
    }

    let mut records: Vec<GtRecord> = Vec::new();

    for eff in effects {
        for (j, &snp_idx) in eff.eqtl_snp_indices.iter().enumerate() {
            let alpha = eff.alpha[j];
            let theta = alpha * eff.beta;
            records.push(GtRecord {
                snp_idx,
                gene_idx: eff.gene_idx,
                role: eff.role,
                is_observed: eff.is_observed,
                alpha,
                beta: eff.beta,
                theta,
            });
        }
    }

    // Sort by position for tabix compatibility
    records.sort_by_key(|r| (positions[r.snp_idx], r.gene_idx));

    let mut writer = open_bgzf_writer(path, tpool)?;
    writeln!(
        writer,
        "#chr\tstart\tend\tsnp_id\tgene_idx\tgene_id\trole\tis_observed\talpha\tbeta\ttheta"
    )?;

    for rec in &records {
        let pos = positions[rec.snp_idx];
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}",
            chromosomes[rec.snp_idx],
            pos.saturating_sub(1),
            pos,
            snp_ids[rec.snp_idx],
            rec.gene_idx,
            gene_ids[rec.gene_idx],
            rec.role,
            rec.is_observed,
            rec.alpha,
            rec.beta,
            rec.theta,
        )?;
    }

    writer.flush()?;
    info!("Wrote mediation ground truth to {}", path);
    Ok(())
}

/// Gene metadata needed for writing the gene table.
pub struct GeneTableInput<'a> {
    pub effects: &'a [MediationGeneEffects],
    pub gene_ids: &'a [Box<str>],
    pub gene_names: &'a [Option<Box<str>>],
    pub gene_chromosomes: &'a [Box<str>],
    pub gene_tss: &'a [u64],
    pub num_cis_snps: &'a [usize],
}

/// Write gene table with causal/observed flags.
///
/// `num_cis_snps` should be precomputed per gene (one entry per effect) to
/// avoid an O(G×M) scan inside the writer.
pub fn write_gene_table(
    path: &str,
    input: &GeneTableInput,
    tpool: Option<&ThreadPool>,
) -> Result<()> {
    let GeneTableInput {
        effects,
        gene_ids,
        gene_names,
        gene_chromosomes,
        gene_tss,
        num_cis_snps,
    } = input;
    let mut writer = open_bgzf_writer(path, tpool)?;
    writeln!(
        writer,
        "#chr\ttss\tgene_idx\tgene_id\tgene_name\trole\tis_observed\tbeta\tnum_cis_snps\tnum_eqtl_snps"
    )?;

    for (idx, eff) in effects.iter().enumerate() {
        let g = eff.gene_idx;
        let name = gene_names[g].as_deref().unwrap_or(".");

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
            gene_chromosomes[g],
            gene_tss[g],
            g,
            gene_ids[g],
            name,
            eff.role,
            eff.is_observed,
            eff.beta,
            num_cis_snps[idx],
            eff.eqtl_snp_indices.len(),
        )?;
    }

    writer.flush()?;
    info!("Wrote gene table ({} genes) to {}", effects.len(), path);
    Ok(())
}
