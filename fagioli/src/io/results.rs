use anyhow::Result;
use log::info;
use rust_htslib::bgzf;
use rust_htslib::tpool::ThreadPool;
use std::io::Write;

use crate::genotype::GenotypeMatrix;
use crate::mapping::map_qtl_helpers::GeneResult;

/// A single row of variant-level fine-mapping results.
pub struct VariantRow {
    /// Global SNP index into the GenotypeMatrix
    pub snp_idx: usize,
    /// Label values for context columns (e.g. gene_id + cell_type, or trait name)
    pub labels: Vec<Box<str>>,
    pub pip: f32,
    pub effect_mean: f32,
    pub effect_std: f32,
    pub z_marginal: f32,
}

/// Write variant-level fine-mapping results to a BED.GZ file.
///
/// Columns: `#chr start end snp_id {label_columns...} pip effect_mean effect_std z_marginal`
pub fn write_variant_results(
    out_file: &str,
    label_columns: &[&str],
    rows: &[VariantRow],
    geno: &GenotypeMatrix,
    tpool: &ThreadPool,
) -> Result<()> {
    info!("Writing results to {}", out_file);

    let mut writer = bgzf::Writer::from_path(out_file)?;
    writer.set_thread_pool(tpool)?;

    // Header
    write!(writer, "#chr\tstart\tend\tsnp_id")?;
    for col in label_columns {
        write!(writer, "\t{}", col)?;
    }
    writeln!(writer, "\tpip\teffect_mean\teffect_std\tz_marginal")?;

    // Rows
    for row in rows {
        let chr = &geno.chromosomes[row.snp_idx];
        let pos = geno.positions[row.snp_idx];
        let snp_id = &geno.snp_ids[row.snp_idx];

        write!(writer, "{}\t{}\t{}\t{}", chr, pos, pos + 1, snp_id)?;
        for label in &row.labels {
            write!(writer, "\t{}", label)?;
        }
        writeln!(
            writer,
            "\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
            row.pip, row.effect_mean, row.effect_std, row.z_marginal,
        )?;
    }

    writer.flush()?;
    info!("Results written: {}", out_file);
    Ok(())
}

/// Write per-gene summary TSV.
pub fn write_gene_summary(
    summary_file: &str,
    gene_results: &[GeneResult],
    n_ct: usize,
    tpool: &ThreadPool,
) -> Result<()> {
    info!("Writing gene summary to {}", summary_file);

    let mut writer = bgzf::Writer::from_path(summary_file)?;
    writer.set_thread_pool(tpool)?;

    writeln!(
        writer,
        "gene_id\tnum_cis_snps\tmax_pip\tavg_elbo\tnum_significant_cell_types"
    )?;

    for gr in gene_results {
        let max_pip = gr
            .detailed
            .result
            .pip
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);

        let mut sig_ct = 0usize;
        for ct_idx in 0..n_ct {
            let has_sig =
                (0..gr.cis_snp_indices.len()).any(|j| gr.detailed.result.pip[(j, ct_idx)] >= 0.5);
            if has_sig {
                sig_ct += 1;
            }
        }

        writeln!(
            writer,
            "{}\t{}\t{:.6}\t{:.2}\t{}",
            gr.gene_id,
            gr.cis_snp_indices.len(),
            max_pip,
            gr.detailed.result.avg_elbo,
            sig_ct,
        )?;
    }

    writer.flush()?;
    info!("Gene summary written: {}", summary_file);
    Ok(())
}

/// Write parameters JSON.
pub fn write_parameters(param_file: &str, params: &serde_json::Value) -> Result<()> {
    std::fs::write(param_file, serde_json::to_string_pretty(params)?)?;
    info!("Wrote parameters: {}", param_file);
    Ok(())
}
