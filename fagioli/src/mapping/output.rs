use anyhow::Result;
use log::info;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

use crate::genotype::GenotypeMatrix;
use super::gene_mapping::GeneQtlResult;

/// A single long-form VCF record: one row per (variant, gene, cell_type).
struct VcfRecord {
    chromosome: Box<str>,
    position: u64,
    snp_id: Box<str>,
    ref_allele: Box<str>,
    alt_allele: Box<str>,
    gene_id: Box<str>,
    cell_type: Box<str>,
    effect_size: f32,
    effect_std: f32,
    z_score: f32,
    p_value: f32,
    pip: Option<f32>,
}

/// Write QTL mapping results to Parquet, TSV.GZ, and VCF.GZ.
///
/// For Susie results, `pip_threshold` filters variants below the threshold
/// from the VCF output (set to 0.0 to include all).
pub fn write_mapping_results(
    output_prefix: &str,
    results: &[GeneQtlResult],
    genotype_matrix: &GenotypeMatrix,
    pip_threshold: f32,
) -> Result<()> {
    if results.is_empty() {
        info!("No mapping results to write");
        return Ok(());
    }

    // Count total rows (gene × snp × cell_type)
    let total_rows: usize = results.iter().map(|r| r.snp_indices.len()).sum();

    if total_rows == 0 {
        info!("No SNP results to write");
        return Ok(());
    }

    // Build columns
    let mut gene_ids: Vec<Box<str>> = Vec::with_capacity(total_rows);
    let mut snp_ids: Vec<Box<str>> = Vec::with_capacity(total_rows);
    let mut chromosomes: Vec<Box<str>> = Vec::with_capacity(total_rows);
    let mut positions: Vec<f32> = Vec::with_capacity(total_rows);
    let mut cell_types: Vec<Box<str>> = Vec::with_capacity(total_rows);
    let mut effect_sizes: Vec<f32> = Vec::with_capacity(total_rows);
    let mut effect_stds: Vec<f32> = Vec::with_capacity(total_rows);
    let mut z_scores: Vec<f32> = Vec::with_capacity(total_rows);
    let mut p_values: Vec<f32> = Vec::with_capacity(total_rows);
    let mut pips: Vec<f32> = Vec::with_capacity(total_rows);

    for result in results {
        for (local_idx, &global_snp_idx) in result.snp_indices.iter().enumerate() {
            gene_ids.push(result.gene_id.clone());
            snp_ids.push(genotype_matrix.snp_ids[global_snp_idx].clone());
            chromosomes.push(genotype_matrix.chromosomes[global_snp_idx].clone());
            positions.push(genotype_matrix.positions[global_snp_idx] as f32);
            cell_types.push(result.cell_type.clone());
            effect_sizes.push(result.effect_sizes[local_idx]);
            effect_stds.push(result.effect_stds[local_idx]);
            z_scores.push(result.z_scores[local_idx]);
            p_values.push(result.p_values[local_idx]);
            pips.push(
                result
                    .pips
                    .as_ref()
                    .map(|p| p[local_idx])
                    .unwrap_or(f32::NAN),
            );
        }
    }

    // Build a numeric matrix: total_rows × 5 columns (effect_size, effect_std, z_score, p_value, pip)
    let num_numeric_cols = 5;
    let mut numeric_data = DMatrix::<f32>::zeros(total_rows, num_numeric_cols);
    for i in 0..total_rows {
        numeric_data[(i, 0)] = effect_sizes[i];
        numeric_data[(i, 1)] = effect_stds[i];
        numeric_data[(i, 2)] = z_scores[i];
        numeric_data[(i, 3)] = p_values[i];
        numeric_data[(i, 4)] = pips[i];
    }

    let col_names: Vec<Box<str>> = vec![
        Box::from("effect_size"),
        Box::from("effect_std"),
        Box::from("z_score"),
        Box::from("p_value"),
        Box::from("pip"),
    ];

    // Row names encode gene_id|snp_id|chr|pos|cell_type for metadata
    let row_labels: Vec<Box<str>> = (0..total_rows)
        .map(|i| {
            Box::from(format!(
                "{}|{}|{}|{}|{}",
                gene_ids[i], snp_ids[i], chromosomes[i], positions[i] as u64, cell_types[i]
            ))
        })
        .collect();

    let out_file = format!("{}.qtl_results.parquet", output_prefix);
    numeric_data.to_parquet_with_names(
        &out_file,
        (Some(&row_labels), Some("variant")),
        Some(&col_names),
    )?;

    info!(
        "Wrote {} QTL results ({} genes × SNPs) to {}",
        total_rows,
        results.len(),
        out_file
    );

    // Also write a TSV.GZ with full metadata columns for easier downstream use
    write_results_tsv(output_prefix, results, genotype_matrix)?;

    // Write VCF.GZ sorted by chromosome/position, with PIP filtering
    write_results_vcf(output_prefix, results, genotype_matrix, pip_threshold)?;

    Ok(())
}

/// Write results as a TSV.GZ file with all columns explicit.
fn write_results_tsv(
    output_prefix: &str,
    results: &[GeneQtlResult],
    genotype_matrix: &GenotypeMatrix,
) -> Result<()> {
    use matrix_util::common_io::open_buf_writer;
    use std::io::Write;

    let out_file = format!("{}.qtl_results.tsv.gz", output_prefix);
    let mut writer = open_buf_writer(&out_file)?;

    writeln!(
        writer,
        "gene_id\tsnp_id\tchromosome\tposition\tcell_type\teffect_size\teffect_std\tz_score\tp_value\tpip"
    )?;

    for result in results {
        for (local_idx, &global_snp_idx) in result.snp_indices.iter().enumerate() {
            let pip_str = result
                .pips
                .as_ref()
                .map(|p| format!("{:.6}", p[local_idx]))
                .unwrap_or_else(|| "NA".to_string());

            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}\t{}",
                result.gene_id,
                genotype_matrix.snp_ids[global_snp_idx],
                genotype_matrix.chromosomes[global_snp_idx],
                genotype_matrix.positions[global_snp_idx],
                result.cell_type,
                result.effect_sizes[local_idx],
                result.effect_stds[local_idx],
                result.z_scores[local_idx],
                result.p_values[local_idx],
                pip_str,
            )?;
        }
    }

    writer.flush()?;
    info!("Wrote TSV results: {}", out_file);
    Ok(())
}

/// Write results as a long-form sorted VCF.GZ file.
///
/// Each row is one (variant, gene, cell_type) combination. The ID column
/// contains `gene@cell_type` for easy filtering. For Susie results, rows
/// with PIP below `pip_threshold` are excluded.
fn write_results_vcf(
    output_prefix: &str,
    results: &[GeneQtlResult],
    genotype_matrix: &GenotypeMatrix,
    pip_threshold: f32,
) -> Result<()> {
    use rust_htslib::bgzf::Writer as BgzfWriter;
    use std::io::Write;

    let has_pips = results.iter().any(|r| r.pips.is_some());

    // Build long-form records: one per (variant, gene, cell_type)
    let mut records: Vec<VcfRecord> = Vec::new();

    for result in results {
        for (local_idx, &global_snp_idx) in result.snp_indices.iter().enumerate() {
            let pip = result.pips.as_ref().map(|p| p[local_idx]);

            // Apply PIP threshold per row
            if has_pips && pip_threshold > 0.0 {
                if let Some(p) = pip {
                    if p < pip_threshold {
                        continue;
                    }
                }
            }

            records.push(VcfRecord {
                chromosome: genotype_matrix.chromosomes[global_snp_idx].clone(),
                position: genotype_matrix.positions[global_snp_idx],
                snp_id: genotype_matrix.snp_ids[global_snp_idx].clone(),
                ref_allele: genotype_matrix.allele1[global_snp_idx].clone(),
                alt_allele: genotype_matrix.allele2[global_snp_idx].clone(),
                gene_id: result.gene_id.clone(),
                cell_type: result.cell_type.clone(),
                effect_size: result.effect_sizes[local_idx],
                effect_std: result.effect_stds[local_idx],
                z_score: result.z_scores[local_idx],
                p_value: result.p_values[local_idx],
                pip,
            });
        }
    }

    // Sort by chromosome (natural order), then position, then gene@celltype
    records.sort_by(|a, b| {
        chr_sort_key(&a.chromosome)
            .cmp(&chr_sort_key(&b.chromosome))
            .then(a.position.cmp(&b.position))
            .then(a.gene_id.cmp(&b.gene_id))
            .then(a.cell_type.cmp(&b.cell_type))
    });

    let out_file = format!("{}.qtl_results.vcf.gz", output_prefix);
    let mut writer = BgzfWriter::from_path(&out_file)
        .map_err(|e| anyhow::anyhow!("Failed to create bgzf writer: {}", e))?;

    // VCF header
    writeln!(writer, "##fileformat=VCFv4.2")?;
    writeln!(
        writer,
        "##INFO=<ID=SNP,Number=1,Type=String,Description=\"SNP identifier\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=ES,Number=1,Type=Float,Description=\"Effect size (posterior mean)\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=SE,Number=1,Type=Float,Description=\"Effect size standard deviation\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=Z,Number=1,Type=Float,Description=\"Z-score\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=P,Number=1,Type=Float,Description=\"P-value (two-sided)\">"
    )?;
    if has_pips {
        writeln!(
            writer,
            "##INFO=<ID=PIP,Number=1,Type=Float,Description=\"Posterior inclusion probability\">"
        )?;
    }
    writeln!(writer, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")?;

    // Write records
    for rec in &records {
        let id = format!("{}@{}", rec.gene_id, rec.cell_type);

        let mut info_parts: Vec<String> = Vec::with_capacity(6);
        info_parts.push(format!("SNP={}", rec.snp_id));
        info_parts.push(format!("ES={:.6}", rec.effect_size));
        info_parts.push(format!("SE={:.6}", rec.effect_std));
        info_parts.push(format!("Z={:.4}", rec.z_score));
        info_parts.push(format!("P={:.6e}", rec.p_value));
        if let Some(pip) = rec.pip {
            info_parts.push(format!("PIP={:.6}", pip));
        }

        let info_str = info_parts.join(";");

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t.\tPASS\t{}",
            rec.chromosome, rec.position, id, rec.ref_allele, rec.alt_allele, info_str
        )?;
    }

    writer.flush()?;
    drop(writer);

    // Build tabix index via system tabix command
    match std::process::Command::new("tabix")
        .args(["-p", "vcf", &out_file])
        .output()
    {
        Ok(output) if output.status.success() => {
            info!("Built tabix index: {}.tbi", out_file);
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::warn!(
                "tabix indexing failed ({}): {}. Run manually: tabix -p vcf {}",
                output.status, stderr.trim(), out_file
            );
        }
        Err(_) => {
            log::warn!(
                "tabix not found in PATH. Run manually: tabix -p vcf {}",
                out_file
            );
        }
    }

    info!(
        "Wrote VCF: {} rows (long-form): {}",
        records.len(),
        out_file
    );

    Ok(())
}

/// Sort key for chromosome names: numeric chromosomes first (1-22),
/// then X=23, Y=24, MT=25, then alphabetical.
fn chr_sort_key(chr: &str) -> (u32, Box<str>) {
    let stripped = chr
        .strip_prefix("chr")
        .or_else(|| chr.strip_prefix("Chr"))
        .unwrap_or(chr);

    if let Ok(n) = stripped.parse::<u32>() {
        (n, Box::from(""))
    } else {
        match stripped.to_uppercase().as_str() {
            "X" => (23, Box::from("")),
            "Y" => (24, Box::from("")),
            "MT" | "M" => (25, Box::from("")),
            _ => (100, Box::from(stripped)),
        }
    }
}
