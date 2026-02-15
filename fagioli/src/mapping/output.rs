use anyhow::Result;
use log::info;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

use crate::genotype::GenotypeMatrix;
use super::gene_mapping::GeneQtlResult;

/// A single long-form row: one per (variant, gene, cell_type).
struct OutputRow {
    chromosome: Box<str>,
    position: u64,
    gene_id: Box<str>,
    cell_type: Box<str>,
    snp_id: Box<str>,
    effect_size: f32,
    effect_std: f32,
    z_score: f32,
    p_value: f32,
    pip: Option<f32>,
}

/// Write QTL mapping results to Parquet and bgzipped TSV (tabix-indexed).
///
/// For Susie results, `pip_threshold` filters rows with PIP below threshold.
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

    let total_rows: usize = results.iter().map(|r| r.snp_indices.len()).sum();

    if total_rows == 0 {
        info!("No SNP results to write");
        return Ok(());
    }

    // Write Parquet
    write_results_parquet(output_prefix, results, genotype_matrix)?;

    // Write bgzipped, sorted, tabix-indexed TSV
    write_results_tsv(output_prefix, results, genotype_matrix, pip_threshold)?;

    Ok(())
}

/// Write results as a Parquet file.
fn write_results_parquet(
    output_prefix: &str,
    results: &[GeneQtlResult],
    genotype_matrix: &GenotypeMatrix,
) -> Result<()> {
    let total_rows: usize = results.iter().map(|r| r.snp_indices.len()).sum();

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

    Ok(())
}

/// Write results as a genomically-sorted, bgzipped TSV with tabix index.
///
/// Format: #chrom, pos, gene@celltype, snp_id, effect_size, effect_std, z_score, p_value, pip
/// Sorted by (chromosome, position, gene, cell_type).
/// Compressed with bgzip for tabix indexing.
fn write_results_tsv(
    output_prefix: &str,
    results: &[GeneQtlResult],
    genotype_matrix: &GenotypeMatrix,
    pip_threshold: f32,
) -> Result<()> {
    use rust_htslib::bgzf::Writer as BgzfWriter;
    use std::io::Write;

    let has_pips = results.iter().any(|r| r.pips.is_some());

    // Build long-form rows
    let mut rows: Vec<OutputRow> = Vec::new();

    for result in results {
        for (local_idx, &global_snp_idx) in result.snp_indices.iter().enumerate() {
            let pip = result.pips.as_ref().map(|p| p[local_idx]);

            // Apply PIP threshold
            if has_pips && pip_threshold > 0.0 {
                if let Some(p) = pip {
                    if p < pip_threshold {
                        continue;
                    }
                }
            }

            rows.push(OutputRow {
                chromosome: genotype_matrix.chromosomes[global_snp_idx].clone(),
                position: genotype_matrix.positions[global_snp_idx],
                gene_id: result.gene_id.clone(),
                cell_type: result.cell_type.clone(),
                snp_id: genotype_matrix.snp_ids[global_snp_idx].clone(),
                effect_size: result.effect_sizes[local_idx],
                effect_std: result.effect_stds[local_idx],
                z_score: result.z_scores[local_idx],
                p_value: result.p_values[local_idx],
                pip,
            });
        }
    }

    // Sort genomically: chromosome (natural order), position, gene, cell_type
    rows.sort_by(|a, b| {
        chr_sort_key(&a.chromosome)
            .cmp(&chr_sort_key(&b.chromosome))
            .then(a.position.cmp(&b.position))
            .then(a.gene_id.cmp(&b.gene_id))
            .then(a.cell_type.cmp(&b.cell_type))
    });

    let out_file = format!("{}.qtl_results.tsv.gz", output_prefix);
    let mut writer = BgzfWriter::from_path(&out_file)
        .map_err(|e| anyhow::anyhow!("Failed to create bgzf writer: {}", e))?;

    // Header (prefixed with # for tabix compatibility)
    if has_pips {
        writeln!(
            writer,
            "#chrom\tpos\tgene@celltype\tsnp_id\teffect_size\teffect_std\tz_score\tp_value\tpip"
        )?;
    } else {
        writeln!(
            writer,
            "#chrom\tpos\tgene@celltype\tsnp_id\teffect_size\teffect_std\tz_score\tp_value"
        )?;
    }

    for row in &rows {
        let id = format!("{}@{}", row.gene_id, row.cell_type);

        if let Some(pip) = row.pip {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}\t{:.6}",
                row.chromosome, row.position, id, row.snp_id,
                row.effect_size, row.effect_std, row.z_score, row.p_value, pip
            )?;
        } else {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}",
                row.chromosome, row.position, id, row.snp_id,
                row.effect_size, row.effect_std, row.z_score, row.p_value
            )?;
        }
    }

    writer.flush()?;
    drop(writer);

    // Build tabix index: sequence col 1, begin col 2, 1-based, skip comment lines
    match std::process::Command::new("tabix")
        .args(["-s", "1", "-b", "2", "-e", "2", "-S", "1", &out_file])
        .output()
    {
        Ok(output) if output.status.success() => {
            info!("Built tabix index: {}.tbi", out_file);
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::warn!(
                "tabix indexing failed ({}): {}. Run manually: tabix -s1 -b2 -e2 -S1 {}",
                output.status, stderr.trim(), out_file
            );
        }
        Err(_) => {
            log::warn!(
                "tabix not found in PATH. Run manually: tabix -s1 -b2 -e2 -S1 {}",
                out_file
            );
        }
    }

    info!(
        "Wrote {} rows (sorted, bgzipped): {}",
        rows.len(),
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
