use std::collections::HashMap;

use anyhow::Result;
use log::info;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

use crate::genotype::GenotypeMatrix;
use super::gene_mapping::GeneQtlResult;

/// A single VCF record aggregating all (gene, cell_type) results for one variant.
struct VcfRecord {
    chromosome: Box<str>,
    position: u64,
    snp_id: Box<str>,
    ref_allele: Box<str>,
    alt_allele: Box<str>,
    /// Per (gene_id, cell_type): (effect_size, effect_std, z_score, p_value, pip)
    annotations: Vec<VcfAnnotation>,
}

struct VcfAnnotation {
    gene_id: Box<str>,
    cell_type: Box<str>,
    effect_size: f32,
    effect_std: f32,
    z_score: f32,
    p_value: f32,
    pip: Option<f32>,
    elbo: f32,
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
    let mut elbos: Vec<f32> = Vec::with_capacity(total_rows);

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
            elbos.push(result.final_elbo);
        }
    }

    // Build a numeric matrix: total_rows × 6 columns (effect_size, effect_std, z_score, p_value, pip, elbo)
    let num_numeric_cols = 6;
    let mut numeric_data = DMatrix::<f32>::zeros(total_rows, num_numeric_cols);
    for i in 0..total_rows {
        numeric_data[(i, 0)] = effect_sizes[i];
        numeric_data[(i, 1)] = effect_stds[i];
        numeric_data[(i, 2)] = z_scores[i];
        numeric_data[(i, 3)] = p_values[i];
        numeric_data[(i, 4)] = pips[i];
        numeric_data[(i, 5)] = elbos[i];
    }

    let col_names: Vec<Box<str>> = vec![
        Box::from("effect_size"),
        Box::from("effect_std"),
        Box::from("z_score"),
        Box::from("p_value"),
        Box::from("pip"),
        Box::from("elbo"),
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
        "gene_id\tsnp_id\tchromosome\tposition\tcell_type\teffect_size\teffect_std\tz_score\tp_value\tpip\telbo"
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
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6e}\t{}\t{:.2}",
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
                result.final_elbo,
            )?;
        }
    }

    writer.flush()?;
    info!("Wrote TSV results: {}", out_file);
    Ok(())
}

/// Write results as a sorted VCF.GZ file.
///
/// Variants are sorted by (chromosome, position). For Susie results, variants
/// where the maximum PIP across all gene/cell_type annotations is below
/// `pip_threshold` are excluded.
fn write_results_vcf(
    output_prefix: &str,
    results: &[GeneQtlResult],
    genotype_matrix: &GenotypeMatrix,
    pip_threshold: f32,
) -> Result<()> {
    use rust_htslib::bgzf::Writer as BgzfWriter;
    use std::io::Write;

    // Aggregate results by global SNP index
    let mut snp_records: HashMap<usize, VcfRecord> = HashMap::new();

    for result in results {
        for (local_idx, &global_snp_idx) in result.snp_indices.iter().enumerate() {
            let pip = result.pips.as_ref().map(|p| p[local_idx]);

            let annotation = VcfAnnotation {
                gene_id: result.gene_id.clone(),
                cell_type: result.cell_type.clone(),
                effect_size: result.effect_sizes[local_idx],
                effect_std: result.effect_stds[local_idx],
                z_score: result.z_scores[local_idx],
                p_value: result.p_values[local_idx],
                pip,
                elbo: result.final_elbo,
            };

            snp_records
                .entry(global_snp_idx)
                .or_insert_with(|| VcfRecord {
                    chromosome: genotype_matrix.chromosomes[global_snp_idx].clone(),
                    position: genotype_matrix.positions[global_snp_idx],
                    snp_id: genotype_matrix.snp_ids[global_snp_idx].clone(),
                    ref_allele: genotype_matrix.allele1[global_snp_idx].clone(),
                    alt_allele: genotype_matrix.allele2[global_snp_idx].clone(),
                    annotations: Vec::new(),
                })
                .annotations
                .push(annotation);
        }
    }

    // Apply PIP threshold: keep variant if max PIP across annotations >= threshold
    let has_pips = results.iter().any(|r| r.pips.is_some());
    let mut records: Vec<VcfRecord> = snp_records
        .into_values()
        .filter(|rec| {
            if !has_pips || pip_threshold <= 0.0 {
                return true;
            }
            let max_pip = rec
                .annotations
                .iter()
                .filter_map(|a| a.pip)
                .fold(0.0f32, f32::max);
            max_pip >= pip_threshold
        })
        .collect();

    let total_before = results.iter().map(|r| r.snp_indices.len()).sum::<usize>();
    let filtered_annotations: usize = records.iter().map(|r| r.annotations.len()).sum();

    // Sort by chromosome (natural order), then position
    records.sort_by(|a, b| {
        chr_sort_key(&a.chromosome)
            .cmp(&chr_sort_key(&b.chromosome))
            .then(a.position.cmp(&b.position))
    });

    let out_file = format!("{}.qtl_results.vcf.gz", output_prefix);
    let mut writer = BgzfWriter::from_path(&out_file)
        .map_err(|e| anyhow::anyhow!("Failed to create bgzf writer: {}", e))?;

    // VCF header
    writeln!(writer, "##fileformat=VCFv4.2")?;
    writeln!(
        writer,
        "##INFO=<ID=GENE,Number=.,Type=String,Description=\"Gene ID\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=CT,Number=.,Type=String,Description=\"Cell type\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=ES,Number=.,Type=Float,Description=\"Effect size (posterior mean)\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=SE,Number=.,Type=Float,Description=\"Effect size standard deviation\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=Z,Number=.,Type=Float,Description=\"Z-score\">"
    )?;
    writeln!(
        writer,
        "##INFO=<ID=P,Number=.,Type=Float,Description=\"P-value (two-sided)\">"
    )?;
    if has_pips {
        writeln!(
            writer,
            "##INFO=<ID=PIP,Number=.,Type=Float,Description=\"Posterior inclusion probability\">"
        )?;
    }
    writeln!(
        writer,
        "##INFO=<ID=ELBO,Number=.,Type=Float,Description=\"Final ELBO\">"
    )?;
    if has_pips && pip_threshold > 0.0 {
        writeln!(
            writer,
            "##FILTER=<ID=LOW_PIP,Description=\"Max PIP below {}\">",
            pip_threshold
        )?;
    }
    writeln!(writer, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")?;

    // Write records
    for rec in &records {
        // Build INFO field: one annotation set per (gene, cell_type)
        let mut info_parts: Vec<String> = Vec::new();
        let mut genes: Vec<&str> = Vec::new();
        let mut cts: Vec<&str> = Vec::new();
        let mut ess: Vec<String> = Vec::new();
        let mut ses: Vec<String> = Vec::new();
        let mut zs: Vec<String> = Vec::new();
        let mut ps: Vec<String> = Vec::new();
        let mut pips_vec: Vec<String> = Vec::new();
        let mut elbos: Vec<String> = Vec::new();

        for ann in &rec.annotations {
            genes.push(&ann.gene_id);
            cts.push(&ann.cell_type);
            ess.push(format!("{:.6}", ann.effect_size));
            ses.push(format!("{:.6}", ann.effect_std));
            zs.push(format!("{:.4}", ann.z_score));
            ps.push(format!("{:.6e}", ann.p_value));
            if let Some(pip) = ann.pip {
                pips_vec.push(format!("{:.6}", pip));
            }
            elbos.push(format!("{:.2}", ann.elbo));
        }

        info_parts.push(format!("GENE={}", genes.join(",")));
        info_parts.push(format!("CT={}", cts.join(",")));
        info_parts.push(format!("ES={}", ess.join(",")));
        info_parts.push(format!("SE={}", ses.join(",")));
        info_parts.push(format!("Z={}", zs.join(",")));
        info_parts.push(format!("P={}", ps.join(",")));
        if !pips_vec.is_empty() {
            info_parts.push(format!("PIP={}", pips_vec.join(",")));
        }
        info_parts.push(format!("ELBO={}", elbos.join(",")));

        let info_str = info_parts.join(";");

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t.\tPASS\t{}",
            rec.chromosome, rec.position, rec.snp_id, rec.ref_allele, rec.alt_allele, info_str
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

    if has_pips && pip_threshold > 0.0 {
        info!(
            "Wrote VCF: {} variants ({} annotations, filtered from {} at PIP >= {}): {}",
            records.len(),
            filtered_annotations,
            total_before,
            pip_threshold,
            out_file
        );
    } else {
        info!(
            "Wrote VCF: {} variants ({} annotations): {}",
            records.len(),
            filtered_annotations,
            out_file
        );
    }

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
