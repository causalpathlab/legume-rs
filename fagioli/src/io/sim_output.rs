//! Output writers for single-cell eQTL simulation results.

use anyhow::Result;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use log::info;
use matrix_util::common_io::open_buf_writer;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;
use std::io::Write;

use crate::genotype::GenotypeMatrix;
use crate::simulation::{CellTypeGeneticEffects, FactorModel, GeneAnnotations, ScCountData};

/// Parameters controlling sim-qtl output format.
pub struct SimQtlOutputParams<'a> {
    pub output_prefix: &'a str,
    pub backend: SparseIoBackend,
    pub num_cell_types: usize,
    pub num_factors: usize,
}

/// Write all outputs from a sim-qtl run.
#[allow(clippy::too_many_arguments)]
pub fn write_sim_qtl_outputs(
    params: &SimQtlOutputParams<'_>,
    geno: &GenotypeMatrix,
    genes: &GeneAnnotations,
    cell_frac: &DMatrix<f32>,
    factor_model: &FactorModel,
    gene_effects: &[Option<CellTypeGeneticEffects>],
    sc_data: &ScCountData,
    use_parquet: bool,
) -> Result<()> {
    let ext = if use_parquet { "parquet" } else { "tsv" };

    let backend_ext = match params.backend {
        SparseIoBackend::Zarr => "zarr",
        SparseIoBackend::HDF5 => "h5",
    };

    let backend_file = format!("{}.counts.{}", params.output_prefix, backend_ext);
    info!("Creating sparse count matrix: {}", backend_file);

    let mtx_shape = (sc_data.num_genes, sc_data.num_cells, sc_data.triplets.len());
    let mut sparse_backend = create_sparse_from_triplets(
        &sc_data.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&params.backend),
    )?;

    let gene_names: Vec<Box<str>> = genes
        .genes
        .iter()
        .map(|g| {
            if let Some(ref name) = g.gene_name {
                Box::from(format!("{}_{}", g.gene_id, name))
            } else {
                Box::from(g.gene_id.to_string())
            }
        })
        .collect();
    sparse_backend.register_row_names_vec(&gene_names);

    let cell_names: Vec<Box<str>> = (0..sc_data.num_cells)
        .map(|i| {
            let ind_id = &geno.individual_ids[sc_data.cell_individuals[i]];
            Box::from(format!("cell_{}@{}", i, ind_id))
        })
        .collect();
    sparse_backend.register_column_names_vec(&cell_names);

    info!(
        "Wrote {} genes × {} cells: {}",
        sc_data.num_genes, sc_data.num_cells, backend_file
    );

    // 2. Cell annotations
    let cell_anno_file = format!("{}.cells.tsv.gz", params.output_prefix);
    let mut writer = open_buf_writer(&cell_anno_file)?;
    writeln!(writer, "cell_id\tindividual_id\tcell_type")?;
    for (cell_idx, (&ind_id, &ct)) in sc_data
        .cell_individuals
        .iter()
        .zip(&sc_data.cell_types)
        .enumerate()
    {
        writeln!(
            writer,
            "cell_{}@{}\t{}\tcell_type_{}",
            cell_idx, geno.individual_ids[ind_id], geno.individual_ids[ind_id], ct
        )?;
    }
    writer.flush()?;
    info!("Wrote cell annotations: {}", cell_anno_file);

    // 3. Cell-to-individual mapping
    let mapping_file = format!("{}.cell_to_individual.tsv.gz", params.output_prefix);
    let mut writer = open_buf_writer(&mapping_file)?;
    writeln!(writer, "cell_id\tindividual_id\tindividual_index")?;
    for (cell_idx, &ind_idx) in sc_data.cell_individuals.iter().enumerate() {
        writeln!(
            writer,
            "cell_{}@{}\t{}\t{}",
            cell_idx, geno.individual_ids[ind_idx], geno.individual_ids[ind_idx], ind_idx
        )?;
    }
    writer.flush()?;
    info!("Wrote cell-to-individual mapping: {}", mapping_file);

    // 4. Gene annotations
    let gene_anno_file = format!("{}.genes.tsv.gz", params.output_prefix);
    let mut writer = open_buf_writer(&gene_anno_file)?;
    writeln!(
        writer,
        "gene_idx\tgene_id\tgene_name\tchromosome\ttss\tstrand"
    )?;
    for (idx, gene) in genes.genes.iter().enumerate() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            idx,
            gene.gene_id,
            gene.gene_name.as_ref().map(|s| s.as_ref()).unwrap_or("NA"),
            gene.chromosome,
            gene.tss,
            gene.strand
        )?;
    }
    writer.flush()?;
    info!("Wrote gene annotations: {}", gene_anno_file);

    // 5. Cell type fractions
    let cell_type_names: Vec<Box<str>> = (0..params.num_cell_types)
        .map(|i| Box::from(format!("cell_type_{}", i)))
        .collect();

    let frac_file = format!("{}.cell_fractions.{}", params.output_prefix, ext);
    if use_parquet {
        cell_frac.to_parquet_with_names(
            &frac_file,
            (Some(&geno.individual_ids), Some("individual")),
            Some(&cell_type_names),
        )?;
    } else {
        cell_frac.to_tsv(&frac_file)?;
    }
    info!("Wrote cell fractions: {}", frac_file);

    // 6. Factor model
    let gene_ids: Vec<Box<str>> = genes
        .genes
        .iter()
        .map(|g| Box::from(g.gene_id.to_string()))
        .collect();
    let factor_names: Vec<Box<str>> = (0..params.num_factors)
        .map(|i| Box::from(format!("factor_{}", i)))
        .collect();

    let loadings_file = format!("{}.gene_loadings.{}", params.output_prefix, ext);
    if use_parquet {
        factor_model.gene_loadings.to_parquet_with_names(
            &loadings_file,
            (Some(&gene_ids), Some("gene")),
            Some(&factor_names),
        )?;
    } else {
        factor_model.gene_loadings.to_tsv(&loadings_file)?;
    }
    info!("Wrote gene loadings: {}", loadings_file);

    let scores_file = format!("{}.factor_celltype.{}", params.output_prefix, ext);
    if use_parquet {
        factor_model.factor_celltype.to_parquet_with_names(
            &scores_file,
            (Some(&factor_names), Some("factor")),
            Some(&cell_type_names),
        )?;
    } else {
        factor_model.factor_celltype.to_tsv(&scores_file)?;
    }
    info!("Wrote factor-celltype scores: {}", scores_file);

    // 7. Individual-level log-rates (N × G per cell type)
    for ct in 0..params.num_cell_types {
        let lr_file = format!(
            "{}.log_rates.cell_type_{}.{}",
            params.output_prefix, ct, ext
        );
        if use_parquet {
            sc_data.individual_log_rates[ct].to_parquet_with_names(
                &lr_file,
                (Some(&geno.individual_ids), Some("individual")),
                Some(&gene_names),
            )?;
        } else {
            sc_data.individual_log_rates[ct].to_tsv(&lr_file)?;
        }
        info!("Wrote individual log-rates: {}", lr_file);
    }

    // 8. eQTL ground truth
    let eqtl_file = format!("{}.eqtl_effects.tsv.gz", params.output_prefix);
    let mut writer = open_buf_writer(&eqtl_file)?;
    writeln!(writer, "gene_idx\tgene_id\teqtl_type\tcell_type\tsnp_idx\tsnp_id\tchromosome\tposition\teffect_size")?;

    for (gene_idx, effects_opt) in gene_effects.iter().enumerate() {
        let Some(effects) = effects_opt else { continue };
        let gene_id = &genes.genes[gene_idx].gene_id;

        for (j, &snp_idx) in effects.shared_causal_indices.iter().enumerate() {
            for ct in 0..effects.num_cell_types {
                let effect = effects.shared_effect_sizes[(ct, j)];
                if effect.abs() > 1e-10 {
                    writeln!(
                        writer,
                        "{}\t{}\tshared\t{}\t{}\t{}\t{}\t{}\t{}",
                        gene_idx,
                        gene_id,
                        ct,
                        snp_idx,
                        geno.snp_ids[snp_idx],
                        geno.chromosomes[snp_idx],
                        geno.positions[snp_idx],
                        effect
                    )?;
                }
            }
        }

        for (ct, ct_indices) in effects.independent_causal_indices.iter().enumerate() {
            for (j, &snp_idx) in ct_indices.iter().enumerate() {
                let effect = effects.independent_effect_sizes[(ct, j)];
                if effect.abs() > 1e-10 {
                    writeln!(
                        writer,
                        "{}\t{}\tindependent\t{}\t{}\t{}\t{}\t{}\t{}",
                        gene_idx,
                        gene_id,
                        ct,
                        snp_idx,
                        geno.snp_ids[snp_idx],
                        geno.chromosomes[snp_idx],
                        geno.positions[snp_idx],
                        effect
                    )?;
                }
            }
        }
    }
    writer.flush()?;
    info!("Wrote eQTL effects: {}", eqtl_file);

    Ok(())
}
