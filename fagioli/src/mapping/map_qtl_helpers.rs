//! Data-wrangling helpers for the `map-qtl` pipeline.
//!
//! This module contains individual matching, covariate merging,
//! gene-spec building, and phenotype/genotype extraction.
//! IO routines live in [`crate::io`]; statistical inference lives in
//! `map_qtl.rs` alongside the pipeline orchestration.

use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;

use crate::genotype::GenotypeMatrix;
use crate::mapping::pseudobulk::CollapsedPseudobulk;
use crate::sgvb::BlockFitResultDetailed;
use crate::simulation::GeneAnnotations;
use matrix_param::traits::Inference;

// ── Data structures ──────────────────────────────────────────────────────────

/// Per-gene specification: which pseudobulk row, gene ID, and cis SNP indices.
pub struct GeneSpec {
    pub gene_idx: usize,
    pub gene_id: String,
    pub cis_indices: Vec<usize>,
}

/// Per-gene result from fine-mapping.
pub struct GeneResult {
    pub gene_id: String,
    pub cis_snp_indices: Vec<usize>,
    pub detailed: BlockFitResultDetailed,
    pub z_marginal: DMatrix<f32>,
}

/// Matched individual indices between pseudobulk and genotypes.
pub struct MatchedIndividuals {
    pub pb_indices: Vec<usize>,
    pub geno_indices: Vec<usize>,
}

// ── Individual matching ──────────────────────────────────────────────────────

/// Match individuals between pseudobulk and genotype data by ID.
pub fn match_individuals(pb_ids: &[Box<str>], geno_ids: &[Box<str>]) -> MatchedIndividuals {
    let geno_id_lookup: std::collections::HashMap<&str, usize> = geno_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_ref(), i))
        .collect();

    let mut pb_indices = Vec::new();
    let mut geno_indices = Vec::new();

    for (pb_idx, pb_id) in pb_ids.iter().enumerate() {
        if let Some(&geno_idx) = geno_id_lookup.get(pb_id.as_ref()) {
            pb_indices.push(pb_idx);
            geno_indices.push(geno_idx);
        }
    }

    MatchedIndividuals {
        pb_indices,
        geno_indices,
    }
}

// ── Covariates ───────────────────────────────────────────────────────────────

/// Compute cell-type composition covariates from pseudobulk weights.
/// Returns a centered (n_matched × n_ct) matrix, or None if only 1 cell type.
pub fn compute_composition_covariates(
    collapsed: &CollapsedPseudobulk,
    matched_pb_indices: &[usize],
    n_ct: usize,
) -> Option<DMatrix<f32>> {
    if n_ct <= 1 {
        return None;
    }

    let n = matched_pb_indices.len();
    let mut comp = DMatrix::<f32>::zeros(n, n_ct);

    for (row, &pb_idx) in matched_pb_indices.iter().enumerate() {
        let mut total_weight = 0.0f32;
        for ct_idx in 0..n_ct {
            let w = collapsed.cell_weights[ct_idx][pb_idx];
            comp[(row, ct_idx)] = w;
            total_weight += w;
        }
        if total_weight > 0.0 {
            for ct_idx in 0..n_ct {
                comp[(row, ct_idx)] /= total_weight;
            }
        }
    }

    center_columns(&mut comp);
    Some(comp)
}

/// Horizontally concatenate an optional composition matrix with extra covariate matrices.
pub fn merge_covariates(
    composition: Option<DMatrix<f32>>,
    extra: &[DMatrix<f32>],
    n_matched: usize,
) -> Option<DMatrix<f32>> {
    if composition.is_none() && extra.is_empty() {
        return None;
    }

    let total_cols = composition.as_ref().map_or(0, |c| c.ncols())
        + extra.iter().map(|m| m.ncols()).sum::<usize>();

    if total_cols == 0 {
        return None;
    }

    let mut combined = DMatrix::<f32>::zeros(n_matched, total_cols);
    let mut col_offset = 0;

    if let Some(ref comp) = composition {
        for j in 0..comp.ncols() {
            for i in 0..n_matched {
                combined[(i, col_offset + j)] = comp[(i, j)];
            }
        }
        col_offset += comp.ncols();
    }
    for mat in extra {
        for j in 0..mat.ncols() {
            for i in 0..n_matched {
                combined[(i, col_offset + j)] = mat[(i, j)];
            }
        }
        col_offset += mat.ncols();
    }

    info!("Total covariates: {} columns", total_cols);
    Some(combined)
}

/// Center each column of a matrix to zero mean.
fn center_columns(mat: &mut DMatrix<f32>) {
    let n = mat.nrows() as f32;
    for j in 0..mat.ncols() {
        let mean = mat.column(j).sum() / n;
        for i in 0..mat.nrows() {
            mat[(i, j)] -= mean;
        }
    }
}

// ── Gene specification building ──────────────────────────────────────────────

/// Build gene specs by matching gene annotations to pseudobulk gene names.
/// In cis mode, each gene gets its cis SNP indices from the annotation.
/// In trans mode, all genes get all SNPs.
pub fn build_gene_specs(
    gene_annot: Option<&GeneAnnotations>,
    collapsed: &CollapsedPseudobulk,
    geno: &GenotypeMatrix,
    m_snps: usize,
    n_genes: usize,
) -> Vec<GeneSpec> {
    if let Some(annot) = gene_annot {
        use data_beans::utilities::name_matching::flexible_name_match;

        annot
            .genes
            .iter()
            .enumerate()
            .filter_map(|(ann_idx, gene)| {
                let gene_id_str = match &gene.gene_id {
                    genomic_data::gff::GeneId::Ensembl(s) => s.as_ref(),
                    genomic_data::gff::GeneId::Missing => return None,
                };

                let pb_gene_idx = collapsed.gene_names.iter().position(|pb_name| {
                    flexible_name_match(gene_id_str, pb_name)
                        || gene
                            .gene_name
                            .as_ref()
                            .is_some_and(|gn| flexible_name_match(gn, pb_name))
                })?;

                let cis_indices =
                    annot.cis_snp_indices(ann_idx, &geno.positions, &geno.chromosomes);

                if cis_indices.is_empty() {
                    return None;
                }

                Some(GeneSpec {
                    gene_idx: pb_gene_idx,
                    gene_id: gene_id_str.to_string(),
                    cis_indices,
                })
            })
            .collect()
    } else {
        let all_snp_indices: Vec<usize> = (0..m_snps).collect();
        (0..n_genes)
            .map(|g| GeneSpec {
                gene_idx: g,
                gene_id: collapsed.gene_names[g].to_string(),
                cis_indices: all_snp_indices.clone(),
            })
            .collect()
    }
}

// ── Per-gene phenotype/genotype extraction ───────────────────────────────────

/// Build the Y (log expression) and V (variance) matrices for a single gene
/// across all matched individuals and cell types.
pub fn build_gene_phenotype(
    collapsed: &CollapsedPseudobulk,
    gene_idx: usize,
    matched_pb_indices: &[usize],
    n_ct: usize,
    min_cell_weight: f32,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let n = matched_pb_indices.len();
    let mut y = DMatrix::<f32>::zeros(n, n_ct);
    let mut v = DMatrix::<f32>::zeros(n, n_ct);

    for ct_idx in 0..n_ct {
        let log_mean = collapsed.gamma_params[ct_idx].posterior_log_mean();
        let log_sd = collapsed.gamma_params[ct_idx].posterior_log_sd();

        for (row, &pb_idx) in matched_pb_indices.iter().enumerate() {
            let lm = log_mean[(gene_idx, pb_idx)];
            let ls = log_sd[(gene_idx, pb_idx)];

            if ls > 0.0 && collapsed.cell_weights[ct_idx][pb_idx] >= min_cell_weight {
                y[(row, ct_idx)] = lm;
                v[(row, ct_idx)] = ls * ls;
            } else {
                y[(row, ct_idx)] = 0.0;
                v[(row, ct_idx)] = 1e6;
            }
        }
    }

    (y, v)
}

/// Build the genotype matrix X for a gene's cis SNPs, standardized.
/// Returns (X, valid_cis_indices) after removing zero-variance columns.
pub fn build_cis_genotypes(
    geno: &GenotypeMatrix,
    matched_geno_indices: &[usize],
    cis_indices: &[usize],
) -> Option<(DMatrix<f32>, Vec<usize>)> {
    let n = matched_geno_indices.len();
    let p_cis = cis_indices.len();

    let mut x = DMatrix::<f32>::zeros(n, p_cis);
    for (col, &snp_idx) in cis_indices.iter().enumerate() {
        for (row, &geno_idx) in matched_geno_indices.iter().enumerate() {
            x[(row, col)] = geno.genotypes[(geno_idx, snp_idx)];
        }
    }
    x.scale_columns_inplace();

    let valid_cols: Vec<usize> = (0..p_cis)
        .filter(|&j| x.column(j).iter().any(|&v| v.abs() > 1e-8))
        .collect();

    if valid_cols.len() < 2 {
        return None;
    }

    let x = if valid_cols.len() < p_cis {
        DMatrix::from_fn(n, valid_cols.len(), |i, j| x[(i, valid_cols[j])])
    } else {
        x
    };

    let valid_cis_indices: Vec<usize> = valid_cols.iter().map(|&j| cis_indices[j]).collect();
    Some((x, valid_cis_indices))
}

// ── EB reweighting ───────────────────────────────────────────────────────────

/// Re-weight PIPs and effects using EB weights for a single gene.
pub fn eb_reweight(
    gr: &GeneResult,
    eb_w: &[f32],
    n_ct: usize,
) -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {
    let p_cis = gr.cis_snp_indices.len();

    let mut pip_avg = DMatrix::<f32>::zeros(p_cis, n_ct);
    let mut eff_avg = DMatrix::<f32>::zeros(p_cis, n_ct);
    let mut std_avg = DMatrix::<f32>::zeros(p_cis, n_ct);

    let max_elbo = gr
        .detailed
        .per_prior_elbos
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let gene_weights: Vec<f32> = gr
        .detailed
        .per_prior_elbos
        .iter()
        .enumerate()
        .map(|(v_idx, &e)| eb_w[v_idx] * (e - max_elbo).exp())
        .collect();
    let sum_gw: f32 = gene_weights.iter().sum();

    for (v_idx, &gw) in gene_weights.iter().enumerate() {
        let w = gw / sum_gw;
        pip_avg += &gr.detailed.per_prior_pips[v_idx] * w;
        eff_avg += &gr.detailed.per_prior_effects[v_idx] * w;
        std_avg += &gr.detailed.per_prior_stds[v_idx] * w;
    }

    (pip_avg, eff_avg, std_avg)
}

