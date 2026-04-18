//! Cluster/topic-informed pseudobulk CNV detection.
//!
//! Shared across SVD, topic, and indexed-topic pipelines:
//! 1. Cell-type membership from K-means (SVD) or topic proportions (topic/itopic)
//! 2. Pseudobulk via `collapse_pseudobulk()` → per-(cell_type, individual) `GammaMatrix`
//! 3. Within each cell type: log-ratio vs cross-individual mean → CNV signal
//! 4. Feed stacked log-ratios to factorial tree for CNV calling

use log::info;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap as HashMap;

use auxiliary_data::cell_annotations::{CellAnnotations, CellTypeMembership};
use cnv::detect::{CnvDetectConfig, CnvDetectResult};
use cnv::genome_order::GenePosition;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::pseudobulk::collapse_pseudobulk;
use matrix_param::traits::Inference;
use matrix_util::clustering::{Kmeans, KmeansArgs};

use matrix_util::traits::IoOps;

use crate::embed_common::*;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// CNV detection using K-means clusters on SVD latent embeddings.
pub fn detect_cnv_cluster_informed(
    data_vec: SparseIoVec,
    latent_nk: &Mat,
    batch_membership: &[Box<str>],
    gene_positions: &[GenePosition],
    n_clusters: usize,
    cnv_config: &CnvDetectConfig,
) -> anyhow::Result<CnvDetectResult> {
    let n_cells = latent_nk.nrows();

    info!("CNV: clustering {n_cells} cells into {n_clusters} groups");
    let cluster_labels = latent_nk.kmeans_rows(KmeansArgs::with_clusters(n_clusters));

    // One-hot membership from cluster labels
    let mut membership_matrix = DMatrix::<f32>::zeros(n_cells, n_clusters);
    for (i, &c) in cluster_labels.iter().enumerate() {
        membership_matrix[(i, c)] = 1.0;
    }
    let cell_type_names: Vec<Box<str>> = (0..n_clusters)
        .map(|c| format!("cluster_{c}").into())
        .collect();
    let membership = CellTypeMembership {
        matrix: membership_matrix,
        cell_type_names,
    };

    detect_cnv_with_membership(
        data_vec,
        &membership,
        batch_membership,
        gene_positions,
        cnv_config,
    )
}

/// CNV detection using topic proportions as soft cell-type membership.
///
/// `topic_proportions` should be on the probability simplex (e.g., `exp(log_softmax_z_nk)`).
pub fn detect_cnv_topic_informed(
    data_vec: SparseIoVec,
    topic_proportions: &Mat,
    batch_membership: &[Box<str>],
    gene_positions: &[GenePosition],
    cnv_config: &CnvDetectConfig,
) -> anyhow::Result<CnvDetectResult> {
    let n_topics = topic_proportions.ncols();

    let cell_type_names: Vec<Box<str>> =
        (0..n_topics).map(|k| format!("topic_{k}").into()).collect();
    let membership = CellTypeMembership {
        matrix: topic_proportions.clone(),
        cell_type_names,
    };

    detect_cnv_with_membership(
        data_vec,
        &membership,
        batch_membership,
        gene_positions,
        cnv_config,
    )
}

// ---------------------------------------------------------------------------
// Core shared logic
// ---------------------------------------------------------------------------

/// Core CNV detection: pseudobulk by (`cell_type`, individual) → log-ratio → factorial tree.
fn detect_cnv_with_membership(
    data_vec: SparseIoVec,
    membership: &CellTypeMembership,
    batch_membership: &[Box<str>],
    gene_positions: &[GenePosition],
    cnv_config: &CnvDetectConfig,
) -> anyhow::Result<CnvDetectResult> {
    let column_names = data_vec.column_names()?;
    let annotations = build_cell_annotations(batch_membership, &column_names);
    let n_individuals = annotations.individual_ids.len();
    let n_cell_types = membership.cell_type_names.len();

    info!("CNV: pseudobulk with {n_cell_types} cell types × {n_individuals} individuals",);

    let collapsed = collapse_pseudobulk(data_vec, &annotations, membership, 1.0, 1.0)?;

    // Within each cell type: log-ratio vs cross-individual mean
    let n_genes = collapsed.gene_names.len();
    let mut log_ratio_columns: Vec<Vec<f32>> = Vec::new();

    for (ct_idx, gamma) in collapsed.gamma_params.iter().enumerate() {
        let log_mean = gamma.posterior_log_mean();
        let weights = &collapsed.cell_weights[ct_idx];

        // Weighted mean across individuals
        let mut mean_g = vec![0.0f32; n_genes];
        let mut total_weight = 0.0f32;
        for i in 0..n_individuals {
            if weights[i] < 1.0 {
                continue;
            }
            total_weight += weights[i];
            for g in 0..n_genes {
                mean_g[g] += weights[i] * log_mean[(g, i)];
            }
        }
        if total_weight > 0.0 {
            for v in &mut mean_g {
                *v /= total_weight;
            }
        }

        for i in 0..n_individuals {
            if weights[i] < 1.0 {
                continue;
            }
            let col: Vec<f32> = (0..n_genes).map(|g| log_mean[(g, i)] - mean_g[g]).collect();
            log_ratio_columns.push(col);
        }
    }

    if log_ratio_columns.is_empty() {
        anyhow::bail!("No valid (cell_type, individual) pairs for CNV detection");
    }

    let valid_count = log_ratio_columns.len();
    let mut signal = DMatrix::<f32>::zeros(n_genes, valid_count);
    for (j, col) in log_ratio_columns.iter().enumerate() {
        for g in 0..n_genes {
            signal[(g, j)] = col[g];
        }
    }

    info!("CNV: log-ratio signal {n_genes} genes × {valid_count} samples",);

    cnv::detect::detect_cnv(&signal, gene_positions, cnv_config)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load gene positions from --gff or --cnv-ground-truth.
pub fn load_gene_positions(
    cnv_args: &CnvArgs,
    gene_names: &[Box<str>],
) -> anyhow::Result<Option<Vec<GenePosition>>> {
    if let Some(gt_path) = &cnv_args.cnv_ground_truth {
        info!("CNV: loading positions from {gt_path}");
        Ok(Some(cnv::detect::read_gene_positions_from_cnv_tsv(
            gt_path,
        )?))
    } else if let Some(gff_path) = &cnv_args.gff {
        info!("CNV: loading gene annotations from {gff_path}");
        let gene_tss = genomic_data::coordinates::load_gene_tss(gff_path, gene_names)?;
        Ok(Some(
            gene_tss
                .iter()
                .enumerate()
                .filter_map(|(idx, tss)| {
                    tss.as_ref().map(|t| GenePosition {
                        gene_idx: idx,
                        chromosome: t.chr.clone(),
                        position: t.tss as u64,
                    })
                })
                .collect(),
        ))
    } else {
        Ok(None)
    }
}

/// Reconstruct per-cell batch labels from a `SparseIoVec`.
///
/// Returns `None` if no batch information is available.
pub fn reconstruct_batch_labels(data_vec: &SparseIoVec) -> Option<Vec<Box<str>>> {
    let batch_names = data_vec.batch_names()?;
    let n_cells = data_vec.num_columns();
    let batch_indices = data_vec.get_batch_membership(0..n_cells);
    Some(
        batch_indices
            .iter()
            .map(|&b| batch_names[b].clone())
            .collect(),
    )
}

/// Build a `CnvDetectConfig` from shared CLI args.
pub fn build_cnv_config(cnv_args: &CnvArgs) -> CnvDetectConfig {
    CnvDetectConfig {
        corr_thresholds: cnv_args.cnv_corr_thresholds.clone(),
        factorial: cnv::factorial_tree::FactorialTreeConfig {
            n_factors: cnv_args.cnv_factors,
            n_states: cnv_args.cnv_states,
            n_iter: cnv_args.cnv_iter,
            warmup: cnv_args.cnv_warmup,
            ..Default::default()
        },
    }
}

/// Write CNV detection results (profiles + loadings) to parquet.
pub fn write_cnv_results(
    cnv_result: &CnvDetectResult,
    out_prefix: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    let ordered_gene_names: Vec<Box<str>> = cnv_result
        .genome_order
        .ordered_indices
        .iter()
        .map(|&i| gene_names[i].clone())
        .collect();

    cnv_result.gene_factor_profiles.to_parquet_with_names(
        &(out_prefix.to_string() + ".cnv_profiles.parquet"),
        (Some(&ordered_gene_names), Some("gene")),
        None,
    )?;

    cnv_result.factorial_result.loadings.to_parquet_with_names(
        &(out_prefix.to_string() + ".cnv_loadings.parquet"),
        (None, None),
        None,
    )?;

    info!(
        "CNV detection: {} factors, {} blocks → {}.cnv_profiles.parquet",
        cnv_result.factorial_result.factor_emission_means.len(),
        cnv_result.coarsening.num_blocks(),
        out_prefix,
    );
    Ok(())
}

/// Build `CellAnnotations` from per-cell batch labels (same pattern as cocoa).
fn build_cell_annotations(batch_labels: &[Box<str>], column_names: &[Box<str>]) -> CellAnnotations {
    let mut indv_set: Vec<Box<str>> = batch_labels
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    indv_set.retain(|s| !s.is_empty() && s.as_ref() != "NA");

    let indv_to_idx: HashMap<Box<str>, usize> = indv_set
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let cell_to_individual: HashMap<Box<str>, usize> = column_names
        .iter()
        .zip(batch_labels.iter())
        .filter_map(|(cell_name, batch_name)| {
            indv_to_idx
                .get(batch_name)
                .map(|&idx| (cell_name.clone(), idx))
        })
        .collect();

    CellAnnotations {
        cell_to_individual,
        individual_ids: indv_set,
    }
}
