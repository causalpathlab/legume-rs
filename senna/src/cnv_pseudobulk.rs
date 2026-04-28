//! Cluster/topic-informed pseudobulk CNV detection.
//!
//! Shared across SVD, topic, and indexed-topic pipelines:
//! 1. Cell-type membership from K-means (SVD) or topic proportions (topic/itopic).
//! 2. Pseudobulk via `collapse_pseudobulk()` → per-(cell_type, individual) `GammaMatrix`.
//! 3. Within each cell type: log-ratio vs cross-individual mean → CNV signal.
//! 4. Hand to [`cnv::per_sample::call_per_sample_cnv`] for per-topic HMM-EM
//!    on the stacked `[G × (cell_types · individuals)]` log-ratio matrix.

use log::info;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap as HashMap;

use auxiliary_data::cell_annotations::{CellAnnotations, CellTypeMembership};
use cnv::genome_order::GenePosition;
use cnv::per_sample::{call_per_sample_cnv, PerSampleCnv, PerSampleCnvConfig};
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
    cfg: &PerSampleCnvConfig,
) -> anyhow::Result<PerSampleCnv> {
    let n_cells = latent_nk.nrows();

    info!("CNV: clustering {n_cells} cells into {n_clusters} groups");
    let cluster_labels = latent_nk.kmeans_rows(KmeansArgs::with_clusters(n_clusters));

    let mut membership_matrix = DMatrix::<f32>::zeros(n_cells, n_clusters);
    for (i, &c) in cluster_labels.iter().enumerate() {
        membership_matrix[(i, c)] = 1.0;
    }
    let cell_type_names: Vec<Box<str>> = (0..n_clusters).map(|c| format!("K{c}").into()).collect();
    let membership = CellTypeMembership {
        matrix: membership_matrix,
        cell_type_names,
    };

    detect_cnv_with_membership(data_vec, &membership, batch_membership, gene_positions, cfg)
}

/// CNV detection using topic proportions as soft cell-type membership.
///
/// `topic_proportions` should be on the probability simplex.
pub fn detect_cnv_topic_informed(
    data_vec: SparseIoVec,
    topic_proportions: &Mat,
    batch_membership: &[Box<str>],
    gene_positions: &[GenePosition],
    cfg: &PerSampleCnvConfig,
) -> anyhow::Result<PerSampleCnv> {
    let n_topics = topic_proportions.ncols();
    let cell_type_names: Vec<Box<str>> = (0..n_topics).map(|k| format!("T{k}").into()).collect();
    let membership = CellTypeMembership {
        matrix: topic_proportions.clone(),
        cell_type_names,
    };

    detect_cnv_with_membership(data_vec, &membership, batch_membership, gene_positions, cfg)
}

// ---------------------------------------------------------------------------
// Core shared logic
// ---------------------------------------------------------------------------

/// Pseudobulk by `(cell_type, individual)` → per-cell-type log-ratio against
/// the cross-individual mean → stack into `[G × (n_cell_types · n_individuals)]`
/// signal → run [`call_per_sample_cnv`].
fn detect_cnv_with_membership(
    data_vec: SparseIoVec,
    membership: &CellTypeMembership,
    batch_membership: &[Box<str>],
    gene_positions: &[GenePosition],
    cfg: &PerSampleCnvConfig,
) -> anyhow::Result<PerSampleCnv> {
    let column_names = data_vec.column_names()?;
    let annotations = build_cell_annotations(batch_membership, &column_names);
    let n_individuals = annotations.individual_ids.len();
    let n_cell_types = membership.cell_type_names.len();

    info!("CNV: pseudobulk with {n_cell_types} cell types × {n_individuals} individuals");

    let collapsed = collapse_pseudobulk(data_vec, &annotations, membership, 1.0, 1.0)?;
    let n_genes = collapsed.gene_names.len();

    let total_cols = n_cell_types * n_individuals;
    let mut signal = DMatrix::<f32>::zeros(n_genes, total_cols);

    for (ct_idx, gamma) in collapsed.gamma_params.iter().enumerate() {
        let log_mean = gamma.posterior_log_mean();
        let weights = &collapsed.cell_weights[ct_idx];

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
        if total_weight <= 0.0 {
            log::warn!(
                "CNV: cell type {} has no fully-weighted individuals; emitting zero signal block",
                membership.cell_type_names[ct_idx]
            );
            continue;
        }
        for v in &mut mean_g {
            *v /= total_weight;
        }

        let col_start = ct_idx * n_individuals;
        for i in 0..n_individuals {
            let col = col_start + i;
            for g in 0..n_genes {
                signal[(g, col)] = log_mean[(g, i)] - mean_g[g];
            }
        }
    }

    info!("CNV: signal {n_genes} genes × {total_cols} samples");
    call_per_sample_cnv(&signal, gene_positions, n_cell_types, n_individuals, cfg)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load gene positions from `--gff` or `--cnv-ground-truth`.
pub fn load_gene_positions(
    cnv_args: &CnvArgs,
    gene_names: &[Box<str>],
) -> anyhow::Result<Option<Vec<GenePosition>>> {
    if let Some(gt_path) = &cnv_args.cnv_ground_truth {
        info!("CNV: loading positions from {gt_path}");
        Ok(Some(cnv::genome_order::read_gene_positions_from_tsv(
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

/// Build a [`PerSampleCnvConfig`] from shared CLI args.
pub fn build_cnv_config(cnv_args: &CnvArgs) -> PerSampleCnvConfig {
    PerSampleCnvConfig {
        n_states: cnv_args.cnv_states,
        gmm_k_max: cnv_args.cnv_gmm_k_max,
        ..PerSampleCnvConfig::default()
    }
}

/// Write CNV detection results: `[G_ord × N]` Viterbi states and continuous
/// cn_score in `[−1, 1]` to parquet.
pub fn write_cnv_results(
    result: &PerSampleCnv,
    out_prefix: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    let ordered_gene_names: Vec<Box<str>> = result
        .genome_order
        .ordered_indices
        .iter()
        .map(|&i| gene_names[i].clone())
        .collect();

    let g = ordered_gene_names.len();
    let n = result.cn_score.ncols();
    let mut states = DMatrix::<f32>::zeros(g, n);
    for (s, path) in result.viterbi_paths.iter().enumerate() {
        for (gi, &v) in path.iter().enumerate() {
            states[(gi, s)] = v as f32;
        }
    }
    states.to_parquet_with_names(
        &(out_prefix.to_string() + ".cnv.states.parquet"),
        (Some(&ordered_gene_names), Some("gene")),
        None,
    )?;
    result.cn_score.to_parquet_with_names(
        &(out_prefix.to_string() + ".cnv.cn_score.parquet"),
        (Some(&ordered_gene_names), Some("gene")),
        None,
    )?;
    info!(
        "CNV outputs: {pref}.cnv.states.parquet, {pref}.cnv.cn_score.parquet",
        pref = out_prefix
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
