//! Bayesian hierarchical clustering over hard cluster labels.
//!
//! Postprocess after `senna cluster`: aggregates raw counts per cluster to form
//! the per-gene sufficient stats `T_{k,g} = Σ_{n ∈ k} y_{n,g}`, cluster size
//! `S_k = |{n : label_n = k}|`, then defers to `data_beans_alg::bhc` for the
//! Dirichlet-Multinomial Bayes-factor merge tree + consensus cut.
//!
//! This is the exact recipe pinto uses for link communities, just over cell
//! clusters instead of edge communities. Hard labels mean no aggregation
//! ambiguity and no ESS rescaling: the effective sample size IS the cluster
//! size.
//!
//! Outputs:
//!   `{out}.bhc.merges.parquet` — merge tree (merge_id, left, right, log_bf, n_cells)
//!   `{out}.bhc.cut.parquet`    — consensus id per original cluster (−1 = empty)

use crate::cluster_aggregation::accumulate_gene_sum;
use crate::embed_common::*;
use data_beans_alg::bhc::{bhc_cut, bhc_merge, BhcInput, BhcMerge};

/// Runtime configuration for cluster BHC.
pub struct ClusterBhcConfig {
    /// Per-gene prior strength. Total Dirichlet concentration is
    /// `gamma = gamma_per_gene · G`. Default `1.0` = Bayes-Laplace (one
    /// pseudo-count per gene). This makes the prior strength invariant to
    /// feature-space dimension so the default works for M in [50, 20000].
    pub gamma_per_gene: f64,
    /// `log_bf ≥ cutoff` → merge. `0.0` is the natural Bayesian break point.
    pub cutoff: f64,
    /// Cells per CSC read block for the sufficient-stat sweep.
    pub block_size: usize,
}

/// Run BHC over the `k` clusters and write merges + consensus cut.
///
/// `labels[n]` is the cluster id for cell n, or `usize::MAX` for unassigned
/// (those cells are skipped). `cluster_sizes[k]` must agree with the same
/// `label < k` filter applied to `labels`. `data_vec` is read in column blocks.
pub(crate) fn run_cluster_bhc(
    data_vec: &SparseIoVec,
    labels: &[usize],
    cluster_sizes: &[usize],
    out_prefix: &str,
    cfg: &ClusterBhcConfig,
) -> anyhow::Result<()> {
    let n = labels.len();
    let k = cluster_sizes.len();
    let m = data_vec.num_rows();
    anyhow::ensure!(
        n == data_vec.num_columns(),
        "BHC: labels has {} cells but data has {}",
        n,
        data_vec.num_columns()
    );
    if k < 2 {
        info!("BHC: only {k} cluster(s); skipping");
        return Ok(());
    }

    let gene_sum = accumulate_gene_sum(data_vec, labels, k, m, cfg.block_size)?;
    let size_sum: Vec<f64> = (0..k)
        .map(|kk| gene_sum[kk * m..(kk + 1) * m].iter().sum::<f64>())
        .collect();

    let active = cluster_sizes.iter().filter(|&&s| s > 0).count();
    let gamma = cfg.gamma_per_gene * m as f64;
    info!(
        "BHC: cluster sizes = [{}] (active {}/{}, γ={:.2} = {:.3}·G)",
        cluster_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        active,
        k,
        gamma,
        cfg.gamma_per_gene
    );

    let merges = bhc_merge(
        BhcInput {
            k,
            m,
            gene_sum: &gene_sum,
            size_sum: &size_sum,
            effective_size: cluster_sizes,
        },
        gamma,
    );
    let consensus = bhc_cut(&merges, k, cfg.cutoff);
    let n_consensus = consensus
        .iter()
        .filter(|&&v| v >= 0)
        .map(|&v| v as usize)
        .max()
        .map_or(0, |max_id| max_id + 1);
    info!(
        "BHC cut (log_bf ≥ {:.3}): {k} clusters → {n_consensus} consensus clusters",
        cfg.cutoff
    );

    write_bhc_merges(&format!("{out_prefix}.bhc.merges.parquet"), &merges)?;
    write_bhc_cut(&format!("{out_prefix}.bhc.cut.parquet"), &consensus)?;
    Ok(())
}

fn write_bhc_merges(file_path: &str, merges: &[BhcMerge]) -> anyhow::Result<()> {
    let n_rows = merges.len();
    let mut mat = Mat::zeros(n_rows, 5);
    for (i, m) in merges.iter().enumerate() {
        mat[(i, 0)] = m.id as f32;
        mat[(i, 1)] = m.left as f32;
        mat[(i, 2)] = m.right as f32;
        mat[(i, 3)] = m.log_bf as f32;
        mat[(i, 4)] = m.n_samples as f32;
    }
    let cols: Vec<Box<str>> = vec![
        "merge_id".into(),
        "left".into(),
        "right".into(),
        "log_bf".into(),
        "n_cells".into(),
    ];
    let rows: Vec<Box<str>> = (0..n_rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    mat.to_parquet_with_names(file_path, (Some(&rows), Some("step")), Some(&cols))?;
    Ok(())
}

fn write_bhc_cut(file_path: &str, labels: &[i32]) -> anyhow::Result<()> {
    let n_rows = labels.len();
    let mut mat = Mat::zeros(n_rows, 2);
    for (i, &lab) in labels.iter().enumerate() {
        mat[(i, 0)] = i as f32;
        mat[(i, 1)] = lab as f32;
    }
    let cols: Vec<Box<str>> = vec!["cluster".into(), "consensus".into()];
    let rows: Vec<Box<str>> = (0..n_rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    mat.to_parquet_with_names(file_path, (Some(&rows), Some("row")), Some(&cols))?;
    Ok(())
}
