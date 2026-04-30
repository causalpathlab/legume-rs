//! Cluster-based annotation: re-aggregates raw counts per cluster (NB-Fisher
//! adjusted), then runs marker-set enrichment on the cluster expression matrix.

use super::args::AnnotateArgs;
use super::inputs::{load_from_manifest, LeidenArgs};
use crate::cluster_aggregation::{accumulate_gene_sum_pair, weighted_mean_profile};
use crate::embed_common::{axis_id_names, Mat};
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;
use enrichment::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs, SpecificityMode};
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn annotate_run(args: &AnnotateArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let leiden_args = LeidenArgs {
        knn: args.knn,
        resolution: args.resolution,
        num_clusters: args.num_clusters,
        min_cluster_size: args.min_cluster_size,
        seed: args.cluster_seed,
    };
    let (loaded, mut manifest, manifest_dir) = load_from_manifest(
        &args.from,
        args.clusters.as_deref(),
        &args.markers,
        args.preload_data,
        &leiden_args,
    )?;

    anyhow::ensure!(
        loaded.n_clusters >= 2,
        "annotate needs ≥ 2 clusters, found {}",
        loaded.n_clusters
    );

    // ----- NB-Fisher per-gene weights -----
    let nb_fisher: Vec<f32> = compute_nb_fisher_weights(loaded.data_vec(), Some(args.block_size))?;
    let (w_min, w_max, w_sum) = nb_fisher.par_iter().map(|&w| (w, w, w)).reduce(
        || (f32::INFINITY, 0.0f32, 0.0f32),
        |(lo, hi, s), (a, b, c)| (lo.min(a), hi.max(b), s + c),
    );
    info!(
        "NB-Fisher weights: min={:.4}, max={:.4}, mean={:.4}",
        w_min,
        w_max,
        w_sum / nb_fisher.len() as f32
    );

    // ----- Fused per-cluster + per-batch gene sums (single zarr sweep) -----
    let g = loaded.gene_names.len();
    let n_clusters = loaded.n_clusters;
    let n_batches = loaded.n_batches;
    let batch_labels_usize: Vec<usize> = loaded.batch_labels.iter().map(|&b| b as usize).collect();
    let (gene_sum_kg, gene_sum_pg) = accumulate_gene_sum_pair(
        loaded.data_vec(),
        &loaded.cluster_labels,
        n_clusters,
        &batch_labels_usize,
        n_batches,
        g,
        args.block_size,
    )?;

    // μ[g, c] = w_NBF[g] · (Σ counts[g, n ∈ c]) / size_sum[c]; Simplex
    // specificity downstream supplies the cross-cluster housekeeping
    // suppression.
    let profile_gk = weighted_mean_profile(&gene_sum_kg, n_clusters, g, &nb_fisher);
    let pb_gene_gp = weighted_mean_profile(&gene_sum_pg, n_batches, g, &nb_fisher);
    let cluster_names = axis_id_names("K", n_clusters);
    info!("Cluster expression: {} genes × {} clusters", g, n_clusters);

    // pb_membership[batch, cluster] = (# cells in batch with cluster id) / batch_size.
    let pb_membership_pk = build_pb_membership(
        &batch_labels_usize,
        &loaded.cluster_labels,
        n_batches,
        n_clusters,
    );

    // One-hot cell membership (N × nClusters).
    let n_cells = loaded.cell_names.len();
    let mut cell_membership_nk = Mat::zeros(n_cells, n_clusters);
    for (n, &c) in loaded.cluster_labels.iter().enumerate() {
        if c < n_clusters {
            cell_membership_nk[(n, c)] = 1.0;
        }
    }

    // Data-aware specificity re-weighting: scale each marker by an empirical
    // specificity score derived from the actual cluster expression matrix.
    // Markers that light up broadly (GZMB across NK + CD8 effector) get
    // attenuated; cluster-exclusive markers (NCAM1 in NK only) keep full
    // weight. This complements the IDF that already runs on the marker TSV.
    let mut markers_gc = loaded.markers_gc.clone();
    if !args.no_empirical_specificity {
        apply_empirical_specificity_weights(&mut markers_gc, &profile_gk);
    }

    let group = GroupInputs {
        profile_gk: profile_gk.clone(),
        pb_gene_gp,
        pb_membership_pk,
        cell_membership_nk,
        gene_names: loaded.gene_names.clone(),
        cell_names: loaded.cell_names.clone(),
    };

    let config = AnnotateConfig {
        specificity: SpecificityMode::Simplex,
        num_row_randomization: args.num_draws,
        num_sample_perm: args.num_perm,
        batch_labels: (n_batches >= 2).then(|| loaded.batch_labels.clone()),
        fdr_alpha: args.fdr_alpha,
        q_softmax_temperature: args.q_temperature,
        min_confidence: args.min_confidence,
        seed: args.seed,
    };

    info!(
        "Running cluster × marker enrichment: {} clusters × {} celltypes, \
         row-rand B={}, sample-perm B={}",
        n_clusters,
        loaded.celltype_names.len(),
        args.num_draws,
        args.num_perm,
    );

    let AnnotateOutputs {
        q_kc,
        es_kc,
        es_restandardized_kc,
        pvalue_kc,
        qvalue_kc,
        cell_annotation_nc,
        argmax_labels,
    } = annotate(&group, &markers_gc, &loaded.celltype_names, &config)?;

    // ----- Outputs -----
    let cell_expr_path = format!("{}.cluster_expression.parquet", args.out);
    profile_gk.to_parquet_with_names(
        &cell_expr_path,
        (Some(&loaded.gene_names), Some("gene")),
        Some(&cluster_names),
    )?;
    info!("wrote {cell_expr_path}");

    let annotation_path = format!("{}.annotation.parquet", args.out);
    cell_annotation_nc.to_parquet_with_names(
        &annotation_path,
        (Some(&loaded.cell_names), Some("cell")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {annotation_path}");

    let argmax_path = format!("{}.argmax.tsv", args.out);
    {
        let mut f = File::create(&argmax_path)?;
        writeln!(f, "cell\tcell_type\tprobability")?;
        for lab in &argmax_labels {
            writeln!(f, "{}\t{}\t{:.4}", lab.cell_name, lab.label, lab.confidence)?;
        }
    }
    info!("wrote {argmax_path}");

    let q_path = format!("{}.cluster_celltype_q.parquet", args.out);
    q_kc.to_parquet_with_names(
        &q_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {q_path}");

    let es_path = format!("{}.cluster_celltype_es.parquet", args.out);
    es_kc.to_parquet_with_names(
        &es_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {es_path}");

    let es_std_path = format!("{}.cluster_celltype_es_std.parquet", args.out);
    es_restandardized_kc.to_parquet_with_names(
        &es_std_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;

    let p_path = format!("{}.cluster_celltype_p.parquet", args.out);
    pvalue_kc.to_parquet_with_names(
        &p_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;

    let q_val_path = format!("{}.cluster_celltype_q_values.parquet", args.out);
    qvalue_kc.to_parquet_with_names(
        &q_val_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;

    display_annotation_histogram(&cell_annotation_nc, &loaded.celltype_names);

    let rel = |abs_path: &str| -> String { rel_to_manifest(&manifest_dir, abs_path) };
    manifest.annotate.annotation = Some(rel(&annotation_path));
    manifest.annotate.argmax = Some(rel(&argmax_path));
    manifest.annotate.cluster_celltype_q = Some(rel(&q_path));
    manifest.annotate.cluster_celltype_es = Some(rel(&es_path));
    manifest.annotate.cluster_expression = Some(rel(&cell_expr_path));
    manifest.annotate.markers = Some(args.markers.to_string());
    manifest.save(Path::new(args.from.as_ref()))?;

    info!("senna annotate complete");
    Ok(())
}

fn rel_to_manifest(manifest_dir: &Path, abs_path: &str) -> String {
    Path::new(abs_path)
        .strip_prefix(manifest_dir)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| abs_path.to_string())
}

/// Multiply each gene's marker entries by an empirical specificity score
/// derived from the cluster expression matrix.
///
/// For gene `g`, score = `(max_c μ[g,c]/Σ_c μ[g,c] − 1/K) / (1 − 1/K)` ∈
/// `[0, 1]`. A gene that fires evenly across all K clusters sits at the
/// uniform floor `1/K`, mapping to score 0. A gene exclusive to one
/// cluster has max simplex value 1, mapping to score 1.
fn apply_empirical_specificity_weights(markers_gc: &mut Mat, profile_gk: &Mat) {
    let g = markers_gc.nrows();
    let c = markers_gc.ncols();
    let k = profile_gk.ncols();
    debug_assert_eq!(profile_gk.nrows(), g);
    if k < 2 {
        return;
    }
    let inv_k = 1.0 / k as f32;
    let denom = (1.0 - inv_k).max(1e-8);

    let scores: Vec<f32> = (0..g)
        .into_par_iter()
        .map(|gi| {
            let row = profile_gk.row(gi);
            let sum: f32 = row.iter().sum();
            if sum <= 1e-12 {
                return 0.0;
            }
            let max = row.iter().fold(0.0f32, |m, &v| m.max(v));
            (((max / sum) - inv_k) / denom).clamp(0.0, 1.0)
        })
        .collect();

    let (mn, mx, sm) = scores
        .iter()
        .fold((f32::INFINITY, 0.0f32, 0.0f32), |(lo, hi, s), &x| {
            (lo.min(x), hi.max(x), s + x)
        });
    info!(
        "Empirical specificity weights: min={:.3}, max={:.3}, mean={:.3}",
        mn,
        mx,
        sm / g as f32
    );

    for gi in 0..g {
        let s = scores[gi];
        for ci in 0..c {
            markers_gc[(gi, ci)] *= s;
        }
    }
}

/// `pb_membership[b, c]` = fraction of batch `b`'s cells assigned to cluster `c`.
/// Rows that observe no cells stay all-zero, which the enrichment crate handles.
fn build_pb_membership(
    batch_labels: &[usize],
    cluster_labels: &[usize],
    n_batches: usize,
    n_clusters: usize,
) -> Mat {
    let mut out = Mat::zeros(n_batches, n_clusters);
    let mut batch_count = vec![0u64; n_batches];
    for (n, &b) in batch_labels.iter().enumerate() {
        if b >= n_batches {
            continue;
        }
        batch_count[b] += 1;
        let c = cluster_labels[n];
        if c < n_clusters {
            out[(b, c)] += 1.0;
        }
    }
    for b in 0..n_batches {
        let s = batch_count[b].max(1) as f32;
        for c in 0..n_clusters {
            out[(b, c)] /= s;
        }
    }
    out
}

fn display_annotation_histogram(annot: &Mat, annot_names: &[Box<str>]) {
    let n_cells = annot.nrows();
    let n_types = annot.ncols();

    // Per-cell argmax: rows are independent, so this fans out cleanly.
    let per_cell: Vec<(f32, Option<usize>)> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let row = annot.row(i);
            let sum: f32 = row.iter().sum();
            if sum < 1e-12 {
                return (0.0, None);
            }
            let (idx, val) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (*val, Some(idx))
        })
        .collect();

    let mut type_counts = vec![0usize; n_types];
    let mut type_prob_sum = vec![0.0f32; n_types];
    let mut unassigned = 0usize;
    for &(prob, assign) in &per_cell {
        match assign {
            Some(c) => {
                type_counts[c] += 1;
                type_prob_sum[c] += prob;
            }
            None => unassigned += 1,
        }
    }
    let mut sorted_types: Vec<usize> = (0..n_types).collect();
    sorted_types.sort_by(|&a, &b| type_counts[b].cmp(&type_counts[a]));

    let max_count = *type_counts.iter().max().unwrap_or(&1).max(&unassigned);
    const MAX_BAR: usize = 20;

    let assigned_cells = n_cells - unassigned;
    let assigned_prob_sum: f32 = per_cell
        .iter()
        .filter_map(|(p, a)| a.map(|_| *p))
        .sum::<f32>();
    let mean_prob = if assigned_cells > 0 {
        assigned_prob_sum / assigned_cells as f32
    } else {
        0.0
    };
    let above_50 = per_cell.iter().filter(|(p, _)| *p > 0.5).count();
    let above_70 = per_cell.iter().filter(|(p, _)| *p > 0.7).count();

    eprintln!();
    eprintln!("Annotation Summary ({n_cells} cells)");
    eprintln!(
        "  Mean max-prob (assigned): {:.3}  >0.5: {} ({:.1}%)  >0.7: {} ({:.1}%)",
        mean_prob,
        above_50,
        100.0 * above_50 as f32 / n_cells as f32,
        above_70,
        100.0 * above_70 as f32 / n_cells as f32
    );
    if unassigned > 0 {
        let bar_len = (unassigned * MAX_BAR) / max_count.max(1);
        eprintln!(
            "  {:24} {:5} ({:5.1}%)      {}",
            "unassigned",
            unassigned,
            100.0 * unassigned as f32 / n_cells as f32,
            "▒".repeat(bar_len)
        );
    }
    eprintln!();

    for &ct in &sorted_types {
        if type_counts[ct] == 0 {
            continue;
        }
        let bar_len = (type_counts[ct] * MAX_BAR) / max_count.max(1);
        let bar: String = "█".repeat(bar_len);
        let avg_prob = type_prob_sum[ct] / type_counts[ct] as f32;
        eprintln!(
            "  {:24} {:5} ({:5.1}%) {:.2} {}",
            annot_names[ct],
            type_counts[ct],
            100.0 * type_counts[ct] as f32 / n_cells as f32,
            avg_prob,
            bar
        );
    }
    eprintln!();
}
