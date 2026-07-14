//! Cluster-based annotation: re-aggregates raw counts per cluster (NB-Fisher
//! adjusted), then runs marker-set enrichment on the cluster expression matrix.

use super::args::AnnotateArgs;
use super::finalize::{
    clean_outputs, finalize_annotation, AnnotationArtifacts, ENRICHMENT_OUTPUT_SUFFIXES,
};
use super::inputs::{load_from_manifest, LeidenArgs};
use crate::cluster_aggregation::{
    accumulate_gene_sum, accumulate_gene_sum_pair, weighted_mean_profile,
};
use crate::embed_common::{axis_id_names, Mat};
use crate::run_manifest;
use data_beans_alg::gene_weighting::{compute_nb_fisher_weights, load_fisher_weights};
use enrichment::consensus::{Abstain, UNASSIGNED};
use enrichment::marker_bootstrap::{ClusterBootstrap, EnrichmentBootstrapConfig};
use enrichment::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs, SpecificityMode};
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use std::path::Path;

pub fn run(args: &AnnotateArgs) -> anyhow::Result<()> {
    let out: Box<str> = match args.out.as_deref() {
        Some(o) => Box::from(o),
        None => crate::run_manifest::derive_out_prefix(&args.from).into_boxed_str(),
    };
    mkdir_parent(&out)?;
    // Exactly one gene-set source. --markers → curated cell-type annotation
    // (+ optional inline CL ontology via --obo/--label-cl); --gaf/--gmt → ontology
    // gene-set mode (cross-cluster-contrasted module-score signature per cluster).
    let ontology_mode = args.gaf.is_some() || args.gmt.is_some();
    let n_sources = [
        !args.markers.is_empty(),
        args.gaf.is_some(),
        args.gmt.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count();
    anyhow::ensure!(
        n_sources == 1,
        "exactly one gene-set source required: --markers, --gaf, or --gmt (got {n_sources})"
    );
    if ontology_mode {
        anyhow::ensure!(
            args.obo.is_some(),
            "--gaf/--gmt require --obo (resolves GO term ids to names in the signature)"
        );
        anyhow::ensure!(
            args.label_cl.is_none(),
            "--label-cl is for --markers (curated CL); GO/GMT term ids are ontology ids already"
        );
    } else {
        anyhow::ensure!(
            args.obo.is_some() == args.label_cl.is_some(),
            "--obo and --label-cl must be given together to run inline ontology annotation \
             (got only one); omit both to skip it"
        );
    }
    if !args.no_clean {
        clean_outputs(&out, ENRICHMENT_OUTPUT_SUFFIXES);
    }

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

    ////////////////////////////////
    // NB-Fisher per-gene weights //
    ////////////////////////////////
    // Try the cached parquet from training first; fall back to recomputing.
    let fisher_prefix = run_manifest::resolve(&manifest_dir, &manifest.prefix)
        .to_string_lossy()
        .into_owned();
    let nb_fisher: Vec<f32> = match load_fisher_weights(&fisher_prefix)? {
        Some((cached_genes, cached_w)) if cached_genes == loaded.gene_names => {
            info!(
                "Loaded {} NB-Fisher weights from {fisher_prefix}.fisher_weights.parquet",
                cached_w.len()
            );
            cached_w
        }
        Some((cached_genes, _)) => {
            info!(
                "Cached fisher_weights gene names ({}) don't match data ({}); recomputing",
                cached_genes.len(),
                loaded.gene_names.len()
            );
            compute_nb_fisher_weights(loaded.data_vec(), Some(args.block_size))?
        }
        None => compute_nb_fisher_weights(loaded.data_vec(), Some(args.block_size))?,
    };
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

    ///////////////////////////
    // Per-cluster gene sums //
    ///////////////////////////
    // The marker path additionally needs a per-batch axis (its sample-
    // permutation null), accumulated in the same fused sweep. The GO/GMT
    // ontology path scores the per-cluster profile directly, so it needs only
    // the cluster sums.
    let g = loaded.gene_names.len();
    let n_clusters = loaded.n_clusters;
    let n_batches = loaded.n_batches;
    let batch_labels_usize: Vec<usize> = loaded.batch_labels.iter().map(|&b| b as usize).collect();
    let (gene_sum_kg, gene_sum_b_opt): (Vec<f64>, Option<Vec<f64>>) = if ontology_mode {
        (
            accumulate_gene_sum(
                loaded.data_vec(),
                &loaded.cluster_labels,
                n_clusters,
                g,
                args.block_size,
            )?,
            None,
        )
    } else {
        let (kg, b) = accumulate_gene_sum_pair(
            loaded.data_vec(),
            &loaded.cluster_labels,
            n_clusters,
            &batch_labels_usize,
            n_batches,
            g,
            args.block_size,
        )?;
        (kg, Some(b))
    };

    // μ[g, c] = w_NBF[g] · (Σ counts[g, n ∈ c]) / size_sum[c]; Simplex
    // specificity downstream supplies the cross-cluster housekeeping
    // suppression.
    let profile_gk = weighted_mean_profile(&gene_sum_kg, n_clusters, g, &nb_fisher);
    let cluster_names = axis_id_names("K", n_clusters);
    info!("Cluster expression: {g} genes × {n_clusters} clusters");
    let profile_max = profile_gk.iter().fold(0f32, |m, &v| m.max(v));
    if profile_max <= 1e-12 {
        anyhow::bail!(
            "Cluster expression matrix is all zero — every cell-axis cluster_label is \
             out of range (>= n_clusters). Check the cluster file's barcodes match the \
             data backend, or that the manifest's `clusters` path resolves correctly."
        );
    }

    ///////////////////////////////////
    // GO/GMT ontology gene-set mode //
    ///////////////////////////////////
    // Descriptive module-score signature on the cluster profile (no cell-level
    // labels, no permutation, no tree). Diverges from the marker path entirely.
    if ontology_mode {
        return run_ontology_gene_sets(
            args,
            &out,
            &profile_gk,
            &loaded.gene_names,
            &cluster_names,
            &mut manifest,
            &manifest_dir,
        );
    }

    // marker path: the second axis was per-batch.
    let gene_sum_pg = gene_sum_b_opt.expect("marker path accumulates the per-batch axis");

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

    // Per-batch β̃ profile — only the marker (sample-permutation) path needs it,
    // so it's built after the GO/GMT early-return above.
    let pb_gene_gp = weighted_mean_profile(&gene_sum_pg, n_batches, g, &nb_fisher);
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
        // pb_membership_pk's rows ARE batches (one pseudobulk per batch),
        // so the sample-permutation null shuffles batches directly with no
        // inner stratification. Cell-level labels would be the wrong length
        // (caused a panic in `permute_indices`).
        batch_labels: None,
        fdr_alpha: args.fdr_alpha,
        q_softmax_temperature: args.q_temperature,
        min_confidence: args.min_confidence,
        seed: args.seed,
        // ON by default, as in `faba annotate`. A single pass over one marker panel always
        // returns a winner, and returns it with a softmaxed `confidence` that says nothing
        // about whether the panel could have said otherwise.
        bootstrap: (!args.no_bootstrap_markers && args.n_boot > 0).then_some(
            EnrichmentBootstrapConfig {
                n_boot: args.n_boot,
                abstain: if args.abstain_separable {
                    Abstain::Separable(args.abstain_alpha)
                } else {
                    Abstain::Support(args.min_support)
                },
                set_coverage: args.set_coverage,
                max_set_size: args.max_set_size,
                boot_num_draws: args.boot_num_draws,
            },
        ),
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
        perm_z_kc,
        pvalue_kc,
        qvalue_kc,
        cell_annotation_nc,
        argmax_labels,
        bootstrap,
    } = annotate(&group, &markers_gc, &loaded.celltype_names, &config)?;

    /////////////
    // Outputs //
    /////////////
    let cell_expr_path = format!("{out}.cluster_expression.parquet");
    profile_gk.to_parquet_with_names(
        &cell_expr_path,
        (Some(&loaded.gene_names), Some("gene")),
        Some(&cluster_names),
    )?;
    info!("wrote {cell_expr_path}");

    let annotation_path = format!("{out}.annotation.parquet");
    cell_annotation_nc.to_parquet_with_names(
        &annotation_path,
        (Some(&loaded.cell_names), Some("cell")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {annotation_path}");

    // Per-cell label files via the SHARED writer (also emits `membership.tsv`),
    // so this pass and `annotate-by-projection` produce an identical contract.
    let argmax_path = format!("{out}.argmax.tsv");
    {
        let cells: Vec<Box<str>> = argmax_labels
            .iter()
            .map(|l| Box::from(l.cell_name.as_ref()))
            .collect();
        let labels: Vec<Box<str>> = argmax_labels
            .iter()
            .map(|l| Box::from(l.label.as_ref()))
            .collect();
        let probs: Vec<f32> = argmax_labels.iter().map(|l| l.confidence).collect();
        graph_embedding_util::type_annotation::write_label_tsvs(&out, &cells, &labels, &probs)?;
    }

    let q_path = format!("{out}.cluster_celltype_q.parquet");
    q_kc.to_parquet_with_names(
        &q_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {q_path}");

    let es_path = format!("{out}.cluster_celltype_es.parquet");
    es_kc.to_parquet_with_names(
        &es_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {es_path}");

    let es_std_path = format!("{out}.cluster_celltype_es_std.parquet");
    es_restandardized_kc.to_parquet_with_names(
        &es_std_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;

    // Correlation-preserving sample-permutation z (when num_perm > 0): the
    // preferred ontology input — graded, unlike the pooled p-value.
    if let Some(pz) = &perm_z_kc {
        let perm_z_path = format!("{out}.cluster_celltype_perm_z.parquet");
        pz.to_parquet_with_names(
            &perm_z_path,
            (Some(&cluster_names), Some("cluster")),
            Some(&loaded.celltype_names),
        )?;
    }

    let p_path = format!("{out}.cluster_celltype_p.parquet");
    pvalue_kc.to_parquet_with_names(
        &p_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;

    let q_val_path = format!("{out}.cluster_celltype_q_values.parquet");
    qvalue_kc.to_parquet_with_names(
        &q_val_path,
        (Some(&cluster_names), Some("cluster")),
        Some(&loaded.celltype_names),
    )?;

    // The bootstrap's own artifacts. The K x C matrices above are NOT withheld under the
    // bootstrap — they are a different granularity in a different file, and the ontology layer
    // below consumes them. What the bootstrap replaces is the per-CELL story: `argmax_labels`
    // now carries `cluster_label_support` instead of a softmaxed test statistic, and
    // `cell_annotation_nc` is the consensus distribution. Both were swapped inside `annotate`.
    if let Some(boot) = &bootstrap {
        write_bootstrap_outputs(&out, boot, &cluster_names, &loaded.celltype_names)?;
    }

    display_annotation_histogram(&cell_annotation_nc, &loaded.celltype_names);

    // Optional inline ontology annotation (TreeBH) — reuses the freshly computed
    // restandardized-ES z-matrix in memory, no parquet round-trip. NON-FATAL: a
    // bad label→CL map / OBO must not discard the already-written enrichment
    // outputs, so errors are logged and the run still finalizes.
    let mut ontology_assign: Option<String> = None;
    let mut ontology_mass: Option<String> = None;
    if let (Some(obo), Some(label_cl)) = (args.obo.as_deref(), args.label_cl.as_deref()) {
        match super::ontology::annotate_ontology_with_obo(
            &out,
            label_cl,
            obo,
            args.ontology_fdr_q,
            args.ontology_by,
            // Prefer the correlation-preserving permutation z; fall back to the
            // row-randomization restandardized ES when no sample permutations ran.
            super::ontology::OntologyScore::Z(perm_z_kc.as_ref().unwrap_or(&es_restandardized_kc)),
            Some(&q_kc),
            &cluster_names,
            &loaded.celltype_names,
        ) {
            Ok((a, m)) => {
                ontology_assign = Some(a);
                ontology_mass = Some(m);
            }
            Err(e) => log::error!(
                "inline ontology annotation failed ({e}); enrichment outputs are intact"
            ),
        }
    }

    finalize_annotation(
        &mut manifest,
        Path::new(args.from.as_ref()),
        &manifest_dir,
        &AnnotationArtifacts {
            argmax_abs: &argmax_path,
            markers: &args.markers,
            annotation_abs: Some(&annotation_path),
            cluster_celltype_q_abs: Some(&q_path),
            cluster_celltype_es_abs: Some(&es_path),
            cluster_expression_abs: Some(&cell_expr_path),
            ontology_assignment_abs: ontology_assign.as_deref(),
            ontology_node_mass_abs: ontology_mass.as_deref(),
        },
    )?;

    info!("senna annotate-by-enrichment complete");
    Ok(())
}

/// GO/GMT ontology gene-set mode: read gene-sets → reconcile to the run's gene
/// dictionary (+ coverage gate) → descriptive module-score signature on the
/// cluster profile. No cell-level labels (GO terms aren't cell types); writes
/// the cluster profile + the per-cluster signature and its `K × T` effect matrix.
fn run_ontology_gene_sets(
    args: &AnnotateArgs,
    out: &str,
    profile_gk: &Mat,
    gene_names: &[Box<str>],
    cluster_names: &[Box<str>],
    manifest: &mut crate::run_manifest::RunManifest,
    manifest_dir: &Path,
) -> anyhow::Result<()> {
    use enrichment::ontology_module_score;

    let obo = args
        .obo
        .as_deref()
        .expect("ontology mode validated to require --obo");
    let gs = super::go_signature::load_go_gene_sets(
        obo,
        args.gaf.as_deref(),
        args.gmt.as_deref(),
        args.no_iea,
        args.min_gene_set,
        args.max_gene_set,
        gene_names,
    )?;

    ////////////////////////////////////////////////////////////
    // Descriptive module-score signature (the GO/GMT scorer) //
    ////////////////////////////////////////////////////////////
    // Per (cluster, term): mean_in − mean_out of log1p(CP10K) on the cluster
    // profile, cross-cluster-contrasted. The top positive-effect terms per
    // cluster ARE the GO signature. This plain effect-size ranking recovers
    // cluster lineage (T-cell, cell-cycle, erythroid, …) more cleanly than a
    // permutation-z + TreeBH walk, whose ÷sd reweighting rewards small,
    // stable-null terms and whose depth preference descends to narrow processes.
    let ms = ontology_module_score(profile_gk, &gs.terms, &gs.universe)?;

    let cell_expr_path = format!("{out}.cluster_expression.parquet");
    profile_gk.to_parquet_with_names(
        &cell_expr_path,
        (Some(gene_names), Some("gene")),
        Some(cluster_names),
    )?;
    info!("wrote {cell_expr_path}");

    let sig_path = format!("{out}.ontology_signature.tsv");
    super::go_signature::write_go_signature(
        &sig_path,
        &gs.onto,
        &ms.effect_kt,
        &ms.term_ids,
        &gs.terms,
        "cluster",
        cluster_names,
    )?;
    let effect_path = format!("{out}.ontology_term_effect.parquet");
    ms.effect_kt.to_parquet_with_names(
        &effect_path,
        (Some(cluster_names), Some("cluster")),
        Some(&ms.term_ids),
    )?;
    info!("wrote {effect_path}");

    manifest.annotate.cluster_expression =
        Some(run_manifest::rel_to_manifest(manifest_dir, &cell_expr_path));
    manifest.annotate.ontology_signature =
        Some(run_manifest::rel_to_manifest(manifest_dir, &sig_path));
    manifest.annotate.ontology_term_effect =
        Some(run_manifest::rel_to_manifest(manifest_dir, &effect_path));
    manifest.save(Path::new(args.from.as_ref()))?;

    info!("senna annotate-by-enrichment (ontology gene-set mode) complete");
    Ok(())
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

    // Defensive guard: if the cluster expression matrix has no specificity
    // signal at all (every gene's row is uniform across clusters or all-zero
    // — typically because cluster_labels are misaligned and gene_sum_kg
    // ended up empty), multiplying by these all-zero scores would silently
    // zero the marker matrix, killing every enrichment downstream and
    // leaving every cell unassigned. Warn loudly and leave `markers_gc`
    // untouched so the caller can still get IDF-weighted enrichment.
    if mx <= 1e-6 {
        log::warn!(
            "Empirical specificity scores are all ~0 — likely cluster_labels misaligned or \
             gene_sum_kg is empty. Falling back to IDF-weighted markers without empirical \
             reweighting. Re-check that the cluster file's cell barcodes match the data, \
             or pass --no-empirical-specificity to silence this fallback."
        );
        return;
    }

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

/////////////////////////
// bootstrap artifacts //
/////////////////////////

/// Write what the marker bootstrap learned: the per-cluster consensus distribution, a per-cluster
/// QC row, and a per-celltype QC row.
///
/// These are all keyed by **cluster** or by **celltype**, never by cell — on this path a cell's
/// call is its cluster's call, and reporting a per-cell support would be inventing resolution the
/// method does not have. See `enrichment::marker_bootstrap`'s module doc.
fn write_bootstrap_outputs(
    out: &str,
    boot: &ClusterBootstrap,
    cluster_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> anyhow::Result<()> {
    use anyhow::Context;
    use std::io::Write;

    let k = cluster_names.len();
    let c = boot.c;
    let width = c + 1; // the trailing `unassigned` column

    ///////////////////////////////////////////////////////////////////////
    // K x (C+1): what the resamples actually said about each cluster.   //
    ///////////////////////////////////////////////////////////////////////
    let support_path = format!("{out}.cluster_celltype_support.parquet");
    let mut support = Mat::zeros(k, width);
    for kk in 0..k {
        for j in 0..width {
            support[(kk, j)] = boot.consensus.post[kk * width + j];
        }
    }
    let mut support_cols: Vec<Box<str>> = celltype_names.to_vec();
    support_cols.push(Box::from(enrichment::UNASSIGNED_LABEL));
    support.to_parquet_with_names(
        &support_path,
        (Some(cluster_names), Some("cluster")),
        Some(&support_cols),
    )?;
    info!("wrote {support_path}");

    ///////////////////////////////////////////////////////
    // Per-cluster: the call, the set, and its stability //
    ///////////////////////////////////////////////////////
    let qc_path = format!("{out}.cluster_qc.tsv");
    let mut f = std::fs::File::create(&qc_path).with_context(|| format!("creating {qc_path}"))?;
    writeln!(
        f,
        "cluster\tconsensus_label\tlabel_set\tsupport\tset_support\tentropy\tdecision_gap\tn_draws"
    )?;
    let name_of = |t: usize| -> &str {
        if t == UNASSIGNED {
            enrichment::UNASSIGNED_LABEL
        } else {
            &celltype_names[t]
        }
    };
    for (kk, cname) in cluster_names.iter().enumerate() {
        // The set is printed in canonical celltype order, NOT support order: a label's position
        // should not shift between runs because two shares swapped by 0.01.
        let mut set: Vec<usize> = boot.consensus.label_set[kk].clone();
        set.sort_unstable();
        let set_str = if set.is_empty() {
            String::from("-")
        } else {
            set.iter()
                .map(|&t| celltype_names[t].to_string())
                .collect::<Vec<_>>()
                .join("/")
        };
        writeln!(
            f,
            "{cname}\t{}\t{set_str}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
            name_of(boot.consensus.label[kk]),
            boot.consensus.support[kk],
            boot.consensus.set_support[kk],
            boot.consensus.entropy[kk],
            boot.decision_gap[kk],
            boot.n_draws,
        )?;
    }
    info!("wrote {qc_path}");

    ////////////////////////////////////////////////////////////////////////
    // Per-celltype: is this panel even in a state to be bootstrapped?    //
    ////////////////////////////////////////////////////////////////////////
    let type_qc_path = format!("{out}.type_qc.tsv");
    let mut f =
        std::fs::File::create(&type_qc_path).with_context(|| format!("creating {type_qc_path}"))?;
    writeln!(
        f,
        "cell_type\tn_live\tusable\tmean_es_std_sd\tclusters_won\tmean_support_where_won"
    )?;
    for (cc, name) in celltype_names.iter().enumerate() {
        let jitter: f32 =
            (0..k).map(|kk| boot.es_std_sd[(kk, cc)]).sum::<f32>() / (k.max(1) as f32);
        let won: Vec<usize> = (0..k)
            .filter(|&kk| boot.consensus.label[kk] == cc)
            .collect();
        let mean_support = if won.is_empty() {
            0.0
        } else {
            won.iter()
                .map(|&kk| boot.consensus.support[kk])
                .sum::<f32>()
                / won.len() as f32
        };
        writeln!(
            f,
            "{name}\t{}\t{}\t{:.4}\t{}\t{:.4}",
            boot.n_live[cc],
            boot.usable[cc],
            jitter,
            won.len(),
            mean_support,
        )?;
    }
    info!("wrote {type_qc_path}");

    let called = boot
        .consensus
        .label
        .iter()
        .filter(|&&t| t != UNASSIGNED)
        .count();
    info!(
        "marker bootstrap: {called}/{k} clusters called over {} resamples \
         (mean cluster_label_support {:.2}); {} celltype(s) unusable",
        boot.n_draws,
        boot.consensus.support.iter().sum::<f32>() / (k.max(1) as f32),
        boot.usable.iter().filter(|&&u| !u).count(),
    );
    Ok(())
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
