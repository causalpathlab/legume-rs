//! Orchestration: load inputs → call `enrichment::annotate` → write outputs.

use super::args::AnnotateArgs;
use super::inputs::load_from_manifest;
use crate::embed_common::Mat;
use enrichment::{annotate, AnnotateConfig, AnnotateOutputs};
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn annotate_run(args: &AnnotateArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let (loaded, mut manifest, manifest_dir) = load_from_manifest(&args.from, &args.markers)?;

    let config = AnnotateConfig {
        specificity: loaded.specificity,
        num_row_randomization: args.num_draws,
        num_sample_perm: args.num_perm,
        batch_labels: None,
        fdr_alpha: args.fdr_alpha,
        q_softmax_temperature: args.q_temperature,
        min_confidence: args.min_confidence,
        seed: args.seed,
    };

    info!(
        "Running bipartite enrichment annotation: {} topics × {} celltypes, \
         row-rand B={}, sample-perm B={}",
        loaded.group.profile_gk.ncols(),
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
    } = annotate(
        &loaded.group,
        &loaded.markers_gc,
        &loaded.celltype_names,
        &config,
    )?;

    // Topic row labels shared across the K × C diagnostics so a reader
    // can cross-reference topic IDs by name with fit-topic / svd outputs.
    let topic_names = crate::embed_common::axis_id_names("T", loaded.group.profile_gk.ncols());

    // Cell annotation (N × C posterior).
    let annotation_path = format!("{}.annotation.parquet", args.out);
    cell_annotation_nc.to_parquet_with_names(
        &annotation_path,
        (Some(&loaded.group.cell_names), Some("cell")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {annotation_path}");

    // Argmax TSV.
    let argmax_path = format!("{}.argmax.tsv", args.out);
    {
        let mut f = File::create(&argmax_path)?;
        writeln!(f, "cell\tcell_type\tprobability")?;
        for lab in &argmax_labels {
            writeln!(f, "{}\t{}\t{:.4}", lab.cell_name, lab.label, lab.confidence)?;
        }
    }
    info!("wrote {argmax_path}");

    // Q matrix (K × C).
    let q_path = format!("{}.topic_celltype_q.parquet", args.out);
    q_kc.to_parquet_with_names(
        &q_path,
        (Some(&topic_names), Some("topic")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {q_path}");

    // ES diagnostic (K × C, written wide as raw ES; companion p/q parquets
    // follow the same shape for easy joins downstream).
    let es_path = format!("{}.topic_celltype_es.parquet", args.out);
    es_kc.to_parquet_with_names(
        &es_path,
        (Some(&topic_names), Some("topic")),
        Some(&loaded.celltype_names),
    )?;
    info!("wrote {es_path}");

    let es_std_path = format!("{}.topic_celltype_es_std.parquet", args.out);
    es_restandardized_kc.to_parquet_with_names(
        &es_std_path,
        (Some(&topic_names), Some("topic")),
        Some(&loaded.celltype_names),
    )?;

    let p_path = format!("{}.topic_celltype_p.parquet", args.out);
    pvalue_kc.to_parquet_with_names(
        &p_path,
        (Some(&topic_names), Some("topic")),
        Some(&loaded.celltype_names),
    )?;

    let q_val_path = format!("{}.topic_celltype_q_values.parquet", args.out);
    qvalue_kc.to_parquet_with_names(
        &q_val_path,
        (Some(&topic_names), Some("topic")),
        Some(&loaded.celltype_names),
    )?;

    // Histogram + summary.
    display_annotation_histogram(&cell_annotation_nc, &loaded.celltype_names);

    // Update manifest with annotate section (paths stored as basenames
    // relative to the manifest directory).
    let rel = |abs_path: &str| -> String { rel_to_manifest(&manifest_dir, abs_path) };
    manifest.annotate.annotation = Some(rel(&annotation_path));
    manifest.annotate.argmax = Some(rel(&argmax_path));
    manifest.annotate.topic_celltype_q = Some(rel(&q_path));
    manifest.annotate.topic_celltype_es = Some(rel(&es_path));
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

fn display_annotation_histogram(annot: &Mat, annot_names: &[Box<str>]) {
    let n_cells = annot.nrows();
    let n_types = annot.ncols();
    let mut max_probs = Vec::with_capacity(n_cells);
    let mut assignments: Vec<Option<usize>> = Vec::with_capacity(n_cells);
    let mut unassigned = 0usize;

    for i in 0..n_cells {
        let row = annot.row(i);
        let sum: f32 = row.iter().sum();
        if sum < 1e-12 {
            max_probs.push(0.0);
            assignments.push(None);
            unassigned += 1;
            continue;
        }
        let (max_idx, max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        max_probs.push(*max_val);
        assignments.push(Some(max_idx));
    }

    let mut type_counts = vec![0usize; n_types];
    let mut type_prob_sum = vec![0.0f32; n_types];
    for (i, ct) in assignments.iter().enumerate() {
        if let Some(c) = ct {
            type_counts[*c] += 1;
            type_prob_sum[*c] += max_probs[i];
        }
    }
    let mut sorted_types: Vec<usize> = (0..n_types).collect();
    sorted_types.sort_by(|&a, &b| type_counts[b].cmp(&type_counts[a]));

    let max_count = *type_counts.iter().max().unwrap_or(&1).max(&unassigned);
    const MAX_BAR: usize = 20;

    let assigned_cells = n_cells - unassigned;
    let mean_prob = if assigned_cells > 0 {
        max_probs
            .iter()
            .zip(assignments.iter())
            .filter_map(|(p, a)| a.map(|_| *p))
            .sum::<f32>()
            / assigned_cells as f32
    } else {
        0.0
    };
    let above_50 = max_probs.iter().filter(|&&x| x > 0.5).count();
    let above_70 = max_probs.iter().filter(|&&x| x > 0.7).count();

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
