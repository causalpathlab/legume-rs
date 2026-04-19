//! `senna layout topic-pb` — cells laid out by (PB ⊗ topic) outer-product features.
//!
//! Computes m × K outer-product features for each cell:
//!   feature[i,j] = 1(cell ∈ PB_i) × P(topic_j | cell)
//! where PB membership is one-hot and topic probabilities come from softmax(latent).
//!
//! Pipeline:
//! 1. Load PB assignments (from collapse) and topic latent (from manifest)
//! 2. Build sparse (m*K)-dimensional feature matrix (only K non-zero per cell)
//! 3. SVD for denoising (top 50-100 PCs)
//! 4. t-SNE or PHATE on PCs → 2D coords
//! 5. Write cell_coords.parquet

use super::fit_visualize_common::{prepare_viz, resolve_inputs, VisualizeCommonArgs};
use crate::embed_common::*;
use crate::geometry::phate::{phate_layout_2d, PhateArgs};
use crate::run_manifest;
use anyhow::Context;
use nalgebra as na;

#[derive(Args, Debug)]
pub struct LayoutTopicPbArgs {
    #[clap(flatten)]
    common: VisualizeCommonArgs,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of SVD components to keep before t-SNE/PHATE"
    )]
    svd_dims: usize,

    #[arg(
        long,
        default_value = "phate",
        help = "Dimensionality reduction method (phate or tsne)"
    )]
    method: String,
    // TODO: add PHATE/t-SNE specific params if needed
}

pub fn fit_layout_topic_pb(args: &LayoutTopicPbArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;

    // Load latent matrix from manifest
    let latent_path = resolved
        .manifest
        .as_ref()
        .and_then(|m| m.outputs.latent.as_deref())
        .context("--from manifest missing outputs.latent (did you run senna topic?)")?;

    let latent_abs = run_manifest::resolve(
        resolved.manifest_path.as_ref().unwrap().parent().unwrap(),
        latent_path,
    );

    info!("Loading topic latent from {}", latent_abs.display());
    let MatWithNames {
        mat: latent_mat, ..
    } = Mat::from_parquet(latent_abs.to_str().unwrap())?;
    let (n_cells, n_topics) = (latent_mat.nrows(), latent_mat.ncols());
    info!("Loaded latent: {} cells × {} topics", n_cells, n_topics);

    // Softmax to get topic probabilities
    let topic_probs = softmax_rows(&latent_mat);

    // Prepare PB data (collapse, membership, etc.)
    let prep = prepare_viz(&args.common, &resolved)?;
    let n_pb_kept = prep.pb_size.len();

    // Build outer-product features: (n_cells × (n_pb_kept * n_topics))
    // Sparse: only n_topics non-zero entries per cell
    info!(
        "Building outer-product features: {} cells × ({} PBs × {} topics) = {} dims",
        n_cells,
        n_pb_kept,
        n_topics,
        n_pb_kept * n_topics
    );

    // Aggregate cell features to PB-level: mean topic distribution per PB
    info!("Aggregating to PB-level: computing mean topic distribution per PB");
    let pb_features = aggregate_to_pb_level(&prep.pb_membership_kept, &topic_probs, n_pb_kept)?;

    info!(
        "PB-level features: {} PBs × {} topics",
        pb_features.nrows(),
        pb_features.ncols()
    );

    // Run PHATE on PB-level features (fast: ~900 PBs instead of 8k cells)
    info!("Running {} on PB-level features", args.method);
    let pb_coords = match args.method.as_str() {
        "phate" => {
            let params = PhateArgs::default();
            phate_layout_2d(&pb_features, &params)
        }
        "tsne" => {
            anyhow::bail!("t-SNE not yet implemented for topic-pb layout")
        }
        _ => anyhow::bail!("Unknown method: {}", args.method),
    };

    // Map cells to their PB coordinates
    info!("Mapping cells to PB coordinates");
    let mut cell_coords = na::DMatrix::zeros(n_cells, 2);
    for i in 0..n_cells {
        let pb_id = prep.pb_membership_kept[i];
        if pb_id != usize::MAX && pb_id < n_pb_kept {
            cell_coords[(i, 0)] = pb_coords[(pb_id, 0)];
            cell_coords[(i, 1)] = pb_coords[(pb_id, 1)];
        } else {
            // Orphan cell
            cell_coords[(i, 0)] = f32::NAN;
            cell_coords[(i, 1)] = f32::NAN;
        }
    }

    // Write output using existing finalize_viz
    // Note: finalize_viz expects PB coords, but we're giving cell coords
    // Need to adapt or create new finalizer
    info!("Writing cell coordinates");
    write_cell_coords(&args.common, &mut resolved, &prep, &cell_coords)
}

/// Softmax each row of a matrix independently
fn softmax_rows(mat: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    let (n, k) = (mat.nrows(), mat.ncols());
    let mut result = na::DMatrix::zeros(n, k);

    for i in 0..n {
        let row = mat.row(i);
        let max_val = row.max();
        let mut exp_sum = 0.0_f32;

        for j in 0..k {
            let exp_val = (row[j] - max_val).exp();
            result[(i, j)] = exp_val;
            exp_sum += exp_val;
        }

        for j in 0..k {
            result[(i, j)] /= exp_sum;
        }
    }

    result
}

/// Aggregate cell-level topic probabilities to PB-level mean features
/// Returns (n_pb × n_topics) matrix
fn aggregate_to_pb_level(
    pb_membership: &[usize],
    topic_probs: &na::DMatrix<f32>,
    n_pb: usize,
) -> anyhow::Result<na::DMatrix<f32>> {
    let n_cells = pb_membership.len();
    let n_topics = topic_probs.ncols();

    anyhow::ensure!(
        topic_probs.nrows() == n_cells,
        "pb_membership length {} != topic_probs rows {}",
        n_cells,
        topic_probs.nrows()
    );

    // Accumulate topic probs per PB
    let mut pb_topic_sum = na::DMatrix::zeros(n_pb, n_topics);
    let mut pb_counts = vec![0usize; n_pb];

    for cell_idx in 0..n_cells {
        let pb_id = pb_membership[cell_idx];
        if pb_id == usize::MAX {
            continue; // Skip orphan cells
        }

        anyhow::ensure!(
            pb_id < n_pb,
            "pb_id {} >= n_pb {} for cell {}",
            pb_id,
            n_pb,
            cell_idx
        );

        for topic_idx in 0..n_topics {
            pb_topic_sum[(pb_id, topic_idx)] += topic_probs[(cell_idx, topic_idx)];
        }
        pb_counts[pb_id] += 1;
    }

    // Compute means
    for pb_id in 0..n_pb {
        if pb_counts[pb_id] > 0 {
            let count = pb_counts[pb_id] as f32;
            for topic_idx in 0..n_topics {
                pb_topic_sum[(pb_id, topic_idx)] /= count;
            }
        }
    }

    Ok(pb_topic_sum)
}

/// Write cell coordinates (adapted from finalize_viz)
fn write_cell_coords(
    _args: &VisualizeCommonArgs,
    resolved: &mut super::fit_visualize_common::ResolvedViz,
    prep: &super::fit_visualize_common::VizPrep,
    cell_coords: &na::DMatrix<f32>,
) -> anyhow::Result<()> {
    use std::path::Path;

    let n_cells = cell_coords.nrows();
    anyhow::ensure!(
        cell_coords.ncols() == 2,
        "Expected 2D coords, got {}",
        cell_coords.ncols()
    );

    // Build output matrix: [x, y, pb_id, (optional cluster)]
    let n_extra = 1; // pb_id column
    let mut cell_out = na::DMatrix::zeros(n_cells, 2 + n_extra);

    for i in 0..n_cells {
        cell_out[(i, 0)] = cell_coords[(i, 0)];
        cell_out[(i, 1)] = cell_coords[(i, 1)];

        let pb_id = prep.pb_membership_kept[i];
        cell_out[(i, 2)] = if pb_id == usize::MAX {
            f32::NAN
        } else {
            pb_id as f32
        };
    }

    let col_names: Vec<Box<str>> = vec!["x".into(), "y".into(), "pb_id".into()];

    let cell_coords_path = format!("{}.cell_coords.parquet", resolved.out);

    // Generate cell names
    let cell_names: Vec<Box<str>> = (0..n_cells).map(|i| format!("cell_{}", i).into()).collect();

    cell_out.to_parquet_with_names(
        &cell_coords_path,
        (Some(&cell_names[..]), Some("cell")),
        Some(&col_names),
    )?;
    info!("Wrote {}", cell_coords_path);

    // Update manifest if present
    if let (Some(manifest), Some(manifest_path)) =
        (resolved.manifest.as_mut(), &resolved.manifest_path)
    {
        let manifest_dir = manifest_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));

        let rel_path = if let Ok(stripped) = Path::new(&cell_coords_path).strip_prefix(manifest_dir)
        {
            stripped.to_string_lossy().into_owned()
        } else {
            cell_coords_path
        };

        manifest.layout.cell_coords = Some(rel_path);
        manifest.save(manifest_path)?;
        info!("Updated manifest {}", manifest_path.display());
    }

    Ok(())
}
