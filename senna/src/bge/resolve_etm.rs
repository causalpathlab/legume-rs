//! `senna bge` ETM resolution (on by default; disable with `--skip-etm`):
//! resolve the ETM topic side from a finished bge run (no further training)
//! and write a topic-model-shaped output layout (`latent` = log θ,
//! `dictionary` = β). Split out of the bge driver.

use super::BgeArgs;
use crate::embed_common::*;
use graph_embedding_util as ge;

//////////////////////////////////////////////////////////////////////
// ETM resolution from the bge cell embedding (default; --skip-etm) //
//////////////////////////////////////////////////////////////////////

/// Resolve the ETM topic side from a finished bge run, with no further
/// training, and write a topic-model-shaped output layout so that
/// `senna {plot, plot-topic, clustering, annotate} --from` consume the
/// topics directly (matching the `senna topic` / `masked-topic` conventions:
/// `latent` = log θ, `dictionary` = β).
///
/// Archetypal analysis on the cell embedding `Z [N,H]` yields archetypes
/// `α [K,H]` (= topic embeddings) and per-cell simplex weights `θ [N,K]`
/// (= topic proportions); the dictionary is `β = log_softmax_d(ρ·αᵀ)`,
/// the same factorization the ETM decoder uses. Writes:
///   - `{out}.latent.parquet`           log θ [N,K]   (topic proportions)
///   - `{out}.dictionary.parquet`       β    [D,K]   (each topic column a gene simplex)
///   - `{out}.topic_embedding.parquet`  α    [K,H]   (for a later `masked-topic` finetune)
///   - `{out}.cell_embedding.parquet`   Z    [N,H]   (raw bge cell embedding)
///   - `{out}.feature_bias.parquet`     `b_feat` [D]
///   - `{out}.cell_bias.parquet`        `b_cell` [N]    (per-cell depth sink)
pub(super) fn resolve_etm_topics(
    model: &ge::JointEmbedModel,
    feature_names: &[Box<str>],
    barcodes: &[Box<str>],
    args: &BgeArgs,
    cell_keep_idx: Option<&[usize]>,
    labels: &[usize],
) -> anyhow::Result<()> {
    use matrix_util::archetypal::topic_dictionary;

    let cpu = candle_core::Device::Cpu;
    let z_full = Mat::from_tensor(&model.e_cell.to_device(&cpu)?)?; // [N, H]
                                                                    // Drop QC-failed cells from archetype fitting + per-cell outputs. `z`
                                                                    // and `barcodes` are subset by the same `keep` so their rows stay
                                                                    // aligned; the dictionary β (from ρ + archetypes) is per-feature and
                                                                    // unaffected.
    let (z, barcodes): (Mat, Vec<Box<str>>) = match cell_keep_idx {
        Some(keep) => (
            z_full.select_rows(keep.iter()),
            keep.iter().map(|&i| barcodes[i].clone()).collect(),
        ),
        None => (z_full, barcodes.to_vec()),
    };
    let rho = Mat::from_tensor(&model.e_feat.to_device(&cpu)?)?; // [D, H]
    let h = z.ncols();
    anyhow::ensure!(
        rho.ncols() == h,
        "resolve-etm: cell embedding H={} != feature embedding H={}",
        h,
        rho.ncols()
    );

    // Robust topic recovery from CELL CLUSTERS, using the Leiden `labels` the
    // bge driver computed ONCE on this same kept-cell embedding (shared with the
    // co-embedding's temperature calibration). The previous approach (Arora/SPA
    // convex-hull anchors on ρ) was outlier-driven: a few extreme features (e.g.
    // immunoglobulin genes) became the hull vertices, SPA picked them, and the
    // min-cells guard nuked the rest → K collapsed (BM1 → 2). Each cluster is a
    // topic:
    //   α [K,H] = L2-normalized cluster centroid (the topic *direction*),
    //   θ [N,K] = softmax over clusters of ⟨z_i, α_k⟩  (soft assignment),
    //   β [D,K] = log_softmax_d(ρ·(α−ᾱ)ᵀ) — a gene scores high in a topic when
    //             its embedding aligns with that cluster's cell direction.
    anyhow::ensure!(
        labels.len() == z.nrows(),
        "resolve-etm: labels ({}) != kept cells ({})",
        labels.len(),
        z.nrows()
    );
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    anyhow::ensure!(k >= 2, "resolve-etm: clustering produced < 2 topics");
    let alpha = cluster_centroids(&z, labels, k); // [K, H]
    let theta = soft_theta(&z, &alpha); // [N, K]
    let mut sizes = vec![0usize; k];
    for &l in labels {
        sizes[l] += 1;
    }
    info!("resolve-etm: cluster-seeded K={k}, cluster sizes={sizes:?}");

    // β = log_softmax_d(ρ · (α − ᾱ)ᵀ): [D, K], each topic column a gene simplex.
    let beta_dk = topic_dictionary(&rho, &alpha);
    // log θ on the simplex.
    let log_theta = theta.map(|x| (x + 1e-8).ln());

    let topic_names = axis_id_names("T", k);
    let h_names = axis_id_names("h", h);
    let out = &args.out;

    // Topic-model layout — latent = log θ, dictionary = β.
    log_theta.to_parquet_with_names(
        &format!("{out}.latent.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&topic_names),
    )?;
    beta_dk.to_parquet_with_names(
        &format!("{out}.dictionary.parquet"),
        (Some(feature_names), Some("gene")),
        Some(&topic_names),
    )?;
    // Resolved topic embeddings α = cluster centroids (warm-start for a later
    // `masked-topic` finetune).
    alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topic_names), Some("topic")),
        Some(&h_names),
    )?;
    // Raw bge cell embedding Z preserved (the reference manifold). The feature
    // side — {out}.feature_embedding.parquet (SIMBA co-embed) — is written by
    // the bge driver before this call, so it is not emitted here.
    z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&h_names),
    )?;
    ge::eval::save_bias(
        &format!("{out}.feature_bias.parquet"),
        &model.b_feat,
        feature_names,
        "feature",
    )?;
    // Per-cell bias `b_cell` (depth sink), subset by the same QC keep mask as
    // the cell rows above so barcodes/biases stay aligned.
    let b_cell = match cell_keep_idx {
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx_t = candle_core::Tensor::from_vec(idx, keep.len(), model.b_cell.device())?;
            model.b_cell.index_select(&idx_t, 0)?
        }
        None => model.b_cell.clone(),
    };
    ge::eval::save_bias(
        &format!("{out}.cell_bias.parquet"),
        &b_cell,
        &barcodes,
        "cell",
    )?;

    info!(
        "resolve-etm: wrote topic-model layout (latent=log θ, dictionary=β) + \
         cell_embedding.parquet to {out}.*"
    );
    Ok(())
}

/// L2-normalized cluster centroids of `z` `[N,H]` → `α` `[K,H]` (topic
/// directions). Empty clusters (shouldn't occur) stay zero.
fn cluster_centroids(z: &Mat, labels: &[usize], k: usize) -> Mat {
    let (n, h) = (z.nrows(), z.ncols());
    let mut alpha = Mat::zeros(k, h);
    let mut counts = vec![0f32; k];
    for i in 0..n {
        let l = labels[i];
        counts[l] += 1.0;
        for j in 0..h {
            alpha[(l, j)] += z[(i, j)];
        }
    }
    for l in 0..k {
        if counts[l] > 0.0 {
            let mut nrm = 0f32;
            for j in 0..h {
                alpha[(l, j)] /= counts[l];
                nrm += alpha[(l, j)] * alpha[(l, j)];
            }
            let nrm = nrm.sqrt().max(1e-8);
            for j in 0..h {
                alpha[(l, j)] /= nrm;
            }
        }
    }
    alpha
}

/// Soft topic assignment `θ` `[N,K]` = softmax over clusters of `⟨z_i, α_k⟩`
/// (both rows ~unit, so this is a temperature-1 cosine softmax). Parameter-free.
fn soft_theta(z: &Mat, alpha: &Mat) -> Mat {
    let s = z * alpha.transpose(); // [N, K]
    let (n, k) = (s.nrows(), s.ncols());
    let mut th = s.clone();
    for i in 0..n {
        let mut mx = f32::NEG_INFINITY;
        for c in 0..k {
            mx = mx.max(s[(i, c)]);
        }
        let mut sum = 0f32;
        for c in 0..k {
            let e = (s[(i, c)] - mx).exp();
            th[(i, c)] = e;
            sum += e;
        }
        let sum = sum.max(1e-8);
        for c in 0..k {
            th[(i, c)] /= sum;
        }
    }
    th
}
