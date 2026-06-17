//! `senna bge` ETM resolution (on by default; disable with `--skip-etm`):
//! resolve the ETM topic side from a finished bge run (no further training)
//! and write a topic-model-shaped output layout (`latent` = log θ,
//! `dictionary` = β). Split out of the bge driver.

use super::BgeArgs;
use crate::embed_common::*;
use graph_embedding_util as ge;

/////////////////////////////////////////////////////////////////////
// ETM resolution from the bge cell embedding (default; --skip-etm)  //
/////////////////////////////////////////////////////////////////////

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
///   - `{out}.feature_embedding.parquet`ρ    [D,H]   (raw bge feature embedding)
///   - `{out}.feature_bias.parquet`     `b_feat` [D]
///   - `{out}.cell_bias.parquet`        `b_cell` [N]    (per-cell depth sink)
pub(super) fn resolve_etm_topics(
    model: &ge::JointEmbedModel,
    feature_names: &[Box<str>],
    barcodes: &[Box<str>],
    args: &BgeArgs,
    cell_keep_idx: Option<&[usize]>,
) -> anyhow::Result<()> {
    use matrix_util::archetypal::{
        anchor_topics, select_anchor_topics, topic_dictionary, AnchorOpts,
    };

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

    // Separable-NMF topic recovery (Arora anchors via SPA) on the feature
    // embedding ρ: anchor features are the convex-hull vertices of the
    // feature cloud — near-pure markers, one per topic. Deterministic and
    // single-pass (no subsample, no nonconvex fit, no per-K refit); θ is then
    // assigned to every cell by projecting it onto the anchors (FW_ITERS
    // Frank–Wolfe steps — the simplex projection converges quickly, not a user
    // knob). The MIN_ANCHOR_CELLS guard drops singleton/outlier-feature topics:
    // a topic claimed by < 10 cells is almost surely an artifact at this scale.
    const FW_ITERS: usize = 30;
    const MIN_ANCHOR_CELLS: usize = 10;
    let opts = AnchorOpts {
        fw_iters: FW_ITERS,
        min_anchor_cells: MIN_ANCHOR_CELLS,
    };
    let res = if let Some(k) = args.num_topics {
        anyhow::ensure!(k >= 2, "resolve-etm: --num-topics must be ≥ 2");
        info!("resolve-etm: separable-NMF (SPA anchors) with fixed K={k}");
        anchor_topics(&z, &rho, k, opts)
    } else {
        // SPA anchors are nested → the sweep is one pass; residual elbow
        // over 2..=H+1 selects K.
        let upper = (h + 1).max(2);
        let krange: Vec<usize> = (2..=upper).collect();
        info!("resolve-etm: auto-selecting K via SPA residual-elbow over 2..={upper}");
        select_anchor_topics(&z, &rho, &krange, opts).1
    };
    // K reflects any anchors the guard dropped, not the requested count.
    let k = res.anchors.len();
    info!(
        "resolve-etm: K={k}, anchor features=[{}], reconstruction RSS={:.4}",
        res.anchors
            .iter()
            .map(|&d| feature_names[d].as_ref())
            .collect::<Vec<&str>>()
            .join(", "),
        res.rss
    );

    // β = log_softmax_d(ρ · (α − ᾱ)ᵀ): [D, K], each topic column a feature
    // simplex; markers surface as each topic's deviation from the mean anchor.
    // α here are the anchor-feature embeddings.
    let beta_dk = topic_dictionary(&rho, &res.alpha);
    // log θ on the simplex.
    let log_theta = res.theta.map(|x| (x + 1e-8).ln());

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
    // Resolved topic embeddings α (warm-start for a later `masked-topic` finetune).
    res.alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topic_names), Some("topic")),
        Some(&h_names),
    )?;
    // Raw bge embeddings preserved under non-conflicting names.
    z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&h_names),
    )?;
    rho.to_parquet_with_names(
        &format!("{out}.feature_embedding.parquet"),
        (Some(feature_names), Some("feature")),
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
         {{cell,feature}}_embedding.parquet to {out}.*"
    );
    Ok(())
}
