//! Archetype-based topic resolution from the trained cell embedding.
//!
//! Mirrors `senna bge --resolve-etm`: archetypal analysis on the cell
//! embedding `Z = e_cell [N, H]` yields archetypes `α [K, H]` (topic
//! embeddings) and per-cell simplex weights `θ [N, K]` (topic
//! proportions); the gene×topic dictionary is `β = log_softmax_g(β_g·αᵀ)`,
//! using the base gene embedding `β_g [G, H]` as the feature side. No
//! retraining — a post-hoc factorisation of the frozen embedding.
//!
//! Writes the senna topic-model layout (consumed by `senna {plot,
//! clustering, annotate} --from` via the `{prefix}.latent` / `.dictionary`
//! suffix convention):
//!   - `{out}.latent.parquet`          log θ [N, K]
//!   - `{out}.dictionary.parquet`      β    [G, K]  (each topic column a gene simplex)
//!   - `{out}.topic_embedding.parquet` α    [K, H]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use log::info;
use matrix_util::archetypal::{anchor_topics, select_anchor_topics, topic_dictionary, AnchorOpts};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{ConvertMatOps, IoOps};

use super::args::GemArgs;
use super::common::candle_core;
use super::feature_table::FeatureTable;
use super::model::GemModel;
use candle_core::Device;
use graph_embedding_util::data::UnifiedData;

/// Paths to the resolved-topic parquet artifacts (recorded in the manifest).
pub struct ResolvedTopics {
    pub k: usize,
    pub latent: String,
    pub dictionary: String,
    pub topic_embedding: String,
}

/// Resolve archetypal topics from the trained embedding and write the
/// topic-model layout. Runs even after a Ctrl+C-interrupted training so
/// the partial embedding still yields a topic report; when `stop` is set
/// it skips the (potentially long) K-sweep and finalises at a fixed K.
pub fn resolve_topics(
    prefix: &str,
    model: &GemModel,
    table: &FeatureTable,
    unified: &UnifiedData,
    args: &GemArgs,
    stop: &Arc<AtomicBool>,
    // QC-passed cell indices (spliced auto cell-calling): archetypes are fit on
    // and θ is assigned to these cells only, and `latent` is written for
    // them — keeping near-empty cells out would otherwise seed a spurious
    // "empty" archetype.
    keep_idx: &[usize],
) -> Result<ResolvedTopics> {
    let cpu = Device::Cpu;
    // Restrict to QC-passed cells (row selection, no backend rewrite).
    let z =
        DMatrix::<f32>::from_tensor(&model.e_cell.to_device(&cpu)?)?.select_rows(keep_idx.iter()); // [N_keep, H]
    let barcodes_kept: Vec<Box<str>> = keep_idx
        .iter()
        .map(|&i| unified.barcodes[i].clone())
        .collect();
    let rho = DMatrix::<f32>::from_tensor(&model.beta.to_device(&cpu)?)?; // [G, H]
    let h = z.ncols();
    anyhow::ensure!(
        rho.ncols() == h,
        "resolve-topics: cell embedding H={} != gene embedding H={}",
        h,
        rho.ncols()
    );

    // Separable-NMF topic recovery (Arora anchors via SPA) on the gene
    // embedding ρ: the K anchor genes are the convex-hull vertices of the
    // gene cloud — each a near-pure marker for one topic. Deterministic and
    // single-pass (no subsample, no nonconvex archetype fit, no per-K
    // refit), so even the auto-K sweep is one SPA pass; θ is then assigned to
    // every QC-passed cell by projecting it onto the anchors (FW_ITERS
    // Frank–Wolfe steps — the simplex projection converges quickly, not a user
    // knob). The MIN_ANCHOR_CELLS guard drops singleton/outlier-gene topics: a
    // topic claimed by < 10 cells is almost surely an artifact at gem's scale.
    const FW_ITERS: usize = 30;
    const MIN_ANCHOR_CELLS: usize = 10;
    let opts = AnchorOpts {
        fw_iters: FW_ITERS,
        min_anchor_cells: MIN_ANCHOR_CELLS,
    };
    let interrupted = stop.load(Ordering::SeqCst);
    let res = match args.num_topics {
        Some(k) => {
            anyhow::ensure!(k >= 2, "resolve-topics: --num-topics must be ≥ 2");
            info!("resolve-topics: separable-NMF (SPA anchors) with fixed K={k}");
            anchor_topics(&z, &rho, k, opts)
        }
        None if interrupted => {
            // Interrupted: skip the K-sweep, finalise fast at a sane K.
            let k = args.n_programs.max(2);
            info!("resolve-topics: interrupted — fixing K={k}");
            anchor_topics(&z, &rho, k, opts)
        }
        None => {
            // SPA anchors are nested, so the sweep is a single pass; the
            // residual elbow over 2..=H+1 (a (K-1)-simplex can't span past
            // H+1 dims) selects K.
            let upper = (h + 1).max(2);
            let krange: Vec<usize> = (2..=upper).collect();
            info!("resolve-topics: auto-selecting K via SPA residual-elbow over 2..={upper}");
            select_anchor_topics(&z, &rho, &krange, opts).1
        }
    };
    // K reflects any anchors the guard dropped, not the requested count.
    let k = res.anchors.len();
    info!(
        "resolve-topics: K={k}, anchor genes=[{}], reconstruction RSS={:.4}",
        res.anchors
            .iter()
            .map(|&g| table.gene_names[g].as_ref())
            .collect::<Vec<&str>>()
            .join(", "),
        res.rss
    );

    // β = log_softmax_g(β_g · (α − ᾱ)ᵀ): [G, K], each topic column a gene
    // simplex; markers surface as each topic's deviation from the mean
    // anchor (matches senna bge). α here are the anchor-gene embeddings.
    let beta_gk = topic_dictionary(&rho, &res.alpha);
    // log θ on the simplex (matches senna's `latent = log θ`).
    let log_theta = res.theta.map(|x| (x + 1e-8).ln());

    let topic_names: Vec<Box<str>> = (0..k).map(|i| format!("T{i}").into_boxed_str()).collect();
    let h_names: Vec<Box<str>> = (0..h)
        .map(|i| format!("dim_{i}").into_boxed_str())
        .collect();

    let path_latent = format!("{prefix}.latent.parquet");
    let path_dict = format!("{prefix}.dictionary.parquet");
    let path_topic_emb = format!("{prefix}.topic_embedding.parquet");

    log_theta
        .to_parquet_with_names(
            &path_latent,
            (Some(&barcodes_kept), Some("cell")),
            Some(&topic_names),
        )
        .with_context(|| format!("writing {path_latent}"))?;
    beta_gk
        .to_parquet_with_names(
            &path_dict,
            (Some(&table.gene_names), Some("gene")),
            Some(&topic_names),
        )
        .with_context(|| format!("writing {path_dict}"))?;
    res.alpha
        .to_parquet_with_names(
            &path_topic_emb,
            (Some(&topic_names), Some("topic")),
            Some(&h_names),
        )
        .with_context(|| format!("writing {path_topic_emb}"))?;

    info!("resolve-topics: wrote {prefix}.{{latent,dictionary,topic_embedding}}.parquet (K={k})");

    Ok(ResolvedTopics {
        k,
        latent: path_latent,
        dictionary: path_dict,
        topic_embedding: path_topic_emb,
    })
}
