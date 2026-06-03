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
use matrix_util::archetypal::{archetypal_analysis, select_archetype_k, AaArgs};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{ConvertMatOps, IoOps, MatOps};

use super::args::RnaModEmbedArgs;
use super::common::candle_core;
use super::feature_table::FeatureTable;
use super::model::RnaModEmbedModel;
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
    model: &RnaModEmbedModel,
    table: &FeatureTable,
    unified: &UnifiedData,
    args: &RnaModEmbedArgs,
    stop: &Arc<AtomicBool>,
) -> Result<ResolvedTopics> {
    let cpu = Device::Cpu;
    let z = DMatrix::<f32>::from_tensor(&model.e_cell.to_device(&cpu)?)?; // [N, H]
    let rho = DMatrix::<f32>::from_tensor(&model.beta.to_device(&cpu)?)?; // [G, H]
    let h = z.ncols();
    anyhow::ensure!(
        rho.ncols() == h,
        "resolve-topics: cell embedding H={} != gene embedding H={}",
        h,
        rho.ncols()
    );

    let base = AaArgs {
        k: 2,
        max_iter: args.aa_iters,
        fw_iters: 30,
        tol: 1e-4,
        seed: args.seed,
        subsample: args.aa_subsample,
    };

    let interrupted = stop.load(Ordering::SeqCst);
    let (k, res) = match args.num_topics {
        Some(k) => {
            anyhow::ensure!(k >= 2, "resolve-topics: --num-topics must be ≥ 2");
            info!("resolve-topics: archetypal analysis with fixed K={k}");
            (k, archetypal_analysis(&z, &AaArgs { k, ..base }))
        }
        None if interrupted => {
            // Interrupted: skip the elbow sweep, finalise fast at a sane K.
            let k = args.n_programs.max(2);
            info!("resolve-topics: interrupted — skipping K-sweep, using fixed K={k}");
            (k, archetypal_analysis(&z, &AaArgs { k, ..base }))
        }
        None => {
            let krange: Vec<usize> = (2..=args.max_k.max(2)).collect();
            info!(
                "resolve-topics: auto-selecting K via archetypal RSS-elbow over 2..={}",
                args.max_k.max(2)
            );
            select_archetype_k(&z, &krange, &base)
        }
    };
    info!("resolve-topics: K={k}, reconstruction RSS={:.4}", res.rss);

    // β = log_softmax_g(β_g · αᵀ): [G, K], each topic column a gene simplex.
    let beta_gk = (&rho * res.alpha.transpose()).log_softmax_columns();
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
            (Some(&unified.barcodes), Some("cell")),
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
