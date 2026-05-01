//! Warm-start initialization for `senna topic` and `senna indexed-topic`.
//!
//! Loads weights saved by a previous training run into the current `VarMap`,
//! so training continues from the previous checkpoint instead of from
//! random init. The current architecture must match the saved one: same K,
//! encoder layers, level decoder dims, and (for indexed) embedding dim and
//! `n_features_full`.
//!
//! Cross-gene-set warm-start (i.e. resuming on a dataset with a different
//! gene list) is not supported here: the encoder's input is gene-keyed for
//! the dense path (D_coarse), and the dictionary tensors at every level are
//! gene-keyed too. Run on the same gene set, or train from scratch.

use crate::topic::model_metadata::TopicModelMetadata;
use candle_util::candle_nn::VarMap;

/// Architecture invariants the saved checkpoint must match.
pub struct WarmStartCheck<'a> {
    /// "topic" or "indexed_topic"
    pub model_type_expected: &'static str,
    pub n_topics: usize,
    pub n_features_full: usize,
    pub n_features_encoder: usize,
    pub encoder_hidden: &'a [usize],
    pub level_decoder_dims: &'a [usize],
    /// Set only for indexed; ignored for dense.
    pub embedding_dim: Option<usize>,
}

/// Validate that the saved checkpoint is architecture-compatible, then load
/// weights into `parameters`.
pub fn warm_start_load(
    parameters: &VarMap,
    prefix: &str,
    expected: &WarmStartCheck<'_>,
) -> anyhow::Result<()> {
    let metadata = TopicModelMetadata::load(prefix)?;

    anyhow::ensure!(
        metadata.model_type.as_ref() == expected.model_type_expected,
        "warm-start: model_type mismatch (saved='{}', current='{}')",
        metadata.model_type,
        expected.model_type_expected,
    );
    anyhow::ensure!(
        metadata.n_topics == expected.n_topics,
        "warm-start: K mismatch (saved={}, current={})",
        metadata.n_topics,
        expected.n_topics,
    );
    anyhow::ensure!(
        metadata.encoder_hidden.as_slice() == expected.encoder_hidden,
        "warm-start: encoder_hidden mismatch (saved={:?}, current={:?})",
        metadata.encoder_hidden,
        expected.encoder_hidden,
    );
    anyhow::ensure!(
        metadata.n_features_full == expected.n_features_full,
        "warm-start: n_features_full mismatch (saved={}, current={}). \
         Cross-gene-set warm-start not supported — train on the same gene set.",
        metadata.n_features_full,
        expected.n_features_full,
    );
    anyhow::ensure!(
        metadata.n_features_encoder == expected.n_features_encoder,
        "warm-start: n_features_encoder (D_coarse) mismatch (saved={}, current={}). \
         Coarsening parameters must match the original run.",
        metadata.n_features_encoder,
        expected.n_features_encoder,
    );
    anyhow::ensure!(
        metadata.level_decoder_dims.as_slice() == expected.level_decoder_dims,
        "warm-start: level_decoder_dims mismatch (saved={:?}, current={:?})",
        metadata.level_decoder_dims,
        expected.level_decoder_dims,
    );
    if let Some(emb) = expected.embedding_dim {
        anyhow::ensure!(
            metadata.embedding_dim == Some(emb),
            "warm-start: embedding_dim mismatch (saved={:?}, current={})",
            metadata.embedding_dim,
            emb,
        );
    }

    let safetensors_path = format!("{prefix}.safetensors");
    log::info!("Warm-starting from {safetensors_path}");

    // VarMap has interior mutability via Arc<Mutex<_>>; clone shares storage
    // and lets us call `.load()` (which takes `&mut self`) without forcing
    // every caller to thread a mutable reference through the pipeline.
    let mut handle = parameters.clone();
    handle.load(&safetensors_path)?;
    log::info!(
        "Warm-start: loaded {} variables from {prefix}",
        handle.all_vars().len()
    );
    Ok(())
}
