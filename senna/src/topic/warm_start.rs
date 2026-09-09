//! Warm-start initialization for `senna topic` and `senna masked-topic`.
//!
//! Loads weights saved by a previous training run into the current `VarMap`,
//! so training continues from the previous checkpoint instead of from
//! random init. The current architecture must match the saved one: same K,
//! encoder layers, level decoder dims, and (for indexed) embedding dim and
//! `n_features_full`.
//!
//! Cross-gene-set warm-start (i.e. resuming on a dataset with a different
//! gene list) is not supported here: the encoder's input is gene-keyed for
//! the dense path (`D_coarse`), and the dictionary tensors at every level are
//! gene-keyed too. Run on the same gene set, or train from scratch.

use crate::topic::model_metadata::TopicModelMetadata;
use candle_util::candle_nn::VarMap;
pub use candle_util::grow::Growth;

/// The per-gene embedding already grown onto this run's axis, when that axis
/// differs from the source run's. `n_source` is what the checkpoint's gene-keyed
/// dimensions are expected to read; `rho` replaces the checkpoint's ρ.
pub struct GrownRho {
    pub n_source: usize,
    pub rho: crate::embed_common::Mat,
}

/// Name of the one gene-keyed learned tensor in a masked checkpoint: the
/// encoder's per-gene embedding ρ. Everything else the decoders hold is keyed
/// to modules or topics.
const RHO_TENSOR: &str = "enc.feature.embeddings";

/// Architecture invariants the saved checkpoint must match.
pub struct WarmStartCheck<'a> {
    /// "topic" or "`indexed_topic`"
    pub model_type_expected: &'static str,
    pub n_topics: usize,
    pub n_features_full: usize,
    pub n_features_encoder: usize,
    pub encoder_hidden: &'a [usize],
    pub level_decoder_dims: &'a [usize],
    /// Set only for indexed; ignored for dense.
    pub embedding_dim: Option<usize>,
    /// When non-zero, `n_topics` / `embedding_dim` above are the *grown* sizes
    /// and the checkpoint is expected to be smaller by exactly this much.
    pub growth: Growth,
    /// `Some` when this run's gene axis differs from the source run's: the
    /// gene-keyed checks read against the source run's length, and ρ is taken from
    /// here rather than from the file. Masked family only.
    pub gene_axis: Option<GrownRho>,
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
        metadata.n_topics + expected.growth.add_topics == expected.n_topics,
        "warm-start: K mismatch (saved={} + {} added = {}, current={})",
        metadata.n_topics,
        expected.growth.add_topics,
        metadata.n_topics + expected.growth.add_topics,
        expected.n_topics,
    );
    anyhow::ensure!(
        metadata.encoder_hidden.as_slice() == expected.encoder_hidden,
        "warm-start: encoder_hidden mismatch (saved={:?}, current={:?})",
        metadata.encoder_hidden,
        expected.encoder_hidden,
    );
    // On a grown gene axis the checkpoint is keyed to the SOURCE run's length.
    let saved_features = expected
        .gene_axis
        .as_ref()
        .map_or(expected.n_features_full, |g| g.n_source);
    anyhow::ensure!(
        metadata.n_features_full == saved_features,
        "warm-start: n_features_full mismatch (saved={}, current={}). The saved weights are \
         keyed to the parent's gene axis, so it has to be the same axis.\n\
         \n\
         Absorbing a new cohort is the usual cause, and it splits two ways. If the axis GREW \
         because the two cohorts spell some genes differently, each unreconciled name became a \
         second row — reconcile them with the family's row-name canonicalization option (for \
         example `ENSG00000105329_TGFB1` vs `TGFB1`). If it grew because the new cohort really \
         does measure genes the model has never seen, those cannot be added to a trained \
         model: restrict the input to the parent's gene set, or re-train.",
        metadata.n_features_full,
        saved_features,
    );
    let saved_encoder = if expected.gene_axis.is_some() {
        saved_features
    } else {
        expected.n_features_encoder
    };
    anyhow::ensure!(
        metadata.n_features_encoder == saved_encoder,
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
        let saved_emb = metadata.embedding_dim.unwrap_or(0);
        anyhow::ensure!(
            saved_emb + expected.growth.add_embedding_dim == emb,
            "warm-start: embedding_dim mismatch (saved={:?} + {} added, current={})",
            metadata.embedding_dim,
            expected.growth.add_embedding_dim,
            emb,
        );
    } else {
        anyhow::ensure!(
            expected.growth.add_embedding_dim == 0,
            "warm-start: --add-embedding-dim has no meaning for a '{}' model — it has no \
             per-gene embedding ρ to widen. Only the masked family does.",
            expected.model_type_expected,
        );
    }
    let safetensors_path = format!("{prefix}.safetensors");
    log::info!("Warm-starting from {safetensors_path}");

    let dims = candle_util::grow::GrowthDims {
        k_old: metadata.n_topics,
        k_new: expected.n_topics,
        h_old: metadata.embedding_dim.unwrap_or(0),
        h_new: expected.embedding_dim.unwrap_or(0),
    };
    if let Some(grown) = expected.gene_axis.as_ref() {
        return load_with_grown_rho(parameters, &safetensors_path, &dims, &grown.rho);
    }

    if expected.growth.is_none() {
        // VarMap has interior mutability via Arc<Mutex<_>>; clone shares storage
        // and lets us call `.load()` (which takes `&mut self`) without forcing
        // every caller to thread a mutable reference through the pipeline.
        let mut handle = parameters.clone();
        handle.load(&safetensors_path)?;
        log::info!(
            "Warm-start: loaded {} variables from {prefix}",
            handle.all_vars().len()
        );
        return Ok(());
    }

    log::info!(
        "Warm-start with growth: K {} → {}, H {} → {}",
        dims.k_old,
        dims.k_new,
        dims.h_old,
        dims.h_new,
    );
    candle_util::grow::load_grown(parameters, &safetensors_path, &dims)
}

/// Load a checkpoint whose gene axis is not this run's.
///
/// Every tensor but ρ is module- or topic-keyed and loads as usual, grown on K
/// or H if asked. ρ is the one gene-keyed tensor, and it comes from `rho`,
/// already grown by name; if H grew too, the same slab fill the other tensors
/// get is applied on top.
fn load_with_grown_rho(
    parameters: &VarMap,
    path: &str,
    dims: &candle_util::grow::GrowthDims,
    rho: &crate::embed_common::Mat,
) -> anyhow::Result<()> {
    use matrix_util::traits::ConvertMatOps;
    let saved = candle_util::candle_core::safetensors::load(path, &candle_util::candle_core::Device::Cpu)?;
    let data = parameters.data().lock().expect("VarMap lock");
    let (mut n_copied, mut n_grown) = (0usize, 0usize);
    for (name, var) in data.iter() {
        let fresh = var.as_tensor();
        let s = if name == RHO_TENSOR {
            rho.to_tensor(var.device())?
        } else {
            saved
                .get(name)
                .ok_or_else(|| {
                    anyhow::anyhow!("warm-start: {path} has no tensor named `{name}`; architectures differ")
                })?
                .to_device(var.device())?
        };
        if s.dims() == fresh.dims() {
            var.set(&s)?;
            n_copied += 1;
        } else {
            anyhow::ensure!(
                s.rank() == fresh.rank(),
                "warm-start: `{name}` has rank {} in the checkpoint and {} here",
                s.rank(),
                fresh.rank(),
            );
            var.set(&candle_util::grow::grow_tensor(name, fresh, &s, dims)?)?;
            n_grown += 1;
        }
    }
    log::info!(
        "Warm-start on a grown gene axis: {n_copied} variables copied, {n_grown} grown; ρ has \
         {} rows",
        rho.nrows()
    );
    Ok(())
}
