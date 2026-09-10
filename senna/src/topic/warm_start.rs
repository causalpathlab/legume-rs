//! Warm-start initialization for `senna topic` and `senna masked-topic`.
//!
//! Loads weights saved by a previous training run into the current `VarMap`,
//! so training continues from the previous checkpoint instead of from
//! random init. The current architecture must match the saved one: same K,
//! encoder layers, level decoder dims, and (for indexed) embedding dim and
//! `n_features_full`.
//!
//! A different gene list is accepted only when the caller hands over a
//! [`GeneAxisGrowth`]: the checkpoint's gene-keyed tensors are then gathered
//! onto this run's order by name (`candle_util::grow`). Without one the axes
//! must match exactly.

use crate::topic::model_metadata::TopicModelMetadata;
use candle_util::candle_nn::VarMap;
pub use candle_util::grow::Growth;

/// This run's gene axis is not the source run's: how the two align, and the
/// modules an unseen gene was placed in.
pub struct GeneAxisGrowth<'a> {
    pub remap: &'a crate::topic::eval::GeneRemap,
    /// Where an unseen gene's per-gene embedding restarts: at the mean of its
    /// module's known members rather than at the global mean the loader gives
    /// it. `None` when there is nothing to refine — the family has no per-gene
    /// embedding at all, or the source run trained at full resolution and has
    /// no modules to take a mean over.
    pub modules: Option<&'a data_beans_alg::feature_coarsening::FeatureCoarsening>,
}

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
    /// gene-keyed checks read against the source run's length, and the
    /// gene-keyed tensors are gathered onto this run's order by name.
    pub gene_axis: Option<GeneAxisGrowth<'a>>,
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
        .map_or(expected.n_features_full, |g| g.remap.d_train);
    anyhow::ensure!(
        metadata.n_features_full == saved_features,
        "warm-start: n_features_full mismatch (saved={}, current={}).\n\
         \n\
         A gene axis this run does not share with the source run is continued by name, so \
         reaching this means the source run's own records disagree with each other: its \
         model.json says one axis length and its feature_mean.parquet another. That is a \
         copied or half-written prefix rather than anything about this cohort — check \
         --init-from.",
        metadata.n_features_full,
        saved_features,
    );
    // A width equal to the gene axis is gene-keyed and follows the axis across
    // a grown one; a module width is not and must match exactly. This covers
    // the encoder's input and every level's decoder output, which is the whole
    // model at `--max-coarse-features 0`.
    let on_source_axis = |width: usize| {
        if width == expected.n_features_full {
            saved_features
        } else {
            width
        }
    };
    let saved_encoder = on_source_axis(expected.n_features_encoder);
    anyhow::ensure!(
        metadata.n_features_encoder == saved_encoder,
        "warm-start: n_features_encoder (D_coarse) mismatch (saved={}, current={}). \
         Coarsening parameters must match the original run.",
        metadata.n_features_encoder,
        saved_encoder,
    );
    let saved_decoders: Vec<usize> = expected
        .level_decoder_dims
        .iter()
        .map(|&w| on_source_axis(w))
        .collect();
    anyhow::ensure!(
        metadata.level_decoder_dims == saved_decoders,
        "warm-start: level_decoder_dims mismatch (saved={:?}, current={:?})",
        metadata.level_decoder_dims,
        saved_decoders,
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
        gene_axis: expected.gene_axis.as_ref().map(|g| candle_util::grow::AxisRemap {
            new_to_old: &g.remap.new_to_train,
            n_old: g.remap.d_train,
        }),
    };
    if let Some(g) = expected.gene_axis.as_ref() {
        log::info!(
            "Warm-start on a gene axis of {} ({} of the checkpoint's {} known)",
            g.remap.new_to_train.len(),
            g.remap.n_mapped,
            g.remap.d_train,
        );
        candle_util::grow::load_grown(parameters, &safetensors_path, &dims)?;
        if let Some(modules) = g.modules {
            crate::topic::gene_axis::refine_rho_by_module(parameters, g.remap, modules)?;
        }
        return Ok(());
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
