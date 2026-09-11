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
/// coarse group an unseen gene was placed in.
pub struct GeneAxisGrowth<'a> {
    pub remap: &'a crate::topic::eval::GeneRemap,
    /// Where an unseen gene's FREE per-gene embedding restarts: at the mean of
    /// its coarse group's known members rather than at the global mean the
    /// loader gives it. `None` when there is nothing to refine — the family has
    /// no per-gene embedding, or the source run trained at full resolution and
    /// has no coarsening to take a mean over. Ignored, present or not, when the
    /// feature side is composed from learned modules: there `n_gene_modules`
    /// decides, and a gained gene's membership starts flat by its own rule.
    pub coarsening: Option<&'a data_beans_alg::feature_coarsening::FeatureCoarsening>,
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
    /// Learned gene modules `M` this run registers; `0` for a free per-gene
    /// table, which is every family but the masked one.
    pub n_gene_modules: usize,
    /// `Some` when this run's gene axis differs from the source run's: the
    /// gene-keyed checks read against the source run's length, and the
    /// gene-keyed tensors are gathered onto this run's order by name.
    pub gene_axis: Option<GeneAxisGrowth<'a>>,
}

/// Whether a checkpoint can be continued onto the feature side this run asks
/// for, decided from the checkpoint's metadata alone.
///
/// Callable before any data is read, which is where it belongs: the answer is
/// one small JSON away, and the alternative is telling a user their flags do
/// not compose after the import, the QC and the collapse. [`warm_start_load`]
/// runs the same two checks against the metadata it has already loaded, so
/// every entry point is covered whether or not it called this first.
pub(crate) fn check_feature_side(
    prefix: &str,
    n_gene_modules: usize,
    add_embedding_dim: usize,
) -> anyhow::Result<()> {
    let metadata = TopicModelMetadata::load(prefix)?;
    check_gene_modules(metadata.gene_modules(), n_gene_modules)?;
    check_embedding_growth(n_gene_modules, add_embedding_dim)
}

/// Whether the checkpoint's feature side is the one this run registers.
///
/// `M` decides which weights exist, not merely how wide they are: at `0` the
/// encoder holds a free `[D, H]` table, and above it a `[D, M]` membership with
/// an `[M, H]` dictionary. Continuing across that switch asks the loader for
/// tensors the checkpoint does not contain; continuing between two counts asks
/// for a membership of a different width, which cannot be reached by appending
/// (`candle_util::feature_embedding`). Both are refused by the flag that caused
/// them, rather than further down by a missing variable name.
fn check_gene_modules(saved: usize, current: usize) -> anyhow::Result<()> {
    if saved == current {
        return Ok(());
    }
    let feature_side = |m: usize| match m {
        0 => "a free per-gene embedding".to_string(),
        m => format!("{m} learned modules"),
    };
    // Two different refusals wearing one shape. Crossing 0 is a different model:
    // the two register different weights and no amount of work makes one
    // continue the other. Moving between two counts is a capability this does
    // not have YET — `M` moves by splitting loaded modules, since an appended
    // one is outside every feature's support and sparsemax would never give it
    // a gradient. Saying so keeps the message from promising the restriction is
    // permanent when only the first half of it is.
    let why = if saved == 0 || current == 0 {
        "the two register different weights, so there is nothing to continue"
    } else {
        "changing M on a continued fit means splitting the loaded modules, which is not \
         implemented yet: a larger M would only append modules that sparsemax can never give \
         a gradient to, and a smaller one would discard trained ones"
    };
    anyhow::bail!(
        "warm-start: --gene-modules mismatch. The source run composes its feature side from \
         {}, this run from {}, and {why}. Pass --gene-modules {saved} to continue this \
         checkpoint, or drop --init-from to train the feature side you asked for from scratch.",
        feature_side(saved),
        feature_side(current),
    );
}

/// Whether `--add-embedding-dim` means anything for this feature side.
///
/// The encoder's input is the composed embedding followed by each module's
/// level and its coverage, `[H | M | M]`. Widening `H` therefore inserts
/// columns in the MIDDLE of a trained weight, and the leading-corner copy every
/// other growth uses would slide the two module halves onto the wrong inputs.
/// Refused by the flags that asked for it, rather than by a shape mismatch
/// several tensors later.
fn check_embedding_growth(n_gene_modules: usize, add_embedding_dim: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        n_gene_modules == 0 || add_embedding_dim == 0,
        "warm-start: --add-embedding-dim {add_embedding_dim} does not compose with \
         --gene-modules {n_gene_modules}. The encoder reads a composed embedding followed by \
         each module's level and coverage, so a wider embedding is inserted into the middle of \
         the trained input weight rather than appended to it. Continue at the checkpoint's \
         embedding width, or train the wider one from scratch.",
    );
    Ok(())
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
    check_gene_modules(metadata.gene_modules(), expected.n_gene_modules)?;
    check_embedding_growth(expected.n_gene_modules, expected.growth.add_embedding_dim)?;
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
        gene_axis: expected
            .gene_axis
            .as_ref()
            .map(|g| candle_util::grow::AxisRemap {
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
        // Which feature side this is, is known here and nowhere below: a free
        // table gets its unseen rows refined, a composed one already has the
        // flat membership `grow` gave it, which composes the dictionary's
        // centroid.
        if expected.n_gene_modules > 0 {
            log::info!(
                "Warm-start: the feature side is composed from {} learned modules, so a gene \
                 the source run did not have starts flat, at the dictionary's centroid",
                expected.n_gene_modules,
            );
        } else if let Some(coarsening) = g.coarsening {
            crate::topic::gene_axis::refine_rho_by_coarsening(parameters, g.remap, coarsening)?;
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

#[cfg(test)]
#[path = "warm_start_tests.rs"]
mod warm_start_tests;
