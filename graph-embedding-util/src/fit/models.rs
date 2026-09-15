//! Allocating the embedding heads: one primary model that owns the feature side, plus
//! one pseudobulk head per collapse level that SHARES it.
//!
//! Split out because the allocation order is load-bearing and easy to break silently.
//! The primary head registers the canonical feature Vars BEFORE any head exists, and
//! each head then clones handles to those same Vars — so a head built too early, or
//! one that misses a handle, trains against a feature side nothing else sees. That
//! last failure mode has shipped once already, as a missing handle whose whole term
//! silently left the loss.

use super::config::FitConfig;
use crate::data::UnifiedData;
use crate::model::{JointEmbedModel, ModelArgs, ModelInit, ShareFeaturesArgs};
use candle_util::candle_nn::VarMap;

/// The primary (per-cell) head and one head per pseudobulk level, coarsest → finest.
pub(super) struct Heads {
    pub cell_model: JointEmbedModel,
    pub level_models: Vec<JointEmbedModel>,
}

/// Allocate every head against one `VarMap`.
///
/// The primary head allocates the canonical feature-side Vars — `e_feat`/`b_feat`
/// — and every level head then shares that feature side while registering its
/// own cell side under a unique `pb_l{idx}` prefix. AdamW over
/// `varmap.all_vars()` therefore updates the feature side once and each head's
/// cell side independently.
pub(super) fn build_heads(
    unified: &UnifiedData,
    pb_blobs: &[UnifiedData],
    config: &FitConfig,
    varmap: &VarMap,
) -> anyhow::Result<Heads> {
    let (n_features, n_cells, h) = (
        unified.n_features(),
        unified.n_cells(),
        config.embedding_dim,
    );
    let zeros_features = vec![0f32; n_features];
    let zeros_cells = vec![0f32; n_cells];

    // Phase 1 trains by the hierarchical engine, which writes its composed
    // dictionary into this free feature table. Gene modules are the hier
    // engine's own partition, never a head parameterization here.
    let cell_model = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features,
            n_cells,
            embedding_dim: h,
            seed: config.seed,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &zeros_features,
            b_cell: &zeros_cells,
        },
        varmap,
        &config.device,
    )?;

    let mut level_models: Vec<JointEmbedModel> = Vec::with_capacity(pb_blobs.len());
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        let prefix = format!("pb_l{level_idx}");
        // Each level's cell side is keyed by its own `{prefix}_e_cell` name, so one base
        // seed yields an independent reproducible init per level.
        let level_model = JointEmbedModel::new_sharing_features(
            ShareFeaturesArgs {
                n_cells: n_pb,
                embedding_dim: h,
                shared_e_feat: cell_model.e_feat.clone(),
                shared_b_feat: cell_model.b_feat.clone(),
                e_cell_init: None,
                b_cell_init: &vec![0f32; n_pb],
                var_prefix: &prefix,
                seed: config.seed,
                shared_modules: cell_model.modules.clone(),
            },
            varmap,
            &config.device,
        )?;
        level_models.push(level_model);
    }
    Ok(Heads {
        cell_model,
        level_models,
    })
}
