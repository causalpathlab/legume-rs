//! Allocating the embedding heads: one primary model that owns the feature side, plus
//! one pseudobulk head per collapse level that SHARES it.
//!
//! Split out because the allocation order is load-bearing and easy to break silently.
//! The primary head registers the canonical feature Vars, the gate is enabled on it
//! BEFORE any head exists, and each head then clones handles to those same Vars — so a
//! head built too early, or one that misses a handle, trains against a feature side
//! nothing else sees. That last failure mode already shipped once: `gate_ibp_bias` was
//! not among the cloned handles, and the gate's whole KL silently left the loss.

use super::config::FitConfig;
use crate::data::UnifiedData;
use crate::model::{
    FactoredInit, JointEmbedModel, ModelArgs, ModelInit, ModuleInit, ShareFeaturesArgs,
};
use candle_util::candle_nn::{VarBuilder, VarMap};
use log::info;

/// The primary (per-cell) head and one head per pseudobulk level, coarsest → finest.
pub(super) struct Heads {
    pub cell_model: JointEmbedModel,
    pub level_models: Vec<JointEmbedModel>,
}

/// Allocate every head against one `VarMap`.
///
/// The primary head allocates the canonical feature-side Vars — `e_feat`/`b_feat` for a
/// free model, `beta`/`b_feat` when β-sharing factored — and every level head then
/// shares that feature side while registering its own cell side under a unique
/// `pb_l{idx}` prefix. AdamW over `varmap.all_vars()` therefore updates the feature side
/// once and each head's cell side independently.
pub(super) fn build_heads(
    unified: &UnifiedData,
    pb_blobs: &[UnifiedData],
    config: &FitConfig,
    module_warm: Option<&[u32]>,
    varmap: &VarMap,
) -> anyhow::Result<Heads> {
    let (n_features, n_cells, h) = (
        unified.n_features(),
        unified.n_cells(),
        config.embedding_dim,
    );
    let vs = VarBuilder::from_varmap(varmap, candle_util::candle_core::DType::F32, &config.device);
    let zeros_features = vec![0f32; n_features];
    let zeros_cells = vec![0f32; n_cells];

    let mut cell_model = match (&config.gene_modules, &config.feat_factor) {
        (Some(gm), factor) => {
            // Hard errors, not silent no-ops: the module model's `e_feat` is a
            // composed snapshot, and both of these write or read it as the trained
            // table.
            anyhow::ensure!(
                factor.is_none(),
                "gene modules are not supported with feat_factor (β-sharing) in this version"
            );
            anyhow::ensure!(
                config.pb_posterior.is_none(),
                "gene modules cannot combine with the phase-1 posterior sampler: the \
                 selection pass writes per-feature loadings into a table the module \
                 model composes rather than trains"
            );
            if config.feature_embedding_l2 > 0.0 {
                info!(
                    "gene modules: feature_embedding_l2 is ignored; the residual ridge \
                     ({}) is the module model's per-row shrinkage",
                    gm.residual_l2
                );
            }
            info!(
                "learned gene modules: {} features → {} modules (mixed membership), \
                 gene dropout {}, exact module term λ={}, balance λ={}",
                n_features, gm.n_modules, gm.gene_dropout, gm.lambda_module, gm.lambda_balance
            );
            JointEmbedModel::new_with_modules(
                ModuleInit {
                    n_features,
                    n_cells,
                    embedding_dim: h,
                    n_modules: gm.n_modules,
                    init_labels: module_warm,
                    init_own_mass: gm.init_own_mass,
                    b_feat: &zeros_features,
                    b_cell: &zeros_cells,
                    seed: config.seed,
                },
                varmap,
                &config.device,
            )?
        }
        (None, Some(spec)) => {
            // β-sharing is incompatible with the free-E_feat L2, which assumes a single
            // free feature table per row.
            anyhow::ensure!(
                config.feature_embedding_l2 == 0.0,
                "feat_factor (β-sharing) is not supported with feature_embedding_l2 > 0"
            );
            anyhow::ensure!(
                spec.row_to_gene.len() == n_features && spec.unspliced_rows.len() == n_features,
                "feat_factor row maps (row_to_gene {}, unspliced_rows {}) must match n_features {}",
                spec.row_to_gene.len(),
                spec.unspliced_rows.len(),
                n_features
            );
            // Dense gene ids ⇒ the count is max + 1. Single source of truth: the row→gene
            // map, with no separately-supplied `n_genes` to keep in sync.
            let n_genes = spec
                .row_to_gene
                .iter()
                .copied()
                .max()
                .map_or(0, |m| m as usize + 1);
            info!(
                "per-gene β-sharing factorization: {} features → {} genes ({} unspliced rows); \
                 splice δ recovered post-hoc on the cell axis (dual phase-2 projection)",
                n_features,
                n_genes,
                spec.unspliced_rows.iter().filter(|&&b| b).count(),
            );
            // Allocate the ridge-shrunk per-gene splice offset δ_g only when its L2
            // penalty is on; otherwise plain β-sharing (spliced ≡ unspliced ≡ β_g).
            let unspliced_rows = (config.delta_l2 > 0.0).then_some(spec.unspliced_rows.as_slice());
            if unspliced_rows.is_some() {
                info!(
                    "δ_g splice offset ON (L2={}): unspliced rows embed as β_g + δ_g",
                    config.delta_l2
                );
            }
            JointEmbedModel::new_factored(
                FactoredInit {
                    n_features,
                    n_cells,
                    embedding_dim: h,
                    n_genes,
                    row_to_gene: &spec.row_to_gene,
                    b_feat: &zeros_features,
                    b_cell: &zeros_cells,
                    seed: config.seed,
                    unspliced_rows,
                },
                varmap,
                vs,
                &config.device,
            )?
        }
        (None, None) => JointEmbedModel::new_with_init(
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
        )?,
    };

    // BEFORE the sharing heads exist, so every head references the one gate Var
    // (`s_feat` free / `s_beta` factored) rather than allocating its own.
    if let Some(g) = config.feature_gate {
        cell_model.enable_feature_gate(g, varmap, &config.device)?;
        info!(
            "Learned feature gate ON — an INDEPENDENT Bernoulli inclusion probability \
             σ(S) per (feature, dim), with each dim's rate π_h learned. τ={}",
            g.temperature
        );
    }

    let mut level_models: Vec<JointEmbedModel> = Vec::with_capacity(pb_blobs.len());
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        let prefix = format!("pb_l{level_idx}");
        // Each level's cell side is keyed by its own `{prefix}_e_cell` name, so one base
        // seed yields an independent reproducible init per level.
        let level_model = if cell_model.factor.is_some() {
            cell_model.new_sharing_factor(n_pb, &prefix, varmap, &config.device, config.seed)?
        } else {
            JointEmbedModel::new_sharing_features(
                ShareFeaturesArgs {
                    n_cells: n_pb,
                    embedding_dim: h,
                    shared_e_feat: cell_model.e_feat.clone(),
                    shared_b_feat: cell_model.b_feat.clone(),
                    e_cell_init: None,
                    b_cell_init: &vec![0f32; n_pb],
                    var_prefix: &prefix,
                    seed: config.seed,
                    // EVERY gate handle, so each head reweights the SAME feature side and
                    // AdamW updates one `s_feat`. `shared_gate_ibp_bias` is the one that
                    // was missing: `train_composite` evaluates the gate KL on `axes[0]`,
                    // which is a pb head whenever `--phase1-cells-per-pb` is 0 (the
                    // default), so without it the KL returns `None` on every ordinary run.
                    shared_s_feat: cell_model.s_feat.clone(),
                    shared_e_feat_raw: cell_model.e_feat_raw.clone(),
                    shared_e_feat_logstd: cell_model.e_feat_logstd.clone(),
                    shared_gate_ibp_bias: cell_model.gate_ibp_bias.clone(),
                    gate: cell_model.gate,
                    shared_modules: cell_model.modules.clone(),
                },
                varmap,
                &config.device,
            )?
        };
        level_models.push(level_model);
    }
    Ok(Heads {
        cell_model,
        level_models,
    })
}
