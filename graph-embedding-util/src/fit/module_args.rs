//! The `--gene-modules` flag group, flattened by the CLIs that expose the
//! learned-module parameterization: one `clap::Args` so every caller shows one
//! help text and resolves one way, and `serde` with `clap_defaults` so a manifest
//! written before a flag existed still deserializes.

use super::config::GeneModuleConfig;

/// The warm-start membership is held fixed for a quarter of the epochs before it
/// trains — [`GeneModuleConfig::warmup_epochs_for`]'s own fallback, so `None`
/// here reproduces it exactly.
const DEFAULT_WARMUP_EPOCHS: Option<usize> = None;
/// Per-step probability a gene is hidden when module counts are pooled.
const DEFAULT_GENE_DROPOUT: f32 = 0.3;
/// Weight of the exact cell-module term relative to the NCE.
const DEFAULT_MODULE_WEIGHT: f32 = 1.0;
/// Weight of the load-balance prior `KL(π̄ ‖ Uniform)`.
const DEFAULT_MODULE_BALANCE: f32 = 1.0;
/// Ridge on the per-gene residual (the module model's only per-row table).
const DEFAULT_RESIDUAL_L2: f32 = 0.1;
/// Units (cells or pseudobulks) pooled per step per axis for the exact term.
const DEFAULT_UNITS_PER_STEP: usize = 64;
/// Share of a gene's warm-start membership on its k-means module.
const DEFAULT_INIT_MASS: f32 = 0.9;

#[derive(clap::Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "matrix_util::clap_defaults::clap_defaults")]
pub struct GeneModuleArgs {
    #[arg(
        long = "gene-modules",
        value_name = "M",
        help = "Learn M gene modules and embed genes THROUGH them",
        long_help = "Put a learned mixed-membership module layer in front of the feature embedding.\n\
                     Each gene's row becomes a sparse mixture of M shared module vectors,\n\
                     plus a small ridge-shrunk residual of its own.\n\
                     Whether it is on without this flag is the command's own choice — see its help.\n\
                     \n\
                     WHY. A free row receives gradient only on the steps that draw its gene.\n\
                     A rare gene, or one absent from a later dataset, has nothing standing in for it.\n\
                     Through a module, every edge on any member trains the shared vector,\n\
                     so a gene that is never drawn still inherits a trained row from its siblings.\n\
                     \n\
                     HOW. Membership is sparsemax over learned logits: a gene sits on a few modules\n\
                     with exact zeros elsewhere, never on a uniform average of all of them.\n\
                     Two terms train it. An EXACT softmax over all M modules scores each unit's\n\
                     module-pooled counts, so every module gets gradient on every step.\n\
                     The usual NCE then draws its negatives from the positive gene's own module,\n\
                     so it resolves genes WITHIN a module and leaves the rest to the exact term.\n\
                     \n\
                     OUTPUTS. {out}.module_membership.parquet (gene x M), \n\
                     {out}.module_dictionary.parquet (M x H), {out}.module_residual.parquet,\n\
                     {out}.module_bias.parquet. The feature dictionary keeps holding the composed\n\
                     row, so nothing that reads it has to know modules exist."
    )]
    pub gene_modules: Option<usize>,
}

impl GeneModuleArgs {
    /// Resolve the group against the CLI's own default: an explicit
    /// `--gene-modules M`, else `default_on` (`Some(M)` for a CLI that trains
    /// modules unless told otherwise, `None` for an opt-in CLI). Every other knob
    /// is fixed at the value the flags used to default to.
    pub fn resolve(&self, default_on: Option<usize>) -> anyhow::Result<Option<GeneModuleConfig>> {
        let Some(m) = self.gene_modules.or(default_on) else {
            return Ok(None);
        };
        anyhow::ensure!(m >= 2, "--gene-modules needs at least 2 modules, got {m}");
        Ok(Some(GeneModuleConfig {
            n_modules: m,
            warmup_epochs: DEFAULT_WARMUP_EPOCHS,
            gene_dropout: DEFAULT_GENE_DROPOUT,
            lambda_module: DEFAULT_MODULE_WEIGHT,
            lambda_balance: DEFAULT_MODULE_BALANCE,
            residual_l2: DEFAULT_RESIDUAL_L2,
            units_per_step: DEFAULT_UNITS_PER_STEP,
            init_own_mass: DEFAULT_INIT_MASS,
            parent: None,
        }))
    }
}

#[cfg(test)]
mod tests;
