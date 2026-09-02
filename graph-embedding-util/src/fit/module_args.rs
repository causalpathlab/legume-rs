//! The `--gene-modules` flag group, flattened by the CLIs that expose the
//! learned-module parameterization. Same pattern as
//! [`crate::posterior::PosteriorArgs`]: one `clap::Args` so every caller shows one
//! help text and resolves one way, and `serde` with `clap_defaults` so a manifest
//! written before a flag existed still deserializes.

use super::config::GeneModuleConfig;

/// Module count a CLI that turns modules on by default uses when `--gene-modules`
/// is not given. `senna bge` passes it to [`GeneModuleArgs::resolve`]; `pinto cage`
/// passes `None` (its default sampled gate is exclusive with modules).
pub const DEFAULT_GENE_MODULES: usize = 128;

#[derive(clap::Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "matrix_util::clap_defaults::clap_defaults")]
pub struct GeneModuleArgs {
    #[arg(
        long = "gene-modules",
        value_name = "M",
        help = "Learn M gene modules and embed genes THROUGH them (senna bge: on by default, M=128)",
        long_help = "Put a learned mixed-membership module layer in front of the feature embedding.\n\
                     Each gene's row becomes a sparse mixture of M shared module vectors,\n\
                     plus a small ridge-shrunk residual of its own.\n\
                     senna bge turns this on by default with M=128; --no-gene-modules turns it off.\n\
                     pinto cage leaves it off unless M is given.\n\
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
                     The membership starts from a k-means over the pseudobulk gene profiles\n\
                     and is held there for the first epochs (--module-warmup-epochs).\n\
                     A load-balance prior (--module-balance) keeps the modules from merging.\n\
                     Every epoch logs occupancy, dead modules, row entropy and support,\n\
                     which is where a collapse shows first.\n\
                     \n\
                     OUTPUTS. {out}.module_membership.parquet (gene x M), \n\
                     {out}.module_dictionary.parquet (M x H), {out}.module_residual.parquet,\n\
                     {out}.module_bias.parquet. The feature dictionary keeps holding the composed\n\
                     row, so nothing that reads it has to know modules exist.\n\
                     \n\
                     Cannot combine with --posterior."
    )]
    pub gene_modules: Option<usize>,

    #[arg(
        long = "no-gene-modules",
        conflicts_with = "gene_modules",
        help = "Train the plain free feature embedding instead of the module layer"
    )]
    pub no_gene_modules: bool,

    #[arg(
        long = "module-warmup-epochs",
        value_name = "N",
        help = "Epochs the warm-start membership is held before it trains (default: a quarter)",
        long_help = "Epochs the k-means membership is held fixed while the module vectors and\n\
                     residuals train. A membership released too early is rich-get-richer:\n\
                     the exact term rewards putting every gene in the module\n\
                     that already scores well. Default: a quarter of the epochs, at least one."
    )]
    pub module_warmup_epochs: Option<usize>,

    #[arg(
        long = "gene-dropout",
        value_name = "P",
        default_value_t = 0.3,
        help = "Per-step probability a gene is hidden when module counts are pooled",
        long_help = "Gene dropout at pooling time. Each step hides a random subset of genes\n\
                     when the module counts are formed, and rescales the survivors of each\n\
                     module so its expected count is unchanged.\n\
                     This is exactly the situation a later dataset with a different gene panel\n\
                     presents, so the module vectors are fit under that perturbation. 0 turns it off."
    )]
    pub gene_dropout: f32,

    #[arg(
        long = "module-balance",
        value_name = "LAMBDA",
        default_value_t = 1.0,
        help = "Weight of the load-balance prior on module occupancy",
        long_help = "Weight of KL(occupancy || uniform), where occupancy is the mean membership\n\
                     each module carries across genes. Zero when every module holds an equal share,\n\
                     largest when one module holds them all. A uniform target on purpose:\n\
                     for modules, a decaying prior is the collapse direction."
    )]
    pub module_balance: f32,

    #[arg(
        long = "module-weight",
        value_name = "LAMBDA",
        default_value_t = 1.0,
        help = "Weight of the exact cell-module term relative to the NCE",
        hide = true
    )]
    pub module_weight: f32,

    #[arg(
        long = "module-entropy",
        value_name = "LAMBDA",
        default_value_t = 0.0,
        help = "Row-entropy penalty on the membership (0 = off)",
        hide = true
    )]
    pub module_entropy: f32,

    #[arg(
        long = "module-residual-l2",
        value_name = "LAMBDA",
        default_value_t = 0.1,
        help = "Ridge on the per-gene residual (the module model's only per-row table)",
        hide = true
    )]
    pub module_residual_l2: f32,

    #[arg(
        long = "module-units-per-step",
        value_name = "U",
        default_value_t = 64,
        help = "Units pooled per step per axis for the exact module term",
        hide = true
    )]
    pub module_units_per_step: usize,

    #[arg(
        long = "module-init-mass",
        value_name = "P",
        default_value_t = 0.9,
        help = "Share of a gene's warm-start membership on its k-means module",
        hide = true
    )]
    pub module_init_mass: f32,
}

impl GeneModuleArgs {
    /// Resolve the group against the CLI's own default: `--no-gene-modules` wins,
    /// then an explicit `--gene-modules M`, then `default_on` (`Some(M)` for a CLI
    /// that trains modules unless told otherwise, `None` for an opt-in CLI).
    pub fn resolve(&self, default_on: Option<usize>) -> anyhow::Result<Option<GeneModuleConfig>> {
        if self.no_gene_modules {
            return Ok(None);
        }
        let Some(m) = self.gene_modules.or(default_on) else {
            return Ok(None);
        };
        anyhow::ensure!(m >= 2, "--gene-modules needs at least 2 modules, got {m}");
        anyhow::ensure!(
            (0.0..1.0).contains(&self.gene_dropout),
            "--gene-dropout must be in [0, 1), got {}",
            self.gene_dropout
        );
        anyhow::ensure!(
            self.module_init_mass > 0.0 && self.module_init_mass <= 1.0,
            "--module-init-mass must be in (0, 1], got {}",
            self.module_init_mass
        );
        Ok(Some(GeneModuleConfig {
            n_modules: m,
            warmup_epochs: self.module_warmup_epochs,
            gene_dropout: self.gene_dropout,
            lambda_module: self.module_weight,
            lambda_balance: self.module_balance,
            lambda_entropy: self.module_entropy,
            residual_l2: self.module_residual_l2,
            units_per_step: self.module_units_per_step,
            init_own_mass: self.module_init_mass,
            parent: None,
        }))
    }
}

#[cfg(test)]
mod tests;
