//! The `--freeze-feature-embedding` / `--init-feature-embedding` /
//! `--lora-feature-embedding` triple, shared by every model whose feature
//! side can start from an earlier run's table: `senna bge`, `senna simba`,
//! `senna fne` and the `masked-*` family. One clap struct, so the flags, their
//! help and the "one of the three" rule read the same everywhere.

use clap::Args;
use graph_embedding_util::{LoraSpec, PresetMode};

#[derive(Args, Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FeatureEmbeddingArgs {
    #[arg(
        long,
        value_name = "PREFIX",
        conflicts_with_all = ["init_feature_embedding", "lora_feature_embedding"],
        help = "Pin the gene embedding ρ to an earlier run's table; everything else trains",
        long_help = "Pin the per-gene embedding ρ to an earlier run's feature table,\n\
                     given by that run's output prefix.\n\
                     It resolves `{prefix}.feature_loading.parquet`,\n\
                     else `{prefix}.dictionary.parquet` or `{prefix}.feature_embedding.parquet`,\n\
                     accepting only a signed table (a log-simplex dictionary is refused).\n\
                     A `senna fne` run qualifies: its table also holds terms, words and\n\
                     cell types, and only its gene rows are read.\n\
                     \n\
                     Genes match by canonical name (`ENSG…_TP53` is `TP53`).\n\
                     A matched gene keeps its row for the whole fit; the rest of the\n\
                     model trains as usual.\n\
                     A gene with no row: `bge` keeps it as a free, trained row;\n\
                     the masked models drop it from the feature axis.\n\
                     \n\
                     H is taken from the table when `--embedding-dim` is 0;\n\
                     an explicit `--embedding-dim` must agree with it."
    )]
    pub freeze_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        value_name = "PREFIX",
        conflicts_with = "lora_feature_embedding",
        help = "Start the gene embedding ρ from an earlier run's table; it keeps training",
        long_help = "Warm-start the per-gene embedding ρ from an earlier run's feature table.\n\
                     The table is found and matched exactly as for\n\
                     `--freeze-feature-embedding`; the difference is that ρ keeps\n\
                     training from there instead of a random init.\n\
                     One of `--freeze-`, `--init-` and `--lora-feature-embedding`."
    )]
    pub init_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        value_name = "PREFIX",
        help = "Anchor ρ to an earlier run's table and train a low-rank residual on top",
        long_help = "Anchor the per-gene embedding ρ to an earlier run's feature table and\n\
                     train a low-rank residual on top: ρ_g = ρ₀_g + u_g · V, with u_g\n\
                     per gene (`--lora-rank` numbers) and V shared by every anchored gene.\n\
                     The table is found and matched exactly as for\n\
                     `--freeze-feature-embedding`; the given row ρ₀_g never moves,\n\
                     and the output ρ carries the residual folded in.\n\
                     Rank 0 would be `--freeze-`, rank H would be `--init-feature-embedding`,\n\
                     so both are refused.\n\
                     One of `--freeze-`, `--init-` and `--lora-feature-embedding`."
    )]
    pub lora_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        value_name = "R",
        requires = "lora_feature_embedding",
        help = "Rank of the LoRA residual (with --lora-feature-embedding; default 16)",
        long_help = "Rank of the LoRA residual, with `--lora-feature-embedding`; default 16.\n\
                     On `bge`, whose gene table is a module dictionary plus per-gene\n\
                     residuals, the same rank serves two residuals: one on the module\n\
                     dictionary (a module's genes move together) and one on the gene\n\
                     rows (a gene moves on its own)."
    )]
    pub lora_rank: Option<usize>,

    #[arg(
        long,
        value_name = "RATIO",
        requires = "lora_feature_embedding",
        help = "LoRA+: the shared factor V trains at this multiple of the learning rate; 1 = plain LoRA (default 4)",
        long_help = "LoRA+: the shared factor V, which starts at zero and is touched by every\n\
                     step, trains at this multiple of the learning rate; the per-gene\n\
                     factor u keeps the base rate. 1 is plain LoRA; the default is 4."
    )]
    pub lora_lr_ratio: Option<f32>,

    #[arg(
        long,
        value_name = "LAMBDA",
        requires = "lora_feature_embedding",
        help = "Ridge on the LoRA residual, per epoch and per anchored row (default 0.05)",
        long_help = "Ridge on the LoRA residual: `λ · Σ_g ‖u_g·V‖²` per epoch, spread over\n\
                     the epoch's steps, on each residual (bge: module and gene). Per row,\n\
                     because the data gradient on the shared factor is a sum over the\n\
                     anchored rows, so one weight means the same thing at any table size.\n\
                     The shrinkage that keeps the shared factor from marching off the\n\
                     anchor under a row optimizer. 0 is none; the default keeps the\n\
                     residual below the anchor's own scale."
    )]
    pub lora_ridge: Option<f32>,
}

impl FeatureEmbeddingArgs {
    /// The run prefix given by whichever flag was used, and what to do with
    /// its rows.
    pub fn resolve(&self) -> Option<(&str, PresetMode)> {
        if let Some(p) = self.freeze_feature_embedding.as_deref() {
            return Some((p, PresetMode::Freeze));
        }
        if let Some(p) = self.init_feature_embedding.as_deref() {
            return Some((p, PresetMode::Init));
        }
        let d = LoraSpec::default();
        self.lora_feature_embedding.as_deref().map(|p| {
            (
                p,
                PresetMode::Lora(LoraSpec {
                    rank: self.lora_rank.unwrap_or(d.rank),
                    lr_ratio: self.lora_lr_ratio.unwrap_or(d.lr_ratio),
                    ridge: self.lora_ridge.unwrap_or(d.ridge),
                }),
            )
        })
    }
}

/// The flag that selects `mode`, for messages.
#[must_use]
pub fn flag_name(mode: PresetMode) -> &'static str {
    match mode {
        PresetMode::Freeze => "--freeze-feature-embedding",
        PresetMode::Init => "--init-feature-embedding",
        PresetMode::Lora(_) => "--lora-feature-embedding",
    }
}

#[cfg(test)]
#[path = "feature_embedding_args_tests.rs"]
mod tests;
