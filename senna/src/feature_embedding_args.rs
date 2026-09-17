//! The `--freeze-feature-embedding` / `--init-feature-embedding` /
//! `--lora-feature-embedding` triple, shared by every model whose feature
//! side can start from an earlier run's table: `senna bge`, `senna simba`,
//! `senna fne`, `senna gem` and the `masked-*` family. One clap struct, so
//! the flags, their help and the "one of the three" rule read the same
//! everywhere.

use clap::Args;
use graph_embedding_util::{LoraArgs, PresetMode};

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
                     `senna gem` reads the table onto its row grammar: a bare gene name\n\
                     is the gene's count/spliced row, a `{gene}/{modality}/{channel}` name\n\
                     is that row. Base rows are pinned as above; a given row on another\n\
                     track pins that track's offset for its gene, and the offsets of\n\
                     every other gene train (see --offset-rank).\n\
                     \n\
                     The table's rows that match no feature of this run — genes the\n\
                     data lacks and every non-gene row — are carried through unchanged\n\
                     into the output ρ table, after the trained rows, so the result is\n\
                     the full table; `{out}.feature_types.parquet` names each row's type.\n\
                     \n\
                     H is taken from the table when `--embedding-dim` is `auto`;\n\
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
                     training from there instead of a random init. Its unmatched rows\n\
                     are not carried through: the trained rows leave the table's space.\n\
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
                     and the output ρ carries the residual folded in; the unmatched\n\
                     rows are carried through as under `--freeze-feature-embedding`.\n\
                     Rank 0 would be `--freeze-`, rank H would be `--init-feature-embedding`,\n\
                     so both are refused.\n\
                     One of `--freeze-`, `--init-` and `--lora-feature-embedding`."
    )]
    pub lora_feature_embedding: Option<Box<str>>,

    /// `--lora-rank`, `--lora-lr-ratio`, `--lora-ridge`; read with
    /// `--lora-feature-embedding` only.
    #[command(flatten)]
    #[serde(flatten)]
    pub lora: LoraArgs,
}

impl FeatureEmbeddingArgs {
    /// The run prefix given by whichever flag was used, and what to do with
    /// its rows. A LoRA knob given without `--lora-feature-embedding` is
    /// refused here: the knobs are a shared group and clap cannot tie them to
    /// this struct's own flag.
    pub fn resolve(&self) -> anyhow::Result<Option<(&str, PresetMode)>> {
        self.lora.refuse_unless_selected(
            self.lora_feature_embedding.is_some(),
            "--lora-feature-embedding",
        )?;
        if let Some(p) = self.lora_feature_embedding.as_deref() {
            return Ok(Some((p, PresetMode::Lora(self.lora.spec()))));
        }
        if let Some(p) = self.freeze_feature_embedding.as_deref() {
            return Ok(Some((p, PresetMode::Freeze)));
        }
        Ok(self
            .init_feature_embedding
            .as_deref()
            .map(|p| (p, PresetMode::Init)))
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
