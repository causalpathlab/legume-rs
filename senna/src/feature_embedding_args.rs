//! The `--freeze-feature-embedding` / `--init-feature-embedding` pair, shared
//! by every model whose feature side can start from an earlier run's table:
//! `senna bge` and the `masked-*` family. One clap struct, so the flags, their
//! help and the "one or the other" rule read the same everywhere.

use clap::Args;

#[derive(Args, Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FeatureEmbeddingArgs {
    #[arg(
        long,
        value_name = "PREFIX",
        conflicts_with = "init_feature_embedding",
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
        help = "Start the gene embedding ρ from an earlier run's table; it keeps training",
        long_help = "Warm-start the per-gene embedding ρ from an earlier run's feature table.\n\
                     The table is found and matched exactly as for\n\
                     `--freeze-feature-embedding`; the difference is that ρ keeps\n\
                     training from there instead of a random init.\n\
                     Mutually exclusive with `--freeze-feature-embedding`."
    )]
    pub init_feature_embedding: Option<Box<str>>,
}

impl FeatureEmbeddingArgs {
    /// The run prefix given by either flag, and whether it is to be pinned.
    pub fn resolve(&self) -> Option<(&str, bool)> {
        match (
            self.freeze_feature_embedding.as_deref(),
            self.init_feature_embedding.as_deref(),
        ) {
            (Some(p), _) => Some((p, true)),
            (None, Some(p)) => Some((p, false)),
            (None, None) => None,
        }
    }
}
