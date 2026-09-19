//! `pinto annotate` moved to `lupin annotate`.
//!
//! This module keeps only the clap [`AnnotateArgs`] surface so the old
//! subcommand still parses, then prints a migration error from `main`.

use clap::Args;

#[derive(Debug, Clone, Args)]
pub struct AnnotateArgs {
    #[arg(
        long,
        help = "Run prefix: reads `{prefix}.feature_embedding.parquet` + `{prefix}.cell_embedding.parquet`"
    )]
    pub from: Option<Box<str>>,

    #[arg(long, help = "Feature × D embedding parquet (overrides --from)")]
    pub feature_embedding: Option<Box<str>>,

    #[arg(long, help = "Cell × D embedding parquet (overrides --from)")]
    pub cell_embedding: Option<Box<str>>,

    #[arg(long, short = 'm', help = "Marker TSV: gene, cell_type")]
    pub markers: Box<str>,

    #[arg(long, short = 'o', help = "Output prefix (defaults to --from)")]
    pub out: Option<Box<str>>,

    #[arg(long, default_value_t = 30)]
    pub knn: usize,

    #[arg(long, default_value_t = 1.0)]
    pub resolution: f64,

    #[arg(long, default_value_t = 500)]
    pub num_perm: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, help = "Disable IDF weights on marker genes")]
    pub no_idf: bool,

    #[arg(long, help = "Skip distance-outlier QC prune")]
    pub no_assign_qc: bool,

    #[arg(long, default_value_t = 2.5)]
    pub assign_mad: f32,

    #[arg(long, default_value_t = 0.1)]
    pub fdr_alpha: f32,

    #[arg(long, default_value_t = 1.0)]
    pub q_temperature: f32,

    #[arg(long, help = "Cell Ontology OBO for TreeBH")]
    pub obo: Option<Box<str>>,

    #[arg(long, help = "marker_label → CL:ID map TSV")]
    pub label_cl: Option<Box<str>>,

    #[arg(long, default_value_t = 0.1)]
    pub ontology_fdr_q: f64,

    #[arg(long, help = "Benjamini–Yekutieli within ontology families")]
    pub ontology_by: bool,
}
