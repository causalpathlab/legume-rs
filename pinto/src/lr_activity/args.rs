//! CLI arguments for the `pinto lr-activity` subcommand.

use crate::lr_activity::matcher::NullScheme;
use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct SrtLrActivityArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        long,
        required = true,
        help = "Prefix of a prior `pinto lc` run (reads {prefix}.link_community.parquet, {prefix}.coord_pairs.parquet)",
        long_help = "Prefix of a prior `pinto lc` run. Reads edge→community assignments\n\
                     from {prefix}.link_community.parquet and cell-pair metadata\n\
                     (including per-edge batch labels when present) from\n\
                     {prefix}.coord_pairs.parquet. Row order in both files must\n\
                     match — they are joined by position."
    )]
    pub lc_prefix: Box<str>,

    #[arg(
        long,
        required = true,
        help = "Two-column TSV of directional ligand→receptor pairs (ligand\\treceptor)",
        long_help = "Directional ligand→receptor pair file. One pair per line, with\n\
                     two whitespace/tab/comma-separated columns: ligand gene,\n\
                     receptor gene. Delimiter auto-detected (.csv → comma, else tab).\n\
                     Gene names must match row names in the expression data; pairs\n\
                     referencing missing genes are dropped with a warning."
    )]
    pub lr_pairs: Box<str>,

    #[arg(
        long,
        default_value_t = 500,
        help = "Number of matched decoy gene pairs per real LR pair"
    )]
    pub n_null: usize,

    #[arg(
        long,
        default_value_t = 4,
        help = "Quantile bins for L and R in the H(R|L) estimator"
    )]
    pub n_bins: usize,

    #[arg(
        long,
        default_value_t = 20,
        help = "Minimum edges per connected component to keep in the test"
    )]
    pub min_cc_edges: usize,

    #[arg(
        long,
        default_value_t = 0.25,
        help = "Relative tolerance for mean-expression matching of decoys (fraction of pooled σ)"
    )]
    pub expr_tol: f32,

    #[arg(
        long,
        default_value_t = 0.10,
        help = "Absolute tolerance for global Moran's I matching of decoys"
    )]
    pub moran_tol: f32,

    #[arg(
        long,
        default_value_t = 50.0,
        help = "Minimum total count for a gene to enter the decoy candidate pool"
    )]
    pub min_gene_count: f32,

    #[arg(
        long,
        default_value = "_",
        help = "Delimiter used to split compound gene row names (e.g. `_` for ENSG..._SYMBOL)",
        long_help = "Delimiter used to alias compound gene row names. Pass `_` (default)\n\
                     to match `ENSG00000105329_TGFB1` by either the Ensembl id or the\n\
                     symbol. Pass an empty string or a non-present character to disable\n\
                     aliasing and require exact full-name matches."
    )]
    pub gene_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching of gene names (forward/reverse) as a fallback"
    )]
    pub gene_allow_prefix: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = NullScheme::Mixed,
        help = "Null scheme: swap both / ligand-only / receptor-only / mixed (default)",
        long_help = "How decoy pairs are drawn for the gene-swap null:\n\
                     \x20 both     — swap both ligand and receptor to matched decoys\n\
                     \x20 ligand   — swap ligand only; receptor stays as the real R\n\
                     \x20 receptor — swap receptor only; ligand stays as the real L\n\
                     \x20 mixed    — draw n_null/3 from each and concatenate (default)\n\
                     \n\
                     ligand-only and receptor-only nulls test whether the specific\n\
                     identity of the swapped gene matters given the real counterpart;\n\
                     both-swap is a stricter composite. The mixed default catches any\n\
                     of these deviations from the (L, R) identity."
    )]
    pub null_scheme: NullScheme,
}
