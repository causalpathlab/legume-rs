//! CLI arguments for the `pinto lr-activity` subcommand.

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
        default_value_t = 10,
        help = "Random-projection dimension for propensity binary-sort (samples ≈ batches × 2^d)",
        long_help = "Number of random-projection axes used to assign each cell to a\n\
                     propensity bin via binary sort. Each batch gets up to 2^d bins;\n\
                     pseudobulk samples are (batch × propensity-bin) combinations.\n\
                     Defaults to 10 (≈1024 bins per batch, typical practice). Larger\n\
                     d → finer pseudobulk resolution and more permutation power, at\n\
                     the cost of fewer cells per pseudobulk; d=12 (~4096 bins) starts\n\
                     to make per-sample pseudobulks too sparse on Visium-scale data."
    )]
    pub propensity_dim: usize,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Number of sample permutations for the null distribution"
    )]
    pub n_permutations: usize,

    #[arg(
        long,
        default_value_t = 4,
        help = "Number of top propensity bits to stratify the shuffle by (0 = unstratified)",
        long_help = "Sample permutation reshuffles L only within sample groups\n\
                     sharing the top --shuffle-stratify-dim bits of the propensity\n\
                     binary code, preserving the cell-population marginal across\n\
                     permutations. Without it, free shuffles across populations\n\
                     pick up cell-type-marginal correlations and become\n\
                     anti-conservative.\n\
                     \n\
                     Default 4 → 16 stratification buckets, holding 2^(d−4) samples\n\
                     each (e.g. ~64 samples/bucket at d=10). Pick s ≤ d − 3 so each\n\
                     bucket retains ≥ ~8 samples. 0 disables stratification."
    )]
    pub shuffle_stratify_dim: usize,

    #[arg(
        long,
        default_value_t = 50.0,
        help = "Minimum total count for a real LR gene's pair to be tested"
    )]
    pub min_gene_count: f32,

    #[arg(
        long,
        default_value_t = 100,
        help = "Skip communities with fewer than this many edges (sparse communities can't calibrate)"
    )]
    pub min_edges_per_community: usize,

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
        default_value_t = true,
        help = "Also write a JSON sidecar with per-stratum participating edge lists for significant pairs"
    )]
    pub emit_json: bool,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Westfall-Young FWER cutoff for including a pair's edge participation in the JSON sidecar"
    )]
    pub json_fwer_threshold: f32,
}
