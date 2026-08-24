//! CLI arguments for the `pinto lr-activity` subcommand.

use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct SrtLrActivityArgs {
    // Deliberately NOT the shared `SrtInputArgs`: this command reads its cell
    // pairs from a prior run, so the graph, coordinate, batch and QC options
    // that struct carries would all parse and do nothing here. Twice that
    // shape produced a command that failed or lied at its own defaults, so
    // lra declares exactly what it consumes, the way `prop` does.
    #[arg(
        required = true,
        value_delimiter(','),
        help = "Spatial gene expression data files (.zarr or .h5)",
        long_help = "Spatial gene expression data files, comma separated.\n\
                     Accepted formats are .zarr and .h5.\n\
                     Each file is a genes-by-cells sparse matrix.\n\
                     Multiple files are concatenated column-wise, over cells.\n\
                     Cells must match the prior run the pairs are read from."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix (e.g., results/my_run)"
    )]
    pub out: Box<str>,

    #[arg(long, default_value_t = 42, help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        long,
        help = "Cells per parallel block (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse data into memory",
        long_help = "Preload all sparse column data into memory up front.\n\
                     Faster when the data fits in RAM.\n\
                     Some parallel access patterns require it. It raises peak memory usage."
    )]
    pub preload_data: bool,

    #[arg(
        long,
        required = true,
        help = "Prefix of a prior `pinto lc` run",
        long_help = "Prefix of a prior `pinto lc` run.\n\
                     Edge-to-community assignments are read from {prefix}.link_community.parquet.\n\
                     Cell-pair metadata comes from {prefix}.coord_pairs.parquet,\n\
                     including per-edge batch labels when present.\n\
                     Row order must match across both files. They are joined by position."
    )]
    pub lc_prefix: Box<str>,

    #[arg(
        long,
        required = true,
        help = "Two-column TSV of ligand-receptor pairs",
        long_help = "Ligand-receptor pair file, one pair per line.\n\
                     Each line holds two columns: ligand gene, then receptor gene.\n\
                     They may be separated by whitespace, tab or comma.\n\
                     The delimiter is auto-detected: comma for .csv, else tab.\n\
                     \n\
                     Gene names must match row names in the expression data.\n\
                     Pairs naming a missing gene are dropped with a warning."
    )]
    pub lr_pairs: Box<str>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Random-projection dimension for propensity binary-sort",
        long_help = "Random-projection axes for binning cells by propensity.\n\
                     A binary sort over d axes gives each batch up to 2^d bins.\n\
                     Pseudobulk samples are (batch × propensity-bin) pairs.\n\
                     \n\
                     The default of 10 gives ≈1024 bins per batch.\n\
                     Larger d buys finer resolution and more permutation power.\n\
                     It also leaves fewer cells per pseudobulk. At d=12, roughly 4096 bins,\n\
                     Visium-scale data turns sparse.",
        hide = true
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
        long_help = "Permutation reshuffles L within sample groups only.\n\
                     A group shares the top s bits of the propensity binary code.\n\
                     That preserves the cell-population marginal across permutations.\n\
                     Free shuffles across populations behave differently.\n\
                     They pick up cell-type marginals, and turn anti-conservative.\n\
                     \n\
                     The default of 4 gives 16 stratification buckets.\n\
                     Each holds 2^(d−4) samples, so about 64 at d=10.\n\
                     Pick s ≤ d − 3 to keep at least ~8 samples per bucket.\n\
                     Pass 0 to disable stratification.",
        hide = true
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
        help = "Skip communities with fewer than this many edges",
        long_help = "Skip communities holding fewer than this many edges.\n\
                     A sparse community cannot calibrate the null reliably.",
        hide = true
    )]
    pub min_edges_per_community: usize,

    #[arg(
        long,
        default_value = "_",
        help = "Delimiter used to split compound gene row names (e.g. `_` for ENSG..._SYMBOL)",
        long_help = "Delimiter used to alias compound gene row names.\n\
                     The default `_` aliases `ENSG00000105329_TGFB1`.\n\
                     Either the Ensembl id or the symbol then matches it. Pass an empty string,\n\
                     or a character that never occurs,\n\
                     to disable aliasing and demand exact full-name matches.",
        hide = true
    )]
    pub gene_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching of gene names (forward/reverse) as a fallback",
        hide = true
    )]
    pub gene_allow_prefix: bool,

    #[arg(
        long,
        default_value_t = true,
        help = "Also write a JSON sidecar with per-stratum edge lists for sig. pairs",
        long_help = "Also write a JSON sidecar of participating edge lists.\n\
                     They are recorded per stratum, for significant pairs."
    )]
    pub emit_json: bool,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "FWER cutoff for the JSON sidecar; gates every LR figure pinto plot draws",
        long_help = "Westfall-Young FWER cutoff for the JSON sidecar.\n\
                     A row enters the sidecar when its FWER is below this\n\
                     AND its restandardized z is positive,\n\
                     meaning the pair is active in that stratum specifically.\n\
                     The sidecar is what pinto plot renders,\n\
                     so this cutoff gates every LR figure.\n\
                     The parquet always carries the full table regardless."
    )]
    pub json_fwer_threshold: f32,
}
