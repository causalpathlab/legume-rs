//! `faba all` command-line surface.
//!
//! One struct for the whole pipeline: each step reads the subset it needs and
//! builds the standalone subcommand's own args from it, so a chained run and a
//! hand-run subcommand cannot drift apart.

use crate::common::*;
use crate::editing::pipeline::M6aContrastArgs;
use crate::quant::mass_enrichment::MassEnrichmentArgs;

/// Serialize a field by its `Debug` form — for foreign enums that carry no `Serialize`.
pub(super) fn ser_debug<T: std::fmt::Debug, S: serde::Serializer>(
    v: &T,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("{v:?}"))
}

#[derive(Args, Debug, serde::Serialize)]
#[command(
    about = "Run unified RNA-seq pipeline: SNP → genes → ATOI → m6A → APA",
    long_about = "Orchestrates the complete RNA-seq analysis pipeline:\n\n\
        0. SNP genotyping (de novo discovery + optional known sites)\n\
        1. Gene expression filtering (identify expressed genes)\n\
        2. ATOI detection (A-to-I editing, masked by SNP)\n\
        3. APA quantification (alternative polyadenylation, masked by SNP+ATOI)\n\
        4. m6A detection (DART C→T, WT-vs-MUT contrast; skipped without --control-bam)\n\n\
        ATOI is reference-anchored and tested per site\n\
        against a beta-binomial sequencing-error null (--edit-error-rate/--edit-overdispersion),\n\
        no control sample.\n\
        m6A instead requires a catalytically-dead control (--control-bam):\n\
        each motif C is tested for higher conversion in the positional BAMs\n\
        than the pooled control (so a genomic C/T variant is rejected);\n\
        the step is skipped when no control is given.\n\
        The SNP mask is off by default for m6A\n\
        (the contrast already rejects variants; opt back in with --m6a-snp-mask).\n\
        Step 0 discovers variants de novo and optionally force-calls at known sites\n\
        (--known-snps, VCF/BCF/Parquet).\n\
        De novo variants are VAF-filtered (--snp-mask-min-vaf)\n\
        so RNA editing sites are preserved in the mask.\n\
        UMI deduplication is applied to all pileup steps (disable with --no-umi-dedup)."
)]
pub struct PipelineArgs {
    // Required inputs
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM files (comma-separated)",
        long_help = "Comma-separated BAM files used across every modality:\n\
                     gene counting, ATOI, APA and m6A quantification."
    )]
    pub bam_files: Vec<Box<str>>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation (GFF) file"
    )]
    pub gff_file: Box<str>,

    #[arg(
        short = 'f',
        long = "genome",
        required = true,
        help = "Reference genome FASTA file (.fa/.fasta, must be indexed)"
    )]
    pub genome_file: Box<str>,

    #[arg(
        short = 'o',
        long = "output",
        required = true,
        help = "Output directory (flat structure)"
    )]
    pub output: Box<str>,

    #[arg(
        long = "control-bam",
        alias = "mut",
        alias = "control",
        value_delimiter = ',',
        help = "Control BAM files (catalytically-dead YTHmut) for the m6A contrast",
        long_help = "Comma-separated control (catalytically-dead YTHmut) BAM files.\n\
                     m6A is called by a WT-vs-MUT contrast:\n\
                     the signal arm is the positional BAMs MINUS these controls.\n\
                     That split is used only for m6A site discovery;\n\
                     otherwise these controls are quantified like positional samples.\n\
                     Their SNP, gene, ATOI, APA and m6A per-cell matrices are produced too,\n\
                     with cells frozen per control BAM in step 1.\n\
                     Optional, but the m6A (DART) step is skipped without it:\n\
                     m6A cannot be separated from genomic C/T variation without a control."
    )]
    pub control_bam_files: Vec<Box<str>>,

    ///////////////////////
    // Shared parameters //
    ///////////////////////
    #[arg(long, default_value = "CB", help = "Cell barcode tag")]
    pub cell_barcode_tag: Box<str>,

    #[arg(long, default_value = "GX", help = "Gene barcode tag")]
    pub gene_barcode_tag: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend (zarr or hdf5)"
    )]
    // `SparseIoBackend` is data-beans' and does not implement `Serialize`; it is a plain
    // enum, so its `Debug` form ("Zarr" / "Hdf5") is exactly what belongs in the summary.
    #[serde(serialize_with = "ser_debug")]
    pub backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive",
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive\n\
                     (zarr backend only; no effect on hdf5)"
    )]
    pub zip: bool,

    #[arg(
        long,
        alias = "threads",
        default_value_t = 16,
        help = "Maximum number of threads"
    )]
    pub max_threads: usize,

    ///////////////////////////////
    // Gene expression filtering //
    ///////////////////////////////
    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum cells per gene (gene filtering; 0 = off, matching \
                Cell Ranger, which keeps every gene/feature)"
    )]
    pub gene_min_cells: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum UMI per gene (gene filtering; 0 = off, matching \
                Cell Ranger, which keeps every gene/feature)"
    )]
    pub gene_min_counts: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum detected genes (nnz) per cell; cells below this are \
                dropped from gene counts and every downstream modality. \
                0 = off (default), so cell calling is pure Cell Ranger \
                EmptyDrops/OrdMag with no extra min-genes floor"
    )]
    pub cell_min_genes: usize,

    #[command(flatten)]
    pub cell_qc: crate::cell_qc::CellQcArgs,

    //////////////////////////////////////////
    // Gene biotype (quantification subset) //
    //////////////////////////////////////////
    #[arg(
        long,
        default_value = "",
        help = "Gene biotype to quantify. Empty (default) keeps all biotypes. \
                Pass a value to restrict: protein_coding, lncRNA, pseudogene. \
                QC/cell-calling always uses ALL biotypes; only the quantified \
                gene set (gene counts + ATOI/APA/m6A) is restricted to this type."
    )]
    pub gene_type: Box<str>,

    //////////////////////
    // Mitochondrial QC //
    //////////////////////
    #[command(flatten)]
    pub mito_qc: crate::quant::MitoQcArgs,

    ////////////////////////////////////////////////////
    // Shared read-quality filters (ATOI / m6A / SNP) //
    ////////////////////////////////////////////////////
    #[arg(
        long,
        default_value_t = 20,
        help = "Minimum base quality for editing/SNP base calls (ATOI/m6A/SNP)"
    )]
    pub min_base_quality: u8,

    #[arg(
        long,
        default_value_t = 20,
        help = "Minimum mapping quality for editing/SNP reads (ATOI/m6A/SNP)"
    )]
    pub min_mapping_quality: u8,

    #[arg(
        long = "no-apa-pdui",
        default_value_t = false,
        help = "Skip the APA PDUI (proximal/distal count) matrix output ({batch}_apa)"
    )]
    pub no_apa_pdui: bool,

    /////////////////////
    // ATOI parameters //
    /////////////////////
    #[arg(
        long,
        default_value_t = 5,
        help = "Minimum coverage for ATOI detection (matches `faba atoi`)"
    )]
    pub atoi_min_coverage: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum A-to-G conversions for ATOI (matches `faba atoi`)"
    )]
    pub atoi_min_conversion: usize,

    #[arg(
        long = "atoi-fdr",
        default_value_t = 0.05,
        help = "ATOI detection FDR target (Benjamini-Hochberg q-value)"
    )]
    pub atoi_fdr_cutoff: f32,

    ///////////////////////////////////////////////////////
    // Editing statistical null (shared by ATOI and m6A) //
    ///////////////////////////////////////////////////////
    #[arg(
        long = "edit-error-rate",
        alias = "error-rate",
        default_value_t = 0.01,
        help = "Sequencing-error rate ε: the beta-binomial null mean the edited \
                fraction is tested against (reference-anchored, no control sample)"
    )]
    pub edit_error_rate: f64,

    #[arg(
        long = "edit-overdispersion",
        alias = "overdispersion",
        default_value_t = 0.1,
        help = "Beta-binomial overdispersion ρ for the editing null (0 ⇒ binomial)"
    )]
    pub edit_overdispersion: f64,

    ////////////////////
    // APA parameters //
    ////////////////////
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for APA detection"
    )]
    pub apa_min_coverage: usize,

    #[arg(
        long = "apa-max-sites",
        default_value_t = 20,
        help = "Cap candidate poly-A sites per UTR for APA BIC selection \
                (top-N by coverage; 0 = unlimited). Bounds EM cost on long 3'UTRs."
    )]
    pub apa_max_sites: usize,

    #[arg(
        long = "apa-em-pdui",
        default_value_t = false,
        help = "Use the full SCAPE EM for PDUI instead of the fast top-2 nearest-site \
                assignment (slower; --mixture also forces the EM)"
    )]
    pub apa_em_pdui: bool,

    #[arg(long, default_value_t = 10, help = "Minimum poly(A) tail length")]
    pub polya_min_tail_length: usize,

    /////////////////////
    // DART parameters //
    /////////////////////
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for m6A detection"
    )]
    pub m6a_min_coverage: usize,

    #[arg(long, default_value_t = 5, help = "Minimum C-to-T conversions for m6A")]
    pub m6a_min_conversion: usize,

    #[arg(
        long = "m6a-fdr",
        default_value_t = 0.05,
        help = "m6A detection FDR target (Benjamini-Hochberg q-value)"
    )]
    pub m6a_fdr_cutoff: f32,

    #[command(flatten)]
    pub m6a_contrast: M6aContrastArgs,

    /// Apply the SNP mask to m6A calls. Off by default: with the WT-vs-MUT
    /// contrast a genomic variant is rejected automatically, so the mask is
    /// redundant (and was over-aggressive).
    #[arg(long = "m6a-snp-mask", default_value_t = false)]
    pub m6a_snp_mask: bool,

    ////////////////////////////////////////////////////////
    // Mixture model weighting (shared by m6A and A-to-I) //
    ////////////////////////////////////////////////////////
    #[arg(
        long = "mixture-weight",
        value_enum,
        default_value_t = crate::editing::pipeline::MixtureWeightMode::Posterior,
        help = "How to weight each (cell, site) observation in the mixture EM"
    )]
    pub mixture_weight: crate::editing::pipeline::MixtureWeightMode,

    #[arg(
        long = "mixture-prior-alpha",
        default_value_t = 1.0,
        help = "Beta prior α for posterior-rate weighting (default: 1.0)"
    )]
    pub mixture_prior_alpha: f32,

    #[arg(
        long = "mixture-prior-beta",
        default_value_t = 1.0,
        help = "Beta prior β for posterior-rate weighting (default: 1.0)"
    )]
    pub mixture_prior_beta: f32,

    #[arg(
        long = "drop-single-component",
        default_value_t = false,
        help = "Drop genes with a single mixture component across m6A/ATOI/APA"
    )]
    pub drop_single_component: bool,

    #[arg(
        long = "mixture",
        default_value_t = false,
        help = "Also produce the per-gene component-mixture `_{modality}_mixture` \
                matrices (EM; slow). Off by default — only the gene-level \
                `{gene}/{modality}/{channel}` counts are produced.",
        long_help = "Also produce the per-gene component-mixture matrices (EM; slow).\n\
                     Off by default — only the gene-level\n\
                     `{gene}/{modality}/{channel}` counts are produced.\n\n\
                     For m6A / A-to-I this SKIPS the 1-D Gaussian mixture EM entirely when off.\n\
                     For APA the SCAPE poly-A fit always runs\n\
                     (PDUI needs it to identify proximal vs distal),\n\
                     so this gates only the extra `_apa_mixture` component-matrix output."
    )]
    pub mixture: bool,

    ////////////////////
    // SNP parameters //
    ////////////////////
    #[arg(
        long = "known-snps",
        help = "Known SNP sites VCF/BCF/Parquet for SNP masking",
        long_help = "Path to known SNP sites. Accepts:\n\
                     - VCF/BCF (.vcf, .vcf.gz, .bcf): standard variant calls\n\
                     - Parquet (.parquet): output from a previous `faba snp` run\n\
                     When provided, force-calls genotypes at these positions\n\
                     in addition to de novo discovery,\n\
                     and builds a mask for ATOI/APA/DART filtering."
    )]
    pub known_snps: Option<Box<str>>,

    #[arg(long, default_value_t = 5, help = "Minimum depth for SNP calling")]
    pub snp_min_depth: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Minimum genotype quality for SNP mask"
    )]
    pub snp_min_gq: f32,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for de novo SNP discovery"
    )]
    pub snp_min_coverage: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum alt allele reads for SNP discovery"
    )]
    pub snp_min_alt_count: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Minimum alt allele frequency for SNP discovery"
    )]
    pub snp_min_alt_freq: f64,

    #[arg(
        long,
        default_value_t = 0.35,
        help = "Minimum VAF for SNP mask (filters RNA editing from de novo variants)",
        long_help = "Minimum variant allele fraction for a de novo discovered variant\n\
                     to enter the SNP mask.\n\
                     A het site needs VAF in the range [min_vaf, 1-min_vaf];\n\
                     a hom-alt site needs VAF >= 1-min_vaf.\n\
                     Sites with lower VAF are likely RNA editing (A-to-I or m6A)\n\
                     rather than germline SNPs.\n\
                     Set to 0 to disable VAF filtering (mask all called variants)."
    )]
    pub snp_mask_min_vaf: f32,

    //////////////////////////////////////////////
    // UMI deduplication (applies to all steps) //
    //////////////////////////////////////////////
    #[arg(
        long = "umi-tag",
        default_value = "UB",
        help = "UMI barcode BAM tag for deduplication (all steps)"
    )]
    pub umi_tag: Box<str>,

    #[arg(
        long = "no-umi-dedup",
        default_value_t = false,
        help = "Disable UMI deduplication (for bulk data without UMIs)"
    )]
    pub no_umi_dedup: bool,

    //////////////////
    // Step control //
    //////////////////
    #[arg(long, default_value_t = false, help = "Skip SNP genotyping step")]
    pub skip_snp: bool,

    #[arg(long, default_value_t = false, help = "Skip gene counting step")]
    pub skip_genes: bool,

    #[arg(long, default_value_t = false, help = "Skip ATOI detection step")]
    pub skip_atoi: bool,

    #[arg(long, default_value_t = false, help = "Skip APA quantification step")]
    pub skip_apa: bool,

    //////////////////////////////
    // Mass-enrichment grouping //
    //////////////////////////////
    #[command(flatten)]
    pub enrich: MassEnrichmentArgs,
}
