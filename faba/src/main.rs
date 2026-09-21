mod apa;
mod atoi;
mod cell_qc;
mod common;
mod data;
mod docs;
mod editing;
mod gene_count;
mod m6a;
mod mixture;
mod pipeline;
mod qc;
mod quant;
mod read_depth;
mod site_analysis;
mod snp;

use crate::common::*;
use crate::qc::{run_qc, run_qc_report, QcArgs, QcReportArgs};
use apa::run::*;
use atoi::run::*;
use docs::*;
use gene_count::run::*;
use m6a::run::*;
use pipeline::args::*;
use pipeline::run::*;
use read_depth::run::*;
use site_analysis::metagene::*;
use site_analysis::pileup::*;
use site_analysis::scan_pwm::*;
use snp::run::*;

const LOGO: &str = include_str!("../logo.txt");

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", line);
    }
    println!(" Feature statistics Accumulator for Base-pair-level Analysis");
    println!();
}

/// Feature statistics Accumulator for Base-pair-level Analysis
#[derive(Parser, Debug)]
#[command(version, about, long_about = None, term_width = 80,
    after_help = "\
Feature naming convention:\n\
  All sparse matrix row names follow: {gene_key}/{modality}/{detail}\n\
  where gene_key = {gene_id}_{symbol} (e.g. ENSG00001234_BRCA2)\n\n\
  \n\
  * count:   {gene_key}/count/spliced, {gene_key}/count/unspliced\n\
  * dartseq: {gene_key}/m6a/{channel} (gene), {gene_key}/m6a/{chr}:{pos}/{channel}\n\
  * (site),  {gene_key}/m6a/{component}/{channel} (mixture)\n\
  * atoi:    {gene_key}/atoi/{channel} (gene), {gene_key}/atoi/{chr}:{pos}/{channel}\n\
  * (site),  {gene_key}/atoi/{component}/{channel} (mixture)\n\
  * apa:     {gene_key}/apa/{proximal|distal} (gene),\n\
  * (mixture){gene_key}/apa/{component}\n\
  * baf:     {chr}:{pos}/baf/{alt|depth}\n\
  \n\
  \n\
Output layout (every matrix is per-replicate — one per input BAM):\n\
  per-modality: {batch}_m6a, {batch}_atoi, ...\n\
  depth:        {batch}_depth —> binned per-cell read depth (--depth-resolution-kb)\n\
  apa:          {batch}_apa (proximal/distal counts)\n\
  \n\
  Mixture components are FIT ON the POOLED shared across batches\n\
  but COUNTED PER BATCH, so per-batch matrices share one row vocabulary\n\
  and stack directly.\n\
  \n\
  The shared definitions are the only single files:\n\
  *_sites.parquet, *_components.parquet.\n\
  \n\
Use `faba <COMMAND> --help` for detailed options on each subcommand.")]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Enable verbose logging")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(name = "dartseq", aliases = ["dart", "m6a"],
        about = "Quantify DART-seq m6A sites from C-to-T conversions",
        long_about = "Quantify DART-seq m6A sites from C-to-T conversions\n\n\
            A site is a PUTATIVE candidate on the sequencing pattern alone:\n\
            the RAC (forward) / GTY (reverse) motif,\n\
            at least --min-conversion converted signal reads,\n\
            and total coverage (signal + control) of at least --min-coverage.\n\
            `faba qc` thresholds variants, and `faba qc-report` shows what each keeps.\n\
	    \n\
            The unit is always the site.\n\
            A genomic C/T variant converts equally in both arms, so a control is REQUIRED.\n\n\
            Outputs (one per input BAM, {batch}-prefixed):\n\
            - m6a_sites.parquet: every putative site with its statistics (single)\n\
            - {batch}_m6a: gene-level two-channel matrix\n\
              (methylated + unmethylated counts per gene, pooled over every putative site)\n\
            - {batch}_m6a_site: per-site two-channel matrix,\n\
              keyed on the single-base {chr}:{pos} site\n\
            - {batch}_m6a_mixture (+ m6a_components.parquet), with --mixture:\n\
              per-replicate mixture counts — components fit on pooled replicates,\n\
              counted per batch (shared row schema)\n\n\
	      \n\
            Reference:\n\
            Meyer, \"DART-seq: an antibody-free method for global m6A detection\",\n\
            Nature Methods, 16(12):1275-1280, 2019.\n\
            https://doi.org/10.1038/s41592-019-0570-0",
        after_long_help = "\
Example:\n  \
  faba dartseq wt.bam --control-bam ctrl.bam -g genes.gff -f genome.fa -o out/\n\
  faba dartseq s1.bam,s2.bam --control-bam c1.bam,c2.bam\n\
    -g genes.gff -f genome.fa -o out/ --mixture\n\
  faba qc-report out/ -o out/qc && faba qc out/ -o out_qc/")]
    DartSeq(DartSeqCountArgs),

    #[command(name = "apa", aliases = ["polya"],
        about = "Quantify alternative polyadenylation (APA) sites per cell",
        long_about = "Quantify alternative polyadenylation (APA) sites per cell\n\n\
            Discovers and quantifies poly(A) site usage from 3'-end sequencing data.\n\
            The mixture mode implements the SCAPE model.\n\n\
            Outputs:\n\
            - apa_components.parquet: shared pA-site component definitions\n\
            (fit on the pooled BAMs)\n\
            - {batch}_apa: per-replicate per-cell proximal/distal counts\n\
            (default; --no-pdui to skip)\n\
            - {batch}_apa_mixture: per-replicate per-cell pA-site usage,\n\
            counted per batch on the shared components (--mixture)\n\
            --method simple instead writes a per-replicate {batch} matrix\n\
            for each input BAM.\n\n\
            Reference:\n\
            Zhou et al., \"SCAPE: a mixture model revealing single-cell polyadenylation diversity and cellular dynamics during cell differentiation and reprogramming\",\n\
            Nucleic Acids Research, 50(11):e66, 2022.\n\
            https://doi.org/10.1093/nar/gkac167",
        after_long_help = "\
	Example:\n\
	faba apa sample.bam -g genes.gff -o out/\n\
	faba apa sample.bam -g genes.gff -o out/ --method simple\n\
	faba apa sample.bam --utr-bed utrs.bed -o out/ --mixture\n\
  faba apa sample.bam -g genes.gff -o out/ --gene-type protein_coding")]
    Apa(CountApaArgs),

    #[command(name = "atoi", aliases = ["a2i", "editing"],
        about = "Detect and quantify A-to-I RNA editing sites",
        long_about = "Detect A-to-I (adenosine-to-inosine) RNA editing sites\n\
                      \n\
                      Discovers editing sites from A->G (forward) or T->C (reverse) conversions in BAM files.\n\
                      A putative site is a reference A/T with observed editing at/above the\n\
                      candidacy floors (--min-coverage, --min-conversion). Every putative site is\n\
                      written with its beta-binomial p-value against the sequencing-error null and\n\
                      quantified per cell; no p-value cutoff is applied here (see `faba qc`).\n\
                      \n\
                      Outputs (one per input BAM, {batch}-prefixed):\n\
                      - atoi_sites.parquet: every putative site with its statistics (single)\n\
                      - {batch}_atoi: gene-level two-channel matrix\n\
                      (edited + unedited counts per gene, pooled over every putative site)\n\
                      - {batch}_atoi_site: per-site two-channel matrix,\n\
                      keyed on the single-base {chr}:{pos} site\n\
                      - {batch}_atoi_mixture (+ atoi_components.parquet), with --mixture:\n\
                      per-replicate mixture counts",
        after_long_help = "\
	Example:\n\
	faba atoi sample.bam -g genes.gff -f genome.fa -o out/\n\
  faba atoi s1.bam,s2.bam -g genes.gff -f genome.fa -o out/ --min-coverage 10")]
    AtoI(AtoICountArgs),

    #[command(name = "count", aliases = ["genes", "count-genes"],
        about = "Count reads per gene for single-cell or bulk RNA-seq",
        long_about = "Count reads per gene for single-cell or bulk RNA-seq\n\
                      \n\
                      Produces ONE sparse (features x cells) count matrix per input BAM,\n\
                      `{batch}_count`, from GFF gene annotations.\n\
                      Supports 10x-style cell barcodes.\n\
                      Rows are `{gene_key}/count/{spliced|unspliced}`:\n\
                      both tracks share the one feature axis,\n\
                      so sum a gene's two rows to recover its total count.",
        after_long_help = "\
	Example:\n\
	faba count sample.bam -g genes.gff -o out/\n\
  faba count s1.bam,s2.bam -g genes.gff -o out/ --gene-type protein_coding"
    )]
    Count(GeneCountArgs),

    #[command(name = "depth", aliases = ["read-depth", "rd"],
        about = "Compute read depth over genomic intervals",
        long_about = "Compute read depth over genomic intervals\n\
                      \n\
                      Bins the genome at the -r/--resolution-kb resolution (in KILOBASES) and counts read coverage per cell,\n\
                      producing a sparse (bins x cells) matrix per input BAM.",
        after_long_help = "\
	Example:\n\
	faba depth sample.bam -r 10 -o out/\n\
  faba depth sample.bam -r 100 -o out/"
    )]
    Depth(ReadDepthArgs),

    #[command(name = "pwm", aliases = ["scan-pwm"],
        about = "Build position weight matrix around genomic sites",
        long_about = "Build position weight matrix around genomic sites\n\
                      \n\
                      Reads site-level parquet files from dartseq, atoi or apa output,\n\
                      collects base frequencies in a +/- window around each site,\n\
                      and outputs a position weight matrix as TSV.",
        after_long_help = "\
	Example:\n\
	faba pwm -s out/m6a_sites.parquet -f genome.fa -o pwm.tsv\n\
  faba pwm -s out/m6a_sites.parquet sample.bam --source reads -o pwm.tsv"
    )]
    Pwm(ScanPwmArgs),

    #[command(
        name = "pileup",
        alias = "inspect",
        about = "ASCII pileup, or a faceted Miami plot (SVG/PDF) for a gene",
        long_about = "Pileup plot for a gene's (or region's) modification sites.\n\
                      \n\
                      Selection is `-q/--genes` (symbols or Ensembl IDs) and/or `--regions chr:lb-ub`;\n\
                      at least one is required,\n\
                      and everything matched is aggregated into one pileup.\n\
                      \n\
                      ASCII mode (default):\n\
                      reads one or more sparse matrices (zarr/h5) from faba output,\n\
                      filters to the selection, bins positions along the gene body,\n\
                      and renders a vertical ASCII histogram.\n\
                      Multiple files (e.g. replicates via a shell glob) are aggregated per position.\n\
                      \n\
                      Miami figure mode: passing --gtf, --bam, --format, --svg,\n\
                      or --png renders a publication SVG/PDF instead —\n\
                      a mirrored Manhattan with epi sites up, a GTF gene model in the middle,\n\
                      and BAM read depth down,\n\
                      faceted into one panel per cell type (--cell-membership).",
        after_long_help = "\
	Examples:\n\
	# ASCII histogram (unchanged)\n\
	faba pileup out/rep1_wt_m6a.zarr.zip -q BRCA2\n\
	faba pileup out/rep*_wt_m6a.zarr.zip -q BRCA2 -s out/m6a_sites.parquet\n\
	# Miami figure: epi sites / gene model / read depth, faceted by cell type\n\
	faba pileup out/rep1_wt_m6a.zarr.zip -q BRCA2 \n\
	--gtf gencode.gtf --bam sample.bam --cell-membership cells.tsv \n\
    --top-modality m6A --out brca2_miami --svg --png"
    )]
    Pileup(PileupArgs),

    #[command(
        name = "metagene",
        alias = "mg",
        about = "Metagene histogram of site positions across gene features",
        long_about = "Metagene histogram of site positions across gene features\n\n\
            Follows the MetaPlotR convention, so a profile can be held against a published one.\n\
            Each site is placed on ONE elected transcript per gene, the longest spliced.\n\
            That transcript's 5'UTR, CDS and 3'UTR are disjoint, so nothing needs a priority order.\n\
            Position runs along the SPLICED region; introns consume no coordinate.\n\
            Non-coding genes are left out unless --include-non-coding asks for them.\n\n\
            Bins split between the regions by each region's MEDIAN spliced length,\n\
            taken over the assigned sites, so widths depend on the sites as well as the annotation.\n\
            Compare the shape of two profiles rather than their bar widths.\n\n\
            Bin heights are RAW counts, never a rate.\n\
            A bin is also taller where more of its positions were deep enough to test,\n\
            which on a 3'-biased library means the last bins of the 3'UTR.\n\
            A terminal peak is therefore not evidence of enrichment on its own.\n\n\
            Neither --isoforms nor --dist-measures exists in MetaPlotR itself.\n\
            Its pipeline emits every overlapping transcript and leaves the choice to a script,\n\
            so `--isoforms all` is that raw output and `longest` is what the script intends.\n\
            --dist-measures is our name for the per-site table that script reads.\n\n\
            See docs/profiling-methods.md sections 1.2 and 7.\n\n\
            Reference:\n\
            Olarerin-George and Jaffrey, \"MetaPlotR: a Perl/R pipeline for plotting metagenes of nucleotide modifications and other transcriptomic sites\",\n\
            Bioinformatics, 33(10):1563-1564, 2017.\n\
            https://doi.org/10.1093/bioinformatics/btx002",
        after_long_help = "\
	Example:\n\
	faba metagene -s out/m6a_sites.parquet -g genes.gff -o metagene.tsv --print\n\
	# write the table MetaPlotR's visualize_metagenes.R reads:\n\
	faba metagene -s out/m6a_sites.parquet -g genes.gff -o metagene.tsv \n\
	--dist-measures m6a.dist.measures.txt"
    )]
    Metagene(MetageneArgs),

    #[command(name = "snp", aliases = ["genotype"],
        about = "Discover and genotype SNP variants from BAM pileup",
        long_about = "Discover and genotype SNP variants from BAM pileup\n\
                      \n\
                      Two modes of operation:\n\
                      1. De novo discovery (default): compare reads to reference genome,\n\
                      call variants where alt allele evidence exceeds thresholds.\n\
                      2. Known-site genotyping (--known-snps): force-call at VCF positions.\n\
                      Both modes can be combined.\n\
                      \n\
                      Supports 10x single-cell (per-cell allele counts + depth for BAF) and bulk WGS/RNA-seq modes.\n\
                      \n\
                      Outputs:\n\
                      - snp_sites.parquet: genotype calls with allele counts and GQ\n\
                      - snp_sites.vcf.gz: the same calls as VCF\n\
                      (skipped with a warning when the FASTA has no readable .fai)\n\
                      - {batch}_baf: per-cell allele frequency matrix (10x)\n\
                      \n\
                      The GENOTYPE CALLS are the parquet/VCF.\n\
                      `{batch}_baf` is a different thing:\n\
                      an allele-frequency track carrying two read counts per cell per locus and no genotype,\n\
                      no GQ, no rsid. Rows are `{chr}:{pos}/baf/{alt|depth}` —\n\
                      keyed on the LOCUS,\n\
                      since a variant is a coordinate and belongs to no gene.\n\
                      BAF = alt / depth per cell per locus; the channels nest (alt ≤ depth),\n\
                      so never sum them. Needs -g/--gff,\n\
                      which is what gives each locus a region to fetch reads from.\n\
                      (matrices are `.zarr.zip` by default; `.zarr` with --no-zip, `.h5` for the hdf5 backend.)\n\
                      \n\
                      Uses a binomial genotype likelihood model (cellSNP-lite; Huang & Huang, Bioinformatics 2021).\n\
                      \n\
                      The call set is an output in its own right; no other faba step consumes it as a mask.\n\
                      Join snp_sites.parquet against a site table downstream if a variant overlap matters.",
        after_long_help = "\
	Example:\n\
	# De novo discovery\n\
	faba snp sample.bam -f genome.fa -g genes.gff -o out/\n\n\
	# Known-site genotyping only\n\
	faba snp sample.bam -f genome.fa --known-snps dbsnp.vcf.gz -o out/ --skip-discovery\n\n\
	# Both: discover + force-call at known sites\n\
	faba snp sample.bam -f genome.fa --known-snps dbsnp.vcf.gz -g genes.gff -o out/\n\n\
	# Bulk mode (genotype calls only, no per-cell matrices)\n\
  faba snp sample.bam -f genome.fa -o out/ --bulk\n\n\
  Known SNP reference files:\n\n\
  dbSNP common variants (hg38):\n\
  wget https://ftp.ncbi.nih.gov/snp/organisms/human_9606/VCF/00-common_all.vcf.gz\n\
  wget https://ftp.ncbi.nih.gov/snp/organisms/human_9606/VCF/00-common_all.vcf.gz.tbi\n\n\
  1000 Genomes (hg38):\n\
  wget https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/\n\
  1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/\n\
  1kGP_high_coverage_Illumina.sites.vcf.gz\n\n\
  gnomAD v4 sites (hg38, per-chromosome):\n\
  wget https://storage.googleapis.com/gcp-public-data--gnomad/\n\
  release/4.1/vcf/genomes/gnomad.genomes.v4.1.sites.chr{1..22}.vcf.bgz\n\n\
  Mouse Genomes Project (mm10/mm39):\n\
  wget https://ftp.ebi.ac.uk/pub/databases/mousegenomes/\n\
    REL-2112-v8-SNPs_Indels/mgp_REL2021_snps.vcf.gz"
    )]
    Snp(SnpArgs),

    #[command(
        name = "qc",
        about = "Filter a faba output directory into a new fileset: cells, features and editing sites",
        long_about = "Filter a faba output directory into a NEW fileset (never in place).\n\
                      \n\
                      The producers are inclusive: they write every called cell, every gene with a count,\n\
                      and every putative editing site with its statistics, and apply no p-value,\n\
                      effect-size or reproducibility cutoff. This is where those cuts live.\n\
                      \n\
                      Cells are decided once per batch on `{batch}_count` (nnz floor + MAD outlier QC)\n\
                      and the same keep set is applied to every matrix of the batch.\n\
                      Editing sites are decided on the parquet columns (pv, log_odds, coverage, ...)\n\
                      plus the kept cells per site read off the `_site` matrices; dropped sites go to\n\
                      `{modality}_sites_dropped.parquet` with a `reason`. Gene-level `{batch}_m6a` /\n\
                      `{batch}_atoi` are re-pooled from the filtered site matrix, so they agree with the cut.\n\
                      Run `faba qc-report` first to see what each threshold keeps.",
        after_long_help = "\
	Example:\n\
	faba qc-report out/ -o out/qc\n\
	faba qc out/ -o out_qc/ --site-max-pv 0.05 --site-min-cells 10 --auto-cutoff"
    )]
    Qc(QcArgs),

    #[command(
        name = "qc-report",
        about = "Sweep every `faba qc` threshold and report what survives",
        long_about = "Sweep every `faba qc` threshold over a grid, one criterion at a time with the\n\
                      others off, and report the kept sites / genes / cells at each value, after a\n\
                      -log10(p) histogram of every putative site per editing modality.\n\
                      Writes {prefix}.qc_report.parquet and draws each panel as ASCII bars on stderr.\n\
                      No error rate is estimated; calibrating a cutoff is left to you."
    )]
    QcReport(QcReportArgs),

    #[command(
        name = "docs",
        about = "Print the method write-ups compiled into this binary",
        long_about = "Print the method write-ups compiled into this binary.\n\
                      \n\
                      Run with no argument to list what there is."
    )]
    Docs(DocsArgs),

    #[command(
        name = "all",
        aliases = ["pipeline", "full", "magic"],
        about = "Run all RNA-seq analyses: SNP → count → ATOI → m6A → APA",
        long_about = "Run all RNA-seq analyses in a unified pipeline\n\
                      \n\
                      Orchestrates the complete analysis workflow:\n\
                      0. SNP genotyping (de novo + optional --known-snps; skip --skip-snp)\n\
                      1. Gene expression filtering (identify expressed genes)\n\
                      2. Per-cell read depth (only with --depth-resolution-kb)\n\
                      3. ATOI detection (every putative A-to-I site, with its p-value)\n\
                      4. m6A detection (DART C→T, WT-vs-MUT contrast; skipped w/o --control-bam)\n\
                      5. APA quantification (alternative polyadenylation)\n\
                      \n\
                      Read depth is independent of every other step: nothing downstream reads it.\n\
                      It runs straight after gene counting only to share that step's called-cell axis,\n\
                      so its columns match every other matrix.\n\
                      \n\
                      APA runs LAST because the SCAPE EM is the heavy step and nothing else waits on it,\n\
                      so the fast modalities finish first.\n\
                      \n\
                      Discovery runs in bulk, over all cells that passed step 1 at once,\n\
                      for both ATOI and m6A —\n\
                      so the two modalities cannot disagree about which cells were compared.\n\
                      \n\
                      ATOI is reference-anchored and tested per site against a beta-binomial error null (no control).\n\
                      m6A instead needs a catalytically-dead control (--control-bam):\n\
                      each motif C is tested for higher conversion in the positional BAMs than the pooled control,\n\
                      so genomic C/T variants are rejected;\n\
                      without a control the m6A step is skipped.\n\
                      The WT-vs-MUT split is only for that contrast:\n\
                      control BAMs are otherwise quantified like the positional samples,\n\
                      so every modality is produced for them too.\n\
                      The gene and cell sets from step 1 apply to every later step.\n\
                      No step masks another and no step applies a p-value or effect-size cutoff:\n\
                      run `faba qc-report` and `faba qc` on the output directory for that.",
        after_long_help = "\
	Example:\n\
	faba all sample.bam -g genes.gff -f genome.fa -o out/\n\
	faba all wt.bam -g genes.gff -f genome.fa -o out/ --control-bam ctrl.bam\n\
  faba all s1.bam,s2.bam -g genes.gff -f genome.fa -o out/ --skip-apa"
    )]
    All(PipelineArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    data_beans::aux::logging::init_logger(cli.verbose);

    // Install the Ctrl+C handler up front, so one keypress means one thing for the whole run.
    //
    // It used to be installed lazily, by whichever library function first asked for the flag —
    // which made the *same* keypress behave three different ways depending on when you pressed
    // it: a hard kill before the handler existed, a graceful stop while a loop was polling, and —
    // worst — a silent no-op afterwards, where the flag was set, nothing was watching it, and the
    // process simply appeared to ignore you until you pressed it a second time.
    let _stop = legume_numeric::matrix::stop::stop_flag();

    match cli.commands {
        Commands::DartSeq(ref args) => run_m6a(args)?,
        Commands::Apa(mut args) => run_apa(&mut args)?,
        Commands::AtoI(ref args) => run_atoi(args)?,
        Commands::Count(ref args) => run_gene_count(args)?,
        Commands::Depth(ref args) => run_read_depth(args)?,
        Commands::Pwm(ref args) => run_scan_pwm(args)?,
        Commands::Pileup(ref args) => run_pileup(args)?,
        Commands::Metagene(ref args) => run_metagene(args)?,
        Commands::Snp(ref args) => run_snp(args)?,
        Commands::Qc(ref args) => run_qc(args)?,
        Commands::QcReport(ref args) => run_qc_report(args)?,
        Commands::Docs(ref args) => run_docs(args)?,
        Commands::All(ref args) => run_pipeline(args)?,
    }

    Ok(())
}
