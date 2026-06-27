mod apa;
mod cell_clustering;
mod cell_qc;
mod common;
mod data;
mod editing;
mod gene_count;
mod mixture;
mod pipeline_util;
mod read_depth;
mod run_apa;
mod run_atoi;
mod run_gem_embedding;
mod run_gene_count;
mod run_m6a;
mod run_pipeline;
mod run_read_depth;
mod run_snp;
mod site_analysis;
mod snp;

use crate::common::*;
use faba::gem::annotate::{run_gem_annotate, GemAnnotateArgs};
use faba::gem::args::GemArgs;
use faba::gem::plot::{run_gem_plot, GemPlotArgs};
use faba::gem::summary::{run_gem_summary, GemSummaryArgs};
use run_apa::*;
use run_atoi::*;
use run_gem_embedding::*;
use run_gene_count::*;
use run_m6a::*;
use run_pipeline::*;
use run_read_depth::*;
use run_snp::*;
use site_analysis::metagene::*;
use site_analysis::pileup::*;
use site_analysis::scan_pwm::*;

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
  genes:   gene_key/count/spliced, gene_key/count/unspliced\n\
  dartseq: gene_key/m6A/{component} (mixture), gene_key/m6A/{chr}:{pos} (site)\n\
  atoi:    gene_key/A2I/{component} (mixture), gene_key/A2I/{chr}:{pos} (site)\n\
  apa:     gene_key/pA/{component} (mixture), gene_key/pA/{chr}:{pos} (site)\n\n\
  snp:     gene_key/SNP/{chr}:{pos} (alt allele count per cell)\n\n\
  Split on '/' to extract (gene_key, modality, detail) for cross-modal joins.\n\n\
Output layout (every matrix is per-replicate — one per input BAM):\n\
  site-level: {batch}_m6a_{ratio,converted,unconverted},\n    \
              {batch}_atoi_ratio, {batch}_genes, {batch}_snp_{alt,depth}\n\
  mixture:    {batch}_m6a_mixture, {batch}_atoi_mixture,\n    \
              {batch}_apa_mixture, {batch}_apa_pdui\n\
  Mixture components are FIT on the pooled replicates (shared across\n\
  batches) but COUNTED per batch, so per-batch mixture matrices share one\n\
  row vocabulary and stack directly. The shared definitions are the only\n\
  single files: *_sites.parquet, *_components.parquet.\n\
  --drop-single-component prunes genes with a lone component (no relative\n\
  signal) from the mixture matrices and component sidecars.\n\n\
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
            Discovers m6A methylation sites from reference-anchored C->T\n\
            (forward) or G->A (reverse) conversions at the DART motif, called by\n\
            a WT-vs-MUT contrast: each motif C must convert significantly more in\n\
            the signal (APOBEC1-YTH) BAMs than in the pooled catalytically-dead\n\
            --control-bam (Fisher exact / beta-binomial LRT, effect-size guards,\n\
            BH-FDR). A genomic C/T variant converts equally in both arms and is\n\
            rejected, so a control is REQUIRED. Then quantifies per-cell\n\
            methylation at the discovered sites.\n\n\
            Outputs:\n  \
            - m6a_sites.parquet: site annotations (single)\n  \
            - {batch}_m6a_{ratio,converted,unconverted}: per-replicate\n    \
              site-level matrices, one per input BAM\n  \
            - {batch}_m6a_mixture (+ m6a_components.parquet): per-replicate\n    \
              mixture counts — components fit on pooled replicates,\n    \
              counted per batch (shared row schema)\n\n\
            Reference:\n  \
            Meyer, \"DART-seq: an antibody-free method for global m6A\n  \
            detection\", Nature Methods, 16(12):1275-1280, 2019.\n  \
            https://doi.org/10.1038/s41592-019-0570-0",
        after_long_help = "\
Example:\n  \
  faba dartseq wt.bam --control-bam ctrl.bam -g genes.gff -f genome.fa -o out/\n  \
  faba dartseq s1.bam,s2.bam --control-bam c1.bam,c2.bam \\\n    \
    -g genes.gff -f genome.fa -o out/ --detect-atoi --min-coverage 20\n  \
  faba dartseq wt.bam --control-bam ctrl.bam -g genes.gff -f genome.fa -o out/ \\\n    \
    --atoi-mask out/atoi_sites.parquet")]
    DartSeq(DartSeqCountArgs),

    #[command(name = "apa", aliases = ["polya"],
        about = "Quantify alternative polyadenylation (APA) sites per cell",
        long_about = "Quantify alternative polyadenylation (APA) sites per cell\n\n\
            Discovers and quantifies poly(A) site usage from 3'-end sequencing\n\
            data. The mixture mode implements the SCAPE model.\n\n\
            Outputs (--method mixture, default):\n  \
            - {batch}_apa_mixture (+ apa_components.parquet): per-replicate\n    \
              pA-site usage — sites fit on pooled BAMs, counted per batch\n    \
              (shared row schema)\n  \
            - {batch}_apa_pdui: per-replicate per-cell PDUI (--compute-pdui)\n  \
            --method simple instead writes a per-replicate {batch} matrix\n    \
            for each input BAM.\n\n\
            Reference:\n  \
            Zhou et al., \"SCAPE: a mixture model revealing single-cell\n  \
            polyadenylation diversity and cellular dynamics during cell\n  \
            differentiation and reprogramming\",\n  \
            Nucleic Acids Research, 50(11):e66, 2022.\n  \
            https://doi.org/10.1093/nar/gkac167",
        after_long_help = "\
Example:\n  \
  faba apa sample.bam -g genes.gff -o out/\n  \
  faba apa sample.bam -g genes.gff -o out/ --method simple\n  \
  faba apa sample.bam --utr-bed utrs.bed -o out/ --compute-pdui\n  \
  faba apa sample.bam -g genes.gff -o out/ --atoi-mask out/atoi_sites.parquet")]
    Apa(CountApaArgs),

    #[command(name = "atoi", aliases = ["a2i", "editing"],
        about = "Detect and quantify A-to-I RNA editing sites",
        long_about = "Detect A-to-I (adenosine-to-inosine) RNA editing sites\n\n\
            Discovers editing sites from A->G (forward) or T->C (reverse)\n\
            conversions in BAM files, then quantifies per-cell editing\n\
            at discovered sites.\n\n\
            Outputs:\n  \
            - atoi_sites.parquet: site annotations (single); usable as\n    \
              --atoi-mask input for `faba dartseq` or `faba apa`\n  \
            - {batch}_atoi_ratio: per-replicate site-level matrix, one\n    \
              per input BAM\n  \
            - {batch}_atoi_mixture (+ atoi_components.parquet): per-replicate\n    \
              mixture counts — shared (pooled) component fit, per-batch counts",
        after_long_help = "\
Example:\n  \
  faba atoi sample.bam -g genes.gff -f genome.fa -o out/\n  \
  faba atoi s1.bam,s2.bam -g genes.gff -f genome.fa -o out/ --min-coverage 10")]
    AtoI(AtoICountArgs),

    #[command(name = "genes", aliases = ["count-genes"],
        about = "Count reads per gene for single-cell or bulk RNA-seq",
        long_about = "Count reads per gene for single-cell or bulk RNA-seq\n\n\
            Produces a sparse (cells x genes) count matrix from BAM files\n\
            using GFF gene annotations. Supports 10x-style cell barcodes.",
        after_long_help = "\
Example:\n  \
  faba genes sample.bam -g genes.gff -o out/\n  \
  faba genes sample.bam -g genes.gff -o out/ --no-splice --backend hdf5"
    )]
    Genes(GeneCountArgs),

    #[command(name = "depth", aliases = ["read-depth", "rd"],
        about = "Compute read depth over genomic intervals",
        long_about = "Compute read depth over genomic intervals\n\n\
            Bins the genome at a given resolution and counts read coverage\n\
            per cell, producing a sparse (cells x bins) matrix.",
        after_long_help = "\
Example:\n  \
  faba depth sample.bam -r 10 -o out/\n  \
  faba depth sample.bam -r 100 -o out/ --backend hdf5"
    )]
    Depth(ReadDepthArgs),

    #[command(name = "pwm", aliases = ["scan-pwm"],
        about = "Build position weight matrix around genomic sites",
        long_about = "Build position weight matrix around genomic sites\n\n\
            Reads site-level parquet files from dartseq or apa output, collects\n\
            base frequencies in a +/- window around each site, and outputs\n\
            a position weight matrix as TSV.",
        after_long_help = "\
Example:\n  \
  faba pwm -s out/m6a_sites.parquet -f genome.fa -o pwm.tsv\n  \
  faba pwm -s out/m6a_sites.parquet sample.bam --source reads -o pwm.tsv"
    )]
    Pwm(ScanPwmArgs),

    #[command(
        name = "pileup",
        alias = "inspect",
        about = "ASCII pileup, or a faceted Miami plot (SVG/PDF) for a gene",
        long_about = "Pileup plot for a single gene's modification sites.\n\n\
            ASCII mode (default): reads one or more sparse matrices (zarr/h5)\n\
            from faba output, filters to a gene, bins positions along the gene\n\
            body, and renders a vertical ASCII histogram. Multiple files (e.g.\n\
            replicates via a shell glob) are aggregated per position.\n\n\
            Miami figure mode: passing --gtf, --bam, --format, --svg, or --png\n\
            renders a publication SVG/PDF instead — a mirrored Manhattan with\n\
            epi sites up, a GTF gene model in the middle, and BAM read depth\n\
            down, faceted into one panel per cell type (--cell-membership).",
        after_long_help = "\
Examples:\n  \
  # ASCII histogram (unchanged)\n  \
  faba pileup out/rep1_wt_m6a_mixture.zarr.zip -q BRCA2\n  \
  faba pileup out/rep*_wt_m6a_ratio.zarr.zip -q BRCA2 -s out/m6a_sites.parquet\n  \
  # Miami figure: epi sites / gene model / read depth, faceted by cell type\n  \
  faba pileup out/rep1_wt_m6a_ratio.zarr.zip -q BRCA2 \\\n  \
    --gtf gencode.gtf --bam sample.bam --cell-membership cells.tsv \\\n  \
    --top-modality m6A --out brca2_miami --svg --png"
    )]
    Pileup(PileupArgs),

    #[command(
        name = "metagene",
        alias = "mg",
        about = "Metagene histogram of site positions across gene features",
        long_about = "Metagene histogram of site positions across gene features\n\n\
            Maps sites from a parquet file onto gene features (5'UTR, CDS,\n\
            3'UTR, non-coding) using GFF annotations, and produces a binned\n\
            histogram showing the distribution of sites across the metagene.",
        after_long_help = "\
Example:\n  \
  faba metagene -s out/m6a_sites.parquet -g genes.gff -o metagene.tsv --print\n  \
  faba metagene -s out/atoi_sites.parquet -g genes.gff -o metagene.tsv -n 30"
    )]
    Metagene(MetageneArgs),

    #[command(name = "snp", aliases = ["genotype"],
        about = "Discover and genotype SNP variants from BAM pileup",
        long_about = "Discover and genotype SNP variants from BAM pileup\n\n\
            Two modes of operation:\n\
            1. De novo discovery (default): compare reads to reference genome,\n\
               call variants where alt allele evidence exceeds thresholds.\n\
            2. Known-site genotyping (--known-snps): force-call at VCF positions.\n\
            Both modes can be combined.\n\n\
            Supports 10x single-cell (per-cell allele counts + depth for BAF)\n\
            and bulk WGS/RNA-seq modes.\n\n\
            Outputs:\n\
            - snp_sites.parquet: genotype calls with allele counts and GQ\n\
            - {batch}_snp_alt.zarr: per-cell alt allele count matrix (10x)\n\
            - {batch}_snp_depth.zarr: per-cell total depth matrix (10x)\n\
            BAF = alt / depth per cell per site.\n\n\
            Uses a binomial genotype likelihood model (cellSNP-lite;\n\
            Huang & Huang, Bioinformatics 2021).\n\n\
            The SNP mask output can be used with --snp-mask in `faba atoi`,\n\
            `faba dartseq`, and `faba apa` to filter genetic variants that\n\
            masquerade as base modifications.",
        after_long_help = "\
Example:\n  \
  # De novo discovery\n  \
  faba snp sample.bam -f genome.fa -g genes.gff -o out/\n\n  \
  # Known-site genotyping only\n  \
  faba snp sample.bam -f genome.fa --known-snps dbsnp.vcf.gz -o out/ --skip-discovery\n\n  \
  # Both: discover + force-call at known sites\n  \
  faba snp sample.bam -f genome.fa --known-snps dbsnp.vcf.gz -g genes.gff -o out/\n\n  \
  # Bulk mode (genotype calls only, no per-cell matrices)\n  \
  faba snp sample.bam -f genome.fa -o out/ --bulk\n\n\
Known SNP reference files:\n\n  \
  dbSNP common variants (hg38):\n    \
    wget https://ftp.ncbi.nih.gov/snp/organisms/human_9606/VCF/00-common_all.vcf.gz\n    \
    wget https://ftp.ncbi.nih.gov/snp/organisms/human_9606/VCF/00-common_all.vcf.gz.tbi\n\n  \
  1000 Genomes (hg38):\n    \
    wget https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/\\\n      \
    1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/\\\n      \
    1kGP_high_coverage_Illumina.sites.vcf.gz\n\n  \
  gnomAD v4 sites (hg38, per-chromosome):\n    \
    wget https://storage.googleapis.com/gcp-public-data--gnomad/\\\n      \
    release/4.1/vcf/genomes/gnomad.genomes.v4.1.sites.chr{1..22}.vcf.bgz\n\n  \
  Mouse Genomes Project (mm10/mm39):\n    \
    wget https://ftp.ebi.ac.uk/pub/databases/mousegenomes/\\\n      \
    REL-2112-v8-SNPs_Indels/mgp_REL2021_snps.vcf.gz"
    )]
    Snp(SnpArgs),

    #[command(
        name = "gem",
        aliases = ["gem-embedding"],
        about = "GEM: Gene Epitranscriptomic Modification embedding",
        long_about = "GEM — joint embedding of gene counts + RNA-modification\n\
            tracks (m6A, A-to-I, poly-A) into one cell/gene space.\n\n\
            A feature row (gene g, modality m, region r) embeds as a base\n\
            gene vector β_g deviated by an exp log-deviation gate:\n  \
              AGG  ({g}/count):    e_f = β_g\n  \
              comp ({g}/{m}/{r}):  e_f = β_g ⊙ exp(Σ_k z_{g,k}·δ_{k,m,:} + γ_{m,r,:})\n\
            where z_g is the gene's K-program loading, δ_{k,m,:} the\n\
            program×modality deviation *direction* (a full H-vector, not a\n\
            scalar), and γ_{m,r,:} a per-(modality, region) offset.\n\
            All factors are shared across input tracks\n\
            (`faba {genes,dartseq,atoi,apa}`), so an unmeasured (gene,\n\
            modality) pair still gets a usable embedding — z_g is learned\n\
            from g's other modalities and δ_{:,m,:} from other genes' m-data.",
        after_long_help = "\
Example:\n  \
  faba gem --genes out/rep1_wt_genes.zarr.zip \\\n    \
              --dartseq out/rep1_wt_m6a_mixture.zarr.zip \\\n    \
              --atoi out/rep1_wt_atoi_mixture.zarr.zip \\\n    \
              --apa out/rep1_wt_apa_mixture.zarr.zip -o out/gem\n\n\
Multiple samples (comma-separated per modality; each sample a batch via\n\
its barcodes' `@batch` tag):\n  \
  faba gem --genes out/rep1_genes.zarr.zip,out/rep2_genes.zarr.zip \\\n    \
              --dartseq out/rep1_m6a.zarr.zip,out/rep2_m6a.zarr.zip \\\n    \
              -o out/gem")]
    Gem(GemArgs),

    #[command(
        name = "gem-plot",
        about = "2D UMAP layouts + cluster plots from a `faba gem` manifest",
        long_about = "Reads a `{prefix}.faba.json` gem manifest and renders two\n\
            UMAP scatter plots — the feature embedding (β_g) and the cell\n\
            embedding (e_cell) — each k-means clustered, with a black frame\n\
            box for easy overlay and per-cluster top-gene labels. Feature\n\
            clusters are labelled by the genes nearest the cluster centroid;\n\
            cell clusters by the genes they most upregulate (centroid · β_gᵀ).\n\
            Layout coords + cluster ids are also written to parquet.",
        after_long_help = "\
Example:\n  \
  faba gem-plot --from out/gem.faba.json\n  \
  faba gem-plot -f out/gem.faba.json --num-clusters 20 --top-features 5 --no-cells"
    )]
    GemPlot(GemPlotArgs),

    #[command(
        name = "gem-annotate",
        aliases = ["annotate"],
        about = "Marker-set cell-type annotation by projection (from a gem manifest)",
        long_about = "Light cell-type annotation: projects each marker-defined cell\n\
            type as a virtual cell onto the frozen gem feature embedding (the\n\
            same operator that placed the cells), then scores every cell by\n\
            cosine in the shared space → per-cell soft posterior + argmax.\n\
            An optional permutation null (random gene sets of matching size)\n\
            calibrates each affinity into a p-value. Emits type-anchor\n\
            embeddings that `faba gem-plot` can overlay.",
        after_long_help = "\
Example:\n  \
  faba gem-annotate --from out/gem.faba.json -m markers.tsv\n  \
  faba gem-annotate -f out/gem.faba.json -m markers.tsv --num-perm 500 --resolution 1.0"
    )]
    GemAnnotate(GemAnnotateArgs),

    #[command(
        name = "gem-summary",
        aliases = ["summary"],
        about = "Per-modality gene × cell-type summary from annotation labels",
        long_about = "Group a count matrix by cell type and report per-feature\n\
            statistics — the tidy \"gene × cell-type, per modality\" summary.\n\n\
            Decoupled from `faba gem-annotate`: annotate once, then summarize\n\
            cheaply and repeatedly across measures (m6a_ratio, m6a_mixture,\n\
            converted, genes, atoi_ratio, …) against the SAME labels. Labels\n\
            come from a gem-annotate `*.annot.parquet` or any 2-column\n\
            `cell<TAB>label` TSV (so labels from any tool work).\n\n\
            Data files are positional, like `data-beans stat`; multiple files\n\
            stack into one matrix (e.g. replicates). Run once per measure (with\n\
            a different -o). The output `{out}.summary.parquet` is long format\n\
            `(gene, modality, component, group, nnz, tot, mu, sig)`: it splits\n\
            each `{gene}/{modality}/{detail}` row name\n\
            into gene/modality/component columns — the modality-aware tidy table\n\
            you can pivot/filter by modality (what `data-beans stat` can't do).\n\
            `mu` is the mean over all cells in the group (incl. uncovered\n\
            zeros); `tot/nnz` is the mean over covered cells — for a ratio\n\
            matrix that distinction matters. Reuses the same grouped-row path\n\
            as `data-beans stat -s row -g`.\n\n\
            Multi-sample: if the membership is `@sample`-tagged (a multi-sample\n\
            `faba gem` run), each matrix file's barcodes are tagged the same\n\
            way (per-file `@sample` id, the `faba gem` convention) so pooled\n\
            replicates match EXACTLY — not by ambiguous bare barcode. The\n\
            strip is auto-derived from the files (their common `_`-suffix); for\n\
            a single file pass `--sample-strip` (e.g. `_m6a_ratio`).",
        after_long_help = "\
Example:\n  \
  # one measure (m6A ratio), two replicates stacked\n  \
  faba gem-summary -l out/gem.gem_annot.annot.parquet \\\n    \
      out/rep1_m6a_ratio.zarr.zip out/rep2_m6a_ratio.zarr.zip -o out/celltype\n\n\
  # another measure → run again (mixture component counts)\n  \
  faba gem-summary -l out/gem.gem_annot.annot.parquet \\\n    \
      out/rep1_m6a_mixture.zarr.zip -o out/celltype\n\n\
  # single-sample: data-beans stat does the same (no modality split)\n  \
  data-beans stat -s row -g out/gem.gem_annot.membership.tsv --delimiter @ \\\n    \
      -o out/m6a_ratio_by_celltype.parquet out/rep1_m6a_ratio.zarr.zip"
    )]
    GemSummary(GemSummaryArgs),

    #[command(
        name = "all",
        aliases = ["pipeline", "full", "magic"],
        about = "Run all RNA-seq analyses: SNP → genes → ATOI → APA → m6A",
        long_about = "Run all RNA-seq analyses in a unified pipeline\n\n\
            Orchestrates the complete analysis workflow:\n  \
            0. SNP genotyping (de novo + optional --known-snps; skip --skip-snp)\n  \
            1. Gene expression filtering (identify expressed genes)\n  \
            2. ATOI detection (A-to-I editing sites, masked by SNP)\n  \
            3. APA quantification (alternative polyadenylation, masked by SNP+ATOI)\n  \
            4. m6A detection (DART C→T, WT-vs-MUT contrast; skipped w/o --control-bam)\n\n\
            ATOI is reference-anchored and FDR-controlled against a beta-binomial\n\
            error null (no control). m6A instead needs a catalytically-dead\n\
            control (--control-bam): each motif C is tested for higher conversion\n\
            in the positional BAMs than the pooled control, so genomic C/T\n\
            variants are rejected; without a control the m6A step is skipped.\n\
            Gene filter applies after step 1; SNP mask to steps 2-4 (m6A only\n\
            with --m6a-snp-mask), ATOI mask to steps 3-4.",
        after_long_help = "\
Example:\n  \
  faba all sample.bam -g genes.gff -f genome.fa -o out/\n  \
  faba all wt.bam -g genes.gff -f genome.fa -o out/ --control-bam ctrl.bam\n  \
  faba all s1.bam,s2.bam -g genes.gff -f genome.fa -o out/ --skip-apa"
    )]
    All(PipelineArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    auxiliary_data::logging::init_logger(cli.verbose);

    match cli.commands {
        Commands::DartSeq(ref args) => run_m6a(args)?,
        Commands::Apa(mut args) => run_apa(&mut args)?,
        Commands::AtoI(ref args) => run_atoi(args)?,
        Commands::Genes(ref args) => run_gene_count(args)?,
        Commands::Depth(ref args) => run_read_depth(args)?,
        Commands::Pwm(ref args) => run_scan_pwm(args)?,
        Commands::Pileup(ref args) => run_pileup(args)?,
        Commands::Metagene(ref args) => run_metagene(args)?,
        Commands::Snp(ref args) => run_snp(args)?,
        Commands::Gem(ref args) => run_gem_embedding(args)?,
        Commands::GemPlot(ref args) => run_gem_plot(args)?,
        Commands::GemAnnotate(ref args) => run_gem_annotate(args)?,
        Commands::GemSummary(ref args) => run_gem_summary(args)?,
        Commands::All(ref args) => run_pipeline(args)?,
    }

    Ok(())
}
