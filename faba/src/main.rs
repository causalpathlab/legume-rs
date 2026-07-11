mod apa;
mod assoc;
mod cell_qc;
mod common;
mod data;
mod editing;
mod gene_count;
mod lineage;
mod mixture;
mod pipeline_util;
mod plot;
mod read_depth;
mod run_annotate;
mod run_apa;
mod run_assoc;
mod run_atoi;
mod run_gem_embedding;
mod run_gene_count;
mod run_lineage;
mod run_m6a;
mod run_pipeline;
mod run_read_depth;
mod run_snp;
mod site_analysis;
mod snp;

use crate::common::*;
use faba::gem::args::GemArgs;
use plot::*;
use run_annotate::*;
use run_apa::*;
use run_assoc::*;
use run_atoi::*;
use run_gem_embedding::*;
use run_gene_count::*;
use run_lineage::*;
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
  dartseq: gene_key/m6a/{channel} (gene), gene_key/m6a/{chr}:{pos}/{channel}\n\
           (site), gene_key/m6a/{component}/{channel} (mixture)\n\
  atoi:    gene_key/atoi/{channel} (gene), gene_key/atoi/{chr}:{pos}/{channel}\n\
           (site), gene_key/atoi/{component}/{channel} (mixture)\n\
  apa:     gene_key/apa/{channel} (gene), gene_key/apa/{component} (mixture)\n\
  snp:     gene_key/snp/{chr}:{pos} (alt allele count per cell)\n\n\
  Split on '/' to extract (gene_key, modality, detail) for cross-modal joins.\n\n\
Output layout (every matrix is per-replicate — one per input BAM):\n\
  per-modality: {batch}_m6a, {batch}_atoi (gene two-channel:\n\
                {gene}/{mod}/{pos} = converted, /{neg} = unconverted),\n\
                plus {batch}_{m6a,atoi}_site (per-site) and _mixture,\n\
                {batch}_genes, {batch}_snp_{alt,depth}\n\
  apa:          {batch}_apa (proximal/distal counts, default; --no-pdui skips),\n\
                {batch}_apa_mixture (--mixture)\n\
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
            Outputs (one per input BAM, {batch}-prefixed):\n\
            - m6a_sites.parquet: site annotations (single)\n\
            - {batch}_m6a: gene-level two-channel matrix\n\
              (methylated + unmethylated counts per gene)\n\
            - {batch}_m6a_site: per-site two-channel matrix, keyed on the\n\
              single-base {chr}:{pos} site (min --site-min-cells)\n\
            - {batch}_m6a_mixture (+ m6a_components.parquet): per-replicate\n\
              mixture counts — components fit on pooled replicates,\n\
              counted per batch (shared row schema)\n\n\
            Reference:\n  \
            Meyer, \"DART-seq: an antibody-free method for global m6A\n\
            detection\", Nature Methods, 16(12):1275-1280, 2019.\n\
            https://doi.org/10.1038/s41592-019-0570-0",
        after_long_help = "\
Example:\n  \
  faba dartseq wt.bam --control-bam ctrl.bam -g genes.gff -f genome.fa -o out/\n\
  faba dartseq s1.bam,s2.bam --control-bam c1.bam,c2.bam \n\
    -g genes.gff -f genome.fa -o out/ --detect-atoi --min-coverage 20\n\
    faba dartseq wt.bam --control-bam ctrl.bam -g genes.gff -f genome.fa -o out/ \n\
    --atoi-mask out/atoi_sites.parquet")]
    DartSeq(DartSeqCountArgs),

    #[command(name = "apa", aliases = ["polya"],
        about = "Quantify alternative polyadenylation (APA) sites per cell",
        long_about = "Quantify alternative polyadenylation (APA) sites per cell\n\n\
            Discovers and quantifies poly(A) site usage from 3'-end sequencing\n\
            data. The mixture mode implements the SCAPE model.\n\n\
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
            Zhou et al., \"SCAPE: a mixture model revealing single-cell\n\
            polyadenylation diversity and cellular dynamics during cell\n\
            differentiation and reprogramming\",\n\
            Nucleic Acids Research, 50(11):e66, 2022.\n\
            https://doi.org/10.1093/nar/gkac167",
        after_long_help = "\
	Example:\n\
	faba apa sample.bam -g genes.gff -o out/\n\
	faba apa sample.bam -g genes.gff -o out/ --method simple\n\
	faba apa sample.bam --utr-bed utrs.bed -o out/ --mixture\n\
  faba apa sample.bam -g genes.gff -o out/ --atoi-mask out/atoi_sites.parquet")]
    Apa(CountApaArgs),

    #[command(name = "atoi", aliases = ["a2i", "editing"],
        about = "Detect and quantify A-to-I RNA editing sites",
        long_about = "Detect A-to-I (adenosine-to-inosine) RNA editing sites\n\n\
            Discovers editing sites from A->G (forward) or T->C (reverse)\n\
            conversions in BAM files, then quantifies per-cell editing\n\
            at discovered sites.\n\n\
            Outputs (one per input BAM, {batch}-prefixed):\n\
            - atoi_sites.parquet: site annotations (single); usable as\n\
            --atoi-mask input for `faba dartseq` or `faba apa`\n\
            - {batch}_atoi: gene-level two-channel matrix\n\
              (edited + unedited counts per gene)\n\
            - {batch}_atoi_site: per-site two-channel matrix, keyed on the\n\
              single-base {chr}:{pos} site (min --site-min-cells)\n\
            - {batch}_atoi_mixture (+ atoi_components.parquet): per-replicate\n\
              mixture counts (unless --no-mixture)",
        after_long_help = "\
	Example:\n\
	faba atoi sample.bam -g genes.gff -f genome.fa -o out/\n\
  faba atoi s1.bam,s2.bam -g genes.gff -f genome.fa -o out/ --min-coverage 10")]
    AtoI(AtoICountArgs),

    #[command(name = "genes", aliases = ["count-genes"],
        about = "Count reads per gene for single-cell or bulk RNA-seq",
        long_about = "Count reads per gene for single-cell or bulk RNA-seq\n\n\
            Produces a sparse (cells x genes) count matrix from BAM files\n\
            using GFF gene annotations. Supports 10x-style cell barcodes.",
        after_long_help = "\
	Example:\n\
	faba genes sample.bam -g genes.gff -o out/\n\
  faba genes sample.bam -g genes.gff -o out/ --no-splice"
    )]
    Genes(GeneCountArgs),

    #[command(name = "depth", aliases = ["read-depth", "rd"],
        about = "Compute read depth over genomic intervals",
        long_about = "Compute read depth over genomic intervals\n\n\
            Bins the genome at a given resolution and counts read coverage\n\
            per cell, producing a sparse (cells x bins) matrix.",
        after_long_help = "\
	Example:\n\
	faba depth sample.bam -r 10 -o out/\n\
  faba depth sample.bam -r 100 -o out/"
    )]
    Depth(ReadDepthArgs),

    #[command(name = "pwm", aliases = ["scan-pwm"],
        about = "Build position weight matrix around genomic sites",
        long_about = "Build position weight matrix around genomic sites\n\n\
            Reads site-level parquet files from dartseq or apa output, collects\n\
            base frequencies in a +/- window around each site, and outputs\n\
            a position weight matrix as TSV.",
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
	Examples:\n\
	# ASCII histogram (unchanged)\n\
	faba pileup out/rep1_wt_m6a.zarr.zip -q BRCA2\n\
	faba pileup out/rep*_wt_m6a.zarr.zip -q BRCA2 -s out/m6a_sites.parquet\n\
	# Miami figure: epi sites / gene model / read depth, faceted by cell type\n\
	faba pileup out/rep1_wt_m6a.zarr.zip -q BRCA2 \\\n\
	--gtf gencode.gtf --bam sample.bam --cell-membership cells.tsv \\\n\
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
	Example:\n\
	faba metagene -s out/m6a_sites.parquet -g genes.gff -o metagene.tsv --print\n\
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
            - {batch}_snp_alt: per-cell alt allele count matrix (10x)\n\
            - {batch}_snp_depth: per-cell total depth matrix (10x)\n\
            (matrices are `.zarr.zip` by default; `.zarr` with --no-zip,\n\
            `.h5` for the hdf5 backend.) BAF = alt / depth per cell per site.\n\n\
            Uses a binomial genotype likelihood model (cellSNP-lite;\n\
            Huang & Huang, Bioinformatics 2021).\n\n\
            The SNP mask output can be used with --snp-mask in `faba atoi`,\n\
            `faba dartseq`, and `faba apa` to filter genetic variants that\n\
            masquerade as base modifications.",
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
        name = "gem",
        aliases = ["gem-embedding"],
        about = "GEM: Geodesic Embedding + Motion — velocity (tangent) + lineage (path) in one cell space",
        long_about = "GEM — Geodesic Embedding + Motion: a joint cell-feature embedding.\n\
            Motion is the local velocity δ (the tangent); the lineage is the geodesic path it traces.\n\
            Runs over the shared graph_embedding_util engine, which is modality-agnostic.\n\
            Fed gene counts (spliced + unspliced) today; embeds any per-feature count.\n\n\
            Per-gene β-sharing: each `{gene}/count/{spliced|unspliced}` row embeds as β_g.\n\
            A gene's spliced and unspliced tracks thus share one identity.\n\
            Cell identity is the spliced projection θ → `{out}.cell_embedding.parquet` (raw).\n\
            A velocity increment δ is fit from the unspliced edges → `{out}.velocity.parquet`.\n\
            The nascent state is just θ+δ; ‖δ‖ is speed.\n\
            Per-gene velocity, if wanted, is the in-model δ_g (`--delta-l2`).\n\n\
            With `--lineage-dag` it also shapes the embedding along a pseudobulk lineage.\n\
            It then writes a per-cell pseudotime + fate backbone.\n\
            That backbone is a prior for `faba lineage`, not a replacement.",
        after_long_help = "\
	Example:\n\
  faba gem out/rep1_wt_genes.zarr.zip -o out/gem\n\n\
  Multiple samples — pass them positionally, so shell globs work.\n\
  Each sample becomes a batch via its barcodes' `@batch` tag.\n\n\
  faba gem out/rep1_genes.zarr.zip out/rep2_genes.zarr.zip -o out/gem\n\
  faba gem out/*_genes.zarr.zip -o out/gem\n\n\
  The `--genes a,b` flag form still works, but not together with the positional one.")]
    Gem(GemArgs),

    #[command(
        name = "annotate",
        aliases = ["annot", "ann"],
        about = "Marker-set cell-type annotation of a `faba gem` run",
        long_about = "Annotate the embeddings from `faba gem` against a marker set.\n\n\
            Reads gem's parquet outputs by prefix (`-f/--from`) and a marker TSV\n\
            (`gene<TAB>celltype`, `-m/--markers`), then runs the shared term-ORA core\n\
            (nearest-centroid assign → distance-outlier QC → Leiden clustering →\n\
            cluster×term hypergeometric over-representation, permutation-calibrated).\n\n\
            gem carries two gene programs, each annotated on its own axis (`--track`):\n\
            spliced:  gene β_g (beta_dictionary) vs cell θ (cell_embedding)  → {out}.spliced.*\n\
              velocity: gene δ_g (delta_dictionary) vs cell velocity   → {out}.velocity.*\n\
            `both` (default) runs both; velocity is skipped with a warning when its\n\
            inputs are absent (spliced-only gem run).",
        after_long_help = "\
	Example:\n\
	faba gem --genes out/rep1_genes.zarr.zip -o out/gem\n\
  faba annotate -f out/gem -m markers.tsv -o out/gem"
    )]
    Annotate(AnnotateArgs),

    #[command(
        name = "lineage",
        aliases = ["trajectory", "traj"],
        about = "Velocity-oriented lineage + principal curves over a `faba gem` run",
        long_about = "Infer a velocity-oriented lineage over the embeddings from `faba gem`.\n\n\
            Reads gem's parquet outputs by prefix (`-f/--from`): the cell embedding θ\n\
            (cell_embedding.parquet) and per-cell velocity δ (velocity.parquet). Fits K\n\
            k-means centroids on θ, an MST over them, orients that tree by the\n\
            per-node mean velocity flux, and fits Slingshot-style smooth principal\n\
            curves per lineage → per-cell pseudotime + branch.\n\n\
            Root selection (priority order): --root-node, --root-cell, --root-type\n\
            (marker-grounded, needs --markers), --root-from-gem (gem's velocity-DAG\n\
            source), else the velocity-flux source.\n\n\
            The low-coverage modalities are NOT embedded here; this produces the\n\
            lineage ordering that a separate confounder-adjusted test runs against.\n\n\
            Outputs (all `{out}`-prefixed parquet): nodes, node_velocity, edges\n\
            (with velocity_flux + directed edges), lineages, pseudotime,\n\
            cell_lineage_weights, lineage_pseudotime, curves; with --markers also\n\
            lineage_annot.* + trajectory_annotation; with --layout phate (default)\n\
            also {cells,nodes,curves}_2d.\n\n\
            Reference:\n  \
            Street et al., \"Slingshot: cell lineage and pseudotime inference for\n\
            single-cell transcriptomics\", BMC Genomics, 19:477, 2018.\n\
            https://doi.org/10.1186/s12864-018-4772-0",
        after_long_help = "\
	Example:\n\
	faba gem --genes out/rep1_genes.zarr.zip -o out/gem\n\
  faba lineage -f out/gem -o out/gem"
    )]
    Lineage(LineageArgs),

    #[command(
        name = "plot",
        aliases = ["plot-lineage", "trajectory-plot"],
        about = "Publication-style figure (PDF/PNG/SVG) of a `faba lineage` trajectory over its 2D embedding",
        long_about = "Render the outputs of `faba lineage --markers` (with the default\n\
            --layout phate) into a single annotated figure: cells laid out on the PHATE\n\
            embedding, coloured by coarse cell type (default) or pseudotime, with the\n\
            Slingshot principal curves + MST trajectory nodes overlaid.\n\n\
            Reads by prefix (`-f/--from`): {from}.cells_2d.parquet (PHATE coords),\n\
            {from}.lineage_annot.annot.parquet (per-cell coarse_label),\n\
            {from}.curves_2d.parquet (principal curves), {from}.nodes_2d.parquet\n\
            (MST nodes), {from}.trajectory_annotation.parquet (node role/cell_type),\n\
            and {from}.pseudotime.parquet (for --color-by pseudotime).\n\n\
            The cells are drawn as one transparent raster layer per cell type from a\n\
            qualitative palette (with a legend), or one continuous blue->red\n\
            pseudotime layer (with a colourbar). Principal curves + nodes are dark\n\
            overlays; each non-Cycling_Progenitor node is labeled with its cell type\n\
            and the root node is marked with a red star. Uses the shared plot-utils\n\
            rasterize -> SVG -> render pipeline; writes {out}.plot.pdf by default\n\
            (--png / --svg add those formats, --no-pdf skips the PDF). The scatter is a\n\
            raster layer, so the PDF is a hybrid (vector text over raster points at --dpi;\n\
            raise --dpi to 300-600 for print).",
        after_long_help = "\
	Example:\n\
	faba lineage -f out/gem -o out/lin --markers markers.tsv\n\
	faba plot -f out/lin\n\
  faba plot -f out/lin -o out/lin_pt --color-by pseudotime"
    )]
    Plot(PlotArgs),

    #[command(
        name = "dyn-assoc",
        aliases = ["assoc", "temporal-assoc", "trend"],
        about = "Bayesian between-branch modality contrast along a `faba lineage`",
        long_about = "Test whether a modality (m6a/apa/atoi) diverges between lineage branches.\n\n\
            Downstream of `faba lineage` (like `annotate` is to `gem`). Fits, per branch, a\n\
            binomial GLM  logit(p_{b,g}) = α_b + β·1[g=L]  where α_b conditions out pseudotime\n\
            (matched-null, à la tradeSeq patternTest / cocoa) and β is the branch's\n\
            pseudotime-adjusted log-odds excess. Coverage (edited + unedited) is the binomial\n\
            denominator so detection bias is conditioned out; a shrinkage prior N(0, τ²) on β\n\
            damps noisy calls (stable across seeds — no permutation machinery). Reports the\n\
            posterior mean effect, 90% credible interval, and lfsr = min(P(β>0), P(β<0)); the\n\
            within-branch trend GAM (--trend-method) runs alongside.\n\n\
            If the lineage was annotated (`faba lineage --markers`), the same two tests are\n\
            also reported per CELL TYPE — cells sharing an annotated type are pooled across\n\
            lineages ({out}.celltype_contrast/_trend.parquet). The between-cell-type contrast\n\
            is the clean deliverable; the within-cell-type trend is secondary (pooling divergent\n\
            lineages onto one pseudotime axis weakens the trend reading). Skip with --no-celltype.\n\n\
            Not double-dipping: branches come from gem θ + velocity, which never see the\n\
            modality.\n\n\
            Reference:\n  \
            Van den Berge et al., \"Trajectory-based differential expression analysis\n\
            for single-cell sequencing data\", Nat Commun 11:1201, 2020.",
        after_long_help = "\
	Example:\n\
	faba lineage -f out/gem -o out/lin --markers markers.tsv\n\
  faba dyn-assoc -f out/lin -s out/rep1_wt_m6a_site.zarr.zip --modality m6a -o out/m6a_assoc"
    )]
    Assoc(AssocArgs),

    #[command(
        name = "all",
        aliases = ["pipeline", "full", "magic"],
        about = "Run all RNA-seq analyses: SNP → genes → ATOI → APA → m6A",
        long_about = "\
	Run all RNA-seq analyses in a unified pipeline\n\n\
	Orchestrates the complete analysis workflow:\n\
	0. SNP genotyping (de novo + optional --known-snps; skip --skip-snp)\n\
	1. Gene expression filtering (identify expressed genes)\n\
	2. ATOI detection (A-to-I editing sites, masked by SNP)\n\
	3. APA quantification (alternative polyadenylation, masked by SNP+ATOI)\n\
        4. m6A detection (DART C→T, WT-vs-MUT contrast; skipped w/o --control-bam)\n\n\
        ATOI is reference-anchored and FDR-controlled against a beta-binomial\n\
        error null (no control). m6A instead needs a catalytically-dead\n\
        control (--control-bam): each motif C is tested for higher conversion\n\
        in the positional BAMs than the pooled control, so genomic C/T\n\
        variants are rejected; without a control the m6A step is skipped.\n\
        The WT-vs-MUT split is only for that contrast: control BAMs are\n\
        otherwise quantified like the positional samples, so every\n\
        modality is produced for them too.\n\
        Gene filter applies after step 1; the SNP mask feeds steps 2-4\n\
        (m6A only with --m6a-snp-mask) and the ATOI mask feeds steps 3-4.",
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
        Commands::Annotate(ref args) => run_annotate(args)?,
        Commands::Lineage(ref args) => run_lineage(args)?,
        Commands::Plot(ref args) => run_plot(args)?,
        Commands::Assoc(ref args) => run_assoc(args)?,
        Commands::All(ref args) => run_pipeline(args)?,
    }

    Ok(())
}
