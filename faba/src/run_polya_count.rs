use crate::common::*;
use crate::data::dna_stat_traits::*;
use crate::data::poly_a_stat_map::{PolyASiteArgs, PolyASiteMap};
use crate::data::util_htslib::*;
use genomic_data::bed::BedWithGene;
use genomic_data::gff::FeatureType as GffFeatureType;
use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::{GeneId, GffRecordMap};

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};

#[derive(Args, Debug)]
pub struct PolyACountArgs {
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM files.",
        long_help = "Comma-separated list of BAM files. \n\
		       These files contain RNA-seq reads with poly-A tails. \n\
		       Example: file1.bam,file2.bam"
    )]
    bam_files: Vec<Box<str>>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation (`GFF`) file",
        long_help = "Path to the gene annotation file in GFF format. \n\
		     This file provides genomic feature information required for analysis. \n\
		     Example: genes.gff"
    )]
    gff_file: Box<str>,

    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode tag",
        long_help = "Cell barcode tag used for cell/sample identification in 10x Genomics BAM files. \n\
		     [See here](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam)"
    )]
    cell_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode tag",
        long_help = "Barcode tag used for gene identification in 10x Genomics BAM files.\n\
		    [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)"
    )]
    gene_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum poly-A/T tail length for filtering",
        long_help = "Minimum length of poly-A/T tail (based on CIGAR soft-clip) required for poly-A site detection."
    )]
    polya_min_tail_length: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Maximum non-A/T bases in poly-A tail",
        long_help = "Maximum number of non-A (forward) or non-T (reverse) bases allowed in the soft-clipped poly-A/T region."
    )]
    polya_max_non_a_or_t: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Bases to check for internal priming",
        long_help = "Number of bases at the end of aligned sequence to check for A-rich (forward) or T-rich (reverse) regions indicating internal priming."
    )]
    polya_internal_prime_window: usize,

    #[arg(
        long,
        default_value_t = 7,
        help = "Minimum A/T count for internal priming",
        long_help = "Minimum number of A (forward) or T (reverse) bases in the internal priming window to flag as internal priming."
    )]
    polya_internal_prime_count: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage at a poly-A site",
        long_help = "Minimum number of reads required at a poly-A site for inclusion in the analysis. \n\
		     Filters out low-coverage sites."
    )]
    min_coverage: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Resolution for binning poly-A sites (in bp)",
        long_help = "Resolution for binning poly-A sites in base pairs (bp). \n\
		     Determines the size of genomic windows for aggregating poly-A sites. \n\
		     Default: 10 bp"
    )]
    resolution_bp: usize,

    #[arg(
        long,
        value_enum,
        help = "Bam record type (gene, transcript, exon, utr)",
        long_help = "Selectively choose BAM record type for analysis. \n\
		     Options include gene, transcript, exon, or UTR."
    )]
    record_type: Option<GffFeatureType>,

    #[arg(
        long,
        value_enum,
        help = "Gene type (protein_coding, pseudogene, lncRNA)",
        long_help = "Filter analysis by gene type. \n\
		     Options include protein_coding, pseudogene, or lncRNA."
    )]
    gene_type: Option<GffGeneType>,

    #[arg(
        long,
        default_value_t = 16,
        help = "Maximum number of threads",
        long_help = "Maximum number of threads to use for parallel processing. \n\
		     Choose the right number in HPC environments."
    )]
    max_threads: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of non-zero cutoff for rows/features",
        long_help = "Minimum number of non-zero entries required for rows/features to be included in the output."
    )]
    row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Minimum number of non-zero entries for the columns/cells",
        long_help = "Minimum number of non-zero entries required for columns/cells to be included in the output."
    )]
    column_nnz_cutoff: usize,

    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Output directory for the output files. \n\
		     This file will contain the results in the selected format."
    )]
    output: Box<str>,

    #[arg(
        long,
        short,
        help = "verbosity",
        long_help = "Enable verbose output `RUST_LOG=info`"
    )]
    verbose: bool,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for the output file",
        long_help = "Backend format for the output file. Options include zarr, hdf5, or other supported sparse IO backends."
    )]
    backend: SparseIoBackend,

    #[arg(
        long,
        default_value_t = false,
        help = "Include reads w/o barcode info",
        long_help = "Include reads that are missing gene and cell barcode information in the analysis."
    )]
    include_missing_barcode: bool,
}

impl PolyACountArgs {
    /// Create PolyASiteArgs from the command-line arguments
    fn polya_site_args(&self) -> PolyASiteArgs {
        PolyASiteArgs {
            min_tail_length: self.polya_min_tail_length,
            max_non_a_or_t_bases: self.polya_max_non_a_or_t,
            internal_prime_in: self.polya_internal_prime_window,
            internal_prime_a_or_t_count: self.polya_internal_prime_count,
        }
    }

    /// Create backend file path for a given batch name
    fn backend_file_path(&self, batch_name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, batch_name),
        }
        .into_boxed_str()
    }

    /// Get QC cutoffs
    fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff,
            column: self.column_nnz_cutoff,
        }
    }
}

/// Bin a position based on resolution
/// Returns (start, stop) where start is inclusive and stop is exclusive: [start, stop)
#[inline]
fn bin_position(position: i64, resolution: usize) -> (i64, i64) {
    let start = ((position as usize) / resolution * resolution) as i64;
    let stop = start + resolution as i64;
    (start, stop)
}

/// Generate unique batch names from BAM files
fn uniq_batch_names(bam_files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    let batch_names: Vec<Box<str>> = bam_files
        .iter()
        .map(|x| basename(x))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let unique_bams: HashSet<_> = bam_files.iter().cloned().collect();
    let unique_names: HashSet<_> = batch_names.iter().cloned().collect();

    // If all names are unique, use them as-is
    if unique_names.len() == bam_files.len() && unique_bams.len() == bam_files.len() {
        Ok(batch_names)
    } else {
        // Otherwise, add index suffix to ensure uniqueness
        info!("bam file (base) names are not unique");
        Ok(batch_names
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{}_{}", name, i).into_boxed_str())
            .collect())
    }
}

/// Count poly-A sites in RNA-seq bam files to quantify
/// poly-A cleavage sites at cell level
pub fn run_count_polya(args: &PolyACountArgs) -> anyhow::Result<()> {
    // Setup logging and environment
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    mkdir(&args.output)?;

    // Setup thread pool
    let max_threads = num_cpus::get().min(args.max_threads);
    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()?;
    info!("will use {} threads", rayon::current_num_threads());

    // Validate inputs
    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need at least one bam file"));
    }

    // Check BAM indices
    for bam_file in &args.bam_files {
        info!("checking .bai file for {}...", bam_file);
        check_bam_index(bam_file, None)?;
    }

    // Load and filter GFF
    info!("parsing GFF file: {}", args.gff_file);
    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    info!("found {} genes", gff_map.len());
    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    ///////////////////////////////////////
    // FIRST PASS: identify poly-A sites //
    ///////////////////////////////////////

    let gene_sites = find_all_polya_sites(&gff_map, args)?;

    if gene_sites.is_empty() {
        info!("no poly-A sites found");
        return Ok(());
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} poly-A sites", ndata);

    /////////////////////////////////////////////////////
    // SECOND PASS: collect cell-level counts at sites //
    /////////////////////////////////////////////////////

    let site_key = |x: &BedWithGene| -> Box<str> { format!("{}@pA", x).into_boxed_str() };
    let cutoffs = args.qc_cutoffs();
    let batch_names = uniq_batch_names(&args.bam_files)?;

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        process_bam_file(
            bam_file,
            &batch_name,
            &gene_sites,
            args,
            &gff_map,
            &site_key,
            &cutoffs,
        )?;
    }

    info!("done");
    Ok(())
}

///////////////////////////////////
// FIRST PASS: Find poly-A sites //
///////////////////////////////////

fn find_all_polya_sites(
    gff_map: &GffRecordMap,
    args: &PolyACountArgs,
) -> anyhow::Result<HashMap<GeneId, Vec<i64>>> {
    let njobs = gff_map.len();
    info!("Searching poly-A sites over {} genes", njobs);

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<i64>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_polya_sites_in_gene(rec, args, arc_gene_sites.clone())
        })?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

fn find_polya_sites_in_gene(
    gff_record: &GffRecord,
    args: &PolyACountArgs,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<i64>>>,
) -> anyhow::Result<()> {
    let mut polya_map = PolyASiteMap::new(args.polya_site_args());

    // Sweep all BAM files to find poly-A sites
    for bam_file in &args.bam_files {
        polya_map.update_bam_file_by_gene(bam_file, gff_record, &args.gene_barcode_tag)?;
    }

    // Filter by minimum coverage and insert into map
    let filtered_positions: Vec<i64> = polya_map
        .positions_with_counts()
        .into_iter()
        .filter(|(_, count)| *count >= args.min_coverage)
        .map(|(pos, _)| pos)
        .collect();

    if !filtered_positions.is_empty() {
        arc_gene_sites.insert(gff_record.gene_id.clone(), filtered_positions);
    }

    Ok(())
}

/////////////////////////////////////////////////////
// SECOND PASS: Collect cell-level counts at sites //
/////////////////////////////////////////////////////

fn process_bam_file(
    bam_file: &str,
    batch_name: &str,
    gene_sites: &HashMap<GeneId, Vec<i64>>,
    args: &PolyACountArgs,
    gff_map: &GffRecordMap,
    site_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    cutoffs: &SqueezeCutoffs,
) -> anyhow::Result<()> {
    info!(
        "collecting cell-level data over {} sites from {} ...",
        gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
        bam_file
    );

    let stats = gather_polya_stats(gene_sites, args, gff_map, bam_file)?;
    info!("collected {} cell-level poly-A counts", stats.len());

    info!(
        "aggregating the poly-A count triplets over {} stats...",
        stats.len()
    );

    let site_data_file = args.backend_file_path(batch_name);
    let triplets = summarize_stats(&stats, site_key);
    let data = triplets.to_backend(&site_data_file)?;
    data.qc(cutoffs.clone())?;
    info!("created data backend: {}", &site_data_file);

    Ok(())
}

fn gather_polya_stats(
    gene_sites: &HashMap<GeneId, Vec<i64>>,
    args: &PolyACountArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_count(gene_sites.len() as u64)
        .try_for_each(|gs| -> anyhow::Result<()> {
            let gene = gs.key();
            let sites = gs.value();

            if let Some(gff) = gff_map.get(gene) {
                let stats = collect_gene_stats(args, bam_file, &gff, gene, sites)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_stats(
    args: &PolyACountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    gene_id: &GeneId,
    positions: &[i64],
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    let mut all_stats = Vec::new();

    for &position in positions {
        let stats = collect_polya_counts_at_site(args, bam_file, gff_record, gene_id, position)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

fn collect_polya_counts_at_site(
    args: &PolyACountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    gene_id: &GeneId,
    position: i64,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    const PADDING: i64 = 100;

    let mut polya_map =
        PolyASiteMap::new_with_cell_barcode(args.polya_site_args(), &args.cell_barcode_tag);

    // Read BAM file for region around the poly-A site
    let mut gff = gff_record.clone();
    gff.start = (position - PADDING).max(0);
    gff.stop = position + PADDING;

    polya_map.update_bam_file_by_gene(bam_file, &gff, &args.gene_barcode_tag)?;

    // Extract and bin cell-level counts
    let Some(cell_counts) = polya_map.get_cell_counts_at(position) else {
        return Ok(Vec::new());
    };

    let (start, stop) = bin_position(position, args.resolution_bp);
    let chr = gff_record.seqname.as_ref();
    let strand = &gff_record.strand;

    let stats = cell_counts
        .iter()
        .filter(|(cb, _)| args.include_missing_barcode || *cb != &CellBarcode::Missing)
        .map(|(cell_barcode, count)| {
            (
                cell_barcode.clone(),
                BedWithGene {
                    chr: chr.into(),
                    start,
                    stop,
                    gene: gene_id.clone(),
                    strand: strand.clone(),
                },
                *count,
            )
        })
        .collect();

    Ok(stats)
}

///////////////////////////////////////////////////////
// Summarize poly-A statistics by aggregating counts //
///////////////////////////////////////////////////////

fn summarize_stats<F, T>(
    stats: &[(CellBarcode, BedWithGene, usize)],
    feature_key_func: F,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
{
    let combined_data: HashMap<(CellBarcode, T), usize> = HashMap::default();

    stats.par_iter().for_each(|(cb, bed, count)| {
        let key = (cb.clone(), feature_key_func(bed));
        combined_data
            .entry(key)
            .and_modify(|c| *c += count)
            .or_insert(*count);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, v as f32))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}
