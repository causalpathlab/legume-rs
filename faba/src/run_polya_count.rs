use crate::common::*;
use crate::data::dna_stat_traits::*;
use crate::data::gff::FeatureType as GffFeatureType;
use crate::data::gff::GeneType as GffGeneType;
use crate::data::gff::{GeneId, GffRecordMap};
use crate::data::poly_a_stat_map::{PolyASiteArgs, PolyASiteMap};
use crate::data::util_htslib::*;

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
        default_value_t = 10,
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

fn uniq_batch_names(bam_files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    let batch_names: Vec<Box<str>> = bam_files
        .iter()
        .map(|x| basename(x))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let n_bam_files = bam_files.len();
    let n_uniq_bam_files = bam_files.iter().cloned().collect::<HashSet<_>>().len();
    let n_batches = batch_names.iter().cloned().collect::<HashSet<_>>().len();

    Ok(
        if n_batches == n_bam_files && n_batches == n_uniq_bam_files {
            batch_names
        } else {
            info!("bam file (base) names are not unique");

            batch_names
                .iter()
                .enumerate()
                .map(|(i, x)| format!("{}_{}", x, i).into_boxed_str())
                .collect()
        },
    )
}

/// Count poly-A sites in RNA-seq bam files to quantify
/// poly-A cleavage sites at cell level
///
pub fn run_count_polya(args: &PolyACountArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // create output directory
    mkdir(&args.output)?;

    let max_threads = num_cpus::get().min(args.max_threads);

    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()?;

    info!("will use {} threads", rayon::current_num_threads());

    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need at least one bam file"));
    }

    for x in args.bam_files.iter() {
        info!("checking .bai file for {}...", x);
        check_bam_index(x, None)?;
    }

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

    mkdir(&args.output)?;

    ///////////////////////////////////////
    // FIRST PASS: identify poly-A sites //
    ///////////////////////////////////////

    let njobs = gff_map.len();
    info!("Searching poly-A sites over {} genes", njobs);

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<i64>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .for_each(|rec| {
            find_polya_sites_in_gene(rec, args, arc_gene_sites.clone())
                .expect("failed in find_polya_sites_in_gene")
        });

    if arc_gene_sites.is_empty() {
        info!("no poly-A sites found");
        return Ok(());
    }

    let gene_sites = Arc::try_unwrap(arc_gene_sites)
        .map_err(|_| anyhow::anyhow!("failed to release gene_sites"))?;

    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    info!("Found {} poly-A sites", ndata);

    /////////////////////////////////////////////////////
    // SECOND PASS: collect cell-level counts at sites //
    /////////////////////////////////////////////////////

    let backend = args.backend.clone();
    let backend_file = |name: &str| -> Box<str> {
        match backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &args.output, name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &args.output, name),
        }
        .into_boxed_str()
    };

    let batch_names = uniq_batch_names(&args.bam_files)?;

    for (bam_file, _batch_name) in args.bam_files.iter().zip(batch_names) {
        info!(
            "collecting cell-level data over {} sites from {} ...",
            gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
            bam_file
        );

        let stats = gather_polya_stats(&gene_sites, args, &gff_map, &bam_file)?;

        info!("collected {} cell-level poly-A counts", stats.len());

        let site_data_file = backend_file(&format!("{}", _batch_name));

        // TODO: Output stats to appropriate format
        // This will be similar to the dartseq output logic
        // Can be zarr, hdf5, or bed format depending on requirements
    }

    info!("done");
    Ok(())
}

////////////////////////////////////////////////
// Step 1: find poly-A sites in each gene    //
////////////////////////////////////////////////

fn find_polya_sites_in_gene(
    gff_record: &GffRecord,
    args: &PolyACountArgs,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<i64>>>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();

    let polya_args = PolyASiteArgs {
        min_tail_length: args.polya_min_tail_length,
        max_non_a_or_t_bases: args.polya_max_non_a_or_t,
        internal_prime_in: args.polya_internal_prime_window,
        internal_prime_a_or_t_count: args.polya_internal_prime_count,
    };

    /////////////////////////////////////////////////////
    // sweep all the bam files to find poly-A sites   //
    /////////////////////////////////////////////////////
    let mut polya_map = PolyASiteMap::new(polya_args);

    for bam_file in args.bam_files.iter() {
        polya_map.update_bam_file_by_gene(bam_file, gff_record, &args.gene_barcode_tag)?;
    }

    let positions_with_counts = polya_map.positions_with_counts();

    if !positions_with_counts.is_empty() {
        // Filter by minimum coverage
        let filtered_positions: Vec<i64> = positions_with_counts
            .into_iter()
            .filter(|(_, count)| *count >= args.min_coverage)
            .map(|(pos, _)| pos)
            .collect();

        if !filtered_positions.is_empty() {
            arc_gene_sites.insert(gene_id, filtered_positions);
        }
    }

    Ok(())
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit poly-A sites and collect cell-level counts   //
///////////////////////////////////////////////////////////////////

fn gather_polya_stats(
    gene_sites: &HashMap<GeneId, Vec<i64>>,
    args: &PolyACountArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, i64, usize)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();

    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_count(gene_sites.len() as u64)
        .for_each(|gs| {
            let gene = gs.key();
            let sites = gs.value();
            if let Some(gff) = gff_map.get(gene) {
                let mut ret = Vec::with_capacity(sites.len());
                for &pos in sites {
                    ret.extend(
                        collect_polya_counts_at_site(args, bam_file, &gff, gene, pos)
                            .expect("failed to collect poly-A counts"),
                    );
                }
                arc_ret.lock().expect("lock").extend(ret);
            }
        });

    let stats = Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()?;
    Ok(stats)
}

fn collect_polya_counts_at_site(
    args: &PolyACountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    gene_id: &GeneId,
    position: i64,
) -> anyhow::Result<Vec<(CellBarcode, GeneId, i64, usize)>> {
    let polya_args = PolyASiteArgs {
        min_tail_length: args.polya_min_tail_length,
        max_non_a_or_t_bases: args.polya_max_non_a_or_t,
        internal_prime_in: args.polya_internal_prime_window,
        internal_prime_a_or_t_count: args.polya_internal_prime_count,
    };

    let mut polya_map = PolyASiteMap::new_with_cell_barcode(polya_args, &args.cell_barcode_tag);

    // Read bam file for this specific region around the poly-A site
    let mut gff = gff_record.clone();
    let padding = 100; // Add some padding around the position
    gff.start = (position - padding).max(0);
    gff.stop = position + padding;

    polya_map.update_bam_file_by_gene(bam_file, &gff, &args.gene_barcode_tag)?;

    // Extract cell-level counts from polya_map at the specific position
    let mut ret = vec![];

    if let Some(cell_counts) = polya_map.get_cell_counts_at(position) {
        for (cell_barcode, count) in cell_counts {
            if args.include_missing_barcode || cell_barcode != &CellBarcode::Missing {
                ret.push((cell_barcode.clone(), gene_id.clone(), position, *count));
            }
        }
    }

    Ok(ret)
}
