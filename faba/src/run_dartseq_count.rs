use crate::common::*;
use crate::dartseq_io::ToParquet;
use crate::dartseq_sifter::*;
use crate::dartseq_stat::Histogram;
use crate::data::dna::Dna;

use crate::data::dna_stat_map::*;
use crate::data::dna_stat_traits::*;
use crate::data::gff::FeatureType as GffFeatureType;
use crate::data::gff::GeneType as GffGeneType;
use crate::data::gff::{GeneId, GffRecordMap};
use crate::data::methylation::*;

use crate::data::util_htslib::*;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};

#[derive(Args, Debug)]
pub struct DartSeqCountArgs {
    #[arg(
        short = 'w',
        long = "wt",
        value_delimiter = ',',
        required = true,
        help = "Observed (wild-type) BAM files.",
        long_help = "Comma-separated list of observed (wild-type) BAM files. \n\
		       These files contain C->U (C->T) conversions, \n\
		       representing the editing events in the wild-type sample. \n\
		       Example: file1.bam,file2.bam"
    )]
    wt_bam_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "mut",
        value_delimiter = ',',
        required = true,
        help = "Background/control (mutant) BAM files.",
        long_help = "Comma-separated list of control (mutant) BAM files. \n\
		     These files are used to calibrate background mutation rates \n\
		     to identify disrupted C->U (C->T) conversions in the mutant sample. \n\
		     Example: mut1.bam,mut2.bam"
    )]
    mut_bam_files: Vec<Box<str>>,

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
        short = 'r',
        long,
        help = "resolution (in kb)",
        long_help = "Resolution for binning in kilobases (kb). \n\
		     Determines the size of site-level reports. \n\
		     Example: 1.0"
    )]
    resolution_kb: Option<f32>,

    #[arg(
        long = "genome-bins",
        default_value_t = 57,
        help = "#bins for genomic locations in histogram",
        long_help = "Number of bins for genomic locations in the histogram. \n\
		     Controls the granularity of the histogram across the genome."
    )]
    num_genomic_bins_histogram: usize,

    #[arg(
        long = "print_width",
        default_value_t = 40,
        help = "#bins in histogram when printing on the screen",
        long_help = "Approximate number of bins in the output histogram. \n\
		     Adjusts the print width for visualization."
    )]
    histogram_print_width: usize,

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
        help = "minimum number of total reads per site",
        long_help = "Minimum number of total reads required per site for inclusion in the analysis. \n\
		     Filters out low-coverage sites."
    )]
    min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = 5,
        help = "Minimum number of converted reads (T) in wild-type",
        long_help = "Minimum number of converted reads (C->T) required in wild-type sample. \n\
		     Ensures sufficient conversion signal beyond just frequency."
    )]
    min_conversion: usize,

    #[arg(
        long = "pseudocount",
        default_value_t = 1,
        help = "Pseudocount for null distribution in binomial test",
        long_help = "Pseudocount added to background/mutant counts for regularization. \n\
		     Helps avoid overly confident p-values with zero background counts."
    )]
    pseudocount: usize,

    #[arg(
        long = "min-wt-maf",
        default_value_t = 0.01,
        help = "Minimum frequency of `C->U` in an edit site on the wild type",
        long_help = "Minimum frequency of C->U conversion at an edit site in the wild-type sample. \n\
		     Used to filter out low-frequency events."
    )]
    min_methylation_maf: f64,

    #[arg(
        long = "max-mut-maf",
        default_value_t = 0.01,
        help = "Maximum frequency `C->U` on the mutant",
        long_help = "Maximum frequency of C->U conversion at an edit site in the mutant sample. \n\
		     Used to filter out background events."
    )]
    max_background_maf: f64,

    #[arg(
        short,
        long,
        default_value_t = 0.01,
        help = "Maximum detection p-value cutoff",
        long_help = "Maximum p-value cutoff for detection. \n\
		     Sites with p-values above this threshold will be excluded."
    )]
    pvalue_cutoff: f64,

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
        short = 't',
        long,
        value_enum,
        default_value = "beta",
        help = "Type of output value to report",
        long_help = "Type of output value to report. Options include beta, count, or other supported types."
    )]
    output_value_type: MethFeatureType,

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

    #[arg(
        long,
        default_value_t = false,
        help = "Output mutant signals",
        long_help = "Output mutant signals (null data) in addition to wild-type signals."
    )]
    output_null_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Output results in BED",
        long_help = "Output results in BED file format for genomic intervals."
    )]
    output_bed_file: bool,

    #[arg(
        long = "gene-level",
        default_value_t = false,
        help = "Output results at a gene level (default: a site level)",
        long_help = "Output results at a gene level (default: a site level).\n\
		     The counts will be aggregated within a gene."
    )]
    gene_level_output: bool,

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
        default_value_t = false,
        help = "Print histogram to stdout",
        long_help = "Print histogram of gene features to stdout. \n\
		     The histogram will be saved to a file regardless of this option."
    )]
    print_histogram: bool,
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

/// Count possibly methylated A positions in DART-seq bam files to
/// quantify m6A β values
///
pub fn run_count_dartseq(args: &DartSeqCountArgs) -> anyhow::Result<()> {
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

    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("need pairs of bam files"));
    }

    for x in args.wt_bam_files.iter() {
        info!("checking .bai file for {}...", x);
        check_bam_index(x, None)?;
    }

    for x in args.mut_bam_files.iter() {
        info!("checking .bai file for {}...", x);
        check_bam_index(x, None)?;
    }

    info!("parsing GFF file: {}", args.gff_file);

    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    // gff_map.add_padding(1000);

    info!("found {} genes", gff_map.len(),);

    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    mkdir(&args.output)?;

    /////////////////////////////////////
    // figure out potential edit sites //
    /////////////////////////////////////

    let njobs = gff_map.len();
    info!("Searching possible edit sites over {} blocks", njobs);

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<MethylatedSite>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .for_each(|rec| {
            find_methylated_sites_in_gene(rec, args, arc_gene_sites.clone())
                .expect("failed in find_methylated_sites")
        });

    if arc_gene_sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    let gene_sites = Arc::try_unwrap(arc_gene_sites)
        .map_err(|_| anyhow::anyhow!("failed to release gene_sites"))?;

    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    info!("Found {} m6A sites", ndata);

    gene_sites.to_parquet(&gff_map, format!("{}/sites.parquet", args.output))?;

    ////////////////////////////////
    // output marginal statistics //
    ////////////////////////////////

    let gene_feature_count =
        gene_sites.count_gene_features(&args.gff_file, args.num_genomic_bins_histogram)?;

    if args.print_histogram {
        gene_feature_count.print(args.histogram_print_width);
    }

    gene_feature_count.to_tsv(&format!("{}/gene_feature_count.tsv.gz", args.output))?;

    //////////////////////////
    // Take cell level data //
    //////////////////////////

    let take_value = |dat: &MethylationData| -> f32 {
        match args.output_value_type {
            MethFeatureType::Beta => {
                let tot = (dat.methylated + dat.unmethylated) as f32;
                (dat.methylated as f32) / tot.max(1.)
            }
            MethFeatureType::Methylated => dat.methylated as f32,
            MethFeatureType::Unmethylated => dat.unmethylated as f32,
        }
    };

    let gene_key = |x: &BedWithGene| -> Box<str> {
        gff_map
            .get(&x.gene)
            .map(|gff| format!("{}_{}", gff.gene_id, gff.gene_name))
            .unwrap_or(format!("{}", x.gene))
            .into_boxed_str()
    };

    let site_key = |x: &BedWithGene| -> Box<str> { x.to_string().into_boxed_str() };

    let backend = args.backend.clone();
    let backend_file = |name: &str| -> Box<str> {
        match backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &args.output, name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &args.output, name),
        }
        .into_boxed_str()
    };

    let bed_file =
        |name: &str| -> Box<str> { format!("{}/{}.bed.gz", &args.output, name).into_boxed_str() };

    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    let mut genes = HashSet::<Box<str>>::default();
    let mut sites = HashSet::<Box<str>>::default();
    let mut gene_data_files: Vec<Box<str>> = vec![];
    let mut site_data_files: Vec<Box<str>> = vec![];

    let mut null_gene_data_files: Vec<Box<str>> = vec![];
    let mut null_site_data_files: Vec<Box<str>> = vec![];

    let wt_batch_names = uniq_batch_names(&args.wt_bam_files)?;
    let mut_batch_names = uniq_batch_names(&args.mut_bam_files)?;

    for (bam_file, batch_name) in args.wt_bam_files.iter().zip(wt_batch_names) {
        ////////////////////////////
        // collect the statistics //
        ////////////////////////////

        info!(
            "collecting data over {} sites from {} ...",
            gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
            bam_file
        );

        let mut stats = gather_m6a_stats(&gene_sites, args, &gff_map, &bam_file)?;

        if args.output_bed_file {
            write_bed(&mut stats, &gff_map, &bed_file(&batch_name))?;
        } else {
            //////////////////////////////////
            // Aggregate them into triplets //
            //////////////////////////////////

            info!(
                "aggregating the '{}' triplets over {} stats...",
                args.output_value_type,
                stats.len()
            );

            if args.gene_level_output {
                let gene_data_file = backend_file(&format!("{}", batch_name));
                let triplets = summarize_stats(&stats, gene_key, take_value);
                let data = triplets.to_backend(&gene_data_file)?;
                data.qc(cutoffs.clone())?;
                genes.extend(data.row_names()?);
                info!("created gene-level data: {}", &gene_data_file);
                gene_data_files.push(gene_data_file);
            } else {
                let site_data_file = backend_file(&format!("{}", batch_name));
                let triplets = summarize_stats(&stats, site_key, take_value);
                let data = triplets.to_backend(&site_data_file)?;
                data.qc(cutoffs.clone())?;
                sites.extend(data.row_names()?);
                info!("created site-level data: {}", &site_data_file);
                site_data_files.push(site_data_file);
            }
        }
    }

    if args.output_null_data {
        info!("output null data");

        for (bam_file, batch_name) in args.mut_bam_files.iter().zip(mut_batch_names) {
            ////////////////////////////
            // collect the statistics //
            ////////////////////////////

            let mut stats = gather_m6a_stats(&gene_sites, args, &gff_map, &bam_file)?;

            if args.output_bed_file {
                write_bed(&mut stats, &gff_map, &bed_file(&batch_name))?;
            } else {
                //////////////////////////////////
                // Aggregate them into triplets //
                //////////////////////////////////
                if args.gene_level_output {
                    let gene_data_file = backend_file(&format!("{}", batch_name));
                    let triplets = summarize_stats(&stats, gene_key, take_value);
                    let data = triplets.to_backend(&gene_data_file)?;
                    data.qc(cutoffs.clone())?;
                    genes.extend(data.row_names()?);
                    info!("created gene-level data: {}", &gene_data_file);
                    null_gene_data_files.push(gene_data_file);
                } else {
                    let site_data_file = backend_file(&format!("{}", batch_name));
                    let triplets = summarize_stats(&stats, site_key, take_value);
                    let data = triplets.to_backend(&site_data_file)?;
                    data.qc(cutoffs.clone())?;
                    sites.extend(data.row_names()?);
                    info!("created site-level data: {}", &site_data_file);
                    null_site_data_files.push(site_data_file);
                }
            }
        }
    }

    if !args.output_bed_file {
        let mut genes_sorted = genes.into_iter().collect::<Vec<_>>();
        genes_sorted.sort();

        for data_file in gene_data_files {
            open_sparse_matrix(&data_file, &backend)?.reorder_rows(&genes_sorted)?;
        }

        if args.output_null_data {
            for data_file in null_gene_data_files {
                open_sparse_matrix(&data_file, &backend)?.reorder_rows(&genes_sorted)?;
            }
        }

        let mut sites_sorted = sites.into_iter().collect::<Vec<_>>();
        sites_sorted.sort();

        for data_file in site_data_files {
            open_sparse_matrix(&data_file, &backend)?.reorder_rows(&sites_sorted)?;
        }

        if args.output_null_data {
            for data_file in null_site_data_files {
                open_sparse_matrix(&data_file, &backend)?.reorder_rows(&sites_sorted)?;
            }
        }
    }
    info!("done");
    Ok(())
}

////////////////////////////////////////////////
// Step 1: find possibly methylated positions //
////////////////////////////////////////////////

fn find_methylated_sites_in_gene(
    gff_record: &GffRecord,
    args: &DartSeqCountArgs,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<MethylatedSite>>>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();
    let strand = &gff_record.strand;

    ///////////////////////////////////////////////////////
    // 1. sweep all the bam files to find variable sites //
    ///////////////////////////////////////////////////////
    let mut wt_base_freq_map = DnaBaseFreqMap::new();

    for wt_file in args.wt_bam_files.iter() {
        wt_base_freq_map.update_bam_file_by_gene(wt_file, gff_record, &args.gene_barcode_tag)?;
    }

    let positions = wt_base_freq_map.sorted_positions();

    if positions.len() >= 5 {
        // 2. find AC/T patterns: Using mutant statistics as null
        // distribution, it will keep possible C->U edit positions.
        let mut sifter = DartSeqSifter {
            min_coverage: args.min_coverage,
            min_conversion: args.min_conversion,
            pseudocount: args.pseudocount,
            min_meth_cutoff: args.min_methylation_maf,
            max_pvalue_cutoff: args.pvalue_cutoff,
            max_mutant_cutoff: args.max_background_maf,
            candidate_sites: Vec::with_capacity(positions.len()),
        };

        // gather background frequency map
        let mut mut_base_freq_map = DnaBaseFreqMap::new();

        for mut_file in args.mut_bam_files.iter() {
            mut_base_freq_map.update_bam_file_by_gene(
                mut_file,
                gff_record,
                &args.gene_barcode_tag,
            )?;
        }

        let wt_freq = wt_base_freq_map
            .marginal_frequency_map()
            .ok_or(anyhow::anyhow!("failed to count wt freq"))?;
        let mut_freq = mut_base_freq_map
            .marginal_frequency_map()
            .ok_or(anyhow::anyhow!("failed to count mut freq"))?;

        match &strand {
            Strand::Forward => {
                sifter.forward_sweep(&positions, &wt_freq, &mut_freq);
            }
            Strand::Backward => {
                sifter.backward_sweep(&positions, &wt_freq, &mut_freq);
            }
        };

        let mut ret = sifter.candidate_sites.clone();

        if !ret.is_empty() {
            ret.sort();
            ret.dedup();
            arc_gene_sites.insert(gene_id, ret);
        }
    }
    Ok(())
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit possible C2U positions and collect m6A sites, //
// locations and samples.                                        //
///////////////////////////////////////////////////////////////////

fn gather_m6a_stats(
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
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
                for x in sites {
                    ret.extend(
                        estimate_m6a_stat(args, bam_file, &gff, x).expect("failed to collect stat"),
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

fn estimate_m6a_stat(
    args: &DartSeqCountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    m6a_c2u: &MethylatedSite,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map = DnaBaseFreqMap::new_with_cell_barcode(&args.cell_barcode_tag);
    let m6apos = m6a_c2u.m6a_pos;
    let c2upos = m6a_c2u.conversion_pos;

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    // need to read bam files with the matching gene contexts
    // but read just what we need
    let mut gff = gff_record.clone();
    gff.start = (lb - 1).max(0); // padding
    gff.stop = ub + 1; // padding
    stat_map.update_bam_file_by_gene(bam_file, &gff, &args.gene_barcode_tag)?;

    let gene = gff.gene_id;
    let chr = gff.seqname.as_ref();
    let strand = &gff.strand;

    let unmutated_base = match strand {
        Strand::Forward => Dna::C,
        Strand::Backward => Dna::G,
    };

    let mutated_base = match strand {
        Strand::Forward => Dna::T,
        Strand::Backward => Dna::A,
    };

    // set the anchor position for m6A
    match strand {
        Strand::Forward => {
            stat_map.set_anchor_position(m6apos, Dna::A);
        }
        Strand::Backward => {
            stat_map.set_anchor_position(m6apos, Dna::T);
        }
    };

    let methylation_stat = stat_map.stratified_frequency_at(c2upos);

    let (lb, ub) = if let Some(r) = args.resolution_kb {
        // report reduced kb resolution
        let r = (r * 1000.0) as usize;
        (
            ((m6apos as usize) / r * r) as i64,
            ((m6apos as usize).div_ceil(r) * r) as i64,
        )
    } else {
        // report bp resolution
        (m6apos, m6apos + 1)
    };

    let mut ret = vec![];

    if let Some(meth_stat) = methylation_stat {
        for (cb, counts) in meth_stat {
            let methylated = counts.get(Some(&mutated_base));
            let unmethylated = counts.get(Some(&unmutated_base));

            if (args.include_missing_barcode || cb != &CellBarcode::Missing) && methylated > 0 {
                ret.push((
                    cb.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start: lb,
                        stop: ub,
                        gene: gene.clone(),
                        strand: strand.clone(),
                    },
                    MethylationData {
                        methylated,
                        unmethylated,
                    },
                ));
            }
        }
    }

    Ok(ret)
}

//////////////////////////////////////////////////////////
// Step 3: repackaging them into desired output formats //
//////////////////////////////////////////////////////////

fn summarize_stats<F, V, T>(
    stats: &[(CellBarcode, BedWithGene, MethylationData)],
    feature_key_func: F,
    value_func: V,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
    V: Fn(&MethylationData) -> f32,
{
    let combined_data: HashMap<(CellBarcode, T), MethylationData> = HashMap::default();
    stats.par_iter().for_each(|(cb, k, dat)| {
        let key = (cb.clone(), feature_key_func(k));
        combined_data.entry(key).or_default().add_assign(dat);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, value_func(&v)))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}

fn write_bed(
    stats: &mut [(CellBarcode, BedWithGene, MethylationData)],
    gff_map: &GffRecordMap,
    file_path: &str,
) -> anyhow::Result<()> {
    use std::io::Write;

    stats.par_sort_by(|a, b| a.1.cmp(&b.1));

    let lines = stats
        .into_iter()
        .map(|(cb, bg, data)| {
            let gene_string = if let Some(gff) = gff_map.get(&bg.gene) {
                match gff.gene_name {
                    GeneSymbol::Symbol(x) => format!("{}_{}", &bg.gene, x),
                    GeneSymbol::Missing => format!("{}", &bg.gene),
                }
            } else {
                format!("{}", &bg.gene)
            };

            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                bg.chr,
                bg.start,
                bg.stop,
                bg.strand,
                gene_string,
                data.methylated,
                data.unmethylated,
                cb
            )
            .into_boxed_str()
        })
        .collect::<Vec<_>>();

    use rust_htslib::bgzf::Writer as BWriter;

    let header = "#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\n";

    let mut writer = BWriter::from_path(file_path)?;
    writer.write_all(header.as_bytes())?;
    for l in lines {
        writer.write_all(l.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
