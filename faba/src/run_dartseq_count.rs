use crate::common::*;
use crate::dartseq_sifter::*;
use crate::data::dna::Dna;

use crate::data::dna_stat_map::*;
use crate::data::dna_stat_traits::*;
use crate::data::gff::FeatureType as GffFeatureType;
use crate::data::gff::GeneType as GffGeneType;
use crate::data::methylation::*;
use crate::data::util_htslib::*;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use matrix_util::common_io;
use rayon::ThreadPoolBuilder;
use std::ops::Div;
use std::sync::{Arc, Mutex};

#[derive(Args, Debug)]
pub struct DartSeqCountArgs {
    /// Observed (WT) `.bam` files where `C->U` (`C->T`) conversions happen
    #[arg(short = 'w', long = "wt", value_delimiter = ',', required = true)]
    wt_bam_files: Vec<Box<str>>,

    /// Control (MUT) `.bam` files where `C->U` (`C->T`) conversion is disrupted
    #[arg(short = 'm', long = "mut", value_delimiter = ',', required = true)]
    mut_bam_files: Vec<Box<str>>,

    /// Gene annotation (`GFF`) file
    #[arg(short = 'g', long = "gff", required = true)]
    gff_file: Box<str>,

    /// resolution (in kb)
    #[arg(short = 'r', long)]
    resolution_kb: Option<f32>,

    /// #bins for genomic locations in histogram
    #[arg(long = "genome-bins", default_value_t = 57)]
    num_genomic_bins_histogram: usize,

    /// (approximate) number of bins in histogram
    #[arg(long, default_value_t = 40)]
    histogram_print_width: usize,

    /// (10x) cell/sample barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "CB")]
    cell_barcode_tag: Box<str>,

    /// gene barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "GX")]
    gene_barcode_tag: Box<str>,

    /// minimum number of total reads per site
    #[arg(long, default_value_t = 3)]
    min_coverage: usize,

    /// minimum frequency of `C->U` on an edit site
    #[arg(long = "min-wt-maf", default_value_t = 0.01)]
    min_methylation_maf: f64,

    /// maximum allele frequency `C->U` on the mutant
    #[arg(long = "max-mut-maf", default_value_t = 0.01)]
    max_background_maf: f64,

    /// maximum detection p-value cutoff
    #[arg(short, long, default_value_t = 0.01)]
    pvalue_cutoff: f64,

    /// selectively choose bam record type (gene, transcript, exon, utr)
    #[arg(long, value_enum)]
    record_type: Option<GffFeatureType>,

    /// gene type (protein_coding, pseudogene, lncRNA)
    #[arg(long, value_enum)]
    gene_type: Option<GffGeneType>,

    /// maximum number of threads
    #[arg(long, default_value_t = 16)]
    max_threads: usize,

    /// number of non-zero cutoff for rows/features
    #[arg(long, default_value_t = 10)]
    row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns/cells
    #[arg(long, default_value_t = 10)]
    column_nnz_cutoff: usize,

    /// output value type
    #[arg(short = 't', long, value_enum, default_value = "beta")]
    output_value_type: MethFeatureType,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// include reads missing gene and cell barcode
    #[arg(long, default_value_t = false)]
    include_missing_barcode: bool,

    /// output mut signals
    #[arg(long, default_value_t = false)]
    output_null_data: bool,

    /// output bed file
    #[arg(long, default_value_t = false)]
    output_bed_file: bool,

    /// output mut file
    #[arg(long, default_value_t = false)]
    output_mut_file: bool,

    /// output header for `data-beans` files
    #[arg(short, long, required = true)]
    output: Box<str>,
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

    ////////////////////////////////
    // output marginal statistics //
    ////////////////////////////////

    let gene_feature_count =
        gene_sites.count_gene_features(&args.gff_file, args.num_genomic_bins_histogram)?;

    gene_feature_count.print(args.histogram_print_width);

    gene_feature_count.to_tsv(&format!("{}.gene_feature_count.tsv.gz", args.output))?;

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
            SparseIoBackend::HDF5 => format!("{}.{}.h5", &args.output, name),
            SparseIoBackend::Zarr => format!("{}.{}.zarr", &args.output, name),
        }
        .into_boxed_str()
    };

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
            let bed_file = format!("{}_{}.bed.gz", args.output, batch_name);
            info!("writing down stats to the bed file: {}", &bed_file);
            write_bed(&mut stats, &bed_file)?;
        } else {
            //////////////////////////////////
            // Aggregate them into triplets //
            //////////////////////////////////

            info!(
                "aggregating the '{}' triplets over {} stats...",
                args.output_value_type,
                stats.len()
            );

            let gene_data_file = backend_file(&format!("gene_{}", batch_name));
            let triplets = summarize_stats(&stats, gene_key, take_value);
            let data = triplets.to_backend(&gene_data_file)?;
            data.qc(cutoffs.clone())?;
            genes.extend(data.row_names()?);
            info!("created gene-level data: {}", &gene_data_file);
            gene_data_files.push(gene_data_file);

            let site_data_file = backend_file(&format!("site_{}", batch_name));
            let triplets = summarize_stats(&stats, site_key, take_value);
            let data = triplets.to_backend(&site_data_file)?;
            data.qc(cutoffs.clone())?;
            sites.extend(data.row_names()?);
            info!("created site-level data: {}", &site_data_file);
            site_data_files.push(site_data_file);
        }
    }

    if args.output_null_data {
        info!("output null data");
        for (bam_file, batch_name) in args.mut_bam_files.iter().zip(mut_batch_names) {
            ////////////////////////////
            // collect the statistics //
            ////////////////////////////

            let stats = gather_m6a_stats(&gene_sites, args, &gff_map, &bam_file)?;

            //////////////////////////////////
            // Aggregate them into triplets //
            //////////////////////////////////

            let gene_data_file = backend_file(&format!("gene_{}", batch_name));
            let triplets = summarize_stats(&stats, gene_key, take_value);
            let data = triplets.to_backend(&gene_data_file)?;
            data.qc(cutoffs.clone())?;
            genes.extend(data.row_names()?);
            info!("created gene-level data: {}", &gene_data_file);
            null_gene_data_files.push(gene_data_file);

            let site_data_file = backend_file(&format!("site_{}", batch_name));
            let triplets = summarize_stats(&stats, site_key, take_value);
            let data = triplets.to_backend(&site_data_file)?;
            data.qc(cutoffs.clone())?;
            sites.extend(data.row_names()?);
            info!("created site-level data: {}", &site_data_file);
            null_site_data_files.push(site_data_file);
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
    let mut wt_freq_map = DnaBaseFreqMap::new();

    for wt_file in args.wt_bam_files.iter() {
        wt_freq_map.update_bam_by_gene(wt_file, gff_record, &args.gene_barcode_tag)?;
    }

    let positions = wt_freq_map.sorted_positions();

    if positions.len() >= 3 {
        // 2. find AC/T patterns: Using mutant statistics as null
        // distribution, it will keep possible C->U edit positions.
        let mut sifter = DartSeqSifter {
            min_coverage: args.min_coverage,
            min_meth_cutoff: args.min_methylation_maf,
            max_pvalue_cutoff: args.pvalue_cutoff,
            max_mutant_cutoff: args.max_background_maf,
            candidate_sites: Vec::with_capacity(positions.len()),
        };

        // gather background frequency map
        let mut mut_freq_map = DnaBaseFreqMap::new();

        for mut_file in args.mut_bam_files.iter() {
            mut_freq_map.update_bam_by_gene(mut_file, gff_record, &args.gene_barcode_tag)?;
        }

        let wt_freq = wt_freq_map
            .marginal_frequency_map()
            .ok_or(anyhow::anyhow!("failed to count wt freq"))?;
        let mut_freq = mut_freq_map
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
    stat_map.update_bam_by_gene(bam_file, &gff, &args.gene_barcode_tag)?;

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
    file_path: &str,
) -> anyhow::Result<()> {
    use std::io::Write;

    stats.par_sort_by(|a, b| a.1.cmp(&b.1));

    let lines = stats
        .into_iter()
        .map(|(cb, bg, data)| {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                bg.chr,
                bg.start,
                bg.stop,
                bg.strand,
                bg.gene,
                data.methylated,
                data.unmethylated,
                cb
            )
            .into_boxed_str()
        })
        .collect::<Vec<_>>();

    use rust_htslib::bgzf::Writer as BWriter;

    let mut writer = BWriter::from_path(file_path)?;
    for l in lines {
        writer.write_all(l.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/////////////////////
// other utilities //
/////////////////////

struct GeneFeatureCount {
    five_prime: HashMap<MethBin, Vec<usize>>,
    cds: HashMap<MethBin, Vec<usize>>,
    three_prime: HashMap<MethBin, Vec<usize>>,
}

impl GeneFeatureCount {
    fn get_meth_bins(&self) -> Vec<MethBin> {
        let mut meth_bins: Vec<MethBin> = self
            .cds
            .iter()
            .map(|x| x.key().clone())
            .chain(self.five_prime.iter().map(|x| x.key().clone()))
            .chain(self.three_prime.iter().map(|x| x.key().clone()))
            .collect();

        meth_bins.sort();
        meth_bins.dedup();
        meth_bins
    }

    fn max_count(map: &HashMap<MethBin, Vec<usize>>, k: &MethBin) -> usize {
        map.get(k)
            .and_then(|v| v.iter().cloned().max())
            .unwrap_or(0)
    }

    fn print(&self, max_width: usize) {
        fn print_row(label: &str, data: &[usize], scale: usize, max_width: usize) {
            for &n in data {
                let n1 = n.div_ceil(scale);
                let n0 = max_width.saturating_sub(n1);
                eprintln!(
                    "{:<6}{}{} {}",
                    label,
                    vec!["*"; n1].join(""),
                    vec![" "; n0].join(""),
                    n
                );
            }
        }

        let meth_bins = self.get_meth_bins();

        for k in &meth_bins {
            eprintln!("{}", k);

            let nmax = [
                GeneFeatureCount::max_count(&self.cds, &k),
                GeneFeatureCount::max_count(&self.five_prime, &k),
                GeneFeatureCount::max_count(&self.three_prime, &k),
            ]
            .into_iter()
            .max()
            .unwrap();

            let scale = nmax.div_ceil(max_width);

            if let Some(x) = self.five_prime.get(k) {
                print_row("5'UTR", x.value(), scale, max_width);
            }
            if let Some(x) = self.cds.get(k) {
                print_row("CDS", x.value(), scale, max_width);
            }
            if let Some(x) = self.three_prime.get(k) {
                print_row("3'UTR", x.value(), scale, max_width);
            }
        }
    }

    fn to_tsv(&self, file_path: &str) -> anyhow::Result<()> {
        fn into_boxed_str(label: &str, meth_bin: &MethBin, data: &[usize]) -> Vec<Box<str>> {
            data.iter()
                .enumerate()
                .map(|(i, &n)| format!("{}\t{}\t{}\t{}", label, i, meth_bin, n).into_boxed_str())
                .collect()
        }

        let mut writer = common_io::open_buf_writer(file_path)?;

        writer.write_all(
            b"#feature\tgenomic_bin\t-log10MAF(methylated)\t-log10MAF(background)\tcount\n",
        )?;

        let meth_bins = self.get_meth_bins();

        for k in &meth_bins {
            if let Some(data) = self.five_prime.get(k) {
                for l in into_boxed_str("5UTR", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
            if let Some(data) = self.three_prime.get(k) {
                for l in into_boxed_str("3UTR", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
            if let Some(data) = self.cds.get(k) {
                for l in into_boxed_str("CDS", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }
}

trait Histogram {
    fn count_on_feature_map(
        &self,
        feature_map: &HashMap<GeneId, GffRecord>,
        n_genomic_bins: usize,
    ) -> HashMap<MethBin, Vec<usize>>;

    fn count_gene_features(
        &self,
        gff_file: &str,
        n_genomic_bins: usize,
    ) -> anyhow::Result<GeneFeatureCount>;
}

impl Histogram for HashMap<GeneId, Vec<MethylatedSite>> {
    fn count_gene_features(
        &self,
        gff_file: &str,
        n_genomic_bins: usize,
    ) -> anyhow::Result<GeneFeatureCount> {
        let gff_records = read_gff_record_vec(gff_file)?;

        let UTRMap {
            five_prime,
            three_prime,
        } = build_utr_map(&gff_records)?;

        let cds = build_gene_map(&gff_records, Some(&FeatureType::CDS))?;

        let n_five_prime = five_prime.take_max_length();
        let n_cds = cds.take_max_length();
        let n_three_prime = three_prime.take_max_length();
        let ntot = n_five_prime + n_cds + n_three_prime;

        let nbins_five_prime = n_five_prime as usize * n_genomic_bins / ntot as usize;
        let nbins_cds = n_cds as usize * n_genomic_bins / ntot as usize;
        let nbins_three_prime = n_three_prime as usize * n_genomic_bins / ntot as usize;

        let five_prime = self.count_on_feature_map(&five_prime, nbins_five_prime);
        let cds = self.count_on_feature_map(&cds, nbins_cds);
        let three_prime = self.count_on_feature_map(&three_prime, nbins_three_prime);

        Ok(GeneFeatureCount {
            five_prime,
            cds,
            three_prime,
        })
    }

    fn count_on_feature_map(
        &self,
        gene_gff_map: &HashMap<GeneId, GffRecord>,
        n_genomic_bins: usize,
    ) -> HashMap<MethBin, Vec<usize>> {
        let ret = HashMap::new();

        self.iter().for_each(|x| {
            let g = x.key();
            let sites = x.value();

            if let Some(gff) = gene_gff_map.get(&g) {
                let lb = gff.start; // 1-based
                let ub = gff.stop; // 1-based
                let length = (ub - lb).max(1) as usize;

                for s in sites.iter() {
                    if s.m6a_pos < ub && s.m6a_pos >= lb {
                        let beta_bin = MethBin::from(s, &gff.strand);

                        let mut entry = ret.entry(beta_bin).or_insert(vec![0; n_genomic_bins + 1]);
                        let genomic = entry.value_mut();

                        // relative position with respect to (lb and ub)
                        let rel_pos = (match gff.strand {
                            Strand::Forward => (s.m6a_pos - lb) as usize,
                            Strand::Backward => (ub - s.m6a_pos) as usize,
                        } * n_genomic_bins)
                            .div(length + 1);

                        genomic[rel_pos] += 1;
                    }
                }
            }
        });

        ret
    }
}

#[derive(Eq, Hash, PartialEq, PartialOrd, Ord, Clone)]
struct MethBin {
    neg_log10_p_wt: usize,
    neg_log10_p_mut: usize,
}

impl MethBin {
    fn from(s: &MethylatedSite, strand: &Strand) -> Self {
        let pmin = 1e-4;
        let (wt_unmeth, wt_meth, mut_unmeth, mut_meth) = match strand {
            Strand::Forward => (
                s.wt_freq.get(Some(&Dna::C)) as f32,
                s.wt_freq.get(Some(&Dna::T)) as f32,
                s.mut_freq.get(Some(&Dna::C)) as f32,
                s.mut_freq.get(Some(&Dna::T)) as f32,
            ),
            Strand::Backward => (
                s.wt_freq.get(Some(&Dna::G)) as f32,
                s.wt_freq.get(Some(&Dna::A)) as f32,
                s.mut_freq.get(Some(&Dna::G)) as f32,
                s.mut_freq.get(Some(&Dna::A)) as f32,
            ),
        };

        Self {
            neg_log10_p_wt: Self::neg_log10_beta(wt_meth, wt_unmeth, pmin),
            neg_log10_p_mut: Self::neg_log10_beta(mut_meth, mut_unmeth, pmin),
        }
    }

    fn neg_log10_beta(meth: f32, unmeth: f32, pmin: f32) -> usize {
        (-(meth / (unmeth + meth).max(1.0)).max(pmin).log10()) as usize
    }
}

impl From<MethBin> for Box<str> {
    fn from(meth_bin: MethBin) -> Self {
        format!("{}\t{}", meth_bin.neg_log10_p_wt, meth_bin.neg_log10_p_mut).into_boxed_str()
    }
}

impl std::fmt::Display for MethBin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\t{}", self.neg_log10_p_wt, self.neg_log10_p_mut)
    }
}
