use crate::common::*;
use crate::dartseq_sifter::*;
use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_htslib::*;
use crate::data::gff::*;
use crate::data::methylation::*;
use crate::data::positions::*;
use crate::data::sam::*;
use crate::data::util_htslib::*;

use data_beans::sparse_io::*;
use matrix_util::common_io::*;
use matrix_util::mtx_io::*;
use std::collections::{HashMap, HashSet};

#[derive(Args, Debug)]
pub struct CountDartSeqArgs {
    /// Observed (WT) `.bam` files where `C->U` (`C->T`) conversions happen
    #[arg(short, long, value_delimiter = ',', required = true)]
    wt_bam_files: Vec<Box<str>>,

    /// Control (MUT) `.bam` files where `C->U` (`C->T`) conversion is disrupted
    #[arg(short, long, value_delimiter = ',', required = true)]
    mut_bam_files: Vec<Box<str>>,

    /// Gene annotation (`GFF`) file
    #[arg(short, long, required = true)]
    gff_file: Box<str>,

    /// block size for parallelism (bp)
    #[arg(short = 'b', long, default_value_t = 100_000)]
    block_size: usize,

    /// (10x) cell/sample barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "CB")]
    cell_barcode_tag: Box<str>,

    /// gene barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "GX")]
    gene_barcode_tag: Box<str>,

    /// resolution (bp)
    #[arg(short = 'r', long)]
    resolution: Option<usize>,

    /// minimum number of total reads per site
    #[arg(long, default_value_t = 3)]
    min_coverage: usize,

    /// minimum number of reads at `C->U` edit events
    #[arg(long, default_value_t = 3)]
    min_conversion: usize,

    /// maximum detection p-value cutoff
    #[arg(short, long, default_value_t = 0.05)]
    pvalue_cutoff: f64,

    /// save .mtx file along with row and column names
    #[arg(long, default_value_t = false)]
    save_mtx: bool,

    /// keep unnamed cell/sample barcode
    #[arg(long, default_value_t = false)]
    keep_unnamed_tag: bool,

    /// gene feature type (gene, transcript, exon, utr)
    #[arg(long)]
    gene_feature_type: Option<Box<str>>,

    /// output value type
    #[arg(short = 't', long, value_enum, default_value = "methylated")]
    output_value_type: MethFeatureType,

    /// output `data-beans` file (with `.h5` or `.zarr` ext)
    #[arg(short, long, default_value = "temp.h5")]
    out_file: Box<str>,
}

/// Count possibly methylated A positions in DART-seq bam files to
/// quantify m6A β values
///
pub fn run_count_dartseq(args: &CountDartSeqArgs) -> anyhow::Result<()> {
    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("need matching pairs of bam files"));
    }

    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("empty bam files"));
    }

    for x in args.wt_bam_files.iter() {
        check_bam_index(x, None)?;
    }

    for x in args.mut_bam_files.iter() {
        check_bam_index(x, None)?;
    }

    let gene_feature_type: Option<FeatureType> =
        args.gene_feature_type.clone().map(|x| x.as_ref().into());

    info!("parsing GFF file: {}", args.gff_file);

    let gff_map = GffRecordMap::from(args.gff_file.as_ref(), gene_feature_type.as_ref())?;

    info!("found {} features", gff_map.len(),);

    ////////////////////////////////////////
    // 1. figure out potential edit sites //
    ////////////////////////////////////////

    let njobs = gff_map.len();
    info!("Searching possible edit sites over {} blocks", njobs);
    let sites = gff_map
        .records()
        .into_iter()
        .par_bridge()
        .progress_count(njobs as u64)
        .map(|rec| find_methylated_sites_in_gene(rec, args))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .filter(|x| x.gene_id != Gene::Missing)
        .collect::<Vec<_>>();

    ///////////////////////////////////
    // 2. collect all the statistics //
    ///////////////////////////////////

    let nsites = sites.len() as u64;
    info!("collecting statistics over {} sites...", nsites);

    let stats = sites
        .into_iter()
        .par_bridge()
        .progress_count(nsites)
        .map(|x| collect_m6a_stat(args, &x))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    /////////////////////////////////////
    // 3. Aggregate them into triplets //
    /////////////////////////////////////

    info!("aggregating {} value triplets...", args.output_value_type);

    // aggregate over gene-level
    let mut gene_level_data: HashMap<(GeneId, CellBarcode), MethylationData> = HashMap::new();

    for (cb, k, dat) in stats {
        let gene = k.gene;
        let accum = gene_level_data.entry((gene, cb)).or_default();
        *accum += dat;
    }

    // let mut triplets: HashMap<(CellGene, MethylationKey), MethylationData> = HashMap::new();

    // for (s, k, dat) in stats {
    //     let accum = triplets.entry((s, k)).or_default();
    //     *accum += dat;
    // }

    // let (mut triplets, row_names, col_names) =
    //     format_data_triplets(triplets.into_iter().collect(), args);

    // if args.save_mtx {
    //     let ext = extension(&args.out_file)?;
    //     let mtx_file = args.out_file.replace(ext.as_ref(), "mtx.gz");
    //     let row_file = args.out_file.replace(ext.as_ref(), "rows.gz");
    //     let col_file = args.out_file.replace(ext.as_ref(), "columns.gz");
    //     let nrow = row_names.len();
    //     let ncol = col_names.len();
    //     triplets.par_sort_by_key(|&(row, _, _)| row);
    //     triplets.par_sort_by_key(|&(_, col, _)| col);
    //     write_mtx_triplets(&triplets, nrow, ncol, &mtx_file)?;
    //     write_lines(&row_names, &row_file)?;
    //     write_lines(&col_names, &col_file)?;
    // }

    // /////////////////////////////
    // // 4. construct data-beans //
    // /////////////////////////////
    // let backend = match extension(&args.out_file)?.as_ref() {
    //     "h5" => SparseIoBackend::HDF5,
    //     "zarr" => SparseIoBackend::Zarr,
    //     _ => SparseIoBackend::Zarr,
    // };

    // let mtx_shape = (row_names.len(), col_names.len(), triplets.len());

    // let mut data =
    //     create_sparse_from_triplets(triplets, mtx_shape, Some(&args.out_file), Some(&backend))?;
    // data.register_column_names_vec(&col_names);
    // data.register_row_names_vec(&row_names);

    info!("done");
    Ok(())
}

fn find_methylated_sites_in_gene(
    rec: &GffRecord,
    args: &CountDartSeqArgs,
) -> anyhow::Result<Vec<MethylatedSite>> {
    let chr = rec.seqname.clone();
    let lb = rec.start;
    let ub = rec.stop;
    let strand = rec.strand.clone();
    let gene_id = rec.gene_id.clone();

    // 1. sweep each pair bam files to find variable sites
    let mut wt_freq_map = DnaBaseFreqMap::new(&args.cell_barcode_tag, &args.gene_barcode_tag);
    let mut mut_freq_map = DnaBaseFreqMap::new(&args.cell_barcode_tag, &args.gene_barcode_tag);

    for wt_file in args.wt_bam_files.iter() {
        wt_freq_map.update_by_gene_in_bam(wt_file, &gene_id, (chr.as_ref(), lb, ub))?;
    }

    for mut_file in args.mut_bam_files.iter() {
        mut_freq_map.update_by_gene_in_bam(mut_file, &gene_id, (chr.as_ref(), lb, ub))?;
    }

    // 2. find AC/T patterns: Using mutant statistics as null
    // distribution, it will keep possible C->U edit positions.
    let mut sifter = DartSeqSifter {
        seqname: chr,
        gene_id: gene_id.clone(),
        min_coverage: args.min_coverage,
        min_conversion: args.min_conversion,
        max_pvalue_cutoff: args.pvalue_cutoff,
        candidate_sites: vec![],
    };

    let positions = wt_freq_map.sorted_positions();

    if positions.len() >= 3 {
        let wt_freq = wt_freq_map.marginal_frequency_by_position();
        let mut_freq = mut_freq_map.marginal_frequency_by_position();

        match &strand {
            Strand::Forward => {
                sifter.forward_sweep(&positions, &wt_freq, &mut_freq);
            }
            Strand::Backward => {
                sifter.backward_sweep(&positions, &wt_freq, &mut_freq);
            }
        };
    }

    Ok(sifter.candidate_sites)
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit possible C2U positions and collect m6A sites, //
// locations and samples.                                        //
///////////////////////////////////////////////////////////////////

fn collect_m6a_stat(
    args: &CountDartSeqArgs,
    chr_m6a_c2u: &MethylatedSite,
) -> anyhow::Result<Vec<(CellBarcode, MethylationKey, MethylationData)>> {
    let mut stat_map = DnaBaseFreqMap::new(&args.cell_barcode_tag, &args.gene_barcode_tag);
    let m6apos = chr_m6a_c2u.m6a_pos;
    let c2upos = chr_m6a_c2u.conversion_pos;
    let chr = chr_m6a_c2u.seqname.as_ref();
    let gene = &chr_m6a_c2u.gene_id;
    let strand = &chr_m6a_c2u.strand;

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    for _file in args.wt_bam_files.iter() {
        stat_map.update_by_region_in_bam(_file.as_ref(), (chr, lb, ub))?;
    }

    let methylation_stat = stat_map.frequency_at(c2upos);

    let unmethylated_base = match strand {
        Strand::Forward => Dna::C,
        Strand::Backward => Dna::G,
    };

    let methylated_base = match strand {
        Strand::Forward => Dna::T,
        Strand::Backward => Dna::A,
    };

    let (lb, ub) = if let Some(r) = args.resolution {
        (
            ((m6apos as usize) / r * r + 1) as i64,
            ((m6apos as usize).div_ceil(r) * r) as i64,
        )
    } else {
        (m6apos, m6apos + 1)
    };

    let mut ret = vec![];

    if let Some(meth_stat) = methylation_stat {
        for (s, counts) in meth_stat {
            ret.push((
                s.clone(),
                MethylationKey {
                    chr: chr.into(),
                    lb,
                    ub,
                    gene: gene.clone(),
                },
                MethylationData {
                    methylated: counts.get(Some(&methylated_base)),
                    unmethylated: counts.get(Some(&unmethylated_base)),
                },
            ));
        }
    }

    Ok(ret)
}

// fn format_data_triplets(
//     triplets: Vec<((CellBarcode, MethylationKey), MethylationData)>,
//     args: &CountDartSeqArgs,
// ) -> (Vec<(u64, u64, f32)>, Vec<Box<str>>, Vec<Box<str>>) {
//     // identify unique samples and sites
//     let mut unique_cells = HashSet::new();
//     let mut unique_sites = HashSet::new();

//     for ((cb, k), _) in &triplets {
//         unique_cells.insert(cb.clone());
//         unique_sites.insert(k.clone());
//     }

//     // assign indices to cells
//     let mut unique_cells = unique_cells.into_iter().collect::<Vec<_>>();
//     unique_cells.sort();

//     let cell_to_index: HashMap<CellBarcode, usize> = unique_cells
//         .into_iter()
//         .enumerate()
//         .map(|(i, sample)| (sample, i))
//         .collect();

//     // assign unique indices to gene site pairs
//     let mut unique_sites = unique_sites.into_iter().collect::<Vec<_>>();
//     unique_sites.par_sort();

//     let site_indices: HashMap<MethylationKey, usize> = unique_sites
//         .into_iter()
//         .enumerate()
//         .map(|(i, site)| (site, i))
//         .collect();

//     // // relabel triplets with indices
//     // let mut relabeled_triplets = Vec::with_capacity(triplets.len());
//     // for ((s, k), dat) in triplets {
//     //     let row_idx = site_indices[&k] as u64;
//     //     let col_idx = sample_indices[&s] as u64;

//     //     let value = match args.feature_type {
//     //         MethFeatureType::Beta => {
//     //             let tot = (dat.methylated + dat.unmethylated) as f32;
//     //             let beta = (dat.methylated as f32) / tot.max(1.);
//     //             beta
//     //         }
//     //         MethFeatureType::Methylated => dat.methylated as f32,

//     //         MethFeatureType::Unmethylated => dat.unmethylated as f32,
//     //     };

//     //     relabeled_triplets.push((row_idx, col_idx, value));
//     // }

//     // let mut samples = vec!["".into(); sample_indices.len()];
//     // for (k, j) in sample_indices {
//     //     samples[j] = k.to_string().into_boxed_str();
//     // }

//     // let mut sites = vec!["".into(); site_indices.len()];
//     // for (k, j) in site_indices {
//     //     sites[j] = k.to_string().into_boxed_str();
//     // }

//     // (relabeled_triplets, sites, samples)
// }
