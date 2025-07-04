use crate::common::*;
use crate::dartseq_sifter::*;
use crate::data::dna::Dna;
use crate::data::dna_stat_map::*;
use crate::data::dna_stat_traits::*;
use crate::data::methylation::*;
use crate::data::positions::*;
use crate::data::util_htslib::*;

use data_beans::qc::*;
use data_beans::sparse_io::*;
use fnv::FnvHashMap as HashMap;
use matrix_util::common_io::remove_file;

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

    /// bam record type (gene, transcript, exon, utr)
    #[arg(long, default_value = "gene")]
    record_type: Box<str>,

    /// gene type (protein_coding, pseudogene, lncRNA)
    #[arg(long, default_value = "protein_coding")]
    gene_type: Box<str>,

    /// number of non-zero cutoff for rows/features
    #[arg(short, long, default_value_t = 100)]
    row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns/cells
    #[arg(short, long, default_value_t = 100)]
    column_nnz_cutoff: usize,

    /// output value type
    #[arg(short = 't', long, value_enum, default_value = "methylated")]
    output_value_type: MethFeatureType,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// output header for `data-beans` files
    #[arg(short, long, required = true)]
    output: Box<str>,
}

/// Count possibly methylated A positions in DART-seq bam files to
/// quantify m6A β values
///
pub fn run_count_dartseq(args: &DartSeqCountArgs) -> anyhow::Result<()> {
    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("need pairs of bam files"));
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

    let record_feature_type: FeatureType = args.record_type.as_ref().into();

    info!("parsing GFF file: {}", args.gff_file);

    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref(), Some(&record_feature_type))?;
    gff_map.subset(args.gene_type.clone().into());

    info!("found {} features", gff_map.len(),);

    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

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

    if sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    ///////////////////////////////////
    // 2. collect all the statistics //
    ///////////////////////////////////

    let nsites = sites.len() as u64;
    info!("collecting statistics over {} sites...", nsites);

    let stats = sites
        .into_iter()
        .par_bridge()
        .progress_count(nsites)
        .map(|x| collect_m6a_stat(args, &gff_map, &x))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    if stats.is_empty() {
        info!("empty stats");
        return Ok(());
    }

    /////////////////////////////////////
    // 3. Aggregate them into triplets //
    /////////////////////////////////////

    let backend = args.backend.clone();

    info!(
        "aggregating the '{}' triplets over {} stats...",
        args.output_value_type,
        stats.len()
    );

    let take_value = |dat: &MethylationData| -> f32 {
        match args.output_value_type {
            MethFeatureType::Beta => {
                let tot = (dat.methylated + dat.unmethylated) as f32;
                let beta = (dat.methylated as f32) / tot.max(1.);
                beta
            }
            MethFeatureType::Methylated => dat.methylated as f32,
            MethFeatureType::Unmethylated => dat.unmethylated as f32,
        }
    };

    {
        let mut site_level_data: HashMap<(BedWithGene, CellBarcode), MethylationData> =
            HashMap::default();

        for (cb, k, dat) in stats.iter() {
            let site = k.clone();
            let cell = cb.clone();
            let accum = site_level_data.entry((site, cell)).or_default();
            accum.add_assign(dat);
        }

        let site_level_data = site_level_data
            .into_iter()
            .map(|((s, c), v)| (c, s, take_value(&v)))
            .collect::<Vec<_>>();

        let (triplets, row_names, col_names) = format_data_triplets(site_level_data);

        let mtx_shape = (row_names.len(), col_names.len(), triplets.len());
        let output = args.output.clone();
        let backend_file = match backend {
            SparseIoBackend::HDF5 => format!("{}_site.h5", &output),
            SparseIoBackend::Zarr => format!("{}_site.zarr", &output),
        };

        remove_file(&backend_file)?;

        let mut data =
            create_sparse_from_triplets(triplets, mtx_shape, Some(&backend_file), Some(&backend))?;
        data.register_column_names_vec(&col_names);
        data.register_row_names_vec(&row_names);
        if args.row_nnz_cutoff > 0 || args.column_nnz_cutoff > 0 {
            squeeze_by_nnz(
                data,
                SqueezeCutoffs {
                    row: args.row_nnz_cutoff,
                    column: args.column_nnz_cutoff,
                },
                args.block_size,
            )?;
        }
    }

    {
        // aggregate over gene-level
        let mut gene_level_data: HashMap<(GeneId, CellBarcode), MethylationData> =
            HashMap::default();

        for (cb, k, dat) in stats.iter() {
            let gene = k.gene.clone();
            let cell = cb.clone();
            let accum = gene_level_data.entry((gene, cell)).or_default();
            accum.add_assign(dat);
        }

        let gene_level_data = gene_level_data
            .into_iter()
            .map(|((s, c), v)| (c, s, take_value(&v)))
            .collect::<Vec<_>>();

        let (triplets, row_names, col_names) = format_data_triplets(gene_level_data);

        let mtx_shape = (row_names.len(), col_names.len(), triplets.len());
        let output = args.output.clone();
        let backend_file = match backend {
            SparseIoBackend::HDF5 => format!("{}_gene.h5", &output),
            SparseIoBackend::Zarr => format!("{}_gene.zarr", &output),
        };

        remove_file(&backend_file)?;

        let mut data =
            create_sparse_from_triplets(triplets, mtx_shape, Some(&backend_file), Some(&backend))?;
        data.register_column_names_vec(&col_names);
        data.register_row_names_vec(&row_names);
        if args.row_nnz_cutoff > 0 || args.column_nnz_cutoff > 0 {
            squeeze_by_nnz(
                data,
                SqueezeCutoffs {
                    row: args.row_nnz_cutoff,
                    column: args.column_nnz_cutoff,
                },
                args.block_size,
            )?;
        }
    }

    info!("done");
    Ok(())
}

////////////////////////////////////////////////
// Step 1: find possibly methylated positions //
////////////////////////////////////////////////

fn find_methylated_sites_in_gene(
    rec: &GffRecord,
    args: &DartSeqCountArgs,
) -> anyhow::Result<Vec<MethylatedSite>> {
    let chr = rec.seqname.clone();
    let strand = rec.strand.clone();
    let gene_id = rec.gene_id.clone();

    // 1. sweep each pair bam files to find variable sites
    let mut wt_freq_map = DnaBaseFreqMap::new(&args.cell_barcode_tag);
    let mut mut_freq_map = DnaBaseFreqMap::new(&args.cell_barcode_tag);

    for wt_file in args.wt_bam_files.iter() {
        wt_freq_map.update_bam_by_gene(wt_file, rec, &args.gene_barcode_tag)?;
    }

    for mut_file in args.mut_bam_files.iter() {
        mut_freq_map.update_bam_by_gene(mut_file, rec, &args.gene_barcode_tag)?;
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
        let wt_freq = wt_freq_map.marginal_frequency_map();
        let mut_freq = mut_freq_map.marginal_frequency_map();

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
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    chr_m6a_c2u: &MethylatedSite,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map = DnaBaseFreqMap::new(&args.cell_barcode_tag);
    let m6apos = chr_m6a_c2u.m6a_pos;
    let c2upos = chr_m6a_c2u.conversion_pos;
    let chr = chr_m6a_c2u.seqname.as_ref();
    let gene = &chr_m6a_c2u.gene_id;
    let strand = &chr_m6a_c2u.strand;

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    // need to read bam files with the matching gene contexts
    if let Some(gff_record) = gff_map.get(gene) {
        let mut gff = gff_record.clone();
        gff.start = (lb - 1).max(0); // padding
        gff.stop = ub + 1; // padding
        for _file in args.wt_bam_files.iter() {
            stat_map.update_bam_by_gene(_file, &gff, &args.gene_barcode_tag)?;
        }
    }

    let methylation_stat = stat_map.stratified_frequency_at(c2upos);

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
            let methylated = counts.get(Some(&methylated_base));
            let unmethylated = counts.get(Some(&unmethylated_base));

            if (s != &CellBarcode::Missing) && (methylated > 0) {
                ret.push((
                    s.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start: lb,
                        stop: ub,
                        gene: gene.clone(),
                    },
                    MethylationData {
                        methylated,
                        unmethylated,
                    },
                ));
            }
        }
    } //  else {
    //     panic!("");
    // }

    Ok(ret)
}
