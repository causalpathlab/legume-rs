use crate::common::*;
use crate::dartseq::sifter::*;
use crate::data::dna::Dna;
use crate::pipeline_util::*;
use rust_htslib::faidx;

use crate::data::cell_membership::CellMembership;
use crate::data::dna_stat_map::*;
use crate::data::methylation::*;
use crate::data::util_htslib::*;
use genomic_data::gff::{GeneId, GffRecordMap};

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use std::sync::{Arc, Mutex};

use crate::run_dartseq_count::DartSeqCountArgs;

/// Minimum number of positions required to attempt finding methylated sites
const MIN_LENGTH_FOR_TESTING: usize = 3;

/// Padding around target region when reading BAM files
const BAM_READ_PADDING: i64 = 1;

pub fn find_all_methylated_sites(
    gff_map: &GffRecordMap,
    args: &DartSeqCountArgs,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<HashMap<GeneId, Vec<MethylatedSite>>> {
    let njobs = gff_map.len();
    info!("Searching possible edit sites over {} blocks", njobs);

    // Validate reference genome
    info!("Loading reference genome: {}", args.genome_file);
    load_fasta_index(&args.genome_file)?;

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<MethylatedSite>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_methylated_sites_in_gene(rec, args, arc_gene_sites.clone(), cell_membership)
        })?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

fn find_methylated_sites_in_gene(
    gff_record: &GffRecord,
    args: &DartSeqCountArgs,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<MethylatedSite>>>,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();
    let strand = &gff_record.strand;
    let chr = gff_record.seqname.as_ref();

    // Each thread creates its own reader (faidx is not thread-safe)
    let faidx_reader = load_fasta_index(&args.genome_file)?;

    let candidate_sites = if let Some(membership) = cell_membership {
        // Use per-cell-type statistics when membership is provided
        find_sites_with_celltype_stats(gff_record, args, &faidx_reader, chr, strand, membership)?
    } else {
        // Use bulk statistics when no membership
        find_sites_with_bulk_stats(gff_record, args, &faidx_reader, chr, strand)?
    };

    if !candidate_sites.is_empty() {
        arc_gene_sites.insert(gene_id, candidate_sites);
    }

    Ok(())
}

/// Find methylated sites using bulk/marginal statistics (no cell type info)
fn find_sites_with_bulk_stats(
    gff_record: &GffRecord,
    args: &DartSeqCountArgs,
    faidx_reader: &faidx::Reader,
    chr: &str,
    strand: &Strand,
) -> anyhow::Result<Vec<MethylatedSite>> {
    let mut wt_base_freq_map = DnaBaseFreqMap::new();

    for wt_file in &args.wt_bam_files {
        wt_base_freq_map.update_from_gene(
            wt_file,
            gff_record,
            &args.gene_barcode_tag,
            args.include_missing_barcode,
        )?;
    }

    let positions = wt_base_freq_map.sorted_positions();

    if positions.len() < MIN_LENGTH_FOR_TESTING {
        return Ok(Vec::new());
    }

    let mut sifter = args.create_sifter(faidx_reader, chr, positions.len());

    let mut mut_base_freq_map = DnaBaseFreqMap::new();

    for mut_file in &args.mut_bam_files {
        mut_base_freq_map.update_from_gene(
            mut_file,
            gff_record,
            &args.gene_barcode_tag,
            args.include_missing_barcode,
        )?;
    }

    let wt_freq = wt_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count wt freq"))?;
    let mut_freq = mut_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count mut freq"))?;

    match strand {
        Strand::Forward => {
            sifter.forward_sweep(&positions, wt_freq, Some(mut_freq));
        }
        Strand::Backward => {
            sifter.backward_sweep(&positions, wt_freq, Some(mut_freq));
        }
    }

    let mut candidate_sites = sifter.candidate_sites;
    candidate_sites.sort();
    candidate_sites.dedup();
    Ok(candidate_sites)
}

/// Find methylated sites using per-cell-type statistics
fn find_sites_with_celltype_stats(
    gff_record: &GffRecord,
    args: &DartSeqCountArgs,
    faidx_reader: &faidx::Reader,
    chr: &str,
    strand: &Strand,
    membership: &CellMembership,
) -> anyhow::Result<Vec<MethylatedSite>> {
    // Get all cell types
    let cell_types = membership.cell_types();

    if cell_types.is_empty() {
        return Ok(Vec::new());
    }

    // Collect bulk frequencies from mut BAM files (background/null distribution)
    let mut mut_base_freq_map = DnaBaseFreqMap::new();
    for mut_file in &args.mut_bam_files {
        mut_base_freq_map.update_from_gene(
            mut_file,
            gff_record,
            &args.gene_barcode_tag,
            args.include_missing_barcode,
        )?;
    }
    let mut_freq = mut_base_freq_map.marginal_frequency_map();

    // Find sites for each cell type
    let mut all_candidate_sites = Vec::new();

    for cell_type in &cell_types {
        // Create frequency map for this cell type only
        let mut wt_base_freq_map =
            DnaBaseFreqMap::new_for_celltype(&args.cell_barcode_tag, membership, cell_type);

        for wt_file in &args.wt_bam_files {
            wt_base_freq_map.update_from_gene(
                wt_file,
                gff_record,
                &args.gene_barcode_tag,
                args.include_missing_barcode,
            )?;
        }

        let positions = wt_base_freq_map.sorted_positions();

        if positions.len() < MIN_LENGTH_FOR_TESTING {
            continue;
        }

        let wt_freq = match wt_base_freq_map.marginal_frequency_map() {
            Some(freq) => freq,
            None => continue,
        };

        let mut sifter = args.create_sifter(faidx_reader, chr, positions.len());

        match strand {
            Strand::Forward => {
                sifter.forward_sweep(&positions, wt_freq, mut_freq);
            }
            Strand::Backward => {
                sifter.backward_sweep(&positions, wt_freq, mut_freq);
            }
        }

        all_candidate_sites.extend(sifter.candidate_sites);
    }

    all_candidate_sites.sort();
    all_candidate_sites.dedup();
    Ok(all_candidate_sites)
}

//////////////////////////////////////////
// SECOND PASS: Collect cell-level data //
//////////////////////////////////////////

pub fn process_all_bam_files_to_bed(
    args: &DartSeqCountArgs,
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    gff_map: &GffRecordMap,
) -> anyhow::Result<()> {
    // Load cell membership file if provided
    let membership = if let Some(ref path) = args.cell_membership_file {
        let m = CellMembership::from_file(
            path,
            args.membership_barcode_col,
            args.membership_celltype_col,
            !args.exact_barcode_match,
        )?;
        info!(
            "Loaded {} cell barcodes from membership file: {}",
            m.num_cells(),
            path
        );
        info!("Prefix matching: {}", !args.exact_barcode_match);
        Some(m)
    } else {
        None
    };

    let wt_batch_names = uniq_batch_names(&args.wt_bam_files)?;

    for (bam_file, batch_name) in args.wt_bam_files.iter().zip(wt_batch_names) {
        let mut stats = gather_m6a_stats(gene_sites, args, gff_map, bam_file, membership.as_ref())?;
        write_bed(
            &mut stats,
            gff_map,
            &args.bed_file_path(&batch_name),
            membership.as_ref(),
            args,
        )?;
    }

    if args.output_null_data {
        info!("output null data");
        let mut_batch_names = uniq_batch_names(&args.mut_bam_files)?;

        for (bam_file, batch_name) in args.mut_bam_files.iter().zip(mut_batch_names) {
            let mut stats =
                gather_m6a_stats(gene_sites, args, gff_map, bam_file, membership.as_ref())?;
            write_bed(
                &mut stats,
                gff_map,
                &args.bed_file_path(&batch_name),
                membership.as_ref(),
                args,
            )?;
        }
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f64 / total as f64
            } else {
                0.0
            }
        );
    }

    Ok(())
}

pub fn process_all_bam_files_to_backend(
    args: &DartSeqCountArgs,
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    gff_map: &GffRecordMap,
) -> anyhow::Result<()> {
    // Load cell membership file if provided
    let membership = if let Some(ref path) = args.cell_membership_file {
        let m = CellMembership::from_file(
            path,
            args.membership_barcode_col,
            args.membership_celltype_col,
            !args.exact_barcode_match,
        )?;
        info!(
            "Loaded {} cell barcodes from membership file: {}",
            m.num_cells(),
            path
        );
        info!("Prefix matching: {}", !args.exact_barcode_match);
        Some(m)
    } else {
        None
    };

    let gene_key = create_gene_key_function(gff_map);
    let site_key = |x: &BedWithGene| -> Box<str> {
        let gene_part = gene_key(x);
        format!("{}_{}_{}_{}/m6A", gene_part, x.chr, x.start, x.stop).into_boxed_str()
    };
    let take_value = args.value_extractor();
    let cutoffs = args.qc_cutoffs();

    let mut genes = HashSet::<Box<str>>::default();
    let mut sites = HashSet::<Box<str>>::default();
    let mut gene_data_files: Vec<Box<str>> = vec![];
    let mut site_data_files: Vec<Box<str>> = vec![];
    let mut null_gene_data_files: Vec<Box<str>> = vec![];
    let mut null_site_data_files: Vec<Box<str>> = vec![];

    let wt_batch_names = uniq_batch_names(&args.wt_bam_files)?;

    for (bam_file, batch_name) in args.wt_bam_files.iter().zip(wt_batch_names) {
        process_bam_to_backend(
            bam_file,
            &batch_name,
            gene_sites,
            args,
            gff_map,
            &gene_key,
            &site_key,
            &take_value,
            &cutoffs,
            &mut genes,
            &mut sites,
            &mut gene_data_files,
            &mut site_data_files,
            membership.as_ref(),
        )?;
    }

    if args.output_null_data {
        info!("output null data");
        let mut_batch_names = uniq_batch_names(&args.mut_bam_files)?;

        for (bam_file, batch_name) in args.mut_bam_files.iter().zip(mut_batch_names) {
            process_bam_to_backend(
                bam_file,
                &batch_name,
                gene_sites,
                args,
                gff_map,
                &gene_key,
                &site_key,
                &take_value,
                &cutoffs,
                &mut genes,
                &mut sites,
                &mut null_gene_data_files,
                &mut null_site_data_files,
                membership.as_ref(),
            )?;
        }
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f64 / total as f64
            } else {
                0.0
            }
        );
    }

    // Reorder rows to ensure consistency across files
    reorder_all_matrices(
        args,
        genes,
        sites,
        gene_data_files,
        site_data_files,
        null_gene_data_files,
        null_site_data_files,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_bam_to_backend(
    bam_file: &str,
    batch_name: &str,
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    gene_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    site_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    take_value: &(impl Fn(&MethylationData) -> f32 + Send + Sync),
    cutoffs: &SqueezeCutoffs,
    genes: &mut HashSet<Box<str>>,
    sites: &mut HashSet<Box<str>>,
    gene_data_files: &mut Vec<Box<str>>,
    site_data_files: &mut Vec<Box<str>>,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    info!(
        "collecting data over {} sites from {} ...",
        gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
        bam_file
    );

    let stats = gather_m6a_stats(gene_sites, args, gff_map, bam_file, cell_membership)?;

    info!(
        "aggregating the '{}' triplets over {} stats...",
        args.output_value_type,
        stats.len()
    );

    if args.gene_level_output {
        let gene_data_file = args.backend_file_path(batch_name);
        let triplets = summarize_stats(&stats, gene_key, take_value);
        let data = triplets.to_backend(&gene_data_file)?;
        data.qc(cutoffs.clone())?;
        genes.extend(data.row_names()?);
        info!("created gene-level data: {}", &gene_data_file);
        gene_data_files.push(gene_data_file);
    } else {
        let site_data_file = args.backend_file_path(batch_name);
        let triplets = summarize_stats(&stats, site_key, take_value);
        let data = triplets.to_backend(&site_data_file)?;
        data.qc(cutoffs.clone())?;
        sites.extend(data.row_names()?);
        info!("created site-level data: {}", &site_data_file);
        site_data_files.push(site_data_file);
    }

    Ok(())
}

fn reorder_all_matrices(
    args: &DartSeqCountArgs,
    genes: HashSet<Box<str>>,
    sites: HashSet<Box<str>>,
    gene_data_files: Vec<Box<str>>,
    site_data_files: Vec<Box<str>>,
    null_gene_data_files: Vec<Box<str>>,
    null_site_data_files: Vec<Box<str>>,
) -> anyhow::Result<()> {
    let mut genes_sorted: Vec<_> = genes.into_iter().collect();
    genes_sorted.sort();

    let backend = &args.backend;

    for data_file in gene_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&genes_sorted)?;
    }

    if args.output_null_data {
        for data_file in null_gene_data_files {
            open_sparse_matrix(&data_file, backend)?.reorder_rows(&genes_sorted)?;
        }
    }

    let mut sites_sorted: Vec<_> = sites.into_iter().collect();
    sites_sorted.sort();

    for data_file in site_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
    }

    if args.output_null_data {
        for data_file in null_site_data_files {
            open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
        }
    }

    Ok(())
}

pub fn gather_m6a_stats(
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
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
                let stats = collect_gene_m6a_stats(args, bam_file, &gff, sites, cell_membership)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_m6a_stats(
    args: &DartSeqCountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    sites: &[MethylatedSite],
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut all_stats = Vec::new();

    for site in sites {
        let stats = estimate_m6a_stat(args, bam_file, gff_record, site, cell_membership)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

fn estimate_m6a_stat(
    args: &DartSeqCountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    m6a_c2u: &MethylatedSite,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map =
        DnaBaseFreqMap::new_with_cell_barcode(&args.cell_barcode_tag, cell_membership);
    let m6apos = m6a_c2u.m6a_pos;
    let c2upos = m6a_c2u.conversion_pos;

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    // Read BAM file for region around the m6A site
    let mut gff = gff_record.clone();
    gff.start = (lb - BAM_READ_PADDING).max(0);
    gff.stop = ub + BAM_READ_PADDING;
    stat_map.update_from_gene(
        bam_file,
        &gff,
        &args.gene_barcode_tag,
        args.include_missing_barcode,
    )?;

    let gene = gff.gene_id;
    let chr = gff.seqname.as_ref();
    let strand = &gff.strand;

    let (unmutated_base, mutated_base) = match strand {
        Strand::Forward => (Dna::C, Dna::T),
        Strand::Backward => (Dna::G, Dna::A),
    };

    // Set the anchor position for m6A
    let anchor_base = match strand {
        Strand::Forward => Dna::A,
        Strand::Backward => Dna::T,
    };
    stat_map.set_anchor_position(m6apos, anchor_base);

    let methylation_stat = stat_map.stratified_frequency_at(c2upos);

    let Some(meth_stat) = methylation_stat else {
        return Ok(Vec::new());
    };

    let (start, stop) = bin_position_kb(c2upos, args.resolution_kb);

    let stats = meth_stat
        .iter()
        .filter_map(|(cb, counts)| {
            let methylated = counts.get(Some(&mutated_base));
            let unmethylated = counts.get(Some(&unmutated_base));

            if (args.include_missing_barcode || cb != &CellBarcode::Missing) && methylated > 0 {
                Some((
                    cb.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start,
                        stop,
                        gene: gene.clone(),
                        strand: *strand,
                    },
                    MethylationData {
                        methylated,
                        unmethylated,
                        m6a_pos: m6apos,
                    },
                ))
            } else {
                None
            }
        })
        .collect();

    Ok(stats)
}

pub fn write_bed(
    stats: &mut [(CellBarcode, BedWithGene, MethylationData)],
    gff_map: &GffRecordMap,
    file_path: &str,
    cell_membership: Option<&CellMembership>,
    args: &DartSeqCountArgs,
) -> anyhow::Result<()> {
    use rust_htslib::bgzf::Writer as BWriter;
    use std::io::Write;

    stats.par_sort_by(|a, b| a.1.cmp(&b.1));

    let lines: Vec<_> = stats
        .iter()
        .map(|(cb, bg, data)| {
            let gene_string = gff_map
                .get(&bg.gene)
                .map(|gff| match gff.gene_name {
                    GeneSymbol::Symbol(x) => format!("{}_{}", &bg.gene, x),
                    GeneSymbol::Missing => format!("{}", &bg.gene),
                })
                .unwrap_or_else(|| format!("{}", &bg.gene));

            if args.output_cell_types {
                if let Some(membership) = cell_membership {
                    let cell_type = membership
                        .matches_barcode(cb)
                        .unwrap_or_else(|| "unknown".into());
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.methylated,
                        data.unmethylated,
                        cb,
                        data.m6a_pos,
                        cell_type
                    )
                    .into_boxed_str()
                } else {
                    // No membership provided but cell types requested - just output unknown
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tunknown",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.methylated,
                        data.unmethylated,
                        cb,
                        data.m6a_pos
                    )
                    .into_boxed_str()
                }
            } else {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    bg.chr,
                    bg.start,
                    bg.stop,
                    bg.strand,
                    gene_string,
                    data.methylated,
                    data.unmethylated,
                    cb,
                    data.m6a_pos
                )
                .into_boxed_str()
            }
        })
        .collect();

    let header: &[u8] = if args.output_cell_types {
        b"#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\tm6a_pos\tcell_type\n"
    } else {
        b"#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\tm6a_pos\n"
    };

    let mut writer = BWriter::from_path(file_path)?;
    writer.write_all(header)?;
    for line in lines {
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(())
}
