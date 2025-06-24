use crate::common::*;
use crate::data::alignment::SamSampleName;
use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat::*;
use crate::data::util::*;
use crate::hypothesis_tests::BinomTest;

use std::collections::{HashMap, HashSet};

#[derive(Args, Debug)]
pub struct CountDartSeqArgs {
    /// Observed (WT) `.bam` files where `C->U` (`C->T`) conversions happen
    #[arg(short, long, value_delimiter = ',', required = true)]
    wt_bam_files: Vec<Box<str>>,

    /// Control (MUT) `.bam` files where `C->U` (`C->T`) conversion is disrupted
    #[arg(short, long, value_delimiter = ',', required = true)]
    mut_bam_files: Vec<Box<str>>,

    /// block size for parallelism (bp)
    #[arg(short = 'b', long, default_value_t = 100_000)]
    block_size: usize,

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

    /// output file
    #[arg(short, long, default_value = "output.h5")]
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

    let bam_file = args.wt_bam_files[0].as_ref();
    let jobs = create_bam_jobs(bam_file, Some(args.block_size))?;

    ////////////////////////////////////////
    // 1. figure out potential edit sites //
    ////////////////////////////////////////

    let c2u_sites_samples = jobs
        .par_iter()
        .progress_count(jobs.len() as u64)
        .map(|(chr, lb, ub)| -> anyhow::Result<_> { find_c2u_site(args, chr.as_ref(), lb, ub) })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut c2u_sites = Vec::with_capacity(c2u_sites_samples.len());
    let mut samples = HashSet::new();

    for (c2u, sam) in c2u_sites_samples {
        c2u_sites.extend(c2u);
        samples.extend(sam.into_iter());
    }

    // mapping sample name -> index
    let sample_index = samples
        .iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect::<HashMap<_, _>>();

    info!("Found {} samples", sample_index.len());

    c2u_sites.dedup();
    c2u_sites.par_sort();

    info!("Found {} C->U sites", c2u_sites.len());

    ///////////////////////////////////
    // 2. collect all the statistics //
    ///////////////////////////////////

    let mut bed_data = c2u_sites
        .par_iter()
        .map(|(chr, c2u_pos)| -> anyhow::Result<Vec<_>> { collect_m6a_variant(args, chr, c2u_pos) })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    ///////////////////////////////////////////////////////////
    // 3. Sort BED and rename/index genomic features&samples //
    ///////////////////////////////////////////////////////////

    info!("sorting {} edit statistics...", &bed_data.len());
    fn compare_chr_pos(a: &(&str, usize), b: &(&str, usize)) -> std::cmp::Ordering {
        let chr_cmp = a.0.cmp(b.0);
        if chr_cmp == std::cmp::Ordering::Equal {
            a.1.cmp(&b.1)
        } else {
            chr_cmp
        }
    }
    if bed_data.len() > 100_000 {
        bed_data.par_sort_by(|a, b| compare_chr_pos(&(a.0.as_ref(), a.1), &(b.0.as_ref(), b.1)));
    } else {
        bed_data.sort_by(|a, b| compare_chr_pos(&(a.0.as_ref(), a.1), &(b.0.as_ref(), b.1)));
    }

    let edit_sites = c2u_sites
        .iter()
        .map(|(chr, pos)| (chr.clone(), *pos - 1))
        .collect::<Vec<_>>();

    let edit_site_index = edit_sites
        .iter()
        .enumerate()
        .map(|(i, chr_pos)| (chr_pos.clone(), i))
        .collect::<HashMap<_, _>>();

    if let Some(resolution) = args.resolution {
        info!("reducing bp resolution");
        for (chr, pos) in edit_sites {
            let k = pos.div_ceil(resolution);
        }
    }

    // bed_data.into_iter().map()

    // tood: write out

    Ok(())
}

//////////////////////////////////////////////////////////////////////
// Step 1: Find variable C2U sites searching over all the BAM files //
//////////////////////////////////////////////////////////////////////
fn find_c2u_site(
    args: &CountDartSeqArgs,
    chr: &str,
    lb: &usize,
    ub: &usize,
) -> anyhow::Result<(Vec<(Box<str>, usize)>, Vec<SamSampleName>)> {
    let region = format!("{}:{}-{}", chr, (*lb).max(1), *ub).into_boxed_str();

    // 1. sweep each pair bam files to find variable sites
    let mut wt_freq_map = DnaBaseFreqMap::new();
    let mut mut_freq_map = DnaBaseFreqMap::new();

    for wt_file in args.wt_bam_files.iter() {
        wt_freq_map.update_bam_region(wt_file, &region)?;
    }

    for mut_file in args.mut_bam_files.iter() {
        mut_freq_map.update_bam_region(mut_file, &region)?;
    }

    // 2. find AC/T patterns: Using mutant statistics as null
    // distribution, it will keep possible C->U edit positions.
    let mut_pos_to_freq = mut_freq_map.marginal_frequency_by_position();
    let positions = wt_freq_map.sorted_positions();
    let samples: Vec<SamSampleName> = wt_freq_map.samples().into_iter().cloned().collect();

    let mut c2u_positions = Vec::with_capacity(positions.len());

    for s in samples.iter() {
        let wt_pos_to_freq = wt_freq_map.frequency_per_sample(s)?;

        if positions.len() > 2 {
            // first position
            let mut prev = positions[0];
            let mut prev_n_a = wt_pos_to_freq[&prev].get(Some(Dna::A));

            // following positions: A followed by C or T
            for curr in positions[1..].iter() {
                if let (Some(wt_freq), Some(mut_freq)) =
                    (wt_pos_to_freq.get(curr), mut_pos_to_freq.get(curr))
                {
                    if *curr == (prev + 1) && prev_n_a > 0 {
                        let wt_n_c = wt_freq.get(Some(Dna::C));
                        let wt_n_t = wt_freq.get(Some(Dna::T));
                        let mut_n_c = mut_freq.get(Some(Dna::C));
                        let mut_n_t = mut_freq.get(Some(Dna::T));

                        let ntot = wt_n_c + wt_n_t;
                        let ntot_mut = mut_n_c + mut_n_t;

                        if wt_n_t > args.min_conversion
                            && ntot > args.min_coverage
                            && ntot_mut > args.min_coverage
                        {
                            let pv = BinomTest {
                                expected: (mut_n_c, mut_n_t),
                                observed: (wt_n_c, wt_n_t),
                            }
                            .pvalue_greater()?;
                            if pv < args.pvalue_cutoff {
                                c2u_positions.push(*curr);
                            }
                        }
                    }
                };
                // - keeping track of the previous #A
                // - we will only consider when A is the most frequent
                prev_n_a = wt_pos_to_freq
                    .get(curr)
                    .and_then(|x| {
                        let major = x.most_frequent();
                        (major.0 == Dna::A).then_some(major.1)
                    })
                    .unwrap_or(0);

                prev = *curr;
            }
        }
    }

    let chr_c2u_positions: Vec<(Box<str>, usize)> =
        c2u_positions.into_iter().map(|x| (chr.into(), x)).collect();

    Ok((chr_c2u_positions, samples))
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit possible C2U positions and collect m6A sites, //
// locations and samples.                                        //
///////////////////////////////////////////////////////////////////
fn collect_m6a_variant(
    args: &CountDartSeqArgs,
    chr: &str,
    c2u_pos: &usize,
) -> anyhow::Result<Vec<(Box<str>, usize, SamSampleName, f32, f32)>> {
    let mut wt_stat_map = DnaBaseFreqMap::new();
    let c2upos = *c2u_pos;
    let m6apos = (c2upos - 1).max(1);

    // let row_index = edit_site_index
    //     .get(&(chr.clone(), m6apos))
    //     .ok_or(anyhow::anyhow!("edit site not found"))?;

    let region = format!("{}:{}-{}", chr, m6apos, c2upos).into_boxed_str();

    for wt_file in args.wt_bam_files.iter() {
        wt_stat_map.update_bam_region(wt_file.as_ref(), &region)?;
    }

    let mut ret = vec![];

    if let Ok(a_stat) = wt_stat_map.frequency_at(&m6apos) {
        let mut a_count = DnaBaseCount::new();
        for counts in a_stat.values() {
            a_count += counts;
        }

        let major = a_count.most_frequent();

        if major.0 == Dna::A && major.1 >= args.min_coverage {
            let c2u_stat = wt_stat_map.frequency_at(c2u_pos)?;

            for (s, counts) in c2u_stat {
                let n_c = counts.get(Some(Dna::C)) as f32;
                let n_t = counts.get(Some(Dna::T)) as f32;

                // let col_index = sample_index
                //     .get(s)
                //     .ok_or(anyhow::anyhow!("sample name not found"))?;

                ret.push((chr.into(), m6apos, s.clone(), n_c, n_t));
            }
        }
    }

    Ok(ret)
}
