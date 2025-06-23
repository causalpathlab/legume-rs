use data_beans::sparse_io::create_sparse_from_triplets;

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

fn find_edit_site(
    args: &CountDartSeqArgs,
    chr: &str,
    lb: &usize,
    ub: &usize,
) -> anyhow::Result<(Vec<(Box<str>, usize)>, Vec<SamSampleName>)> {
    let region = format!("{}:{}-{}", chr, (*lb).max(1), *ub).into_boxed_str();

    // 1. sweep each pair bam files to find variable sites
    let mut wt_stat_map = DnaBaseStatMap::new();
    let mut mut_stat_map = DnaBaseStatMap::new();

    for wt_file in args.wt_bam_files.iter() {
        wt_stat_map.update_bam_region(&wt_file, &region)?;
    }

    for mut_file in args.mut_bam_files.iter() {
        mut_stat_map.update_bam_region(&mut_file, &region)?;
    }

    // 2. find AC/T patterns
    let wt_pos_stat = wt_stat_map.combine_statistic_by_position();
    let mut_pos_stat = mut_stat_map.combine_statistic_by_position();
    let positions = wt_stat_map.sorted_positions();
    let mut edit_positions = Vec::with_capacity(positions.len());
    if positions.len() > 2 {
        // first position
        let mut prev = positions[0];
        let mut prev_n_a = wt_pos_stat[&prev].get(Some(Dna::A));

        // following positions: A followed by C or T
        for curr in positions[1..].iter() {
            if let (Some(wt_stat), Some(mut_stat)) = (wt_pos_stat.get(curr), mut_pos_stat.get(curr))
            {
                if *curr == (prev + 1) && prev_n_a > 0 {
                    let wt_n_c = wt_stat.get(Some(Dna::C));
                    let wt_n_t = wt_stat.get(Some(Dna::T));
                    let mut_n_c = mut_stat.get(Some(Dna::C));
                    let mut_n_t = mut_stat.get(Some(Dna::T));

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
                            edit_positions.push(*curr);
                        }
                    }
                }
            };
            // - keeping track of the previous #A
            // - we will only consider when A is the most frequent
            prev_n_a = wt_pos_stat
                .get(curr)
                .and_then(|x| {
                    let major = x.most_frequent();
                    (major.0 == Dna::A).then_some(major.1)
                })
                .unwrap_or(0);

            // prev_n_a = wt_pos_stat
            //     .get(curr)
            //     .map(|x| x.get(Some(Dna::A)))
            //     .unwrap_or(0);

            prev = *curr;
        }
    }

    let samples: Vec<SamSampleName> = wt_stat_map
        .samples()
        .into_iter()
        .map(|x| x.clone())
        .collect();

    let ret: Vec<(Box<str>, usize)> = edit_positions
        .into_iter()
        .map(|x| (chr.into(), x))
        .collect();

    Ok((ret, samples))
}

///
/// Count DART-seq bam files to quantify m6A β values
///
///
pub fn run_count_dartseq(args: &CountDartSeqArgs) -> anyhow::Result<()> {
    if args.wt_bam_files.len() < 1 || args.mut_bam_files.len() < 1 {
        return Err(anyhow::anyhow!("need matching pairs of bam files"));
    }

    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("empty bam files"));
    }

    for x in args.wt_bam_files.iter() {
        check_bam_index(&x, None)?;
    }

    for x in args.mut_bam_files.iter() {
        check_bam_index(&x, None)?;
    }

    let bam_file = args.wt_bam_files[0].as_ref();
    let jobs = create_bam_jobs(bam_file, Some(args.block_size))?;

    // 1. figure out potential edit sites
    let c2u_sites_samples = jobs
        .par_iter()
        .progress_count(jobs.len() as u64)
        .map(|(chr, lb, ub)| -> anyhow::Result<_> { find_edit_site(args, chr.as_ref(), lb, ub) })
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

    // 2. collect all the statistics
    let edit_sites = c2u_sites
        .iter()
        .map(|(chr, pos)| (chr.clone(), *pos - 1))
        .collect::<Vec<_>>();

    let edit_site_index = edit_sites
        .iter()
        .enumerate()
        .map(|(i, chr_pos)| (chr_pos.clone(), i))
        .collect::<HashMap<_, _>>();

    let mut bed_data = c2u_sites
        .par_iter()
        .map(|(chr, c2u_pos)| -> anyhow::Result<Vec<_>> {
            let mut wt_stat_map = DnaBaseStatMap::new();
            let c2upos = *c2u_pos;
            let m6apos = (c2upos - 1).max(1);

            let row_index = edit_site_index
                .get(&(chr.clone(), m6apos))
                .ok_or(anyhow::anyhow!("edit site not found"))?;

            let region = format!("{}:{}-{}", chr.as_ref(), m6apos, c2upos).into_boxed_str();

            for wt_file in args.wt_bam_files.iter() {
                wt_stat_map.update_bam_region(&wt_file, &region)?;
            }

            let mut ret = vec![];

            if let Ok(a_stat) = wt_stat_map.statistic_at(&m6apos) {
                let mut a_count = DnaBaseCount::new();
                for (_, counts) in a_stat {
                    a_count += counts;
                }

                let major = a_count.most_frequent();

                if major.0 == Dna::A && major.1 >= args.min_coverage {
                    let c2u_stat = wt_stat_map.statistic_at(c2u_pos)?;

                    for (s, counts) in c2u_stat {
                        let n_c = counts.get(Some(Dna::C)) as f32;
                        let n_t = counts.get(Some(Dna::T)) as f32;

                        let col_index = sample_index
                            .get(s)
                            .ok_or(anyhow::anyhow!("sample name not found"))?;

                        ret.push((*row_index, *col_index, n_c, n_t));
                    }
                }
            }

            Ok(ret)
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    info!("sorting {} edit statistics...", &bed_data.len());
    bed_data.par_sort_by_key(|x| (x.0, x.1));

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
