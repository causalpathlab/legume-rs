use crate::common::*;
use crate::data::alignment::SamSampleName;
use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat::*;
use crate::data::util::*;
use crate::hypothesis_tests::BinomTest;

use coitrees::Interval;
use coitrees::IntervalTree;
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
    // we add +/- 2bp to capture RAC patterns in the left and right boundaries
    let jobs = create_bam_jobs(bam_file, Some(args.block_size), Some(2))?;

    ////////////////////////////////////////
    // 1. figure out potential edit sites //
    ////////////////////////////////////////

    let c2u_sites_samples = jobs
        .par_iter()
        .progress_count(jobs.len() as u64)
        .map(|(chr, lb, ub)| -> anyhow::Result<_> { find_c2u_site(args, chr.as_ref(), lb, ub) })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut c2u_stratified_by_chr: HashMap<Box<str>, Vec<ChrM6aC2u>> = HashMap::new();
    let mut samples = HashSet::new();

    for (c2u, sam) in c2u_sites_samples {
        for x in c2u {
            let sites = c2u_stratified_by_chr.entry(x.chr.clone()).or_default();
            sites.push(x);
        }
        samples.extend(sam.into_iter());
    }

    ///////////////////////////////////
    // 2. collect all the statistics //
    ///////////////////////////////////

    let mut sample_chr_to_coitree: HashMap<(SamSampleName, Box<str>), coitrees::COITree<_, usize>> =
        HashMap::new();

    for (_, c2u_vec) in c2u_stratified_by_chr.iter() {
        for x in c2u_vec {
            println!("{:?}", x);
        }
    }

    for (chr, c2u_vec) in c2u_stratified_by_chr {
        let sample_intervals = c2u_vec
            .iter()
            .map(|x| collect_m6a_stat(args, x))
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let mut sample_to_intervals: HashMap<SamSampleName, Vec<Interval<_>>> = HashMap::new();

        for (s, interv) in sample_intervals {
            sample_to_intervals.entry(s).or_default().push(interv);
        }

        for (s, interv) in sample_to_intervals {
            let tree: coitrees::COITree<_, usize> = coitrees::COITree::new(&interv);
            let k = (s, chr.clone());
            sample_chr_to_coitree.entry(k).or_insert(tree);
        }
    }

    /////////////////////////////////////
    // 3. Aggregate them into triplets //
    /////////////////////////////////////

    for ((s, chr), tree) in sample_chr_to_coitree.iter() {
        for x in tree.into_iter() {
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                chr, x.first, x.last, s, x.metadata.methylated, x.metadata.unmethylated
            );
        }
    }

    Ok(())
}

#[derive(PartialEq, Eq, PartialOrd, Ord, std::fmt::Debug)]
struct ChrM6aC2u {
    chr: Box<str>,
    m6a_pos: usize,
    conversion_pos: usize,
    direction: Direction,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, std::fmt::Debug)]
enum Direction {
    Forward,
    Backward,
}

fn find_c2u_site(
    args: &CountDartSeqArgs,
    chr: &str,
    lb: &usize,
    ub: &usize,
) -> anyhow::Result<(Vec<ChrM6aC2u>, Vec<SamSampleName>)> {
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

    let positions = wt_freq_map.sorted_positions();

    if positions.len() < 3 {
        return Ok((vec![], vec![]));
    }

    // 2. find AC/T patterns: Using mutant statistics as null
    // distribution, it will keep possible C->U edit positions.
    // Find forward: "A/G" "A" "C", reverse: "C" "A" "A/G"
    let mut chr_m6a_c2u_positions: Vec<ChrM6aC2u> = Vec::with_capacity(positions.len());

    let binomial_test = |wt_freq: Option<&DnaBaseCount>,
                         mut_freq: Option<&DnaBaseCount>,
                         failure: Dna,
                         success: Dna|
     -> bool {
        match (wt_freq, mut_freq) {
            (Some(wt_freq), Some(mut_freq)) => {
                let (wt_n_failure, wt_n_success) = (
                    wt_freq.get(Some(failure.clone())),
                    wt_freq.get(Some(success.clone())),
                );
                let (mut_n_failure, mut_n_success) = (
                    mut_freq.get(Some(failure.clone())),
                    mut_freq.get(Some(success.clone())),
                );
                let (ntot, ntot_mut) = (wt_n_failure + wt_n_success, mut_n_failure + mut_n_success);

                if wt_n_success > args.min_conversion
                    && ntot > args.min_coverage
                    && ntot_mut > args.min_coverage
                {
                    let pv_greater = BinomTest {
                        expected: (mut_n_failure, mut_n_success),
                        observed: (wt_n_failure, wt_n_success),
                    }
                    .pvalue_greater()
                    .unwrap_or(1.0);
                    return pv_greater < args.pvalue_cutoff;
                } else {
                    false
                }
            }
            _ => false,
        }
    };

    let c_to_u_binomial_test = |wt_freq: Option<&DnaBaseCount>,
                                mut_freq: Option<&DnaBaseCount>|
     -> bool { binomial_test(wt_freq, mut_freq, Dna::C, Dna::T) };

    let complement_binomial_test =
        |wt_freq: Option<&DnaBaseCount>, mut_freq: Option<&DnaBaseCount>| -> bool {
            binomial_test(wt_freq, mut_freq, Dna::G, Dna::A)
        };

    // search forward RAC patterns
    let forward_sweep_wt_pos_freq = |wt_pos_to_freq: &HashMap<usize, DnaBaseCount>,
                                     mut_pos_to_freq: &HashMap<usize, DnaBaseCount>|
     -> Vec<ChrM6aC2u> {
        let mut ret = vec![];
        for j in 2..positions.len() {
            let first = positions[j - 2];
            let second = positions[j - 1];
            let third = positions[j];

            if third - first != 2 {
                continue;
            }

            // !s.is_mono_allelic() || s.most_frequent().0 != Dna::A

            if wt_pos_to_freq
                .get(&second)
                .map(|s| s.most_frequent().0 != Dna::A)
                .unwrap_or(true)
            {
                continue;
            }

            // R = A/G
            let first_biallele = wt_pos_to_freq.get(&first).map(|r| r.bi_allelic_stat());

            if first_biallele
                .map(|ba| {
                    (ba.a1 == Dna::A && ba.a2 == Dna::G) || (ba.a1 == Dna::G && ba.a2 == Dna::A)
                })
                .unwrap_or(false)
            {
                if c_to_u_binomial_test(wt_pos_to_freq.get(&third), mut_pos_to_freq.get(&third)) {
                    ret.push(ChrM6aC2u {
                        chr: chr.into(),
                        m6a_pos: second,
                        conversion_pos: third,
                        direction: Direction::Forward,
                    });
                }
            }
        }
        ret
    };

    // search reverse CAR patterns
    let reverse_sweep_wt_pos_freq = |wt_pos_to_freq: &HashMap<usize, DnaBaseCount>,
                                     mut_pos_to_freq: &HashMap<usize, DnaBaseCount>|
     -> Vec<ChrM6aC2u> {
        let mut ret = vec![];
        for j in 0..(positions.len() - 2) {
            let first = positions[j];
            let second = positions[j + 1];
            let third = positions[j + 2];

            if third - first != 2 {
                continue;
            }

            if wt_pos_to_freq
                .get(&second)
                // .map(|s| s.most_frequent().0 != Dna::A)
                // complement seq
                .map(|s| s.most_frequent().0 != Dna::T)
                .unwrap_or(true)
            {
                continue;
            }

            // R = A/G
            let third_biallele = wt_pos_to_freq.get(&third).map(|r| r.bi_allelic_stat());
            if third_biallele
                .map(|ba| {
                    // (ba.a1 == Dna::A && ba.a2 == Dna::G) || (ba.a1 == Dna::G && ba.a2 == Dna::A)
                    // complement sequence
                    (ba.a1 == Dna::T && ba.a2 == Dna::C) || (ba.a1 == Dna::C && ba.a2 == Dna::T)
                })
                .unwrap_or(false)
            {
                if complement_binomial_test(wt_pos_to_freq.get(&first), mut_pos_to_freq.get(&first))
                {
                    ret.push(ChrM6aC2u {
                        chr: chr.into(),
                        m6a_pos: second,
                        conversion_pos: first,
                        direction: Direction::Backward,
                    });
                }
            }
        }
        ret
    };

    let samples: Vec<SamSampleName> = wt_freq_map.samples().into_iter().cloned().collect();

    for s in samples.iter() {
        if let (Some(_wt), Some(_mut)) = (
            wt_freq_map.forward_frequency_per_sample(s),
            mut_freq_map.forward_frequency_per_sample(s),
        ) {
            chr_m6a_c2u_positions.extend(forward_sweep_wt_pos_freq(_wt, _mut));
        }

        if let (Some(_wt), Some(_mut)) = (
            wt_freq_map.reverse_frequency_per_sample(s),
            mut_freq_map.reverse_frequency_per_sample(s),
        ) {
            chr_m6a_c2u_positions.extend(reverse_sweep_wt_pos_freq(_wt, _mut));
        }
    }

    let wt_pos_to_freq = wt_freq_map.forward_marginal_frequency_by_position();
    let mut_pos_to_freq = mut_freq_map.forward_marginal_frequency_by_position();
    chr_m6a_c2u_positions.extend(forward_sweep_wt_pos_freq(&wt_pos_to_freq, &mut_pos_to_freq));

    let wt_pos_to_freq = wt_freq_map.reverse_marginal_frequency_by_position();
    let mut_pos_to_freq = mut_freq_map.reverse_marginal_frequency_by_position();
    chr_m6a_c2u_positions.extend(reverse_sweep_wt_pos_freq(&wt_pos_to_freq, &mut_pos_to_freq));

    chr_m6a_c2u_positions.sort();
    chr_m6a_c2u_positions.dedup();

    Ok((chr_m6a_c2u_positions, samples))
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit possible C2U positions and collect m6A sites, //
// locations and samples.                                        //
///////////////////////////////////////////////////////////////////
#[derive(Clone, Copy, Default)]
struct M6aData {
    methylated: usize,
    unmethylated: usize,
}

fn collect_m6a_stat(
    args: &CountDartSeqArgs,
    chr_m6a_c2u: &ChrM6aC2u,
) -> anyhow::Result<Vec<(SamSampleName, Interval<M6aData>)>> {
    let mut stat_map = DnaBaseFreqMap::new();
    let m6apos = chr_m6a_c2u.m6a_pos;
    let c2upos = chr_m6a_c2u.conversion_pos;
    let chr = chr_m6a_c2u.chr.as_ref();

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);
    let region = format!("{}:{}-{}", chr, lb, ub).into_boxed_str();

    for _file in args.wt_bam_files.iter() {
        stat_map.update_bam_region(_file.as_ref(), &region)?;
    }

    let dir = &chr_m6a_c2u.direction;

    let c2u_stat = match dir {
        Direction::Forward => stat_map.forward_frequency_at(&c2upos),
        Direction::Backward => stat_map.reverse_frequency_at(&c2upos),
    };

    let unmethylated_base = match dir {
        Direction::Forward => Dna::C,
        Direction::Backward => Dna::G,
    };

    let methylated_base = match dir {
        Direction::Forward => Dna::T,
        Direction::Backward => Dna::A,
    };

    // let before = match dir {
    //     Direction::Forward => stat_map.forward_frequency_at(&(m6apos - 1)),
    //     Direction::Backward => stat_map.reverse_frequency_at(&(m6apos + 1)),
    // }
    // .unwrap();

    // let after = match dir {
    //     Direction::Forward => stat_map.forward_frequency_at(&(m6apos + 1)),
    //     Direction::Backward => stat_map.reverse_frequency_at(&(m6apos - 1)),
    // }
    // .unwrap();

    // for (_, x) in before {
    //     println!("before: {:?}", x);
    // }

    // for (_, x) in after {
    //     println!("after: {:?}", x);
    // }

    // todo: we can reduce resolution
    // args.resolution;

    let mut ret = vec![];

    if let Some(c2u_stat) = c2u_stat {
        for (s, counts) in c2u_stat {
            let meth_data = M6aData {
                methylated: counts.get(Some(methylated_base.clone())),
                unmethylated: counts.get(Some(unmethylated_base.clone())),
            };

            ret.push((
                s.clone(),
                Interval {
                    first: lb as i32,
                    last: ub as i32,
                    metadata: meth_data,
                },
            ));
        }
    }

    Ok(ret)
}
