use crate::common::*;
use crate::data::alignment::*;
use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_htslib::*;
use crate::data::methylation::*;
use crate::data::util_htslib::*;
use crate::hypothesis_tests::BinomTest;

use clap::ValueEnum;
use data_beans::sparse_io::*;
use matrix_util::common_io::*;
use matrix_util::mtx_io::*;
use std::collections::{HashMap, HashSet};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum FeatureType {
    Beta,
    Methylated,
    Unmethylated,
}

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

    /// save .mtx file along with row and column names
    #[arg(long, default_value_t = false)]
    save_mtx: bool,

    /// feature type
    #[arg(long, value_enum, default_value = "beta")]
    feature_type: FeatureType,

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

    let bam_file = args.wt_bam_files[0].as_ref();
    // we add +/- 2bp to capture RAC patterns in the left and right boundaries
    let jobs = create_bam_jobs(bam_file, Some(args.block_size), Some(2))?;

    ////////////////////////////////////////
    // 1. figure out potential edit sites //
    ////////////////////////////////////////

    let njobs = jobs.len();
    info!("Searching possible edit sites over {} blocks", njobs);
    let sites: Vec<_> = jobs
        .into_iter()
        .par_bridge()
        .progress_count(njobs as u64)
        .map(|(chr, lb, ub)| find_methylated_site(args, chr.as_ref(), lb, ub))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

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

    info!("β value triplets...");
    let mut triplets: HashMap<(SamSampleName, MethylationKey), MethylationData> = HashMap::new();

    for (s, k, dat) in stats {
        let accum = triplets.entry((s, k)).or_default();
        *accum += dat;
    }

    let (mut triplets, row_names, col_names) =
        format_data_triplets(triplets.into_iter().collect(), args);

    if args.save_mtx {
        let ext = extension(&args.out_file)?;
        let mtx_file = args.out_file.replace(ext.as_ref(), "mtx.gz");
        let row_file = args.out_file.replace(ext.as_ref(), "rows.gz");
        let col_file = args.out_file.replace(ext.as_ref(), "columns.gz");
        let nrow = row_names.len();
        let ncol = col_names.len();
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);
        write_mtx_triplets(&triplets, nrow, ncol, &mtx_file)?;
        write_lines(&row_names, &row_file)?;
        write_lines(&col_names, &col_file)?;
    }

    /////////////////////////////
    // 4. construct data beans //
    /////////////////////////////
    let backend = match extension(&args.out_file)?.as_ref() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    let mtx_shape = (row_names.len(), col_names.len(), triplets.len());

    let mut data =
        create_sparse_from_triplets(triplets, mtx_shape, Some(&args.out_file), Some(&backend))?;
    data.register_column_names_vec(&col_names);
    data.register_row_names_vec(&row_names);

    info!("done");
    Ok(())
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
struct MethylatedSite {
    chr: Box<str>,
    m6a_pos: i64,
    conversion_pos: i64,
    direction: Direction,
}

fn find_methylated_site(
    args: &CountDartSeqArgs,
    chr: &str,
    lb: i64,
    ub: i64,
) -> anyhow::Result<Vec<MethylatedSite>> {
    // 1. sweep each pair bam files to find variable sites
    let mut wt_freq_map = DnaBaseFreqMap::new();
    let mut mut_freq_map = DnaBaseFreqMap::new();

    for wt_file in args.wt_bam_files.iter() {
        wt_freq_map.update_bam_region(wt_file, (chr, lb, ub))?;
    }

    for mut_file in args.mut_bam_files.iter() {
        mut_freq_map.update_bam_region(mut_file, (chr, lb, ub))?;
    }

    let positions = wt_freq_map.sorted_positions();

    // 2. find AC/T patterns: Using mutant statistics as null
    // distribution, it will keep possible C->U edit positions.
    // Find forward: "A/G" "A" "C", reverse: "C" "A" "A/G"
    let mut chr_m6a_c2u_positions: Vec<MethylatedSite> = Vec::with_capacity(positions.len());

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

                if ntot >= args.min_coverage
                    && ntot_mut >= args.min_coverage
                    && wt_n_success >= args.min_conversion
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
    let forward_sweep_wt_pos_freq = |wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
                                     mut_pos_to_freq: &HashMap<i64, DnaBaseCount>|
     -> Vec<MethylatedSite> {
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
            let is_first_r = wt_pos_to_freq
                .get(&first)
                .map(|r| {
                    let major = &r.most_frequent().0;
                    major == &Dna::A || major == &Dna::G
                })
                .unwrap_or(false);

            if is_first_r {
                if c_to_u_binomial_test(wt_pos_to_freq.get(&third), mut_pos_to_freq.get(&third)) {
                    ret.push(MethylatedSite {
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
    let reverse_sweep_wt_pos_freq = |wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
                                     mut_pos_to_freq: &HashMap<i64, DnaBaseCount>|
     -> Vec<MethylatedSite> {
        let mut ret = vec![];
        for j in 0..(positions.len() - 2) {
            let first = positions[j];
            let second = positions[j + 1];
            let third = positions[j + 2];

            if third - first != 2 {
                continue;
            }

            // A <=> T (complement)
            if wt_pos_to_freq
                .get(&second)
                .map(|s| s.most_frequent().0 != Dna::T)
                .unwrap_or(true)
            {
                continue;
            }

            // R = A/G <=> T/C (complement)
            let is_first_r = wt_pos_to_freq
                .get(&first)
                .map(|r| {
                    let major = &r.most_frequent().0;
                    major == &Dna::T || major == &Dna::C
                })
                .unwrap_or(false);

            if is_first_r {
                if complement_binomial_test(wt_pos_to_freq.get(&first), mut_pos_to_freq.get(&first))
                {
                    ret.push(MethylatedSite {
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

    if positions.len() < 3 {
        return Ok(vec![]);
    }

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

    Ok(chr_m6a_c2u_positions)
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit possible C2U positions and collect m6A sites, //
// locations and samples.                                        //
///////////////////////////////////////////////////////////////////

fn collect_m6a_stat(
    args: &CountDartSeqArgs,
    chr_m6a_c2u: &MethylatedSite,
) -> anyhow::Result<Vec<(SamSampleName, MethylationKey, MethylationData)>> {
    let mut stat_map = DnaBaseFreqMap::new();
    let m6apos = chr_m6a_c2u.m6a_pos;
    let c2upos = chr_m6a_c2u.conversion_pos;
    let chr = chr_m6a_c2u.chr.as_ref();

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    for _file in args.wt_bam_files.iter() {
        stat_map.update_bam_region(_file.as_ref(), (chr, lb, ub))?;
    }

    let dir = &chr_m6a_c2u.direction;

    let methylation_stat = match dir {
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

    let (lb, ub) = if let Some(r) = args.resolution {
        (
            (m6apos as usize) / r * r + 1,
            (m6apos as usize).div_ceil(r) * r,
        )
    } else {
        (m6apos as usize, m6apos as usize + 1)
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
                    dir: dir.clone(),
                },
                MethylationData {
                    methylated: counts.get(Some(methylated_base.clone())),
                    unmethylated: counts.get(Some(unmethylated_base.clone())),
                },
            ));
        }
    }

    Ok(ret)
}

fn format_data_triplets(
    triplets: Vec<((SamSampleName, MethylationKey), MethylationData)>,
    args: &CountDartSeqArgs,
) -> (Vec<(u64, u64, f32)>, Vec<Box<str>>, Vec<Box<str>>) {
    // identify unique samples and sites
    let mut unique_samples = HashSet::new();
    let mut unique_sites = HashSet::new();

    for ((s, k), _) in &triplets {
        unique_samples.insert(s.clone());
        unique_sites.insert(k.clone());
    }

    // assign indices to samples and sites
    let mut unique_samples = unique_samples.into_iter().collect::<Vec<_>>();
    unique_samples.sort();

    let sample_indices: HashMap<SamSampleName, usize> = unique_samples
        .into_iter()
        .enumerate()
        .map(|(i, sample)| (sample, i))
        .collect();

    let mut unique_sites = unique_sites.into_iter().collect::<Vec<_>>();
    unique_sites.par_sort();

    let site_indices: HashMap<MethylationKey, usize> = unique_sites
        .into_iter()
        .enumerate()
        .map(|(i, site)| (site, i))
        .collect();

    // relabel triplets with indices
    let mut relabeled_triplets = Vec::with_capacity(triplets.len());
    for ((s, k), dat) in triplets {
        let row_idx = site_indices[&k] as u64;
        let col_idx = sample_indices[&s] as u64;

        let value = match args.feature_type {
            FeatureType::Beta => {
                let tot = (dat.methylated + dat.unmethylated) as f32;
                let beta = (dat.methylated as f32) / tot.max(1.);
                beta
            }
            FeatureType::Methylated => dat.methylated as f32,

            FeatureType::Unmethylated => dat.unmethylated as f32,
        };

        relabeled_triplets.push((row_idx, col_idx, value));
    }

    let mut samples = vec!["".into(); sample_indices.len()];
    for (k, j) in sample_indices {
        samples[j] = k.to_string().into_boxed_str();
    }

    let mut sites = vec!["".into(); site_indices.len()];
    for (k, j) in site_indices {
        sites[j] = k.to_string().into_boxed_str();
    }

    (relabeled_triplets, sites, samples)
}
