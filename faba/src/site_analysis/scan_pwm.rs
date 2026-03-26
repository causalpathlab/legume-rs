use super::site_io::*;
use crate::common::*;
use crate::data::dna::{Dna, DnaBaseCount};
use crate::data::dna_stat_map::DnaBaseFreqMap;
use crate::data::util_htslib;

use genomic_data::bed::Bed;
use genomic_data::sam::Strand;

use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum PwmSource {
    Reads,
    Reference,
}

#[derive(Args, Debug)]
pub struct ScanPwmArgs {
    /// Site-level parquet file (from dartseq or apa output)
    #[arg(short = 's', long = "sites", required = true)]
    site_file: Box<str>,

    /// Input BAM file(s), comma-separated (required for --source reads)
    #[arg(value_delimiter = ',')]
    bam_files: Vec<Box<str>>,

    /// Reference genome FASTA (required for --source reference)
    #[arg(short = 'f', long = "genome")]
    genome_file: Option<Box<str>>,

    /// Base frequency source: reads (from BAM) or reference (from FASTA)
    #[arg(long, value_enum, default_value = "reference")]
    source: PwmSource,

    /// Half-window size: collect +/- this many bp around each site
    #[arg(short = 'w', long, default_value_t = 10)]
    window: i64,

    /// Number of threads
    #[arg(long, default_value_t = 16)]
    threads: usize,

    /// Output file path (TSV, or .gz for gzipped)
    #[arg(short, long, required = true)]
    output: Box<str>,
}

/// Swap A<->T and G<->C counts to complement a DnaBaseCount.
fn complement_base_count(count: &DnaBaseCount) -> DnaBaseCount {
    let mut out = DnaBaseCount::new();
    out.add(Some(&Dna::A), count.count_t());
    out.add(Some(&Dna::T), count.count_a());
    out.add(Some(&Dna::G), count.count_c());
    out.add(Some(&Dna::C), count.count_g());
    out
}

/// Collect PWM from reference FASTA sequence around each site.
fn collect_from_reference(
    sites: &[GenomicSite],
    genome_file: &str,
    window: i64,
) -> anyhow::Result<Vec<DnaBaseCount>> {
    let width = (2 * window + 1) as usize;
    let mut pwm: Vec<DnaBaseCount> = (0..width).map(|_| DnaBaseCount::new()).collect();

    let faidx = util_htslib::load_fasta_index(genome_file)?;

    for site in sites {
        let start = site.position - window;
        let end = site.position + window; // inclusive
        let seq = util_htslib::fetch_reference_seq(&faidx, &site.chr, start, end)?;

        if let Some(bases) = seq {
            if bases.len() != width {
                continue; // skip sites near chromosome boundaries
            }
            match site.strand {
                Strand::Forward => {
                    for (j, base) in bases.iter().enumerate() {
                        pwm[j].add(Some(base), 1);
                    }
                }
                Strand::Backward => {
                    // Reverse complement: reverse order and complement each base
                    for (j, base) in bases.iter().rev().enumerate() {
                        let comp = match base {
                            Dna::A => Dna::T,
                            Dna::T => Dna::A,
                            Dna::G => Dna::C,
                            Dna::C => Dna::G,
                        };
                        pwm[j].add(Some(&comp), 1);
                    }
                }
            }
        }
    }

    Ok(pwm)
}

/// Collect PWM from BAM read base frequencies around each site.
fn collect_from_reads(
    sites: &[GenomicSite],
    bam_files: &[Box<str>],
    window: i64,
) -> anyhow::Result<Vec<DnaBaseCount>> {
    let width = (2 * window + 1) as usize;

    // Ensure BAM indexes exist
    for bam_file in bam_files {
        util_htslib::check_bam_index(bam_file, None)?;
    }

    let pwm: Vec<DnaBaseCount> = sites
        .par_iter()
        .progress_count(sites.len() as u64)
        .map(|site| {
            let mut local_pwm: Vec<DnaBaseCount> =
                (0..width).map(|_| DnaBaseCount::new()).collect();

            let bed = Bed {
                chr: site.chr.clone(),
                start: (site.position - window).max(0),
                stop: site.position + window + 1,
            };

            let mut freq_map = DnaBaseFreqMap::new();
            for bam_file in bam_files {
                freq_map
                    .update_from_region(bam_file, &bed)
                    .expect("failed to read BAM file");
            }

            if let Some(counts) = freq_map.marginal_frequency_map() {
                for (j, p) in ((site.position - window)..=(site.position + window)).enumerate() {
                    if let Some(count) = counts.get(&p) {
                        match site.strand {
                            Strand::Forward => {
                                local_pwm[j] += count;
                            }
                            Strand::Backward => {
                                // Reverse offset and complement
                                let rev_j = width - 1 - j;
                                let comp = complement_base_count(count);
                                local_pwm[rev_j] += &comp;
                            }
                        }
                    }
                }
            }

            local_pwm
        })
        .reduce(
            || (0..width).map(|_| DnaBaseCount::new()).collect(),
            |mut acc, local| {
                for (a, b) in acc.iter_mut().zip(local.iter()) {
                    *a += b;
                }
                acc
            },
        );

    Ok(pwm)
}

/// Write PWM table as TSV.
fn write_pwm(pwm: &[DnaBaseCount], window: i64, output: &str) -> anyhow::Result<()> {
    let file = File::create(output)?;
    let mut writer: Box<dyn Write> = if output.ends_with(".gz") {
        Box::new(BufWriter::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::default(),
        )))
    } else {
        Box::new(BufWriter::new(file))
    };

    writeln!(
        writer,
        "position\tcount_A\tcount_T\tcount_G\tcount_C\ttotal\tfreq_A\tfreq_T\tfreq_G\tfreq_C"
    )?;

    for (j, count) in pwm.iter().enumerate() {
        let rel_pos = j as i64 - window;
        let a = count.count_a();
        let t = count.count_t();
        let g = count.count_g();
        let c = count.count_c();
        let total = a + t + g + c;
        let tot_f = total as f32;
        if total > 0 {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                rel_pos,
                a,
                t,
                g,
                c,
                total,
                a as f32 / tot_f,
                t as f32 / tot_f,
                g as f32 / tot_f,
                c as f32 / tot_f,
            )?;
        } else {
            writeln!(writer, "{}\t0\t0\t0\t0\t0\t0\t0\t0\t0", rel_pos)?;
        }
    }

    Ok(())
}

pub fn run_scan_pwm(args: &ScanPwmArgs) -> anyhow::Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok(); // ignore if already set

    let sites = read_sites(&args.site_file)?;

    let pwm = match args.source {
        PwmSource::Reference => {
            let genome = args
                .genome_file
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--genome is required when --source reference"))?;
            collect_from_reference(&sites, genome, args.window)?
        }
        PwmSource::Reads => {
            if args.bam_files.is_empty() {
                return Err(anyhow::anyhow!("BAM file(s) required when --source reads"));
            }
            collect_from_reads(&sites, &args.bam_files, args.window)?
        }
    };

    write_pwm(&pwm, args.window, &args.output)?;
    info!("wrote PWM to {}", args.output);

    Ok(())
}
