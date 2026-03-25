use super::site_io::*;
use clap::Args;
use dashmap::DashMap as HashMap;
use genomic_data::gff::*;
use genomic_data::sam::Strand;
use log::info;
use std::io::Write;

#[derive(Args, Debug)]
pub struct MetageneArgs {
    /// Site-level parquet file (from dart or apa output)
    #[arg(short = 's', long = "sites", required = true)]
    site_file: Box<str>,

    /// GFF annotation file
    #[arg(short = 'g', long = "gff", required = true)]
    gff_file: Box<str>,

    /// Total number of bins across the metagene
    #[arg(short = 'n', long = "bins", default_value_t = 57)]
    num_bins: usize,

    /// Output TSV file path
    #[arg(short, long, required = true)]
    output: Box<str>,

    /// Print ASCII histogram to stderr
    #[arg(long = "print")]
    print_histogram: bool,

    /// Maximum width of ASCII histogram
    #[arg(long = "max-width", default_value_t = 60)]
    max_width: usize,
}

/// Per-chromosome sorted interval index for mapping positions to gene features.
struct FeatureIndex {
    /// (start, stop, strand) sorted by start within each chromosome
    intervals: rustc_hash::FxHashMap<Box<str>, Vec<(i64, i64, Strand)>>,
}

impl FeatureIndex {
    fn from_feature_map(map: &HashMap<GeneId, GffRecord>) -> Self {
        let mut by_chr: rustc_hash::FxHashMap<Box<str>, Vec<(i64, i64, Strand)>> =
            rustc_hash::FxHashMap::default();
        for entry in map.iter() {
            let rec = entry.value();
            by_chr
                .entry(rec.seqname.clone())
                .or_default()
                .push((rec.start, rec.stop, rec.strand));
        }
        for intervals in by_chr.values_mut() {
            intervals.sort_by_key(|&(s, _, _)| s);
        }
        FeatureIndex { intervals: by_chr }
    }

    /// Find the interval containing `position` (1-based GFF coords).
    /// Returns (start, stop, strand, length) if found.
    fn find(&self, chr: &str, position: i64) -> Option<(i64, i64, Strand, usize)> {
        let intervals = self.intervals.get(chr)?;
        // Binary search: find rightmost interval with start <= position
        let idx = intervals.partition_point(|&(s, _, _)| s <= position);
        if idx == 0 {
            return None;
        }
        // Scan backwards from the candidate
        for &(start, stop, strand) in intervals[..idx].iter().rev() {
            if start > position {
                continue;
            }
            if position <= stop {
                let length = (stop - start + 1).max(1) as usize;
                return Some((start, stop, strand, length));
            }
            // Past this point, earlier intervals have smaller start, won't contain position
            if start < position {
                break;
            }
        }
        None
    }
}

pub struct GeneFeatureHistogram {
    five_prime: Vec<usize>,
    cds: Vec<usize>,
    three_prime: Vec<usize>,
    non_coding: Vec<usize>,
}

impl GeneFeatureHistogram {
    pub fn print(&self, max_width: usize) {
        fn print_row(label: &str, data: &[usize], scale: usize, max_width: usize) {
            for &n in data {
                let n1 = n.div_ceil(scale);
                let n0 = max_width.saturating_sub(n1);
                eprintln!("{:<6}{}{} {}", label, "*".repeat(n1), " ".repeat(n0), n);
            }
        }

        let nmax = self
            .five_prime
            .iter()
            .chain(&self.cds)
            .chain(&self.three_prime)
            .chain(&self.non_coding)
            .cloned()
            .max()
            .unwrap_or(0);

        if nmax == 0 {
            eprintln!("(no sites mapped to gene features)");
            return;
        }

        let scale = nmax.div_ceil(max_width);

        if !self.five_prime.is_empty() {
            print_row("5'UTR", &self.five_prime, scale, max_width);
        }
        if !self.cds.is_empty() {
            print_row("CDS", &self.cds, scale, max_width);
        }
        if !self.three_prime.is_empty() {
            print_row("3'UTR", &self.three_prime, scale, max_width);
        }
        if !self.non_coding.is_empty() {
            print_row("ncRNA", &self.non_coding, scale, max_width);
        }
    }

    pub fn to_tsv(&self, file_path: &str) -> anyhow::Result<()> {
        let mut writer = matrix_util::common_io::open_buf_writer(file_path)?;
        writeln!(writer, "#feature\tgenomic_bin\tcount")?;

        for (label, data) in [
            ("5UTR", &self.five_prime),
            ("CDS", &self.cds),
            ("3UTR", &self.three_prime),
            ("ncRNA", &self.non_coding),
        ] {
            for (i, &n) in data.iter().enumerate() {
                writeln!(writer, "{}\t{}\t{}", label, i, n)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

fn count_metagene(
    sites: &[GenomicSite],
    gff_file: &str,
    n_genomic_bins: usize,
) -> anyhow::Result<GeneFeatureHistogram> {
    let gff_records = read_gff_record_vec(gff_file)?;

    let protein_coding_records: Vec<GffRecord> = gff_records
        .iter()
        .filter(|rec| rec.gene_type == GeneType::CodingGene)
        .cloned()
        .collect();

    let non_coding_records: Vec<GffRecord> = gff_records
        .iter()
        .filter(|rec| rec.gene_type != GeneType::CodingGene)
        .cloned()
        .collect();

    let UnionGeneModel {
        gene_boundaries: _,
        cds,
        five_prime_utr,
        three_prime_utr,
    } = build_union_gene_model(&protein_coding_records)?;

    let UnionGeneModel {
        gene_boundaries: nc_gene_boundaries,
        ..
    } = build_union_gene_model(&non_coding_records)?;

    // Proportional bin allocation by max feature length
    let n_five_prime = five_prime_utr.take_max_length().max(10);
    let n_cds = cds.take_max_length();
    let n_three_prime = three_prime_utr.take_max_length().max(20);
    let ntot = n_five_prime + n_cds + n_three_prime;

    let nbins_five_prime = n_five_prime as usize * n_genomic_bins / ntot as usize;
    let nbins_cds = n_cds as usize * n_genomic_bins / ntot as usize;
    let nbins_three_prime = n_three_prime as usize * n_genomic_bins / ntot as usize;

    // Build feature indices
    let five_prime_idx = FeatureIndex::from_feature_map(&five_prime_utr);
    let cds_idx = FeatureIndex::from_feature_map(&cds);
    let three_prime_idx = FeatureIndex::from_feature_map(&three_prime_utr);
    let nc_idx = FeatureIndex::from_feature_map(&nc_gene_boundaries);

    let mut five_prime_hist = vec![0usize; nbins_five_prime];
    let mut cds_hist = vec![0usize; nbins_cds];
    let mut three_prime_hist = vec![0usize; nbins_three_prime];
    let mut non_coding_hist = vec![0usize; n_genomic_bins];

    for site in sites {
        let chr = site.chr.as_ref();
        // Sites use 0-based positions; GFF uses 1-based
        let gff_pos = site.position + 1;

        if let Some((start, _stop, strand, length)) = five_prime_idx.find(chr, gff_pos) {
            if nbins_five_prime > 0 {
                let rel = strand_relative_pos(gff_pos, start, strand, length);
                let bin = rel * nbins_five_prime / length;
                five_prime_hist[bin.min(nbins_five_prime - 1)] += 1;
            }
        } else if let Some((start, _stop, strand, length)) = cds_idx.find(chr, gff_pos) {
            if nbins_cds > 0 {
                let rel = strand_relative_pos(gff_pos, start, strand, length);
                let bin = rel * nbins_cds / length;
                cds_hist[bin.min(nbins_cds - 1)] += 1;
            }
        } else if let Some((start, _stop, strand, length)) = three_prime_idx.find(chr, gff_pos) {
            if nbins_three_prime > 0 {
                let rel = strand_relative_pos(gff_pos, start, strand, length);
                let bin = rel * nbins_three_prime / length;
                three_prime_hist[bin.min(nbins_three_prime - 1)] += 1;
            }
        } else if let Some((start, _stop, strand, length)) = nc_idx.find(chr, gff_pos) {
            let rel = strand_relative_pos(gff_pos, start, strand, length);
            let bin = rel * n_genomic_bins / length;
            non_coding_hist[bin.min(n_genomic_bins - 1)] += 1;
        }
    }

    Ok(GeneFeatureHistogram {
        five_prime: five_prime_hist,
        cds: cds_hist,
        three_prime: three_prime_hist,
        non_coding: non_coding_hist,
    })
}

/// Strand-aware relative position (0-based offset from feature start).
#[inline]
fn strand_relative_pos(pos: i64, start: i64, strand: Strand, length: usize) -> usize {
    match strand {
        Strand::Forward => (pos - start) as usize,
        Strand::Backward => (length - 1).saturating_sub((pos - start) as usize),
    }
}

pub fn run_metagene(args: &MetageneArgs) -> anyhow::Result<()> {
    let sites = read_sites(&args.site_file)?;

    let histogram = count_metagene(&sites, &args.gff_file, args.num_bins)?;

    histogram.to_tsv(&args.output)?;
    info!("wrote metagene histogram to {}", args.output);

    if args.print_histogram {
        histogram.print(args.max_width);
    }

    Ok(())
}
