use super::miami::bin::BinEdges;
use super::miami::depth::accumulate_block;
use super::site_io::*;
use crate::data::bam_io::{self, BamReaderCache};
use clap::Args;
use dashmap::DashMap as HashMap;
use genomic_data::bed::Bed;
use genomic_data::gff::*;
use genomic_data::sam::Strand;
use log::info;
use rayon::prelude::*;
use rust_htslib::bam::ext::BamRecordExtensions;
use std::io::Write;

#[derive(Args, Debug)]
pub struct MetageneArgs {
    #[arg(
        short = 's',
        long = "sites",
        required = true,
        help = "Site-level parquet file (from dartseq or apa output)"
    )]
    site_file: Box<str>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "GFF annotation file"
    )]
    gff_file: Box<str>,

    #[arg(
        short = 'n',
        long = "bins",
        default_value_t = 57,
        help = "Total bins across the metagene (default: 19 per region x 3 regions)",
        long_help = "Total number of bins across the metagene profile.\n\
                     Bins are split equally among the three gene regions:\n\
                     5'UTR, CDS, and 3'UTR. Default 57 = 19 bins per region.\n\
                     Choose a multiple of 3 for equal region widths."
    )]
    num_bins: usize,

    #[arg(short, long, required = true, help = "Output TSV file path")]
    output: Box<str>,

    #[arg(
        long = "bam",
        value_delimiter = ',',
        help = "BAM file(s) for a read-depth coverage track (comma-separated)",
        long_help = "Optional BAM file(s). When given, faba also bins per-base read depth\n\
                     over the same 5'UTR/CDS/3'UTR grid and emits `coverage` and\n\
                     `count_per_covered_mb` (sites per Mb of coverage) columns.\n\
                     For 10x 3' libraries the coverage track spikes at the transcript\n\
                     terminus — the read pileup that a raw site-count metagene renders\n\
                     as a spurious 3' 'peak'; the ratio track divides it out."
    )]
    bam_files: Vec<Box<str>>,

    #[arg(
        long = "cell-barcode-tag",
        default_value = "CB",
        help = "Cell barcode tag (read-depth track; reads are pooled across cells)"
    )]
    cell_barcode_tag: Box<str>,

    #[arg(long = "print", help = "Print ASCII histogram to stderr")]
    print_histogram: bool,

    #[arg(
        long = "max-width",
        default_value_t = 60,
        help = "Maximum width of ASCII histogram"
    )]
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

/////////////////////
// Feature labels  //
/////////////////////

/// The four feature classes, in report order, named ONCE.
///
/// These strings are an output contract: downstream scripts select rows by
/// `#feature`. Spelled without apostrophes because that is what the established
/// TSV has always emitted — the coverage writer briefly used `5'UTR`/`3'UTR`,
/// so the same logical row was named two ways depending on whether `--bam` was
/// passed, and a script grepping `^5UTR` silently matched nothing.
const FEATURE_LABELS: [&str; 4] = ["5UTR", "CDS", "3UTR", "ncRNA"];

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
            print_row(FEATURE_LABELS[0], &self.five_prime, scale, max_width);
        }
        if !self.cds.is_empty() {
            print_row(FEATURE_LABELS[1], &self.cds, scale, max_width);
        }
        if !self.three_prime.is_empty() {
            print_row(FEATURE_LABELS[2], &self.three_prime, scale, max_width);
        }
        if !self.non_coding.is_empty() {
            print_row(FEATURE_LABELS[3], &self.non_coding, scale, max_width);
        }
    }

    pub fn to_tsv(&self, file_path: &str) -> anyhow::Result<()> {
        let mut writer = matrix_util::common_io::open_buf_writer(file_path)?;
        writeln!(writer, "#feature\tgenomic_bin\tcount")?;

        for (label, data) in [
            (FEATURE_LABELS[0], &self.five_prime),
            (FEATURE_LABELS[1], &self.cds),
            (FEATURE_LABELS[2], &self.three_prime),
            (FEATURE_LABELS[3], &self.non_coding),
        ] {
            for (i, &n) in data.iter().enumerate() {
                writeln!(writer, "{}\t{}\t{}", label, i, n)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

//////////////////////////////
// Shared metagene bin grid  //
//////////////////////////////

/// The union gene model and bin allocation that BOTH metagene tracks are
/// measured on.
///
/// Built once by [`run_metagene`] and handed to the count and the coverage
/// pass, because `count_per_covered_mb` divides one track by the other: when
/// each pass re-derived its own bin counts, a change to a bin floor in one of
/// the two places would have silently misaligned the grids and turned every
/// ratio wrong-but-plausible. One owner, so the two cannot differ.
struct MetageneGrid {
    /// Union 5'UTR / CDS / 3'UTR over protein-coding genes.
    five_prime_utr: HashMap<GeneId, GffRecord>,
    cds: HashMap<GeneId, GffRecord>,
    three_prime_utr: HashMap<GeneId, GffRecord>,
    /// Whole-gene boundaries of non-coding genes — no UTR/CDS split to make.
    non_coding: HashMap<GeneId, GffRecord>,
    /// Bins for `[5'UTR, CDS, 3'UTR]`, proportional to each region's max
    /// length. Non-coding genes are spread over the full `n_genomic_bins`.
    nbins: [usize; 3],
    n_genomic_bins: usize,
}

impl MetageneGrid {
    fn build(gff_file: &str, n_genomic_bins: usize) -> anyhow::Result<Self> {
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
            gene_boundaries: non_coding,
            ..
        } = build_union_gene_model(&non_coding_records)?;

        // Proportional bin allocation by max feature length. The floors stop a
        // long CDS from starving the short UTRs down to a couple of bins.
        let n_five_prime = five_prime_utr.take_max_length().max(10);
        let n_cds = cds.take_max_length();
        let n_three_prime = three_prime_utr.take_max_length().max(20);
        let ntot = (n_five_prime + n_cds + n_three_prime) as usize;

        let nbins = [
            n_five_prime as usize * n_genomic_bins / ntot,
            n_cds as usize * n_genomic_bins / ntot,
            n_three_prime as usize * n_genomic_bins / ntot,
        ];

        Ok(Self {
            five_prime_utr,
            cds,
            three_prime_utr,
            non_coding,
            nbins,
            n_genomic_bins,
        })
    }
}

fn count_metagene(sites: &[GenomicSite], grid: &MetageneGrid) -> GeneFeatureHistogram {
    let [nbins_five_prime, nbins_cds, nbins_three_prime] = grid.nbins;
    let n_genomic_bins = grid.n_genomic_bins;

    // Build feature indices
    let five_prime_idx = FeatureIndex::from_feature_map(&grid.five_prime_utr);
    let cds_idx = FeatureIndex::from_feature_map(&grid.cds);
    let three_prime_idx = FeatureIndex::from_feature_map(&grid.three_prime_utr);
    let nc_idx = FeatureIndex::from_feature_map(&grid.non_coding);

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

    GeneFeatureHistogram {
        five_prime: five_prime_hist,
        cds: cds_hist,
        three_prime: three_prime_hist,
        non_coding: non_coding_hist,
    }
}

/// Per-feature read-depth (coverage) histograms, parallel to
/// [`GeneFeatureHistogram`] but f64. Built by [`coverage_metagene`].
struct GeneFeatureCoverage {
    five_prime: Vec<f64>,
    cds: Vec<f64>,
    three_prime: Vec<f64>,
    non_coding: Vec<f64>,
}

/// Read-depth over the SAME [`MetageneGrid`] the count track is binned on,
/// streamed from `bam_files`. For 10x 3' chemistry this spikes at the
/// transcript terminus — the pileup a raw site-count metagene renders as a
/// spurious 3' peak.
fn coverage_metagene(
    bam_files: &[Box<str>],
    grid: &MetageneGrid,
    // The depth track pools every read into one profile, so no per-read cell
    // barcode is ever inspected — `--cell-barcode-tag` only picks a grouping in
    // the miami depth track. Threaded through so the documented flag keeps its
    // meaning if a per-cell-type metagene is ever added.
    _cell_barcode_tag: &str,
) -> anyhow::Result<GeneFeatureCoverage> {
    let [nbins_five, nbins_cds, nbins_three] = grid.nbins;

    Ok(GeneFeatureCoverage {
        five_prime: accumulate_feature_coverage(&grid.five_prime_utr, nbins_five, bam_files)?,
        cds: accumulate_feature_coverage(&grid.cds, nbins_cds, bam_files)?,
        three_prime: accumulate_feature_coverage(&grid.three_prime_utr, nbins_three, bam_files)?,
        non_coding: accumulate_feature_coverage(&grid.non_coding, grid.n_genomic_bins, bam_files)?,
    })
}

//////////////////////////////
// Per-gene read-depth scan  //
//////////////////////////////

/// One gene's feature interval, flattened out of the `DashMap` so the per-gene
/// scan can be a rayon `par_iter` over a slice.
struct FeatureRegion {
    region: Bed,
    edges: BinEdges,
    strand: Strand,
}

/// Per-rayon-job state for the depth scan: the BAM readers this worker has
/// opened, its own running histogram, and one reusable per-gene buffer.
struct CoverageWorker {
    cache: BamReaderCache,
    /// Genes this worker has folded in, already oriented 5'→3'.
    hist: Vec<f64>,
    /// Cleared per gene — the strand flip needs a gene's own bins before they
    /// can land in `hist`.
    gene_bins: Vec<f64>,
}

impl CoverageWorker {
    fn new(nbins: usize) -> Self {
        Self {
            cache: BamReaderCache::new(),
            hist: vec![0.0; nbins],
            gene_bins: vec![0.0; nbins],
        }
    }
}

/// Sum binned read depth across every gene's feature, oriented 5'→3' (reverse
/// strand bins flipped, matching [`strand_relative_pos`]).
///
/// This used to go through `read_depth_binned`, whose uncached entry point
/// builds a fresh `BamReaderCache` inside — so every one of ~10^5 genes, times
/// four feature maps, times every BAM, re-opened the file and re-parsed the
/// whole-genome `.bai` (tens to hundreds of ms each), serially. One cache per
/// worker plus a parallel fold is the fix `gene_count::splice` already applies.
///
/// The result does not depend on thread scheduling: every value added is a bp
/// overlap, a non-negative integer far below 2^53, so the f64 sums are exact
/// and therefore order-independent.
fn accumulate_feature_coverage(
    feature_map: &HashMap<GeneId, GffRecord>,
    nbins: usize,
    bam_files: &[Box<str>],
) -> anyhow::Result<Vec<f64>> {
    if nbins == 0 {
        return Ok(vec![]);
    }

    let regions: Vec<FeatureRegion> = feature_map
        .iter()
        .filter_map(|entry| {
            let rec = entry.value();
            if rec.stop <= rec.start {
                return None;
            }
            // GFF is 1-based inclusive; Bed / BinEdges are 0-based.
            Some(FeatureRegion {
                region: Bed {
                    chr: rec.seqname.clone(),
                    start: rec.start - 1,
                    stop: rec.stop,
                },
                edges: BinEdges::new(rec.start - 1, rec.stop - 1, nbins),
                strand: rec.strand,
            })
        })
        .collect();

    regions
        .par_iter()
        .try_fold(
            || CoverageWorker::new(nbins),
            |mut worker, feature| -> anyhow::Result<CoverageWorker> {
                worker.gene_bins.fill(0.0);
                let CoverageWorker {
                    cache, gene_bins, ..
                } = &mut worker;

                for bam in bam_files {
                    bam_io::for_each_record_in_region_cached(
                        cache,
                        bam,
                        &feature.region,
                        |_chr, rec| {
                            // Aligned blocks only, so introns are not counted —
                            // same convention as the splice-aware counter.
                            for [bs, be] in rec.aligned_blocks() {
                                accumulate_block(gene_bins, &feature.edges, bs, be);
                            }
                        },
                    )?;
                }

                match feature.strand {
                    Strand::Forward => {
                        for (h, v) in worker.hist.iter_mut().zip(worker.gene_bins.iter()) {
                            *h += v;
                        }
                    }
                    Strand::Backward => {
                        for (h, v) in worker.hist.iter_mut().zip(worker.gene_bins.iter().rev()) {
                            *h += v;
                        }
                    }
                }
                Ok(worker)
            },
        )
        .map(|worker| worker.map(|w| w.hist))
        .try_reduce(
            || vec![0.0; nbins],
            |mut acc, part| {
                for (a, b) in acc.iter_mut().zip(part) {
                    *a += b;
                }
                Ok(acc)
            },
        )
}

/// Write combined counts + coverage + normalized TSV. `count_per_covered_mb =
/// count / coverage * 1e6` divides out read depth: a 3' spike that survives
/// normalization is real methylation, not a coverage pileup.
fn write_combined_tsv(
    counts: &GeneFeatureHistogram,
    coverage: &GeneFeatureCoverage,
    path: &str,
) -> anyhow::Result<()> {
    let mut w = matrix_util::common_io::open_buf_writer(path)?;
    writeln!(
        w,
        "#feature\tgenomic_bin\tcount\tcoverage\tcount_per_covered_mb"
    )?;
    for (label, c, cov) in [
        (FEATURE_LABELS[0], &counts.five_prime, &coverage.five_prime),
        (FEATURE_LABELS[1], &counts.cds, &coverage.cds),
        (
            FEATURE_LABELS[2],
            &counts.three_prime,
            &coverage.three_prime,
        ),
        (FEATURE_LABELS[3], &counts.non_coding, &coverage.non_coding),
    ] {
        for (i, &n) in c.iter().enumerate() {
            let d = cov.get(i).copied().unwrap_or(0.0);
            let norm = if d > 0.0 { n as f64 / d * 1e6 } else { 0.0 };
            writeln!(w, "{}\t{}\t{}\t{}\t{:.3}", label, i, n, d as u64, norm)?;
        }
    }
    w.flush()?;
    Ok(())
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

    // Built once and shared: `count_per_covered_mb` divides the count track by
    // the coverage track, so both must be binned on the same gene model.
    let grid = MetageneGrid::build(&args.gff_file, args.num_bins)?;
    let histogram = count_metagene(&sites, &grid);

    if args.bam_files.is_empty() {
        histogram.to_tsv(&args.output)?;
    } else {
        info!(
            "computing read-depth coverage track from {} BAM(s)...",
            args.bam_files.len()
        );
        let coverage = coverage_metagene(&args.bam_files, &grid, &args.cell_barcode_tag)?;
        write_combined_tsv(&histogram, &coverage, &args.output)?;
    }
    info!("wrote metagene histogram to {}", args.output);

    if args.print_histogram {
        histogram.print(args.max_width);
    }

    Ok(())
}
