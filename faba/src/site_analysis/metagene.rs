use super::miami::bin::BinEdges;
use super::miami::depth::accumulate_block;
use super::site_io::*;
use crate::data::bam_io::{self, BamReaderCache};
use clap::Args;
use genomic_data::bed::Bed;
use genomic_data::gff::*;
use genomic_data::sam::Strand;
use log::info;
use rayon::prelude::*;
use rust_htslib::bam::ext::BamRecordExtensions;
use rustc_hash::FxHashMap;
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

/////////////////////////////////////
// Merged real-interval gene model  //
/////////////////////////////////////

/// One gene's 5'UTR, CDS, 3'UTR (or whole non-coding body) as the intervals
/// the GFF actually annotates, merged where they overlap or touch.
///
/// NOT a min..max span over the gene's records. `build_union_gene_model`
/// collapses every isoform's every CDS record into one interval reaching from
/// the first CDS base to the last, which swallows the introns AND, in 95.9% of
/// genes, the 3'UTR sitting inside that reach. Since a site is tested
/// 5'UTR -> CDS -> 3'UTR, that span used to CLAIM 3'UTR sites and — being at
/// the far end of it — pile them into the last CDS bin.
struct MergedFeature {
    seqname: Box<str>,
    strand: Strand,
    /// Sorted, disjoint, 1-based inclusive `(start, stop)`.
    intervals: Vec<(i64, i64)>,
    /// Summed length of `intervals`: the SPLICED length, which is what a
    /// metagene coordinate is a fraction of. An intron is not part of the
    /// feature, so it must not consume any of the feature's bins.
    total_len: i64,
}

/// Sort and merge intervals that overlap or abut.
///
/// Adjacency (`s == last_stop + 1`) merges too: two records that meet
/// base-to-base are one uninterrupted stretch of transcript, and leaving them
/// split would only add a seam with no coordinate consequence.
fn merge_intervals(intervals: &mut Vec<(i64, i64)>) {
    intervals.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(intervals.len());
    for &(start, stop) in intervals.iter() {
        match merged.last_mut() {
            Some(last) if start <= last.1 + 1 => last.1 = last.1.max(stop),
            _ => merged.push((start, stop)),
        }
    }
    *intervals = merged;
}

/// The identity a feature's records are pooled under: gene AND sequence name.
/// `parse_ensembl_id` drops the `_PAR_Y` suffix, so the X and Y copies of a
/// pseudoautosomal gene share a gene id; pooling on the id alone would place
/// chrY intervals on chrX and inflate the spliced length.
type GeneLocus = (GeneId, Box<str>);

/// One locus' raw, unmerged `(start, stop)` records and the strand they read on.
type RawIntervals = (Strand, Vec<(i64, i64)>);

/// Collects one feature kind's raw records per gene until they can be merged.
#[derive(Default)]
struct FeatureBuilder {
    by_gene: FxHashMap<GeneLocus, RawIntervals>,
}

impl FeatureBuilder {
    fn push(&mut self, rec: &GffRecord) {
        if rec.stop < rec.start {
            return;
        }
        self.by_gene
            .entry((rec.gene_id.clone(), rec.seqname.clone()))
            .or_insert_with(|| (rec.strand, Vec::new()))
            .1
            .push((rec.start, rec.stop));
    }

    fn finish(self) -> Vec<MergedFeature> {
        self.by_gene
            .into_iter()
            .filter_map(|((_gene, seqname), (strand, mut intervals))| {
                merge_intervals(&mut intervals);
                let total_len: i64 = intervals.iter().map(|&(s, e)| e - s + 1).sum();
                if total_len <= 0 {
                    return None;
                }
                Some(MergedFeature {
                    seqname,
                    strand,
                    intervals,
                    total_len,
                })
            })
            .collect()
    }
}

/// Gap between two closed intervals; `0` when they overlap.
fn distance_between_regions(a_start: i64, a_stop: i64, b_start: i64, b_stop: i64) -> i64 {
    if a_stop < b_start {
        b_start - a_stop
    } else if a_start > b_stop {
        a_start - b_stop
    } else {
        0
    }
}

/// The four feature tracks, one merged model per gene per kind.
struct FeatureTracks {
    five_prime_utr: Vec<MergedFeature>,
    cds: Vec<MergedFeature>,
    three_prime_utr: Vec<MergedFeature>,
    non_coding: Vec<MergedFeature>,
}

/// Sort raw GFF records into the four tracks.
///
/// GENCODE writes a generic `UTR` feature rather than `five_prime_UTR` /
/// `three_prime_UTR`, so which end a UTR record belongs to has to be decided
/// per record, by whether it sits closer to the gene's canonical start codon
/// or its stop codon — the same rule `genomic_data::gff::build_utr_maps`
/// applies, except that one can only rule on already-collapsed spans.
/// Explicit `five_prime_UTR` / `three_prime_UTR` records are taken as given.
fn build_feature_tracks(records: &[GffRecord]) -> anyhow::Result<FeatureTracks> {
    let start_codons = build_codon_map(records, &FeatureType::StartCodon)?;
    let stop_codons = build_codon_map(records, &FeatureType::StopCodon)?;

    let mut five_prime = FeatureBuilder::default();
    let mut cds = FeatureBuilder::default();
    let mut three_prime = FeatureBuilder::default();
    let mut non_coding = FeatureBuilder::default();

    for rec in records.iter() {
        if rec.gene_type != GeneType::CodingGene {
            // Non-coding genes have no UTR/CDS split to make, so they keep
            // their whole-gene boundaries.
            if rec.feature_type == FeatureType::Gene {
                non_coding.push(rec);
            }
            continue;
        }

        match rec.feature_type {
            FeatureType::CDS => cds.push(rec),
            FeatureType::FivePrimeUTR => five_prime.push(rec),
            FeatureType::ThreePrimeUTR => three_prime.push(rec),
            FeatureType::UTR => {
                let start_codon = start_codons.get(&rec.gene_id).map(|c| (c.start, c.stop));
                let stop_codon = stop_codons.get(&rec.gene_id).map(|c| (c.start, c.stop));
                // Without both codons the end cannot be named, and guessing
                // would put the record on a track it may not belong to.
                if let (Some((sc_start, sc_stop)), Some((pc_start, pc_stop))) =
                    (start_codon, stop_codon)
                {
                    let to_start = distance_between_regions(rec.start, rec.stop, sc_start, sc_stop);
                    let to_stop = distance_between_regions(rec.start, rec.stop, pc_start, pc_stop);
                    if to_start <= to_stop {
                        five_prime.push(rec);
                    } else {
                        three_prime.push(rec);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(FeatureTracks {
        five_prime_utr: five_prime.finish(),
        cds: cds.finish(),
        three_prime_utr: three_prime.finish(),
        non_coding: non_coding.finish(),
    })
}

////////////////////////////
// Feature interval index  //
////////////////////////////

/// One merged interval plus what it takes to place a genomic position along
/// the spliced feature the interval belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndexedInterval {
    start: i64,
    stop: i64,
    strand: Strand,
    /// Spliced length of the same feature lying genomically BEFORE this
    /// interval — the offset that turns a genomic position into a spliced one.
    cum_before: i64,
    total_len: i64,
}

impl IndexedInterval {
    /// 0-based offset of `pos` along the spliced feature, read 5'->3'.
    ///
    /// A reverse-strand transcript reads 5'->3' as the genomic coordinate
    /// DECREASES, so its offsets are mirrored about the feature's length.
    fn relative_pos(&self, pos: i64) -> i64 {
        let rel_genomic = self.cum_before + (pos - self.start);
        let rel = match self.strand {
            Strand::Forward => rel_genomic,
            Strand::Backward => self.total_len - 1 - rel_genomic,
        };
        rel.clamp(0, (self.total_len - 1).max(0))
    }

    /// Bin of `pos` within a track of `nbins` bins.
    fn bin(&self, pos: i64, nbins: usize) -> usize {
        let rel = self.relative_pos(pos) as usize;
        let total = self.total_len.max(1) as usize;
        (rel * nbins / total).min(nbins.saturating_sub(1))
    }
}

/// One chromosome's intervals, sorted by start.
struct ChromIntervals {
    intervals: Vec<IndexedInterval>,
    /// `max_stop[i]` = the largest `stop` among `intervals[..=i]`. Sorted-ness
    /// bounds where an interval BEGINS, not how far it reaches, so this is the
    /// only thing that licenses stopping a backward scan early.
    max_stop: Vec<i64>,
}

/// Per-chromosome sorted interval index for mapping positions to gene features.
struct FeatureIndex {
    by_chr: FxHashMap<Box<str>, ChromIntervals>,
}

impl FeatureIndex {
    fn from_features(features: &[MergedFeature]) -> Self {
        let mut by_chr: FxHashMap<Box<str>, Vec<IndexedInterval>> = FxHashMap::default();
        for feature in features {
            let mut cum_before = 0;
            for &(start, stop) in feature.intervals.iter() {
                by_chr
                    .entry(feature.seqname.clone())
                    .or_default()
                    .push(IndexedInterval {
                        start,
                        stop,
                        strand: feature.strand,
                        cum_before,
                        total_len: feature.total_len,
                    });
                cum_before += stop - start + 1;
            }
        }

        let by_chr = by_chr
            .into_iter()
            .map(|(chr, mut intervals)| {
                // A total order, not just by start: where two genes' features
                // both cover a site, `find` returns whichever the backward
                // scan reaches first, so ordering ties by start alone would
                // leave that site's bin at the mercy of insertion order.
                // Intervals equal on every key place a site in the same bin
                // of the same track, so the run is reproducible.
                intervals
                    .sort_by_key(|iv| (iv.start, iv.stop, iv.total_len, iv.cum_before, iv.strand));
                let mut running = i64::MIN;
                let max_stop = intervals
                    .iter()
                    .map(|iv| {
                        running = running.max(iv.stop);
                        running
                    })
                    .collect();
                (
                    chr,
                    ChromIntervals {
                        intervals,
                        max_stop,
                    },
                )
            })
            .collect();

        FeatureIndex { by_chr }
    }

    /// Find an interval containing `position` (1-based GFF coords).
    fn find(&self, chr: &str, position: i64) -> Option<IndexedInterval> {
        let chrom = self.by_chr.get(chr)?;
        // Rightmost interval with start <= position; everything left of it
        // also starts at or before `position`, so only `stop` is still open.
        let idx = chrom.intervals.partition_point(|iv| iv.start <= position);
        for i in (0..idx).rev() {
            // An earlier-STARTING interval can still reach much further than
            // its neighbours — a first CDS exon of a long gene outruns every
            // short exon that starts after it. Only "nothing up to here
            // reaches `position`" is a sound reason to stop looking.
            if chrom.max_stop[i] < position {
                break;
            }
            let iv = chrom.intervals[i];
            if position <= iv.stop {
                return Some(iv);
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

/// The gene model and bin allocation that BOTH metagene tracks are measured on.
///
/// Built once by [`run_metagene`] and handed to the count and the coverage
/// pass, because `count_per_covered_mb` divides one track by the other: when
/// each pass re-derived its own bin counts, a change to a bin floor in one of
/// the two places would have silently misaligned the grids and turned every
/// ratio wrong-but-plausible. One owner, so the two cannot differ.
struct MetageneGrid {
    /// Merged 5'UTR / CDS / 3'UTR intervals over protein-coding genes.
    five_prime_utr: Vec<MergedFeature>,
    cds: Vec<MergedFeature>,
    three_prime_utr: Vec<MergedFeature>,
    /// Whole-gene boundaries of non-coding genes — no UTR/CDS split to make.
    non_coding: Vec<MergedFeature>,
    /// Bins for `[5'UTR, CDS, 3'UTR]`, proportional to each region's max
    /// length. Non-coding genes are spread over the full `n_genomic_bins`.
    nbins: [usize; 3],
    n_genomic_bins: usize,
}

/// Longest spliced feature in a track, the scale each region's bin share is
/// proportional to.
fn max_total_len(features: &[MergedFeature]) -> i64 {
    features.iter().map(|f| f.total_len).max().unwrap_or(1)
}

impl MetageneGrid {
    fn build(gff_file: &str, n_genomic_bins: usize) -> anyhow::Result<Self> {
        Self::from_records(&read_gff_record_vec(gff_file)?, n_genomic_bins)
    }

    fn from_records(records: &[GffRecord], n_genomic_bins: usize) -> anyhow::Result<Self> {
        let FeatureTracks {
            five_prime_utr,
            cds,
            three_prime_utr,
            non_coding,
        } = build_feature_tracks(records)?;

        // Proportional bin allocation by max feature length. The floors stop a
        // long CDS from starving the short UTRs down to a couple of bins.
        let n_five_prime = max_total_len(&five_prime_utr).max(10);
        let n_cds = max_total_len(&cds);
        let n_three_prime = max_total_len(&three_prime_utr).max(20);
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

/// Add one site to the bin its spliced-relative position falls in.
fn tally(hist: &mut [usize], iv: &IndexedInterval, pos: i64) {
    if hist.is_empty() {
        return;
    }
    let bin = iv.bin(pos, hist.len());
    hist[bin] += 1;
}

fn count_metagene(sites: &[GenomicSite], grid: &MetageneGrid) -> GeneFeatureHistogram {
    let [nbins_five_prime, nbins_cds, nbins_three_prime] = grid.nbins;
    let n_genomic_bins = grid.n_genomic_bins;

    // Build feature indices
    let five_prime_idx = FeatureIndex::from_features(&grid.five_prime_utr);
    let cds_idx = FeatureIndex::from_features(&grid.cds);
    let three_prime_idx = FeatureIndex::from_features(&grid.three_prime_utr);
    let nc_idx = FeatureIndex::from_features(&grid.non_coding);

    let mut five_prime_hist = vec![0usize; nbins_five_prime];
    let mut cds_hist = vec![0usize; nbins_cds];
    let mut three_prime_hist = vec![0usize; nbins_three_prime];
    let mut non_coding_hist = vec![0usize; n_genomic_bins];

    for site in sites {
        let chr = site.chr.as_ref();
        // Sites use 0-based positions; GFF uses 1-based
        let gff_pos = site.position + 1;

        if let Some(iv) = five_prime_idx.find(chr, gff_pos) {
            tally(&mut five_prime_hist, &iv, gff_pos);
        } else if let Some(iv) = cds_idx.find(chr, gff_pos) {
            tally(&mut cds_hist, &iv, gff_pos);
        } else if let Some(iv) = three_prime_idx.find(chr, gff_pos) {
            tally(&mut three_prime_hist, &iv, gff_pos);
        } else if let Some(iv) = nc_idx.find(chr, gff_pos) {
            tally(&mut non_coding_hist, &iv, gff_pos);
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

/// One gene's feature, flattened out of [`MergedFeature`] so the per-gene scan
/// can be a rayon `par_iter` over a slice.
struct FeatureRegion {
    /// Window to fetch reads over: the outer span of the merged intervals.
    /// Reads landing in the introns inside it are clipped away by `exons`.
    region: Bed,
    /// Bins over the SPLICED feature, `[0, total_len - 1]` — the same
    /// coordinate the count track bins sites in.
    edges: BinEdges,
    /// Per merged interval: 0-based half-open `(genomic start, genomic stop,
    /// spliced offset)`.
    exons: Vec<(i64, i64, i64)>,
    strand: Strand,
}

impl FeatureRegion {
    fn new(feature: &MergedFeature, nbins: usize) -> Option<Self> {
        let (&(first_start, _), &(_, last_stop)) =
            (feature.intervals.first()?, feature.intervals.last()?);

        let mut offset = 0;
        let mut exons = Vec::with_capacity(feature.intervals.len());
        for &(start, stop) in feature.intervals.iter() {
            // GFF is 1-based inclusive; BAM blocks are 0-based half-open.
            exons.push((start - 1, stop, offset));
            offset += stop - start + 1;
        }

        Some(FeatureRegion {
            region: Bed {
                chr: feature.seqname.clone(),
                start: first_start - 1,
                stop: last_stop,
            },
            edges: BinEdges::new(0, feature.total_len - 1, nbins),
            exons,
            strand: feature.strand,
        })
    }
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
/// strand bins flipped, matching [`IndexedInterval::relative_pos`]).
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
    features: &[MergedFeature],
    nbins: usize,
    bam_files: &[Box<str>],
) -> anyhow::Result<Vec<f64>> {
    if nbins == 0 {
        return Ok(vec![]);
    }

    let regions: Vec<FeatureRegion> = features
        .iter()
        .filter_map(|feature| FeatureRegion::new(feature, nbins))
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
                                // Depth is binned in spliced coordinates, so a
                                // block is clipped to each exon and shifted by
                                // that exon's offset before it is accumulated.
                                for &(gstart, gstop, offset) in feature.exons.iter() {
                                    let lo = bs.max(gstart);
                                    let hi = be.min(gstop);
                                    if lo < hi {
                                        accumulate_block(
                                            gene_bins,
                                            &feature.edges,
                                            offset + lo - gstart,
                                            offset + hi - gstart,
                                        );
                                    }
                                }
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

#[cfg(test)]
mod tests;
