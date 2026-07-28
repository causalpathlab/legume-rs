use super::site_io::*;
use clap::Args;
use genomic_data::gff::*;
use genomic_data::sam::Strand;
use log::info;
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
        help = "Total bins across the metagene (default: 57)",
        long_help = "Total number of bins across the metagene profile.\n\
                     \n\
                     Bins are split by each region's longest SPLICED length.\n\
                     Floors of 10 bins for the 5'UTR and 20 for the 3'UTR apply.\n\
                     Otherwise a long CDS would starve them.\n\
                     The split therefore depends on the annotation.\n\
                     Compare the shape of two profiles, not their bar widths.\n\
                     \n\
                     Sites are assigned to a feature's MERGED annotated intervals.\n\
                     Position runs along the spliced feature.\n\
                     Introns consume no metagene coordinate.\n\
                     A min-start..max-stop span would be the obvious shortcut.\n\
                     It is also wrong: that CDS span covers a median 83% of the gene.\n\
                     It overlaps the 3'UTR in 96% of genes.\n\
                     CDS then claims 3'UTR sites and piles them into its LAST bin.\n\
                     The result is a terminal spike that reads as biology.\n\
                     See docs/profiling-methods.md section 1.1."
    )]
    num_bins: usize,

    #[arg(
        long = "include-non-coding",
        help = "Also profile non-coding genes, as a separate ncRNA track",
        long_help = "Also profile non-coding genes, as a separate ncRNA track.\n\
                     \n\
                     A non-coding gene has no start or stop codon to split on.\n\
                     Its whole body becomes one undivided ncRNA track.\n\
                     That is not the coordinate the other three are on.\n\
                     Reading all four as one profile compares unlike lengths."
    )]
    include_non_coding: bool,

    #[arg(short, long, required = true, help = "Output TSV file path")]
    output: Box<str>,

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

/// Sort raw GFF records into the feature tracks, keeping the non-coding track
/// only when `include_non_coding` asks for it.
///
/// GENCODE writes a generic `UTR` feature rather than `five_prime_UTR` /
/// `three_prime_UTR`, so which end a UTR record belongs to has to be decided
/// per record, by whether it sits closer to the gene's canonical start codon
/// or its stop codon — the same rule `genomic_data::gff::build_utr_maps`
/// applies, except that one can only rule on already-collapsed spans.
/// Explicit `five_prime_UTR` / `three_prime_UTR` records are taken as given.
fn build_feature_tracks(
    records: &[GffRecord],
    include_non_coding: bool,
) -> anyhow::Result<FeatureTracks> {
    // Only the generic-`UTR` arm below consults the codons, and GFF3 names both
    // UTR ends outright — so on that input the two maps would be built, cloning
    // a record apiece, and never read. The scan stops at the first bare `UTR`,
    // which is the first record of interest on the GTF input that does need it.
    let needs_codons = records
        .iter()
        .any(|rec| rec.feature_type == FeatureType::UTR);
    let (start_codons, stop_codons) = if needs_codons {
        (
            build_codon_map(records, &FeatureType::StartCodon)?,
            build_codon_map(records, &FeatureType::StopCodon)?,
        )
    } else {
        Default::default()
    };

    let mut five_prime = FeatureBuilder::default();
    let mut cds = FeatureBuilder::default();
    let mut three_prime = FeatureBuilder::default();
    let mut non_coding = FeatureBuilder::default();

    for rec in records.iter() {
        if rec.gene_type != GeneType::CodingGene {
            // Whole-gene boundaries: there is no UTR/CDS split to make.
            // Off by default; the reason is on `--include-non-coding`.
            if include_non_coding && rec.feature_type == FeatureType::Gene {
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
            // Once per FEATURE, not once per interval: the lookup and the
            // seqname clone are the same for every interval of one feature.
            let chrom = by_chr.entry(feature.seqname.clone()).or_default();
            let mut cum_before = 0;
            for &(start, stop) in feature.intervals.iter() {
                chrom.push(IndexedInterval {
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
/// TSV has always emitted; a second speller once wrote `5'UTR`/`3'UTR` for the
/// same logical row, and a script grepping `^5UTR` silently matched nothing.
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

/// The merged gene model and the bin width of each track, together — the whole
/// input `count_metagene` measures sites against.
struct MetageneGrid {
    /// Merged 5'UTR / CDS / 3'UTR intervals over protein-coding genes.
    five_prime_utr: Vec<MergedFeature>,
    cds: Vec<MergedFeature>,
    three_prime_utr: Vec<MergedFeature>,
    /// Whole-gene boundaries of non-coding genes — no UTR/CDS split to make.
    /// Empty unless `--include-non-coding` asked for the track.
    non_coding: Vec<MergedFeature>,
    /// Bin widths for `[5'UTR, CDS, 3'UTR, ncRNA]`, in the order the tracks are
    /// reported. The three coding widths are proportional to each region's max
    /// length; ncRNA gets the whole budget when it is on and `0` when it is
    /// off, so a track nobody asked for writes no rows rather than a run of
    /// zeros a reader would have to know to ignore. One owner for all four, so
    /// no track can end up sized against a different grid than its neighbours.
    nbins: [usize; 4],
}

/// Longest spliced feature in a track, the scale each region's bin share is
/// proportional to.
fn max_total_len(features: &[MergedFeature]) -> i64 {
    features.iter().map(|f| f.total_len).max().unwrap_or(1)
}

impl MetageneGrid {
    fn from_records(
        records: &[GffRecord],
        n_genomic_bins: usize,
        include_non_coding: bool,
    ) -> anyhow::Result<Self> {
        let FeatureTracks {
            five_prime_utr,
            cds,
            three_prime_utr,
            non_coding,
        } = build_feature_tracks(records, include_non_coding)?;

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
            if include_non_coding {
                n_genomic_bins
            } else {
                0
            },
        ];

        Ok(Self {
            five_prime_utr,
            cds,
            three_prime_utr,
            non_coding,
            nbins,
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
    let [nbins_five_prime, nbins_cds, nbins_three_prime, nbins_non_coding] = grid.nbins;

    // Build feature indices
    let five_prime_idx = FeatureIndex::from_features(&grid.five_prime_utr);
    let cds_idx = FeatureIndex::from_features(&grid.cds);
    let three_prime_idx = FeatureIndex::from_features(&grid.three_prime_utr);
    let nc_idx = FeatureIndex::from_features(&grid.non_coding);

    let mut five_prime_hist = vec![0usize; nbins_five_prime];
    let mut cds_hist = vec![0usize; nbins_cds];
    let mut three_prime_hist = vec![0usize; nbins_three_prime];
    let mut non_coding_hist = vec![0usize; nbins_non_coding];

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

pub fn run_metagene(args: &MetageneArgs) -> anyhow::Result<()> {
    let sites = read_sites(&args.site_file)?;

    let records = read_gff_record_vec(&args.gff_file)?;
    let grid = MetageneGrid::from_records(&records, args.num_bins, args.include_non_coding)?;
    let histogram = count_metagene(&sites, &grid);
    histogram.to_tsv(&args.output)?;
    info!("wrote metagene histogram to {}", args.output);

    if args.print_histogram {
        histogram.print(args.max_width);
    }

    Ok(())
}

#[cfg(test)]
mod tests;
