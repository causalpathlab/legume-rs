//! Metagene profiles over a simplified 5'UTR / CDS / 3'UTR transcript.
//!
//! This follows MetaPlotR, which is what the published m6A metagenes we compare
//! against were made with, so that a difference between our profile and theirs
//! is a difference in the DATA rather than in the procedure:
//!
//! Olarerin-George AO, Jaffrey SR. *MetaPlotR: a Perl/R pipeline for plotting
//! metagenes of nucleotide modifications and other transcriptomic sites.*
//! Bioinformatics 33, 1563–1564 (2017).
//! <https://doi.org/10.1093/bioinformatics/btx002>
//!
//! Two places where we follow its stated procedure rather than its published
//! code are called out on [`elect_longest_isoform`] and [`ScaleFactors`].

use super::site_io::*;
use clap::{Args, ValueEnum};
use genomic_data::gff::*;
use genomic_data::sam::Strand;
use genomic_data::transcript::{
    build_transcript_models, elect_longest_isoform, merge_intervals, TranscriptModel,
};
use log::info;
use rustc_hash::FxHashMap;
use std::io::Write;

/// Which isoforms carry the sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum IsoformPolicy {
    /// One transcript per gene, the longest spliced one. MetaPlotR's procedure.
    Longest,
    /// Every coding transcript. A site in several isoforms is counted in each.
    All,
}

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
        default_value_t = 200,
        help = "Total bins across the metagene (default: 200)",
        long_help = "Total number of bins across the 5'UTR, CDS and 3'UTR.\n\
                     \n\
                     MetaPlotR plots its metagene with 200 breaks, hence the default.\n\
                     Bins are split between the three regions in proportion to each\n\
                     region's MEDIAN spliced length over the assigned sites.\n\
                     A maximum would be set by a single gene: titin's merged CDS is\n\
                     114,586 nt against a median of 1,347, which is what the previous\n\
                     allocation keyed on.\n\
                     The split depends on the annotation and on the sites, so compare\n\
                     the shape of two profiles rather than their bar widths."
    )]
    num_bins: usize,

    #[arg(
        long = "isoforms",
        value_enum,
        default_value = "longest",
        help = "Which isoforms sites are placed on",
        long_help = "Which isoforms sites are placed on.\n\
                     \n\
                     `longest` keeps one transcript per gene, the longest spliced one.\n\
                     That is MetaPlotR's stated procedure.\n\
                     `all` keeps every coding transcript.\n\
                     A site inside several isoforms is then counted once per isoform,\n\
                     which is what MetaPlotR's own distance table does."
    )]
    isoforms: IsoformPolicy,

    #[arg(
        long = "include-non-coding",
        help = "Also profile non-coding genes, as a separate ncRNA track",
        long_help = "Also profile non-coding genes, as a separate ncRNA track.\n\
                     \n\
                     This has no counterpart in MetaPlotR, which profiles coding\n\
                     transcripts only.\n\
                     A non-coding gene has no start or stop codon to split on.\n\
                     Its whole body becomes one undivided track on its own [0,1] axis,\n\
                     and its density is normalized within that track alone."
    )]
    include_non_coding: bool,

    #[arg(short, long, required = true, help = "Output TSV file path")]
    output: Box<str>,

    #[arg(
        long = "dist-measures",
        help = "Also write MetaPlotR's per-site distance table to this path",
        long_help = "Also write the per-site distance table, one row per site and\n\
                     transcript it was placed on.\n\
                     \n\
                     Column names match MetaPlotR's own `dist_measures` output, so its\n\
                     `visualize_metagenes.R` runs on this file unmodified.\n\
                     That turns \"our profile looks like theirs\" into \"their script,\n\
                     our data\"."
    )]
    dist_measures: Option<Box<str>>,

    #[arg(long = "print", help = "Print ASCII histogram to stderr")]
    print_histogram: bool,

    #[arg(
        long = "max-width",
        default_value_t = 60,
        help = "Maximum width of ASCII histogram"
    )]
    max_width: usize,
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

/// Region indices into [`FEATURE_LABELS`], and the base of each region's
/// MetaPlotR coordinate: 5'UTR spans [0,1), CDS [1,2), 3'UTR [2,3).
const UTR5: usize = 0;
const CDS: usize = 1;
const UTR3: usize = 2;
const NCRNA: usize = 3;

////////////////////////////
// Feature interval index  //
////////////////////////////

/// One interval of one region of one transcript, plus what it takes to place a
/// genomic position along that region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndexedInterval {
    start: i64,
    stop: i64,
    strand: Strand,
    /// Spliced length of the same region lying genomically BEFORE this
    /// interval — the offset that turns a genomic position into a spliced one.
    cum_before: i64,
    /// Spliced length of the region this interval belongs to.
    total_len: i64,
    /// Which of [`FEATURE_LABELS`] this interval is.
    region: usize,
    /// Index into the model table, so an assignment can report its
    /// transcript's three region sizes. `u32::MAX` for the ncRNA track, which
    /// has no transcript.
    model: u32,
}

impl IndexedInterval {
    /// 0-based offset of `pos` along the spliced region, read 5'->3'.
    ///
    /// A reverse-strand transcript reads 5'->3' as the genomic coordinate
    /// DECREASES, so its offsets are mirrored about the region's length.
    fn relative_pos(&self, pos: i64) -> i64 {
        let rel_genomic = self.cum_before + (pos - self.start);
        let rel = match self.strand {
            Strand::Forward => rel_genomic,
            Strand::Backward => self.total_len - 1 - rel_genomic,
        };
        rel.clamp(0, (self.total_len - 1).max(0))
    }

    /// This interval's placement of `pos`, ready to bin.
    ///
    /// Binning lives on [`SiteAssignment`] rather than here: the bin widths are
    /// not known until every site has been placed, because MetaPlotR weights its
    /// medians by the sites.
    fn place(&self, site: u32, pos: i64) -> SiteAssignment {
        SiteAssignment {
            site,
            model: (self.region != NCRNA).then_some(self.model),
            region: self.region,
            rel: self.relative_pos(pos),
            total_len: self.total_len,
        }
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

/// Per-chromosome sorted interval index over every region of every transcript.
///
/// `IndexedInterval::model` indexes the SAME `models` slice this was built from;
/// keeping a private copy of each transcript's region sizes here would be a
/// second table that has to stay index-aligned with the first.
struct RegionIndex {
    by_chr: FxHashMap<Box<str>, ChromIntervals>,
}

impl RegionIndex {
    fn build(models: &[TranscriptModel], non_coding: &[NonCodingBody]) -> Self {
        let mut by_chr: FxHashMap<Box<str>, Vec<IndexedInterval>> = FxHashMap::default();

        for (mi, m) in models.iter().enumerate() {
            let chrom = by_chr.entry(m.seqname.clone()).or_default();
            for (region, intervals, total_len) in [
                (UTR5, &m.utr5, m.utr5_size),
                (CDS, &m.cds, m.cds_size),
                (UTR3, &m.utr3, m.utr3_size),
            ] {
                let mut cum_before = 0;
                for &(start, stop) in intervals.iter() {
                    chrom.push(IndexedInterval {
                        start,
                        stop,
                        strand: m.strand,
                        cum_before,
                        total_len,
                        region,
                        model: mi as u32,
                    });
                    cum_before += stop - start + 1;
                }
            }
        }

        // Same shape as the coding loop: one lookup and one clone per LOCUS.
        for body in non_coding.iter() {
            let chrom = by_chr.entry(body.seqname.clone()).or_default();
            let total_len: i64 = body.intervals.iter().map(|&(s, e)| e - s + 1).sum();
            let mut cum_before = 0;
            for &(start, stop) in body.intervals.iter() {
                chrom.push(IndexedInterval {
                    start,
                    stop,
                    strand: body.strand,
                    cum_before,
                    total_len,
                    region: NCRNA,
                    model: u32::MAX,
                });
                cum_before += stop - start + 1;
            }
        }

        let by_chr = by_chr
            .into_iter()
            .map(|(chr, mut intervals)| {
                // A total order, not just by start: ties on every key place a
                // site identically, so the run is reproducible whatever order
                // the parser handed the records over in.
                intervals.sort_by_key(|iv| {
                    (
                        iv.start,
                        iv.stop,
                        iv.region,
                        iv.model,
                        iv.total_len,
                        iv.cum_before,
                        iv.strand,
                    )
                });
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

        RegionIndex { by_chr }
    }

    /// Every interval containing `position` (1-based GFF coords).
    ///
    /// All of them, not the first: MetaPlotR emits one row per site and
    /// transcript, so a site inside two elected transcripts is counted in each.
    fn find_all(&self, chr: &str, position: i64, out: &mut Vec<IndexedInterval>) {
        out.clear();
        let Some(chrom) = self.by_chr.get(chr) else {
            return;
        };
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
                out.push(iv);
            }
        }
    }
}

///////////////////////
// Site assignments  //
///////////////////////

/// One site placed on one transcript.
///
/// Placement only — nothing here depends on how many bins a track ends up with,
/// which matters because the bin widths are not known until every site has been
/// placed (they come from the site-weighted medians below).
struct SiteAssignment {
    /// Index into the site list. The row owns no strings: with ~240k rows over
    /// ~25 distinct chromosome names, a `Box<str>` per row is pure waste, and
    /// only `--dist-measures` ever reads the name back.
    site: u32,
    /// `None` on the ncRNA track, which has no transcript. An in-band sentinel
    /// here would have to be remembered at every site that indexes the model
    /// table; this makes the compiler ask.
    model: Option<u32>,
    region: usize,
    /// 0-based offset along the spliced region, and that region's spliced
    /// length. Kept as integers so [`SiteAssignment::bin`] is the same exact
    /// arithmetic as [`IndexedInterval::bin`] rather than a float round-trip.
    rel: i64,
    total_len: i64,
}

impl SiteAssignment {
    /// Bin within a track of `nbins` bins.
    fn bin(&self, nbins: usize) -> usize {
        let total = self.total_len.max(1) as usize;
        ((self.rel as usize) * nbins / total).min(nbins.saturating_sub(1))
    }

    /// MetaPlotR's coordinate: 5'UTR in [0,1), CDS in [1,2), 3'UTR in [2,3).
    /// The ncRNA track is on its own [0,1) axis.
    fn rel_location(&self) -> f64 {
        let base = if self.region == NCRNA {
            0.0
        } else {
            self.region as f64
        };
        base + self.rel as f64 / self.total_len.max(1) as f64
    }
}

/// Collect every (site, transcript) placement.
fn assign_sites(sites: &[GenomicSite], index: &RegionIndex) -> (Vec<SiteAssignment>, usize) {
    let mut out = Vec::new();
    let mut hits = Vec::new();
    let mut unassigned = 0usize;

    for (si, site) in sites.iter().enumerate() {
        // Sites use 0-based positions; GFF uses 1-based.
        let gff_pos = site.position + 1;
        index.find_all(site.chr.as_ref(), gff_pos, &mut hits);
        if hits.is_empty() {
            unassigned += 1;
            continue;
        }
        for iv in hits.iter() {
            out.push(iv.place(si as u32, gff_pos));
        }
    }
    (out, unassigned)
}

//////////////////////
// Scale factors    //
//////////////////////

/// MetaPlotR's display widths: each UTR's median size relative to the CDS's.
///
/// The medians are taken over the ASSIGNED SITES, not over the transcript set —
/// `visualize_metagenes.R` computes them from `dist`, which holds one row per
/// site, so a transcript carrying many sites weighs more than one carrying few.
/// Measured on our own m6A calls the two differ by 59% in the 3'UTR
/// (site-weighted 1.68 against transcript-weighted 1.06), which is enough to
/// make a profile look comparable to a published one when it is not.
struct ScaleFactors {
    /// Each region's median size, DOUBLED — which is what everything downstream
    /// actually computes on, so it is what gets stored. Halving here to report
    /// a "median" would throw away the exactness the doubling exists for, and
    /// an even-n median is not an integer.
    twice_median: [i64; 3],
    utr5_sf: f64,
    utr3_sf: f64,
}

impl ScaleFactors {
    /// Place a within-region fraction on MetaPlotR's rescaled axis, where the
    /// CDS keeps width 1 and each UTR is scaled to its median size relative to
    /// the CDS. Bin edges and per-site coordinates both go through here so the
    /// two cannot describe different axes.
    fn rescale(&self, region: usize, within: f64) -> f64 {
        match region {
            UTR5 => 1.0 - self.utr5_sf * (1.0 - within),
            CDS => 1.0 + within,
            UTR3 => 2.0 + self.utr3_sf * within,
            _ => within, // ncRNA: its own [0,1] axis
        }
    }

    /// Median region sizes, for reporting only.
    fn median(&self) -> [f64; 3] {
        [
            self.twice_median[0] as f64 / 2.0,
            self.twice_median[1] as f64 / 2.0,
            self.twice_median[2] as f64 / 2.0,
        ]
    }
}

/// Median of a slice, doubled, so an even-length median stays an integer.
///
/// Doubled because the bin allocation divides medians by their sum and must
/// stay in integer arithmetic to be reproducible. Only the two middle order
/// statistics are needed, so this selects rather than sorts.
fn twice_median(values: &mut [i64]) -> i64 {
    if values.is_empty() {
        return 0;
    }
    let n = values.len();
    let (lo, mid, _) = values.select_nth_unstable(n / 2);
    if n % 2 == 1 {
        2 * *mid
    } else {
        // `lo` holds the n/2 smallest; its max is the lower middle value.
        *mid + *lo.iter().max().expect("n >= 2 when n is even and non-zero")
    }
}

fn scale_factors(
    assignments: &[SiteAssignment],
    models: &[TranscriptModel],
) -> anyhow::Result<ScaleFactors> {
    let mut per_region: [Vec<i64>; 3] = Default::default();
    for a in assignments.iter() {
        let Some(mi) = a.model else {
            continue; // ncRNA: no transcript, no region sizes
        };
        let m = &models[mi as usize];
        per_region[UTR5].push(m.utr5_size);
        per_region[CDS].push(m.cds_size);
        per_region[UTR3].push(m.utr3_size);
    }
    let mut m = [0i64; 3];
    for r in 0..3 {
        m[r] = twice_median(&mut per_region[r]);
    }
    if m[CDS] == 0 {
        anyhow::bail!(
            "no site was placed on a coding sequence, so MetaPlotR's scale factors \
             (which are relative to the median CDS) are undefined"
        );
    }
    Ok(ScaleFactors {
        twice_median: m,
        utr5_sf: m[UTR5] as f64 / m[CDS] as f64,
        utr3_sf: m[UTR3] as f64 / m[CDS] as f64,
    })
}

/// Split `n` bins between the three regions in proportion to their medians.
///
/// Integer throughout, with the remainder going to the largest fractional
/// parts, so the three always sum to `n` and the result does not depend on
/// float rounding.
fn allocate_bins(n: usize, m: &[i64; 3]) -> [usize; 3] {
    let total: i64 = m.iter().sum();
    if total <= 0 {
        return [0, n, 0];
    }
    let mut out = [0usize; 3];
    let mut rem = [(0i64, 0usize); 3];
    let mut used = 0usize;
    for r in 0..3 {
        let exact = m[r] * n as i64;
        out[r] = (exact / total) as usize;
        used += out[r];
        rem[r] = (exact % total, r);
    }
    // Largest remainder first; ties to the wider region, then to the earlier.
    rem.sort_by_key(|&(f, r)| (std::cmp::Reverse(f), std::cmp::Reverse(m[r]), r));
    for &(_, r) in rem.iter().take(n.saturating_sub(used)) {
        out[r] += 1;
    }
    out
}

/// The bin width of every track, decided in one place.
///
/// All four widths come out of one constructor on purpose. They were split once
/// before — the ncRNA width sized by its own statement beside a `[usize; 3]` —
/// and a track sized against a different grid than its neighbours is exactly
/// the drift this type exists to prevent.
struct BinGrid([usize; 4]);

impl BinGrid {
    fn new(n: usize, scale: &ScaleFactors, include_non_coding: bool) -> Self {
        let coding = allocate_bins(n, &scale.twice_median);
        // The ncRNA track is on its own axis, so it gets the whole budget —
        // and `0` when it was not asked for, which is what makes `to_tsv` emit
        // no rows for it rather than a run of zeros a reader must know to skip.
        BinGrid([
            coding[UTR5],
            coding[CDS],
            coding[UTR3],
            if include_non_coding { n } else { 0 },
        ])
    }
}

//////////////////
// Histogram    //
//////////////////

pub struct GeneFeatureHistogram {
    /// One row of bins per track. Each track's bin count is its own length —
    /// carrying `nbins` beside this would be a second copy that could disagree.
    counts: [Vec<usize>; 4],
    scale: ScaleFactors,
}

impl GeneFeatureHistogram {
    /// Tally every placement, once the grid has fixed the bin widths.
    fn accumulate(grid: &BinGrid, scale: ScaleFactors, assignments: &[SiteAssignment]) -> Self {
        let mut counts: [Vec<usize>; 4] =
            std::array::from_fn(|region| vec![0usize; grid.0[region]]);
        for a in assignments.iter() {
            let track = &mut counts[a.region];
            let width = track.len();
            if width > 0 {
                // `bin` already clamps to the track width.
                track[a.bin(width)] += 1;
            }
        }
        GeneFeatureHistogram { counts, scale }
    }

    /// Rescaled coordinate spanned by one bin, as MetaPlotR's plot draws it.
    fn bin_edges(&self, region: usize, i: usize) -> (f64, f64) {
        let b = self.counts[region].len().max(1) as f64;
        (
            self.scale.rescale(region, i as f64 / b),
            self.scale.rescale(region, (i + 1) as f64 / b),
        )
    }

    pub fn print(&self, max_width: usize) {
        let nmax = self
            .counts
            .iter()
            .flat_map(|c| c.iter())
            .cloned()
            .max()
            .unwrap_or(0);
        if nmax == 0 {
            eprintln!("(no sites mapped to gene features)");
            return;
        }
        let scale = nmax.div_ceil(max_width);
        for (region, data) in self.counts.iter().enumerate() {
            for &n in data.iter() {
                let n1 = n.div_ceil(scale);
                let n0 = max_width.saturating_sub(n1);
                eprintln!(
                    "{:<6}{}{} {}",
                    FEATURE_LABELS[region],
                    "*".repeat(n1),
                    " ".repeat(n0),
                    n
                );
            }
        }
    }

    pub fn to_tsv(&self, file_path: &str) -> anyhow::Result<()> {
        let mut writer = matrix_util::common_io::open_buf_writer(file_path)?;
        // Line 1 is unchanged: downstream scripts read the first three columns
        // positionally and select rows by `#feature`.
        writeln!(
            writer,
            "#feature\tgenomic_bin\tcount\tbin_start\tbin_end\tfrac\tdensity"
        )?;

        // Coding regions share one density; the ncRNA track is on a different
        // axis and normalizes within itself, or `density` would mean two things.
        let coding_total: usize = self.counts[..3].iter().flat_map(|c| c.iter()).sum();
        let nc_total: usize = self.counts[NCRNA].iter().sum();

        for (region, data) in self.counts.iter().enumerate() {
            let total = if region == NCRNA {
                nc_total
            } else {
                coding_total
            };
            for (i, &n) in data.iter().enumerate() {
                let (lo, hi) = self.bin_edges(region, i);
                let width = hi - lo;
                let (frac, density) = if total == 0 || width <= 0.0 {
                    (0.0, 0.0)
                } else {
                    let f = n as f64 / total as f64;
                    (f, f / width)
                };
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                    FEATURE_LABELS[region], i, n, lo, hi, frac, density
                )?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}

/// MetaPlotR's `*.dist.measures.txt` schema, so its `visualize_metagenes.R`
/// runs on this file with only its input path changed.
///
/// The first fourteen columns and their order are `rel_and_abs_dist_calc.pl`'s,
/// not ours; `strand` and `rescaled_location` are appended after them so a
/// positional reader of theirs is unaffected.
///
/// The six `_st`/`_end` columns are that script's ABSOLUTE distances:
/// `mrna_pos - endpoint`, in 1-based spliced coordinates running 5'->3'
/// (`make_annot_bed.pl` numbers the first exonic base 1 on either strand).
/// `utr3_st` is therefore the signed distance from the stop codon — negative
/// inside the CDS, positive into the 3'UTR — which is the coordinate its
/// feature-distance plot is drawn on, and the one a landmark-anchored profile
/// needs. A region the transcript does not have prints `NA`, as theirs does.
///
/// `coord` is 1-BASED. MetaPlotR reads it off the `end` field of a 0-based BED
/// (`chr1 566859 566860` yields 566860), whereas the site parquet stores the
/// 0-based position — verified against hg38: at `primary_pos` chr1:169804275
/// the reference base is G, and only at `primary_pos + 1` is it the A of the
/// RAC, with the deaminated C following.
const DIST_MEASURES_HEADER: &str = "chr\tcoord\tgene_name\trefseqID\trel_location\t\
     utr5_st\tutr5_end\tcds_st\tcds_end\tutr3_st\tutr3_end\t\
     utr5_size\tcds_size\tutr3_size\tstrand\trescaled_location";

fn write_dist_measures(
    path: &str,
    sites: &[GenomicSite],
    assignments: &[SiteAssignment],
    models: &[TranscriptModel],
    scale: &ScaleFactors,
) -> anyhow::Result<()> {
    let mut w = matrix_util::common_io::open_buf_writer(path)?;
    writeln!(w, "{}", DIST_MEASURES_HEADER)?;

    for a in assignments.iter() {
        let Some(mi) = a.model else {
            continue; // ncRNA has no transcript, and no MetaPlotR counterpart
        };
        let m = &models[mi as usize];
        let site = &sites[a.site as usize];
        let rel_location = a.rel_location();

        // 1-based position along the mature transcript.
        let preceding = match a.region {
            UTR5 => 0,
            CDS => m.utr5_size,
            _ => m.utr5_size + m.cds_size,
        };
        let mrna_pos = preceding + a.rel + 1;

        // Region boundaries in the same frame, inclusive on both ends.
        let bounds = [
            (m.utr5_size > 0).then_some((1, m.utr5_size)),
            (m.cds_size > 0).then_some((m.utr5_size + 1, m.utr5_size + m.cds_size)),
            (m.utr3_size > 0).then_some((
                m.utr5_size + m.cds_size + 1,
                m.utr5_size + m.cds_size + m.utr3_size,
            )),
        ];
        let mut abs = String::new();
        for b in bounds.iter() {
            match b {
                Some((st, end)) => {
                    abs.push_str(&format!("{}\t{}\t", mrna_pos - st, mrna_pos - end))
                }
                None => abs.push_str("NA\tNA\t"),
            }
        }

        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.6}\t{}{}\t{}\t{}\t{}\t{:.6}",
            site.chr,
            site.position + 1,
            m.gene_name,
            m.transcript_id,
            rel_location,
            abs,
            m.utr5_size,
            m.cds_size,
            m.utr3_size,
            m.strand,
            scale.rescale(a.region, rel_location - a.region as f64)
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Whole-gene bodies of non-coding genes, which have no UTR/CDS split.
/// One non-coding gene's merged body, in the same shape the coding path uses:
/// a locus with its intervals, so the index builder clones the sequence name
/// once per gene rather than once per interval.
struct NonCodingBody {
    seqname: Box<str>,
    strand: Strand,
    intervals: Vec<(i64, i64)>,
}

fn non_coding_bodies(records: &[GffRecord]) -> Vec<NonCodingBody> {
    // Keyed on gene AND sequence name: `parse_ensembl_id` drops the `_PAR_Y`
    // suffix, so keying on the id alone would fuse the chrX and chrY copies of
    // a pseudoautosomal gene into one cross-chromosome body.
    let mut by_gene: FxHashMap<(GeneId, Box<str>), NonCodingBody> = FxHashMap::default();
    for rec in records.iter() {
        if rec.gene_type == GeneType::CodingGene
            || rec.feature_type != FeatureType::Gene
            || rec.stop < rec.start
        {
            continue;
        }
        by_gene
            .entry((rec.gene_id.clone(), rec.seqname.clone()))
            .or_insert_with(|| NonCodingBody {
                seqname: rec.seqname.clone(),
                strand: rec.strand,
                intervals: Vec::new(),
            })
            .intervals
            .push((rec.start, rec.stop));
    }
    by_gene
        .into_values()
        .map(|mut b| {
            merge_intervals(&mut b.intervals);
            b
        })
        .collect()
}

pub fn run_metagene(args: &MetageneArgs) -> anyhow::Result<()> {
    let sites = read_sites(&args.site_file)?;
    let records = read_gff_record_vec(&args.gff_file)?;

    let models = build_transcript_models(&records);
    let n_models = models.len();
    let models = match args.isoforms {
        IsoformPolicy::Longest => elect_longest_isoform(models),
        IsoformPolicy::All => models,
    };
    let non_coding = if args.include_non_coding {
        non_coding_bodies(&records)
    } else {
        Vec::new()
    };
    drop(records);

    info!(
        "{} coding transcripts, {} kept under --isoforms {:?}",
        n_models,
        models.len(),
        args.isoforms
    );

    let index = RegionIndex::build(&models, &non_coding);

    // Placement first, widths second: MetaPlotR's scale factors are weighted by
    // the sites themselves, so no bin width is known until every site is placed.
    let (assignments, unassigned) = assign_sites(&sites, &index);
    let scale = scale_factors(&assignments, &models)?;
    let grid = BinGrid::new(args.num_bins, &scale, args.include_non_coding);
    let nbins = grid.0;

    let n_rows: usize = assignments.iter().filter(|a| a.model.is_some()).count();
    info!(
        "sites {} | assigned rows {} | unassigned {} ({:.2}%)",
        sites.len(),
        n_rows,
        unassigned,
        100.0 * unassigned as f64 / sites.len().max(1) as f64
    );
    let median = scale.median();
    info!(
        "site-weighted medians 5'UTR/CDS/3'UTR = {:.1}/{:.1}/{:.1} nt; SF5 = {:.4}, SF3 = {:.4}",
        median[0], median[1], median[2], scale.utr5_sf, scale.utr3_sf
    );
    info!(
        "bins 5'UTR/CDS/3'UTR/ncRNA = {}/{}/{}/{}",
        nbins[0], nbins[1], nbins[2], nbins[3]
    );

    if let Some(path) = args.dist_measures.as_ref() {
        write_dist_measures(path, &sites, &assignments, &models, &scale)?;
        info!("wrote per-site distance table to {}", path);
    }

    let histogram = GeneFeatureHistogram::accumulate(&grid, scale, &assignments);
    histogram.to_tsv(&args.output)?;
    info!("wrote metagene histogram to {}", args.output);

    if args.print_histogram {
        histogram.print(args.max_width);
    }

    Ok(())
}

#[cfg(test)]
mod tests;
