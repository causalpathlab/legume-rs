use genomic_data::bed::Bed;
use genomic_data::gff::{build_codon_map, FeatureType, GeneId, GeneSymbol, GffRecord};
use genomic_data::sam::Strand;
use rustc_hash::FxHashMap;
use std::io::BufRead;

/// A 3'-UTR as the transcript splices it: the exons the annotation actually
/// gives, not the genomic span those exons sit in.
///
/// APA's entire estimand is *position within the 3'UTR* — proximal vs distal
/// poly-A — so the coordinate a read is placed on has to be the spliced one. A
/// `min(start)..max(stop)` span over a gene's 3'UTR records swallows the
/// introns between them and, wherever one isoform's 3'UTR begins upstream of
/// another's stop codon, reaches back into CDS; a position measured along that
/// span is not a 3'UTR position at all.
pub struct UtrRegion {
    pub chr: Box<str>,
    /// Low end of the region. A BAM **fetch** bound only, never a coordinate origin.
    ///
    /// The two constructors disagree about its frame: the GFF builder stores the
    /// 1-based lowest exon start, the BED loader the raw 0-based BED start.
    /// Nothing may read it as a genomic coordinate for that reason — use
    /// [`UtrRegion::to_bed`], which derives the fetch window from `exons`.
    pub start: i64,
    /// Highest exon stop, 1-based inclusive. A BAM **fetch** bound only, never a
    /// coordinate origin.
    pub end: i64,
    pub strand: Strand,
    pub name: Box<str>,
    /// Spliced length — the summed length of `exons`, and the denominator every
    /// spliced offset here is taken against. Both constructors keep
    /// `utr_length == sum(exons)`; the offset arithmetic reads it instead of
    /// re-summing the exons on every read.
    pub utr_length: usize,
    /// Merged, sorted, disjoint exons in 1-based inclusive coordinates.
    pub exons: Vec<(i64, i64)>,
}

/// Where a read's aligned blocks land on the spliced 3'UTR.
///
/// All three fields are in transcript orientation, so on the reverse strand
/// `x_rel` sits at a HIGHER genomic coordinate than `three_prime_rel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplicedCover {
    /// 1-based spliced offset of the 5'-most base the read covers.
    pub x_rel: i64,
    /// Spliced bases the read genuinely covers; skipped bases are not counted.
    pub len: i64,
    /// 1-based spliced offset of the 3'-most base the read covers — the poly-A
    /// end of a junction read.
    ///
    /// This is `x_rel + len - 1` exactly while the covered offsets form one
    /// unbroken run, which is every read whose skips fall inside annotated
    /// introns. A skip over exonic bases punches a hole in the run, and then
    /// the 3' end is further out than the charged length reaches.
    pub three_prime_rel: i64,
}

impl UtrRegion {
    /// The genomic superset to fetch reads over: the exons' bounding box, which
    /// is a fetch window and not a coordinate system (see
    /// `overlap_spliced_blocks`).
    ///
    /// The window is returned in htslib's fetch frame — 0-based, `start`
    /// inclusive and `stop` exclusive — because that is what the `Bed` is handed
    /// to.
    /// `exons` is 1-based inclusive, so the window opens one base below the
    /// lowest exon start; taking `start` as the bound instead left a read ending
    /// exactly on the UTR's first base outside the fetch.
    ///
    /// The frame is read off `exons` rather than `start`/`end` deliberately: the
    /// two constructors disagree about `start` — `build_utr_regions_from_gff`
    /// sets it to the 1-based lowest exon start, `load_utr_regions_from_bed` to
    /// the raw 0-based BED start — while `exons` means the same thing on both
    /// paths.
    pub fn to_bed(&self) -> Bed {
        Bed {
            chr: self.chr.clone(),
            start: self
                .exons
                .first()
                .map_or(self.start, |&(s, _)| s - 1)
                .max(0),
            stop: self.exons.last().map_or(self.end, |&(_, e)| e),
        }
    }

    /// 1-based position of `genomic_pos` along the spliced 3'UTR, read 5'->3'.
    /// `None` when the position sits in an intron or outside the exons.
    ///
    /// Offset 1 is the UTR's 5'-most base: the LOWEST genomic coordinate on the
    /// forward strand, the HIGHEST on the reverse.
    ///
    /// Read extraction goes through `overlap_spliced_blocks`, which needs a
    /// read's whole covered stretch rather than one position, so nothing on the
    /// hot path calls this. It stays because it is the direction
    /// `genomic_from_spliced` inverts: the round-trip test is what pins the two
    /// together, and a mapping with only one side written down drifts.
    #[allow(dead_code)]
    pub fn spliced_offset(&self, genomic_pos: i64) -> Option<i64> {
        let mut before = 0i64;
        for &(exon_start, exon_stop) in self.exons.iter() {
            if genomic_pos >= exon_start && genomic_pos <= exon_stop {
                let forward = before + genomic_pos - exon_start;
                return Some(match self.strand {
                    Strand::Forward => forward + 1,
                    Strand::Backward => self.utr_length as i64 - forward,
                });
            }
            before += exon_stop - exon_start + 1;
        }
        None
    }

    /// Exact inverse of [`UtrRegion::spliced_offset`]: the genomic coordinate at
    /// spliced position `offset`. `None` outside `1..=utr_length`.
    pub fn genomic_from_spliced(&self, offset: i64) -> Option<i64> {
        let spliced_len = self.utr_length as i64;
        if offset < 1 || offset > spliced_len {
            return None;
        }
        // Walk the exons in genomic order; a reverse-strand offset is counted
        // from the far end, which is the same mirror `spliced_offset` applies.
        let mut forward = match self.strand {
            Strand::Forward => offset - 1,
            Strand::Backward => spliced_len - offset,
        };
        for &(exon_start, exon_stop) in self.exons.iter() {
            let len = exon_stop - exon_start + 1;
            if forward < len {
                return Some(exon_start + forward);
            }
            forward -= len;
        }
        None
    }

    /// Intersect a read's ALIGNED BLOCKS with the exons.
    ///
    /// Each block is one uninterrupted run of reference bases the read aligns
    /// to, 1-based inclusive, ascending and disjoint — what htslib's
    /// `aligned_blocks` yields once shifted out of its 0-based half-open frame.
    /// Only bases inside both a block and an exon are charged, so the bases a
    /// read skips cost nothing whether or not the skip lines up with an
    /// annotated intron. A read's outer `reference_start..reference_end` span
    /// cannot make that distinction: it credits everything between the ends, so
    /// an `N` gap landing on exonic sequence inflates the length the model then
    /// spends on a poly-A position.
    ///
    /// `None` when no block touches an exon, which is what keeps intronic
    /// coverage out of a 3'UTR position estimate.
    pub fn overlap_spliced_blocks(
        &self,
        blocks: impl IntoIterator<Item = (i64, i64)>,
    ) -> Option<SplicedCover> {
        let mut first = i64::MAX;
        let mut last = i64::MIN;
        let mut covered = 0i64;

        // Exons are the inner loop so `blocks` is consumed once, straight off
        // the CIGAR, without collecting it.
        for (block_start, block_stop) in blocks {
            let mut before = 0i64;
            for &(exon_start, exon_stop) in self.exons.iter() {
                let lo = exon_start.max(block_start);
                let hi = exon_stop.min(block_stop);
                if lo <= hi {
                    covered += hi - lo + 1;
                    first = first.min(before + lo - exon_start);
                    last = last.max(before + hi - exon_start);
                }
                before += exon_stop - exon_start + 1;
            }
        }

        if covered <= 0 {
            return None;
        }

        // `first`/`last` count from the low genomic end; a reverse-strand
        // transcript reads the same run from the other end, which swaps which
        // of the two is 5'-most.
        let (x_rel, three_prime_rel) = match self.strand {
            Strand::Forward => (first + 1, last + 1),
            Strand::Backward => (
                self.utr_length as i64 - last,
                self.utr_length as i64 - first,
            ),
        };
        Some(SplicedCover {
            x_rel,
            len: covered,
            three_prime_rel,
        })
    }

    /// Convert a UTR-relative alpha to a genomic range `[alpha - beta, alpha + beta]`.
    ///
    /// Alpha is a spliced position, so it maps back through the exons; beta
    /// stays a raw genomic half-width because callers intersect the result with
    /// genomic coordinates (edit/SNP masks, annotation output), not with a
    /// second spliced interval.
    ///
    /// 0-based, inheriting the frame of [`UtrRegion::alpha_to_genomic`].
    pub fn alpha_to_genomic_range(&self, alpha: f64, beta: f64) -> (i64, i64) {
        let genomic_alpha = self.alpha_to_genomic(alpha);
        let start = (genomic_alpha - beta as i64).max(0);
        let stop = genomic_alpha + beta as i64;
        (start, stop)
    }

    /// A UTR-relative alpha as a **0-based** genomic coordinate, through the exons.
    ///
    /// The one place this conversion is written. It used to be spelled inline
    /// as `start + alpha` / `end - alpha` at three call sites, which was the
    /// same map only while a UTR was one contiguous block: on a spliced model a
    /// linear step lands in an intron, and the reported point could fall
    /// outside the `[start, stop]` range emitted beside it on the same row.
    ///
    /// This is also where a position LEAVES `UtrRegion`'s internal frame, so it
    /// is where the frame changes.
    /// `exons` — and every offset taken against them — is 1-based inclusive,
    /// while every genomic position faba compares or writes elsewhere is
    /// 0-based: the A-to-I / SNP masks (`record.pos()`, `DnaBaseFreqMap` gpos
    /// keys), the `primary_pos` and `pos` columns of the other site parquets,
    /// and `fetch_reference_seq`.
    /// Exporting the 1-based value made `genomic_alpha` the one position column
    /// out of step with the rest, which masked the wrong base, shifted APA's
    /// metagene assignment, and mis-registered its PWM window by 1bp.
    pub fn alpha_to_genomic(&self, alpha: f64) -> i64 {
        // Alpha is seeded by site discovery and nudged by EM, so it can drift a
        // hair past either end of the UTR; clamp rather than lose the site.
        let offset = (alpha as i64).clamp(1, (self.utr_length as i64).max(1));
        // The fallback bound comes off `exons` for the same reason `to_bed` does:
        // `start`/`end` do not share one frame across the two constructors.
        let one_based = self
            .genomic_from_spliced(offset)
            .unwrap_or(match self.strand {
                Strand::Forward => self.exons.first().map_or(self.start, |&(s, _)| s),
                Strand::Backward => self.exons.last().map_or(self.end, |&(_, e)| e),
            });
        (one_based - 1).max(0)
    }
}

////////////////////////////////
// Spliced 3'-UTR gene model   //
////////////////////////////////

/// The identity a 3'-UTR's records are pooled under: gene AND sequence name.
///
/// `parse_ensembl_id` splits a gene id at its first `.`, which drops the
/// `_PAR_Y` suffix with the version, so the X and Y copies of a pseudoautosomal
/// gene arrive under one gene id. Pooling on the id alone would splice the chrY
/// exons into the chrX region and hand the fit a UTR twice as long as either.
type Utr3Locus = (GeneId, Box<str>);

/// One locus' 3'-UTR exons and what naming its region takes.
struct Utr3Exons {
    strand: Strand,
    gene_name: GeneSymbol,
    /// Sorted, disjoint, 1-based inclusive `(start, stop)`.
    exons: Vec<(i64, i64)>,
}

/// Sort and merge intervals that overlap or abut.
///
/// Adjacency (`start == last_stop + 1`) merges too: two records meeting
/// base-to-base are one uninterrupted stretch of transcript, and leaving them
/// split would add a seam with no coordinate consequence.
fn merge_exon_intervals(intervals: &mut Vec<(i64, i64)>) {
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

/// Pool each gene's 3'-UTR **records** — not the span they sit in — and merge
/// them into the exons the transcript splices together.
///
/// GENCODE writes a generic `UTR` feature rather than `five_prime_UTR` /
/// `three_prime_UTR`, so which end a UTR record belongs to has to be decided
/// per record, by whether it sits nearer the gene's canonical start codon or
/// its stop codon. Without both codons the end cannot be named, and guessing
/// would put 5' sequence — or CDS-adjacent sequence — on the 3' track, so such
/// records are dropped. Explicit `three_prime_UTR` records are taken as given.
fn build_three_prime_utr_exons(
    records: &[GffRecord],
) -> anyhow::Result<FxHashMap<Utr3Locus, Utr3Exons>> {
    let start_codons = build_codon_map(records, &FeatureType::StartCodon)?;
    let stop_codons = build_codon_map(records, &FeatureType::StopCodon)?;

    let mut by_locus: FxHashMap<Utr3Locus, Utr3Exons> = FxHashMap::default();

    for rec in records.iter() {
        if rec.stop < rec.start {
            continue;
        }
        let is_three_prime = match rec.feature_type {
            FeatureType::ThreePrimeUTR => true,
            FeatureType::UTR => {
                let (Some(start_codon), Some(stop_codon)) = (
                    start_codons.get(&rec.gene_id),
                    stop_codons.get(&rec.gene_id),
                ) else {
                    continue;
                };
                let to_start = distance_between_regions(
                    rec.start,
                    rec.stop,
                    start_codon.start,
                    start_codon.stop,
                );
                let to_stop = distance_between_regions(
                    rec.start,
                    rec.stop,
                    stop_codon.start,
                    stop_codon.stop,
                );
                to_stop < to_start
            }
            _ => false,
        };
        if !is_three_prime {
            continue;
        }

        by_locus
            .entry((rec.gene_id.clone(), rec.seqname.clone()))
            .or_insert_with(|| Utr3Exons {
                strand: rec.strand,
                gene_name: rec.gene_name.clone(),
                exons: Vec::new(),
            })
            .exons
            .push((rec.start, rec.stop));
    }

    by_locus.retain(|_, locus| {
        merge_exon_intervals(&mut locus.exons);
        !locus.exons.is_empty()
    });

    Ok(by_locus)
}

/// Build one spliced [`UtrRegion`] per gene locus from raw GFF records.
///
/// Unfiltered — the caller applies `--min-utr-length` to the spliced length.
pub fn build_utr_regions_from_gff(records: &[GffRecord]) -> anyhow::Result<Vec<UtrRegion>> {
    let loci = build_three_prime_utr_exons(records)?;

    // A gene id present on more than one sequence is a pseudoautosomal pair
    // (see `Utr3Locus`). Both copies are real regions, but they would otherwise
    // carry the same name — hence the same `site_id` row — so the seqname joins
    // the name where, and only where, it has to tell them apart.
    let mut seqnames_per_gene: FxHashMap<&GeneId, usize> = FxHashMap::default();
    for (gene_id, _) in loci.keys() {
        *seqnames_per_gene.entry(gene_id).or_insert(0) += 1;
    }

    let mut regions: Vec<UtrRegion> = loci
        .iter()
        .map(|((gene_id, seqname), locus)| {
            let mut name = match &locus.gene_name {
                GeneSymbol::Symbol(s) => format!("{}_{}", gene_id, s),
                GeneSymbol::Missing => format!("{}", gene_id),
            };
            if seqnames_per_gene.get(gene_id).copied().unwrap_or(0) > 1 {
                name = format!("{}_{}", name, seqname);
            }
            UtrRegion {
                chr: seqname.clone(),
                // Fetch bounds: the exons' bounding box, which is a superset of
                // the region and not the coordinate the model reads positions on.
                start: locus.exons[0].0,
                end: locus.exons[locus.exons.len() - 1].1,
                strand: locus.strand,
                name: name.into(),
                utr_length: locus.exons.iter().map(|&(s, e)| (e - s + 1) as usize).sum(),
                exons: locus.exons.clone(),
            }
        })
        .collect();

    // `read_gff_record_vec` parses in parallel, so record order — and with it
    // the map order above — varies run to run. Sort so the region list handed
    // to the fit is the same list every time.
    regions.sort_by(|a, b| (&a.chr, a.start, &a.name).cmp(&(&b.chr, b.start, &b.name)));
    Ok(regions)
}

/// Load UTR regions from a BED file.
///
/// Supports multiple formats:
/// - 4-col SCAPE format: chr, start, end, strand (+/-)
/// - 6-col standard BED: chr, start, end, name, score, strand
/// - 3-col minimal: chr, start, end (assumes forward strand)
pub fn load_utr_regions_from_bed(path: &str) -> anyhow::Result<Vec<UtrRegion>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut regions = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }

        let chr: Box<str> = fields[0].into();
        let start: i64 = fields[1].parse()?;
        let end: i64 = fields[2].parse()?;

        // Detect format based on column count and content
        let (name, strand) = if fields.len() >= 6 {
            // Standard 6-col BED: chr, start, end, name, score, strand
            let name: Box<str> = fields[3].into();
            let strand = match fields[5] {
                "-" => Strand::Backward,
                _ => Strand::Forward,
            };
            (name, strand)
        } else if fields.len() == 4 {
            // Could be SCAPE format (chr, start, end, strand) or (chr, start, end, name)
            let col3 = fields[3];
            if col3 == "+" || col3 == "-" {
                // SCAPE 4-col format: strand in column 4
                let strand = if col3 == "-" {
                    Strand::Backward
                } else {
                    Strand::Forward
                };
                let name: Box<str> = format!("{}:{}-{}", chr, start, end).into();
                (name, strand)
            } else {
                // name in column 4, assume forward strand
                (col3.into(), Strand::Forward)
            }
        } else {
            // 3-col: chr, start, end
            let name: Box<str> = format!("{}:{}-{}", chr, start, end).into();
            (name, Strand::Forward)
        };

        let utr_length = (end - start) as usize;

        regions.push(UtrRegion {
            chr,
            start,
            end,
            strand,
            name,
            utr_length,
            // A BED interval carries no exon structure, so the region is one
            // unspliced block. BED is 0-based half-open and `exons` is 1-based
            // inclusive, so the first base is `start + 1` — which leaves the
            // block's length at `end - start`, exactly the `utr_length` this
            // path has always reported, and every offset unchanged.
            exons: vec![(start + 1, end)],
        });
    }

    Ok(regions)
}

#[cfg(test)]
mod tests;
