//! The cell scan: tally per-cell conversions genome-wide, before discovery.
//!
//! This runs **first**, ahead of site discovery, which is possible because the
//! RAC/GTY motif is fixed by the *reference sequence* plus gene strand. Discovery
//! contributes only the coverage / `--min-conversion` floors, and the per-cell
//! statistic does not use them — so there is no dependency to invert, and the
//! selected set is already in hand when discovery runs. That in turn means each
//! site's WT counts are de-diluted *as they are first computed*, so nothing
//! downstream has to be recomputed or reordered.
//!
//! The scan keeps **no genomic state**. Motif positions are used purely as a
//! lookup while walking each read; the only thing accumulated is
//! `barcode -> (edited, covered)`. Memory is O(cells), not O(positions × cells),
//! which makes this lighter than the existing per-cell quantification pass rather
//! than an extra burden.

use super::ActivityTally;
use crate::common::*;
use crate::data::bam_io::{self, BamReaderCache};
use crate::data::dna::Dna;
use crate::data::util_htslib::{fetch_reference_bases, load_fasta_index};
use crate::editing::pipeline::ConversionParams;
use crate::editing::sifter::ModificationType;
use genomic_data::gff::{GffRecord, GffRecordMap};
use genomic_data::sam::{CellBarcode, Strand, UmiBarcode};
use rust_htslib::bam::ext::BamRecordExtensions;
use rustc_hash::FxHashSet;
use std::sync::Mutex;

/// The `(reference, converted)` base pair this modality reads out on the given
/// strand. m6A is a C→U deamination, seen as C→T on a forward-strand transcript
/// and G→A on a reverse-strand one; A-to-I is an A→I deamination, read as A→G
/// and T→C respectively.
pub fn channel_bases(mod_type: &ModificationType, forward: bool) -> (Dna, Dna) {
    match (mod_type, forward) {
        (ModificationType::M6A { .. }, true) => (Dna::C, Dna::T),
        (ModificationType::M6A { .. }, false) => (Dna::G, Dna::A),
        (ModificationType::AtoI, true) => (Dna::A, Dna::G),
        (ModificationType::AtoI, false) => (Dna::T, Dna::C),
    }
}

/// Keep-out distance around a motif: DART's tether reaches beyond the motif C,
/// so a nearby "background" position can still catch m6A-directed editing and
/// would smuggle signal into the activity proxy. Measured on rep1, the WT/MUT
/// background ratio is FLAT from 1 nt out to 500 nt, so leakage is not in fact
/// visible — but the guard costs nothing and keeps the proxy interpretable.
const BACKGROUND_MIN_DISTANCE: i64 = 25;

/// Sample every Nth eligible background position. The rate is an average over
/// many sites, so a deterministic stride is unbiased and bounds the per-gene set
/// (non-motif C's outnumber motif C's several-fold).
const BACKGROUND_STRIDE: usize = 4;

//////////////////////////
// Motif classification //
//////////////////////////

/// Two bases of context each side, so candidates at the span's edges still
/// classify (m6A needs them; A-to-I does not, but the uniform window costs
/// nothing). Both the motif pass and the background sweep read this one window,
/// so the reference is fetched once per gene.
fn context_window(start: i64, stop: i64) -> (i64, i64) {
    ((start - 2).max(0), stop + 2)
}

/// The motif hits inside an already-fetched window, ascending.
///
/// `lo` is the genomic coordinate of `seq[0]`; the outermost two bases at each
/// end are context for the classifier, never candidates themselves.
fn motif_hits(
    seq: &[Option<Dna>],
    lo: i64,
    start: i64,
    stop: i64,
    forward: bool,
    mod_type: &ModificationType,
) -> Vec<i64> {
    let (ref_base, _) = channel_bases(mod_type, forward);
    let mut out = Vec::new();
    for i in 2..seq.len().saturating_sub(2) {
        let pos = lo + i as i64;
        if pos < start || pos >= stop || seq[i] != Some(ref_base) {
            continue;
        }
        // Discovery's rule, not a copy of it: see `sifter::is_m6a_motif` for why
        // these two must admit the same motif set (they had already drifted once
        // on `check_r_site`).
        let hit = match mod_type {
            ModificationType::AtoI => true,
            ModificationType::M6A { check_r_site, .. } => {
                crate::editing::sifter::is_m6a_motif(seq, i, forward, *check_r_site)
            }
        };
        if hit {
            out.push(pos);
        }
    }
    out
}

/// Reference positions this modality can edit, on this gene's strand.
///
/// m6A is motif-constrained — `[AG]AC` with the edit at the C, or its reverse
/// complement `GT[CT]` with the edit at the G — mirroring
/// `sifter::validate_rac_pattern` / `validate_gty_pattern`. A-to-I is
/// reference-anchored with no motif, so every reference A (forward) or T
/// (reverse) is a candidate.
///
/// Either way this is derived from the reference alone — no coverage, no counts
/// — which is exactly why the scan can precede discovery.
// Kept for tests / callers that want candidates without the background channel;
// the scan itself goes through `candidate_and_background`.
#[allow(dead_code)]
pub fn candidate_positions(
    faidx: &rust_htslib::faidx::Reader,
    chr: &str,
    start: i64,
    stop: i64,
    forward: bool,
    mod_type: &ModificationType,
) -> FxHashSet<i64> {
    let (lo, hi) = context_window(start, stop);
    // Length-preserving: `pos = lo + i` below is a genomic coordinate, so a
    // dropped `N` would shift every position after an assembly gap.
    let Ok(Some(seq)) = fetch_reference_bases(faidx, chr, lo, hi) else {
        return FxHashSet::default();
    };
    motif_hits(&seq, lo, start, stop, forward, mod_type)
        .into_iter()
        .collect()
}

/////////////////////////////
// The dense position mask //
/////////////////////////////

/// What a reference position contributes to a read's tally.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PositionClass {
    /// Neither channel — the base is not tallied at all.
    Neither,
    /// A motif candidate: the m6A signal.
    Motif,
    /// Same chemistry, no motif: the APOBEC1 activity proxy.
    Background,
}

const NEITHER: u8 = 0;
const MOTIF: u8 = 1;
const BACKGROUND: u8 = 2;
/// Scaffolding for the sweep below: inside a motif's keep-out zone, hence
/// ineligible for background. Erased before the mask leaves the constructor.
const KEEP_OUT: u8 = 3;

/// The two position sets a gene contributes — motif candidates (the m6A signal)
/// and non-motif background (the APOBEC1 activity proxy) — as one byte per base
/// of the gene span.
///
/// Dense rather than hashed because the consumer walks reads *sequentially*: a
/// set probe per aligned base is a random miss into a table far larger than L2,
/// while an indexed byte rides cache lines the walk is already pulling in. At
/// ~2.7e10 aligned bases per BAM that difference is the pass.
pub struct GenePositions {
    /// Genomic coordinate of `mask[0]`.
    origin: i64,
    mask: Vec<u8>,
    n_motif: usize,
}

impl GenePositions {
    fn empty() -> Self {
        Self {
            origin: 0,
            mask: Vec::new(),
            n_motif: 0,
        }
    }

    /// Reads overhang the gene they were fetched for, so a position outside the
    /// span is ordinary, not an error: it simply contributes to neither channel.
    #[inline]
    pub fn class_at(&self, pos: i64) -> PositionClass {
        match usize::try_from(pos - self.origin)
            .ok()
            .and_then(|i| self.mask.get(i))
        {
            Some(&MOTIF) => PositionClass::Motif,
            Some(&BACKGROUND) => PositionClass::Background,
            _ => PositionClass::Neither,
        }
    }

    /// A gene with no motif candidate cannot contribute signal, so its BAM
    /// fetch — by far the expensive half — can be skipped outright.
    pub fn has_motif(&self) -> bool {
        self.n_motif > 0
    }
}

/// Motif candidates plus the non-motif background positions of the same channel,
/// from one reference fetch.
///
/// The background is the same chemistry (C→T forward, G→A reverse) at reference
/// bases that are *not* candidates and sit >= `BACKGROUND_MIN_DISTANCE` from any
/// — promiscuous deamination with no m6A path, hence a per-cell readout of how
/// much active fusion protein a cell carries. It is the ONLY such readout: the
/// construct is invisible to gene quantification (rat APOBEC1 does not map to
/// the human reference, and 10x 3' capture lands in the vector 3'UTR).
pub fn candidate_and_background(
    faidx: &rust_htslib::faidx::Reader,
    chr: &str,
    start: i64,
    stop: i64,
    forward: bool,
    mod_type: &ModificationType,
) -> GenePositions {
    let (lo, hi) = context_window(start, stop);
    let Ok(Some(seq)) = fetch_reference_bases(faidx, chr, lo, hi) else {
        return GenePositions::empty();
    };
    let Ok(span) = usize::try_from(stop - start) else {
        return GenePositions::empty();
    };
    let motif = motif_hits(&seq, lo, start, stop, forward, mod_type);
    let mut mask = vec![NEITHER; span];
    for &pos in &motif {
        mask[(pos - start) as usize] = MOTIF;
    }

    // The keep-out zone is painted once per motif rather than probed once per
    // base: the sweep is sequential anyway, so each zone is a contiguous write,
    // and `frontier` stops overlapping zones from repainting ground an earlier
    // motif already covered. Motif marks are all in place before this runs, so
    // a motif inside another's zone is never overwritten.
    let keep_out = BACKGROUND_MIN_DISTANCE as usize;
    let mut frontier = 0usize;
    for &pos in &motif {
        let j = (pos - start) as usize;
        let from = j.saturating_sub(keep_out).max(frontier);
        let upto = (j + keep_out + 1).min(span);
        for m in mask[from..upto].iter_mut() {
            if *m == NEITHER {
                *m = KEEP_OUT;
            }
        }
        frontier = upto;
    }

    let (ref_base, _) = channel_bases(mod_type, forward);
    let mut eligible = 0usize;
    for (i, b) in seq.iter().enumerate() {
        let pos = lo + i as i64;
        if pos < start || pos >= stop || *b != Some(ref_base) {
            continue;
        }
        let j = (pos - start) as usize;
        if mask[j] != NEITHER {
            continue;
        }
        eligible += 1;
        if eligible.is_multiple_of(BACKGROUND_STRIDE) {
            mask[j] = BACKGROUND;
        }
    }

    // The keep-out marks were only ever scaffolding; what leaves here says
    // motif, background, or nothing.
    for m in mask.iter_mut() {
        if *m == KEEP_OUT {
            *m = NEITHER;
        }
    }
    GenePositions {
        origin: start,
        mask,
        n_motif: motif.len(),
    }
}

////////////////////
// Tallying reads //
////////////////////

/// Tally one gene from one BAM into `out`.
fn tally_gene(
    cache: &mut BamReaderCache,
    bam_file: &str,
    gff_record: &GffRecord,
    pos: &GenePositions,
    forward: bool,
    params: &ConversionParams,
    out: &mut ActivityTally,
) -> anyhow::Result<()> {
    let (ref_base, alt_base) = channel_bases(&params.mod_type, forward);
    let cb_tag = params.cell_barcode_tag.as_bytes();
    let umi_tag: Option<&[u8]> = params.umi_tag.as_deref().map(str::as_bytes);
    // Per-molecule dedup scoped to the gene, matching the main scan.
    let mut umi_seen: FxHashSet<(CellBarcode, u64)> = FxHashSet::default();

    bam_io::for_each_record_in_gene_cached(
        cache,
        bam_file,
        gff_record,
        &params.gene_barcode_tag,
        params.include_missing_barcode,
        |rec| {
            if !bam_io::passes_alignment_filters(rec, params.min_mapping_quality) {
                return;
            }
            let cb = bam_io::extract_cell_barcode(rec, cb_tag);
            if !params.include_missing_barcode && cb == CellBarcode::Missing {
                return;
            }
            // A read without the UMI tag is counted, not dropped — same as
            // `dna_stat_map`, which simply cannot dedup it.
            if let Some(tag) = umi_tag {
                if let UmiBarcode::Hash(umi) = bam_io::extract_umi(rec, tag) {
                    if !umi_seen.insert((cb.clone(), umi)) {
                        return;
                    }
                }
            }
            // `Seq::index` is O(1); `as_bytes()` would decode and allocate the
            // whole read to look at one or two motif positions.
            let seq = rec.seq();
            let quals = rec.qual();
            let (mut edited, mut covered) = (0u64, 0u64);
            let (mut bg_edited, mut bg_covered) = (0u64, 0u64);
            for [rpos, gpos] in rec.aligned_pairs() {
                let class = pos.class_at(gpos);
                if class == PositionClass::Neither {
                    continue;
                }
                let r = rpos as usize;
                if r >= seq.len() || r >= quals.len() || quals[r] < params.min_base_quality {
                    continue;
                }
                let (cov, edit) = if class == PositionClass::Motif {
                    (&mut covered, &mut edited)
                } else {
                    (&mut bg_covered, &mut bg_edited)
                };
                match Dna::from_byte(seq[r].to_ascii_uppercase()) {
                    Some(b) if b == ref_base => *cov += 1,
                    Some(b) if b == alt_base => {
                        *cov += 1;
                        *edit += 1;
                    }
                    // Any other base is a different kind of mismatch (a genomic
                    // variant, a sequencing error) and belongs in neither the
                    // numerator nor the denominator of a conversion rate.
                    _ => {}
                }
            }
            if covered > 0 || bg_covered > 0 {
                let e = out.entry(cb).or_default();
                e.add(edited, covered);
                e.add_background(bg_edited, bg_covered);
            }
        },
    )
}

/// Fold one tally into another.
pub fn merge_tally(dst: &mut ActivityTally, src: ActivityTally) {
    for (cb, act) in src {
        *dst.entry(cb).or_default() += &act;
    }
}

/// Scan every gene of **one** BAM and return its per-cell tally.
///
/// One library per call, deliberately: cell barcodes are only unique within a
/// single 10x run, so pooling libraries into one tally would merge two unrelated
/// cells that happen to share a barcode — and a competent cell in one library
/// would rescue a null cell in another, reintroducing exactly the dilution this
/// stage exists to remove.
pub fn scan_cell_activity(
    gff_map: &GffRecordMap,
    params: &ConversionParams,
    bam_file: &str,
    label: &str,
) -> anyhow::Result<ActivityTally> {
    // `records()` clones out of the underlying map, so bind it before borrowing.
    let records: Vec<GffRecord> = gff_map.records();
    if records.is_empty() {
        return Ok(ActivityTally::default());
    }
    info!("cell scan ({label}): {} gene(s)", records.len());

    // Accumulate per gene and fold under one lock. Rayon's `*_init` state is
    // dropped without a hook, so a worker-local tally would be silently lost;
    // one brief lock per gene (not per read) is cheap enough.
    let merged: Mutex<ActivityTally> = Mutex::new(ActivityTally::default());
    records
        .par_iter()
        .progress_with(new_progress_bar(records.len() as u64))
        .try_for_each_init(
            || {
                let faidx = load_fasta_index(&params.genome_file)
                    .expect("cell scan: failed to load the genome index");
                (faidx, BamReaderCache::new())
            },
            |(faidx, cache), rec| -> anyhow::Result<()> {
                let forward = matches!(rec.strand, Strand::Forward);
                let pos = candidate_and_background(
                    faidx,
                    rec.seqname.as_ref(),
                    rec.start,
                    rec.stop,
                    forward,
                    &params.mod_type,
                );
                if !pos.has_motif() {
                    return Ok(());
                }
                let mut local = ActivityTally::default();
                tally_gene(cache, bam_file, rec, &pos, forward, params, &mut local)?;
                if !local.is_empty() {
                    merge_tally(&mut merged.lock().expect("cell scan: merge lock"), local);
                }
                Ok(())
            },
        )?;

    let out = merged.into_inner().expect("cell scan: merge lock");
    info!("cell scan ({label}): tallied {} cells", out.len());
    Ok(out)
}
