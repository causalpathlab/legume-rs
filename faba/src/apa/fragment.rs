use crate::apa::utr_region::UtrRegion;
use crate::data::bam_io;
use crate::data::poly_a_utils::*;
use genomic_data::bed::Bed;
use genomic_data::sam::{CellBarcode, UmiBarcode};

/// Clustered fragment: one representative tuple + a count of how many
/// original `FragmentRecord`s share the same (coarsened) features.
///
/// The SCAPE likelihood `p(x_n, l_n, r_n | site_k)` depends *only* on
/// `(x, l, r, is_junction, pa_site)` — cell_barcode and umi don't enter.
/// So `(x, l, r, is_junction, pa_site)` is a **sufficient statistic** and
/// fragments sharing this tuple are indistinguishable to the model.
/// Counting them and running EM on the (≪ N) cluster representatives
/// is exact, not an approximation; the per-cluster weight `count` enters
/// the M-step and log-likelihood accumulator linearly.
#[derive(Clone)]
pub struct FragmentCluster {
    pub x: f32,
    pub l: f32,
    pub r: f32,
    pub is_junction: bool,
    /// Kept on the representative for symmetry with `FragmentRecord` and
    /// in case downstream code needs it; the SCAPE per-fragment
    /// likelihood doesn't depend on it (only site discovery does, and
    /// that runs on the un-clustered fragments).
    #[allow(dead_code)]
    pub pa_site: Option<f32>,
    /// Number of original fragments collapsed into this cluster (the EM
    /// multiplicity weight for the representative).
    pub count: u32,
}

/// Quantisation grid for fragment clustering. Bin widths chosen so the
/// representative tuple sits well inside the SCAPE Gaussian's σ_f (≈50bp
/// at default `LikelihoodParams`): a 5bp x-bin perturbs log-lik by at most
/// O((5/50)² / 2) ≈ 0.005, two orders of magnitude below the EM tolerance.
/// Setting any bin to 0 disables coarsening on that axis and keeps the
/// raw value as the key (exact dedup only).
#[derive(Clone, Copy)]
pub struct ClusterBins {
    pub x: f32,
    pub l: f32,
    pub r: f32,
    pub pa_site: f32,
}

impl Default for ClusterBins {
    fn default() -> Self {
        Self {
            x: 5.0,
            l: 10.0,
            r: 10.0,
            pa_site: 5.0,
        }
    }
}

#[inline]
fn quantise(v: f32, bin: f32) -> i32 {
    if bin <= 0.0 {
        // Disabled: fall back to nearest integer so we still get exact dedup.
        v.round() as i32
    } else {
        (v / bin).round() as i32
    }
}

/// Group fragments by their coarsened feature tuple. Returns
/// `(clusters, cluster_idx)` where `cluster_idx[n]` is the cluster the
/// nth original fragment fell into — used downstream by cell assignment
/// to look up the per-cluster posterior γ for each (cell, umi).
///
/// The representative `(x, l, r, pa_site)` for a cluster is the **mean**
/// of its members (within the bin width); cheap and slightly more
/// accurate than a bin centroid.
pub fn cluster_fragments(
    fragments: &[FragmentRecord],
    bins: ClusterBins,
) -> (Vec<FragmentCluster>, Vec<u32>) {
    use rustc_hash::FxHashMap;

    // Key: rounded tuple. pa_site uses an Option-aware key (None ≠ 0).
    // f32 isn't Hash, so we go through quantised i32 representatives.
    type Key = (i32, i32, i32, bool, Option<i32>);

    let mut acc: FxHashMap<Key, (usize, f64, f64, f64, f64, u32)> = FxHashMap::default();
    // acc[key] = (cluster_idx, sum_x, sum_l, sum_r, sum_pa, count)
    let mut cluster_idx = Vec::with_capacity(fragments.len());

    for frag in fragments {
        let key: Key = (
            quantise(frag.x, bins.x),
            quantise(frag.l, bins.l),
            quantise(frag.r, bins.r),
            frag.is_junction,
            frag.pa_site.map(|p| quantise(p, bins.pa_site)),
        );
        let next_idx = acc.len();
        let entry = acc.entry(key).or_insert((next_idx, 0.0, 0.0, 0.0, 0.0, 0));
        entry.1 += frag.x as f64;
        entry.2 += frag.l as f64;
        entry.3 += frag.r as f64;
        entry.4 += frag.pa_site.unwrap_or(0.0) as f64;
        entry.5 += 1;
        cluster_idx.push(entry.0 as u32);
    }

    // Materialise clusters in insertion order so `cluster_idx` indexing
    // matches the Vec layout. acc's idx counter was assigned in
    // insertion order, so collect-then-sort-by-idx reproduces that.
    let mut clusters: Vec<(usize, FragmentCluster)> = acc
        .into_iter()
        .map(|(key, (idx, sx, sl, sr, spa, count))| {
            let c = count as f64;
            (
                idx,
                FragmentCluster {
                    x: (sx / c) as f32,
                    l: (sl / c) as f32,
                    r: (sr / c) as f32,
                    is_junction: key.3,
                    pa_site: if key.4.is_some() {
                        Some((spa / c) as f32)
                    } else {
                        None
                    },
                    count,
                },
            )
        })
        .collect();
    clusters.sort_by_key(|(idx, _)| *idx);
    let clusters = clusters.into_iter().map(|(_, c)| c).collect();

    (clusters, cluster_idx)
}

/// Parameters for polyA tail filtering and internal priming detection.
pub struct PolyAFilterParams {
    /// Minimum soft-clipped tail length to call a junction read
    pub min_tail: usize,
    /// Maximum number of non-A/T bases allowed in the tail
    pub max_non_at: usize,
    /// Window size for internal priming check
    pub internal_prime_window: usize,
    /// Minimum A/T count in window to flag internal priming
    pub internal_prime_count: usize,
}

/// A fragment extracted from a BAM record within a UTR region.
/// Following SCAPE notation: x = UTR break position, l = aligned length, r = polyA length.
pub struct FragmentRecord {
    /// UTR-relative start position (1-based, from UTR start)
    pub x: f32,
    /// Aligned length within the UTR
    pub l: f32,
    /// PolyA tail length (from soft-clip or DT tag)
    pub r: f32,
    /// Whether this is a junction read (covers pA site boundary)
    pub is_junction: bool,
    /// Junction-detected pA site position (UTR-relative, if junction read)
    pub pa_site: Option<f32>,
    /// Cell barcode
    pub cell_barcode: CellBarcode,
    /// UMI
    pub umi: UmiBarcode,
}

/// Extract fragment records from a BAM file for a given UTR region.
/// SE-only implementation. Returns fragments within the UTR boundaries.
/// Reuses a per-thread `BamReaderCache` so callers inside `par_iter` don't
/// re-open the BAM on every UTR.
pub fn extract_fragments_cached(
    cache: &mut bam_io::BamReaderCache,
    bam_file: &str,
    utr: &UtrRegion,
    cb_tag: &[u8],
    umi_tag: &[u8],
    polya: &PolyAFilterParams,
    min_mapping_quality: u8,
) -> anyhow::Result<Vec<FragmentRecord>> {
    let mut fragments = Vec::new();

    // Try fetching with the original chr name; if empty, try alternate (chr1 <-> 1)
    let bed = utr.to_bed();
    let result = try_fetch_region_cached(cache, bam_file, &bed);
    let bed = match result {
        Some(b) => b,
        None => {
            let alt_chr = alternate_chr_name(&utr.chr);
            Bed {
                chr: alt_chr.into(),
                start: utr.start,
                stop: utr.end,
            }
        }
    };

    bam_io::for_each_record_in_region_cached(cache, bam_file, &bed, |_chr, rec| {
        // Skip unmapped and low mapping-quality reads (consistent with the
        // editing/SNP modalities, which gate reads at min_mapping_quality).
        if rec.is_unmapped() || rec.mapq() < min_mapping_quality {
            return;
        }

        // Extract cell barcode and UMI
        let cell_barcode = bam_io::extract_cell_barcode(rec, cb_tag);
        let umi = bam_io::extract_umi(rec, umi_tag);

        // Every test below is a walk over the same CIGAR, so unpack it once and
        // lend it out. `Record::cigar()` allocates a fresh `Vec<Cigar>` per call
        // — and `rec.aligned_blocks()` calls it internally — so asking the
        // record for its CIGAR three or four times is three or four
        // malloc+frees on every read of every 3'UTR of every BAM.
        let cigar = rec.cigar();

        // Determine if this is a valid junction read (poly-A/T tail passes all checks)
        let poly_tail_len = get_polya_tail_length(&cigar, rec.is_reverse());
        let is_valid_junction = if poly_tail_len >= polya.min_tail as i64 {
            let at_count = count_a_or_t_bases_in_tail_from_cigar(&cigar, rec);
            let non_at = poly_tail_len as usize - at_count;
            non_at <= polya.max_non_at
                && !check_internal_prime_from_cigar(
                    &cigar,
                    rec,
                    polya.internal_prime_window,
                    polya.internal_prime_count,
                )
        } else {
            false
        };

        // Place the read on the SPLICED 3'UTR, block by block. The fetch window
        // is the exons' bounding box, so it also returns reads sitting in the
        // UTR's introns: those touch no exon and drop out here. Passing the
        // read's outer span instead would charge it for the bases under an `N`
        // gap whenever the gap does not coincide with an annotated intron.
        // `aligned_blocks` is 0-based half-open, so a block `[s, e)` is the
        // 1-based inclusive `(s + 1, e)` that `exons` is written in.
        let blocks = aligned_blocks(&cigar, rec.pos()).map(|[s, e]| (s + 1, e));
        let Some(cover) = utr.overlap_spliced_blocks(blocks) else {
            return;
        };

        let r = if is_valid_junction {
            poly_tail_len as f32
        } else {
            0.0
        };

        // The pA site a junction read reports is its 3'-most aligned base as a
        // spliced offset. That is `x + l - 1` only while the read's covered
        // offsets run unbroken, so take the end the blocks actually reached.
        let pa_pos = if is_valid_junction {
            Some(cover.three_prime_rel as f32)
        } else {
            None
        };

        fragments.push(FragmentRecord {
            x: cover.x_rel as f32,
            l: cover.len as f32,
            r,
            is_junction: pa_pos.is_some(),
            pa_site: pa_pos,
            cell_barcode,
            umi,
        });
    })?;

    Ok(fragments)
}

/// Try to fetch from BAM with given Bed. Returns Some(bed) if the chr exists, None otherwise.
fn try_fetch_region_cached(
    cache: &mut bam_io::BamReaderCache,
    bam_file: &str,
    bed: &Bed,
) -> Option<Bed> {
    let reader = cache.get_or_open(bam_file).ok()?;
    match reader.fetch((bed.chr.as_ref(), bed.start, bed.stop)) {
        Ok(_) => Some(bed.clone()),
        Err(_) => None,
    }
}

/// Generate alternate chromosome name: "chr1" <-> "1", "chrX" <-> "X", etc.
fn alternate_chr_name(chr: &str) -> String {
    if let Some(stripped) = chr.strip_prefix("chr") {
        stripped.to_string()
    } else {
        format!("chr{}", chr)
    }
}
