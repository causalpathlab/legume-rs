use crate::apa::utr_region::UtrRegion;
use crate::data::bam_io;
use crate::data::polya_utils::*;
use genomic_data::bed::Bed;
use genomic_data::sam::{CellBarcode, Strand, UmiBarcode};
use rust_htslib::bam;
use rust_htslib::bam::ext::BamRecordExtensions;
use rust_htslib::bam::record::Aux;

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
    pub x: f64,
    /// Aligned length within the UTR
    pub l: f64,
    /// PolyA tail length (from soft-clip or DT tag)
    pub r: f64,
    /// Whether this is a junction read (covers pA site boundary)
    pub is_junction: bool,
    /// Junction-detected pA site position (UTR-relative, if junction read)
    pub pa_site: Option<f64>,
    /// Cell barcode
    pub cell_barcode: CellBarcode,
    /// UMI
    pub umi: UmiBarcode,
}

/// Extract fragment records from a BAM file for a given UTR region.
/// SE-only implementation. Returns fragments within the UTR boundaries.
pub fn extract_fragments(
    bam_file: &str,
    utr: &UtrRegion,
    cb_tag: &[u8],
    umi_tag: &[u8],
    polya: &PolyAFilterParams,
) -> anyhow::Result<Vec<FragmentRecord>> {
    let mut fragments = Vec::new();

    let utr_start = utr.start;
    let utr_end = utr.end;
    let utr_len = utr.utr_length as f64;

    // Try fetching with the original chr name; if empty, try alternate (chr1 <-> 1)
    let bed = utr.to_bed();
    let result = try_fetch_region(bam_file, &bed);
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

    bam_io::for_each_record_in_region(bam_file, &bed, |_chr, rec| {
        let ref_start = rec.reference_start();
        let ref_end = rec.reference_end();

        // Skip unmapped reads
        if rec.is_unmapped() {
            return;
        }

        // Extract cell barcode and UMI
        let cell_barcode = bam_io::extract_cell_barcode(&rec, cb_tag);
        let umi = match rec.aux(umi_tag) {
            Ok(Aux::String(s)) => UmiBarcode::Barcode(s.into()),
            _ => UmiBarcode::Missing,
        };

        // Determine if this is a junction read (has poly-A/T soft-clip)
        let poly_tail_len = get_polya_tail_length(&rec);
        let is_junction = poly_tail_len >= polya.min_tail as i64;

        // Check internal priming for junction reads
        if is_junction {
            let num_at = count_a_or_t_bases_in_tail(&rec);
            let non_at = poly_tail_len as usize - num_at;
            if non_at > polya.max_non_at {
                // Treat as non-junction (poly-A tail too noisy)
            } else if check_internal_prime(
                &rec,
                polya.internal_prime_window,
                polya.internal_prime_count,
            ) {
                // Internal priming — skip as junction but still use as regular fragment
            }
        }

        // Compute UTR-relative coordinates based on strand
        let (x, l, r, pa_pos) = match utr.strand {
            Strand::Forward => {
                // Forward strand: UTR position 1 is at utr_start
                let x_abs = ref_start.max(utr_start);
                let end_abs = ref_end.min(utr_end);
                let x_rel = (x_abs - utr_start + 1) as f64;
                let l_val = (end_abs - x_abs) as f64;

                if l_val <= 0.0 || x_rel < 1.0 || x_rel > utr_len {
                    return;
                }

                let r_val = if is_junction {
                    poly_tail_len as f64
                } else {
                    0.0
                };

                // Junction read pA position: reference_end for forward
                let pa = if is_junction && poly_tail_len >= polya.min_tail as i64 {
                    let at_count = count_a_or_t_bases_in_tail(&rec);
                    let non_at = poly_tail_len as usize - at_count;
                    if non_at <= polya.max_non_at
                        && !check_internal_prime(
                            &rec,
                            polya.internal_prime_window,
                            polya.internal_prime_count,
                        )
                    {
                        Some((ref_end - utr_start) as f64)
                    } else {
                        None
                    }
                } else {
                    None
                };

                (x_rel, l_val, r_val, pa)
            }
            Strand::Backward => {
                // Reverse strand: UTR position 1 is at utr_end (UTR is reversed)
                let x_abs = ref_end.min(utr_end);
                let start_abs = ref_start.max(utr_start);
                let x_rel = (utr_end - x_abs + 1) as f64;
                let l_val = (x_abs - start_abs) as f64;

                if l_val <= 0.0 || x_rel < 1.0 || x_rel > utr_len {
                    return;
                }

                let r_val = if is_junction {
                    poly_tail_len as f64
                } else {
                    0.0
                };

                // Junction read pA position: reference_start for reverse
                let pa = if is_junction && poly_tail_len >= polya.min_tail as i64 {
                    let at_count = count_a_or_t_bases_in_tail(&rec);
                    let non_at = poly_tail_len as usize - at_count;
                    if non_at <= polya.max_non_at
                        && !check_internal_prime(
                            &rec,
                            polya.internal_prime_window,
                            polya.internal_prime_count,
                        )
                    {
                        Some((utr_end - ref_start) as f64)
                    } else {
                        None
                    }
                } else {
                    None
                };

                (x_rel, l_val, r_val, pa)
            }
        };

        fragments.push(FragmentRecord {
            x,
            l,
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
fn try_fetch_region(bam_file: &str, bed: &Bed) -> Option<Bed> {
    let index_file = bam_file.to_string() + ".bai";
    let mut reader = bam::IndexedReader::from_path_and_index(bam_file, &index_file).ok()?;
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
