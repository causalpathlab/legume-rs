use super::*;
use crate::data::dna::{Dna, DnaBaseCount};
use crate::data::dna_stat_map::HashMap;
use rust_htslib::faidx;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_fasta(seq: &str) -> (NamedTempFile, faidx::Reader) {
    let mut f = NamedTempFile::with_suffix(".fa").unwrap();
    writeln!(f, ">chr1").unwrap();
    writeln!(f, "{}", seq).unwrap();
    f.flush().unwrap();

    let path = f.path().to_str().unwrap().to_string();
    let reader = faidx::Reader::from_path(&path).unwrap();
    (f, reader)
}

fn build_freq_map(entries: &[(i64, Dna, usize)]) -> HashMap<i64, DnaBaseCount> {
    let mut map = HashMap::default();
    for &(pos, base, count) in entries {
        let freq: &mut DnaBaseCount = map.entry(pos).or_default();
        freq.add(Some(&base), count);
    }
    map
}

fn make_m6a_sifter<'a>(faidx: &'a faidx::Reader) -> ConversionSifter<'a> {
    ConversionSifter {
        faidx,
        chr: "chr1",
        min_coverage: 10,
        min_conversion: 5,
        error_rate: 0.01,
        overdispersion: 0.1,
        mod_type: ModificationType::M6A {
            check_r_site: true,
            contrast: M6aContrast {
                min_control_coverage: 10,
                min_delta: 0.05,
            },
        },
        candidate_sites: Vec::new(),
    }
}

fn make_atoi_sifter<'a>(faidx: &'a faidx::Reader) -> ConversionSifter<'a> {
    ConversionSifter {
        faidx,
        chr: "chr1",
        min_coverage: 10,
        min_conversion: 5,
        error_rate: 0.01,
        overdispersion: 0.1,
        mod_type: ModificationType::AtoI,
        candidate_sites: Vec::new(),
    }
}

// -- m6A forward (RAC), WT vs MUT --

#[test]
fn test_forward_sweep_discovers_rac_site() {
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    // WT: 80/100 edited at the motif C; MUT control: essentially unedited.
    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 50), (10, Dna::T, 0)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(sifter.candidate_sites.len(), 1);
    let site = &sifter.candidate_sites[0];
    assert_eq!(site.primary_pos(), 9);
    assert_eq!(site.conversion_pos(), 10);
    assert!(site.pv() < 0.01, "pv should be < 0.01, got {}", site.pv());
}

#[test]
fn test_forward_sweep_no_control_is_still_putative() {
    // Discovery is putative on the WT pattern alone; the control check is part of
    // the downstream test, so a strong WT motif with no control is a candidate
    // (it fails the control-coverage test later, not here).
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, None);

    assert_eq!(
        sifter.candidate_sites.len(),
        1,
        "putative on the WT pattern"
    );
    // No control ⇒ the contrast has nothing to reject against ⇒ non-significant.
    assert!(sifter.candidate_sites[0].pv() > 0.5);
}

#[test]
fn test_forward_sweep_variant_equal_in_mut_is_putative_but_nonsignificant() {
    // A genomic C/T variant edits equally in WT and MUT. It is still a putative
    // site (the WT shows the motif + C→U), but the contrast p-value is ~1 (no
    // WT-specificity) and the downstream delta test records it as rejected.
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(sifter.candidate_sites.len(), 1);
    assert!(
        sifter.candidate_sites[0].pv() > 0.5,
        "WT ≈ MUT ⇒ contrast non-significant"
    );
}

#[test]
fn test_forward_sweep_rejects_non_rac() {
    let seq = "NNNNNNNNTAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 50), (10, Dna::T, 0)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(
        sifter.candidate_sites.len(),
        0,
        "TAC should not match RAC pattern"
    );
}

// -- m6A backward (GTY) --

#[test]
fn test_backward_sweep_discovers_gty_site() {
    let seq = "NNNNNNNNGTC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    // Reverse strand: C->T shows as G->A on the reference.
    let wt = build_freq_map(&[(8, Dna::G, 20), (8, Dna::A, 80)]);
    let mutc = build_freq_map(&[(8, Dna::G, 50), (8, Dna::A, 0)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.backward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(sifter.candidate_sites.len(), 1);
    let site = &sifter.candidate_sites[0];
    assert_eq!(site.primary_pos(), 9);
    assert_eq!(site.conversion_pos(), 8);
    assert!(site.pv() < 0.01, "pv should be < 0.01, got {}", site.pv());
}

// -- Coverage / conversion floors --

#[test]
fn test_sweep_respects_min_coverage() {
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);
    sifter.min_coverage = 10;

    // n_ref + n_alt = 5 < 10
    let wt = build_freq_map(&[(10, Dna::C, 1), (10, Dna::T, 4)]);
    let mutc = build_freq_map(&[(10, Dna::C, 50), (10, Dna::T, 0)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(sifter.candidate_sites.len(), 0);
}

#[test]
fn test_sweep_respects_min_conversion() {
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);
    sifter.min_conversion = 5;

    // Only 3 edited reads (< 5).
    let wt = build_freq_map(&[(10, Dna::C, 97), (10, Dna::T, 3)]);
    let mutc = build_freq_map(&[(10, Dna::C, 50), (10, Dna::T, 0)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(sifter.candidate_sites.len(), 0);
}

#[test]
fn test_thin_or_absent_control_is_still_putative() {
    // Control depth no longer gates candidacy — that is the downstream
    // control-coverage test. A strong WT motif is putative whether the control is
    // thin or absent; the p-value and the recorded reason handle it later.
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 3), (10, Dna::T, 0)]); // thin control, n_m = 3
    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(
        sifter.candidate_sites.len(),
        1,
        "putative regardless of control depth"
    );
}

#[test]
fn test_multiple_sites_kept() {
    let seq = "GACGACGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[
        (2, Dna::C, 20),
        (2, Dna::T, 80),
        (5, Dna::C, 40),
        (5, Dna::T, 60),
        (8, Dna::C, 85),
        (8, Dna::T, 15),
    ]);
    let mutc = build_freq_map(&[(2, Dna::C, 50), (5, Dna::C, 50), (8, Dna::C, 50)]);

    let positions: Vec<i64> = (0..=8).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(
        sifter.candidate_sites.len(),
        3,
        "all three WT-enriched motif sites pass the contrast"
    );
}

#[test]
fn test_nonconsecutive_positions_skip() {
    let seq = "NNNNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 50), (10, Dna::T, 0)]);

    let positions = vec![5i64, 9, 10];
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(
        sifter.candidate_sites.len(),
        0,
        "Non-consecutive positions should yield no sites"
    );
}

// -- A-to-I (single-sample, unchanged) --

#[test]
fn test_atoi_forward_scan_discovers_a_to_g() {
    let seq = "NNNNNANNNN";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_atoi_sifter(&reader);

    let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);

    let positions: Vec<i64> = (0..=9).collect();
    sifter.forward_scan(&positions, &wt);

    assert_eq!(sifter.candidate_sites.len(), 1);
    let site = &sifter.candidate_sites[0];
    assert_eq!(site.primary_pos(), 5);
    assert!(site.pv() < 0.01, "pv should be < 0.01, got {}", site.pv());
}

#[test]
fn test_atoi_forward_scan_skips_non_a_ref() {
    let seq = "NNNNNGNNN";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_atoi_sifter(&reader);

    let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);

    let positions: Vec<i64> = (0..=8).collect();
    sifter.forward_scan(&positions, &wt);

    assert_eq!(sifter.candidate_sites.len(), 0);
}

#[test]
fn test_atoi_backward_scan_discovers_t_to_c() {
    let seq = "NNNNNTNNN";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_atoi_sifter(&reader);

    let wt = build_freq_map(&[(5, Dna::T, 20), (5, Dna::C, 80)]);

    let positions: Vec<i64> = (0..=8).collect();
    sifter.backward_scan(&positions, &wt);

    assert_eq!(sifter.candidate_sites.len(), 1);
    let site = &sifter.candidate_sites[0];
    assert_eq!(site.primary_pos(), 5);
    assert!(site.pv() < 0.01);
}

#[test]
fn test_atoi_respects_min_coverage() {
    let seq = "NNNNNANNNN";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_atoi_sifter(&reader);
    sifter.min_coverage = 10;

    let wt = build_freq_map(&[(5, Dna::A, 1), (5, Dna::G, 4)]);

    let positions: Vec<i64> = (0..=9).collect();
    sifter.forward_scan(&positions, &wt);

    assert_eq!(sifter.candidate_sites.len(), 0);
}

#[test]
fn test_atoi_weak_editing_is_nonsignificant() {
    let seq = "NNNNNANNNN";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_atoi_sifter(&reader);
    sifter.min_conversion = 1;

    let wt = build_freq_map(&[(5, Dna::A, 99), (5, Dna::G, 5)]);

    let positions: Vec<i64> = (0..=9).collect();
    sifter.forward_scan(&positions, &wt);

    assert_eq!(sifter.candidate_sites.len(), 1);
    assert!(
        sifter.candidate_sites[0].pv() > 0.01,
        "near-error editing should be non-significant: {}",
        sifter.candidate_sites[0].pv()
    );
}

// -- Dispatch --

#[test]
fn test_scan_dispatch_m6a_forward() {
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 50), (10, Dna::T, 0)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.scan(&positions, &wt, Some(&mutc), true);

    assert_eq!(sifter.candidate_sites.len(), 1);
    assert!(sifter.candidate_sites[0].is_m6a());
}

#[test]
fn test_scan_dispatch_atoi_forward() {
    let seq = "NNNNNANNNN";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_atoi_sifter(&reader);

    let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);

    let positions: Vec<i64> = (0..=9).collect();
    sifter.scan(&positions, &wt, None, true);

    assert_eq!(sifter.candidate_sites.len(), 1);
    assert!(sifter.candidate_sites[0].is_atoi());
}

/// Discovery and the per-cell scan must admit the SAME motif set — the scan
/// decides which cells are competent for exactly the sites discovery calls, so a
/// cell judged on a narrower motif than the sites it gates is judged on the
/// wrong evidence. They were written out twice and had already drifted once (the
/// scan hardcoded the R check while discovery honoured `--no-check-r-site`).
///
/// Both are checked against an **independent oracle** read straight off the
/// sequence literal, not against each other: now that they share
/// `is_m6a_motif`, asserting they agree is vacuous — a bug in the shared rule
/// moves both sides equally and equality still holds. (Verified: two deliberate
/// mutations slipped past the earlier cross-check version of this test.)
///
/// The reference exercises every branch — matches on both strands, each
/// single-base mismatch, the `[AG]`/`[CT]` degenerate slots, an `N`, and both
/// contig edges, where the sifter clamps a window fetch and the scan does not.
/// The leading `AC` and trailing `GT` are load-bearing: they put a real motif in
/// the first and last two bases, which is the only place the clamp is reachable.
#[test]
fn discovery_and_the_cell_scan_both_implement_the_motif_rule() {
    use crate::data::util_htslib::fetch_reference_bases;
    use crate::editing::sifter::is_m6a_motif;

    let seq = "ACCACGACTACGTCGACNACCGTCGTTGTAGTCAACGTAGGTCACGT";
    let (_tmp, faidx) = create_test_fasta(seq);
    let (bytes, n) = (seq.as_bytes(), seq.len() as i64);

    for check_r in [true, false] {
        let mut sifter = make_m6a_sifter(&faidx);
        sifter.mod_type = ModificationType::M6A {
            check_r_site: check_r,
            contrast: M6aContrast {
                min_control_coverage: 10,
                min_delta: 0.05,
            },
        };
        let bases = fetch_reference_bases(&faidx, "chr1", 0, n - 1)
            .unwrap()
            .unwrap();

        // RAC / GTY straight off the literal, in raw bytes: shares no code with
        // either implementation, so a bug in the shared rule cannot hide here.
        let at = |k: i64| -> Option<u8> { (0..n).contains(&k).then(|| bytes[k as usize]) };
        let oracle = |pos: i64, forward: bool| -> bool {
            if forward {
                at(pos) == Some(b'C')
                    && at(pos - 1) == Some(b'A')
                    && (!check_r || matches!(at(pos - 2), Some(b'A') | Some(b'G')))
            } else {
                at(pos) == Some(b'G')
                    && at(pos + 1) == Some(b'T')
                    && (!check_r || matches!(at(pos + 2), Some(b'C') | Some(b'T')))
            }
        };

        let mut hits = 0;
        for pos in 0..n {
            for forward in [true, false] {
                let want = oracle(pos, forward);
                hits += usize::from(want);
                let discovery = if forward {
                    sifter.validate_rac_pattern(pos - 2, pos - 1, pos)
                } else {
                    sifter.validate_gty_pattern(pos, pos + 1, pos + 2)
                };
                assert_eq!(
                    discovery, want,
                    "discovery disagrees at pos {pos} forward={forward} check_r={check_r}"
                );
                assert_eq!(
                    is_m6a_motif(&bases, pos as usize, forward, check_r),
                    want,
                    "the scan disagrees at pos {pos} forward={forward} check_r={check_r}"
                );
            }
        }
        // Guard the guard: a reference with no motifs would pass everything above.
        assert!(
            hits >= 8,
            "only {hits} motif hits -- the fixture stopped exercising the rule"
        );
    }
}
