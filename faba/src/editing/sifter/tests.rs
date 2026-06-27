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
                min_ratio: 2.0,
                rho: 0.02,
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
fn test_forward_sweep_no_control_no_call() {
    // Without control coverage we cannot confirm WT-specificity → no call.
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, None);

    assert_eq!(sifter.candidate_sites.len(), 0, "no control → no m6A call");
}

#[test]
fn test_forward_sweep_rejects_variant_equal_in_mut() {
    // A genomic C/T variant: high conversion in BOTH WT and MUT → effect-size
    // guard rejects it (this is the bug class the contrast fixes).
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(
        sifter.candidate_sites.len(),
        0,
        "WT≈MUT (variant) must be rejected"
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
fn test_sweep_respects_min_control_coverage() {
    // Strong WT signal but too little control coverage → cannot confirm.
    let seq = "NNNNNNNNGAC";
    let (_f, reader) = create_test_fasta(seq);
    let mut sifter = make_m6a_sifter(&reader);

    let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
    let mutc = build_freq_map(&[(10, Dna::C, 3), (10, Dna::T, 0)]); // n_m = 3 < 10

    let positions: Vec<i64> = (0..=10).collect();
    sifter.forward_sweep(&positions, &wt, Some(&mutc));

    assert_eq!(sifter.candidate_sites.len(), 0);
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
