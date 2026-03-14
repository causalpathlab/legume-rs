use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::data::util_htslib::fetch_reference_base;
use crate::editing::ConversionSite;
use crate::hypothesis_tests::{BinomTest, BinomialCounts};
use rust_htslib::faidx;

/// Controls which scanning logic to use
#[derive(Clone, Debug)]
pub enum ModificationType {
    /// DART-seq m6A: RAC/GTY pattern, requires triplet validation
    M6A { check_r_site: bool },
    /// A-to-I editing: single-position A->G / T->C
    AtoI,
}

/// Unified sifter for detecting base conversion sites
pub struct ConversionSifter<'a> {
    pub faidx: &'a faidx::Reader,
    pub chr: &'a str,
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub pseudocount: usize,
    pub max_pvalue_cutoff: f64,
    pub mod_type: ModificationType,
    pub candidate_sites: Vec<ConversionSite>,
}

impl<'a> ConversionSifter<'a> {
    /// Dispatch to the appropriate scan method based on modification type
    pub fn scan(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
        forward: bool,
    ) {
        match &self.mod_type {
            ModificationType::M6A { .. } => {
                if forward {
                    self.forward_sweep(positions, wt_pos_to_freq, mut_pos_to_freq);
                } else {
                    self.backward_sweep(positions, wt_pos_to_freq, mut_pos_to_freq);
                }
            }
            ModificationType::AtoI => {
                if forward {
                    self.forward_scan(positions, wt_pos_to_freq, mut_pos_to_freq);
                } else {
                    self.backward_scan(positions, wt_pos_to_freq, mut_pos_to_freq);
                }
            }
        }
    }

    // ========== m6A (DART-seq) methods ==========

    /// Validate RAC pattern in reference: R=A/G, A, C
    fn validate_rac_pattern(&self, r_site: i64, m6a_site: i64, conv_site: i64) -> bool {
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();

        let check_r = matches!(&self.mod_type, ModificationType::M6A { check_r_site: true });
        let r_site_valid = if check_r {
            let ref_r = fetch_reference_base(self.faidx, self.chr, r_site)
                .ok()
                .flatten();
            matches!(ref_r, Some(Dna::A) | Some(Dna::G))
        } else {
            true
        };

        r_site_valid && ref_m6a == Some(Dna::A) && ref_conv == Some(Dna::C)
    }

    /// Validate GTY pattern in reference: G, T, Y=C/T (complement of RAC)
    fn validate_gty_pattern(&self, conv_site: i64, m6a_site: i64, r_site: i64) -> bool {
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();

        let check_r = matches!(&self.mod_type, ModificationType::M6A { check_r_site: true });
        let r_site_valid = if check_r {
            let ref_r = fetch_reference_base(self.faidx, self.chr, r_site)
                .ok()
                .flatten();
            matches!(ref_r, Some(Dna::C) | Some(Dna::T))
        } else {
            true
        };

        ref_conv == Some(Dna::G) && ref_m6a == Some(Dna::T) && r_site_valid
    }

    /// Search over RAC patterns (forward strand m6A)
    pub fn forward_sweep(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
    ) {
        for j in 2..positions.len() {
            let r_site = positions[j - 2];
            let m6a_site = positions[j - 1];
            let conv_site = positions[j];

            if conv_site - r_site != 2 {
                continue;
            }

            if !self.validate_rac_pattern(r_site, m6a_site, conv_site) {
                continue;
            }

            let Some(wt_conv) = wt_pos_to_freq.get(&conv_site) else {
                continue;
            };

            match mut_pos_to_freq {
                Some(mut_freq_map) => {
                    let Some(mut_conv) = mut_freq_map.get(&conv_site) else {
                        continue;
                    };

                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_conv), Some(mut_conv), &Dna::C, &Dna::T)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(ConversionSite::M6A {
                                m6a_pos: m6a_site,
                                conversion_pos: conv_site,
                                wt_freq: wt_conv.clone(),
                                mut_freq: mut_conv.clone(),
                                pv,
                            });
                        }
                    }
                }
                None => {
                    self.candidate_sites.push(ConversionSite::M6A {
                        m6a_pos: m6a_site,
                        conversion_pos: conv_site,
                        wt_freq: wt_conv.clone(),
                        mut_freq: DnaBaseCount::default(),
                        pv: 0.0,
                    });
                }
            }
        }
    }

    /// Search backward GTY patterns (reverse strand m6A)
    pub fn backward_sweep(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
    ) {
        for j in 0..positions.len().saturating_sub(2) {
            let conv_site = positions[j];
            let m6a_site = positions[j + 1];
            let r_site = positions[j + 2];

            if r_site - conv_site != 2 {
                continue;
            }

            if !self.validate_gty_pattern(conv_site, m6a_site, r_site) {
                continue;
            }

            let Some(wt_conv) = wt_pos_to_freq.get(&conv_site) else {
                continue;
            };

            match mut_pos_to_freq {
                Some(mut_freq_map) => {
                    let Some(mut_conv) = mut_freq_map.get(&conv_site) else {
                        continue;
                    };

                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_conv), Some(mut_conv), &Dna::G, &Dna::A)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(ConversionSite::M6A {
                                m6a_pos: m6a_site,
                                conversion_pos: conv_site,
                                wt_freq: wt_conv.clone(),
                                mut_freq: mut_conv.clone(),
                                pv,
                            });
                        }
                    }
                }
                None => {
                    self.candidate_sites.push(ConversionSite::M6A {
                        m6a_pos: m6a_site,
                        conversion_pos: conv_site,
                        wt_freq: wt_conv.clone(),
                        mut_freq: DnaBaseCount::default(),
                        pv: 0.0,
                    });
                }
            }
        }
    }

    // ========== A-to-I methods ==========

    /// Forward strand scan: ref=A, look for A->G conversion
    pub fn forward_scan(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
    ) {
        for &pos in positions {
            let ref_base = fetch_reference_base(self.faidx, self.chr, pos)
                .ok()
                .flatten();
            if ref_base != Some(Dna::A) {
                continue;
            }

            let Some(wt_freq) = wt_pos_to_freq.get(&pos) else {
                continue;
            };

            match mut_pos_to_freq {
                Some(mut_freq_map) => {
                    let Some(mut_freq) = mut_freq_map.get(&pos) else {
                        continue;
                    };
                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_freq), Some(mut_freq), &Dna::A, &Dna::G)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(ConversionSite::AtoI {
                                editing_pos: pos,
                                wt_freq: wt_freq.clone(),
                                mut_freq: mut_freq.clone(),
                                pv,
                            });
                        }
                    }
                }
                None => {
                    let n_a = wt_freq.get(Some(&Dna::A));
                    let n_g = wt_freq.get(Some(&Dna::G));
                    if n_a + n_g >= self.min_coverage && n_g >= self.min_conversion {
                        self.candidate_sites.push(ConversionSite::AtoI {
                            editing_pos: pos,
                            wt_freq: wt_freq.clone(),
                            mut_freq: DnaBaseCount::default(),
                            pv: 0.0,
                        });
                    }
                }
            }
        }
    }

    /// Backward strand scan: ref=T, look for T->C conversion
    pub fn backward_scan(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
    ) {
        for &pos in positions {
            let ref_base = fetch_reference_base(self.faidx, self.chr, pos)
                .ok()
                .flatten();
            if ref_base != Some(Dna::T) {
                continue;
            }

            let Some(wt_freq) = wt_pos_to_freq.get(&pos) else {
                continue;
            };

            match mut_pos_to_freq {
                Some(mut_freq_map) => {
                    let Some(mut_freq) = mut_freq_map.get(&pos) else {
                        continue;
                    };
                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_freq), Some(mut_freq), &Dna::T, &Dna::C)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(ConversionSite::AtoI {
                                editing_pos: pos,
                                wt_freq: wt_freq.clone(),
                                mut_freq: mut_freq.clone(),
                                pv,
                            });
                        }
                    }
                }
                None => {
                    let n_t = wt_freq.get(Some(&Dna::T));
                    let n_c = wt_freq.get(Some(&Dna::C));
                    if n_t + n_c >= self.min_coverage && n_c >= self.min_conversion {
                        self.candidate_sites.push(ConversionSite::AtoI {
                            editing_pos: pos,
                            wt_freq: wt_freq.clone(),
                            mut_freq: DnaBaseCount::default(),
                            pv: 0.0,
                        });
                    }
                }
            }
        }
    }

    // ========== Shared binomial test ==========

    fn binomial_test_pvalue(
        &self,
        observed_count: Option<&DnaBaseCount>,
        background_count: Option<&DnaBaseCount>,
        failure_base: &Dna,
        success_base: &Dna,
    ) -> Option<f64> {
        match (observed_count, background_count) {
            (Some(wt_freq), Some(mut_freq)) => {
                let (wt_n_failure, wt_n_success) = (
                    wt_freq.get(Some(failure_base)),
                    wt_freq.get(Some(success_base)),
                );
                let (mut_n_failure, mut_n_success) = (
                    mut_freq.get(Some(failure_base)),
                    mut_freq.get(Some(success_base)),
                );
                let (ntot_wt, ntot_mut) =
                    (wt_n_failure + wt_n_success, mut_n_failure + mut_n_success);

                if ntot_wt >= self.min_coverage
                    && ntot_mut >= self.min_coverage
                    && wt_n_success >= self.min_conversion
                {
                    let pv_greater = BinomTest {
                        expected: BinomialCounts::with_pseudocount(
                            mut_n_failure,
                            mut_n_success,
                            self.pseudocount,
                        ),
                        observed: BinomialCounts::new(wt_n_failure, wt_n_success),
                    }
                    .pvalue_greater()
                    .unwrap_or(1.0);

                    Some(pv_greater)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            pseudocount: 1,
            max_pvalue_cutoff: 0.05,
            mod_type: ModificationType::M6A { check_r_site: true },
            candidate_sites: Vec::new(),
        }
    }

    fn make_atoi_sifter<'a>(faidx: &'a faidx::Reader) -> ConversionSifter<'a> {
        ConversionSifter {
            faidx,
            chr: "chr1",
            min_coverage: 10,
            min_conversion: 5,
            pseudocount: 1,
            max_pvalue_cutoff: 0.05,
            mod_type: ModificationType::AtoI,
            candidate_sites: Vec::new(),
        }
    }

    // -- m6A Forward sweep tests --

    #[test]
    fn test_forward_sweep_discovers_rac_site() {
        let seq = "NNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.primary_pos(), 9);
        assert_eq!(site.conversion_pos(), 10);
        assert!(site.pv() < 0.01, "pv should be < 0.01, got {}", site.pv());
    }

    #[test]
    fn test_forward_sweep_rejects_non_rac() {
        let seq = "NNNNNNNNTAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(
            sifter.candidate_sites.len(),
            0,
            "TAC should not match RAC pattern"
        );
    }

    // -- m6A Backward sweep tests --

    #[test]
    fn test_backward_sweep_discovers_gty_site() {
        let seq = "NNNNNNNNGTC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);

        let wt = build_freq_map(&[(8, Dna::G, 20), (8, Dna::A, 80)]);
        let mt = build_freq_map(&[(8, Dna::G, 90), (8, Dna::A, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.backward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.primary_pos(), 9);
        assert_eq!(site.conversion_pos(), 8);
        assert!(site.pv() < 0.01, "pv should be < 0.01, got {}", site.pv());
    }

    // -- Coverage / conversion thresholds --

    #[test]
    fn test_sweep_respects_min_coverage() {
        let seq = "NNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);
        sifter.min_coverage = 10;

        let wt = build_freq_map(&[(10, Dna::C, 1), (10, Dna::T, 4)]);
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(
            sifter.candidate_sites.len(),
            0,
            "Should reject when WT coverage < min_coverage"
        );
    }

    #[test]
    fn test_sweep_respects_min_conversion() {
        let seq = "NNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);
        sifter.min_conversion = 5;

        let wt = build_freq_map(&[(10, Dna::C, 97), (10, Dna::T, 3)]);
        let mt = build_freq_map(&[(10, Dna::C, 99), (10, Dna::T, 1)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(
            sifter.candidate_sites.len(),
            0,
            "Should reject when conversion count < min_conversion"
        );
    }

    // -- No-mutant mode --

    #[test]
    fn test_forward_sweep_no_mutant() {
        let seq = "NNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, None);

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.primary_pos(), 9);
        assert_eq!(site.pv(), 0.0, "No-mutant mode should report pv=0.0");
    }

    // -- Multiple sites --

    #[test]
    fn test_multiple_sites_varying_strength() {
        let seq = "GACGACGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);
        sifter.max_pvalue_cutoff = 0.01;

        let wt = build_freq_map(&[
            (2, Dna::C, 20),
            (2, Dna::T, 80),
            (5, Dna::C, 40),
            (5, Dna::T, 60),
            (8, Dna::C, 85),
            (8, Dna::T, 15),
        ]);
        let mt = build_freq_map(&[
            (2, Dna::C, 90),
            (2, Dna::T, 10),
            (5, Dna::C, 90),
            (5, Dna::T, 10),
            (8, Dna::C, 90),
            (8, Dna::T, 10),
        ]);

        let positions: Vec<i64> = (0..=8).collect();
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(
            sifter.candidate_sites.len(),
            2,
            "Expected 2 significant sites, got {}",
            sifter.candidate_sites.len()
        );
        let conv_positions: Vec<i64> = sifter
            .candidate_sites
            .iter()
            .map(|s| s.conversion_pos())
            .collect();
        assert!(conv_positions.contains(&2));
        assert!(conv_positions.contains(&5));
    }

    // -- Non-consecutive positions --

    #[test]
    fn test_nonconsecutive_positions_skip() {
        let seq = "NNNNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        let positions = vec![5i64, 9, 10];
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(
            sifter.candidate_sites.len(),
            0,
            "Non-consecutive positions should yield no sites"
        );
    }

    // -- A-to-I tests --

    #[test]
    fn test_atoi_forward_scan_discovers_a_to_g() {
        let seq = "NNNNNANNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);

        let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);
        let mt = build_freq_map(&[(5, Dna::A, 90), (5, Dna::G, 10)]);

        let positions: Vec<i64> = (0..=9).collect();
        sifter.forward_scan(&positions, &wt, Some(&mt));

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
        let mt = build_freq_map(&[(5, Dna::A, 90), (5, Dna::G, 10)]);

        let positions: Vec<i64> = (0..=8).collect();
        sifter.forward_scan(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 0);
    }

    #[test]
    fn test_atoi_backward_scan_discovers_t_to_c() {
        let seq = "NNNNNTNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);

        let wt = build_freq_map(&[(5, Dna::T, 20), (5, Dna::C, 80)]);
        let mt = build_freq_map(&[(5, Dna::T, 90), (5, Dna::C, 10)]);

        let positions: Vec<i64> = (0..=8).collect();
        sifter.backward_scan(&positions, &wt, Some(&mt));

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
        let mt = build_freq_map(&[(5, Dna::A, 90), (5, Dna::G, 10)]);

        let positions: Vec<i64> = (0..=9).collect();
        sifter.forward_scan(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 0);
    }

    #[test]
    fn test_atoi_no_mutant_mode() {
        let seq = "NNNNNANNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);

        let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);

        let positions: Vec<i64> = (0..=9).collect();
        sifter.forward_scan(&positions, &wt, None);

        assert_eq!(sifter.candidate_sites.len(), 1);
        assert_eq!(sifter.candidate_sites[0].pv(), 0.0);
    }

    // -- Dispatch test --

    #[test]
    fn test_scan_dispatch_m6a_forward() {
        let seq = "NNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_m6a_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.scan(&positions, &wt, Some(&mt), true);

        assert_eq!(sifter.candidate_sites.len(), 1);
        assert!(sifter.candidate_sites[0].is_m6a());
    }

    #[test]
    fn test_scan_dispatch_atoi_forward() {
        let seq = "NNNNNANNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);

        let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);
        let mt = build_freq_map(&[(5, Dna::A, 90), (5, Dna::G, 10)]);

        let positions: Vec<i64> = (0..=9).collect();
        sifter.scan(&positions, &wt, Some(&mt), true);

        assert_eq!(sifter.candidate_sites.len(), 1);
        assert!(sifter.candidate_sites[0].is_atoi());
    }
}
