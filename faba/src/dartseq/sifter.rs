use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::data::util_htslib::fetch_reference_base;
use crate::hypothesis_tests::{BinomTest, BinomialCounts};
use rust_htslib::faidx;

#[derive(Clone, Debug)]
pub struct MethylatedSite {
    pub m6a_pos: i64,
    pub conversion_pos: i64,
    pub wt_freq: DnaBaseCount,
    pub mut_freq: DnaBaseCount,
    pub pv: f64,
}

// Compare only positions for ordering (ignore freq and pv)
impl PartialEq for MethylatedSite {
    fn eq(&self, other: &Self) -> bool {
        self.m6a_pos == other.m6a_pos && self.conversion_pos == other.conversion_pos
    }
}

impl Eq for MethylatedSite {}

impl PartialOrd for MethylatedSite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MethylatedSite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.m6a_pos
            .cmp(&other.m6a_pos)
            .then_with(|| self.conversion_pos.cmp(&other.conversion_pos))
    }
}

pub struct DartSeqSifter<'a> {
    pub faidx: &'a faidx::Reader,
    pub chr: &'a str,
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub pseudocount: usize,
    pub max_pvalue_cutoff: f64,
    pub check_r_site: bool,
    pub candidate_sites: Vec<MethylatedSite>,
}

impl<'a> DartSeqSifter<'a> {
    /// Validate RAC pattern in reference: R=A/G, A, C
    /// If `check_r_site` is false, only validates A and C positions
    fn validate_rac_pattern(&self, r_site: i64, m6a_site: i64, conv_site: i64) -> bool {
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();

        let r_site_valid = if self.check_r_site {
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
    /// If `check_r_site` is false, only validates G and T positions
    fn validate_gty_pattern(&self, conv_site: i64, m6a_site: i64, r_site: i64) -> bool {
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();

        let r_site_valid = if self.check_r_site {
            let ref_r = fetch_reference_base(self.faidx, self.chr, r_site)
                .ok()
                .flatten();
            matches!(ref_r, Some(Dna::C) | Some(Dna::T))
        } else {
            true
        };

        ref_conv == Some(Dna::G) && ref_m6a == Some(Dna::T) && r_site_valid
    }

    /// Search over RAC patterns
    /// * R site: `R=A/G`
    /// * m6A site: `A`
    /// * Conversion site: `C->T`
    ///
    /// If `mut_pos_to_freq` is None, skips binomial test and reports all sites matching RAC pattern
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

            // Check if positions are consecutive
            if conv_site - r_site != 2 {
                continue;
            }

            // Validate reference sequence matches RAC pattern
            if !self.validate_rac_pattern(r_site, m6a_site, conv_site) {
                continue;
            }

            // Get wild-type frequency data
            let Some(wt_conv) = wt_pos_to_freq.get(&conv_site) else {
                continue;
            };

            // If mutant data provided, perform binomial test; otherwise just record the site
            match mut_pos_to_freq {
                Some(mut_freq_map) => {
                    let Some(mut_conv) = mut_freq_map.get(&conv_site) else {
                        continue;
                    };

                    // Perform binomial test and store result if significant
                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_conv), Some(mut_conv), &Dna::C, &Dna::T)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(MethylatedSite {
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
                    // No statistical test, just record sites matching RAC pattern
                    self.candidate_sites.push(MethylatedSite {
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

    /// Search backward GTY patterns (complement of RAC)
    /// * Conversion site: `G->A`
    /// * m6A site: `T` (complement of A)
    /// * R site: `Y=C/T` (complement of R=A/G)
    ///
    /// If `mut_pos_to_freq` is None, skips binomial test and reports all sites matching GTY pattern
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

            // Check if positions are consecutive
            if r_site - conv_site != 2 {
                continue;
            }

            // Validate reference sequence matches GTY pattern
            if !self.validate_gty_pattern(conv_site, m6a_site, r_site) {
                continue;
            }

            // Get wild-type frequency data
            let Some(wt_conv) = wt_pos_to_freq.get(&conv_site) else {
                continue;
            };

            // If mutant data provided, perform binomial test; otherwise just record the site
            match mut_pos_to_freq {
                Some(mut_freq_map) => {
                    let Some(mut_conv) = mut_freq_map.get(&conv_site) else {
                        continue;
                    };

                    // Perform binomial test and store result if significant
                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_conv), Some(mut_conv), &Dna::G, &Dna::A)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(MethylatedSite {
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
                    // No statistical test, just record sites matching GTY pattern
                    self.candidate_sites.push(MethylatedSite {
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
                        // Add pseudocount to null/background distribution for regularization
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

    /// Create a temp FASTA with a single chromosome "chr1" containing `seq`,
    /// and build its .fai index via rust_htslib.
    fn create_test_fasta(seq: &str) -> (NamedTempFile, faidx::Reader) {
        let mut f = NamedTempFile::with_suffix(".fa").unwrap();
        writeln!(f, ">chr1").unwrap();
        writeln!(f, "{}", seq).unwrap();
        f.flush().unwrap();

        let path = f.path().to_str().unwrap().to_string();
        let reader = faidx::Reader::from_path(&path).unwrap();
        (f, reader)
    }

    /// Build a HashMap<i64, DnaBaseCount> from (pos, base, count) tuples.
    fn build_freq_map(entries: &[(i64, Dna, usize)]) -> HashMap<i64, DnaBaseCount> {
        let mut map = HashMap::default();
        for &(pos, base, count) in entries {
            let freq: &mut DnaBaseCount = map.entry(pos).or_default();
            freq.add(Some(&base), count);
        }
        map
    }

    fn make_sifter<'a>(faidx: &'a faidx::Reader) -> DartSeqSifter<'a> {
        DartSeqSifter {
            faidx,
            chr: "chr1",
            min_coverage: 10,
            min_conversion: 5,
            pseudocount: 1,
            max_pvalue_cutoff: 0.05,
            check_r_site: true,
            candidate_sites: Vec::new(),
        }
    }

    // -- Forward sweep tests --

    #[test]
    fn test_forward_sweep_discovers_rac_site() {
        //                  0123456789012
        let seq = "NNNNNNNNGAC";
        // positions:       8=G(R), 9=A(m6A), 10=C(conv)
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_sifter(&reader);

        // WT: high conversion C->T at pos 10
        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
        // MUT: low conversion (background)
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.m6a_pos, 9);
        assert_eq!(site.conversion_pos, 10);
        assert!(site.pv < 0.01, "pv should be < 0.01, got {}", site.pv);
    }

    #[test]
    fn test_forward_sweep_rejects_non_rac() {
        //                  0123456789012
        // T at pos 8 -> not R={A,G}
        let seq = "NNNNNNNNTAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_sifter(&reader);

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

    // -- Backward sweep tests --

    #[test]
    fn test_backward_sweep_discovers_gty_site() {
        //                  0123456789012
        // GTY at pos 8=G(conv), 9=T(m6A), 10=C(Y)
        let seq = "NNNNNNNNGTC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_sifter(&reader);

        // WT: high conversion G->A at pos 8
        let wt = build_freq_map(&[(8, Dna::G, 20), (8, Dna::A, 80)]);
        // MUT: low conversion
        let mt = build_freq_map(&[(8, Dna::G, 90), (8, Dna::A, 10)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.backward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.m6a_pos, 9);
        assert_eq!(site.conversion_pos, 8);
        assert!(site.pv < 0.01, "pv should be < 0.01, got {}", site.pv);
    }

    // -- Coverage / conversion thresholds --

    #[test]
    fn test_sweep_respects_min_coverage() {
        let seq = "NNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_sifter(&reader);
        sifter.min_coverage = 10;

        // Only 5 total WT reads — below min_coverage
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
        let mut sifter = make_sifter(&reader);
        sifter.min_conversion = 5;

        // Only 3 T reads (conversions) — below min_conversion=5
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
        let mut sifter = make_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);

        let positions: Vec<i64> = (0..=10).collect();
        sifter.forward_sweep(&positions, &wt, None);

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.m6a_pos, 9);
        assert_eq!(site.pv, 0.0, "No-mutant mode should report pv=0.0");
    }

    // -- Multiple sites with varying strength --

    #[test]
    fn test_multiple_sites_varying_strength() {
        // Three RAC motifs at different positions
        //  pos: 0  1  2  3  4  5  6  7  8
        //       G  A  C  G  A  C  G  A  C
        let seq = "GACGACGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_sifter(&reader);
        sifter.max_pvalue_cutoff = 0.01;

        // Site 1 (conv=2): strong — 80% conversion in WT vs 10% in MUT
        // Site 2 (conv=5): moderate — 60% conversion in WT vs 10% in MUT
        // Site 3 (conv=8): weak — 15% conversion in WT vs 10% in MUT
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

        // Strong and moderate should pass; weak should not
        assert_eq!(
            sifter.candidate_sites.len(),
            2,
            "Expected 2 significant sites, got {}",
            sifter.candidate_sites.len()
        );
        let conv_positions: Vec<i64> = sifter
            .candidate_sites
            .iter()
            .map(|s| s.conversion_pos)
            .collect();
        assert!(conv_positions.contains(&2));
        assert!(conv_positions.contains(&5));
    }

    // -- Non-consecutive positions --

    #[test]
    fn test_nonconsecutive_positions_skip() {
        let seq = "NNNNNNNNNNGAC";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_sifter(&reader);

        let wt = build_freq_map(&[(10, Dna::C, 20), (10, Dna::T, 80)]);
        let mt = build_freq_map(&[(10, Dna::C, 90), (10, Dna::T, 10)]);

        // Gap: positions [5, 9, 10] — conv_site(10) - r_site(5) = 5 != 2
        let positions = vec![5i64, 9, 10];
        sifter.forward_sweep(&positions, &wt, Some(&mt));

        assert_eq!(
            sifter.candidate_sites.len(),
            0,
            "Non-consecutive positions should yield no sites"
        );
    }
}
