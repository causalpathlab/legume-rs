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
    pub min_meth_cutoff: f64,
    pub max_pvalue_cutoff: f64,
    pub max_mutant_cutoff: f64,
    pub candidate_sites: Vec<MethylatedSite>,
}

impl<'a> DartSeqSifter<'a> {
    /// Validate RAC pattern in reference: R=A/G, A, C
    fn validate_rac_pattern(&self, r_site: i64, m6a_site: i64, conv_site: i64) -> bool {
        let ref_r = fetch_reference_base(self.faidx, self.chr, r_site)
            .ok()
            .flatten();
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();

        matches!(ref_r, Some(Dna::A) | Some(Dna::G))
            && ref_m6a == Some(Dna::A)
            && ref_conv == Some(Dna::C)
    }

    /// Validate GTY pattern in reference: G, T, Y=C/T (complement of RAC)
    fn validate_gty_pattern(&self, conv_site: i64, m6a_site: i64, r_site: i64) -> bool {
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();
        let ref_r = fetch_reference_base(self.faidx, self.chr, r_site)
            .ok()
            .flatten();

        ref_conv == Some(Dna::G)
            && ref_m6a == Some(Dna::T)
            && matches!(ref_r, Some(Dna::C) | Some(Dna::T))
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
                    if let Some(pv) = self.binomial_test_pvalue(Some(wt_conv), Some(mut_conv), &Dna::C, &Dna::T) {
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
        for j in 0..(positions.len().max(2) - 2) {
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
                    if let Some(pv) = self.binomial_test_pvalue(Some(wt_conv), Some(mut_conv), &Dna::G, &Dna::A) {
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

                let p_wt = wt_n_success as f64 / ntot_wt as f64;
                let p_mut = mut_n_success as f64 / ntot_mut as f64;

                if ntot_wt >= self.min_coverage
                    && ntot_mut >= self.min_coverage
                    && wt_n_success >= self.min_conversion
                    && p_wt >= self.min_meth_cutoff
                    && p_mut < self.max_mutant_cutoff
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
