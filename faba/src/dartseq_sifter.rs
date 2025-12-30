use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::hypothesis_tests::{BinomTest, BinomialCounts};

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

pub struct DartSeqSifter {
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub pseudocount: usize,
    pub min_meth_cutoff: f64,
    pub max_pvalue_cutoff: f64,
    pub max_mutant_cutoff: f64,
    pub candidate_sites: Vec<MethylatedSite>,
}

impl DartSeqSifter {
    /// search over RAC patterns
    /// * first: `R=A/G`
    /// * second: `A`
    /// * third: `C->T`
    pub fn forward_sweep(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: &HashMap<i64, DnaBaseCount>,
    ) {
        for j in 2..positions.len() {
            let first = positions[j - 2];
            let third = positions[j];

            if third - first != 2 {
                continue;
            }

            let r_site = positions[j - 2];
            let m6a_site = positions[j - 1];
            let conv_site = positions[j];

            // Cache lookups to avoid redundant hash map queries
            let wt_m6a = wt_pos_to_freq.get(&m6a_site);
            let mut_m6a = mut_pos_to_freq.get(&m6a_site);
            let wt_r = wt_pos_to_freq.get(&r_site);
            let mut_r = mut_pos_to_freq.get(&r_site);
            let wt_conv = wt_pos_to_freq.get(&conv_site);
            let mut_conv = mut_pos_to_freq.get(&conv_site);

            // Check if both WT and mutant m6a site are 'A'
            let both_m6a_valid = wt_m6a
                .zip(mut_m6a)
                .map(|(wt, mt)| {
                    wt.most_frequent().0 == Dna::A && mt.most_frequent().0 == Dna::A
                })
                .unwrap_or(false);
            if !both_m6a_valid {
                continue;
            }

            // Ensure mutant's conversion site is mostly C
            if !mut_conv
                .map(|s| s.most_frequent().0 == Dna::C)
                .unwrap_or(false)
            {
                continue;
            }

            // Check if both WT and mutant R site are valid (A or G)
            let both_r_valid = wt_r
                .zip(mut_r)
                .map(|(wt, mt)| {
                    let wt_major = &wt.most_frequent().0;
                    let mt_major = &mt.most_frequent().0;
                    (wt_major == &Dna::A || wt_major == &Dna::G)
                        && (mt_major == &Dna::A || mt_major == &Dna::G)
                })
                .unwrap_or(false);

            if !both_r_valid {
                continue;
            }

            // Perform binomial test and store result if significant
            if let Some(pv) = self.binomial_test_pvalue(wt_conv, mut_conv, &Dna::C, &Dna::T) {
                if pv < self.max_pvalue_cutoff {
                    self.candidate_sites.push(MethylatedSite {
                        m6a_pos: m6a_site,
                        conversion_pos: conv_site,
                        wt_freq: wt_conv.unwrap().clone(),
                        mut_freq: mut_conv.unwrap().clone(),
                        pv,
                    });
                }
            }
        }
    }

    /// search backward CAR patterns with complement
    /// * conversion site: first `C->T <=> G->A`
    /// * m6A site: second `A <=> T`
    /// * R site: third `(G/A) <=> (C/T)`
    pub fn backward_sweep(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: &HashMap<i64, DnaBaseCount>,
    ) {
        for j in 0..(positions.len().max(2) - 2) {
            let first = positions[j];
            let third = positions[j + 2];

            if third - first != 2 {
                continue;
            }

            let r_site = positions[j + 2];
            let m6a_site = positions[j + 1];
            let conv_site = positions[j];

            // Cache lookups to avoid redundant hash map queries
            let wt_m6a = wt_pos_to_freq.get(&m6a_site);
            let mut_m6a = mut_pos_to_freq.get(&m6a_site);
            let wt_r = wt_pos_to_freq.get(&r_site);
            let mut_r = mut_pos_to_freq.get(&r_site);
            let wt_conv = wt_pos_to_freq.get(&conv_site);
            let mut_conv = mut_pos_to_freq.get(&conv_site);

            // Check if both WT and mutant m6a site are 'T' (complement of A)
            let both_m6a_valid = wt_m6a
                .zip(mut_m6a)
                .map(|(wt, mt)| {
                    wt.most_frequent().0 == Dna::T && mt.most_frequent().0 == Dna::T
                })
                .unwrap_or(false);
            if !both_m6a_valid {
                continue;
            }

            // Ensure mutant's conversion site is mostly G (complement of C)
            if !mut_conv
                .map(|s| s.most_frequent().0 == Dna::G)
                .unwrap_or(false)
            {
                continue;
            }

            // Check if both WT and mutant R site are valid (T or C, complement of A/G)
            let both_r_valid = wt_r
                .zip(mut_r)
                .map(|(wt, mt)| {
                    let wt_major = &wt.most_frequent().0;
                    let mt_major = &mt.most_frequent().0;
                    (wt_major == &Dna::T || wt_major == &Dna::C)
                        && (mt_major == &Dna::T || mt_major == &Dna::C)
                })
                .unwrap_or(false);

            if !both_r_valid {
                continue;
            }

            // Perform binomial test and store result if significant
            if let Some(pv) = self.binomial_test_pvalue(wt_conv, mut_conv, &Dna::G, &Dna::A) {
                if pv < self.max_pvalue_cutoff {
                    self.candidate_sites.push(MethylatedSite {
                        m6a_pos: m6a_site,
                        conversion_pos: conv_site,
                        wt_freq: wt_conv.unwrap().clone(),
                        mut_freq: mut_conv.unwrap().clone(),
                        pv,
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
