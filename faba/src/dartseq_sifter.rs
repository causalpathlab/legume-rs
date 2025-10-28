use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::hypothesis_tests::BinomTest;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Hash)]
pub struct MethylatedSite {
    pub m6a_pos: i64,
    pub conversion_pos: i64,
    // pub strand: Strand,
}

pub struct DartSeqSifter {
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub max_pvalue_cutoff: f64,
    pub candidate_sites: Vec<MethylatedSite>,
}

impl DartSeqSifter {
    /// search over RAC patterns
    /// * first: `R=A/G`
    /// * second: `A`
    /// * third: `C->T`
    pub fn forward_sweep(
        &mut self,
        positions: &Vec<i64>,
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

            // to make sure the second position is A
            let is_wt_m6a_a = wt_pos_to_freq
                .get(&m6a_site)
                .map(|s| s.most_frequent().0 == Dna::A)
                .unwrap_or(false);
            if !is_wt_m6a_a {
                continue;
            }

            let is_mut_m6a_a = mut_pos_to_freq
                .get(&m6a_site)
                .map(|s| s.most_frequent().0 == Dna::A)
                .unwrap_or(false);
            if !is_mut_m6a_a {
                continue;
            }

            // to ensure mutant's conversion site is mostly C
            let is_mut_c = mut_pos_to_freq
                .get(&conv_site)
                .map(|s| s.most_frequent().0 == Dna::C)
                .unwrap_or(false);

            if !is_mut_c {
                continue;
            }

            // test R = A/G for the first position
            let is_wt_r_site_valid = wt_pos_to_freq
                .get(&r_site)
                .map(|r| {
                    let major = &r.most_frequent().0;
                    major == &Dna::A || major == &Dna::G
                })
                .unwrap_or(false);

            let is_mut_r_site_valid = mut_pos_to_freq
                .get(&r_site)
                .map(|r| {
                    let major = &r.most_frequent().0;
                    major == &Dna::A || major == &Dna::G
                })
                .unwrap_or(false);

            if is_wt_r_site_valid && is_mut_r_site_valid {
                let wt_conv = wt_pos_to_freq.get(&conv_site);
                let mut_conv = mut_pos_to_freq.get(&conv_site);
                let pv = self
                    .binomial_test_pvalue(wt_conv, mut_conv, &Dna::C, &Dna::T)
                    .unwrap_or(1.0);

                if pv < self.max_pvalue_cutoff {
                    let meth = MethylatedSite {
                        // seqname: self.seqname.clone(),
                        // gene_id: self.gene_id.clone(),
                        m6a_pos: m6a_site,
                        conversion_pos: conv_site,
                        // strand: Strand::Forward,
                    };
                    self.candidate_sites.push(meth);
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
        positions: &Vec<i64>,
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

            // A <=> T (complement)
            let is_wt_m6a_t = wt_pos_to_freq
                .get(&m6a_site)
                .map(|s| s.most_frequent().0 == Dna::T)
                .unwrap_or(false);
            if !is_wt_m6a_t {
                continue;
            }

            let is_mut_m6a_t = mut_pos_to_freq
                .get(&m6a_site)
                .map(|s| s.most_frequent().0 == Dna::T)
                .unwrap_or(false);
            if !is_mut_m6a_t {
                continue;
            }

            // mutant's conversion site should be mostly G (C <=> G)
            let is_mut_g = mut_pos_to_freq
                .get(&conv_site)
                .map(|s| s.most_frequent().0 == Dna::G)
                .unwrap_or(false);

            if !is_mut_g {
                continue;
            }

            // R = A/G <=> T/C (complement)
            let is_wt_r_site_valid = wt_pos_to_freq
                .get(&r_site)
                .map(|r| {
                    let major = &r.most_frequent().0;
                    major == &Dna::T || major == &Dna::C
                })
                .unwrap_or(false);

            let is_mut_r_site_valid = mut_pos_to_freq
                .get(&r_site)
                .map(|r| {
                    let major = &r.most_frequent().0;
                    major == &Dna::T || major == &Dna::C
                })
                .unwrap_or(false);

            if is_wt_r_site_valid && is_mut_r_site_valid {
                let wt_conv = wt_pos_to_freq.get(&conv_site);
                let mut_conv = mut_pos_to_freq.get(&conv_site);
                let pv = self
                    .binomial_test_pvalue(wt_conv, mut_conv, &Dna::G, &Dna::A)
                    .unwrap_or(1.0);

                if pv < self.max_pvalue_cutoff {
                    let meth = MethylatedSite {
                        // seqname: self.seqname.clone(),
                        // gene_id: self.gene_id.clone(),
                        m6a_pos: m6a_site,
                        conversion_pos: conv_site,
                        // strand: Strand::Backward,
                    };
                    self.candidate_sites.push(meth);
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
                let (ntot, ntot_mut) = (wt_n_failure + wt_n_success, mut_n_failure + mut_n_success);

                if ntot >= self.min_coverage
                    && ntot_mut >= self.min_coverage
                    && wt_n_success >= self.min_conversion
                {
                    let pv_greater = BinomTest {
                        expected: (mut_n_failure, mut_n_success),
                        observed: (wt_n_failure, wt_n_success),
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
