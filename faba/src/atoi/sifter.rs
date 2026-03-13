use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::data::util_htslib::fetch_reference_base;
use crate::hypothesis_tests::{BinomTest, BinomialCounts};
use rust_htslib::faidx;

/// A-to-I (adenosine-to-inosine) RNA editing site detected by A->G conversion
#[derive(Clone, Debug)]
pub struct AtoISite {
    pub editing_pos: i64,
    pub wt_freq: DnaBaseCount,
    pub mut_freq: DnaBaseCount,
    pub pv: f64,
}

impl PartialEq for AtoISite {
    fn eq(&self, other: &Self) -> bool {
        self.editing_pos == other.editing_pos
    }
}

impl Eq for AtoISite {}

impl PartialOrd for AtoISite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtoISite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.editing_pos.cmp(&other.editing_pos)
    }
}

/// Sifter for detecting A-to-I editing sites (A->G on fwd, T->C on rev)
pub struct AtoISifter<'a> {
    pub faidx: &'a faidx::Reader,
    pub chr: &'a str,
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub pseudocount: usize,
    pub max_pvalue_cutoff: f64,
    pub candidate_sites: Vec<AtoISite>,
}

impl<'a> AtoISifter<'a> {
    /// Forward strand scan: ref=A, look for A->G conversion
    /// WT should have higher A->G rate than MUT (editing enriched in WT)
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
                    // failure=A (unchanged), success=G (edited)
                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_freq), Some(mut_freq), &Dna::A, &Dna::G)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(AtoISite {
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
                        self.candidate_sites.push(AtoISite {
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

    /// Backward strand scan: ref=T, look for T->C conversion (complement of A->G)
    /// WT should have higher T->C rate than MUT
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
                    // failure=T (unchanged), success=C (edited)
                    if let Some(pv) =
                        self.binomial_test_pvalue(Some(wt_freq), Some(mut_freq), &Dna::T, &Dna::C)
                    {
                        if pv < self.max_pvalue_cutoff {
                            self.candidate_sites.push(AtoISite {
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
                        self.candidate_sites.push(AtoISite {
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

    // Duplicated from DartSeqSifter — kept as independent copy per Step 6 of the plan.
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

    fn make_atoi_sifter<'a>(faidx: &'a faidx::Reader) -> AtoISifter<'a> {
        AtoISifter {
            faidx,
            chr: "chr1",
            min_coverage: 10,
            min_conversion: 5,
            pseudocount: 1,
            max_pvalue_cutoff: 0.05,
            candidate_sites: Vec::new(),
        }
    }

    #[test]
    fn test_atoi_forward_scan_discovers_a_to_g() {
        // Position 5 = A in reference
        let seq = "NNNNNANNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);

        // WT: high A->G conversion at pos 5
        let wt = build_freq_map(&[(5, Dna::A, 20), (5, Dna::G, 80)]);
        // MUT: low conversion (background)
        let mt = build_freq_map(&[(5, Dna::A, 90), (5, Dna::G, 10)]);

        let positions: Vec<i64> = (0..=9).collect();
        sifter.forward_scan(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.editing_pos, 5);
        assert!(site.pv < 0.01, "pv should be < 0.01, got {}", site.pv);
    }

    #[test]
    fn test_atoi_forward_scan_skips_non_a_ref() {
        // Position 5 = G (not A), should be skipped
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
        // Position 5 = T in reference (complement of A on rev strand)
        let seq = "NNNNNTNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);

        // WT: high T->C conversion at pos 5
        let wt = build_freq_map(&[(5, Dna::T, 20), (5, Dna::C, 80)]);
        // MUT: low conversion
        let mt = build_freq_map(&[(5, Dna::T, 90), (5, Dna::C, 10)]);

        let positions: Vec<i64> = (0..=8).collect();
        sifter.backward_scan(&positions, &wt, Some(&mt));

        assert_eq!(sifter.candidate_sites.len(), 1);
        let site = &sifter.candidate_sites[0];
        assert_eq!(site.editing_pos, 5);
        assert!(site.pv < 0.01);
    }

    #[test]
    fn test_atoi_respects_min_coverage() {
        let seq = "NNNNNANNNN";
        let (_f, reader) = create_test_fasta(seq);
        let mut sifter = make_atoi_sifter(&reader);
        sifter.min_coverage = 10;

        // Only 5 total reads — below min_coverage
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
        assert_eq!(sifter.candidate_sites[0].pv, 0.0);
    }
}
