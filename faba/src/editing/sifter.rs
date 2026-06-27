use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::data::util_htslib::fetch_reference_base;
use crate::editing::ConversionSite;
use faba::hypothesis_tests::{betabinom_pvalue_greater, contrast_pvalue};
use rust_htslib::faidx;

/// Statistical guards + dispersion for the m6A WT-vs-MUT contrast.
///
/// m6A-only: A-to-I is single-sample and carries none of this, which is why the
/// config hangs on the `M6A` arm below rather than living as flat sifter/param
/// fields that A-to-I would have to fill with never-read placeholders.
#[derive(Clone, Copy, Debug)]
pub struct M6aContrast {
    /// minimum MUT (control) coverage required to confirm WT-specificity. A site
    /// with too little control coverage cannot be shown to be control-low, so it
    /// is left uncalled rather than assumed real.
    pub min_control_coverage: usize,
    /// minimum absolute effect size `p_WT − p_MUT` (Jeffreys-regularized).
    pub min_delta: f32,
    /// minimum relative effect size `p_WT / p_MUT` (Jeffreys-regularized).
    pub min_ratio: f32,
    /// overdispersion ρ for the two-sample beta-binomial LRT contrast.
    pub rho: f64,
}

/// Controls which scanning logic to use
#[derive(Clone, Debug)]
pub enum ModificationType {
    /// DART-seq m6A: RAC/GTY pattern, triplet validation, and a WT-vs-MUT
    /// `contrast` against the pooled control.
    M6A {
        check_r_site: bool,
        contrast: M6aContrast,
    },
    /// A-to-I editing: single-position A->G / T->C
    AtoI,
}

/// Unified sifter for detecting base conversion sites.
///
/// m6A (DART) is a **two-sample** call: at each motif C the WT conversion is
/// tested against the matched MUT (catalytically-dead YTHmut) control — a
/// genomic C/T variant converts equally in both arms and is rejected. A-to-I is
/// a **single-sample** reference-anchored call (ADAR is active in the YTHmut
/// too, so there is no control to contrast against) tested against a
/// beta-binomial sequencing-error null.
pub struct ConversionSifter<'a> {
    pub faidx: &'a faidx::Reader,
    pub chr: &'a str,
    pub min_coverage: usize,
    pub min_conversion: usize,
    /// Sequencing-error rate ε for the single-sample (A-to-I) beta-binomial null.
    pub error_rate: f64,
    /// Overdispersion ρ of the single-sample (A-to-I) beta-binomial null.
    pub overdispersion: f64,
    /// Scan mode. The m6A contrast guards live on the `M6A` arm ([`M6aContrast`]);
    /// A-to-I carries none.
    pub mod_type: ModificationType,
    pub candidate_sites: Vec<ConversionSite>,
}

impl<'a> ConversionSifter<'a> {
    /// Dispatch to the appropriate scan method based on modification type.
    ///
    /// `mut_pos_to_freq` is the pooled MUT (control) frequency map; it is used
    /// only for m6A and ignored for A-to-I.
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
                    self.forward_scan(positions, wt_pos_to_freq);
                } else {
                    self.backward_scan(positions, wt_pos_to_freq);
                }
            }
        }
    }

    /// Single-sample editing p-value (A-to-I): the probability that `n_alt` of
    /// `n_ref + n_alt` ref+alt reads are sequencing noise, under a beta-binomial
    /// null (error rate ε, overdispersion ρ). Returns `None` if the site fails
    /// the coverage or minimum-conversion floor.
    fn edit_pvalue(&self, n_ref: usize, n_alt: usize) -> Option<f32> {
        let n = n_ref + n_alt;
        if n < self.min_coverage || n_alt < self.min_conversion {
            return None;
        }
        Some(betabinom_pvalue_greater(
            n_alt as u64,
            n as u64,
            self.error_rate,
            self.overdispersion,
        ))
    }

    /// Two-sample m6A call at a motif C: test WT conversion against the pooled
    /// MUT control. Returns `(p-value, cloned MUT counts)` when the site passes
    /// every guard, else `None`. Guards: WT coverage / minimum conversion,
    /// control coverage (`min_control_coverage`), and absolute/relative effect
    /// size (`min_delta`, `min_ratio`) on Jeffreys-regularized rates. The
    /// effect-size guards are what reject genomic C/T variants (equal in both
    /// arms) and constitutive editing.
    fn m6a_contrast(
        &self,
        contrast: &M6aContrast,
        wt: &DnaBaseCount,
        mut_conv: Option<&DnaBaseCount>,
        ref_base: Dna,
        alt_base: Dna,
    ) -> Option<(f32, DnaBaseCount)> {
        let a_w = wt.get(Some(&alt_base)) as u64; // WT converted (edited)
        let u_w = wt.get(Some(&ref_base)) as u64; // WT unconverted
        let n_w = a_w + u_w;
        if (n_w as usize) < self.min_coverage || (a_w as usize) < self.min_conversion {
            return None;
        }

        // Read control counts through the borrow; a missing control ⇒ zero counts
        // ⇒ fails the coverage guard below (cannot confirm WT-specificity).
        let a_m = mut_conv.map_or(0, |m| m.get(Some(&alt_base))) as u64; // MUT converted
        let u_m = mut_conv.map_or(0, |m| m.get(Some(&ref_base))) as u64; // MUT unconverted
        let n_m = a_m + u_m;
        if (n_m as usize) < contrast.min_control_coverage {
            return None;
        }

        // Jeffreys-regularized rates for the effect-size guards.
        let pw = (a_w as f64 + 0.5) / (n_w as f64 + 1.0);
        let pm = (a_m as f64 + 0.5) / (n_m as f64 + 1.0);
        if (pw - pm) < contrast.min_delta as f64 {
            return None;
        }
        if pw < (contrast.min_ratio as f64) * pm {
            return None;
        }

        let pv = contrast_pvalue(a_w, u_w, a_m, u_m, contrast.rho);
        // Clone the control counts only now that the site is kept.
        Some((pv, mut_conv.cloned().unwrap_or_default()))
    }

    // ========== m6A (DART) methods ==========

    /// Validate RAC pattern in reference: R=A/G, A, C
    fn validate_rac_pattern(&self, r_site: i64, m6a_site: i64, conv_site: i64) -> bool {
        let ref_m6a = fetch_reference_base(self.faidx, self.chr, m6a_site)
            .ok()
            .flatten();
        let ref_conv = fetch_reference_base(self.faidx, self.chr, conv_site)
            .ok()
            .flatten();

        let check_r = matches!(
            &self.mod_type,
            ModificationType::M6A {
                check_r_site: true,
                ..
            }
        );
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

        let check_r = matches!(
            &self.mod_type,
            ModificationType::M6A {
                check_r_site: true,
                ..
            }
        );
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

    /// Search over RAC patterns (forward strand m6A), WT vs MUT contrast.
    pub fn forward_sweep(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
    ) {
        let contrast = match &self.mod_type {
            ModificationType::M6A { contrast, .. } => *contrast,
            ModificationType::AtoI => return, // sweeps are only dispatched for m6A
        };
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
            let mut_conv = mut_pos_to_freq.and_then(|m| m.get(&conv_site));

            // DART edits C→T at the motif C; test WT T-fraction vs the MUT control.
            if let Some((pv, mut_freq)) =
                self.m6a_contrast(&contrast, wt_conv, mut_conv, Dna::C, Dna::T)
            {
                self.candidate_sites.push(ConversionSite::M6A {
                    m6a_pos: m6a_site,
                    conversion_pos: conv_site,
                    wt_freq: wt_conv.clone(),
                    mut_freq,
                    pv,
                    qv: 1.0,
                });
            }
        }
    }

    /// Search backward GTY patterns (reverse strand m6A), WT vs MUT contrast.
    pub fn backward_sweep(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
        mut_pos_to_freq: Option<&HashMap<i64, DnaBaseCount>>,
    ) {
        let contrast = match &self.mod_type {
            ModificationType::M6A { contrast, .. } => *contrast,
            ModificationType::AtoI => return, // sweeps are only dispatched for m6A
        };
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
            let mut_conv = mut_pos_to_freq.and_then(|m| m.get(&conv_site));

            // Reverse strand: motif C→T appears as G→A on the reference.
            if let Some((pv, mut_freq)) =
                self.m6a_contrast(&contrast, wt_conv, mut_conv, Dna::G, Dna::A)
            {
                self.candidate_sites.push(ConversionSite::M6A {
                    m6a_pos: m6a_site,
                    conversion_pos: conv_site,
                    wt_freq: wt_conv.clone(),
                    mut_freq,
                    pv,
                    qv: 1.0,
                });
            }
        }
    }

    // ========== A-to-I methods (single-sample, reference-anchored) ==========

    /// Forward strand scan: ref=A, look for A->G conversion
    pub fn forward_scan(&mut self, positions: &[i64], wt_pos_to_freq: &HashMap<i64, DnaBaseCount>) {
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

            // A-to-I reads as A→G against the reference; test the G fraction.
            let n_ref = wt_freq.get(Some(&Dna::A));
            let n_alt = wt_freq.get(Some(&Dna::G));
            if let Some(pv) = self.edit_pvalue(n_ref, n_alt) {
                self.candidate_sites.push(ConversionSite::AtoI {
                    editing_pos: pos,
                    wt_freq: wt_freq.clone(),
                    mut_freq: DnaBaseCount::default(),
                    pv,
                    qv: 1.0,
                });
            }
        }
    }

    /// Backward strand scan: ref=T, look for T->C conversion
    pub fn backward_scan(
        &mut self,
        positions: &[i64],
        wt_pos_to_freq: &HashMap<i64, DnaBaseCount>,
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

            // Reverse strand: A→G appears as T→C on the reference.
            let n_ref = wt_freq.get(Some(&Dna::T));
            let n_alt = wt_freq.get(Some(&Dna::C));
            if let Some(pv) = self.edit_pvalue(n_ref, n_alt) {
                self.candidate_sites.push(ConversionSite::AtoI {
                    editing_pos: pos,
                    wt_freq: wt_freq.clone(),
                    mut_freq: DnaBaseCount::default(),
                    pv,
                    qv: 1.0,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests;
