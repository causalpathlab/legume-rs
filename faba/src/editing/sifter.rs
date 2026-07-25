use crate::data::dna::Dna;
use crate::data::dna::DnaBaseCount;
use crate::data::dna_stat_map::HashMap;
use crate::data::util_htslib::{fetch_reference_base, fetch_reference_bases};
use crate::editing::{CallReason, ConversionSite};
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
    /// minimum absolute effect size `p_WT − p_MUT` (raw rates).
    pub min_delta: f32,
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

//////////////////////
// The m6A motif    //
//////////////////////

/// The DART m6A motif rule, in ONE place.
///
/// `ctx[i]` is the **conversion** base; the caller supplies at least
/// `i-2 ..= i+2` (forward reads down, reverse reads up), with `None` for
/// anything off the contig or non-ACGT. Forward matches RAC — `[AG]`, `A`, `C`
/// with the conversion at the C — and reverse its complement GTY: `G`, `T`,
/// `[CT]` with the conversion at the G. `check_r_site` false relaxes both to
/// `[ACGT]AC` / `GT[ACGT]`, which is what `--no-check-r-site` means.
///
/// Shared by site discovery ([`ConversionSifter`]) and the per-cell scan
/// ([`crate::editing::cell_activity::scan`]) because the two MUST admit the
/// same motif set: the scan decides which cells are competent for precisely the
/// sites discovery calls, so a cell judged on a narrower motif than the sites it
/// gates is judged on the wrong evidence. Written out twice, they had already
/// drifted once — the scan hardcoded the R check while discovery honoured the
/// flag.
pub fn is_m6a_motif(ctx: &[Option<Dna>], i: usize, forward: bool, check_r_site: bool) -> bool {
    let at = |k: isize| -> Option<Dna> {
        usize::try_from(i as isize + k)
            .ok()
            .and_then(|j| ctx.get(j).copied())
            .flatten()
    };
    if forward {
        at(0) == Some(Dna::C)
            && at(-1) == Some(Dna::A)
            && (!check_r_site || matches!(at(-2), Some(Dna::A) | Some(Dna::G)))
    } else {
        at(0) == Some(Dna::G)
            && at(1) == Some(Dna::T)
            && (!check_r_site || matches!(at(2), Some(Dna::C) | Some(Dna::T)))
    }
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

    /// Score a *putative* m6A site at a motif C. A site is putative on the WT
    /// sequencing evidence alone — the RAC/GTY motif (validated by the caller)
    /// plus observed WT C→U at or above the coverage / minimum-conversion floors.
    /// It is materialized here with its raw WT-vs-MUT contrast p-value and a copy
    /// of the control counts; the control-coverage, effect-size (`min_delta`)
    /// and FDR checks are the *test*, applied downstream in
    /// [`crate::editing::pipeline::find_all_conversion_sites`], where a failing
    /// site is recorded with its reason rather than dropped. The `contrast`'s
    /// `rho` still parameterizes the p-value; its other fields are read by the
    /// test pass. Returns `(p-value, cloned MUT counts)`, or `None` when the WT
    /// evidence does not clear the coverage / minimum-conversion floors.
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

        // Control counts (may be zero / thin — that is the test's concern, not
        // candidacy). A missing control ⇒ zero counts ⇒ the site is still
        // putative but will fail the downstream control-coverage check.
        let a_m = mut_conv.map_or(0, |m| m.get(Some(&alt_base))) as u64; // MUT converted
        let u_m = mut_conv.map_or(0, |m| m.get(Some(&ref_base))) as u64; // MUT unconverted

        let pv = contrast_pvalue(a_w, u_w, a_m, u_m, contrast.rho);
        Some((pv, mut_conv.cloned().unwrap_or_default()))
    }

    ////////////////////////
    // m6A (DART) methods //
    ////////////////////////

    /// Validate RAC pattern in reference: R=A/G, A, C
    fn validate_rac_pattern(&self, _r_site: i64, _m6a_site: i64, conv_site: i64) -> bool {
        self.validate_motif(conv_site, true)
    }

    /// Validate GTY pattern in reference: G, T, Y=C/T (complement of RAC)
    fn validate_gty_pattern(&self, conv_site: i64, _m6a_site: i64, _r_site: i64) -> bool {
        self.validate_motif(conv_site, false)
    }

    /// Both patterns, through the one shared rule. The triplet is derived from
    /// `conv_site` rather than passed in, because the offsets are fixed by the
    /// motif itself — handing them in separately let a caller ask about a
    /// triplet that is not one.
    ///
    /// One `fetch_reference_bases` over the window replaces up to three
    /// single-base fetches, and the window is the exact span the old code read:
    /// bases beyond it are never consulted, so a candidate at a contig edge
    /// resolves the same way it always did.
    fn validate_motif(&self, conv_site: i64, forward: bool) -> bool {
        let check_r = matches!(
            &self.mod_type,
            ModificationType::M6A {
                check_r_site: true,
                ..
            }
        );
        // Forward reads [conv-2, conv]; reverse reads [conv, conv+2].
        let (lo, hi) = if forward {
            (conv_site - 2, conv_site)
        } else {
            (conv_site, conv_site + 2)
        };
        // Only the LOW end needs clamping: `fetch_reference_bases` refuses a
        // negative start outright, so an unclamped window would drop a motif in
        // the first two bases of a contig. The high end needs nothing — htslib
        // truncates a past-the-end request and returns the short slice, which
        // the shift loop already handles (pinned by the contig-edge cases in
        // `discovery_and_the_cell_scan_both_implement_the_motif_rule`).
        let clamped_lo = lo.max(0);
        let mut ctx = [None; 3];
        if let Ok(Some(seq)) = fetch_reference_bases(self.faidx, self.chr, clamped_lo, hi) {
            let shift = (clamped_lo - lo) as usize;
            for (k, b) in seq.iter().enumerate() {
                if shift + k < ctx.len() {
                    ctx[shift + k] = *b;
                }
            }
        }
        let i = if forward { 2 } else { 0 };
        is_m6a_motif(&ctx, i, forward, check_r)
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
                    gene_pv: f32::NAN,
                    reason: CallReason::default(),
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
                    gene_pv: f32::NAN,
                    reason: CallReason::default(),
                });
            }
        }
    }

    ////////////////////////////////////////////////////////
    // A-to-I methods (single-sample, reference-anchored) //
    ////////////////////////////////////////////////////////

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
                    gene_pv: f32::NAN,
                    reason: CallReason::default(),
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
                    gene_pv: f32::NAN,
                    reason: CallReason::default(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests;
