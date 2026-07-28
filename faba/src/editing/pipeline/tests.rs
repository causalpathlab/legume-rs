use super::{m6a_effect_reason, partition_by_site};
use crate::data::dna::{Dna, DnaBaseCount};
use crate::editing::sifter::M6aContrast;
use crate::editing::{CallReason, ConversionSite};
use dashmap::DashMap;
use genomic_data::gff::{FeatureType, GeneId, GeneSymbol, GeneType, GffRecord, GffRecordMap};
use genomic_data::sam::Strand;

fn base_count(entries: &[(Dna, usize)]) -> DnaBaseCount {
    let mut c = DnaBaseCount::new();
    for (b, n) in entries {
        c.add(Some(b), *n);
    }
    c
}

/// A forward-strand m6A candidate: converted reads are T, unconverted are C.
fn fwd_site(
    pos: i64,
    wt_t: usize,
    wt_c: usize,
    mu_t: usize,
    mu_c: usize,
    pv: f32,
) -> ConversionSite {
    ConversionSite::M6A {
        m6a_pos: pos,
        conversion_pos: pos + 1,
        wt_freq: base_count(&[(Dna::T, wt_t), (Dna::C, wt_c)]),
        mut_freq: base_count(&[(Dna::T, mu_t), (Dna::C, mu_c)]),
        pv,
        reason: CallReason::default(),
    }
}

/// A reverse-strand m6A candidate: converted reads are A, unconverted are G.
fn rev_site(
    pos: i64,
    wt_a: usize,
    wt_g: usize,
    mu_a: usize,
    mu_g: usize,
    pv: f32,
) -> ConversionSite {
    ConversionSite::M6A {
        m6a_pos: pos,
        conversion_pos: pos - 1,
        wt_freq: base_count(&[(Dna::A, wt_a), (Dna::G, wt_g)]),
        mut_freq: base_count(&[(Dna::A, mu_a), (Dna::G, mu_g)]),
        pv,
        reason: CallReason::default(),
    }
}

fn gene_id(name: &str) -> GeneId {
    GeneId::Ensembl(name.into())
}

fn gff(genes: &[(&str, Strand)]) -> GffRecordMap {
    let map: DashMap<GeneId, GffRecord> = DashMap::new();
    for (name, strand) in genes {
        let gid = gene_id(name);
        map.insert(
            gid.clone(),
            GffRecord {
                seqname: "chr1".into(),
                feature_type: FeatureType::Gene,
                start: 1,
                stop: 10_000,
                strand: *strand,
                gene_id: gid,
                gene_name: GeneSymbol::Symbol((*name).into()),
                gene_type: GeneType::CodingGene,
            },
        );
    }
    GffRecordMap::from_map(map)
}

fn contrast(min_control_coverage: usize) -> M6aContrast {
    M6aContrast {
        min_control_coverage,
        // The shipped default, so these tests exercise what users get.
        min_log_odds: 1e-4,
    }
}

/// Run one site through the whole test pass and report the verdict it earned.
/// The p-value is derived from the same 2×2 the counts describe, so a caller
/// cannot accidentally pair a table with someone else's significance.
fn verdict_for(a_w: u64, u_w: u64, a_m: u64, u_m: u64, c: M6aContrast) -> CallReason {
    let gm = gff(&[("G", Strand::Forward)]);
    let sites: DashMap<GeneId, Vec<ConversionSite>> = DashMap::new();
    let pv = faba::hypothesis_tests::contrast_pvalue(a_w, u_w, a_m, u_m);
    sites.insert(
        gene_id("G"),
        vec![fwd_site(
            100,
            a_w as usize,
            u_w as usize,
            a_m as usize,
            u_m as usize,
            pv,
        )],
    );
    let discovered = partition_by_site(sites, &gm, 0.05, Some(c));
    let found = discovered
        .selected
        .get(&gene_id("G"))
        .or_else(|| discovered.rejected.get(&gene_id("G")))
        .expect("the site lands in exactly one of the two maps");
    let reason = found[0].reason();
    drop(found);
    reason
}

/// The site is the only test unit, and this is why. `FOCAL` is a scaled-down
/// MYC: one fully methylated C among four putative-but-quiet ones. Per site the
/// focal C is called and the quiet ones are rejected on their own evidence; the
/// pooled 2x2 the deleted gene-level mode would have built is checked here too,
/// and it now clears both guards -- so pooling would emit FIVE calls where the
/// site test emits one.
/// `WEAK_P` clears the guards but not the p-value cutoff, `HOT_REV`
/// keeps the reverse-strand A/G branch of `contrast_counts` honest (read as T/C
/// it would see an empty 2x2), and `COLD` is a genomic C/T variant.
#[test]
fn site_level_attributes_the_call_to_the_c_that_carries_it() {
    let gm = gff(&[
        ("FOCAL", Strand::Forward),
        ("WEAK_P", Strand::Forward),
        ("HOT_REV", Strand::Backward),
        ("COLD", Strand::Forward),
    ]);

    let sites: DashMap<GeneId, Vec<ConversionSite>> = DashMap::new();
    sites.insert(
        gene_id("FOCAL"),
        vec![
            // The focal C: 100% signal vs 0% control, and significant.
            fwd_site(100, 20, 0, 0, 50, 1e-4),
            // Four quiet C's, deep but flat: 1% in both arms.
            fwd_site(200, 1, 99, 1, 99, 0.5),
            fwd_site(300, 1, 99, 1, 99, 0.5),
            fwd_site(400, 1, 99, 1, 99, 0.5),
            fwd_site(500, 1, 99, 1, 99, 0.5),
        ],
    );
    sites.insert(gene_id("WEAK_P"), vec![fwd_site(600, 20, 0, 0, 50, 0.9)]);
    sites.insert(gene_id("HOT_REV"), vec![rev_site(700, 20, 0, 0, 50, 1e-4)]);
    sites.insert(gene_id("COLD"), vec![fwd_site(800, 80, 20, 80, 20, 1e-9)]);

    let c = contrast(10);
    let discovered = partition_by_site(sites, &gm, 0.05, Some(c));

    let selected = discovered.selected.get(&gene_id("FOCAL")).unwrap();
    assert_eq!(selected.len(), 1, "only the focal C survives");
    assert_eq!(selected[0].primary_pos(), 100);
    // One p-value per site, reported as-is: there is no separate corrected
    // statistic, which is why the parquet carries `pv` and nothing beside it.
    assert_eq!(selected[0].pv(), 1e-4);

    // What pooling FOCAL into one 2x2 would decide NOW. Summing the five sites
    // gives a_w = 20+4 = 24, u_w = 4x99 = 396, a_m = 4, u_m = 50+4x99 = 446 --
    // an odds ratio of 6.76 at a Fisher p of 3.4e-5, so pooling clears BOTH the
    // guard and the cutoff and hands one verdict to all five C's. Four of them
    // carry nothing.
    //
    // Gene pooling failed on the delta floor before and fails by false
    // ATTRIBUTION now: the direction of the error flipped, the argument did not.
    // Either way it cannot say WHICH C carries the mark, which is the whole
    // reason the gene-level mode is gone. That one site clears the guard and four
    // do not is already asserted above and below.
    assert_eq!(m6a_effect_reason(24, 396, 4, 446, &c), None);

    // Every other putative C of the same gene is kept with the reason it missed.
    // Equal rates in both arms give an odds ratio of exactly 1, whose log is
    // exactly 0.0 in f64 -- the two cross-products are the same float, so the
    // logs cancel rather than landing near zero.
    let rejected = discovered.rejected.get(&gene_id("FOCAL")).unwrap();
    let reasons: Vec<CallReason> = rejected.iter().map(|s| s.reason()).collect();
    assert_eq!(reasons, vec![CallReason::OddsRatio; 4]);

    assert_eq!(
        discovered.rejected.get(&gene_id("WEAK_P")).unwrap()[0].reason(),
        CallReason::Pvalue
    );
    assert!(discovered.selected.contains_key(&gene_id("HOT_REV")));
    assert_eq!(
        discovered.rejected.get(&gene_id("COLD")).unwrap()[0].reason(),
        CallReason::OddsRatio
    );
    assert!(!discovered.selected.contains_key(&gene_id("COLD")));
}

#[test]
fn effect_reason_flags_each_rejection_kind() {
    let c = contrast(10);
    // Thin control (n_m = 2 < 10): cannot confirm WT-specificity.
    assert_eq!(
        m6a_effect_reason(80, 20, 1, 1, &c),
        Some(CallReason::LowControl)
    );
    // WT == MUT ⇒ odds ratio exactly 1, log exactly 0 < 1e-4.
    assert_eq!(
        m6a_effect_reason(80, 20, 80, 20, &c),
        Some(CallReason::OddsRatio)
    );
    // Odds ratio 2520/1920 = 1.3125, log 0.272 ⇒ clears the guard and goes to
    // the p-value cutoff. This IS the fold guard: the separate 1.25x rate-fold
    // gate that once sat here was measured inert (94–99% of sites passed it) and
    // is now subsumed, on odds rather than rates, at ~1.0 rather than 1.25.
    assert_eq!(m6a_effect_reason(36, 64, 30, 70, &c), None);
    // Strong WT over a clean control ⇒ passes the guards, eligible for the test.
    assert_eq!(m6a_effect_reason(80, 20, 1, 99, &c), None);
}

#[test]
fn effect_reason_zero_control_is_finite_and_rejected() {
    // Numerical-safety: with the control floor disabled a 2×2 can reach the
    // guards with n_m = 0. Both cross-products are then zero, which is the one
    // case that means "this table carries no information", so the guard reads
    // an odds ratio of 1 and rejects. The hazard the old rate rule had to dodge
    // with `.max(1)` denominators -- 0/0 = NaN making `NaN < floor` false and
    // silently KEEPING the site -- cannot arise: every branch is explicit and
    // none returns NaN.
    let c = contrast(0);
    assert_eq!(
        m6a_effect_reason(4, 96, 0, 0, &c),
        Some(CallReason::OddsRatio)
    );
}

/// The MYC case, scaled: delta 0.0056, well under the retired 0.02 floor, but an
/// odds ratio of 15.08. The old rule rejected this class at 3x the WT coverage
/// and a 4x cleaner control than the sites it kept -- 36,830 candidates, median
/// odds ratio 4.83. A real flip, not a relabel: the guard passes AND the p-value
/// (~1.2e-7) selects, where the old guard would never have let it be tested.
#[test]
fn a_small_delta_with_a_large_odds_ratio_is_now_called() {
    assert_eq!(
        verdict_for(30, 4970, 2, 4998, contrast(3)),
        CallReason::Selected
    );
}

/// A strong site whose control is clean but thin must not be rejected for
/// having no effect. This pins the decision to use the RAW cross-product: a
/// Haldane-corrected guard reads this 2×2 as log-odds −3.148, i.e. a claim that
/// the control converts 23x MORE, on the strength of three reads (pinned exactly
/// by `haldane_disagrees_in_sign_with_the_raw_guard_at_an_empty_control`). It
/// dies at the p-value either way (p = 0.982); what matters here is that it dies
/// labelled `Pvalue` ("no evidence"), not `OddsRatio` ("no effect").
#[test]
fn a_clean_thin_control_does_not_reject_a_strong_site() {
    assert_eq!(m6a_effect_reason(30, 4970, 0, 3, &contrast(3)), None);
    assert_eq!(verdict_for(30, 4970, 0, 3, contrast(3)), CallReason::Pvalue);
}

/// The low-abundance lesson, in code. One converted read of five scores
/// `delta = 0.20` -- the largest delta anywhere in these tests, four times a
/// genuinely strong deep site -- because a rate difference at tiny denominators
/// reports depth, not effect. The odds ratio passes it through (the control is
/// clean) and the exact test rejects it for what is actually wrong: five reads
/// is not evidence (p = 5/55 = 0.0909). The reported SE says the same thing,
/// which is why it ships beside the estimate.
#[test]
fn low_abundance_clears_the_guard_and_dies_at_the_p_value() {
    assert_eq!(m6a_effect_reason(1, 4, 0, 50, &contrast(3)), None);
    assert_eq!(verdict_for(1, 4, 0, 50, contrast(3)), CallReason::Pvalue);

    let (_, se) = faba::hypothesis_tests::log_odds_ratio_woolf(1, 4, 0, 50);
    assert!(se > 1.7, "the SE flags the thin evidence: {se}");
}

/// The one property worth keeping from the delta guard: a genomic C/T variant
/// converts equally in both arms, so its odds ratio is exactly 1 at ANY depth.
/// Deep coverage must not rescue it. (The balanced case is covered by
/// `effect_reason_flags_each_rejection_kind`; what this adds is the scaling.)
#[test]
fn a_genomic_variant_is_rejected_at_every_depth() {
    let c = contrast(3);
    for (a_w, u_w, a_m, u_m) in [(8u64, 2u64, 80u64, 20u64), (800, 200, 80, 20)] {
        assert_eq!(
            m6a_effect_reason(a_w, u_w, a_m, u_m, &c),
            Some(CallReason::OddsRatio),
            "({a_w},{u_w},{a_m},{u_m}) is a variant, not a modification"
        );
    }
}

/// Clearing the guard is not the same as being called. This is the MEDIAN
/// profile of the 36,830 candidates the delta rule rejected (delta 0.0032, odds
/// ratio ~4.8, WT coverage 744, control background 0.0008): the odds-ratio guard
/// lets it through, and the exact test then rejects it anyway at p = 0.15, so
/// the rejection lands on the honest reason.
///
/// This is a statement about THIS 2×2, not about the population. Measured on
/// chr19+MYC at the pre-0.12.5 floors, of the 994 putative sites the delta rule
/// would have cut, 542 are now Selected and 425 land here — so a majority of
/// delta rejections really were suppressed calls, not relabelled ones. An
/// earlier draft predicted the opposite; the data said otherwise.
#[test]
fn a_median_delta_rejection_still_dies_at_the_p_value() {
    assert_eq!(m6a_effect_reason(3, 741, 1, 1249, &contrast(3)), None);
    assert_eq!(
        verdict_for(3, 741, 1, 1249, contrast(3)),
        CallReason::Pvalue
    );
}
