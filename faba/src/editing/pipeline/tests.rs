use super::{m6a_effect_reason, m6a_site_counts, partition_by_site};
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
        min_delta: 0.05,
    }
}

/// The site is the only test unit, and this is why. `FOCAL` is a scaled-down
/// MYC: one fully methylated C among four putative-but-quiet ones. Per site the
/// focal C is called and the quiet ones are rejected on their own evidence; the
/// pooled 2x2 the deleted gene-level mode would have built is checked here too,
/// and it falls under the delta floor -- which would have condemned all five.
/// `WEAK_P` clears the effect guards but not the p-value cutoff, `HOT_REV`
/// keeps the reverse-strand A/G branch of `m6a_site_counts` honest (read as T/C
/// it would see an empty 2x2), and `COLD` is a genomic C/T variant.
#[test]
fn site_level_calls_the_focal_c_a_pooled_gene_test_would_bury() {
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
    let discovered = partition_by_site(sites, &gm, 0.05, move |site, strand| {
        let (a_w, u_w, a_m, u_m) = m6a_site_counts(site, strand);
        m6a_effect_reason(a_w, u_w, a_m, u_m, &c)
    });

    let selected = discovered.selected.get(&gene_id("FOCAL")).unwrap();
    assert_eq!(selected.len(), 1, "only the focal C survives");
    assert_eq!(selected[0].primary_pos(), 100);
    // One p-value per site, reported as-is: there is no separate corrected
    // statistic, which is why the parquet carries `pv` and nothing beside it.
    assert_eq!(selected[0].pv(), 1e-4);

    // What pooling FOCAL into one 2x2 would have decided: 24/420 = 5.71% signal
    // vs 4/450 = 0.89% control is a 4.83% excess, under the 5% floor. The gene
    // would have been rejected on delta and all five sites with it -- including
    // the one the site test just called at a 100-point excess.
    assert_eq!(
        m6a_effect_reason(24, 396, 4, 396, &c),
        Some(CallReason::Delta),
        "the pooled gene 2x2 buries its own focal site"
    );

    // Every other putative C of the same gene is kept with the reason it missed.
    let rejected = discovered.rejected.get(&gene_id("FOCAL")).unwrap();
    let reasons: Vec<CallReason> = rejected.iter().map(|s| s.reason()).collect();
    assert_eq!(reasons, vec![CallReason::Delta; 4]);

    assert_eq!(
        discovered.rejected.get(&gene_id("WEAK_P")).unwrap()[0].reason(),
        CallReason::Pvalue
    );
    assert!(discovered.selected.contains_key(&gene_id("HOT_REV")));
    assert_eq!(
        discovered.rejected.get(&gene_id("COLD")).unwrap()[0].reason(),
        CallReason::Delta
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
    // WT == MUT ⇒ delta 0 < 0.05.
    assert_eq!(
        m6a_effect_reason(80, 20, 80, 20, &c),
        Some(CallReason::Delta)
    );
    // Delta ok (0.36 − 0.30 = 0.06) at a 1.2× fold. There is no fold guard, so
    // this clears the effect-size test and goes to the p-value cutoff — the fold
    // gate was measured inert on real DART data (94–99% of sites passed it).
    assert_eq!(m6a_effect_reason(36, 64, 30, 70, &c), None);
    // Strong WT over a clean control ⇒ passes the guards, eligible for the test.
    assert_eq!(m6a_effect_reason(80, 20, 1, 99, &c), None);
}

#[test]
fn effect_reason_zero_control_is_not_nan_kept() {
    // Numerical-safety: with the control floor disabled a 2×2 can reach the
    // guards with n_m = 0. Raw rates floor the denominator so pm = 0 (not
    // 0/0 = NaN, which would make `NaN < delta` false and keep the site). WT is
    // 4% here (< 5% delta), so it must be rejected on delta, not kept.
    let c = contrast(0);
    assert_eq!(m6a_effect_reason(4, 96, 0, 0, &c), Some(CallReason::Delta));
}
