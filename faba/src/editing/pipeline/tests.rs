use super::{m6a_effect_reason, m6a_partition_by_gene, GeneContrastStat};
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
fn fwd_site(pos: i64, wt_t: usize, wt_c: usize, mu_t: usize, mu_c: usize) -> ConversionSite {
    ConversionSite::M6A {
        m6a_pos: pos,
        conversion_pos: pos + 1,
        wt_freq: base_count(&[(Dna::T, wt_t), (Dna::C, wt_c)]),
        mut_freq: base_count(&[(Dna::T, mu_t), (Dna::C, mu_c)]),
        pv: 0.5,
        gene_pv: f32::NAN,
        qv: 1.0,
        reason: CallReason::default(),
    }
}

/// A reverse-strand m6A candidate: converted reads are A, unconverted are G.
fn rev_site(pos: i64, wt_a: usize, wt_g: usize, mu_a: usize, mu_g: usize) -> ConversionSite {
    ConversionSite::M6A {
        m6a_pos: pos,
        conversion_pos: pos - 1,
        wt_freq: base_count(&[(Dna::A, wt_a), (Dna::G, wt_g)]),
        mut_freq: base_count(&[(Dna::A, mu_a), (Dna::G, mu_g)]),
        pv: 0.5,
        gene_pv: f32::NAN,
        qv: 1.0,
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

#[test]
fn gene_level_pools_by_strand_and_partitions_by_gene_q() {
    // Three genes: a forward "hot" gene (strong WT>MUT), a reverse "hot" gene
    // (exercises the A/G branch), and a "cold" gene where WT == MUT (a variant,
    // no WT-specificity). Only the hot genes should clear the gene-level FDR.
    let gm = gff(&[
        ("HOT_FWD", Strand::Forward),
        ("HOT_REV", Strand::Backward),
        ("COLD", Strand::Forward),
    ]);

    let sites: DashMap<GeneId, Vec<ConversionSite>> = DashMap::new();
    // Two candidate C's per gene: WT 80/20 converted, control ~1/49.
    sites.insert(
        gene_id("HOT_FWD"),
        vec![fwd_site(100, 80, 20, 1, 49), fwd_site(200, 80, 20, 1, 49)],
    );
    sites.insert(
        gene_id("HOT_REV"),
        vec![rev_site(300, 80, 20, 1, 49), rev_site(400, 80, 20, 1, 49)],
    );
    // Cold: WT and control identical → equal rates → not WT-specific.
    sites.insert(
        gene_id("COLD"),
        vec![fwd_site(500, 80, 20, 80, 20), fwd_site(600, 80, 20, 80, 20)],
    );

    let contrast = M6aContrast {
        min_control_coverage: 10,
        min_delta: 0.05,
        rho: 0.02,
    };
    let discovered = m6a_partition_by_gene(sites, &gm, &contrast, 0.05);

    let n_total: usize = discovered.gene_stats.iter().map(|g| g.n_sites).sum();
    assert_eq!(n_total, 6, "3 genes × 2 sites");
    assert!(discovered.selected.contains_key(&gene_id("HOT_FWD")));
    assert!(discovered.selected.contains_key(&gene_id("HOT_REV")));
    assert!(discovered.rejected.contains_key(&gene_id("COLD")));
    assert!(!discovered.selected.contains_key(&gene_id("COLD")));

    let stat = |g: &str| -> GeneContrastStat {
        discovered
            .gene_stats
            .iter()
            .find(|s| s.gene_id == gene_id(g))
            .cloned()
            .expect("gene present in stats")
    };

    // Forward pooling reads T (converted) / C (unconverted)...
    let hf = stat("HOT_FWD");
    assert_eq!((hf.wt_converted, hf.wt_unconverted), (160, 40));
    assert_eq!((hf.mut_converted, hf.mut_unconverted), (2, 98));
    assert!(hf.reason == CallReason::Selected && hf.qv <= 0.05);

    // ...reverse pooling reads A / G. Same input magnitudes ⇒ same pooled 2×2.
    let hr = stat("HOT_REV");
    assert_eq!(
        (hr.wt_converted, hr.wt_unconverted),
        (160, 40),
        "reverse strand must pool A/G, not T/C"
    );
    assert!(hr.reason.is_selected());

    // COLD has WT == MUT, so the pooled delta is 0 < 0.05: it is rejected on the
    // delta guard (before the FDR even runs), and that reason is recorded.
    let cold = stat("COLD");
    assert!(!cold.reason.is_selected());
    assert_eq!(cold.reason, CallReason::Delta);
    assert_eq!(cold.n_sites, 2);

    // Every site of a selected gene inherits its gene's q and outcome...
    for s in discovered.selected.get(&gene_id("HOT_FWD")).unwrap().iter() {
        assert!(s.qv() <= 0.05, "selected site q should match its gene's q");
        assert_eq!(s.reason(), CallReason::Selected);
    }
    // ...and every site of a rejected gene carries the gene's rejection reason.
    for s in discovered.rejected.get(&gene_id("COLD")).unwrap().iter() {
        assert_eq!(s.reason(), CallReason::Delta);
    }
}

fn contrast(min_control_coverage: usize) -> M6aContrast {
    M6aContrast {
        min_control_coverage,
        min_delta: 0.05,
        rho: 0.02,
    }
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
    // this now clears the effect-size test and goes to the FDR — the fold gate
    // was measured inert on real DART data (94–99% of sites passed it).
    assert_eq!(m6a_effect_reason(36, 64, 30, 70, &c), None);
    // Strong WT over a clean control ⇒ passes the guards, eligible for the FDR.
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

/// An A-to-I site: forward edits A→G (G / A), reverse edits T→C (C / T).
fn atoi_site(pos: i64, strand: Strand, edited: usize, unedited: usize) -> ConversionSite {
    let wt = if matches!(strand, Strand::Forward) {
        base_count(&[(Dna::G, edited), (Dna::A, unedited)])
    } else {
        base_count(&[(Dna::C, edited), (Dna::T, unedited)])
    };
    ConversionSite::AtoI {
        editing_pos: pos,
        wt_freq: wt,
        mut_freq: DnaBaseCount::new(),
        pv: 0.5,
        gene_pv: f32::NAN,
        qv: 1.0,
        reason: CallReason::default(),
    }
}

#[test]
fn atoi_gene_level_pools_edits_and_partitions_by_gene_q() {
    use super::atoi_partition_by_gene;
    // Two strongly-edited genes (forward + reverse, to exercise both base sets)
    // and one near the error rate. Single-sample: no effect guards, FDR decides.
    let gm = gff(&[
        ("HOT_F", Strand::Forward),
        ("HOT_R", Strand::Backward),
        ("COLD", Strand::Forward),
    ]);
    let sites: DashMap<GeneId, Vec<ConversionSite>> = DashMap::new();
    sites.insert(
        gene_id("HOT_F"),
        vec![
            atoi_site(100, Strand::Forward, 80, 20),
            atoi_site(200, Strand::Forward, 80, 20),
        ],
    );
    sites.insert(
        gene_id("HOT_R"),
        vec![
            atoi_site(300, Strand::Backward, 80, 20),
            atoi_site(400, Strand::Backward, 80, 20),
        ],
    );
    // ~0.5% editing, below the 1% error null → non-significant.
    sites.insert(
        gene_id("COLD"),
        vec![
            atoi_site(500, Strand::Forward, 1, 199),
            atoi_site(600, Strand::Forward, 1, 199),
        ],
    );

    let discovered = atoi_partition_by_gene(sites, &gm, 0.01, 0.1, 0.05);

    assert!(discovered.selected.contains_key(&gene_id("HOT_F")));
    assert!(discovered.selected.contains_key(&gene_id("HOT_R")));
    assert!(discovered.rejected.contains_key(&gene_id("COLD")));

    let stat = |g: &str| -> GeneContrastStat {
        discovered
            .gene_stats
            .iter()
            .find(|s| s.gene_id == gene_id(g))
            .cloned()
            .expect("gene present")
    };
    // Forward pools G(edited)/A(unedited); no control arm ⇒ MUT columns are 0.
    let hf = stat("HOT_F");
    assert_eq!((hf.wt_converted, hf.wt_unconverted), (160, 40));
    assert_eq!((hf.mut_converted, hf.mut_unconverted), (0, 0));
    assert!(hf.reason == CallReason::Selected);
    // Reverse pools C/T to the same magnitudes.
    let hr = stat("HOT_R");
    assert_eq!(
        (hr.wt_converted, hr.wt_unconverted),
        (160, 40),
        "reverse must pool C/T, not G/A"
    );
    // COLD is eligible (no effect guards) but rejected by the FDR.
    let cold = stat("COLD");
    assert!(!cold.reason.is_selected());
    assert_eq!(cold.reason, CallReason::Fdr);
}
