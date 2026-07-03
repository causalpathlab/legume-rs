use super::*;

fn make_counts(a: usize, t: usize, g: usize, c: usize) -> DnaBaseCount {
    let mut counts = DnaBaseCount::new();
    counts.add(Some(&Dna::A), a);
    counts.add(Some(&Dna::T), t);
    counts.add(Some(&Dna::G), g);
    counts.add(Some(&Dna::C), c);
    counts
}

fn call(counts: DnaBaseCount) -> SnpSite {
    let params = GenotypeParams::default();
    genotype_site(
        SiteInput {
            chr: "chr1".into(),
            pos: 100,
            ref_allele: b'A',
            alt_allele: b'G',
            rsid: None,
            counts,
            qual: None,
        },
        &params,
    )
}

#[test]
fn test_hom_ref_call() {
    let site = call(make_counts(30, 0, 0, 0));
    assert_eq!(site.genotype, SnpGenotype::HomRef);
    assert!(site.gq > 20.0);
}

#[test]
fn test_het_call() {
    let site = call(make_counts(15, 0, 15, 0));
    assert_eq!(site.genotype, SnpGenotype::Het);
    assert!(site.gq > 20.0);
}

#[test]
fn test_hom_alt_call() {
    let site = call(make_counts(0, 0, 30, 0));
    assert_eq!(site.genotype, SnpGenotype::HomAlt);
    assert!(site.gq > 20.0);
}

#[test]
fn test_no_call_low_depth() {
    let site = call(make_counts(1, 0, 1, 0));
    assert_eq!(site.genotype, SnpGenotype::NoCall);
}

#[test]
fn test_no_call_zero_depth() {
    let site = call(DnaBaseCount::new());
    assert_eq!(site.genotype, SnpGenotype::NoCall);
}

#[test]
fn test_qual_model_matches_count_at_uniform_quality() {
    // At uniform Q20 (ε=0.01), quality model should give same calls as count model
    let phred = 20u8;
    let mut qual = DnaBaseQual::default();
    for _ in 0..30 {
        qual.add(Some(&Dna::A), phred);
    }
    let counts = make_counts(30, 0, 0, 0);
    let gl_count = compute_genotype_likelihoods(&counts, b'A', b'G', 0.01);
    let gl_qual = compute_genotype_likelihoods_qual(&qual, b'A', b'G');

    // Both should agree ref >> het >> alt
    assert!(gl_count.ll_ref > gl_count.ll_het);
    assert!(gl_qual.ll_ref > gl_qual.ll_het);
    assert!(gl_count.ll_ref > gl_count.ll_alt);
    assert!(gl_qual.ll_ref > gl_qual.ll_alt);

    // Values should be close (not exact due to het model difference)
    assert!((gl_count.ll_ref - gl_qual.ll_ref).abs() < 1.0);
}

#[test]
fn test_qual_model_het_call() {
    let phred = 30u8;
    let mut qual = DnaBaseQual::default();
    for _ in 0..15 {
        qual.add(Some(&Dna::A), phred);
        qual.add(Some(&Dna::G), phred);
    }
    let gl = compute_genotype_likelihoods_qual(&qual, b'A', b'G');
    assert!(gl.ll_het > gl.ll_ref);
    assert!(gl.ll_het > gl.ll_alt);
}

#[test]
fn test_genotype_likelihoods_ordering() {
    // Pure ref -> ll_ref should be highest
    let counts = make_counts(30, 0, 0, 0);
    let gl = compute_genotype_likelihoods(&counts, b'A', b'G', 0.01);
    assert!(gl.ll_ref > gl.ll_het);
    assert!(gl.ll_ref > gl.ll_alt);

    // Pure alt -> ll_alt should be highest
    let counts = make_counts(0, 0, 30, 0);
    let gl = compute_genotype_likelihoods(&counts, b'A', b'G', 0.01);
    assert!(gl.ll_alt > gl.ll_het);
    assert!(gl.ll_alt > gl.ll_ref);

    // 50/50 -> ll_het should be highest
    let counts = make_counts(15, 0, 15, 0);
    let gl = compute_genotype_likelihoods(&counts, b'A', b'G', 0.01);
    assert!(gl.ll_het > gl.ll_ref);
    assert!(gl.ll_het > gl.ll_alt);
}

#[test]
fn test_snp_site_accessors() {
    let counts = make_counts(20, 0, 10, 0);
    let site = SnpSite {
        chr: "chr1".into(),
        pos: 100,
        ref_allele: b'A',
        alt_allele: b'G',
        rsid: Some("rs123".into()),
        counts,
        genotype: SnpGenotype::Het,
        gq: 30.0,
    };
    assert_eq!(site.ref_count(), 20);
    assert_eq!(site.alt_count(), 10);
    assert_eq!(site.depth(), 30);
}

#[test]
fn test_genotype_display() {
    assert_eq!(format!("{}", SnpGenotype::HomRef), "0/0");
    assert_eq!(format!("{}", SnpGenotype::Het), "0/1");
    assert_eq!(format!("{}", SnpGenotype::HomAlt), "1/1");
    assert_eq!(format!("{}", SnpGenotype::NoCall), "./.");
}
