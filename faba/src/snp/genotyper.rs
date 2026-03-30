use crate::data::dna::{Dna, DnaBaseCount, DnaBaseQual};
use crate::snp::{SnpGenotype, SnpSite};

/// Log-likelihoods for each genotype hypothesis
pub struct GenotypeLikelihoods {
    /// log P(data | 0/0)
    pub ll_ref: f64,
    /// log P(data | 0/1)
    pub ll_het: f64,
    /// log P(data | 1/1)
    pub ll_alt: f64,
}

/// Parameters for genotype calling and de novo discovery
pub struct GenotypeParams {
    /// Minimum read depth to attempt calling (known-site mode)
    pub min_depth: usize,
    /// Minimum genotype quality (Phred) to emit a call
    pub min_gq: f32,
    /// Prior probability of heterozygous genotype
    pub het_prior: f64,
    /// Prior probability of homozygous alt genotype
    pub hom_alt_prior: f64,
    /// Sequencing error rate (probability of observing wrong base)
    pub base_error_rate: f64,
    /// Minimum total depth for de novo discovery
    pub min_coverage: usize,
    /// Minimum alt allele read count for de novo discovery
    pub min_alt_count: usize,
    /// Minimum alt allele frequency for de novo discovery
    pub min_alt_freq: f64,
}

impl Default for GenotypeParams {
    fn default() -> Self {
        Self {
            min_depth: 5,
            min_gq: 20.0,
            het_prior: 0.001,
            hom_alt_prior: 0.0001,
            base_error_rate: 0.01,
            min_coverage: 10,
            min_alt_count: 3,
            min_alt_freq: 0.1,
        }
    }
}

/// Compute genotype log-likelihoods from allele counts using a binomial model.
///
/// Simplified pileup model with uniform error rate, as used by cellSNP-lite
/// and Vartrix for single-cell genotyping at known variant sites.
///
/// Reference:
///   Huang X, Huang Y, "Cellsnp-lite: an efficient tool for genotyping
///   single cells", Bioinformatics, 37(23):4569-4571, 2021.
///   <https://doi.org/10.1093/bioinformatics/btab358>
///
/// The full per-read-quality model (Li H, "A statistical framework for SNP
/// calling, mutation discovery, association mapping and population genetical
/// parameter estimation from sequencing data", Bioinformatics, 27(21):
/// 2987-2993, 2011) uses per-base quality scores. This simplified version
/// assumes a constant error rate across all reads.
///
/// For genotype G in {RR, RA, AA}:
///   P(data|RR) = Binom(n_alt; depth, epsilon)
///   P(data|RA) = Binom(n_alt; depth, 0.5)
///   P(data|AA) = Binom(n_alt; depth, 1 - epsilon)
///
/// The binomial coefficient C(n,k) cancels across genotypes, so we omit it.
pub fn compute_genotype_likelihoods(
    counts: &DnaBaseCount,
    ref_allele: u8,
    alt_allele: u8,
    error_rate: f64,
) -> GenotypeLikelihoods {
    let ref_base = Dna::from_byte(ref_allele);
    let alt_base = Dna::from_byte(alt_allele);

    let n_ref = counts.get(ref_base.as_ref()) as f64;
    let n_alt = counts.get(alt_base.as_ref()) as f64;
    let depth = n_ref + n_alt;

    if depth == 0.0 {
        return GenotypeLikelihoods {
            ll_ref: 0.0,
            ll_het: 0.0,
            ll_alt: 0.0,
        };
    }

    let eps = error_rate.clamp(1e-10, 1.0 - 1e-10);

    // log P(data | RR): alt reads are errors
    let ll_ref = n_alt * eps.ln() + n_ref * (1.0 - eps).ln();
    // log P(data | RA): expect 50/50
    let ll_het = depth * 0.5_f64.ln();
    // log P(data | AA): ref reads are errors
    let ll_alt = n_ref * eps.ln() + n_alt * (1.0 - eps).ln();

    GenotypeLikelihoods {
        ll_ref,
        ll_het,
        ll_alt,
    }
}

/// Call genotype from likelihoods using MAP with priors.
/// Returns (genotype, Phred-scaled genotype quality).
pub fn call_genotype(gl: &GenotypeLikelihoods, params: &GenotypeParams) -> (SnpGenotype, f32) {
    let ref_prior = 1.0 - params.het_prior - params.hom_alt_prior;

    // Log posteriors (unnormalized)
    let lp_ref = gl.ll_ref + ref_prior.ln();
    let lp_het = gl.ll_het + params.het_prior.ln();
    let lp_alt = gl.ll_alt + params.hom_alt_prior.ln();

    // Find MAP genotype
    let (gt, best_lp) = if lp_ref >= lp_het && lp_ref >= lp_alt {
        (SnpGenotype::HomRef, lp_ref)
    } else if lp_het >= lp_ref && lp_het >= lp_alt {
        (SnpGenotype::Het, lp_het)
    } else {
        (SnpGenotype::HomAlt, lp_alt)
    };

    // GQ = Phred-scaled probability of the second-best genotype
    // GQ = -10 * log10(1 - P(best)) = -10 * log10(P(second-best) + P(third))
    // Using log-sum-exp for numerical stability
    let all = [lp_ref, lp_het, lp_alt];
    let log_total = log_sum_exp(&all);
    let log_p_best = best_lp - log_total;
    let log_p_error = (1.0_f64 - log_p_best.exp()).max(1e-300).ln();
    let gq = (-10.0 * log_p_error * std::f64::consts::LOG10_E) as f32;

    (gt, gq.max(0.0))
}

/// Compute genotype log-likelihoods from pre-accumulated per-base quality stats
/// (Li 2011 model). Uses quality-weighted sums stored in `DnaBaseQual`.
pub fn compute_genotype_likelihoods_qual(
    qual: &DnaBaseQual,
    ref_allele: u8,
    alt_allele: u8,
) -> GenotypeLikelihoods {
    GenotypeLikelihoods {
        ll_ref: qual.ll_hom(ref_allele),
        ll_het: qual.ll_het(ref_allele, alt_allele),
        ll_alt: qual.ll_hom(alt_allele),
    }
}

/// Input data for genotyping a single SNP site.
pub struct SiteInput {
    pub chr: Box<str>,
    pub pos: i64,
    pub ref_allele: u8,
    pub alt_allele: u8,
    pub rsid: Option<Box<str>>,
    pub counts: DnaBaseCount,
    pub qual: Option<DnaBaseQual>,
}

/// Genotype a single known SNP site from pileup counts.
/// When `qual` is provided, uses the Li 2011 per-base quality model;
/// otherwise falls back to the simplified constant-error-rate model.
pub fn genotype_site(input: SiteInput, params: &GenotypeParams) -> SnpSite {
    let depth = input.counts.total();

    if depth < params.min_depth {
        return SnpSite {
            chr: input.chr,
            pos: input.pos,
            ref_allele: input.ref_allele,
            alt_allele: input.alt_allele,
            rsid: input.rsid,
            counts: input.counts,
            genotype: SnpGenotype::NoCall,
            gq: 0.0,
        };
    }

    let gl = if let Some(ref q) = input.qual {
        compute_genotype_likelihoods_qual(q, input.ref_allele, input.alt_allele)
    } else {
        compute_genotype_likelihoods(
            &input.counts,
            input.ref_allele,
            input.alt_allele,
            params.base_error_rate,
        )
    };
    let (genotype, gq) = call_genotype(&gl, params);

    let genotype = if gq < params.min_gq {
        SnpGenotype::NoCall
    } else {
        genotype
    };

    SnpSite {
        chr: input.chr,
        pos: input.pos,
        ref_allele: input.ref_allele,
        alt_allele: input.alt_allele,
        rsid: input.rsid,
        counts: input.counts,
        genotype,
        gq,
    }
}

/// Find the most frequent non-reference allele from base counts.
/// Returns (alt_base_byte, alt_count) or None if no non-ref bases observed.
pub fn find_top_alt_allele(counts: &DnaBaseCount, ref_allele: u8) -> Option<(u8, usize)> {
    let alleles = [
        (b'A', Dna::A),
        (b'T', Dna::T),
        (b'G', Dna::G),
        (b'C', Dna::C),
    ];

    alleles
        .iter()
        .filter(|(byte, _)| *byte != ref_allele)
        .map(|(byte, dna)| (*byte, counts.get(Some(dna))))
        .max_by_key(|(_, count)| *count)
        .filter(|(_, count)| *count > 0)
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
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
}
