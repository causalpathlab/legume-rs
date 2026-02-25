/// Compare chromosome names ignoring optional "chr" prefix.
///
/// Handles mixed conventions (e.g., "chr1" vs "1", "chrX" vs "X").
pub fn chr_eq(a: &str, b: &str) -> bool {
    fn strip(s: &str) -> &str {
        s.strip_prefix("chr").unwrap_or(s)
    }
    strip(a) == strip(b)
}

/// Strip the "chr" prefix from a chromosome name for use as a lookup key.
pub fn chr_stripped(s: &str) -> &str {
    s.strip_prefix("chr").unwrap_or(s)
}

/// Result of comparing sumstat alleles against reference panel alleles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlleleMatch {
    /// Same orientation — keep z/beta as-is.
    Same,
    /// Effect allele is swapped — negate z and beta.
    Flipped,
    /// Strand-ambiguous pair (A/T or C/G) — should drop.
    Ambiguous,
    /// Alleles don't correspond at all — should drop.
    Mismatch,
}

/// Returns true for strand-ambiguous allele pairs: A/T, T/A, C/G, G/C.
pub fn is_strand_ambiguous(a1: &str, a2: &str) -> bool {
    matches!(
        (
            a1.to_ascii_uppercase().as_str(),
            a2.to_ascii_uppercase().as_str()
        ),
        ("A", "T") | ("T", "A") | ("C", "G") | ("G", "C")
    )
}

/// Compare sumstat alleles (sum_a1, sum_a2) against reference alleles (ref_a1, ref_a2).
pub fn alleles_match(sum_a1: &str, sum_a2: &str, ref_a1: &str, ref_a2: &str) -> AlleleMatch {
    let (s1, s2) = (sum_a1.to_ascii_uppercase(), sum_a2.to_ascii_uppercase());
    let (r1, r2) = (ref_a1.to_ascii_uppercase(), ref_a2.to_ascii_uppercase());

    if is_strand_ambiguous(&r1, &r2) {
        return AlleleMatch::Ambiguous;
    }

    if s1 == r1 && s2 == r2 {
        AlleleMatch::Same
    } else if s1 == r2 && s2 == r1 {
        AlleleMatch::Flipped
    } else {
        AlleleMatch::Mismatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chr_eq_same_prefix() {
        assert!(chr_eq("chr1", "chr1"));
        assert!(chr_eq("chrX", "chrX"));
        assert!(!chr_eq("chr1", "chr2"));
    }

    #[test]
    fn test_chr_eq_no_prefix() {
        assert!(chr_eq("1", "1"));
        assert!(chr_eq("X", "X"));
        assert!(!chr_eq("1", "2"));
    }

    #[test]
    fn test_chr_eq_mixed_prefix() {
        assert!(chr_eq("chr1", "1"));
        assert!(chr_eq("1", "chr1"));
        assert!(chr_eq("chrX", "X"));
        assert!(chr_eq("X", "chrX"));
        assert!(chr_eq("chrMT", "MT"));
        assert!(chr_eq("MT", "chrMT"));
    }

    #[test]
    fn test_chr_eq_mixed_no_match() {
        assert!(!chr_eq("chr1", "2"));
        assert!(!chr_eq("1", "chr2"));
        assert!(!chr_eq("chrX", "Y"));
    }

    #[test]
    fn test_is_strand_ambiguous() {
        assert!(is_strand_ambiguous("A", "T"));
        assert!(is_strand_ambiguous("T", "A"));
        assert!(is_strand_ambiguous("C", "G"));
        assert!(is_strand_ambiguous("G", "C"));
        assert!(is_strand_ambiguous("a", "t")); // case-insensitive
        assert!(!is_strand_ambiguous("A", "C"));
        assert!(!is_strand_ambiguous("A", "G"));
        assert!(!is_strand_ambiguous("T", "C"));
        assert!(!is_strand_ambiguous("T", "G"));
    }

    #[test]
    fn test_alleles_match_same() {
        assert_eq!(alleles_match("A", "C", "A", "C"), AlleleMatch::Same);
        assert_eq!(alleles_match("a", "c", "A", "C"), AlleleMatch::Same);
    }

    #[test]
    fn test_alleles_match_flipped() {
        assert_eq!(alleles_match("C", "A", "A", "C"), AlleleMatch::Flipped);
        assert_eq!(alleles_match("c", "a", "A", "C"), AlleleMatch::Flipped);
    }

    #[test]
    fn test_alleles_match_ambiguous() {
        assert_eq!(alleles_match("A", "T", "A", "T"), AlleleMatch::Ambiguous);
        assert_eq!(alleles_match("C", "G", "C", "G"), AlleleMatch::Ambiguous);
        assert_eq!(alleles_match("T", "A", "A", "T"), AlleleMatch::Ambiguous);
    }

    #[test]
    fn test_alleles_match_mismatch() {
        assert_eq!(alleles_match("A", "G", "C", "T"), AlleleMatch::Mismatch); // ref C/T not ambiguous, sumstat A/G doesn't match
        assert_eq!(alleles_match("A", "G", "A", "C"), AlleleMatch::Mismatch);
        assert_eq!(alleles_match("T", "G", "A", "C"), AlleleMatch::Mismatch);
    }
}
