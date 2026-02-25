/// Compare chromosome names ignoring optional "chr" prefix.
///
/// Handles mixed conventions (e.g., "chr1" vs "1", "chrX" vs "X").
pub fn chr_eq(a: &str, b: &str) -> bool {
    fn strip(s: &str) -> &str {
        s.strip_prefix("chr").unwrap_or(s)
    }
    strip(a) == strip(b)
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
}
