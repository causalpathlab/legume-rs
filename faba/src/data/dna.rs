#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy, Ord, Hash)]
pub enum Dna {
    A,
    T,
    G,
    C,
}

impl Dna {
    pub fn from_byte(b: u8) -> Option<Dna> {
        match b {
            b'A' => Some(Dna::A),
            b'T' => Some(Dna::T),
            b'G' => Some(Dna::G),
            b'C' => Some(Dna::C),
            _ => None,
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            Dna::A => b'A',
            Dna::T => b'T',
            Dna::G => b'G',
            Dna::C => b'C',
        }
    }

    /// Array index: A=0, T=1, G=2, C=3.
    #[inline]
    pub fn to_index(self) -> usize {
        match self {
            Dna::A => 0,
            Dna::T => 1,
            Dna::G => 2,
            Dna::C => 3,
        }
    }

    /// Array index from raw byte. Returns None for non-ATGC bytes.
    #[inline]
    pub fn byte_to_index(b: u8) -> Option<usize> {
        Self::from_byte(b).map(|d| d.to_index())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct DnaBaseCount {
    data: [(Dna, usize); 4],
}

impl Default for DnaBaseCount {
    fn default() -> Self {
        Self::new()
    }
}

impl DnaBaseCount {
    pub fn new() -> Self {
        DnaBaseCount {
            data: [
                (Dna::A, 0usize),
                (Dna::T, 0usize),
                (Dna::G, 0usize),
                (Dna::C, 0usize),
            ],
        }
    }

    pub fn add(&mut self, b: Option<&Dna>, val: usize) {
        if let Some(base) = b {
            self.data[base.to_index()].1 += val;
        }
    }

    pub fn get(&self, b: Option<&Dna>) -> usize {
        match b {
            Some(base) => self.data[base.to_index()].1,
            None => 0,
        }
    }

    pub fn count_a(&self) -> usize {
        self.data[0].1
    }

    pub fn count_t(&self) -> usize {
        self.data[1].1
    }

    pub fn count_g(&self) -> usize {
        self.data[2].1
    }

    pub fn count_c(&self) -> usize {
        self.data[3].1
    }

    pub fn total(&self) -> usize {
        self.data[0].1 + self.data[1].1 + self.data[2].1 + self.data[3].1
    }
}

/// Precomputed log-likelihood terms for a single Phred quality score.
struct PhredEntry {
    log_correct: f64, // log(1 - ε)
    log_error3: f64,  // log(ε / 3)
    log_het: f64,     // log(0.5 * (1 - 2ε/3))
}

static PHRED_TABLE: std::sync::LazyLock<[PhredEntry; 256]> = std::sync::LazyLock::new(|| {
    let zero = PhredEntry {
        log_correct: 0.0,
        log_error3: 0.0,
        log_het: 0.0,
    };
    // Initialize with zeros, then fill
    let mut table: [PhredEntry; 256] = std::array::from_fn(|_| PhredEntry { ..zero });
    for (q, entry) in table.iter_mut().enumerate() {
        let eps = (10.0_f64)
            .powf(-(q as f64) / 10.0)
            .clamp(1e-10, 1.0 - 1e-10);
        entry.log_correct = (1.0 - eps).ln();
        entry.log_error3 = (eps / 3.0).ln();
        entry.log_het = (0.5 * (1.0 - 2.0 * eps / 3.0)).ln();
    }
    table
});

/// Per-base quality accumulator for the Li 2011 genotype likelihood model.
///
/// For each base b ∈ {A, T, G, C}, accumulates log-likelihood terms from
/// individual read quality scores so that genotype likelihoods can be computed
/// in O(1) at call time without storing per-read data.
///
/// Given a read with base b and Phred quality q, ε = 10^(-q/10):
///   sum_log_correct[b] += log(1 - ε)       — read matches true allele
///   sum_log_error3[b]  += log(ε / 3)        — read is a sequencing error
///   sum_log_het[b]     += log(0.5·(1-2ε/3)) — read from a het (ref or alt)
///
/// Reference: Li H, Bioinformatics 27(21):2987-2993, 2011.
#[derive(Clone, Debug)]
pub struct DnaBaseQual {
    sum_log_correct: [f64; 4],
    sum_log_error3: [f64; 4],
    sum_log_het: [f64; 4],
}

impl Default for DnaBaseQual {
    fn default() -> Self {
        Self {
            sum_log_correct: [0.0; 4],
            sum_log_error3: [0.0; 4],
            sum_log_het: [0.0; 4],
        }
    }
}

impl DnaBaseQual {
    /// Accumulate one base observation with its Phred quality score.
    #[inline]
    pub fn add(&mut self, base: Option<&Dna>, phred: u8) {
        let idx = match base {
            Some(b) => b.to_index(),
            None => return,
        };
        let entry = &PHRED_TABLE[phred as usize];
        self.sum_log_correct[idx] += entry.log_correct;
        self.sum_log_error3[idx] += entry.log_error3;
        self.sum_log_het[idx] += entry.log_het;
    }

    /// Compute quality-weighted log-likelihood for homozygous genotype.
    pub fn ll_hom(&self, allele: u8) -> f64 {
        let target = match Dna::byte_to_index(allele) {
            Some(i) => i,
            None => return f64::NEG_INFINITY,
        };
        let mut ll = self.sum_log_correct[target];
        for (i, &v) in self.sum_log_error3.iter().enumerate() {
            if i != target {
                ll += v;
            }
        }
        ll
    }

    /// Compute quality-weighted log-likelihood for het genotype.
    pub fn ll_het(&self, ref_allele: u8, alt_allele: u8) -> f64 {
        let (ri, ai) = match (
            Dna::byte_to_index(ref_allele),
            Dna::byte_to_index(alt_allele),
        ) {
            (Some(r), Some(a)) => (r, a),
            _ => return f64::NEG_INFINITY,
        };
        let mut ll = self.sum_log_het[ri] + self.sum_log_het[ai];
        for (i, &v) in self.sum_log_error3.iter().enumerate() {
            if i != ri && i != ai {
                ll += v;
            }
        }
        ll
    }
}

use std::ops::AddAssign;

impl AddAssign<&DnaBaseCount> for DnaBaseCount {
    fn add_assign(&mut self, other: &Self) {
        self.add(Some(&Dna::A), other.get(Some(&Dna::A)));
        self.add(Some(&Dna::T), other.get(Some(&Dna::T)));
        self.add(Some(&Dna::G), other.get(Some(&Dna::G)));
        self.add(Some(&Dna::C), other.get(Some(&Dna::C)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_count_add_get() {
        let mut count = DnaBaseCount::new();
        count.add(Some(&Dna::A), 5);
        count.add(Some(&Dna::T), 3);
        count.add(Some(&Dna::G), 7);
        count.add(Some(&Dna::C), 2);

        assert_eq!(count.count_a(), 5);
        assert_eq!(count.count_t(), 3);
        assert_eq!(count.count_g(), 7);
        assert_eq!(count.count_c(), 2);

        assert_eq!(count.get(Some(&Dna::A)), 5);
        assert_eq!(count.get(Some(&Dna::T)), 3);
        assert_eq!(count.get(None), 0);

        // Add more to existing
        count.add(Some(&Dna::A), 10);
        assert_eq!(count.count_a(), 15);
    }

    #[test]
    fn test_add_assign() {
        let mut a = DnaBaseCount::new();
        a.add(Some(&Dna::A), 10);
        a.add(Some(&Dna::T), 5);

        let mut b = DnaBaseCount::new();
        b.add(Some(&Dna::A), 3);
        b.add(Some(&Dna::G), 7);

        a += &b;
        assert_eq!(a.count_a(), 13);
        assert_eq!(a.count_t(), 5);
        assert_eq!(a.count_g(), 7);
        assert_eq!(a.count_c(), 0);
    }
}
