#![allow(dead_code)]

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
            _ => None, // Handle invalid bases
        }
    }
    pub fn from_byte_complement(b: u8) -> Option<Dna> {
        Dna::from_byte(b).map(|b| match b {
            Dna::A => Dna::T,
            Dna::C => Dna::G,
            Dna::G => Dna::C,
            Dna::T => Dna::A,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BiAllele {
    pub a1: Dna,
    pub a2: Dna,
    pub n1: usize,
    pub n2: usize,
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

    pub fn set(&mut self, b: Option<&Dna>, val: usize) {
        if let Some(base) = b {
            match base {
                Dna::A => self.data[0].1 = val,
                Dna::T => self.data[1].1 = val,
                Dna::G => self.data[2].1 = val,
                Dna::C => self.data[3].1 = val,
            }
        }
    }

    pub fn add(&mut self, b: Option<&Dna>, val: usize) {
        if let Some(base) = b {
            match base {
                Dna::A => self.data[0].1 += val,
                Dna::T => self.data[1].1 += val,
                Dna::G => self.data[2].1 += val,
                Dna::C => self.data[3].1 += val,
            }
        }
    }

    pub fn get(&self, b: Option<&Dna>) -> usize {
        match b {
            Some(Dna::A) => self.data[0].1,
            Some(Dna::T) => self.data[1].1,
            Some(Dna::G) => self.data[2].1,
            Some(Dna::C) => self.data[3].1,
            None => 0,
        }
    }

    pub fn most_frequent(&self) -> &(Dna, usize) {
        self.data
            .iter()
            .max_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap()
    }

    pub fn is_mono_allelic(&self) -> bool {
        let mfa = self.most_frequent();
        let tot = self.data.iter().map(|&(_, x)| x).sum::<usize>();
        tot == mfa.1
    }

    pub fn second_most_frequent(&self) -> &(Dna, usize) {
        let mfa = self.most_frequent();
        self.data
            .iter()
            .filter(|&s| s.0 != mfa.0)
            .max_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap()
    }

    pub fn bi_allelic_stat(&self) -> BiAllele {
        let fst = self.most_frequent();
        let snd = self
            .data
            .iter()
            .filter(|&s| s.0 != fst.0)
            .max_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();

        BiAllele {
            a1: fst.0,
            a2: snd.0,
            n1: fst.1,
            n2: snd.1,
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
    fn test_most_frequent() {
        let mut count = DnaBaseCount::new();
        count.add(Some(&Dna::A), 5);
        count.add(Some(&Dna::T), 20);
        count.add(Some(&Dna::G), 3);
        count.add(Some(&Dna::C), 10);

        let (base, n) = count.most_frequent();
        assert_eq!(*base, Dna::T);
        assert_eq!(*n, 20);

        let (base2, n2) = count.second_most_frequent();
        assert_eq!(*base2, Dna::C);
        assert_eq!(*n2, 10);
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

    #[test]
    fn test_bi_allelic_stat() {
        let mut count = DnaBaseCount::new();
        count.add(Some(&Dna::C), 80);
        count.add(Some(&Dna::T), 20);

        let bi = count.bi_allelic_stat();
        assert_eq!(bi.a1, Dna::C);
        assert_eq!(bi.n1, 80);
        assert_eq!(bi.a2, Dna::T);
        assert_eq!(bi.n2, 20);
    }
}
