#![allow(dead_code)]

#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, Clone)]
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
            a1: fst.0.clone(),
            a2: snd.0.clone(),
            n1: fst.1,
            n2: snd.1,
        }
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
