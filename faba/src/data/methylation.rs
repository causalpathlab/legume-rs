use crate::data::alignment::*;
use std::hash::Hash;
use std::ops::{Add, AddAssign};

#[derive(Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MethylationKey {
    pub chr: Box<str>,
    pub lb: usize,
    pub ub: usize,
    pub dir: Direction,
}

#[derive(Default)]
pub struct MethylationData {
    pub methylated: usize,
    pub unmethylated: usize,
}

impl Add for MethylationData {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            methylated: self.methylated + other.methylated,
            unmethylated: self.unmethylated + other.unmethylated,
        }
    }
}

impl AddAssign for MethylationData {
    fn add_assign(&mut self, other: Self) {
        self.methylated += other.methylated;
        self.unmethylated += other.unmethylated;
    }
}

/// display sample names
impl std::fmt::Display for MethylationKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}@{}", self.chr, self.lb, self.ub, self.dir)
    }
}
