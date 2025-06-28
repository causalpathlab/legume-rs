use crate::data::positions::*;
use clap::ValueEnum;
use std::hash::Hash;
use std::ops::{Add, AddAssign};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum MethFeatureType {
    Beta,
    Methylated,
    Unmethylated,
}

impl std::fmt::Display for MethFeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Beta => {
                write!(f, "beta")
            }
            Self::Methylated => {
                write!(f, "methylated")
            }
            Self::Unmethylated => {
                write!(f, "unmethylated")
            }
        }
    }
}

#[derive(Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MethylationKey {
    pub chr: Box<str>,
    pub lb: i64,
    pub ub: i64,
    pub gene: Gene,
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
        write!(f, "{}:{}-{}@{}", self.chr, self.lb, self.ub, self.gene)
    }
}
