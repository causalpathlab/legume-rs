#![allow(dead_code)]

use clap::ValueEnum;
pub use genomic_data::bed::*;

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

#[derive(Default, Clone)]
pub struct MethylationData {
    pub methylated: usize,
    pub unmethylated: usize,
}

pub trait UpdateMethData {
    fn add_assign(&mut self, other: &Self);
}

impl UpdateMethData for MethylationData {
    fn add_assign(&mut self, other: &Self) {
        self.methylated += other.methylated;
        self.unmethylated += other.unmethylated;
    }
}
