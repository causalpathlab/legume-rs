#![allow(dead_code)]

use crate::data::positions::*;
use clap::ValueEnum;
use std::hash::Hash;

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

pub struct MethylationData {
    pub methylated: usize,
    pub unmethylated: usize,
}

impl Default for MethylationData {
    fn default() -> Self {
        Self {
            methylated: 0,
            unmethylated: 0,
        }
    }
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

/// display methylation site name
impl std::fmt::Display for MethylationKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}@{}", self.chr, self.lb, self.ub, self.gene)
    }
}
