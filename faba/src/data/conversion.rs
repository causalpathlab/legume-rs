#![allow(dead_code)]

use clap::ValueEnum;
pub use genomic_data::bed::*;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ConversionValueType {
    Ratio,
    Converted,
    Unconverted,
}

impl std::fmt::Display for ConversionValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Ratio => {
                write!(f, "ratio")
            }
            Self::Converted => {
                write!(f, "converted")
            }
            Self::Unconverted => {
                write!(f, "unconverted")
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct ConversionData {
    pub converted: usize,
    pub unconverted: usize,
    /// 0-based site position, used for BED annotation
    pub site_pos: i64,
}

pub trait UpdateConversionData {
    fn add_assign(&mut self, other: &Self);
}

impl UpdateConversionData for ConversionData {
    fn add_assign(&mut self, other: &Self) {
        self.converted += other.converted;
        self.unconverted += other.unconverted;
    }
}
