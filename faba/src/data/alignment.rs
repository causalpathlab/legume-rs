#![allow(dead_code)]
use std::hash::Hash;

/// alignment file sample name
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum SamSampleName {
    Combined,
    Barcode(Box<str>),
}

/// alignment file UMI name
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum SamUmiName {
    Combined,
    Barcode(Box<str>),
}

/// forward vs. backward(reverse) alignment reads
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

/// display sample names
impl std::fmt::Display for SamSampleName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Combined => write!(f, "."),
            Self::Barcode(barcode) => write!(f, "{}", barcode),
        }
    }
}

/// display UMI names
impl std::fmt::Display for SamUmiName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Combined => write!(f, "."),
            Self::Barcode(barcode) => write!(f, "{}", barcode),
        }
    }
}

/// display direction
impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward => write!(f, "+"),
            Self::Backward => write!(f, "-"),
        }
    }
}
