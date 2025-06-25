#![allow(dead_code)]

/// alignment file sample name
///
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum SamSampleName {
    Combined,
    Barcode(Box<str>),
}

/// alignment file UMI name
///
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum SamUmiName {
    Combined,
    Barcode(Box<str>),
}

/// Display sample names
///
impl std::fmt::Display for SamSampleName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamSampleName::Combined => write!(f, "."),
            SamSampleName::Barcode(barcode) => write!(f, "{}", barcode),
        }
    }
}

impl std::fmt::Display for SamUmiName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamUmiName::Combined => write!(f, "."),
            SamUmiName::Barcode(barcode) => write!(f, "{}", barcode),
        }
    }
}
