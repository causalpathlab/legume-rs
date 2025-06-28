#![allow(dead_code)]
use std::hash::Hash;

/// cell barcode
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum CellBarcode {
    Barcode(Box<str>),
    Missing,
}

/// UMI barcode
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum UmiBarcode {
    Barcode(Box<str>),
    Missing,
}

/// forward vs. backward(reverse) alignment reads
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum Strand {
    Forward,
    Backward,
}

////////////
// Traits //
////////////

impl std::fmt::Display for CellBarcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = self.clone().into();
        write!(f, "{}", x)
    }
}

impl std::fmt::Display for UmiBarcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = self.clone().into();
        write!(f, "{}", x)
    }
}

impl std::fmt::Display for Strand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = self.clone().into();
        write!(f, "{}", x)
    }
}

impl From<CellBarcode> for Box<str> {
    fn from(cell_barcode: CellBarcode) -> Self {
        match cell_barcode {
            CellBarcode::Missing => Box::from("."),
            CellBarcode::Barcode(boxed_str) => boxed_str,
        }
    }
}

impl From<UmiBarcode> for Box<str> {
    fn from(umi_barcode: UmiBarcode) -> Self {
        match umi_barcode {
            UmiBarcode::Missing => Box::from("."),
            UmiBarcode::Barcode(boxed_str) => boxed_str,
        }
    }
}

impl From<Strand> for Box<str> {
    fn from(read_direction: Strand) -> Self {
        match read_direction {
            Strand::Forward => Box::from("+"),
            Strand::Backward => Box::from("-"),
        }
    }
}
