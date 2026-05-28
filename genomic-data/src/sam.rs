#![allow(dead_code)]
use std::hash::Hash;
use std::sync::Arc;

/// Cell barcode.
///
/// `Arc<str>` (not `Box<str>`): downstream code (cell_assign, dedup maps,
/// PDUI, simulate) clones the barcode many times per fragment. With
/// `Box<str>` each clone is a fresh heap allocation; with `Arc<str>` a
/// clone is just an atomic refcount bump. Combined with the thread-local
/// interner in `faba::data::bam_io`, extraction from a BAM aux tag also
/// dedupes — repeated barcodes hand back the same `Arc<str>`.
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum CellBarcode {
    Barcode(Arc<str>),
    Missing,
}

/// UMI identity, stored as a 64-bit hash.
///
/// UMIs are only used for deduplication (per cell × site), never displayed
/// to the user, so we never need the original string. Storing the hash
/// directly eliminates the per-read `Box<str>` allocation that previously
/// dominated `extract_fragments` on highly-expressed UTRs. Collision
/// probability across 10^5 UMIs in a single cell is ~10^-10 — well below
/// the empirical UMI error rate.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum UmiBarcode {
    Hash(u64),
    Missing,
}

/// forward vs. backward(reverse) alignment reads
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Copy, Default)]
pub enum Strand {
    #[default]
    Forward,
    Backward,
}

////////////
// Traits //
////////////

impl std::fmt::Display for CellBarcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellBarcode::Barcode(s) => f.write_str(s),
            CellBarcode::Missing => f.write_str("."),
        }
    }
}

impl std::fmt::Display for UmiBarcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UmiBarcode::Hash(h) => write!(f, "{:016x}", h),
            UmiBarcode::Missing => f.write_str("."),
        }
    }
}

impl std::fmt::Display for Strand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = (*self).into();
        write!(f, "{}", x)
    }
}

impl From<CellBarcode> for Box<str> {
    fn from(cell_barcode: CellBarcode) -> Self {
        match cell_barcode {
            CellBarcode::Missing => Box::from("."),
            CellBarcode::Barcode(arc_str) => Box::from(&*arc_str),
        }
    }
}

impl From<UmiBarcode> for Box<str> {
    fn from(umi_barcode: UmiBarcode) -> Self {
        match umi_barcode {
            UmiBarcode::Missing => Box::from("."),
            UmiBarcode::Hash(h) => format!("{:016x}", h).into_boxed_str(),
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
