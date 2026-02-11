pub mod bed_reader;
pub mod genotype_reader;

pub use bed_reader::BedReader;
pub use genotype_reader::{GenomicRegion, GenotypeMatrix, GenotypeReader};
