pub mod genotype_reader;
pub mod bed_reader;

pub use genotype_reader::{GenotypeMatrix, GenotypeReader, GenomicRegion};
pub use bed_reader::BedReader;
