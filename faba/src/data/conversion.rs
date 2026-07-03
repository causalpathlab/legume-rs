pub use genomic_data::bed::*;

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
