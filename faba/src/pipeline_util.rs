use crate::common::*;
use crate::data::methylation::*;
use crate::data::util_htslib::*;

use dashmap::DashMap as HashMap;

/// Bin a position based on resolution (in kb)
/// Returns (start, stop) where start is inclusive and stop is exclusive: [start, stop)
#[inline]
pub fn bin_position_kb(position: i64, resolution_kb: Option<f32>) -> (i64, i64) {
    if let Some(r) = resolution_kb {
        let r = (r * 1000.0) as usize;
        let start = ((position as usize) / r * r) as i64;
        let stop = start + r as i64;
        (start, stop)
    } else {
        (position, position + 1)
    }
}

/// Check BAM indices for all files
pub fn check_all_bam_indices(bam_files: &[Box<str>]) -> anyhow::Result<()> {
    for bam_file in bam_files {
        info!("checking .bai file for {}...", bam_file);
        check_bam_index(bam_file, None)?;
    }
    Ok(())
}

pub fn create_gene_key_function(
    gff_map: &GffRecordMap,
) -> impl Fn(&BedWithGene) -> Box<str> + Send + Sync + '_ {
    |x: &BedWithGene| -> Box<str> {
        gff_map
            .get(&x.gene)
            .map(|gff| format!("{}_{}", gff.gene_id, gff.gene_name))
            .unwrap_or_else(|| format!("{}", x.gene))
            .into_boxed_str()
    }
}

pub fn summarize_stats<F, V, T>(
    stats: &[(CellBarcode, BedWithGene, MethylationData)],
    feature_key_func: F,
    value_func: V,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
    V: Fn(&MethylationData) -> f32 + Send + Sync,
{
    let combined_data: HashMap<(CellBarcode, T), MethylationData> = HashMap::default();

    stats.par_iter().for_each(|(cb, k, dat)| {
        let key = (cb.clone(), feature_key_func(k));
        combined_data.entry(key).or_default().add_assign(dat);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, value_func(&v)))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}
