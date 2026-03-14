use crate::common::*;
use crate::data::methylation::*;
use crate::data::util_htslib::*;

use dashmap::DashMap as HashMap;

/// Check BAM indices for all files
pub fn check_all_bam_indices(bam_files: &[Box<str>]) -> anyhow::Result<()> {
    for bam_file in bam_files {
        info!("checking .bai file for {}...", bam_file);
        check_bam_index(bam_file, None)?;
    }
    Ok(())
}

/// Build gene key as `{gene_id}_{symbol}` for feature naming.
///
/// Feature naming convention: `{gene_key}/{modality}/{detail}`
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
