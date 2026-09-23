//! The two-track gene axis: RNA gene rows are the base track, and each gene's
//! peak-aggregated row is a second track tied to the same gene.
//!
//! Rows are named `{gene}/{tag}` by the loader's per-file suffix. The engine
//! sees only track and gene ids; what a track means stays here. Every gene must
//! have a base (RNA) row, since the other track is read as an offset from it.

use graph_embedding_util::{TrackInfo, TrackSpec};
use rustc_hash::FxHashMap;

/// Suffix of the RNA rows (the base track).
pub const RNA_TAG: &str = "rna";
/// Suffix of the peak-aggregated rows.
pub const ATAC_TAG: &str = "atac";

/// Track and gene ids for every row of a two-track gene axis.
pub fn gene_tracks(feature_names: &[Box<str>]) -> anyhow::Result<TrackSpec> {
    let mut gene_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut has_base: Vec<bool> = Vec::new();
    let mut track_of_row = Vec::with_capacity(feature_names.len());
    let mut gene_of_row = Vec::with_capacity(feature_names.len());
    for name in feature_names {
        let (gene, tag) = name
            .rsplit_once('/')
            .ok_or_else(|| anyhow::anyhow!("feature `{name}` carries no track tag"))?;
        let track = match tag {
            RNA_TAG => 0,
            ATAC_TAG => 1,
            _ => anyhow::bail!("feature `{name}`: unknown track tag `{tag}`"),
        };
        let next = gene_id.len() as u32;
        let g = *gene_id.entry(gene).or_insert(next);
        if g as usize == has_base.len() {
            has_base.push(false);
        }
        has_base[g as usize] |= track == 0;
        track_of_row.push(track);
        gene_of_row.push(g);
    }
    if let Some((gene, _)) = gene_id.iter().find(|(_, &g)| !has_base[g as usize]) {
        anyhow::bail!("gene `{gene}` has a peak-aggregated row but no RNA row");
    }
    let spec = TrackSpec {
        track_of_row,
        gene_of_row,
        tracks: [RNA_TAG, ATAC_TAG]
            .into_iter()
            .map(|t| TrackInfo {
                name: t.into(),
                is_count: true,
            })
            .collect(),
    };
    spec.validate(feature_names.len())?;
    Ok(spec)
}
