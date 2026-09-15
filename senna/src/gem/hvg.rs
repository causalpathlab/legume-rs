//! Per-gene pooling of a gem feature axis for HVG ranking.
//!
//! `senna gem`'s HVG selection ranks GENES, not rows: a gene's
//! `{gene}/count/spliced` and `{gene}/count/unspliced` rows are pooled onto
//! one entry before the variance-trend ranking runs, so both tracks of a
//! selected gene carry projection weight together (see `gem::run`'s HVG
//! block for why).
//!
//! This used to go through `gem::rows::build_gene_track_map`, which also
//! paired tracks for the now-deleted β-sharing engine and tolerated a row
//! that didn't fit the `{gene}/count/{spliced|unspliced}` grammar by giving
//! it its own single-track gene id. That tolerance existed for the engine's
//! sake, not HVG pooling's — a pooled ranking has nowhere harmless to put an
//! unpooled row, so [`build_gene_index`] rejects one outright, naming it.

use auxiliary_data::feature_rows::parse_feature_row;

/// Per-row gene index over a gem feature axis: `row_to_gene[r]` is the dense
/// gene id of row `r`, and the returned `Vec<Box<str>>` is the id-ordered
/// gene keys. `GENE1/count/spliced` and `GENE1/count/unspliced` pool onto the
/// same id; every row must parse as a feature row ([`parse_feature_row`]), or
/// this errors naming the row.
pub(crate) fn build_gene_index(
    feature_names: &[Box<str>],
) -> anyhow::Result<(Vec<u32>, Vec<Box<str>>)> {
    let mut ids: rustc_hash::FxHashMap<Box<str>, u32> = rustc_hash::FxHashMap::default();
    let mut row_to_gene: Vec<u32> = Vec::with_capacity(feature_names.len());
    let mut gene_names: Vec<Box<str>> = Vec::new();
    for name in feature_names {
        let row = parse_feature_row(name).ok_or_else(|| {
            anyhow::anyhow!(
                "{name}: not a `{{gene}}/count/{{spliced|unspliced}}` feature row — gem's HVG \
                 pooling needs every row on the trained axis to parse"
            )
        })?;
        let gid = match ids.get(row.gene) {
            Some(&g) => g,
            None => {
                let g = gene_names.len() as u32;
                ids.insert(row.gene.into(), g);
                gene_names.push(row.gene.into());
                g
            }
        };
        row_to_gene.push(gid);
    }
    Ok((row_to_gene, gene_names))
}

#[cfg(test)]
#[path = "hvg/tests.rs"]
mod tests;
