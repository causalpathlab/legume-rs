//! The gene-count row grammar shared by both gem models.
//!
//! `gem` and `gem-encoder` fit different architectures over the *same* input:
//! a gene-level count matrix whose rows are `{gene}/count/{spliced|unspliced}`.
//! How a row is read — which gene it belongs to, which track it is, and what
//! happens to a row that is neither — is not an architectural choice, so it is
//! not one either model gets to make for itself. It lives here and both call it.
//!
//! That was not always true. `gem` used to match on `/count/` with a bare
//! `rsplit_once`, which cannot separate "spliced" from "not a count row": a
//! `{gene}/count/total` row (the pooled gene-QC track faba also emits) parsed as
//! a *second spliced row* of that gene and was silently added to it, while
//! `gem-encoder` rejected the same row. Same input, two answers.

use candle_util::data::indexed::GeneTrackMap;
use rustc_hash::FxHashMap;

/// Split a gem feature row `{gene}/count/{spliced|unspliced}` into its gene key
/// and whether it is the **nascent** (unspliced) track. `None` when the row is
/// not a gene-level count row at all.
///
/// Goes through [`auxiliary_data::feature_rows::parse_feature_row`] rather than matching
/// on `/count/` directly, because a bare `rsplit_once` **cannot tell "spliced"
/// apart from "not a count row"** — both fall to the same branch. It used to,
/// and the consequence was silent: `BRCA2/m6a/methylated` became a mature gene
/// literally named `BRCA2/m6a/methylated`, and the sub-gene form
/// `{gene}/count/{site}/{channel}` became a mature row of the right gene. The
/// `n_nascent > 0` guard below catches a wholly spliced input, not a
/// contaminated one.
///
/// A `subunit` is rejected on purpose: this model is gene-resolution, so a
/// per-site or per-component row is not a thing it can pair across tracks.
#[must_use]
pub fn split_count_row(name: &str) -> Option<(&str, bool)> {
    use auxiliary_data::feature_rows::{parse_feature_row, COUNT, SPLICED, UNSPLICED};
    let row = parse_feature_row(name)?;
    if row.modality != COUNT || row.subunit.is_some() {
        return None;
    }
    match row.channel {
        SPLICED => Some((row.gene, false)),
        UNSPLICED => Some((row.gene, true)),
        _ => None,
    }
}

/// Intern each row's gene key to a dense gene id, returning the row→gene map,
/// the nascent flags, and the id-ordered gene names.
#[must_use]
pub fn build_gene_track_map(feature_names: &[Box<str>]) -> (GeneTrackMap, Vec<Box<str>>) {
    let mut ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut row_to_gene = Vec::with_capacity(feature_names.len());
    let mut row_is_nascent = Vec::with_capacity(feature_names.len());
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();
    for name in feature_names {
        let Some((gene, is_nascent)) = split_count_row(name) else {
            // Keep the row on the axis so every index still lines up with the
            // matrix, but give it its own gene so it can never be paired with a
            // real one. The warning below is the only place this surfaces.
            if skipped.len() < 3 {
                skipped.push(name);
            } else {
                skipped.push("");
            }
            let g = ids.len() as u32;
            ids.insert(name.clone(), g);
            gene_names.push(name.clone());
            row_to_gene.push(g);
            row_is_nascent.push(false);
            continue;
        };
        let gid = match ids.get(gene) {
            Some(&g) => g,
            None => {
                let g = ids.len() as u32;
                ids.insert(gene.into(), g);
                gene_names.push(gene.into());
                g
            }
        };
        row_to_gene.push(gid);
        row_is_nascent.push(is_nascent);
    }
    if !skipped.is_empty() {
        let shown: Vec<&str> = skipped.iter().copied().filter(|s| !s.is_empty()).collect();
        log::warn!(
            "{} of {} feature rows are not `{{gene}}/count/{{spliced|unspliced}}` and were \
             given their own single-track gene id — they cannot pair across tracks and will \
             contribute nothing but noise. Examples: {}. Both gem models expect a \
             gene-level count matrix (`*_genes.zarr.zip`); a mixed-modality or per-site \
             matrix is not a supported input.",
            skipped.len(),
            feature_names.len(),
            shown.join(", ")
        );
    }
    let n_genes = gene_names.len();
    (
        GeneTrackMap {
            row_to_gene,
            row_is_nascent,
            n_genes,
        },
        gene_names,
    )
}

#[cfg(test)]
#[path = "rows/tests.rs"]
mod tests;
