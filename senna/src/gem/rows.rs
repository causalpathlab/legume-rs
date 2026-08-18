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
//!
//! The grammar itself now lives one crate down, in
//! [`auxiliary_data::feature_rows`], because `pinto` needs the same split for
//! spatial matrices and must not depend on an application crate to get it. What
//! stays here is the part that is genuinely gem's: the choice to keep an
//! unpairable row on the axis as its own single-track gene
//! ([`UnparsedRowPolicy::OwnGene`]) rather than fail. A gene-keyed model can
//! carry such a row harmlessly; a consumer that POOLS the two tracks cannot, and
//! pinto takes the strict policy for exactly that reason.

use auxiliary_data::feature_rows::{intern_count_rows, UnparsedRowPolicy};
use candle_util::data::indexed::GeneTrackMap;

/// Intern each row's gene key to a dense gene id, returning the row→gene map,
/// the nascent flags, and the id-ordered gene names.
#[must_use]
pub fn build_gene_track_map(feature_names: &[Box<str>]) -> (GeneTrackMap, Vec<Box<str>>) {
    let map = intern_count_rows(feature_names, UnparsedRowPolicy::OwnGene);
    if !map.unparsed.is_empty() {
        let shown: Vec<&str> = map
            .unparsed
            .iter()
            .take(3)
            .map(|&r| feature_names[r].as_ref())
            .collect();
        log::warn!(
            "{} of {} feature rows are not `{{gene}}/count/{{spliced|unspliced}}` and were \
             given their own single-track gene id — they cannot pair across tracks and will \
             contribute nothing but noise. Examples: {}. Both gem models expect a \
             gene-level count matrix (`*_genes.zarr.zip`); a mixed-modality or per-site \
             matrix is not a supported input.",
            map.unparsed.len(),
            feature_names.len(),
            shown.join(", ")
        );
    }
    let n_genes = map.n_genes();
    (
        GeneTrackMap {
            row_to_gene: map.row_to_gene,
            row_is_nascent: map.row_is_nascent,
            n_genes,
        },
        map.gene_names,
    )
}

#[cfg(test)]
#[path = "rows/tests.rs"]
mod tests;
