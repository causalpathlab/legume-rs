//! The gene/track map shared by the gem-format loaders.
//!
//! gem-format inputs carry two rows per gene (`{gene}/count/spliced` and
//! `{gene}/count/unspliced`); a gene-keyed model needs to know, for each row
//! of that axis, which gene it belongs to and which track it is. That map is
//! built once ([`GeneTrackMap`]) and shared by every consumer that needs the
//! pairing rather than the raw row axis.

/// Maps each row of a gem-format feature axis to `(gene id, is_nascent)`.
///
/// Built by the caller from the row names (faba splits
/// `{gene}/count/{spliced|unspliced}`), so this crate stays free of any naming
/// convention.
#[derive(Clone)]
pub struct GeneTrackMap {
    /// `row_to_gene[r]` = gene id of row `r`.
    pub row_to_gene: Vec<u32>,
    /// `row_is_nascent[r]` = true when row `r` is the unspliced track.
    pub row_is_nascent: Vec<bool>,
    /// Number of distinct genes `G`.
    pub n_genes: usize,
}

impl GeneTrackMap {
    /// Per-gene row ids `(nascent_row, mature_row)`, `None` where a gene lacks
    /// that track (a spliced-only input has no nascent rows at all).
    #[must_use]
    pub fn per_gene_rows(&self) -> (Vec<Option<u32>>, Vec<Option<u32>>) {
        let mut nascent = vec![None; self.n_genes];
        let mut mature = vec![None; self.n_genes];
        for (r, (&g, &is_n)) in self
            .row_to_gene
            .iter()
            .zip(self.row_is_nascent.iter())
            .enumerate()
        {
            let slot = if is_n { &mut nascent } else { &mut mature };
            slot[g as usize] = Some(r as u32);
        }
        (nascent, mature)
    }
}
