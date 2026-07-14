//! **Abundance-stratified sampling of marker genes** — so a null gene set is matched on the
//! nuisance covariate that would otherwise decide the test on its own.
//!
//! # The bias
//!
//! This is GOseq's problem [Young et al., *Genome Biol.* 2010]. GOseq observed that in RNA-seq a
//! *longer* gene collects more reads, so it is more likely to be called differentially expressed —
//! and a plain hypergeometric over a DE gene list therefore rewards categories that happen to be
//! full of long genes, not categories that are biologically real. The remedy is to make the null
//! share that covariate. (`graph-embedding-util`'s `type_annotation::gene_strata` is the same fix
//! in the embedding's coordinate system, where the covariate is `‖e_g‖`.)
//!
//! Our covariate is the gene's **abundance** in the cluster profile, `Σ_k profile[g, k]`.
//!
//! Two things make it decisive here. A gene the data never detected has zero specificity in every
//! cluster, so it sorts to the **bottom of every ranking** and can never contribute a hit step —
//! and a uniform draw from all genes is full of them — measured, **~30% of the gene pool is
//! undetected**, and the pool's mean expression is a *quarter* of a real marker panel's. A null
//! made largely of genes that cannot possibly be enriched is trivially easy to beat.
//!
//! That alone would only inflate everything by a constant. What makes it *decision-relevant* is
//! that the inflation **differs by cell type**, because panels differ in how well-expressed their
//! markers are. The label is `argmax_c es_std[k, c]`, so a cell type whose markers happen to be
//! abundant out-scores one whose markers are not — on abundance, not on biology. Measured before
//! this module existed: the rank correlation between a cell type's mean `es_std` and its markers'
//! mean expression was **+1.000** across every type — a *perfect* monotone, which no biology
//! produces.
//!
//! # The fix
//!
//! Bin the genes by abundance and draw **within strata**, reproducing each panel's own abundance
//! profile exactly. The abundance advantage then appears on *both* sides of the comparison and
//! cancels, and what remains to be tested is what we wanted all along: are these genes specific to
//! this cluster?
//!
//! (GOseq cannot permute — it has one gene list — so it approximates with a length-weighted
//! Wallenius noncentral hypergeometric. We *can* permute, so we get the exact conditional null and
//! skip the noncentral approximation entirely.)

use crate::Mat;
use rand::RngExt;

#[cfg(test)]
mod tests;

/// Genes a stratum needs before it is allowed to exist.
///
/// **A stratum with one gene in it cannot be shuffled.** Stratifying is a *constraint* on the
/// null, and pushed far enough it constrains the null into the identity: with as many bins as
/// genes, every gene is alone in its bin, no swap is possible, the "random" panel is the real one,
/// and the p-value is 1 for everything. That failure is silent — the null still runs, still
/// produces numbers, and tests nothing at all. So the bin count is capped to leave partners.
const MIN_PER_STRATUM: usize = 10;

/// Abundance bins to aim for. Deciles are the usual quantile choice, and are a **ceiling** — see
/// [`GeneStrata::by_abundance`].
const N_STRATA: usize = 10;

/// The gene pool, binned by abundance.
pub struct GeneStrata {
    /// `stratum[g]` = which bin gene `g` falls in.
    stratum: Vec<usize>,
    /// `members[s]` = the gene indices in bin `s`. Never empty for a bin that exists.
    members: Vec<Vec<u32>>,
}

impl GeneStrata {
    /// Bin every gene into at most [`N_STRATA`] equal-count bins of increasing abundance
    /// (`Σ_k profile[g, k]`).
    ///
    /// Equal-count (quantile) bins rather than equal-width: expression is heavy-tailed, so
    /// equal-width bins would put almost every gene in the bottom bin and leave the top bin with
    /// nothing to swap a highly-expressed marker with.
    ///
    /// [`N_STRATA`] is a ceiling, not a target: the count is reduced until every bin holds at least
    /// [`MIN_PER_STRATUM`] genes, so a tiny gene pool gets few bins (and, in the limit, one — an
    /// unstratified draw) rather than a null that cannot move.
    #[must_use]
    pub fn by_abundance(profile_gk: &Mat) -> Self {
        let g = profile_gk.nrows();
        let k = profile_gk.ncols();
        let abundance: Vec<f32> = (0..g)
            .map(|gi| (0..k).map(|kk| profile_gk[(gi, kk)].max(0.0)).sum())
            .collect();
        Self::by_covariate(&abundance)
    }

    /// [`Self::by_abundance`] against a caller-supplied covariate, one value per gene.
    #[must_use]
    pub fn by_covariate(covariate: &[f32]) -> Self {
        let n = covariate.len();
        let k = N_STRATA.min(n / MIN_PER_STRATUM).max(1);

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| covariate[a].total_cmp(&covariate[b]));

        let mut stratum = vec![0usize; n];
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (rank, &gi) in order.iter().enumerate() {
            // Equal counts, remainder spread over the low bins.
            let s = (rank * k / n.max(1)).min(k - 1);
            stratum[gi] = s;
            members[s].push(gi as u32);
        }
        Self { stratum, members }
    }

    /// A single-bin pool — i.e. the old **unstratified** draw, uniform over every gene.
    ///
    /// This is the escape hatch (`--no-gene-strata`) and the control the tests measure against, so
    /// what stratification actually buys is checkable rather than asserted.
    #[must_use]
    pub fn unstratified(n_genes: usize) -> Self {
        Self {
            stratum: vec![0; n_genes],
            members: vec![(0..n_genes as u32).collect()],
        }
    }

    /// How many bins exist.
    #[must_use]
    pub fn n_strata(&self) -> usize {
        self.members.len()
    }

    /// A panel's **stratum profile**: `out[s]` holds the weights of the panel's genes that fall in
    /// bin `s`. This is what a null draw must reproduce — both the per-bin *counts* (so abundance
    /// cancels) and the *weights* those genes carried (so the walk's mass distribution does too).
    ///
    /// Weights stay attached to their own stratum rather than being pooled and re-dealt: IDF weight
    /// and expression are not independent, and a null that gave a highly-expressed gene a
    /// low-expressed marker's weight would be matched on neither.
    #[must_use]
    pub fn profile_of(&self, panel: &[(u32, f32)]) -> Vec<Vec<f32>> {
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); self.members.len()];
        for &(gi, w) in panel {
            out[self.stratum[gi as usize]].push(w);
        }
        out
    }

    /// Scratch for [`Self::draw_matched`] — a per-stratum copy of `members`, shuffled in place.
    ///
    /// Owned by the caller so a null running thousands of draws allocates it once, not per draw.
    #[must_use]
    pub fn scratch(&self) -> Vec<Vec<u32>> {
        self.members.clone()
    }

    /// Draw a gene set with the **same stratum profile** as the real panel: for each bin, as many
    /// genes as the panel had there, uniformly and without replacement within the bin, each
    /// carrying one of the panel's own weights from that bin.
    ///
    /// `out` is `(gene, weight)`, cleared first. A bin holding fewer genes than the panel wants
    /// yields everything it has — it cannot do better, and silently drawing from a neighbouring bin
    /// would break the very matching this exists for.
    pub fn draw_matched(
        &self,
        profile: &[Vec<f32>],
        scratch: &mut [Vec<u32>],
        out: &mut Vec<(u32, f32)>,
        rng: &mut impl RngExt,
    ) {
        out.clear();
        for (s, weights) in profile.iter().enumerate() {
            if weights.is_empty() {
                continue;
            }
            let bin = &mut scratch[s];
            let take = weights.len().min(bin.len());
            // Partial Fisher–Yates: only the first `take` of the permutation is needed, and a full
            // shuffle of a decile of the genome per draw would cost more than the ES walk it feeds.
            for j in 0..take {
                let r = rng.random_range(j..bin.len());
                bin.swap(j, r);
                out.push((bin[j], weights[j]));
            }
        }
    }
}
