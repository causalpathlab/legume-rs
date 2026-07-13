//! **Norm-stratified sampling of marker genes** — so a null panel is matched on the nuisance
//! covariate that would otherwise decide the test on its own.
//!
//! # The bias
//!
//! This is GOseq's problem [Young et al., *Genome Biol.* 2010] in a different coordinate system.
//! GOseq observed that in RNA-seq a *longer* gene collects more reads, so it is more likely to be
//! called differentially expressed — and therefore a plain hypergeometric over a DE gene list
//! rewards categories that happen to be full of long genes, not categories that are biologically
//! real. The remedy is to make the null share that covariate.
//!
//! Our covariate is not length. It is the gene's **embedding norm** `‖e_g‖`.
//!
//! A type's prototype is the (IDF-weighted) mean of its markers' embeddings, so a type whose
//! markers are long vectors gets a long centroid. And because the cells sit far from every
//! centroid (`‖cell‖ ≫ ‖centroid‖`), the Euclidean rule degenerates —
//! `argmin_t ‖x − c_t‖² = argmax_t (⟨x, c_t⟩ − ½‖c_t‖²)` — and a longer centroid simply wins more
//! cells, largely irrespective of *where it points*. Measured on cord blood: the rank correlation
//! between a type's centroid norm and its share of the cells is **+0.93**.
//!
//! So if the null draws genes **uniformly** from the marker pool, every null panel inherits the
//! *pool's* mean norm. A type whose own markers are above that mean beats its null on norm alone;
//! one below it loses on norm alone — and no biology is tested either way. Measured, before this
//! module existed: the panel null's p-value was almost a monotone function of a type's mean gene
//! norm (the three types that "passed" were simply the three with the longest markers).
//!
//! # The fix
//!
//! Bin the pool by `‖e_g‖` and draw **within strata**, so the null panel reproduces the type's own
//! norm profile exactly. The norm advantage then appears on *both* sides of the comparison and
//! cancels, and what remains to be tested is what we wanted all along: do these genes point the
//! same way, and do they point at cells.
//!
//! (GOseq cannot permute — it has one gene list — so it approximates with a length-weighted
//! Wallenius noncentral hypergeometric. We *can* permute, so we get the exact conditional null and
//! skip the noncentral approximation entirely.)

use crate::null_call::live_row;

/// The marker-gene pool, binned by embedding norm.
pub(super) struct GeneStrata {
    /// `stratum[i]` = which bin pool position `i` falls in.
    pub stratum: Vec<usize>,
    /// `members[k]` = the pool positions in bin `k`. Never empty for a bin that exists.
    pub members: Vec<Vec<usize>>,
}

/// Genes a stratum needs before it is allowed to exist.
///
/// **A stratum with one gene in it cannot be shuffled.** Stratifying is a *constraint* on the
/// null, and pushed far enough it constrains the null into the identity: with as many bins as
/// genes, every gene is alone in its bin, no swap is possible, the "shuffled" panel is the real
/// one, and the p-value is 1 for everything. That failure is silent — the null still runs, still
/// produces numbers, and tests nothing at all. So the bin count is capped to leave partners.
const MIN_PER_STRATUM: usize = 10;

/// Norm bins to aim for. Deciles are the usual quantile choice, and are a ceiling — see
/// [`GeneStrata::by_norm`]. No caller wants a different value, so it is not a parameter.
const N_STRATA: usize = 10;

impl GeneStrata {
    /// Bin `pool` (indices into `feature_emb`'s rows) into at most `n_strata` equal-count bins of
    /// increasing `‖e_g‖`. Dead rows are assumed already filtered out by the caller.
    ///
    /// Equal-count (quantile) bins rather than equal-width: the norm distribution is heavy-tailed
    /// (measured 0.87 to 6.22, with the top decile carrying most of the range), so equal-width
    /// bins would leave the top bin nearly empty and there would be nothing to swap a long gene
    /// with.
    ///
    /// `n_strata` is a **ceiling**, not a target: the count is reduced until every bin holds at
    /// least [`MIN_PER_STRATUM`] genes, so a small panel gets few bins (and, in the limit, one —
    /// an unstratified shuffle) rather than a null that cannot move.
    pub fn by_norm(feature_emb: &[f32], pool: &[u32], h: usize) -> Self {
        let n = pool.len();
        let k = N_STRATA.min(n / MIN_PER_STRATUM).max(1);

        // Compute each norm once. Calling it from inside the comparator would evaluate an
        // `h`-length dot product twice per comparison — `O(n log n)` of them instead of `n`.
        let norms: Vec<f32> = pool
            .iter()
            .map(|&g| {
                live_row(feature_emb, g as usize, h)
                    .map(|r| r.iter().map(|&x| x * x).sum::<f32>().sqrt())
                    .unwrap_or(0.0)
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| norms[a].total_cmp(&norms[b]));

        let mut stratum = vec![0usize; n];
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (rank, &i) in order.iter().enumerate() {
            // Equal counts, remainder spread over the low bins.
            let s = (rank * k / n.max(1)).min(k - 1);
            stratum[i] = s;
            members[s].push(i);
        }
        Self { stratum, members }
    }
}
