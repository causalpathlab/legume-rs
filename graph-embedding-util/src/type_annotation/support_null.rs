//! **Permutation null for the bootstrap support** — turning `label_support` into a p-value.
//!
//! The bootstrap ([`super::marker_bootstrap`]) reports, per cell, the fraction of replicates that
//! agreed on its label. That number is then compared against a hand-picked bar (`--min-support`,
//! 0.5) to decide whether to call the cell at all. The bar is arbitrary, and worse, it is **not
//! scale-free**: with `C` types, chance agreement is `1/C`, so 0.5 sits at 3× chance on a 6-type
//! panel and 12× chance on a 24-type one — the *same* flag is a different test on different
//! panels, and their abstention rates are not comparable.
//!
//! So calibrate it. Ask what support a cell gets when **the panel carries no information about
//! cell type**, and keep the cells whose support beats that, at a controlled FDR.
//!
//! # What is permuted
//!
//! The **gene → type assignment**. The flattened list of `(gene, type)` marker pairs has its gene
//! column shuffled, so every type keeps its exact panel size and its exact IDF weight multiset,
//! and the gene multiset as a whole is unchanged — only *which type a gene belongs to* is
//! destroyed. That is the literal statement of "the panel means nothing", and it is the direct
//! analogue of the label shuffle the over-representation test already runs on the cells.
//!
//! **The shuffle is stratified by gene norm** ([`super::gene_strata`]), and it has to be. A free
//! shuffle moves long genes between types, so a type whose real markers are long vectors would get
//! a short-vector null panel — and since a longer centroid wins cells almost irrespective of
//! direction, its null support would collapse and it would look significant for a reason that has
//! nothing to do with biology. Permuting within norm strata keeps every type's norm profile
//! exactly and destroys only gene *identity*. This is GOseq's correction, in the coordinate the
//! decision is actually made in.
//!
//! Two alternatives were considered and rejected:
//!
//! * **Shuffling the cells' coordinates** destroys the cluster structure too, so the null run is
//!   no longer doing the same computation — Leiden would be clustering noise. Not comparable.
//! * **Rotating the centroids** relative to the cells preserves norms, pairwise distances and the
//!   panel's internal scatter, which is attractive. But it also preserves panel *coherence*, so a
//!   rotated-but-coherent panel stably picks the wrong cells and yields *high* null support. That
//!   asks "does this panel point at anything real" — which is [`super::panel_null`]'s question,
//!   not this one's.
//!
//! # The statistic, and why there is no closed form
//!
//! The statistic is the support itself, `s_i = max_t n_it / B` over the `B` replicates. The null
//! **must use the same `B`**: `s_i` is a maximum over noisy proportions and is therefore biased
//! upward, the more so the smaller `B` is. Comparing a `B = 200` support against a null built at a
//! different `B` (or against exact probabilities) would tilt the whole test. Matching `B` makes
//! that bias appear on both sides and cancel — the same reason [`super::panel_null`] draws its
//! null panels at the same *size*.
//!
//! There is no CLT shortcut, for two reasons:
//!
//! 1. **It is a maximum, not a mean.** Even under exchangeable labels, `max_t Binomial(B, 1/C)/B`
//!    is an extreme-value statistic — skewed, not asymptotically normal in `B`.
//! 2. **The variance that matters is not the one CLT describes.** A central limit in `B` gives the
//!    Monte-Carlo error of `s_i` *given a panel*. The null we want is the spread of `s_i` *across*
//!    shuffled panels, which is a property of where the collapsed null centroids land in this
//!    particular embedding. No closed form; simulate it.
//!
//! Nor can the null be pooled across cells to save draws. Null support is strongly
//! **cell-dependent**: a cell buried deep inside a dense cluster gets high support under *any*
//! panel, because it is stably assigned to whichever centroid happens to be nearest. Pooling would
//! hand it a small p-value for a reason that has nothing to do with markers. The null is per-cell.
//!
//! # Why it is affordable
//!
//! Naively this is `P × B` full replicates. It is not, because **the partitions do not depend on
//! the panel**. Re-clustering is ~94% of a replicate's cost, and the `B` partitions are drawn once
//! for the observed bootstrap and then *reused* by every shuffle ([`super::term_ora::Partition`]).
//! What is left per replicate — rebuild `C` centroids, re-assign `n` cells, re-run the
//! over-representation test — is cheap and vectorized. `P` shuffles then fan out over rayon.
//!
//! It also means the null shares the *exact* partitions with the observed run, so partition
//! variability is held fixed between them rather than being an extra source of difference.

use super::gene_strata::GeneStrata;
use super::marker_bootstrap::{assign_nearest, label_support, LivePanel, MarkerBootstrapConfig};
use super::markers::marker_gene_pool;
use super::term_ora::{replicate_label, Partition, TermOraConfig};
use crate::null_call::live_row;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[cfg(test)]
mod tests;

/// Domain-separates this null's RNG from the ORA's label shuffle, the bootstrap's panel resample,
/// and the panel null's gene draws, so the four cannot share a stream off one `--seed`.
const SUPPORT_NULL_STREAM: u64 = 0x5044_0E17_5EED;

/// What the support null found. All per cell, length `n`.
pub struct SupportNull {
    /// Shuffled panels drawn.
    pub n_perm: usize,
    /// `P(support under a meaningless panel ≥ the support observed)`, +1-smoothed. **Small ⇒ this
    /// cell's call is more reproducible than a panel carrying no type information could manage.**
    pub p: Vec<f32>,
    /// Benjamini–Hochberg `q` of `p`, across the cells. This is what a cutoff should be set on:
    /// unlike `--min-support`, an FDR means the same thing whatever the number of types.
    pub q: Vec<f32>,
    /// Mean support the cell attained under the shuffled panels — the bar it had to clear.
    pub null_support: Vec<f32>,
}

/// The gene column of the flat `(gene, type)` marker list, grouped by norm stratum — shuffling
/// happens *within* these buckets, so no type's norm profile can change.
struct StratifiedShuffle {
    /// `slots[k]` = the positions in the flat marker list whose gene sits in stratum `k`.
    slots: Vec<Vec<usize>>,
    /// `genes[k]` = the genes currently occupying those positions (permuted in place).
    genes: Vec<Vec<u32>>,
    /// `(type, weight)` of each position in the flat list, in order.
    slot_of: Vec<(usize, f32)>,
}

impl StratifiedShuffle {
    fn new(feature_emb: &[f32], type_markers: &[Vec<(u32, f32)>], h: usize) -> Self {
        let g = feature_emb.len() / h;
        let pool: Vec<u32> = marker_gene_pool(type_markers, g)
            .into_iter()
            .filter(|&gi| live_row(feature_emb, gi as usize, h).is_some())
            .collect();
        let strata = GeneStrata::by_norm(feature_emb, &pool, h);
        let bin_of: std::collections::HashMap<u32, usize> = pool
            .iter()
            .enumerate()
            .map(|(i, &gi)| (gi, strata.stratum[i]))
            .collect();

        // A dead gene is in no stratum; park it in its own bucket so it never moves (it carries
        // no signal either way, and shuffling it would only add noise).
        let n_bins = strata.members.len() + 1;
        let dead = n_bins - 1;
        let mut slots: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
        let mut genes: Vec<Vec<u32>> = vec![Vec::new(); n_bins];
        let mut slot_of = Vec::new();
        for (t, markers) in type_markers.iter().enumerate() {
            for &(gi, w) in markers {
                let k = bin_of.get(&gi).copied().unwrap_or(dead);
                slots[k].push(slot_of.len());
                genes[k].push(gi);
                slot_of.push((t, w));
            }
        }
        Self {
            slots,
            genes,
            slot_of,
        }
    }

    /// One shuffled panel: permute the genes **within each norm stratum**, then rebuild the
    /// per-type marker lists. Every type keeps its panel size, its weight multiset, and — the
    /// point of the strata — its norm profile.
    fn draw(&self, rng: &mut SmallRng, out: &mut [Vec<(u32, f32)>]) {
        // Permute within each bin. `genes` is cloned per draw rather than shuffled in place, so
        // this is `&self` — which is what lets one shuffler be shared across the rayon fan-out
        // instead of being rebuilt (pool + sort + hash map) on every one of `P` shuffles.
        let mut genes = self.genes.clone();
        for g in &mut genes {
            g.shuffle(rng);
        }
        for v in out.iter_mut() {
            v.clear();
        }
        for (bin, slots) in self.slots.iter().enumerate() {
            for (j, &slot) in slots.iter().enumerate() {
                let (t, w) = self.slot_of[slot];
                out[t].push((genes[bin][j], w));
            }
        }
    }
}

/// Run the support null. `partitions` are the *same* `B` groupings the observed bootstrap used;
/// `obs_support` is its per-cell `label_support`.
///
/// `None` means **do not gate on this null** — it did not run, or it ran too few shuffles to say
/// anything at the requested α. The annotation itself is unaffected either way: this is a
/// calibration layer on top of a result that already exists, so when it cannot speak it must fall
/// silent rather than take the run down with it.
#[allow(clippy::too_many_arguments)]
pub fn run_support_null(
    feature_emb: &[f32],
    cell_flat: &[f32],
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
    partitions: &[Partition],
    lnfact: &[f64],
    obs_support: &[f32],
    n_perm: usize,
    cfg: &TermOraConfig,
    bcfg: &MarkerBootstrapConfig,
) -> Result<Option<SupportNull>> {
    let c = type_markers.len();
    let n = cell_flat.len() / h;
    let b_draws = bcfg.n_boot.max(1);

    // The bootstrap already finished, and its annotation is shippable. If the user interrupted it,
    // the flag is *latched*, so every shuffle below would be filtered out and we would come back
    // with nothing — and reporting that as an error would throw away the annotation the user is
    // waiting for. Decline instead.
    if crate::stop::stopped() {
        log::warn!(
            "skipping the support null: the run was interrupted, and the bootstrap's own result \
             is what you asked to keep. `label_support` is reported uncalibrated."
        );
        return Ok(None);
    }

    // Built ONCE. It is a pure function of the embedding and the panel — nothing about it varies
    // per shuffle — and it costs a pool scan, a sort and a hash map, so rebuilding it inside the
    // fan-out would pay for all of that `P` times over.
    let shuffler = StratifiedShuffle::new(feature_emb, type_markers, h);

    // Folded, not collected: only the two running totals below survive a shuffle. Keeping the
    // per-shuffle vectors instead would hold `P × 2n` of them at once — 1.8 GB at `P = 10 000`,
    // for numbers that are summed and discarded. See `stop::par_reduce_replicates`.
    let folded = crate::stop::par_reduce_replicates(
        n_perm,
        "support null",
        |p| {
            let mut rng = SmallRng::seed_from_u64(
                cfg.seed ^ SUPPORT_NULL_STREAM ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut shuffled: Vec<Vec<(u32, f32)>> = vec![Vec::new(); c];
            shuffler.draw(&mut rng, &mut shuffled);

            // The live/dead status of a gene is a property of the embedding, not of the type it is
            // listed under, so the shuffled panel is exactly as *trained* as the real one — this
            // holds "is the gene in the model?" fixed and isolates "does the panel mean anything?".
            let panel = LivePanel::new(feature_emb, &shuffled, h);

            // `counts[i * (c + 1) + t]` over the B replicates; column `c` is `unassigned`.
            let mut counts = vec![0f32; n * (c + 1)];
            let mut drawn = 0f32;
            let mut centroids = vec![0f32; c * h];
            for d in 0..b_draws {
                panel.resample_into(
                    feature_emb,
                    h,
                    cfg.seed ^ (p as u64) << 40,
                    d,
                    &mut centroids,
                );
                let fine = assign_nearest(cell_flat, n, &centroids, c, h, &panel.usable);
                let partition = &partitions[d % partitions.len()];
                if let Some((per_cell, _)) = replicate_label(
                    &fine, &centroids, cell_flat, partition, n, c, h, lnfact, cfg,
                )? {
                    drawn += 1.0;
                    for (i, &col) in per_cell.iter().enumerate() {
                        counts[i * (c + 1) + col] += 1.0;
                    }
                }
            }

            // The same statistic as the observed run, computed by the same function — see
            // `marker_bootstrap::label_support`. (These were two different maxima once, and the
            // p-value was quietly comparing incomparable things.)
            let denom = drawn.max(1.0);
            let (ge, sum): (Vec<u32>, Vec<f64>) = counts
                .chunks_mut(c + 1)
                .enumerate()
                .map(|(i, row)| {
                    for v in row.iter_mut() {
                        *v /= denom;
                    }
                    let s = label_support(row);
                    (u32::from(s >= obs_support[i]), f64::from(s))
                })
                .unzip();
            Ok(Tally { ge, sum })
        },
        Tally::merge,
    )?;

    let Some((done, acc)) = folded else {
        log::warn!("the support null completed no shuffles; `label_support` is uncalibrated.");
        return Ok(None);
    };

    // **A permutation null cannot report a p below `1 / (P + 1)`.** That floor is a property of the
    // +1-smoothed estimator, not of the data, so with too few shuffles *every* cell lands above
    // `fdr_alpha` — and the gate downstream would then quietly unassign the entire dataset while
    // exiting 0. An interrupted null is not a smaller null; it is one that cannot answer the
    // question it was asked, and the honest move is to say so and leave the calls alone.
    let floor = 1.0 / (done as f32 + 1.0);
    if floor >= cfg.fdr_alpha {
        log::warn!(
            "the support null ran only {done} shuffle(s): the smallest p it can report is \
             1/{} = {floor:.3}, which is not below --fdr-alpha {}. It cannot reject anything, so \
             the calibrated cutoff is NOT applied and the calls stand on `--min-support` alone. \
             Re-run without interrupting, or with --support-perm >= {}.",
            done + 1,
            cfg.fdr_alpha,
            (1.0 / cfg.fdr_alpha).ceil() as usize,
        );
        return Ok(None);
    }

    let p: Vec<f32> = acc
        .ge
        .iter()
        .map(|&k| (f64::from(k) + 1.0) as f32 / (done as f32 + 1.0))
        .collect();
    let q = enrichment::bh_fdr(&p);
    let null_support = acc
        .sum
        .iter()
        .map(|&s| (s / done as f64) as f32)
        .collect::<Vec<f32>>();

    Ok(Some(SupportNull {
        n_perm: done,
        p,
        q,
        null_support,
    }))
}

/// The only two things a shuffle leaves behind, per cell.
///
/// `sum` is `f64` and stays `f64` until the final divide. It is a sum of `P` values in `[0, 1]`,
/// and at `P = 10 000` an `f32` running total is already large enough that each new term lands
/// ~13 bits down in the mantissa — the classic "adding a small number to a big one" drift, right
/// in the mean the p-value is reported against.
struct Tally {
    /// Shuffles whose support for this cell reached the observed one.
    ge: Vec<u32>,
    /// Sum, over shuffles, of the support this cell got.
    sum: Vec<f64>,
}

impl Tally {
    /// Associative — required by `try_reduce_with`, which combines subtrees in whatever order
    /// rayon split them, never in shuffle order.
    fn merge(mut a: Self, b: Self) -> Self {
        for (x, y) in a.ge.iter_mut().zip(&b.ge) {
            *x += y;
        }
        for (x, y) in a.sum.iter_mut().zip(&b.sum) {
            *x += y;
        }
        a
    }
}
