//! Gene-gene co-occurrence edges with hop-sampled negatives.
//!
//! A cell is a bag of gene tokens. Two genes drawn from the same bag are a
//! positive pair; the negative for that pair keeps the first gene and replaces
//! the second with one drawn from ANOTHER cell. Which other cell is the whole
//! design. A discriminator trained against a noise distribution `q` converges
//! to `log p(g,h) − log p(g)·q(h)`, so the negatives decide what is learned:
//!
//! - any other cell → the corpus-marginal mutual information, dominated by
//!   cell type, which the cell-gene edges already carry;
//! - a cell of the same type in a different state → co-occurrence *beyond*
//!   type, the part a topic latent cannot represent;
//! - the nearest neighbour → positive ≈ negative, and nothing to learn.
//!
//! The collapse tree already ranks "how different". A negative drawn from under
//! the ancestor `t` levels above the positive's finest group is a cell that
//! agrees with it to that depth and differs below. Drawing `t` at random per
//! negative puts every resolution into every step, weighted by [`HopWeights`],
//! and makes the plain uniform negative the special case `t = root`. A group
//! with no partner at `t` resolves at the first hop above it that has one, and
//! the root always does — the tree from the collapse is shallow and nearly half
//! its leaves are singletons all the way up, so this fallback is the common
//! path, not the exception.
//!
//! The score is symmetric, `ρ_g·ρ_h + b_g + b_h`, on the same `ρ` the cell-gene
//! edges train: co-occurrence in a bag has no direction, and a second
//! "context" matrix would take the signal somewhere the cell side never reads.

use crate::loss::feat::{gather_feature_rows, PerBatchStratifiedCellSampler};
use crate::loss::{logistic_nce, softmax_nce, NceObjective};
use crate::model::JointEmbedModel;
use candle_util::candle_core::{Device, Result, Tensor};
use rand::{Rng, RngExt};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

/// Rejection draws before a deterministic fallback. Every rejection loop here
/// has success probability ≥ 1/2 per draw, so this bound is never reached in
/// practice; it exists so a pathological input cannot spin.
const MAX_REJECT: usize = 32;

/////////////////
// Hop weights //
/////////////////

/// How the hop count `t ∈ 1..=T` is distributed, `T` being the tree height:
/// `t = 1` is the finest group's parent (sisters), `t = T` is the root (any
/// other finest group, i.e. the plain uniform negative).
#[derive(Clone, Debug, PartialEq)]
pub enum HopWeights {
    /// Equal mass on every hop.
    Uniform,
    /// Mass `T + 1 − t`: sisters most, the root least.
    Near,
    /// Mass `t`: the root most, sisters least.
    Far,
    /// All mass on `t = 1`.
    Sisters,
    /// All mass on `t = T` — every negative is a plain uniform draw over the
    /// other finest groups.
    Root,
    /// Explicit unnormalized mass per hop, length `T`.
    Explicit(Vec<f32>),
}

impl HopWeights {
    /// Unnormalized mass on hops `1..=n_hops`.
    pub fn weights(&self, n_hops: usize) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(n_hops >= 1, "a hop tree needs at least one level");
        let w = match self {
            Self::Uniform => vec![1.0; n_hops],
            Self::Near => (1..=n_hops).map(|t| (n_hops + 1 - t) as f32).collect(),
            Self::Far => (1..=n_hops).map(|t| t as f32).collect(),
            Self::Sisters => {
                let mut v = vec![0.0; n_hops];
                v[0] = 1.0;
                v
            }
            Self::Root => {
                let mut v = vec![0.0; n_hops];
                v[n_hops - 1] = 1.0;
                v
            }
            Self::Explicit(v) => {
                anyhow::ensure!(
                    v.len() == n_hops,
                    "hop weights have {} entries but the tree has {n_hops} hops",
                    v.len()
                );
                anyhow::ensure!(
                    v.iter().all(|&x| x >= 0.0 && x.is_finite()),
                    "hop weights must be finite and non-negative"
                );
                anyhow::ensure!(v.iter().sum::<f32>() > 0.0, "hop weights sum to zero");
                v.clone()
            }
        };
        Ok(w)
    }
}

//////////////
// Hop tree //
//////////////

/// The collapse tree, indexed for hop-sampled partner groups.
///
/// Built from `cell_to_pb_per_level` (coarsest first). Only cells flagged
/// `active` — those with at least one expressed feature in some batch sampler —
/// are placed, so every group listed here can supply a gene.
pub struct HopTree {
    n_levels: usize,
    /// Coarsest first; `[level][cell]` is the cell's group at that level.
    cell_to_pb_per_level: Vec<Vec<u32>>,
    /// Finest group → the active cells in it.
    cells_in_fine: Vec<Vec<u32>>,
    /// For each ancestor level `ℓ < n_levels − 1`: group → the distinct finest
    /// groups beneath it (sorted).
    fine_under: Vec<Vec<Vec<u32>>>,
    /// Every finest group with an active cell — the root's children.
    all_fine: Vec<u32>,
}

impl HopTree {
    pub fn new(cell_to_pb_per_level: &[Vec<usize>], active: &[bool]) -> anyhow::Result<Self> {
        let n_levels = cell_to_pb_per_level.len();
        anyhow::ensure!(n_levels >= 1, "hop tree: no collapse levels");
        let n_cells = cell_to_pb_per_level[0].len();
        anyhow::ensure!(
            cell_to_pb_per_level.iter().all(|l| l.len() == n_cells),
            "hop tree: levels disagree on the cell count"
        );
        anyhow::ensure!(
            active.len() == n_cells,
            "hop tree: `active` has {} entries for {n_cells} cells",
            active.len()
        );
        let c2p: Vec<Vec<u32>> = cell_to_pb_per_level
            .iter()
            .map(|l| l.iter().map(|&p| p as u32).collect())
            .collect();
        let finest = &c2p[n_levels - 1];
        let n_fine = finest.iter().max().map_or(0, |&m| m as usize + 1);

        let mut cells_in_fine: Vec<Vec<u32>> = vec![Vec::new(); n_fine];
        for (c, &f) in finest.iter().enumerate() {
            if active[c] {
                cells_in_fine[f as usize].push(c as u32);
            }
        }
        let all_fine: Vec<u32> = (0..n_fine as u32)
            .filter(|&f| !cells_in_fine[f as usize].is_empty())
            .collect();

        let mut fine_under: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n_levels - 1);
        for level in &c2p[..n_levels - 1] {
            let n_groups = level.iter().max().map_or(0, |&m| m as usize + 1);
            let mut under: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
            for (c, &g) in level.iter().enumerate() {
                if active[c] {
                    under[g as usize].push(finest[c]);
                }
            }
            for v in &mut under {
                v.sort_unstable();
                v.dedup();
            }
            fine_under.push(under);
        }

        Ok(Self {
            n_levels,
            cell_to_pb_per_level: c2p,
            cells_in_fine,
            fine_under,
            all_fine,
        })
    }

    /// Tree height: the number of distinct hops, `1..=n_hops`.
    #[must_use]
    pub fn n_hops(&self) -> usize {
        self.n_levels
    }

    /// The number of finest groups that hold an active cell.
    #[must_use]
    pub fn n_finest_groups(&self) -> usize {
        self.all_fine.len()
    }

    /// A cell's finest group.
    #[must_use]
    pub fn fine_of(&self, cell: usize) -> u32 {
        self.cell_to_pb_per_level[self.n_levels - 1][cell]
    }

    /// The active cells in a finest group.
    #[must_use]
    pub fn cells_in(&self, fine: u32) -> &[u32] {
        &self.cells_in_fine[fine as usize]
    }

    /// The finest groups under the ancestor `t` hops above `cell`'s finest
    /// group; the whole forest at `t == n_hops`. Includes the cell's own group.
    #[must_use]
    pub fn groups_at_hop(&self, cell: usize, t: usize) -> &[u32] {
        debug_assert!((1..=self.n_levels).contains(&t), "hop {t} out of range");
        if t >= self.n_levels {
            return &self.all_fine;
        }
        let level = self.n_levels - 1 - t;
        let g = self.cell_to_pb_per_level[level][cell] as usize;
        &self.fine_under[level][g]
    }

    /// Draw a finest group other than `cell`'s own, from the smallest hop `≥ t`
    /// whose ancestor holds one. Returns the group and the hop it resolved at;
    /// `None` only if no other finest group exists at all.
    pub fn draw_partner(&self, cell: usize, t: usize, rng: &mut impl Rng) -> Option<(u32, usize)> {
        let own = self.fine_of(cell);
        for hop in t.max(1)..=self.n_levels {
            let cands = self.groups_at_hop(cell, hop);
            let n = cands.len();
            if n == 0 || (n == 1 && cands[0] == own) {
                continue;
            }
            // `own` is at most one entry, so each draw misses it with probability
            // at least one half.
            for _ in 0..MAX_REJECT {
                let f = cands[rng.random_range(0..n)];
                if f != own {
                    return Some((f, hop));
                }
            }
            if let Some(&f) = cands.iter().find(|&&f| f != own) {
                return Some((f, hop));
            }
        }
        None
    }
}

//////////////////////////
// Sampler and the batch //
//////////////////////////

/// One minibatch of gene-pair edges. Row-major like [`crate::loss::EdgeBatch`]:
/// the negatives for positive `b` sit at `[b*K..(b+1)*K]`.
pub struct GenePairBatch {
    /// `[B]` the cell each positive pair was drawn from (diagnostics; the loss
    /// does not read it).
    pub pos_cells: Vec<u32>,
    /// `[B]` first gene of the pair — kept in the negatives.
    pub pos_g: Vec<u32>,
    /// `[B]` second gene of the pair — replaced in the negatives.
    pub pos_h: Vec<u32>,
    /// `[B*K]` the cell each negative gene was drawn from (diagnostics).
    pub neg_cells: Vec<u32>,
    /// `[B*K]` negative genes.
    pub neg_h: Vec<u32>,
    /// `[B*K]` the hop each negative resolved at (diagnostics).
    pub hops: Vec<u8>,
    pub n_negatives: usize,
}

/// Draws gene-pair positives from the per-batch cell samplers and hop-matched
/// negatives from the collapse tree.
///
/// Built against a specific `&[PerBatchStratifiedCellSampler]` and only valid
/// with that same slice at sample time: the lookup from a global cell id to its
/// `(sampler, local)` position is taken at construction.
pub struct GenePairSampler {
    tree: HopTree,
    /// Global cell id → `(batch sampler index, local index)`, `None` for a cell
    /// with no expressed feature.
    cell_lookup: Vec<Option<(u32, u32)>>,
    /// Picks `t − 1` for `t ∈ 1..=n_hops`. Built over `f64`: `WeightedIndex`
    /// accumulates a running sum and `f32` loses the tail.
    hop_picker: WeightedIndex<f64>,
    hop_weights: Vec<f32>,
}

impl GenePairSampler {
    pub fn new(
        samplers: &[PerBatchStratifiedCellSampler],
        cell_to_pb_per_level: &[Vec<usize>],
        hops: &HopWeights,
    ) -> anyhow::Result<Self> {
        let n_cells = cell_to_pb_per_level.first().map_or(0, Vec::len);
        let mut cell_lookup: Vec<Option<(u32, u32)>> = vec![None; n_cells];
        for (bi, s) in samplers.iter().enumerate() {
            for (li, &c) in s.active_cells.iter().enumerate() {
                cell_lookup[c as usize] = Some((bi as u32, li as u32));
            }
        }
        let active: Vec<bool> = cell_lookup.iter().map(Option::is_some).collect();
        let tree = HopTree::new(cell_to_pb_per_level, &active)?;
        anyhow::ensure!(
            tree.n_finest_groups() >= 2,
            "gene-pair negatives need at least two finest groups with active cells; found {}",
            tree.n_finest_groups()
        );
        let hop_weights = hops.weights(tree.n_hops())?;
        let hop_picker = WeightedIndex::new(hop_weights.iter().map(|&w| f64::from(w)))?;
        Ok(Self {
            tree,
            cell_lookup,
            hop_picker,
            hop_weights,
        })
    }

    #[must_use]
    pub fn n_hops(&self) -> usize {
        self.tree.n_hops()
    }

    /// The unnormalized mass per hop this sampler draws from.
    #[must_use]
    pub fn hop_weights(&self) -> &[f32] {
        &self.hop_weights
    }

    #[must_use]
    pub fn tree(&self) -> &HopTree {
        &self.tree
    }

    /// A gene of `cell`, drawn by count. `None` for an inactive cell.
    fn gene_of(
        &self,
        samplers: &[PerBatchStratifiedCellSampler],
        cell: u32,
        rng: &mut impl Rng,
    ) -> Option<u32> {
        let (bi, li) = self.cell_lookup[cell as usize]?;
        let pf = &samplers[bi as usize].per_cell[li as usize];
        Some(pf.features[pf.picker.sample(rng)])
    }

    /// Draw `batch_size` positives and `n_negatives` hop-matched negatives each.
    pub fn sample_batch(
        &self,
        samplers: &[PerBatchStratifiedCellSampler],
        batch_size: usize,
        n_negatives: usize,
        rng: &mut impl Rng,
    ) -> GenePairBatch {
        let mut pos_cells = Vec::with_capacity(batch_size);
        let mut pos_g = Vec::with_capacity(batch_size);
        let mut pos_h = Vec::with_capacity(batch_size);

        // Positives: a cell by the sampler's own picker, then two distinct genes
        // of it by count. A cell with a single expressed gene has no pair.
        'positives: for _ in 0..batch_size {
            for _ in 0..MAX_REJECT {
                let s = &samplers[rng.random_range(0..samplers.len())];
                let lc = s.cell_picker.sample(rng);
                let pf = &s.per_cell[lc];
                if pf.features.len() < 2 {
                    continue;
                }
                let g = pf.features[pf.picker.sample(rng)];
                let mut h = g;
                for _ in 0..MAX_REJECT {
                    h = pf.features[pf.picker.sample(rng)];
                    if h != g {
                        break;
                    }
                }
                if h == g {
                    // A gene holding nearly all the mass: pick uniformly among the rest.
                    let others: Vec<u32> =
                        pf.features.iter().copied().filter(|&f| f != g).collect();
                    h = others[rng.random_range(0..others.len())];
                }
                pos_cells.push(s.active_cells[lc]);
                pos_g.push(g);
                pos_h.push(h);
                continue 'positives;
            }
        }

        // Negatives: per positive, `n_negatives` draws of (hop → partner group →
        // cell → gene).
        let n_neg = pos_cells.len() * n_negatives;
        let mut neg_cells = Vec::with_capacity(n_neg);
        let mut neg_h = Vec::with_capacity(n_neg);
        let mut hops = Vec::with_capacity(n_neg);
        for &c in &pos_cells {
            for _ in 0..n_negatives {
                let t = self.hop_picker.sample(rng) + 1;
                let (fine, hop) = self
                    .tree
                    .draw_partner(c as usize, t, rng)
                    .expect("at least two finest groups, checked at construction");
                let cells = self.tree.cells_in(fine);
                let cn = cells[rng.random_range(0..cells.len())];
                let h = self
                    .gene_of(samplers, cn, rng)
                    .expect("cells_in holds active cells only");
                neg_cells.push(cn);
                neg_h.push(h);
                hops.push(hop as u8);
            }
        }

        GenePairBatch {
            pos_cells,
            pos_g,
            pos_h,
            neg_cells,
            neg_h,
            hops,
            n_negatives,
        }
    }
}

//////////
// Loss //
//////////

/// NCE over gene pairs: the positive `ρ_g·ρ_h + b_g + b_h` against its
/// negatives `ρ_g·ρ_h′ + b_g + b_h′`, mean over the batch. The first gene plays
/// the "cell side" of the bilinear score, so the same kernels as the cell-gene
/// loss apply.
pub fn gene_pair_nce(
    model: &JointEmbedModel,
    batch: GenePairBatch,
    objective: NceObjective,
    dev: &Device,
) -> Result<Tensor> {
    let b = batch.pos_g.len();
    if b == 0 {
        return Tensor::zeros((), candle_util::candle_core::DType::F32, dev);
    }
    let k = batch.n_negatives;

    let g_idx = Tensor::from_slice(&batch.pos_g, b, dev)?;
    let h_idx = Tensor::from_slice(&batch.pos_h, b, dev)?;
    let n_idx = Tensor::from_slice(&batch.neg_h, b * k, dev)?;

    let e_g = gather_feature_rows(model, &g_idx)?; // [B, H]
    let b_g = model.b_feat.index_select(&g_idx, 0)?; // [B]
    let e_h = gather_feature_rows(model, &h_idx)?; // [B, H]
    let b_h = model.b_feat.index_select(&h_idx, 0)?; // [B]
    let e_n_flat = gather_feature_rows(model, &n_idx)?; // [B*K, H]
    let hdim = e_n_flat.dim(1)?;
    let e_n = e_n_flat.reshape((b, k, hdim))?;
    let b_n = model.b_feat.index_select(&n_idx, 0)?.reshape((b, k))?;

    let pos = JointEmbedModel::score_diag(&e_h, &e_g, &b_h, &b_g)?; // [B]
    let neg = JointEmbedModel::score_negatives(&e_n, &e_g, &b_n, &b_g)?; // [B, K]

    let negs = std::slice::from_ref(&neg);
    let per_pair = match objective {
        NceObjective::Logistic => logistic_nce(&pos, negs)?,
        NceObjective::Softmax => softmax_nce(&pos, negs)?,
    };
    per_pair.mean(0)
}

#[cfg(test)]
#[path = "gene_pair_tests.rs"]
mod gene_pair_tests;
