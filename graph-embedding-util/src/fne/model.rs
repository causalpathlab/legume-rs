//! One flat node table and PBG's fused softmax step over a padded batch.
//!
//! For a batch of `k` chunks of `c` positives with `u` uniform negatives
//! per chunk (`L` lhs rows, `R` rhs rows, `[k, c, D]`):
//!
//! ```text
//!   pos       = Σ_d L·R                                  [k, c]
//!   rhs_bat   = L·Rᵀ + mask,  lhs_bat = R·Lᵀ + mask      [k, c, c]
//!   rhs_uni   = L·URᵀ,        lhs_uni = R·ULᵀ            [k, c, u]
//!   ℓ         = softmax_nce(pos, [rhs_bat, rhs_uni]) + softmax_nce(pos, [lhs_bat, lhs_uni])
//!   loss      = Σ ℓ · row_w          (row_w = relation × edge weight, 0 on pad rows)
//! ```
//!
//! `mask` is PBG's `−1e9` on the diagonal (a positive never competes with
//! itself) and on every pad column (a padded last chunk contributes no
//! negatives). Both sides gather from the same table; a batch is one
//! relation, so each side is one node type.

use super::batch::PaddedBatch;
use super::graph::NodeTypeTable;
use super::{INIT_STDEV, MASK_NEG};
use crate::loss::softmax_nce;
use legume_numeric::candle::candle_core::{DType, Device, Result, Tensor, Var};
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::lora::PinnedLora;
use legume_numeric::candle::masking::additive_pad_mask;
use legume_numeric::matrix::rand_util::name_seed;
use legume_numeric::matrix::traits::SampleOps;

/// The score blocks of one batch, before the loss.
pub(crate) struct ScoreBlocks {
    pub pos: Tensor,
    pub rhs_bat: Tensor,
    pub lhs_bat: Tensor,
    pub rhs_uni: Option<Tensor>,
    pub lhs_uni: Option<Tensor>,
    /// `[k·c]` per-row loss weights, already on the device.
    pub row_w: Tensor,
}

pub(crate) struct FneModel {
    /// `[N_total, D]`, every node type stacked in table order.
    pub e: Var,
    /// `−1e9` on the diagonal, `[1, c, c]`, built once.
    diag_neg: Tensor,
    /// The low-rank residual on the anchored rows, under `PresetMode::Lora`.
    pub lora: Option<PinnedLora>,
}

impl FneModel {
    /// A fresh table, `N(0, INIT_STDEV)` per coordinate, seeded per node
    /// type by name and stacked in table order — so a two-type table named
    /// like simba's reproduces simba's tables exactly.
    pub(crate) fn new(
        types: &NodeTypeTable,
        dim: usize,
        c: usize,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let blocks: Vec<Tensor> = (0..types.len())
            .map(|t| {
                Tensor::rnorm_seeded(types.n_nodes(t), dim, name_seed(seed, types.name(t)))
                    .affine(INIT_STDEV, 0.0)
            })
            .collect::<Result<_>>()?;
        let e = Tensor::cat(&blocks, 0)?.to_device(dev)?.contiguous()?;
        Self::assemble(Var::from_tensor(&e)?, c, dev)
    }

    /// Per-type tables supplied by the caller (tests), stacked in order.
    #[cfg(test)]
    pub(crate) fn from_type_tables(tables: &[Tensor], c: usize) -> Result<Self> {
        let dev = tables[0].device().clone();
        let e = Tensor::cat(tables, 0)?.contiguous()?;
        Self::assemble(Var::from_tensor(&e)?, c, &dev)
    }

    /// Overwrite the listed rows with `preset.rows` and, when the mode pins
    /// them, return the `[N, 1]` gradient mask that is `0` on those rows.
    /// Under `Lora` the residual factors are set up as well: `u` drawn on the
    /// anchored rows from `seed`, `V` zero.
    pub(crate) fn apply_preset(
        &mut self,
        preset: &super::PresetRows,
        seed: u64,
        dev: &Device,
    ) -> anyhow::Result<Option<Tensor>> {
        let (n, d) = self.e.dims2()?;
        preset.mode.validate(d)?;
        anyhow::ensure!(
            preset.rows.len() == preset.ids.len() * d,
            "fne: preset rows are {} values for {} nodes at D={d}",
            preset.rows.len(),
            preset.ids.len()
        );
        let mut flat = self.e.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let mut keep = vec![1f32; n];
        for (i, &g) in preset.ids.iter().enumerate() {
            let g = g as usize;
            anyhow::ensure!(g < n, "fne: preset node {g} is outside the {n}-node table");
            flat[g * d..(g + 1) * d].copy_from_slice(&preset.rows[i * d..(i + 1) * d]);
            keep[g] = 0.0;
        }
        self.e.set(&Tensor::from_vec(flat, (n, d), dev)?)?;
        if let Some(l) = preset.mode.lora() {
            self.lora = Some(PinnedLora::new(n, d, l.rank, &preset.ids, seed, dev)?);
        }
        Ok(preset
            .mode
            .pins()
            .then(|| Tensor::from_vec(keep, (n, 1), dev))
            .transpose()?)
    }

    /// The rows named by `ids`, `[ids, D]`: the table's, plus the low-rank
    /// residual on anchored rows. Every lookup in the model goes through here.
    fn rows(&self, ids: &Tensor) -> Result<Tensor> {
        let rows = gather_rows(self.e.as_tensor(), ids)?;
        match self.lora.as_ref() {
            None => Ok(rows),
            Some(l) => rows + l.residual_rows(ids)?,
        }
    }

    /// The whole `[N, D]` table with the residual folded in — for the output
    /// only; training never forms it.
    pub(crate) fn composed(&self) -> Result<Tensor> {
        let e = self.e.as_tensor().detach();
        match self.lora.as_ref() {
            None => Ok(e),
            Some(l) => e + l.residual()?.detach(),
        }
    }

    fn assemble(e: Var, c: usize, dev: &Device) -> Result<Self> {
        let diag_neg = Tensor::eye(c.max(1), DType::F32, dev)?
            .affine(MASK_NEG, 0.0)?
            .unsqueeze(0)?;
        Ok(Self {
            e,
            diag_neg,
            lora: None,
        })
    }

    pub(crate) fn score_blocks(&self, b: &PaddedBatch, dev: &Device) -> Result<ScoreBlocks> {
        let (k, c, u) = (b.k, b.c, b.u);
        let d = self.e.dim(1)?;
        let p = k * c;
        // Two host→device copies per batch: every id array in one, every
        // float array in the other, sliced on the device.
        let ids: Vec<u32> = [&b.lhs[..], &b.rhs[..], &b.uni_lhs[..], &b.uni_rhs[..]].concat();
        let ids = Tensor::from_slice(&ids, ids.len(), dev)?;
        let floats: Vec<f32> = [&b.col_valid[..], &b.row_w[..]].concat();
        let floats = Tensor::from_slice(&floats, floats.len(), dev)?;
        let l = self.rows(&ids.narrow(0, 0, p)?)?.reshape((k, c, d))?;
        let r = self.rows(&ids.narrow(0, p, p)?)?.reshape((k, c, d))?;
        // Polarity flips raw Dots before the pad/diag mask: MASK_NEG must
        // stay large-negative under enemy (−Dot) scoring.
        let flip = |t: Tensor| -> Result<Tensor> {
            if b.polarity.flips_dot() {
                t.neg()
            } else {
                Ok(t)
            }
        };
        let pos = flip((&l * &r)?.sum(2)?)?; // [k, c]
        let pad = additive_pad_mask(&floats.narrow(0, 0, p)?.reshape((k, 1, c))?)?;
        let row_w = floats.narrow(0, p, p)?;
        let mask = (self.diag_neg.broadcast_as((k, c, c))? + pad.broadcast_as((k, c, c))?)?;
        let rhs_bat = (flip(l.matmul(&r.t()?)?)? + &mask)?;
        let lhs_bat = (flip(r.matmul(&l.t()?)?)? + &mask)?;
        let (rhs_uni, lhs_uni) = if u > 0 {
            let ul = self
                .rows(&ids.narrow(0, 2 * p, k * u)?)?
                .reshape((k, u, d))?;
            let ur = self
                .rows(&ids.narrow(0, 2 * p + k * u, k * u)?)?
                .reshape((k, u, d))?;
            (
                Some(flip(l.matmul(&ur.t()?)?)?),
                Some(flip(r.matmul(&ul.t()?)?)?),
            )
        } else {
            (None, None)
        };
        Ok(ScoreBlocks {
            pos,
            rhs_bat,
            lhs_bat,
            rhs_uni,
            lhs_uni,
            row_w,
        })
    }

    /// PBG's batch loss: weighted sum over positives of the lhs- and
    /// rhs-corrupted softmax losses. Weight decay is added by the caller.
    /// Enemy relations enter as −Dot (see [`Self::score_blocks`]).
    pub(crate) fn batch_loss(&self, b: &PaddedBatch, dev: &Device) -> Result<Tensor> {
        let s = self.score_blocks(b, dev)?;
        let p = b.k * b.c;
        let pos = s.pos.reshape(p)?;
        let mut rhs_negs = vec![s.rhs_bat.reshape((p, b.c))?];
        let mut lhs_negs = vec![s.lhs_bat.reshape((p, b.c))?];
        if let Some(t) = s.rhs_uni {
            rhs_negs.push(t.reshape((p, b.u))?);
        }
        if let Some(t) = s.lhs_uni {
            lhs_negs.push(t.reshape((p, b.u))?);
        }
        let per_row = (softmax_nce(&pos, &rhs_negs)? + softmax_nce(&pos, &lhs_negs)?)?;
        (per_row * s.row_w)?.sum_all()
    }

    /// `Σ‖E‖²` — PBG's `l2_norm()` over the node table, through one
    /// `[D, D]` gemm so no `[N, D]` square is retained for backward.
    pub(crate) fn frob_sq(&self) -> Result<Tensor> {
        let x = self.e.as_tensor();
        let d = x.dim(1)?;
        let gram = x.t()?.matmul(x)?;
        let eye = Tensor::eye(d, gram.dtype(), gram.device())?;
        (gram * eye)?.sum_all()
    }
}
