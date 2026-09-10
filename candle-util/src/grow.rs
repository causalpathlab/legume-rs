//! Growing a trained checkpoint into a larger model.
//!
//! Continuing training over a new cohort can need capacity the parent does not
//! have — a topic for biology it never saw, or a wider gene embedding. Loading
//! a checkpoint into a bigger `VarMap` is a shape mismatch, so the saved
//! tensors are copied into the leading corner and the new slab is filled by a
//! rule that keeps the model's output unchanged at step 0. A cohort can also
//! measure a different set of genes: every gene-keyed tensor is then gathered
//! onto the new order by name ([`AxisRemap`]) before the corner copy.
//!
//! This lives here rather than in the caller because the rule keys on the
//! variable **names** — `topic.embeddings`, `feature.embeddings`, `attn.query`
//! — and those are registered by this crate's own decoders and encoders. A
//! policy that reads them belongs next to the code that writes them.

use crate::candle_core::{Device, Tensor};
use crate::candle_nn::VarMap;

/// Bias given to a newly added topic's encoder output.
///
/// The new rows of `z.mean` / `z.lnvar` get zero weights, so their pre-activation
/// is exactly this constant. Under the softmax that shares the latent with the
/// parent's topics — whose logits are O(1) — `e^-10 ≈ 4.5e-5` of the mass, i.e.
/// the added topics start switched off and have to earn their way in.
pub const NEW_TOPIC_LOGIT_BIAS: f64 = -10.0;

/// Extra capacity the child adds on top of the parent's.
///
/// Both zero is an ordinary exact-match warm start, which takes the original
/// `VarMap::load` path unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct Growth {
    /// Topics appended to `K`.
    pub add_topics: usize,
    /// Dimensions appended to the per-gene embedding `H` (masked family only —
    /// the dense decoder has no `ρ`).
    pub add_embedding_dim: usize,
}

impl Growth {
    #[must_use]
    pub fn is_none(self) -> bool {
        self.add_topics == 0 && self.add_embedding_dim == 0
    }
}

/// This run's gene axis expressed on the checkpoint's, when the two differ.
///
/// Growth on `K` and `H` appends capacity, so a saved tensor sits in the
/// leading corner of the new one. A changed gene axis is a *reordering* with
/// gaps: gene `g` of this run is gene `new_to_old[g]` of the checkpoint, or
/// `None` for a gene the checkpoint never saw. Every tensor with an axis of
/// the checkpoint's gene count is gathered along it by that map, and the
/// unseen entries start at the mean of the seen ones, so a new gene enters at
/// the model's average and nothing it computes for the known genes moves.
pub struct AxisRemap<'a> {
    /// For each gene of this run, its position on the checkpoint's gene axis.
    pub new_to_old: &'a [Option<usize>],
    /// The checkpoint's gene count.
    pub n_old: usize,
}

/// The old and new sizes of the axes that may change.
pub struct GrowthDims<'a> {
    pub k_old: usize,
    pub k_new: usize,
    pub h_old: usize,
    pub h_new: usize,
    /// `Some` when this run's gene axis is not the checkpoint's; see [`AxisRemap`].
    pub gene_axis: Option<AxisRemap<'a>>,
}

impl GrowthDims<'_> {
    /// Name the axis a `old → new` change on some tensor corresponds to, or
    /// `None` when it matches neither declared growth.
    ///
    /// This is what keeps growth from papering over a real architecture change:
    /// a widened hidden layer or an undeclared change of the gene axis grows a
    /// tensor too, and must still be an error rather than a silent zero-pad.
    pub fn classify(&self, old: usize, new: usize) -> Option<&'static str> {
        if self.is_gene_axis(old, new) {
            Some("D")
        } else if old == self.k_old && new == self.k_new {
            Some("K")
        } else if old == self.h_old && new == self.h_new {
            Some("H")
        } else {
            None
        }
    }

    /// Whether a saved axis of `old` entries meeting a fresh one of `new` is the
    /// declared gene axis (the two lengths may be equal: a permutation).
    fn is_gene_axis(&self, old: usize, new: usize) -> bool {
        self.gene_axis
            .as_ref()
            .is_some_and(|g| old == g.n_old && new == g.new_to_old.len())
    }

    /// Whether a saved tensor of this shape has to be gathered onto the new gene
    /// order even when its shape already matches — a same-length permuted axis
    /// is not an exact match.
    #[must_use]
    pub fn reorders(&self, saved_dims: &[usize], fresh_dims: &[usize]) -> bool {
        saved_dims
            .iter()
            .zip(fresh_dims)
            .any(|(&o, &n)| self.is_gene_axis(o, n))
    }
}

/// Value to write into a tensor's newly created slab, or `None` to keep the
/// freshly-initialized values already there.
///
/// The overrides are exactly the ones function preservation requires:
///
/// - **`α`'s new H columns → 0.** `β = softmax_g(α·ρᵀ)`, so zeroing them leaves
///   `β` bit-for-bit unchanged whatever `ρ`'s new columns hold.
/// - **`ρ`'s new H columns → kept random.** This is the one that must *not* be
///   zero: `∂β/∂α_new ∝ ρ_new`, so a zeroed `ρ_new` would leave both sides at
///   zero with no gradient into either — a dead subspace that never learns.
/// - **encoder input weights and the attention query → 0** on their new H
///   columns, so the encoder's output is unchanged despite `ρ` being wider.
/// - **new topic rows of `z.mean` / `z.lnvar` → 0 weight**, with the bias at
///   [`NEW_TOPIC_LOGIT_BIAS`], so added topics start at ~0 mass.
///
/// Everything else keeps its fresh init, which is what new capacity should
/// start from.
pub fn new_slab_value(name: &str, dim: usize) -> Option<f64> {
    // α [K, H] — column growth must not disturb β.
    if name.ends_with("topic.embeddings") {
        return (dim == 1).then_some(0.0);
    }
    // The encoder's view of a wider ρ must start out ignoring the new part.
    if name.ends_with("attn.query") || name.ends_with("fc.relu_linear_stack.0.weight") {
        return (dim == 1).then_some(0.0);
    }
    if name.contains("z.mean") || name.contains("z.lnvar") {
        return Some(if name.ends_with(".bias") {
            NEW_TOPIC_LOGIT_BIAS
        } else {
            0.0
        });
    }
    // ρ (`feature.embeddings`) deliberately falls through: its new columns stay
    // random so the added subspace can receive gradient.
    None
}

/// Copy the checkpoint into a larger `VarMap`, padding the axes that grew.
pub fn load_grown(parameters: &VarMap, path: &str, dims: &GrowthDims<'_>) -> anyhow::Result<()> {
    let saved = candle_core::safetensors::load(path, &Device::Cpu)?;
    let data = parameters.data().lock().expect("VarMap lock");

    let (mut n_copied, mut n_grown) = (0usize, 0usize);
    for (name, var) in data.iter() {
        let s = saved.get(name).ok_or_else(|| {
            anyhow::anyhow!("warm-start: {path} has no tensor named `{name}`; architectures differ")
        })?;
        let s = s.to_device(var.device())?;
        let fresh = var.as_tensor();

        if s.dims() == fresh.dims() && !dims.reorders(s.dims(), fresh.dims()) {
            var.set(&s)?;
            n_copied += 1;
            continue;
        }
        var.set(&grow_tensor(name, fresh, &s, dims)?)?;
        n_grown += 1;
    }
    log::info!("Warm-start: {n_copied} variables copied, {n_grown} grown");
    Ok(())
}

/// One tensor, widened: the checkpoint occupies the leading corner and the new
/// slab is filled per [`new_slab_value`].
pub fn grow_tensor(
    name: &str,
    fresh: &Tensor,
    saved: &Tensor,
    dims: &GrowthDims<'_>,
) -> anyhow::Result<Tensor> {
    anyhow::ensure!(
        saved.rank() == fresh.rank(),
        "warm-start: `{name}` has rank {} in the checkpoint and {} here",
        saved.rank(),
        fresh.rank(),
    );
    // The gene axis first: it is a reordering, not an appended slab, so the
    // saved tensor is gathered onto this run's order and then treated as an
    // exact match on that axis by the corner copy below.
    let mut saved = saved.clone();
    if let Some(g) = dims.gene_axis.as_ref() {
        for dim in 0..saved.rank() {
            if dims.is_gene_axis(saved.dims()[dim], fresh.dims()[dim]) {
                saved = gather_gene_axis(name, &saved, dim, g)?;
            }
        }
    }
    let saved = &saved;

    // Validate every axis BEFORE touching the data. `slice_assign` fails first
    // otherwise, and reports "upper bound is out of range for dim 1, 18 12" —
    // true, but it names neither the tensor nor what the caller did wrong.
    for (dim, (&old, &new)) in saved.dims().iter().zip(fresh.dims()).enumerate() {
        anyhow::ensure!(
            new >= old,
            "warm-start: `{name}` shrank on axis {dim} ({old} → {new}); growth only adds capacity",
        );
        anyhow::ensure!(
            old == new || dims.classify(old, new).is_some(),
            "warm-start: `{name}` axis {dim} changed {old} → {new}, which is neither the \
             requested K growth ({} → {}) nor H growth ({} → {}). That is an architecture \
             change, not added capacity.",
            dims.k_old,
            dims.k_new,
            dims.h_old,
            dims.h_new,
        );
    }

    // Start from the freshly-initialized tensor: it already has the right shape
    // and the model's own init in the region the checkpoint does not cover.
    let corner: Vec<std::ops::Range<usize>> = saved.dims().iter().map(|&d| 0..d).collect();
    let mut out = fresh.slice_assign(&corner, saved)?;

    for (dim, (&old, &new)) in saved.dims().iter().zip(fresh.dims()).enumerate() {
        if old == new {
            continue;
        }
        let axis = dims.classify(old, new).unwrap_or("?");

        if let Some(v) = new_slab_value(name, dim) {
            let mut slab_shape: Vec<usize> = fresh.dims().to_vec();
            slab_shape[dim] = new - old;
            let slab =
                Tensor::full(v, slab_shape.as_slice(), fresh.device())?.to_dtype(fresh.dtype())?;
            let mut r: Vec<std::ops::Range<usize>> = fresh.dims().iter().map(|&d| 0..d).collect();
            r[dim] = old..new;
            out = out.slice_assign(&r, &slab)?;
            log::debug!("warm-start: `{name}` axis {dim} ({axis}) {old} → {new}, new slab = {v}");
        } else {
            log::debug!(
                "warm-start: `{name}` axis {dim} ({axis}) {old} → {new}, new slab keeps its init"
            );
        }
    }
    Ok(out)
}

/// One tensor's gene axis, gathered onto this run's order.
///
/// Known genes take their saved entries. An unseen gene takes the slab value
/// the tensor's growth rule names ([`new_slab_value`]), or — where the rule
/// keeps the fresh init, as it does for ρ — the mean of the saved entries
/// along that axis, so it enters at the checkpoint's average rather than at a
/// random point. A caller that knows more (a module the gene was placed in)
/// refines from there.
fn gather_gene_axis(
    name: &str,
    saved: &Tensor,
    dim: usize,
    remap: &AxisRemap<'_>,
) -> anyhow::Result<Tensor> {
    let n_new = remap.new_to_old.len();
    let dev = saved.device();
    let idx: Vec<u32> = remap
        .new_to_old
        .iter()
        .map(|p| p.unwrap_or(0) as u32)
        .collect();
    let gathered = saved.index_select(&Tensor::from_vec(idx, n_new, dev)?, dim)?;

    let mut along: Vec<usize> = vec![1; saved.rank()];
    along[dim] = n_new;
    let known: Vec<f32> = remap
        .new_to_old
        .iter()
        .map(|p| if p.is_some() { 1.0 } else { 0.0 })
        .collect();
    let known = Tensor::from_vec(known, along.as_slice(), dev)?.to_dtype(saved.dtype())?;
    let unknown = known.affine(-1.0, 1.0)?;

    let fill = match new_slab_value(name, dim) {
        Some(v) => {
            let mut one: Vec<usize> = saved.dims().to_vec();
            one[dim] = 1;
            Tensor::full(v, one.as_slice(), dev)?.to_dtype(saved.dtype())?
        }
        None => saved.mean_keepdim(dim)?,
    };
    let n_known = remap.new_to_old.iter().filter(|p| p.is_some()).count();
    log::debug!(
        "warm-start: `{name}` axis {dim} (D) gathered {} → {n_new} ({n_known} known)",
        remap.n_old,
    );
    Ok(gathered
        .broadcast_mul(&known)?
        .broadcast_add(&fill.broadcast_mul(&unknown)?)?)
}
