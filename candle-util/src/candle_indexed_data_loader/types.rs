//! Public data types for the indexed data loader.

use candle_core::{Device, Tensor};
use matrix_util::traits::CandleDataLoaderOps;

/// Per-sample: top-K features selected from dense data.
#[derive(Clone)]
pub struct IndexedSample {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Packed top-K minibatch.
///
/// Encoder side is fully packed `[N, K]` — no union, no host `[N, S]`.
/// Decoder side keeps the union for the importance-weighted conditional
/// softmax denominator and a per-cell `scatter_pos` so the per-cell
/// likelihood can be gathered without ever returning a dense `[N, S]`.
///
/// Padding when a sample has fewer than `K` features: indices and
/// scatter positions are filled with `0`, values with `0.0`. Gathers
/// and weighted sums against zero values are no-ops and pass silently.
pub struct IndexedMinibatchData {
    /// [N, K_in] u32 in [0, D_in) — encoder feature ids
    pub input_indices: Tensor,
    /// [N, K_in] f32 — encoder feature values
    pub input_values: Tensor,
    /// [N, K_in] f32 — encoder null (μ_residual) gathered at `input_indices`
    pub input_values_null: Option<Tensor>,
    /// [N, K_in] f32 — per-gene mean expression rate `μ_d` gathered at
    /// `input_indices` (when an `input_mean` was supplied). The encoder
    /// composes it with `input_values_null` as a multiplicative count-rate
    /// divisor before Anscombe — joint correction for batch effect ×
    /// gene-typical-rate, leaving the cell's biological deviation.
    pub input_values_mean: Option<Tensor>,
    /// [S] u32 in [0, D_out) — sorted union of decoder per-cell top-K ids
    pub output_union_indices: Tensor,
    /// [N, K_out] u32 in [0, S) — per-cell positions of decoder values in the union
    pub output_scatter_pos: Tensor,
    /// [N, K_out] f32 — decoder feature values (unpacked, in per-cell index order)
    pub output_values: Tensor,
    /// [N, K_out] f32 — per-gene NB-Fisher info weight gathered at the
    /// decoder's per-cell ids (when an `output_fisher_weights` was supplied).
    /// The decoder multiplies this into the `(value+1).log()` likelihood
    /// weight so housekeeping observations contribute less to β's gradient.
    pub output_values_weight: Option<Tensor>,
    /// [1, S] f32 — log selection frequency at union positions
    pub output_log_q_s: Tensor,
}

impl IndexedMinibatchData {
    /// Upload every tensor field to `dev`. Cached minibatches are built
    /// host-side by `precompute_all_minibatches`; the training loop calls
    /// this once per minibatch so a GPU run uploads incrementally instead
    /// of holding the whole epoch resident on device. A no-op copy when
    /// `dev` is already CPU.
    pub fn to_device(&self, dev: &Device) -> anyhow::Result<IndexedMinibatchData> {
        let opt = |t: &Option<Tensor>| -> anyhow::Result<Option<Tensor>> {
            t.as_ref()
                .map(|x| x.to_device(dev))
                .transpose()
                .map_err(Into::into)
        };
        Ok(IndexedMinibatchData {
            input_indices: self.input_indices.to_device(dev)?,
            input_values: self.input_values.to_device(dev)?,
            input_values_null: opt(&self.input_values_null)?,
            input_values_mean: opt(&self.input_values_mean)?,
            output_union_indices: self.output_union_indices.to_device(dev)?,
            output_scatter_pos: self.output_scatter_pos.to_device(dev)?,
            output_values: self.output_values.to_device(dev)?,
            output_values_weight: opt(&self.output_values_weight)?,
            output_log_q_s: self.output_log_q_s.to_device(dev)?,
        })
    }
}

pub struct IndexedInMemoryArgs<'a, D>
where
    D: CandleDataLoaderOps,
{
    pub input: &'a D,
    pub input_null: Option<&'a D>,
    pub output: &'a D,
    pub input_context_size: usize,
    pub output_context_size: usize,
    /// Per-feature weights used to *score* candidates during top-K selection
    /// on the encoder (input) side. Stored values remain raw row values.
    /// Pass `&[1.0; n_features]` to fall back to raw value-only selection.
    pub input_shortlist_weights: &'a [f32],
    /// Same role on the decoder (output) side.
    pub output_shortlist_weights: &'a [f32],
    /// Optional per-feature mean expression rate `μ_d` (encoder side,
    /// length = D_in). When supplied, the loader gathers it for each
    /// per-cell top-K position and packs it as `input_values_mean [N, K]`,
    /// which the encoder composes with the batch null as a multiplicative
    /// count-rate divisor before Anscombe.
    pub input_mean: Option<&'a [f32]>,
    /// Optional per-feature NB-Fisher weight (decoder side, length = D_out).
    /// When supplied, the loader gathers `output_fisher_weights[idx]` and
    /// packs it as `output_values_weight [N, K]` for the decoder to apply
    /// as a multiplicative loss-term weight.
    pub output_fisher_weights: Option<&'a [f32]>,
}
