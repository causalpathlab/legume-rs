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
/// Fully packed `[N, K]` — no union, no host `[N, S]`. This is the encoder's
/// bounded view of each row; what the decoder scores is chosen by the trainer
/// over the whole feature axis and is deliberately not a loader concern.
///
/// Padding when a sample has fewer than `K` features: indices are filled with
/// `0`, values with `0.0`. Gathers and weighted sums against zero values are
/// no-ops and pass silently; consumers that must not confuse a pad with
/// feature `0` derive a validity mask from `values > 0`.
pub struct IndexedMinibatchData {
    /// [N] u32 — the source row of each minibatch row. Bootstrap duplicates
    /// are kept, so this is a multiset, not a permutation. Lets a consumer
    /// index per-row state (a target table, a free latent) by source row.
    pub row_ids: Tensor,
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
            row_ids: self.row_ids.to_device(dev)?,
            input_indices: self.input_indices.to_device(dev)?,
            input_values: self.input_values.to_device(dev)?,
            input_values_null: opt(&self.input_values_null)?,
            input_values_mean: opt(&self.input_values_mean)?,
        })
    }
}

pub struct IndexedInMemoryArgs<'a, D>
where
    D: CandleDataLoaderOps,
{
    pub input: &'a D,
    pub input_null: Option<&'a D>,
    pub input_context_size: usize,
    /// Per-feature weights used to *score* candidates during top-K selection.
    /// Stored values remain raw row values. Pass `&[1.0; n_features]` to fall
    /// back to raw value-only selection.
    pub input_shortlist_weights: &'a [f32],
    /// Optional per-feature mean expression rate `μ_d` (length = D). When
    /// supplied, the loader gathers it for each per-cell top-K position and
    /// packs it as `input_values_mean [N, K]`, which the encoder composes with
    /// the batch null as a multiplicative count-rate divisor before Anscombe.
    pub input_mean: Option<&'a [f32]>,
}
