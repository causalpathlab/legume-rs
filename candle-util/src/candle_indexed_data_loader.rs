use crate::candle_data_loader_util::Minibatches;

use candle_core::{Device, Tensor};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use matrix_util::traits::CandleDataLoaderOps;
use rayon::prelude::*;
use std::cell::RefCell;

fn labeled_bar(label: &str, len: u64) -> ProgressBar {
    ProgressBar::new(len).with_style(
        ProgressStyle::with_template(&format!("{} {{bar:40}} {{pos}}/{{len}} ({{eta}})", label))
            .unwrap()
            .progress_chars("##-"),
    )
}

/// Per-sample: top-K features selected from dense data
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

/// Adaptive feature window data loader with decoupled encoder/decoder windows.
///
/// Each sample keeps its top-K features by value independently for input (encoder)
/// and output (decoder) sides. Encoder consumes packed `[N, K_in]` (indices,
/// values); decoder consumes the per-batch union plus per-cell scatter
/// positions into that union.
///
/// Call `precompute_all_minibatches` after `shuffle_minibatch` to cache all
/// minibatch tensors. Use `minibatch_cached` to retrieve.
pub struct IndexedInMemoryData {
    input_samples: Vec<IndexedSample>,
    input_null_rows: Option<Vec<Vec<f32>>>,
    output_samples: Vec<IndexedSample>,
    n_input_features: usize,
    n_output_features: usize,
    input_context_size: usize,
    output_context_size: usize,
    /// Per-feature log selection frequency: log(q_d) where q_d = P(feature d in top-K).
    /// Used for importance-weighted conditional softmax (Jean et al., 2015).
    output_log_q: Vec<f32>,
    /// Per-feature mean expression rate `μ_d` (encoder side) gathered
    /// into `input_values_mean [N, K]` at minibatch build time.
    input_mean: Option<Vec<f32>>,
    /// Per-feature NB-Fisher weight (decoder side) gathered into
    /// `output_values_weight [N, K]` at minibatch build time.
    output_fisher_weights: Option<Vec<f32>>,
    minibatches: Minibatches,
    cached_batches: Vec<IndexedMinibatchData>,
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

/// Select top-K indices by value from a dense row.
/// Returns (sorted_indices, values_at_those_indices).
pub fn top_k_indices(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let k = k.min(row.len());

    // Collect (value, index) pairs and partial sort to find top-K
    let mut indexed: Vec<(f32, u32)> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    // Partial sort: move top-K elements to the front
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top-K, then sort by index for deterministic order
    let mut top_k: Vec<(u32, f32)> = indexed[..k].iter().map(|&(v, i)| (i, v)).collect();
    top_k.sort_unstable_by_key(|&(idx, _)| idx);

    let indices: Vec<u32> = top_k.iter().map(|&(i, _)| i).collect();
    let values: Vec<f32> = top_k.iter().map(|&(_, v)| v).collect();
    (indices, values)
}

/// Like [`top_k_indices`] but ranks candidates by `log(row[i] + 1) · weights[i]`
/// while returning the **raw** `row[i]` for the selected indices.
///
/// The `log1p` compression on the scoring side is essential when
/// `weights` are NB-Fisher info: without it, a housekeeping gene with raw
/// count `v=200` and `w=0.04` scores `8`, beating a marker with `v=10`,
/// `w=0.24` (score `2.4`), so housekeeping dominates top-K selection
/// despite Fisher attenuation. With `log1p`, the same pair scores
/// `0.21` (housekeeping) vs `0.58` (marker) — markers win, and
/// housekeeping is crowded out of top-K. The values returned remain
/// raw counts so the encoder/decoder math is unchanged.
///
/// Returns at most `k` indices: features whose score is non-positive
/// (`v == 0` or `w == 0`) are dropped before selection, so a cell with
/// fewer non-zero, positively-weighted features than `k` returns a short
/// vector. Every downstream consumer already uses `s.indices.len().min(k)`
/// when packing into `[N, K]` buffers and leaves the rest at the
/// zero-initialized `(idx=0, val=0)`, so the loss `Σ_k v_k · log_recon_at_k`
/// is unaffected by the missing slots, the encoder weighted-sum is
/// unaffected, and the per-batch union skips features no cell actually
/// observed.
pub fn top_k_indices_weighted(row: &[f32], weights: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    debug_assert_eq!(row.len(), weights.len());

    let mut scored: Vec<(f32, u32)> = row
        .iter()
        .zip(weights.iter())
        .enumerate()
        .filter_map(|(i, (&v, &w))| {
            // log1p on the score, not the value — keeps the per-cell K
            // raw values untouched while the *ranking* compresses
            // dynamic range so high-mean genes don't auto-win.
            let score = v.max(0.0).ln_1p() * w;
            (score > 0.0).then_some((score, i as u32))
        })
        .collect();

    let k = k.min(scored.len());
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    scored.select_nth_unstable_by(k - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut idx: Vec<u32> = scored[..k].iter().map(|&(_, i)| i).collect();
    idx.sort_unstable();

    let values: Vec<f32> = idx.iter().map(|&i| row[i as usize]).collect();
    (idx, values)
}

/// Per-feature lookup slot. `gen` carries the call generation that wrote
/// `pos`; a slot is "in the current union" iff its `gen` matches the call's
/// generation. Reusing entries via a generation tag instead of a sentinel
/// reset means a panic mid-call cannot leak stale state into the next call
/// on the same rayon worker — the next call simply bumps the generation and
/// every prior write is invalidated.
#[derive(Default, Clone, Copy)]
struct PosSlot {
    gen: u32,
    pos: u32,
}

#[derive(Default)]
struct PosLookup {
    entries: Vec<PosSlot>,
    current_gen: u32,
}

thread_local! {
    /// Per-thread `feature_id -> position` lookup. Grows monotonically to the
    /// largest `n_features` seen on this thread; never resets between calls.
    static POS_LOOKUP: RefCell<PosLookup> = const {
        RefCell::new(PosLookup { entries: Vec::new(), current_gen: 0 })
    };
}

/// Walk every cell's top-K positions and fill an `[N, K]` `f32` buffer.
///
/// `fill(row, kk, sample_index, feature_id) -> value` is invoked once per
/// observed slot; padded slots (when the sample has fewer than `k`
/// indices) keep the buffer's initial `0.0`. Pad indices are harmless
/// because every consumer either pairs them with a zero value side or
/// only multiplies them in (zero · anything = 0).
fn pack_at_indices<F>(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
    mut fill: F,
) -> anyhow::Result<Tensor>
where
    F: FnMut(usize, usize, usize, u32) -> f32,
{
    let n = sample_indices.len();
    let mut buf = vec![0.0f32; n * k];
    for (row, &si) in sample_indices.iter().enumerate() {
        let s = &samples[si];
        let off = row * k;
        let take = s.indices.len().min(k);
        for (kk, &feat) in s.indices[..take].iter().enumerate() {
            buf[off + kk] = fill(row, kk, si, feat);
        }
    }
    Ok(Tensor::from_vec(buf, (n, k), target_device)?)
}

/// Pack per-cell `(indices, values)` into `[N, K]` u32/f32 tensors. Both
/// share the same loop walk; the index buffer is built directly so we
/// can also dtype-cast it to u32 in one shot.
fn pack_indices_values(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n = sample_indices.len();
    let mut idx_buf = vec![0u32; n * k];
    let mut val_buf = vec![0.0f32; n * k];
    for (row, &si) in sample_indices.iter().enumerate() {
        let s = &samples[si];
        let off = row * k;
        let take = s.indices.len().min(k);
        idx_buf[off..off + take].copy_from_slice(&s.indices[..take]);
        val_buf[off..off + take].copy_from_slice(&s.values[..take]);
    }
    let indices =
        Tensor::from_vec(idx_buf, (n, k), target_device)?.to_dtype(candle_core::DType::U32)?;
    let values = Tensor::from_vec(val_buf, (n, k), target_device)?;
    Ok((indices, values))
}

/// Pack a per-cell row `(per_sample[si][feat])` at `samples[si].indices`
/// into `[N, K] f32`. Used for the encoder's μ_residual batch null.
fn pack_null_at_indices(
    samples: &[IndexedSample],
    null_rows: &[Vec<f32>],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<Tensor> {
    pack_at_indices(
        samples,
        sample_indices,
        k,
        target_device,
        |_, _, si, feat| null_rows[si][feat as usize],
    )
}

/// Gather a per-feature `[D]` slice at each cell's top-K positions into
/// `[N, K] f32`. Used for both encoder gene-mean (`μ_d`) and decoder
/// NB-Fisher weights — each cell sees the same constant per feature.
fn gather_per_feature_at_indices(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    per_feature: &[f32],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<Tensor> {
    pack_at_indices(
        samples,
        sample_indices,
        k,
        target_device,
        |_, _, _, feat| per_feature[feat as usize],
    )
}

/// Decoder-side union + per-cell scatter positions.
///
/// Returns:
/// - `union_indices [S] u32` — sorted-by-discovery union of feature ids
///   appearing in any selected cell's top-K.
/// - `scatter_pos   [N, K] u32 in [0, S)` — for each cell row and slot k,
///   the position of `samples[si].indices[k]` in `union_indices`. Padded
///   slots get position `0` (matched by zero values, harmless).
/// - `union_vec` — the same `[S]` ids as a host `Vec<u32>` for downstream
///   indexing into per-feature arrays (e.g. `output_log_q`).
fn build_union_and_scatter_pos(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    n_features: usize,
    k: usize,
    target_device: &Device,
) -> anyhow::Result<(Tensor, Tensor, Vec<u32>)> {
    let n_batch = sample_indices.len();

    POS_LOOKUP.with(|cell| {
        let mut lookup = cell.borrow_mut();
        let PosLookup {
            entries,
            current_gen,
        } = &mut *lookup;
        if entries.len() < n_features {
            entries.resize(n_features, PosSlot::default());
        }
        let mut gen = current_gen.wrapping_add(1);
        if gen == 0 {
            // Wrapped past u32::MAX — zero all gens and restart at 1 so
            // entries written before the wrap are correctly invalidated.
            for slot in entries.iter_mut() {
                slot.gen = 0;
            }
            gen = 1;
        }
        *current_gen = gen;

        let mut union_vec: Vec<u32> = Vec::new();
        let mut scatter = vec![0u32; n_batch * k];
        for (row, &si) in sample_indices.iter().enumerate() {
            let s = &samples[si];
            let off = row * k;
            let take = s.indices.len().min(k);
            for (kk, &feat) in s.indices[..take].iter().enumerate() {
                let fi = feat as usize;
                if entries[fi].gen != gen {
                    entries[fi] = PosSlot {
                        gen,
                        pos: union_vec.len() as u32,
                    };
                    union_vec.push(feat);
                }
                scatter[off + kk] = entries[fi].pos;
            }
            // Padded slots [take..k] keep scatter=0 (matches zero value).
        }
        let s = union_vec.len();

        let union_indices = Tensor::from_slice(&union_vec, (s,), target_device)?
            .to_dtype(candle_core::DType::U32)?;
        let scatter_pos = Tensor::from_vec(scatter, (n_batch, k), target_device)?
            .to_dtype(candle_core::DType::U32)?;

        Ok((union_indices, scatter_pos, union_vec))
    })
}

/// Build IndexedSamples from a data source in parallel.
fn build_indexed_samples<D: CandleDataLoaderOps + Sync>(
    data: &D,
    n_samples: usize,
    context_size: usize,
    shortlist_weights: &[f32],
    label: &str,
) -> Vec<IndexedSample> {
    let pb = labeled_bar(label, n_samples as u64);
    let out = (0..n_samples)
        .into_par_iter()
        .progress_with(pb.clone())
        .map(|i| {
            let row = data.row_to_f32_vec(i);
            let (indices, values) = top_k_indices_weighted(&row, shortlist_weights, context_size);
            IndexedSample { indices, values }
        })
        .collect();
    pb.finish_and_clear();
    out
}

/// Compute log selection frequency for each feature from indexed samples.
///
/// q_d = (# samples containing feature d) / (total samples), clamped to [1/n, 1].
/// Returns log(q_d) for each of n_features features.
///
/// Used for importance-weighted conditional softmax (Jean et al., 2015,
/// "On Using Very Large Target Vocabulary for Neural Machine Translation").
fn compute_log_selection_freq(samples: &[IndexedSample], n_features: usize) -> Vec<f32> {
    let n = samples.len().max(1) as f32;
    let mut counts = vec![0u32; n_features];
    for sample in samples {
        for &idx in &sample.indices {
            counts[idx as usize] += 1;
        }
    }
    counts
        .iter()
        .map(|&c| ((c.max(1) as f32) / n).ln())
        .collect()
}

impl IndexedInMemoryData {
    /// Build indexed data from dense matrices.
    ///
    /// Input and output get independent top-K selections from their respective sources.
    pub fn from_dense<D>(args: IndexedInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + Sync,
    {
        let (n_samples, n_input_features) = args.input.data_shape();
        let (_, n_output_features) = args.output.data_shape();
        let input_context_size = args.input_context_size.min(n_input_features);
        let output_context_size = args.output_context_size.min(n_output_features);

        let input_samples = build_indexed_samples(
            args.input,
            n_samples,
            input_context_size,
            args.input_shortlist_weights,
            "Top-K (encoder)",
        );
        let output_samples = build_indexed_samples(
            args.output,
            n_samples,
            output_context_size,
            args.output_shortlist_weights,
            "Top-K (decoder)",
        );

        let output_log_q = compute_log_selection_freq(&output_samples, n_output_features);

        // Pre-extract null rows in parallel
        let null_rows: Option<Vec<Vec<f32>>> = args.input_null.map(|d| {
            let (n, _) = d.data_shape();
            let pb = labeled_bar("Null rows", n as u64);
            let rows: Vec<Vec<f32>> = (0..n)
                .into_par_iter()
                .progress_with(pb.clone())
                .map(|i| d.row_to_f32_vec(i))
                .collect();
            pb.finish_and_clear();
            rows
        });

        let rows: Vec<usize> = (0..n_samples).collect();

        let input_mean = args.input_mean.map(|s| s.to_vec());
        let output_fisher_weights = args.output_fisher_weights.map(|s| s.to_vec());
        if let Some(ref b) = input_mean {
            anyhow::ensure!(
                b.len() == n_input_features,
                "input_mean length {} != n_input_features {}",
                b.len(),
                n_input_features
            );
        }
        if let Some(ref w) = output_fisher_weights {
            anyhow::ensure!(
                w.len() == n_output_features,
                "output_fisher_weights length {} != n_output_features {}",
                w.len(),
                n_output_features
            );
        }

        Ok(IndexedInMemoryData {
            input_samples,
            input_null_rows: null_rows,
            output_samples,
            n_input_features,
            n_output_features,
            input_context_size,
            output_context_size,
            output_log_q,
            input_mean,
            output_fisher_weights,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
            cached_batches: vec![],
        })
    }

    /// Build indexed data from pre-computed samples (e.g., from sparse I/O).
    ///
    /// Both input and output share the same samples (encoder = decoder target).
    pub fn from_samples(
        samples: Vec<IndexedSample>,
        n_features: usize,
        context_size: usize,
    ) -> Self {
        let n = samples.len();
        let rows: Vec<usize> = (0..n).collect();
        let output_samples = samples;
        let input_samples = output_samples.clone();
        let output_log_q = compute_log_selection_freq(&output_samples, n_features);
        IndexedInMemoryData {
            input_samples,
            input_null_rows: None,
            output_samples,
            n_input_features: n_features,
            n_output_features: n_features,
            input_context_size: context_size,
            output_context_size: context_size,
            output_log_q,
            input_mean: None,
            output_fisher_weights: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
            cached_batches: vec![],
        }
    }

    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
        self.cached_batches.clear();
    }

    /// Pre-build all minibatch tensors for the current shuffle order.
    ///
    /// Call this once after `shuffle_minibatch` to avoid rebuilding
    /// union+scatter on every `minibatch_cached` call within an epoch.
    pub fn precompute_all_minibatches(&mut self, target_device: &Device) -> anyhow::Result<()> {
        let n_chunks = self.minibatches.chunks.len() as u64;
        let pb = labeled_bar("Minibatch precompute", n_chunks);
        self.cached_batches = self
            .minibatches
            .chunks
            .par_iter()
            .progress_with(pb.clone())
            .map(|sample_indices| self.build_minibatch(sample_indices, target_device))
            .collect::<anyhow::Result<Vec<_>>>()?;
        pb.finish_and_clear();
        Ok(())
    }

    /// Retrieve a pre-computed minibatch. Panics if `precompute_all_minibatches`
    /// was not called after the last `shuffle_minibatch`.
    pub fn minibatch_cached(&self, batch_idx: usize) -> &IndexedMinibatchData {
        &self.cached_batches[batch_idx]
    }

    pub fn num_data(&self) -> usize {
        self.minibatches.samples.len()
    }

    pub fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    pub fn input_context_size(&self) -> usize {
        self.input_context_size
    }

    pub fn output_context_size(&self) -> usize {
        self.output_context_size
    }

    pub fn n_input_features(&self) -> usize {
        self.n_input_features
    }

    pub fn n_output_features(&self) -> usize {
        self.n_output_features
    }

    /// Build a packed minibatch.
    ///
    /// Encoder side packs `(indices, values)` directly into `[N, K_in]` —
    /// no union and no `[N, S_in]`. Decoder side builds the union and
    /// per-cell scatter positions; values are packed in per-cell index
    /// order into `[N, K_out]`. Nothing is materialized at `[N, S]` shape
    /// on host or device.
    #[allow(clippy::type_complexity)]
    fn build_minibatch(
        &self,
        sample_indices: &[usize],
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let k_in = self.input_context_size;
        let k_out = self.output_context_size;

        let (input_result, output_result) = rayon::join(
            || -> anyhow::Result<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
                let (input_indices, input_values) =
                    pack_indices_values(&self.input_samples, sample_indices, k_in, target_device)?;
                let input_values_null = match self.input_null_rows.as_ref() {
                    Some(rows) => Some(pack_null_at_indices(
                        &self.input_samples,
                        rows,
                        sample_indices,
                        k_in,
                        target_device,
                    )?),
                    None => None,
                };
                let input_values_mean = match self.input_mean.as_ref() {
                    Some(b) => Some(gather_per_feature_at_indices(
                        &self.input_samples,
                        sample_indices,
                        b,
                        k_in,
                        target_device,
                    )?),
                    None => None,
                };
                Ok((
                    input_indices,
                    input_values,
                    input_values_null,
                    input_values_mean,
                ))
            },
            || -> anyhow::Result<(Tensor, Tensor, Tensor, Vec<u32>, Option<Tensor>)> {
                // Decoder side: union + per-cell scatter positions, plus
                // values packed in per-cell index order. The decoder never
                // needs the raw feature ids — only their positions in the
                // union — so we don't keep an `indices [N, K_out]` tensor.
                let (union_indices, scatter_pos, union_vec) = build_union_and_scatter_pos(
                    &self.output_samples,
                    sample_indices,
                    self.n_output_features,
                    k_out,
                    target_device,
                )?;
                let n_batch = sample_indices.len();
                let mut val_buf = vec![0.0f32; n_batch * k_out];
                for (row, &si) in sample_indices.iter().enumerate() {
                    let s = &self.output_samples[si];
                    let off = row * k_out;
                    let take = s.values.len().min(k_out);
                    val_buf[off..off + take].copy_from_slice(&s.values[..take]);
                }
                let values = Tensor::from_vec(val_buf, (n_batch, k_out), target_device)?;
                let output_values_weight = match self.output_fisher_weights.as_ref() {
                    Some(w) => Some(gather_per_feature_at_indices(
                        &self.output_samples,
                        sample_indices,
                        w,
                        k_out,
                        target_device,
                    )?),
                    None => None,
                };
                Ok((
                    union_indices,
                    scatter_pos,
                    values,
                    union_vec,
                    output_values_weight,
                ))
            },
        );
        let (input_indices, input_values, input_values_null, input_values_mean) = input_result?;
        let (
            output_union_indices,
            output_scatter_pos,
            output_values,
            output_union_vec,
            output_values_weight,
        ) = output_result?;

        // Slice log selection frequency at output union positions.
        let log_q_s: Vec<f32> = output_union_vec
            .iter()
            .map(|&idx| self.output_log_q[idx as usize])
            .collect();
        let s_out = output_union_vec.len();
        let output_log_q_s = Tensor::from_vec(log_q_s, (1, s_out), target_device)?;

        Ok(IndexedMinibatchData {
            input_indices,
            input_values,
            input_values_null,
            input_values_mean,
            output_union_indices,
            output_scatter_pos,
            output_values,
            output_values_weight,
            output_log_q_s,
        })
    }

    /// Build an indexed minibatch from the shuffled indices.
    pub fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let sample_indices = self.minibatches.chunks.get(batch_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid batch index {} vs total {}",
                batch_idx,
                self.minibatches.chunks.len()
            )
        })?;

        self.build_minibatch(sample_indices, target_device)
    }

    /// Build an indexed minibatch from ordered (non-shuffled) sample range.
    pub fn minibatch_ordered(
        &self,
        lb: usize,
        ub: usize,
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let sample_indices: Vec<usize> = (lb..ub).collect();
        self.build_minibatch(&sample_indices, target_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn assert_union_eq_set(union: &[u32], expected: &[u32]) {
        let mut sorted = union.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, expected);
    }

    fn pos_of(union: &[u32], feat: u32) -> usize {
        union
            .iter()
            .position(|&x| x == feat)
            .unwrap_or_else(|| panic!("feature {feat} not in union {union:?}"))
    }

    #[test]
    fn test_top_k_indices() {
        let row = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.7];
        let (indices, values) = top_k_indices(&row, 3);
        assert_eq!(indices, vec![1, 3, 5]);
        assert_eq!(values, vec![0.5, 0.9, 0.7]);
    }

    #[test]
    fn test_top_k_all_features() {
        let row = vec![0.1, 0.5, 0.3];
        let (indices, values) = top_k_indices(&row, 10);
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(values, vec![0.1, 0.5, 0.3]);
    }

    #[test]
    fn test_top_k_indices_weighted() {
        // raw row top-3 by value would be {3,5,1} (0.9, 0.7, 0.5).
        // Down-weight idx 3 and 5 ("housekeeping"); idx 0,2,4 should win.
        // Scoring is log1p(v) * w — same ordering on this small example.
        let row = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.7];
        let weights = vec![1.0, 0.1, 1.0, 0.01, 1.0, 0.05];
        let (indices, values) = top_k_indices_weighted(&row, &weights, 3);
        // log1p scores:
        //   0: ln1p(0.1)*1.0  ≈ 0.0953
        //   1: ln1p(0.5)*0.1  ≈ 0.0405
        //   2: ln1p(0.3)*1.0  ≈ 0.2624
        //   3: ln1p(0.9)*0.01 ≈ 0.0064
        //   4: ln1p(0.2)*1.0  ≈ 0.1823
        //   5: ln1p(0.7)*0.05 ≈ 0.0265
        // Top-3 = {2, 4, 0}, sorted by index → {0, 2, 4}.
        assert_eq!(indices, vec![0, 2, 4]);
        // Values returned must be the raw row entries — log1p only re-ranks.
        assert_eq!(values, vec![0.1, 0.3, 0.2]);
    }

    #[test]
    fn test_top_k_drops_zero_valued_features() {
        // Only 2 of 6 features are non-zero; with k=5, the result must be
        // length 2 (no zero-padding into top-K). Downstream packers handle
        // the short vec via `s.indices.len().min(k)`.
        let row = vec![0.0, 0.4, 0.0, 0.0, 0.7, 0.0];
        let weights = vec![1.0; 6];
        let (indices, values) = top_k_indices_weighted(&row, &weights, 5);
        assert_eq!(indices, vec![1, 4]);
        assert_eq!(values, vec![0.4, 0.7]);

        // All-zero row → empty result.
        let zero_row = vec![0.0; 6];
        let (idx0, val0) = top_k_indices_weighted(&zero_row, &weights, 3);
        assert!(idx0.is_empty());
        assert!(val0.is_empty());

        // Zero-weight features are also dropped even when their value is
        // positive (score = log1p(v) * 0 = 0).
        let row_wz = vec![0.4, 0.7, 0.3];
        let w_wz = vec![0.0, 1.0, 0.0];
        let (idx_wz, val_wz) = top_k_indices_weighted(&row_wz, &w_wz, 3);
        assert_eq!(idx_wz, vec![1]);
        assert_eq!(val_wz, vec![0.7]);
    }

    #[test]
    fn test_indexed_from_dense() {
        let data = DMatrix::<f32>::from_row_slice(
            4,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // sample 0: top-3 = {1,3,5}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // sample 1: top-3 = {0,2,4}
                0.3, 0.7, 0.1, 0.4, 0.6, 0.5, // sample 2: top-3 = {1,4,5}
                0.2, 0.3, 0.8, 0.1, 0.5, 0.9, // sample 3: top-3 = {2,4,5}
            ],
        );

        let w = vec![1.0f32; 6];
        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 3,
            output_context_size: 3,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.num_data(), 4);
        assert_eq!(indexed.n_input_features(), 6);
        assert_eq!(indexed.input_context_size(), 3);

        assert_eq!(indexed.input_samples[0].indices, vec![1, 3, 5]);
        assert_eq!(indexed.input_samples[0].values, vec![0.5, 0.9, 0.7]);

        assert_eq!(indexed.input_samples[1].indices, vec![0, 2, 4]);
        assert_eq!(indexed.input_samples[1].values, vec![0.8, 0.6, 0.9]);
    }

    #[test]
    fn test_packed_minibatch_shapes() {
        let data = DMatrix::<f32>::from_row_slice(
            3,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // top-2 = {3,5}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // top-2 = {0,4}
                0.3, 0.7, 0.1, 0.4, 0.6, 0.5, // top-2 = {1,4}
            ],
        );

        let w = vec![1.0f32; 6];
        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 2,
            output_context_size: 2,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        let mb = indexed.minibatch_ordered(0, 3, &Device::Cpu).unwrap();

        // Encoder side: packed [N, K_in].
        assert_eq!(mb.input_indices.dims(), &[3, 2]);
        assert_eq!(mb.input_values.dims(), &[3, 2]);

        let in_idx: Vec<Vec<u32>> = mb.input_indices.to_vec2().unwrap();
        let in_val: Vec<Vec<f32>> = mb.input_values.to_vec2().unwrap();
        assert_eq!(in_idx[0], vec![3, 5]);
        assert!((in_val[0][0] - 0.9).abs() < 1e-6);
        assert!((in_val[0][1] - 0.7).abs() < 1e-6);

        // Decoder side: union + scatter_pos + values [N, K_out].
        let union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_union_eq_set(&union, &[0, 1, 3, 4, 5]);
        assert_eq!(mb.output_scatter_pos.dims(), &[3, 2]);
        assert_eq!(mb.output_values.dims(), &[3, 2]);

        let scat: Vec<Vec<u32>> = mb.output_scatter_pos.to_vec2().unwrap();
        let vals: Vec<Vec<f32>> = mb.output_values.to_vec2().unwrap();

        // For sample 0, the per-cell ids are [3, 5]; their scatter positions
        // must point at exactly those entries in the union.
        assert_eq!(union[scat[0][0] as usize], 3);
        assert_eq!(union[scat[0][1] as usize], 5);
        assert!((vals[0][0] - 0.9).abs() < 1e-6);
        assert!((vals[0][1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_packed_minibatch_separate_output() {
        let input = DMatrix::<f32>::from_row_slice(
            2,
            4,
            &[
                0.9, 0.1, 0.5, 0.3, // top-2 = {0,2}
                0.2, 0.8, 0.4, 0.7, // top-2 = {1,3}
            ],
        );
        let output = DMatrix::<f32>::from_row_slice(
            2,
            4,
            &[
                10.0, 20.0, 30.0, 40.0, // top-2 = {2,3}
                50.0, 60.0, 70.0, 80.0, // top-2 = {2,3}
            ],
        );

        let w = vec![1.0f32; 4];
        let args = IndexedInMemoryArgs {
            input: &input,
            input_null: None,
            output: &output,
            input_context_size: 2,
            output_context_size: 2,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        let in_idx: Vec<Vec<u32>> = mb.input_indices.to_vec2().unwrap();
        assert_eq!(in_idx[0], vec![0, 2]);
        assert_eq!(in_idx[1], vec![1, 3]);

        let output_union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_union_eq_set(&output_union, &[2, 3]);

        let vals: Vec<Vec<f32>> = mb.output_values.to_vec2().unwrap();
        assert!((vals[0][0] - 30.0).abs() < 1e-6);
        assert!((vals[0][1] - 40.0).abs() < 1e-6);
        assert!((vals[1][0] - 70.0).abs() < 1e-6);
        assert!((vals[1][1] - 80.0).abs() < 1e-6);

        let scat: Vec<Vec<u32>> = mb.output_scatter_pos.to_vec2().unwrap();
        for row in &scat {
            for (kk, &pos) in row.iter().enumerate() {
                let expected_feat = if kk == 0 { 2u32 } else { 3u32 };
                assert_eq!(output_union[pos as usize], expected_feat);
            }
        }
    }

    #[test]
    fn test_packed_minibatch_different_context_sizes() {
        let data = DMatrix::<f32>::from_row_slice(
            2,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // input top-3={1,3,5}, output top-1={3}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // input top-3={0,2,4}, output top-1={4}
            ],
        );

        let w = vec![1.0f32; 6];
        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 3,
            output_context_size: 1,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.input_context_size(), 3);
        assert_eq!(indexed.output_context_size(), 1);

        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        // Encoder packed at K_in=3.
        assert_eq!(mb.input_indices.dims(), &[2, 3]);
        assert_eq!(mb.input_values.dims(), &[2, 3]);

        // Decoder packed at K_out=1; union {3,4} → S=2.
        assert_eq!(mb.output_scatter_pos.dims(), &[2, 1]);
        assert_eq!(mb.output_values.dims(), &[2, 1]);
        let union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_union_eq_set(&union, &[3, 4]);
    }

    /// A panic mid-`build_union_and_scatter_pos` must not poison the
    /// thread-local `POS_LOOKUP` for subsequent calls on the same thread.
    #[test]
    fn test_pos_lookup_panic_recovery() {
        use std::panic;

        // First call OOBs on idx=99, leaving slots 0..=2 with a dirty gen.
        let bad_samples = vec![
            IndexedSample {
                indices: vec![0, 1],
                values: vec![1.0, 1.0],
            },
            IndexedSample {
                indices: vec![2, 99],
                values: vec![1.0, 1.0],
            },
        ];
        let bad_indices = [0usize, 1];
        let result = panic::catch_unwind(|| {
            let _ = build_union_and_scatter_pos(&bad_samples, &bad_indices, 6, 2, &Device::Cpu);
        });
        assert!(result.is_err(), "expected the OOB sample to panic");

        // Clean call on the same thread: every feature in `good_samples`
        // must end up in the union — the gen bump invalidates the dirty
        // slots from the panicking call.
        let good_samples = vec![
            IndexedSample {
                indices: vec![0, 1],
                values: vec![10.0, 11.0],
            },
            IndexedSample {
                indices: vec![2, 3],
                values: vec![12.0, 13.0],
            },
        ];
        let (union_t, scatter_t, union_vec) =
            build_union_and_scatter_pos(&good_samples, &[0, 1], 6, 2, &Device::Cpu).unwrap();
        let union: Vec<u32> = union_t.to_vec1().unwrap();
        assert_union_eq_set(&union, &[0, 1, 2, 3]);
        assert_eq!(union_vec, union);

        let scat: Vec<Vec<u32>> = scatter_t.to_vec2().unwrap();
        // sample 0 → ids [0,1]; positions must point at those features.
        assert_eq!(union[scat[0][0] as usize], 0);
        assert_eq!(union[scat[0][1] as usize], 1);
        assert_eq!(union[scat[1][0] as usize], 2);
        assert_eq!(union[scat[1][1] as usize], 3);

        // A fresh union value lookup at scatter position 0 must read the
        // first union entry, not stale state from the panicking call.
        let _ = pos_of(&union, 0);
    }
}
