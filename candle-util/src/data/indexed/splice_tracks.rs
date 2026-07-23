//! Splice-aware indexed loader for `faba gem-encoder`.
//!
//! The single-track [`super::IndexedInMemoryData`] selects a per-cell top-K over
//! the **row** axis. That is wrong here: gem-format inputs carry two rows per
//! gene (`{gene}/count/spliced` and `{gene}/count/unspliced`), and the model is
//! gene-keyed — a gene's nascent and mature tracks must be selected *together*
//! or the pairing the encoder and decoder depend on is broken.
//!
//! So selection runs on the **pooled** `s + u` score and the packed minibatch
//! carries one gene id with both of its values:
//!
//! ```text
//! gene_indices   [N, K] u32   per-cell top-K gene ids
//! nascent_observed [N, K] f32   unspliced counts at those genes
//! mature_observed  [N, K] f32   spliced counts at those genes
//! ```
//!
//! A gene may legitimately be selected with `u = 0` and a large `s` (or the
//! reverse); that zero is a real observation, not a padding slot, and the
//! trainer's mask policy — not this loader — decides whether it is scored.
//! Padding (a cell with fewer than `K` positively-scoring genes) is the only
//! case where BOTH values are zero, and it is marked by `slot_valid = 0` —
//! that flag, not a value test, is what separates padding from a real zero.

use super::labeled_bar;
use super::top_k::top_k_from_entries;
use crate::data::loader_util::Minibatches;
use candle_core::{Device, Tensor};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

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

/// Per-cell top-K genes with both tracks' values.
#[derive(Clone)]
pub struct GemSample {
    /// Ascending gene ids.
    pub genes: Vec<u32>,
    /// Nascent counts, aligned with `genes`.
    pub nascent: Vec<f32>,
    /// Mature counts, aligned with `genes`.
    pub mature: Vec<f32>,
}

/// Packed splice-aware minibatch.
pub struct GemMinibatchData {
    /// `[N, K]` u32 — per-cell top-K gene ids.
    pub gene_indices: Tensor,
    /// `[N, K]` f32 — nascent OBSERVED counts (`μ_observed`, batch-mixed).
    pub nascent_observed: Tensor,
    /// `[N, K]` f32 — mature OBSERVED counts (`μ_observed`, batch-mixed).
    pub mature_observed: Tensor,
    /// `[N, K]` f32 — 1 where the slot is a real gene, 0 where it is padding.
    /// Padding must never be visible to the encoder nor scored by the decoder.
    pub slot_valid: Tensor,
    /// `[N, K]` f32 — batch RESIDUAL (`μ_residual`) at the genes' NASCENT rows.
    ///
    /// Per track, not shared. The residual is fitted per row, and intronic
    /// capture is exactly the kind of thing that varies by batch differently
    /// from exonic — so dividing both tracks by one of them would leave a
    /// residual `r^u/r^s` that is itself a per-gene splice-ratio distortion,
    /// i.e. it would land straight in `δ`, the estimand.
    pub nascent_residual: Option<Tensor>,
    /// `[N, K]` f32 — batch RESIDUAL (`μ_residual`) at the genes' MATURE rows.
    pub mature_residual: Option<Tensor>,
    /// `[N, K]` f32 — nascent ADJUSTED counts (`μ_adjusted`), the decoder's target.
    ///
    /// Separate from `nascent_observed` so the encoder can read the batch-mixed
    /// observation while the decoder is scored against the batch-free one; see
    /// [`GemIndexedArgs::output`]. `None` ⇒ score the encoder's own values,
    /// which is the un-adjusted behaviour.
    pub nascent_adjusted: Option<Tensor>,
    /// `[N, K]` f32 — mature ADJUSTED counts (`μ_adjusted`), the decoder's target.
    pub mature_adjusted: Option<Tensor>,
    /// `[N, K]` f32 — per-gene NB-Fisher weight at `gene_indices`.
    ///
    /// Multiplies the decoder's per-position log-likelihood, so a gene carrying
    /// little information about cell state contributes less to `β`'s gradient
    /// even once it is inside a cell's top-K. This mirrors senna's
    /// `output_values_weight` (`traits::indexed::forward_indexed_with_log_beta`,
    /// "Fisher-weighted multinomial NLL"), which gem-encoder had been applying
    /// to SHORTLIST SELECTION ONLY.
    pub values_weight: Option<Tensor>,
    /// `[N, K]` f32 — per-gene mean nascent rate at `gene_indices`.
    pub nascent_mean: Option<Tensor>,
    /// `[N, K]` f32 — per-gene mean mature rate at `gene_indices`.
    pub mature_mean: Option<Tensor>,
}

impl GemMinibatchData {
    /// Upload every field to `dev`. Batches are precomputed host-side, so a GPU
    /// run uploads incrementally rather than holding the epoch resident.
    pub fn to_device(&self, dev: &Device) -> anyhow::Result<GemMinibatchData> {
        let opt = |t: &Option<Tensor>| -> anyhow::Result<Option<Tensor>> {
            t.as_ref()
                .map(|x| x.to_device(dev))
                .transpose()
                .map_err(Into::into)
        };
        Ok(GemMinibatchData {
            gene_indices: self.gene_indices.to_device(dev)?,
            nascent_observed: self.nascent_observed.to_device(dev)?,
            mature_observed: self.mature_observed.to_device(dev)?,
            slot_valid: self.slot_valid.to_device(dev)?,
            nascent_residual: opt(&self.nascent_residual)?,
            mature_residual: opt(&self.mature_residual)?,
            nascent_adjusted: opt(&self.nascent_adjusted)?,
            mature_adjusted: opt(&self.mature_adjusted)?,
            values_weight: opt(&self.values_weight)?,
            nascent_mean: opt(&self.nascent_mean)?,
            mature_mean: opt(&self.mature_mean)?,
        })
    }
}

/// Select a cell's top-K genes from its full **row**-space vector.
///
/// Scoring pools the two tracks (`log1p(s + u) · w_g`) so a gene is chosen on
/// its total evidence and both tracks come along. `gene_weights` is the
/// per-gene shortlist weight (NB-Fisher style); pooling the weight to the gene
/// level is the caller's job.
#[must_use]
pub fn top_k_genes_from_row(
    row: &[f32],
    map: &GeneTrackMap,
    gene_weights: &[f32],
    k: usize,
) -> GemSample {
    debug_assert_eq!(row.len(), map.row_to_gene.len());
    debug_assert_eq!(gene_weights.len(), map.n_genes);

    let mut nascent = vec![0f32; map.n_genes];
    let mut mature = vec![0f32; map.n_genes];
    for (r, &v) in row.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let g = map.row_to_gene[r] as usize;
        if map.row_is_nascent[r] {
            nascent[g] += v;
        } else {
            mature[g] += v;
        }
    }

    let (genes, _) = top_k_from_entries(
        (0..map.n_genes).map(|g| (g as u32, nascent[g] + mature[g], gene_weights[g])),
        k,
    );
    let n_vals = genes.iter().map(|&g| nascent[g as usize]).collect();
    let m_vals = genes.iter().map(|&g| mature[g as usize]).collect();
    GemSample {
        genes,
        nascent: n_vals,
        mature: m_vals,
    }
}

/// Select each column's top-K genes straight from a sparse `[D_rows, N]` CSC
/// matrix — columns are cells, rows are the gem feature axis.
///
/// The inference path uses this: materializing the dense `[N_cells, D_rows]`
/// the [`GemIndexedData::from_dense`] constructor needs is not viable at cell
/// scale, and only the stored nonzeros matter for selection anyway.
///
/// Parallel over chunks of columns. An earlier version of this comment claimed
/// it was "sequential by design — callers already run this inside a block-level
/// parallel loop"; that was simply wrong. Its only caller walks cell blocks in a
/// plain `while` loop, so the claim left the whole inference pass single-threaded.
#[must_use]
pub fn gem_samples_from_csc(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    map: &GeneTrackMap,
    gene_weights: &[f32],
    k: usize,
) -> Vec<GemSample> {
    debug_assert_eq!(x_dn.nrows(), map.row_to_gene.len());

    // Parallel for the same reason `from_dense` is: a per-cell scan of a sparse
    // column plus a top-K rank over the whole gene axis is the dominant CPU cost
    // of the inference pass, and this was the one path still running it on a
    // single thread.
    //
    // Chunked rather than per-cell. A cell is far too small a unit of work: the
    // two dense scratch buffers are `n_genes` floats each (~270 kB together at
    // 34 k genes), so a per-cell task would allocate and free that once per
    // cell. Per chunk, they are allocated once and reused, and rayon schedules
    // hundreds of tasks instead of tens of thousands.
    //
    // Order is load-bearing — `samples[j]` must stay cell `j` — so this collects
    // per-chunk vectors through an INDEXED `par_chunks`/`map` and flattens them
    // sequentially, rather than relying on a flattening combinator to preserve
    // order.
    const CELLS_PER_TASK: usize = 64;

    let cols: Vec<usize> = (0..x_dn.ncols()).collect();
    let per_chunk: Vec<Vec<GemSample>> = cols
        .par_chunks(CELLS_PER_TASK)
        .map(|chunk| {
            let mut nascent = vec![0f32; map.n_genes];
            let mut mature = vec![0f32; map.n_genes];
            let mut touched: Vec<usize> = Vec::with_capacity(1024);
            let mut out = Vec::with_capacity(chunk.len());
            for &j in chunk {
                touched.clear();
                let col = x_dn.col(j);
                for (&r, &v) in col.row_indices().iter().zip(col.values().iter()) {
                    if v == 0.0 {
                        continue;
                    }
                    let g = map.row_to_gene[r] as usize;
                    // First time this gene is hit in this cell. Counts are
                    // non-negative and `v == 0` is skipped above, so a zero sum
                    // is exactly "not yet touched".
                    if nascent[g] + mature[g] == 0.0 {
                        touched.push(g);
                    }
                    if map.row_is_nascent[r] {
                        nascent[g] += v;
                    } else {
                        mature[g] += v;
                    }
                }
                // Rank over the genes this cell actually observed, NOT all
                // `n_genes` — the same O(nnz) shape as the sibling
                // `csc_columns_to_indexed_samples`, which streams sparse
                // entries into `top_k_from_entries` and never goes dense. A
                // zero-count gene cannot outrank an observed one, so scanning
                // the ~80 % of the axis that is zero only costs time. This also
                // matches the sibling's behaviour when a cell has fewer than `k`
                // observed genes: it yields fewer than `k` rather than padding
                // with zero-count genes.
                let (genes, _) = top_k_from_entries(
                    touched
                        .iter()
                        .map(|&g| (g as u32, nascent[g] + mature[g], gene_weights[g])),
                    k,
                );
                let n_vals = genes.iter().map(|&g| nascent[g as usize]).collect();
                let m_vals = genes.iter().map(|&g| mature[g as usize]).collect();
                out.push(GemSample {
                    genes,
                    nascent: n_vals,
                    mature: m_vals,
                });
                // Reset only what was written. The buffers are shared across the
                // whole chunk, so skipping this would leak cell j's counts into
                // cell j+1; doing it over `touched` rather than the full axis is
                // what keeps the per-cell cost O(nnz).
                for &g in &touched {
                    nascent[g] = 0.0;
                    mature[g] = 0.0;
                }
            }
            out
        })
        .collect();

    per_chunk.into_iter().flatten().collect()
}

/// Splice-aware minibatch source over dense `[N_samples, D_rows]` input.
pub struct GemIndexedData {
    samples: Vec<GemSample>,
    /// Per-sample dense RESIDUAL row (`μ_residual`), row-space; gathered at each
    /// selected gene's OWN track row.
    residual_rows: Option<Vec<Vec<f32>>>,
    /// Per-sample dense ADJUSTED row (`μ_adjusted`), row-space.
    adjusted_rows: Option<Vec<Vec<f32>>>,
    /// Per-gene mean nascent / mature rate, gene-space.
    nascent_mean: Option<Vec<f32>>,
    mature_mean: Option<Vec<f32>>,
    /// Per-gene NB-Fisher weight, gene-space — the SAME weights that score the
    /// top-K shortlist, carried through so the decoder can weight the loss with
    /// them too. `None` on the inference path, which scores nothing.
    fisher_weights: Option<Vec<f32>>,
    /// Row id of each gene's nascent / mature track, for the null gather.
    gene_nascent_row: Vec<Option<u32>>,
    gene_mature_row: Vec<Option<u32>>,
    context_size: usize,
    n_genes: usize,
    minibatches: Minibatches,
    cached: Vec<GemMinibatchData>,
}

pub struct GemIndexedArgs<'a, D>
where
    D: matrix_util::traits::CandleDataLoaderOps,
{
    /// `[N_samples, D_rows]` **observed** counts (`μ_observed`, batch-mixed) —
    /// rows are the gem feature axis. What the ENCODER reads.
    pub observed: &'a D,
    /// `[N_samples, D_rows]` per-batch **residual** (`μ_residual`), optional.
    /// The batch signal handed to the encoder alongside `observed`.
    pub residual: Option<&'a D>,
    /// `[N_samples, D_rows]` batch-free **adjusted** counts (`μ_adjusted`),
    /// optional. What the DECODER is scored against.
    ///
    /// When given, the encoder reads `observed` (batch-mixed) with `residual` as
    /// its batch signal, while the decoder is scored against this batch-free
    /// matrix. That is what makes batch adjustment symmetric: nothing has to be
    /// "restored" on the decoder side, so the multinomial head — which ignores
    /// the residual entirely — needs no special case.
    ///
    /// `None` ⇒ decoder scores `observed`, the un-adjusted behaviour.
    /// Mirrors `masked_topic`'s `(observed, residual, adjusted)`
    /// triple, which this loader had collapsed to a single matrix.
    pub adjusted: Option<&'a D>,
    /// Row → (gene, track) map for `input`'s feature axis.
    pub map: &'a GeneTrackMap,
    /// Top-K genes per cell.
    pub context_size: usize,
    /// Per-**gene** shortlist weight (length `G`).
    pub gene_weights: &'a [f32],
    /// Per-**gene** mean nascent rate (length `G`), optional.
    pub nascent_mean: Option<&'a [f32]>,
    /// Per-**gene** mean mature rate (length `G`), optional.
    pub mature_mean: Option<&'a [f32]>,
}

impl GemIndexedData {
    pub fn from_dense<D>(args: GemIndexedArgs<D>) -> anyhow::Result<Self>
    where
        D: matrix_util::traits::CandleDataLoaderOps + Sync,
    {
        let (n_samples, n_rows) = args.observed.data_shape();
        anyhow::ensure!(
            n_rows == args.map.row_to_gene.len(),
            "GemIndexedData: input has {n_rows} rows but the gene/track map covers {}",
            args.map.row_to_gene.len()
        );
        anyhow::ensure!(
            args.gene_weights.len() == args.map.n_genes,
            "GemIndexedData: gene_weights has {} entries but the map has {} genes",
            args.gene_weights.len(),
            args.map.n_genes
        );
        // The `(observed, residual, adjusted)` triple must share ONE axis: the
        // per-track gathers below index `rows[si][r]` with `r` taken from the
        // gene/track map, which is built against `observed`. A short matrix
        // panics mid-epoch; a wide one reads the WRONG ROW silently and scores
        // the decoder against a different gene's counts. Only `observed` was
        // checked before, so the two matrices that can go wrong quietly were
        // the two that went unchecked.
        for (label, d) in [("residual", args.residual), ("adjusted", args.adjusted)] {
            if let Some(d) = d {
                let (rn, rd) = d.data_shape();
                anyhow::ensure!(
                    (rn, rd) == (n_samples, n_rows),
                    "GemIndexedData: `{label}` is {rn}×{rd} but `observed` is \
                     {n_samples}×{n_rows} — the three must be the same matrix shape, \
                     on the same feature axis, from the same collapse"
                );
            }
        }
        for (label, v) in [
            ("nascent_mean", args.nascent_mean),
            ("mature_mean", args.mature_mean),
        ] {
            if let Some(v) = v {
                anyhow::ensure!(
                    v.len() == args.map.n_genes,
                    "GemIndexedData: {label} has {} entries but the map has {} genes",
                    v.len(),
                    args.map.n_genes
                );
            }
        }

        let context_size = args.context_size.min(args.map.n_genes);
        let prog = labeled_bar("Top-K (genes)", n_samples as u64);
        let samples: Vec<GemSample> = (0..n_samples)
            .into_par_iter()
            .progress_with(prog.clone())
            .map(|i| {
                let row = args.observed.row_to_f32_vec(i);
                top_k_genes_from_row(&row, args.map, args.gene_weights, context_size)
            })
            .collect();
        prog.finish_and_clear();

        let read_rows = |d: &D, label: &str| -> Vec<Vec<f32>> {
            let (n, _) = d.data_shape();
            let bar = labeled_bar(label, n as u64);
            let rows = (0..n)
                .into_par_iter()
                .progress_with(bar.clone())
                .map(|i| d.row_to_f32_vec(i))
                .collect();
            bar.finish_and_clear();
            rows
        };
        let adjusted_rows = args.adjusted.map(|d| read_rows(d, "Target rows"));
        let residual_rows = args.residual.map(|d| read_rows(d, "Residual rows"));

        let (gene_nascent_row, gene_mature_row) = args.map.per_gene_rows();

        Ok(Self {
            samples,
            residual_rows,
            adjusted_rows,
            fisher_weights: Some(args.gene_weights.to_vec()),
            nascent_mean: args.nascent_mean.map(<[f32]>::to_vec),
            mature_mean: args.mature_mean.map(<[f32]>::to_vec),
            gene_nascent_row,
            gene_mature_row,
            context_size,
            n_genes: args.map.n_genes,
            minibatches: Minibatches {
                samples: (0..n_samples).collect(),
                chunks: vec![],
            },
            cached: vec![],
        })
    }

    /// Wrap already-selected samples (e.g. from [`gem_samples_from_csc`]).
    ///
    /// The inference path streams CSC blocks and builds one of these per block,
    /// so the packing, gathering, and device upload stay in one place instead of
    /// being reimplemented cell-side.
    ///
    /// `residual_rows`, when supplied, is one dense **row-space** residual vector per
    /// sample — the caller has already expanded whatever per-cell batch offset
    /// applies. It must be given here whenever it was given at training time, or
    /// the encoder sees a different input distribution than it was fitted on.
    pub fn from_samples(
        samples: Vec<GemSample>,
        map: &GeneTrackMap,
        context_size: usize,
        nascent_mean: Option<&[f32]>,
        mature_mean: Option<&[f32]>,
        residual_rows: Option<Vec<Vec<f32>>>,
    ) -> anyhow::Result<Self> {
        let n = samples.len();
        // Same axis contract as `from_dense`, enforced here too: the gathers
        // index `residual_rows[si][r]` with `si` a sample and `r` a ROW id from
        // the map. The doc above states this invariant; nothing used to check
        // it, and inference is exactly where a mismatch would be invisible —
        // the encoder would simply be handed a different input distribution
        // than it was fitted on, and still produce a latent.
        if let Some(rows) = residual_rows.as_ref() {
            anyhow::ensure!(
                rows.len() == n,
                "GemIndexedData::from_samples: {} residual rows for {n} samples",
                rows.len()
            );
            let n_rows = map.row_to_gene.len();
            if let Some((i, bad)) = rows.iter().enumerate().find(|(_, r)| r.len() != n_rows) {
                anyhow::bail!(
                    "GemIndexedData::from_samples: residual row {i} has {} entries but the \
                     gene/track map covers {n_rows} feature rows",
                    bad.len()
                );
            }
        }
        let (gene_nascent_row, gene_mature_row) = map.per_gene_rows();
        Ok(Self {
            samples,
            residual_rows,
            // Inference never scores a decoder, so it needs neither a target nor
            // the loss weights.
            adjusted_rows: None,
            fisher_weights: None,
            nascent_mean: nascent_mean.map(<[f32]>::to_vec),
            mature_mean: mature_mean.map(<[f32]>::to_vec),
            gene_nascent_row,
            gene_mature_row,
            context_size: context_size.min(map.n_genes),
            n_genes: map.n_genes,
            minibatches: Minibatches {
                samples: (0..n).collect(),
                chunks: vec![],
            },
            cached: vec![],
        })
    }

    #[must_use]
    pub fn num_data(&self) -> usize {
        self.minibatches.samples.len()
    }
    #[must_use]
    pub fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }
    #[must_use]
    pub fn context_size(&self) -> usize {
        self.context_size
    }
    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.n_genes
    }

    /// Total observed counts per track across the whole epoch — the
    /// denominators the trainer reports likelihood against.
    #[must_use]
    pub fn total_counts(&self) -> (f32, f32) {
        self.samples.iter().fold((0.0, 0.0), |(u, s), smp| {
            (
                u + smp.nascent.iter().sum::<f32>(),
                s + smp.mature.iter().sum::<f32>(),
            )
        })
    }

    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
        self.cached.clear();
    }

    pub fn precompute_all_minibatches(&mut self) -> anyhow::Result<()> {
        let bar = labeled_bar("Minibatch precompute", self.minibatches.chunks.len() as u64);
        self.cached = self
            .minibatches
            .chunks
            .par_iter()
            .progress_with(bar.clone())
            .map(|idx| self.build_minibatch(idx, &Device::Cpu))
            .collect::<anyhow::Result<Vec<_>>>()?;
        bar.finish_and_clear();
        Ok(())
    }

    /// Retrieve a precomputed minibatch. Panics if
    /// [`Self::precompute_all_minibatches`] has not run since the last shuffle.
    #[must_use]
    pub fn minibatch_cached(&self, batch_idx: usize) -> &GemMinibatchData {
        &self.cached[batch_idx]
    }

    /// Build a minibatch over an ordered (non-shuffled) range — the inference
    /// path, which must visit cells in storage order.
    pub fn minibatch_ordered(
        &self,
        lb: usize,
        ub: usize,
        dev: &Device,
    ) -> anyhow::Result<GemMinibatchData> {
        let idx: Vec<usize> = (lb..ub).collect();
        self.build_minibatch(&idx, dev)
    }

    fn build_minibatch(
        &self,
        sample_indices: &[usize],
        dev: &Device,
    ) -> anyhow::Result<GemMinibatchData> {
        let k = self.context_size;
        let n = sample_indices.len();

        // The three [N, K] core buffers fill in lockstep over one pass.
        let mut idx_buf = vec![0u32; n * k];
        let mut nas_buf = vec![0f32; n * k];
        let mut mat_buf = vec![0f32; n * k];
        let mut valid_buf = vec![0f32; n * k];
        idx_buf
            .par_chunks_mut(k)
            .zip(nas_buf.par_chunks_mut(k))
            .zip(mat_buf.par_chunks_mut(k))
            .zip(valid_buf.par_chunks_mut(k))
            .zip(sample_indices.par_iter())
            .for_each(|((((ic, nc), mc), vc), &si)| {
                let s = &self.samples[si];
                let take = s.genes.len().min(k);
                ic[..take].copy_from_slice(&s.genes[..take]);
                nc[..take].copy_from_slice(&s.nascent[..take]);
                mc[..take].copy_from_slice(&s.mature[..take]);
                vc[..take].fill(1.0);
            });

        let gene_indices =
            Tensor::from_vec(idx_buf, (n, k), dev)?.to_dtype(candle_core::DType::U32)?;
        let nascent_observed = Tensor::from_vec(nas_buf, (n, k), dev)?;
        let mature_observed = Tensor::from_vec(mat_buf, (n, k), dev)?;
        let slot_valid = Tensor::from_vec(valid_buf, (n, k), dev)?;

        // The residual and the adjusted counts are both per-ROW quantities, so
        // each track reads its OWN row. `missing` is the value for a gene that
        // lacks that track: 1.0 for the residual (a neutral multiplier, since it
        // is a divisor) and 0.0 for the adjusted counts (nothing to score).
        let per_track = |rows: &Option<Vec<Vec<f32>>>,
                         gene_row: &[Option<u32>],
                         missing: f32|
         -> Option<anyhow::Result<Tensor>> {
            rows.as_ref().map(|rows| {
                self.gather_gene(sample_indices, k, dev, |si, g| {
                    gene_row[g as usize].map_or(missing, |r| rows[si][r as usize])
                })
            })
        };

        let nascent_residual =
            per_track(&self.residual_rows, &self.gene_nascent_row, 1.0).transpose()?;
        let mature_residual =
            per_track(&self.residual_rows, &self.gene_mature_row, 1.0).transpose()?;
        let nascent_adjusted =
            per_track(&self.adjusted_rows, &self.gene_nascent_row, 0.0).transpose()?;
        let mature_adjusted =
            per_track(&self.adjusted_rows, &self.gene_mature_row, 0.0).transpose()?;

        let values_weight = self
            .fisher_weights
            .as_ref()
            .map(|w| self.gather_gene(sample_indices, k, dev, |_, g| w[g as usize]))
            .transpose()?;
        let nascent_mean = self
            .nascent_mean
            .as_ref()
            .map(|mu| self.gather_gene(sample_indices, k, dev, |_, g| mu[g as usize]))
            .transpose()?;
        let mature_mean = self
            .mature_mean
            .as_ref()
            .map(|mu| self.gather_gene(sample_indices, k, dev, |_, g| mu[g as usize]))
            .transpose()?;

        Ok(GemMinibatchData {
            gene_indices,
            nascent_observed,
            mature_observed,
            slot_valid,
            nascent_residual,
            mature_residual,
            nascent_adjusted,
            mature_adjusted,
            values_weight,
            nascent_mean,
            mature_mean,
        })
    }

    /// Gather a per-`(sample, gene)` scalar at each cell's selected genes into
    /// `[N, K]`. Padding slots keep `0.0`, which is inert everywhere it is used
    /// (the divisive Anscombe null floors it, and `slot_valid` masks it out).
    fn gather_gene<F>(
        &self,
        sample_indices: &[usize],
        k: usize,
        dev: &Device,
        f: F,
    ) -> anyhow::Result<Tensor>
    where
        F: Fn(usize, u32) -> f32 + Sync,
    {
        let n = sample_indices.len();
        let mut buf = vec![0f32; n * k];
        buf.par_chunks_mut(k)
            .zip(sample_indices.par_iter())
            .for_each(|(chunk, &si)| {
                let genes = &self.samples[si].genes;
                for (kk, &g) in genes.iter().take(k).enumerate() {
                    chunk[kk] = f(si, g);
                }
            });
        Ok(Tensor::from_vec(buf, (n, k), dev)?)
    }
}

#[cfg(test)]
#[path = "splice_tracks_tests.rs"]
mod tests;
