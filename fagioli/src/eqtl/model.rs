//! CP / PARAFAC hyperedge model over variants, genes, and contexts.
//!
//! ```text
//! score(variant j, gene g, context k) = sum_h u[j,h] * v[g,h] * c[k,h]
//! ```
//!
//! A cell type is a GATE over programs: `c_k` says which programs are
//! active there, so a variant-gene pair is expressed in `k` only where
//! their shared programs overlap `c_k`'s. The two pseudo-contexts are
//! special VALUES of `c`, never special cases in the code:
//!
//! - `c[ubiquitous] = 1` (fixed, never learned) — the score collapses to
//!   `<u_j, v_g>`, the context-free anchor.
//! - `empty` — an abstention candidate scoring exactly 0 with no
//!   parameters, so every real candidate competes against an absolute
//!   threshold and not only against its siblings.
//!
//! Negatives corrupt ONE slot of the hyperedge: the context slot (the
//! specificity contrast) or the gene slot (which gene this variant-context
//! pair targets). Both pools are drawn from CERTIFIED-ABSENT cells only;
//! unknown cells are sampled in neither class.
//!
//! ## Why this form
//!
//! Validated on real single-cell eQTL summary statistics: the fit
//! separates held-out edges from certified absences well above chance,
//! the gate's effective rank settles below H, and contexts sharing a
//! label prefix have more similar gates than contexts that do not.
//! Permuting the edge/absent labels collapses every one of those, so
//! the geometry is learned rather than fitted.
//!
//! Two earlier architectures failed on the same data, with no recoverable
//! structure: free per-tuple deviation vectors, and additive scalar
//! deviations. The three-way multiplicative form with an explicit
//! cell-type embedding is what works.
//!
//! ## Two traps this code is written around
//!
//! - **Freezing `c[ubiquitous]`.** Grad masking alone does not freeze a row
//!   under AdamW; decay and stale momentum still move it. The row is masked
//!   out of the forward pass AND restored after every optimizer step.
//! - **Sparse regularization.** The ridge penalty touches only the rows in
//!   the batch. Applied densely it runs thousands of penalty updates against
//!   tens of data updates per entity and crushes every parameter to zero
//!   regardless of evidence — the loss pins at `log(1 + nu)`.

use anyhow::{ensure, Result};
use candle_util::candle_core::{DType, Device, Tensor, Var};
use candle_util::candle_nn::{ops, AdamW, Optimizer, ParamsAdamW};
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rustc_hash::FxHashMap as HashMap;

use super::evidence::{EvidenceTable, State};

/// Everything the fit needs beyond the evidence table.
#[derive(Debug, Clone)]
pub struct EqtlModelConfig {
    pub embedding_dim: usize,
    pub num_negatives: usize,
    pub num_iterations: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Ridge weight, applied only to the rows a batch touched.
    pub ridge: f64,
    /// Fraction of real-context edges hidden from training entirely.
    pub holdout_frac: f64,
    /// Permute the edge/absent labels before training (negative control).
    pub shuffle_control: bool,
    pub seed: u64,
}

/// A compacted index space: only entities appearing in a fitted row get a
/// row of the embedding, so nothing is emitted that no evidence trained.
#[derive(Debug)]
pub struct EntityMap {
    pub names: Vec<Box<str>>,
    /// Source id -> fitted row, `None` when the entity never appeared.
    pub slot: Vec<Option<u32>>,
}

impl EntityMap {
    fn new(source: &[Box<str>], used: &[bool]) -> Self {
        let mut names = Vec::new();
        let mut slot = vec![None; source.len()];
        for (id, name) in source.iter().enumerate() {
            if used[id] {
                slot[id] = Some(names.len() as u32);
                names.push(name.clone());
            }
        }
        Self { names, slot }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Diagnostics reported next to the fit.
///
/// Serialized wholesale into `parameters.json`, so a field added here reaches
/// the output without a second edit somewhere else.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FitMetrics {
    pub embedding_dim: usize,
    pub n_variants: usize,
    pub n_genes: usize,
    pub n_contexts: usize,
    pub n_edges_real: usize,
    pub n_edges_ubiquitous: usize,
    pub n_absent_real: usize,
    pub n_heldout: usize,
    pub n_positives: usize,
    pub n_positives_without_negatives: usize,
    pub n_variants_with_positives: usize,
    pub auc_in_sample: f64,
    pub auc_heldout: f64,
    pub auprc_heldout: f64,
    pub auprc_base: f64,
    pub auprc_lift: f64,
    pub generalization_gap: f64,
    pub effective_rank: f64,
    pub within_group_cosine: Option<f64>,
    pub between_group_cosine: Option<f64>,
    pub mean_score_train_edge: f64,
    pub mean_score_heldout_edge: f64,
    pub mean_score_absent: f64,
    pub final_loss: f32,
}

/// The fitted embeddings, their index spaces, and the diagnostics.
#[derive(Debug)]
pub struct EqtlFit {
    pub variants: EntityMap,
    pub genes: EntityMap,
    pub contexts: EntityMap,
    /// Fitted row of the ubiquitous pseudo-context.
    pub ubiquitous: Option<u32>,
    pub u: DMatrix<f32>,
    pub v: DMatrix<f32>,
    pub c: DMatrix<f32>,
    pub metrics: FitMetrics,
}

impl EqtlFit {
    /// The elementwise product `u_j * v_g`, shared by every context of a pair.
    ///
    /// The gate is the only per-context factor, so a caller scoring one pair
    /// across many contexts computes this once instead of once per context.
    pub fn pair_product(&self, variant: usize, gene: usize) -> Vec<f32> {
        (0..self.u.ncols())
            .map(|h| self.u[(variant, h)] * self.v[(gene, h)])
            .collect()
    }

    /// The context-free anchor `<u_j, v_g>`, i.e. the ubiquitous-gated score.
    ///
    /// This is [`Self::score`] with the gate at one, written separately
    /// because the ubiquitous row is absent whenever no pair had two cell
    /// types to meta-analyse.
    pub fn anchor(&self, variant: usize, gene: usize) -> f32 {
        self.pair_product(variant, gene).iter().sum()
    }

    /// The gated score of one hyperedge.
    pub fn score(&self, variant: usize, gene: usize, context: usize) -> f32 {
        self.gated(&self.pair_product(variant, gene), context)
    }

    /// One context's gate applied to a precomputed [`Self::pair_product`].
    pub fn gated(&self, pair_product: &[f32], context: usize) -> f32 {
        pair_product
            .iter()
            .enumerate()
            .map(|(h, uv)| uv * self.c[(context, h)])
            .sum()
    }

    /// Every context but the ubiquitous pseudo-context, in fitted-row order.
    pub fn real_contexts(&self) -> Vec<usize> {
        (0..self.contexts.len())
            .filter(|&k| Some(k as u32) != self.ubiquitous)
            .collect()
    }
}

/// One row of the fitted table, in compacted slot space.
#[derive(Debug, Clone, Copy)]
struct FittedRow {
    variant: u32,
    gene: u32,
    context: u32,
    state: State,
    heldout: bool,
}

/// Which slot of the hyperedge a negative corrupts.
///
/// A negative replaces exactly ONE slot, and the two choices are the same
/// mechanism seen from two sides: fix (variant, gene) and vary the context
/// for the specificity contrast, or fix (variant, context) and vary the gene
/// for the targeting contrast. Everything downstream — pool construction,
/// sampling, the fallback when a pool is empty — is shared; only reading the
/// key out of a row and writing the pick back into its index array differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Context = 0,
    Gene = 1,
}

impl Slot {
    const N: usize = 2;
    const ALL: [Slot; Slot::N] = [Slot::Context, Slot::Gene];

    /// The part of the hyperedge held FIXED while this slot varies.
    fn key(self, row: &FittedRow) -> (u32, u32) {
        match self {
            Slot::Context => (row.variant, row.gene),
            Slot::Gene => (row.variant, row.context),
        }
    }

    /// The entity this slot contributes to a pool.
    fn value(self, row: &FittedRow) -> u32 {
        match self {
            Slot::Context => row.context,
            Slot::Gene => row.gene,
        }
    }
}

/// One training positive together with its per-slot negative pools.
#[derive(Debug, Clone, Copy)]
struct Positive {
    variant: u32,
    gene: u32,
    context: u32,
    /// `(start, len)` into the pool of each [`Slot`].
    pool: [(u32, u32); Slot::N],
}

/// Fit the hyperedge model.
pub fn train(
    evidence: &EvidenceTable,
    variant_names: &[Box<str>],
    gene_names: &[Box<str>],
    config: &EqtlModelConfig,
    device: &Device,
) -> Result<EqtlFit> {
    ensure!(config.embedding_dim > 0, "--embedding-dim must be positive");
    ensure!(config.num_negatives > 0, "--num-negatives must be positive");
    ensure!(config.batch_size > 0, "--batch-size must be positive");
    ensure!(
        (0.0..1.0).contains(&config.holdout_frac),
        "--holdout-frac must be in [0, 1)"
    );

    let mut rng = StdRng::seed_from_u64(config.seed);

    // ── Fitted table: edges and certified absences only ──────────────────
    // Unknown cells never enter, in either class.
    let kept: Vec<usize> = (0..evidence.rows.len())
        .filter(|&i| evidence.rows[i].state != State::Unknown)
        .collect();
    ensure!(
        !kept.is_empty(),
        "No cell was an edge or a certified absence; nothing to fit. \
         Check --detect-z and the input effect sizes."
    );

    let mut states: Vec<State> = kept.iter().map(|&i| evidence.rows[i].state).collect();
    if config.shuffle_control {
        states.shuffle(&mut rng);
        info!("Shuffle control: edge/absent labels permuted before training");
    }

    // ── Compact the three index spaces ───────────────────────────────────
    let mut variant_used = vec![false; variant_names.len()];
    let mut gene_used = vec![false; gene_names.len()];
    let mut context_used = vec![false; evidence.contexts.len()];
    for &i in &kept {
        let row = &evidence.rows[i];
        variant_used[row.variant as usize] = true;
        gene_used[row.gene as usize] = true;
        context_used[row.context as usize] = true;
    }
    let variants = EntityMap::new(variant_names, &variant_used);
    let genes = EntityMap::new(gene_names, &gene_used);
    let contexts = EntityMap::new(&evidence.contexts, &context_used);
    let ubiquitous = contexts.slot[evidence.ubiquitous as usize];

    let (nv, ng, nc) = (variants.len(), genes.len(), contexts.len());
    let h = config.embedding_dim;
    info!("Fitting {nv} variants x {ng} genes x {nc} contexts, H={h}");

    let mut rows: Vec<FittedRow> = kept
        .iter()
        .zip(&states)
        .map(|(&i, &state)| {
            let row = &evidence.rows[i];
            FittedRow {
                variant: variants.slot[row.variant as usize].expect("variant used"),
                gene: genes.slot[row.gene as usize].expect("gene used"),
                context: contexts.slot[row.context as usize].expect("context used"),
                state,
                heldout: false,
            }
        })
        .collect();

    // ── Held-out split: hide real-context edges from training entirely ───
    let is_real = |row: &FittedRow| Some(row.context) != ubiquitous;
    let edge_ids: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].state == State::Edge && is_real(&rows[i]))
        .collect();
    let n_heldout = (config.holdout_frac * edge_ids.len() as f64).round() as usize;
    {
        // A separate stream, so the split does not depend on the model's
        // initialization or on how many batches were drawn.
        let mut split_rng = StdRng::seed_from_u64(config.seed.wrapping_mul(977));
        let mut pool = edge_ids.clone();
        for i in 0..n_heldout.min(pool.len()) {
            let j = split_rng.random_range(i..pool.len());
            pool.swap(i, j);
            rows[pool[i]].heldout = true;
        }
    }
    info!(
        "Held out {} of {} real-context edges ({:.0}%)",
        n_heldout,
        edge_ids.len(),
        100.0 * config.holdout_frac
    );

    // ── Negative pools, from certified absences only ─────────────────────
    let mut absent: [HashMap<(u32, u32), Vec<u32>>; Slot::N] = Default::default();
    for row in rows.iter().filter(|r| r.state == State::CertifiedAbsent) {
        for slot in Slot::ALL {
            absent[slot as usize]
                .entry(slot.key(row))
                .or_default()
                .push(slot.value(row));
        }
    }

    let mut pools: [Vec<u32>; Slot::N] = Default::default();
    let mut positives: Vec<Positive> = Vec::new();
    let mut n_without_negatives = 0usize;
    for row in rows.iter().filter(|r| r.state == State::Edge && !r.heldout) {
        let mut pool = [(0u32, 0u32); Slot::N];
        for slot in Slot::ALL {
            let s = slot as usize;
            let start = pools[s].len() as u32;
            if let Some(found) = absent[s].get(&slot.key(row)) {
                pools[s].extend_from_slice(found);
            }
            pool[s] = (start, pools[s].len() as u32 - start);
        }
        if pool.iter().all(|&(_, len)| len == 0) {
            n_without_negatives += 1;
            continue;
        }
        positives.push(Positive {
            variant: row.variant,
            gene: row.gene,
            context: row.context,
            pool,
        });
    }
    ensure!(
        !positives.is_empty(),
        "No training positive had a certified-absent negative. \
         Every edge would need an absent context or an absent gene of the \
         same variant; try a smaller --detect-z or more input blocks."
    );

    // ── Balance per variant ──────────────────────────────────────────────
    // A ubiquitous variant offers ~30 positives against a specific one's 1.
    // Sampling positives directly lets exactly the class we are trying to
    // distinguish dominate the loss, so a VARIANT is drawn first.
    let mut order: Vec<u32> = (0..positives.len() as u32).collect();
    order.sort_unstable_by_key(|&i| positives[i as usize].variant);
    let mut variant_starts: Vec<(u32, u32)> = Vec::new(); // (start, len) into order
    {
        let mut start = 0usize;
        while start < order.len() {
            let variant = positives[order[start] as usize].variant;
            let mut end = start;
            while end < order.len() && positives[order[end] as usize].variant == variant {
                end += 1;
            }
            variant_starts.push((start as u32, (end - start) as u32));
            start = end;
        }
    }
    info!(
        "{} positives over {} variants ({} edges had no certified-absent negative)",
        positives.len(),
        variant_starts.len(),
        n_without_negatives,
    );

    // ── Parameters ───────────────────────────────────────────────────────
    let normal = Normal::new(0.0f32, 0.1)?;
    let u0: Vec<f32> = (0..nv * h).map(|_| normal.sample(&mut rng)).collect();
    let v0: Vec<f32> = (0..ng * h).map(|_| normal.sample(&mut rng)).collect();
    let mut c0: Vec<f32> = (0..nc * h).map(|_| rng.random_range(0.3f32..0.9)).collect();

    // The ubiquitous gate is all-ones and stays there: it is masked out of
    // the forward pass and restored after every step.
    let mut mask = vec![1.0f32; nc * h];
    let mut ubi_ones = vec![0.0f32; nc * h];
    if let Some(k) = ubiquitous {
        for j in 0..h {
            c0[k as usize * h + j] = 1.0;
            mask[k as usize * h + j] = 0.0;
            ubi_ones[k as usize * h + j] = 1.0;
        }
    }

    let u_var = Var::from_tensor(&Tensor::from_vec(u0, (nv, h), device)?)?;
    let v_var = Var::from_tensor(&Tensor::from_vec(v0, (ng, h), device)?)?;
    let c_var = Var::from_tensor(&Tensor::from_vec(c0, (nc, h), device)?)?;
    let mask = Tensor::from_vec(mask, (nc, h), device)?;
    let ubi_ones = Tensor::from_vec(ubi_ones, (nc, h), device)?;

    let mut opt = AdamW::new(
        vec![u_var.clone(), v_var.clone(), c_var.clone()],
        ParamsAdamW {
            lr: config.learning_rate,
            // Decoupled decay would hit every row every step, which is the
            // dense-penalty failure this model is written around. The ridge
            // lives in the loss, on the batch's rows only.
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;

    // ── Train ────────────────────────────────────────────────────────────
    let nu = config.num_negatives;
    let cand = nu + 1; // positive + negatives; the empty candidate is free
    let batch = config.batch_size;
    let mut final_loss = f32::NAN;
    let zeros_col = Tensor::zeros((batch, 1), DType::F32, device)?;

    let mut vidx = vec![0u32; batch * cand];
    let mut gidx = vec![0u32; batch * cand];
    let mut cidx = vec![0u32; batch * cand];

    for step in 0..config.num_iterations {
        for i in 0..batch {
            let (start, len) = variant_starts[rng.random_range(0..variant_starts.len())];
            let p = positives[order[start as usize + rng.random_range(0..len as usize)] as usize];

            let base = i * cand;
            for j in 0..cand {
                vidx[base + j] = p.variant;
                gidx[base + j] = p.gene;
                cidx[base + j] = p.context;
            }
            // Corrupt ONE slot, roughly half and half, falling back to
            // whichever pool is non-empty.
            let ctx_len = p.pool[Slot::Context as usize].1;
            let gene_len = p.pool[Slot::Gene as usize].1;
            let slot = if gene_len == 0 || (ctx_len > 0 && rng.random_range(0.0f32..1.0) < 0.5) {
                Slot::Context
            } else {
                Slot::Gene
            };
            let (start, len) = p.pool[slot as usize];
            let pool = &pools[slot as usize];
            let target = match slot {
                Slot::Context => &mut cidx,
                Slot::Gene => &mut gidx,
            };
            for j in 1..cand {
                let pick = rng.random_range(0..len as usize);
                target[base + j] = pool[start as usize + pick];
            }
        }

        let vt = Tensor::from_slice(&vidx, vidx.len(), device)?;
        let gt = Tensor::from_slice(&gidx, gidx.len(), device)?;
        let ct = Tensor::from_slice(&cidx, cidx.len(), device)?;

        let c_eff = c_var.as_tensor().mul(&mask)?.add(&ubi_ones)?;
        let uu = u_var.index_select(&vt, 0)?;
        let vv = v_var.index_select(&gt, 0)?;
        let cc = c_eff.index_select(&ct, 0)?;
        let scores = uu.mul(&vv)?.mul(&cc)?.sum(1)?.reshape((batch, cand))?;

        // The empty candidate scores exactly 0 and carries no parameters.
        let logits = Tensor::cat(&[&scores, &zeros_col], 1)?;
        let log_p = ops::log_softmax(&logits, 1)?;
        let mut loss = log_p.narrow(1, 0, 1)?.mean_all()?.neg()?;

        if config.ridge > 0.0 {
            // Only the rows this batch touched are penalized, once each.
            let pen = |t: &Tensor, mut unique: Vec<u32>| -> Result<Tensor> {
                unique.sort_unstable();
                unique.dedup();
                let n = unique.len();
                let idx = Tensor::from_vec(unique, n, device)?;
                Ok(t.index_select(&idx, 0)?.sqr()?.sum_all()?)
            };
            // Corruption never touches the variant slot, so `vidx` repeats
            // each positive's variant `cand` times; dedup the stride instead.
            let variant_ids: Vec<u32> = vidx.iter().step_by(cand).copied().collect();
            let ridge = ((pen(u_var.as_tensor(), variant_ids)?
                + pen(v_var.as_tensor(), gidx.clone())?)?
                + pen(&c_eff, cidx.clone())?)?;
            loss = (loss + (ridge * (0.5 * config.ridge))?)?;
        }

        let grads = loss.backward()?;
        opt.step(&grads)?;
        // Freeze the ubiquitous gate at one. Masking the gradient is not
        // enough on its own under AdamW; the row is written back too.
        c_var.set(&c_var.as_tensor().mul(&mask)?.add(&ubi_ones)?)?;

        let l = loss.to_scalar::<f32>()?;
        final_loss = l;
        if step % 500 == 0 || step + 1 == config.num_iterations {
            info!("  step {:>5}: loss {:.5}", step, l);
        }
    }

    // ── Read the fit back and score ──────────────────────────────────────
    let u = to_dmatrix(u_var.as_tensor(), nv, h)?;
    let v = to_dmatrix(v_var.as_tensor(), ng, h)?;
    let c = to_dmatrix(c_var.as_tensor(), nc, h)?;

    let metrics = evaluate(
        &rows,
        &u,
        &v,
        &c,
        &contexts,
        ubiquitous,
        &positives,
        n_without_negatives,
        variant_starts.len(),
        final_loss,
    );
    report(&metrics);

    Ok(EqtlFit {
        variants,
        genes,
        contexts,
        ubiquitous,
        u,
        v,
        c,
        metrics,
    })
}

fn to_dmatrix(t: &Tensor, rows: usize, cols: usize) -> Result<DMatrix<f32>> {
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(DMatrix::from_row_slice(rows, cols, &flat))
}

// ── Diagnostics ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn evaluate(
    rows: &[FittedRow],
    u: &DMatrix<f32>,
    v: &DMatrix<f32>,
    c: &DMatrix<f32>,
    contexts: &EntityMap,
    ubiquitous: Option<u32>,
    positives: &[Positive],
    n_without_negatives: usize,
    n_variants_with_positives: usize,
    final_loss: f32,
) -> FitMetrics {
    let h = u.ncols();
    let score = |row: &FittedRow| -> f32 {
        (0..h)
            .map(|k| {
                u[(row.variant as usize, k)]
                    * v[(row.gene as usize, k)]
                    * c[(row.context as usize, k)]
            })
            .sum()
    };

    let mut train_edges = Vec::new();
    let mut heldout_edges = Vec::new();
    let mut absent = Vec::new();
    let mut n_edges_ubiquitous = 0usize;
    for row in rows {
        if Some(row.context) == ubiquitous {
            if row.state == State::Edge {
                n_edges_ubiquitous += 1;
            }
            continue;
        }
        let s = score(row) as f64;
        match (row.state, row.heldout) {
            (State::Edge, false) => train_edges.push(s),
            (State::Edge, true) => heldout_edges.push(s),
            (State::CertifiedAbsent, _) => absent.push(s),
            _ => {}
        }
    }

    let auc_in_sample = auc(&train_edges, &absent);
    let auc_heldout = auc(&heldout_edges, &absent);
    let auprc_heldout = auprc(&heldout_edges, &absent);
    let auprc_base =
        heldout_edges.len() as f64 / (heldout_edges.len() + absent.len()).max(1) as f64;

    // The gate geometry is read over real contexts only: the ubiquitous row
    // is a constant, so including it would inflate every spread.
    let real: Vec<usize> = (0..contexts.len())
        .filter(|&k| Some(k as u32) != ubiquitous)
        .collect();
    let gate = DMatrix::from_fn(real.len(), h, |i, j| c[(real[i], j)]);
    let names: Vec<&str> = real.iter().map(|&k| contexts.names[k].as_ref()).collect();
    let (within, between) = group_cosines(&names, &gate).unzip();

    FitMetrics {
        embedding_dim: h,
        n_variants: u.nrows(),
        n_genes: v.nrows(),
        n_contexts: contexts.len(),
        n_edges_real: train_edges.len() + heldout_edges.len(),
        n_edges_ubiquitous,
        n_absent_real: absent.len(),
        n_heldout: heldout_edges.len(),
        n_positives: positives.len(),
        n_positives_without_negatives: n_without_negatives,
        n_variants_with_positives,
        auc_in_sample,
        auc_heldout,
        auprc_heldout,
        auprc_base,
        auprc_lift: auprc_heldout / auprc_base.max(f64::MIN_POSITIVE),
        generalization_gap: auc_in_sample - auc_heldout,
        effective_rank: effective_rank(&gate),
        within_group_cosine: within,
        between_group_cosine: between,
        mean_score_train_edge: mean(&train_edges),
        mean_score_heldout_edge: mean(&heldout_edges),
        mean_score_absent: mean(&absent),
        final_loss,
    }
}

fn report(m: &FitMetrics) {
    info!(
        "IN-SAMPLE edge vs certified-absent: AUC {:.3} (n {} / {})",
        m.auc_in_sample,
        m.n_edges_real - m.n_heldout,
        m.n_absent_real,
    );
    info!(
        "HELD-OUT  edge vs certified-absent: AUC {:.3}, AUPRC {:.3} (base {:.3}, lift {:.1}x, n {})",
        m.auc_heldout, m.auprc_heldout, m.auprc_base, m.auprc_lift, m.n_heldout,
    );
    info!("generalization gap: {:.3}", m.generalization_gap);
    info!(
        "cell-type gate: effective rank {:.2} of {} programs",
        m.effective_rank, m.embedding_dim,
    );
    match (m.within_group_cosine, m.between_group_cosine) {
        (Some(w), Some(b)) => info!(
            "context similarity: within-group {:.3} vs between {:.3}  [want within > between]",
            w, b
        ),
        _ => info!("context similarity: skipped, labels carry no group prefix"),
    }
    info!(
        "mean score: train-edges {:+.3} | held-out edges {:+.3} | absent {:+.3} | empty 0",
        m.mean_score_train_edge, m.mean_score_heldout_edge, m.mean_score_absent,
    );
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        f64::NAN
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Mann-Whitney AUC with average ranks for ties.
pub fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut all: Vec<(f64, bool)> = pos
        .iter()
        .map(|&s| (s, true))
        .chain(neg.iter().map(|&s| (s, false)))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        // Ranks are 1-based; a tied block shares its average rank.
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        rank_sum += all[i..=j].iter().filter(|(_, p)| *p).count() as f64 * avg;
        i = j + 1;
    }
    let n_pos = pos.len() as f64;
    (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * neg.len() as f64)
}

/// Area under the precision-recall curve, positives first on ties.
pub fn auprc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut all: Vec<(f64, bool)> = pos
        .iter()
        .map(|&s| (s, true))
        .chain(neg.iter().map(|&s| (s, false)))
        .collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut hits = 0usize;
    let mut sum = 0.0f64;
    for (seen, (_, is_pos)) in all.iter().enumerate() {
        if *is_pos {
            hits += 1;
            sum += hits as f64 / (seen + 1) as f64;
        }
    }
    sum / pos.len() as f64
}

/// Participation ratio of the gate matrix's principal variances: how many
/// programs the cell types actually spread over.
pub fn effective_rank(gate: &DMatrix<f32>) -> f64 {
    if gate.nrows() < 2 {
        return f64::NAN;
    }
    let mut centered = DMatrix::<f64>::zeros(gate.nrows(), gate.ncols());
    for j in 0..gate.ncols() {
        let mean =
            (0..gate.nrows()).map(|i| gate[(i, j)] as f64).sum::<f64>() / gate.nrows() as f64;
        for i in 0..gate.nrows() {
            centered[(i, j)] = gate[(i, j)] as f64 - mean;
        }
    }
    let sv = centered.singular_values();
    let var: Vec<f64> = sv.iter().map(|s| s * s).collect();
    let total: f64 = var.iter().sum();
    if total <= 0.0 {
        return f64::NAN;
    }
    let sum_sq: f64 = var.iter().map(|v| (v / total).powi(2)).sum();
    1.0 / sum_sq
}

/// Group of a context name, taken from the label itself.
///
/// The prefix before the first `-` or `_` is the group, so `L1-0` and `L1-3`
/// share one and `L2-0` does not. Deriving it from the DATA is deliberate: a
/// curated table of real label names would hard-code one dataset's annotation
/// scheme into library code. A label with no delimiter has no group and is
/// left out of the within-group statistic rather than guessed into a bucket.
fn group(name: &str) -> Option<&str> {
    let cut = name.find(['-', '_'])?;
    (cut > 0).then(|| &name[..cut])
}

/// Mean cosine similarity within and between label groups, or `None` when
/// too few labels carry a group prefix to make the contrast.
pub fn group_cosines(names: &[&str], gate: &DMatrix<f32>) -> Option<(f64, f64)> {
    let known: Vec<(usize, &str)> = names
        .iter()
        .enumerate()
        .filter_map(|(i, n)| group(n).map(|g| (i, g)))
        .collect();
    if known.len() < 3 {
        return None;
    }

    let unit: Vec<Vec<f64>> = known
        .iter()
        .map(|&(i, _)| {
            let row: Vec<f64> = (0..gate.ncols()).map(|j| gate[(i, j)] as f64).collect();
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                row.iter().map(|x| x / norm).collect()
            } else {
                row
            }
        })
        .collect();

    let (mut w_sum, mut w_n, mut b_sum, mut b_n) = (0.0f64, 0usize, 0.0f64, 0usize);
    for a in 0..known.len() {
        for b in (a + 1)..known.len() {
            let cos: f64 = unit[a].iter().zip(&unit[b]).map(|(x, y)| x * y).sum();
            if known[a].1 == known[b].1 {
                w_sum += cos;
                w_n += 1;
            } else {
                b_sum += cos;
                b_n += 1;
            }
        }
    }
    (w_n > 0 && b_n > 0).then(|| (w_sum / w_n as f64, b_sum / b_n as f64))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
