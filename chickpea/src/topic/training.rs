use crate::coarsening::coarsen_tensor;
use crate::common::*;
use crate::linkage::{precompute_expand_indices, rna_dictionary_from_m};
use crate::topic::decoder::DecoderArgs;
use crate::topic::{ChickpeaDecoder, ChickpeaEncoder, SuSiE};
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{self, AdamW, Optimizer, ParamsAdamW, VarMap};
use data_beans_alg::collapse_data::CollapsedOut;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use std::sync::atomic::{AtomicBool, Ordering};

/* ---- Context and params ---- */

pub struct TrainingContext<'a> {
    pub collapsed: &'a [Vec<CollapsedOut>],
    pub rna_coarsenings: &'a [Option<FeatureCoarsening>],
    pub atac_coarsenings: &'a [Option<FeatureCoarsening>],
    pub cis_mask: &'a Tensor,
    pub flat_cis_indices: &'a Tensor,
    pub n_genes: usize,
    pub n_peaks: usize,
    pub c_max: usize,
    pub dev: &'a Device,
}

pub struct TrainingParams {
    pub n_topics: usize,
    pub n_ser_components: usize,
    pub prior_var: f64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub minibatch_size: usize,
    pub jitter_interval: usize,
    pub topic_smoothing: f64,
    pub gate_prior: f64,
    pub row_budget: usize,
    pub sort_dim: usize,
    pub embedding_dim: usize,
    pub context_size: usize,
}

pub struct TrainedModel {
    pub decoders: Vec<ChickpeaDecoder>,
    pub susies: Vec<SuSiE>,
    pub encoder: ChickpeaEncoder,
    pub enc_expand_indices: Option<Tensor>,
    pub level_dims: Vec<(usize, usize)>,
}

/* ---- Per-modality paired tensors ---- */

/// Paired tensors for one modality at one level: observed, batch, target, coarsened.
struct ModalityData {
    observed: Tensor,  // [N, D] full-resolution
    coarsened: Tensor, // [N, d] coarsened for decoder
    batch_residual: Option<Tensor>,
    coarsened_target: Option<Tensor>, // batch-corrected, coarsened
}

impl ModalityData {
    fn from_collapsed(
        co: &CollapsedOut,
        sample: bool,
        fc: Option<&FeatureCoarsening>,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let observed = gamma_to_tensor(&co.mu_observed, sample, dev)?;
        let coarsened = coarsen_tensor(&observed, fc)?;
        let batch_residual = co
            .mu_residual
            .as_ref()
            .map(|r| gamma_to_tensor(r, sample, dev))
            .transpose()?;
        let coarsened_target = co
            .mu_adjusted
            .as_ref()
            .map(|a| {
                let t = gamma_to_tensor(a, sample, dev)?;
                coarsen_tensor(&t, fc).map_err(|e| anyhow::anyhow!(e))
            })
            .transpose()?;
        Ok(Self {
            observed,
            coarsened,
            batch_residual,
            coarsened_target,
        })
    }

    fn subsample(&self, idx: &Tensor, fc: Option<&FeatureCoarsening>) -> anyhow::Result<Self> {
        let observed = self.observed.index_select(idx, 0)?;
        let coarsened = coarsen_tensor(&observed, fc)?;
        let batch_residual = self
            .batch_residual
            .as_ref()
            .map(|t| t.index_select(idx, 0))
            .transpose()?;
        let coarsened_target = self
            .coarsened_target
            .as_ref()
            .map(|t| t.index_select(idx, 0))
            .transpose()?;
        Ok(Self {
            observed,
            coarsened,
            batch_residual,
            coarsened_target,
        })
    }

    fn recoarsen(&self, fc: Option<&FeatureCoarsening>) -> anyhow::Result<Self> {
        Ok(Self {
            observed: self.observed.clone(),
            coarsened: coarsen_tensor(&self.observed, fc)?,
            batch_residual: self.batch_residual.clone(),
            coarsened_target: self.coarsened_target.clone(),
        })
    }

    fn num_samples(&self) -> candle_util::candle_core::Result<usize> {
        self.observed.dim(0)
    }
}

/* ---- Training ---- */

pub fn train(ctx: &TrainingContext, params: &TrainingParams) -> anyhow::Result<TrainedModel> {
    let num_levels = ctx.collapsed.len();
    let dev = ctx.dev;

    /* Level dimensions */
    let level_dims: Vec<(usize, usize)> = (0..num_levels)
        .map(|i| {
            let dg = ctx.rna_coarsenings[i]
                .as_ref()
                .map(|c| c.num_coarse)
                .unwrap_or(ctx.n_genes);
            let dp = ctx.atac_coarsenings[i]
                .as_ref()
                .map(|c| c.num_coarse)
                .unwrap_or(ctx.n_peaks);
            (dg, dp)
        })
        .collect();
    for (i, &(dg, dp)) in level_dims.iter().enumerate() {
        info!("Level {} dims: {} genes, {} peaks", i, dg, dp);
    }

    /* Model */
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, dev);

    let decoders: Vec<ChickpeaDecoder> = level_dims
        .iter()
        .enumerate()
        .map(|(i, &(dg, dp))| {
            ChickpeaDecoder::new(
                DecoderArgs {
                    n_features_atac: dp,
                    n_features_rna: dg,
                    n_topics: params.n_topics,
                },
                vs.pp(format!("dec_{i}")),
            )
            .expect("decoder creation")
        })
        .collect();

    let susies: Vec<SuSiE> = level_dims
        .iter()
        .enumerate()
        .map(|(i, &(dg, dp))| {
            let (n_cands, mask) =
                if ctx.rna_coarsenings[i].is_none() && ctx.atac_coarsenings[i].is_none() {
                    (ctx.c_max, Some(ctx.cis_mask.clone()))
                } else {
                    (dp, None)
                };
            SuSiE::new(
                dg,
                n_cands,
                params.n_ser_components,
                mask,
                vs.pp(format!("susie_{i}")),
            )
            .expect("SuSiE creation")
        })
        .collect();

    let finest_gene_members: Option<Vec<usize>> = ctx
        .rna_coarsenings
        .last()
        .and_then(|c| c.as_ref())
        .map(|c| c.fine_to_coarse.clone());
    let finest_peak_members: Option<Vec<usize>> = ctx
        .atac_coarsenings
        .last()
        .and_then(|c| c.as_ref())
        .map(|c| c.fine_to_coarse.clone());

    let encoder = ChickpeaEncoder::new(
        ctx.n_genes,
        ctx.n_peaks,
        params.n_topics,
        params.embedding_dim,
        params.context_size,
        &[128],
        vs.pp("encoder"),
    )?;

    let mut adam = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: params.learning_rate,
            ..Default::default()
        },
    )?;

    info!(
        "Model: {} topics, {} levels, {} params",
        params.n_topics,
        num_levels,
        varmap.all_vars().len(),
    );

    /* Ctrl-C handler */
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            info!("Interrupt — saving results...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("signal handler");
    }

    let enc_expand_indices: Option<Tensor> = match (&finest_gene_members, &finest_peak_members) {
        (Some(gm), Some(pm)) => Some(precompute_expand_indices(
            gm,
            pm,
            ctx.flat_cis_indices,
            ctx.n_genes,
            ctx.c_max,
            dev,
        )?),
        _ => None,
    };

    /* Progress */
    let n_epochs = params.epochs;
    let jitter = params.jitter_interval.max(1);
    let verbose = log::log_enabled!(log::Level::Info);
    info!("Training: {} epochs, jitter={}", n_epochs, jitter);

    let pb = if verbose {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(n_epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{bar:40.green/dim} {pos}/{len} [{elapsed}<{eta}] {msg}")
                .unwrap(),
        );
        pb
    };

    /* Row budgets: default 2^sort_dim per level */
    let level_sizes: Vec<usize> = ctx
        .collapsed
        .iter()
        .map(|ld| ld[0].mu_observed.ncols())
        .collect();
    let budget = if params.row_budget > 0 {
        params.row_budget
    } else {
        1usize << params.sort_dim
    };
    let level_budgets: Vec<usize> = level_sizes.iter().map(|&sz| sz.min(budget)).collect();
    for (i, (&sz, &b)) in level_sizes.iter().zip(&level_budgets).enumerate() {
        info!("Level {}: {}/{} samples", i, b, sz);
    }

    /* ===================== TRAINING LOOP ===================== */

    let mut epoch_counter = 0usize;
    let mut rng = rand::rng();

    'outer: for jitter_start in (0..n_epochs).step_by(jitter) {
        /* Sample RNA from posterior (jittered); ATAC uses posterior mean (stable) */
        let sampled: Vec<(ModalityData, ModalityData)> = ctx
            .collapsed
            .iter()
            .enumerate()
            .map(|(i, ld)| {
                let rna_fc = ctx.rna_coarsenings[i].as_ref();
                let atac_fc = ctx.atac_coarsenings[i].as_ref();
                Ok((
                    ModalityData::from_collapsed(&ld[0], true, rna_fc, dev)?,
                    ModalityData::from_collapsed(&ld[1], false, atac_fc, dev)?,
                ))
            })
            .collect::<anyhow::Result<_>>()?;

        let jitter_end = jitter.min(n_epochs - jitter_start);

        for _j in 0..jitter_end {
            /* Fresh subsample each epoch */
            let epoch_data: Vec<(ModalityData, ModalityData)> = sampled
                .iter()
                .zip(level_budgets.iter())
                .enumerate()
                .map(|(i, ((rna, atac), &b))| {
                    let ns = rna.num_samples()?;
                    let rna_fc = ctx.rna_coarsenings[i].as_ref();
                    let atac_fc = ctx.atac_coarsenings[i].as_ref();
                    if b < ns {
                        let idx: Vec<u32> = rand::seq::index::sample(&mut rng, ns, b)
                            .into_vec()
                            .into_iter()
                            .map(|i| i as u32)
                            .collect();
                        let idx_t = Tensor::from_vec(idx, b, dev)?;
                        Ok((
                            rna.subsample(&idx_t, rna_fc)?,
                            atac.subsample(&idx_t, atac_fc)?,
                        ))
                    } else {
                        Ok((rna.recoarsen(rna_fc)?, atac.recoarsen(atac_fc)?))
                    }
                })
                .collect::<anyhow::Result<_>>()?;

            /* Encoder weights from finest SuSiE */
            let finest_m = susies.last().unwrap().forward()?;
            let enc_m_weights = match &enc_expand_indices {
                Some(idx) => finest_m
                    .flatten_all()?
                    .index_select(idx, 0)?
                    .reshape((ctx.n_genes, ctx.c_max))?
                    .exp()?,
                None => finest_m.exp()?,
            };

            let (mut la, mut lr, mut ke, mut ks, mut n_tot) = (0f32, 0f32, 0f32, 0f32, 0usize);

            /* Per-level training */
            for (i, (rna, atac)) in epoch_data.iter().enumerate() {
                let dec = &decoders[i];
                let susie = &susies[i];
                let ns = rna.num_samples()?;
                let mb = params.minibatch_size.min(ns);

                /* Shuffle */
                let perm: Vec<u32> = {
                    let mut v: Vec<u32> = (0..ns as u32).collect();
                    v.shuffle(&mut rng);
                    v
                };
                let perm_t = Tensor::from_vec(perm, ns, dev)?;
                let rna_fc = ctx.rna_coarsenings[i].as_ref();
                let atac_fc = ctx.atac_coarsenings[i].as_ref();
                let s_rna = rna.subsample(&perm_t, rna_fc)?;
                let s_atac = atac.subsample(&perm_t, atac_fc)?;

                let m_gc = susie.forward()?;
                let log_w_linked = if rna_fc.is_some() || atac_fc.is_some() {
                    m_gc.exp()?.matmul(&dec.log_beta_atac.exp()?)?.log()?
                } else {
                    rna_dictionary_from_m(&m_gc, &dec.log_beta_atac, ctx.flat_cis_indices)?.log()?
                };
                let kl_susie = susie.kl(params.prior_var, params.gate_prior)?;

                /* Minibatch loop */
                for b in 0..n_batches(ns, mb) {
                    let (start, len) = mb_range(b, mb, ns);

                    let mb_rna = s_rna.observed.narrow(0, start, len)?;
                    let mb_atac = s_atac.observed.narrow(0, start, len)?;
                    let mb_batch_rna = s_rna
                        .batch_residual
                        .as_ref()
                        .map(|t| t.narrow(0, start, len))
                        .transpose()?;
                    let mb_batch_atac = s_atac
                        .batch_residual
                        .as_ref()
                        .map(|t| t.narrow(0, start, len))
                        .transpose()?;

                    let enc_inp = crate::topic::encoder::EncoderInput {
                        x_rna: &mb_rna,
                        x_atac: &mb_atac,
                        batch_rna: mb_batch_rna.as_ref(),
                        batch_atac: mb_batch_atac.as_ref(),
                        m_weights: &enc_m_weights,
                        flat_cis_indices: ctx.flat_cis_indices,
                        c_max: ctx.c_max,
                    };
                    let (log_z, kl_enc) = encoder.forward(&enc_inp, true)?;

                    let log_z = if params.topic_smoothing > 0.0 {
                        let kk = log_z.dim(1)? as f64;
                        ((log_z.exp()? * (1.0 - params.topic_smoothing))?
                            + params.topic_smoothing / kk)?
                            .log()?
                    } else {
                        log_z
                    };

                    let rna_target = match &s_rna.coarsened_target {
                        Some(t) => t.narrow(0, start, len)?,
                        None => s_rna.coarsened.narrow(0, start, len)?,
                    };
                    let atac_target = match &s_atac.coarsened_target {
                        Some(t) => t.narrow(0, start, len)?,
                        None => s_atac.coarsened.narrow(0, start, len)?,
                    };

                    let llik_atac = dec.forward_atac(&log_z, &atac_target)?;
                    let llik_rna = dec.forward_rna(&log_z, &rna_target, &log_w_linked)?;

                    let nb = len as f64;
                    let elbo = (&kl_enc - &llik_atac - &llik_rna)?.sum_all()?;
                    let loss = ((elbo + &kl_susie * (nb / ns as f64))? / nb)?;
                    adam.backward_step(&loss)?;

                    la += llik_atac.sum_all()?.to_scalar::<f32>()?;
                    lr += llik_rna.sum_all()?.to_scalar::<f32>()?;
                    ke += kl_enc.sum_all()?.to_scalar::<f32>()?;
                }
                ks += kl_susie.to_scalar::<f32>()?;
                n_tot += ns;
            }

            let ns = n_tot as f32;
            let msg = format!(
                "rna={:.1} atac={:.1} kl_e={:.1} kl_s={:.1}",
                lr / ns,
                la / ns,
                ke / ns,
                ks / ns,
            );
            if verbose {
                info!("  {}/{}: {}", epoch_counter + 1, n_epochs, msg);
            } else {
                pb.set_message(msg);
                pb.inc(1);
            }

            epoch_counter += 1;
            if stop.load(Ordering::SeqCst) {
                info!("Stopping early at epoch {}", epoch_counter);
                break 'outer;
            }
        }
    }
    pb.finish_and_clear();

    Ok(TrainedModel {
        decoders,
        susies,
        encoder,
        enc_expand_indices,
        level_dims,
    })
}

/* ---- Helpers ---- */

fn n_batches(n: usize, mb: usize) -> usize {
    n.div_ceil(mb)
}

fn mb_range(b: usize, mb: usize, n: usize) -> (usize, usize) {
    let start = b * mb;
    (start, (start + mb).min(n) - start)
}

pub fn gamma_to_tensor(
    gamma: &matrix_param::dmatrix_gamma::GammaMatrix,
    sample: bool,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let mat = if sample {
        gamma.posterior_sample()?
    } else {
        gamma.posterior_mean().clone()
    };
    Ok(mat.transpose().to_tensor(dev)?.contiguous()?)
}
