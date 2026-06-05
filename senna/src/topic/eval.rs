use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::topic_refinement::*;
use candle_util::traits::*;

/// Configuration for latent evaluation by encoder
pub(crate) struct EvaluateLatentConfig<'a, Dec> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub feature_coarsening: Option<&'a FeatureCoarsening>,
    pub decoder: Option<&'a Dec>,
    pub refine_config: Option<&'a TopicRefinementConfig>,
}

pub(crate) fn evaluate_latent_by_encoder<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    // Delta coarsened to D_coarse — encoder operates at D_coarse
    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| x.posterior_mean().clone())
    .map(|mut delta_db| {
        if let Some(fc) = config.feature_coarsening {
            delta_db = fc.aggregate_rows_ds(&delta_db);
        }
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        delta: delta.as_ref(),
        feature_coarsening: config.feature_coarsening,
        decoder: config.decoder,
        refine_config: config.refine_config,
        adj_method: config.adj_method.clone(),
        gene_remap: None,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_block(block, data_vec, encoder, &block_config)
    })
}

/// Mapping from new-data row indices to training gene positions.
pub(crate) struct GeneRemap {
    /// For each new-data compact row, the training gene position (or None).
    pub new_to_train: Vec<Option<usize>>,
    /// Number of training genes (`D_train`).
    pub d_train: usize,
    /// Number of new genes that mapped to training genes.
    pub n_mapped: usize,
}

/// Optional query (held-out) row-name transforms applied *before*
/// matching against the training dictionary.
///
/// Order per query name: (1) if `suffix_delim` is set, split once into
/// `(base, suffix)` and — when `keep_suffix` is set — drop the row unless
/// its `suffix` equals `keep_suffix`; (2) canonicalize `base` with `kind`
/// (e.g. `Gene { delim: '_' }` → bare symbol); (3) resolve as usual.
///
/// Defaults (`kind = Exact`, no delimiter, no filter) reproduce the legacy
/// exact-then-flexible behavior — `FeatureNameKind`'s own derived default is
/// `Exact`, so the whole struct derives `Default`.
#[derive(Default)]
pub(crate) struct QueryNameOpts {
    pub kind: auxiliary_data::feature_names::FeatureNameKind,
    pub suffix_delim: Option<char>,
    pub keep_suffix: Option<Box<str>>,
}

/// Build a gene remap from training gene names and new-data gene names.
///
/// Tries case-insensitive exact match first; falls back to
/// `flexible_gene_match` (handles aliases like `ENSG..._CD8A` ↔ `CD8A`,
/// case differences, and `_`-delimited prefixes/suffixes).
pub(crate) fn build_gene_remap(
    training_genes: &[Box<str>],
    new_data_genes: &[Box<str>],
) -> GeneRemap {
    build_gene_remap_with(training_genes, new_data_genes, &QueryNameOpts::default())
}

/// Like [`build_gene_remap`] but applies [`QueryNameOpts`] (modality-suffix
/// filter, base-key trim, and name-kind canonicalization) to each query row
/// name before resolution. Multiple query rows may resolve to the same
/// training gene (many-to-one); the scatter sites accumulate them.
pub(crate) fn build_gene_remap_with(
    training_genes: &[Box<str>],
    new_data_genes: &[Box<str>],
    opts: &QueryNameOpts,
) -> GeneRemap {
    use crate::marker_support::flexible_gene_match;

    // Lowercased exact-match index — fast path for matching name sets.
    let train_pos: rustc_hash::FxHashMap<String, usize> = training_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.to_lowercase(), i))
        .collect();

    let mut n_exact = 0usize;
    let mut n_flexible = 0usize;
    let mut n_dropped = 0usize;
    let new_to_train: Vec<Option<usize>> = new_data_genes
        .iter()
        .map(|g| {
            // (1) suffix split + modality filter
            let base: &str = match opts.suffix_delim {
                Some(d) => match g.split_once(d) {
                    Some((base, suffix)) => {
                        if let Some(keep) = opts.keep_suffix.as_deref() {
                            if suffix != keep {
                                n_dropped += 1;
                                return None;
                            }
                        }
                        base
                    }
                    // no delimiter present: if a suffix filter is in
                    // force, the row has no qualifying suffix → drop.
                    None => {
                        if opts.keep_suffix.is_some() {
                            n_dropped += 1;
                            return None;
                        }
                        g
                    }
                },
                None => g,
            };

            // (2) name-kind canonicalization (Gene → bare symbol, etc.)
            let key = opts.kind.canonicalize(base);

            // (3) resolve: lowercased exact, then flexible fallback
            if let Some(&i) = train_pos.get(&key.to_lowercase()) {
                n_exact += 1;
                Some(i)
            } else if let Some(i) = training_genes
                .iter()
                .position(|t| flexible_gene_match(&key, t))
            {
                n_flexible += 1;
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let n_mapped = n_exact + n_flexible;
    log::info!(
        "Gene alignment: {n_mapped}/{} new genes mapped to {}/{} training genes \
         ({n_exact} exact, {n_flexible} flexible, {n_dropped} dropped by suffix filter)",
        new_data_genes.len(),
        n_mapped,
        training_genes.len()
    );

    GeneRemap {
        new_to_train,
        d_train: training_genes.len(),
        n_mapped,
    }
}

/// Evaluate latent states with optional gene remapping and pre-computed delta.
///
/// When `gene_remap` is `Some`, per-block CSC data is scattered from new-data
/// row order to training gene order. When `None`, data is used as-is.
pub(crate) fn evaluate_latent_with_gene_remap<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    delta_db: Option<&Mat>,
    gene_remap: Option<&GeneRemap>,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    let delta = delta_db.map(|db| {
        let mut db = db.clone();
        if let Some(fc) = config.feature_coarsening {
            db = fc.aggregate_rows_ds(&db);
        }
        db.to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        delta: delta.as_ref(),
        feature_coarsening: config.feature_coarsening,
        decoder: config.decoder,
        refine_config: config.refine_config,
        adj_method: config.adj_method.clone(),
        gene_remap,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_block(block, data_vec, encoder, &block_config)
    })
}

/// Scatter CSC rows from new-data order to training gene order.
fn remap_csc_to_dense(csc: &nalgebra_sparse::CscMatrix<f32>, remap: &GeneRemap) -> Mat {
    let ncols = csc.ncols();
    let mut out = Mat::zeros(remap.d_train, ncols);
    for j in 0..ncols {
        let col = csc.col(j);
        for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
            if let Some(row_train) = remap.new_to_train[row_new] {
                out[(row_train, j)] += val;
            }
        }
    }
    out
}

/// Configuration for block-wise evaluation
struct EvaluateBlockConfig<'a, Dec> {
    dev: &'a Device,
    delta: Option<&'a Tensor>,
    feature_coarsening: Option<&'a FeatureCoarsening>,
    decoder: Option<&'a Dec>,
    refine_config: Option<&'a TopicRefinementConfig>,
    adj_method: AdjMethod,
    gene_remap: Option<&'a GeneRemap>,
}

fn evaluate_block<Enc, Dec>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &Enc,
    config: &EvaluateBlockConfig<Dec>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = config
        .delta
        .map(|delta_bm| {
            expand_delta_for_block(data_vec, delta_bm, &config.adj_method, lb, ub, config.dev)
        })
        .transpose()?;

    let x_dn_csc = data_vec.read_columns_csc(lb..ub)?;

    let x_enc_nd = if let Some(remap) = config.gene_remap {
        let x_dn_train = remap_csc_to_dense(&x_dn_csc, remap);
        if let Some(fc) = config.feature_coarsening {
            fc.aggregate_rows_ds(&x_dn_train)
                .to_tensor(config.dev)?
                .transpose(0, 1)?
        } else {
            x_dn_train.to_tensor(config.dev)?.transpose(0, 1)?
        }
    } else if let Some(fc) = config.feature_coarsening {
        fc.aggregate_sparse_csc(&x_dn_csc)
            .to_tensor(config.dev)?
            .transpose(0, 1)?
    } else {
        x_dn_csc.to_tensor(config.dev)?.transpose(0, 1)?
    };

    let (log_z_nk, _) = encoder.forward_t(&x_enc_nd, x0_nd.as_ref(), false)?;

    // Apply per-cell refinement (data already at D_coarse)
    let log_z_nk = if let (Some(dec), Some(cfg)) = (config.decoder, config.refine_config) {
        refine_topic_proportions(&log_z_nk, &x_enc_nd, dec, cfg)?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use auxiliary_data::feature_names::FeatureNameKind;

    fn names(xs: &[&str]) -> Vec<Box<str>> {
        xs.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn spliced_filter_trim_and_gene_alias() {
        let training = names(&["TSPAN6", "A1BG"]);
        let query = names(&[
            "ENSG00000000003_TSPAN6/count/spliced",
            "ENSG00000000003_TSPAN6/count/unspliced",
            "ENSGX_A1BG/count/spliced",
            "ENSGY_NOTFOUND/count/spliced",
        ]);
        let opts = QueryNameOpts {
            kind: FeatureNameKind::Gene { delim: '_' },
            suffix_delim: Some('/'),
            keep_suffix: Some("count/spliced".into()),
        };
        let remap = build_gene_remap_with(&training, &query, &opts);
        // spliced TSPAN6 → training row 0
        assert_eq!(remap.new_to_train[0], Some(0));
        // unspliced TSPAN6 → dropped by suffix filter
        assert_eq!(remap.new_to_train[1], None);
        // spliced A1BG → training row 1
        assert_eq!(remap.new_to_train[2], Some(1));
        // spliced gene absent from dictionary → unmapped
        assert_eq!(remap.new_to_train[3], None);
        assert_eq!(remap.n_mapped, 2);
        assert_eq!(remap.d_train, 2);
    }

    #[test]
    fn many_to_one_sums_in_dense_scatter() {
        // Two query rows resolve to the same training gene; remap_csc_to_dense
        // must accumulate, not overwrite.
        let training = names(&["TSPAN6"]);
        let query = names(&["AAA_TSPAN6", "BBB_TSPAN6"]);
        let opts = QueryNameOpts {
            kind: FeatureNameKind::Gene { delim: '_' },
            suffix_delim: None,
            keep_suffix: None,
        };
        let remap = build_gene_remap_with(&training, &query, &opts);
        assert_eq!(remap.new_to_train[0], Some(0));
        assert_eq!(remap.new_to_train[1], Some(0));

        // 2 query rows × 1 cell, values 3 and 4 → training row 0 should hold 7.
        let coo = nalgebra_sparse::CooMatrix::try_from_triplets(
            2,
            1,
            vec![0, 1],
            vec![0, 0],
            vec![3.0f32, 4.0f32],
        )
        .unwrap();
        let csc = nalgebra_sparse::CscMatrix::from(&coo);
        let dense = remap_csc_to_dense(&csc, &remap);
        assert_eq!(dense[(0, 0)], 7.0);
    }

    #[test]
    fn default_opts_preserve_exact_match() {
        let training = names(&["FOO", "BAR"]);
        let query = names(&["BAR", "FOO"]);
        let remap = build_gene_remap(&training, &query);
        assert_eq!(remap.new_to_train[0], Some(1));
        assert_eq!(remap.new_to_train[1], Some(0));
    }
}
