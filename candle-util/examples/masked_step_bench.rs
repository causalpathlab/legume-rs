//! Where does a masked-topic training step spend its time?
//!
//! Times each piece of the step in isolation, forward and backward, on the
//! device given by `--device` (`cpu` or `cuda`), at the shapes the sim uses:
//! N rows, K context slots, D genes, H embedding, r projection rank, T topics.
//!
//! The training step is the DENSE read: scores over every gene, a softmax over
//! the gene axis, one gemm to pool. Lines marked `legacy:` are the slot-window
//! path, which now only scores OLD models recorded with a context size — they
//! stay so the before/after remains legible, not because a step runs them. Run
//! with
//! `cargo run --release --features cuda --target-dir target-cuda -p candle-util --example masked_step_bench -- cuda`.

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use candle_util::decoder::masked_etm::{EmbeddedNbTopicDecoder, MaskedDenseTarget};
use candle_util::encoder::dense_pool::{attention_scores_dense, pool_dense};
use candle_util::encoder::scatter_pool::{
    attention_scores_from_vector, pool_by_scatter, query_over_features,
};
use candle_util::fast_index::gather_rows;
use candle_util::fast_index::scatter_add_cols;
use std::time::Instant;

const N: usize = 100;
const K: usize = 512;
const D: usize = 4000;
const H: usize = 128;
const R: usize = 32;
const T: usize = 8;
const ITERS: usize = 30;

/// `[N, D]` scored-position mask: 1 everywhere the windowed encoder could not
/// see. Local to this bench now — the trainer's window is gone, and with it the
/// scatter this measured; the number is kept so the before/after stays legible.
fn target_mask_nd(
    indices: &Tensor,
    visible: &Tensor,
    n_features: usize,
) -> candle_core::Result<Tensor> {
    let n = indices.dim(0)?;
    let zeros = Tensor::zeros((n, n_features), visible.dtype(), visible.device())?;
    zeros.scatter_add(indices, visible, 1)?.affine(-1.0, 1.0)
}

fn time<F: FnMut() -> candle_core::Result<()>>(dev: &Device, label: &str, mut f: F) {
    // warm-up
    f().unwrap();
    dev.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f().unwrap();
    }
    dev.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    println!("{label:<48} {ms:8.2} ms/step");
}

fn main() -> anyhow::Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "cpu".into());
    let dev = if which == "cuda" {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("device {which}; N={N} K={K} D={D} H={H} r={R} T={T}; {ITERS} iterations");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let rho = vb.get_with_hints((D, H), "rho", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
    let features = candle_util::feature_embedding::FeatureEmbedding::fixed(rho.clone());
    let dec = EmbeddedNbTopicDecoder::new(T, std::sync::Arc::clone(&features), vb.pp("dec"))?;
    let log_theta = Var::from_tensor(&candle_nn::ops::log_softmax(
        &Tensor::randn(0f32, 1.0, (N, T), &dev)?,
        1,
    )?)?;

    // Loader-shaped inputs.
    let idx_host: Vec<u32> = (0..N * K).map(|i| ((i * 7919) % D) as u32).collect();
    let indices = Tensor::from_vec(idx_host, (N, K), &dev)?;
    let gate = Tensor::rand(0.5f32, 3.0, (N, K), &dev)?;
    let visible = Tensor::rand(0f32, 1.0, (N, K), &dev)?
        .gt(0.3)?
        .to_dtype(DType::F32)?;
    let values_nd = Tensor::rand(0f32, 20.0, (N, D), &dev)?.floor()?;
    let lib_n1 = (values_nd.sum_keepdim(1)? + 1.0)?;

    // 1. The [K, D] logits and the partition.
    time(&dev, "full_kd + partition (fwd)", || {
        let full_kd = dec.full_logits_kd()?;
        let _ = EmbeddedNbTopicDecoder::log_partition_k1(&full_kd)?;
        Ok(())
    });

    // 2. Dense NB head, forward only.
    // The hidden block the loader hands the head: a fixed count per row, drawn
    // without replacement and sorted, as `masked_dense::draw_row` produces it.
    let hidden_ids = {
        use rand::seq::SliceRandom;
        let dh = (0.4 * D as f64).round() as usize;
        let mut rng = rand::rng();
        let mut ids = Vec::with_capacity(N * dh);
        for _ in 0..N {
            let mut all: Vec<u32> = (0..D as u32).collect();
            all.shuffle(&mut rng);
            let mut row = all[..dh].to_vec();
            row.sort_unstable();
            ids.extend(row);
        }
        Tensor::from_vec(ids, (N, dh), &dev)?
    };
    time(&dev, "dense NB head (fwd)", || {
        let full_kd = dec.full_logits_kd()?;
        let dense = MaskedDenseTarget {
            values: &values_nd,
            residual: None,
            lib: &lib_n1,
            hidden_ids: &hidden_ids,
            hidden_weight: None,
        };
        let _ = dec.impute_dense_nb(&log_theta, &dense, &full_kd)?;
        Ok(())
    });
    // 3. Dense NB head, forward + backward.
    time(&dev, "dense NB head (fwd+bwd)", || {
        let full_kd = dec.full_logits_kd()?;
        let dense = MaskedDenseTarget {
            values: &values_nd,
            residual: None,
            lib: &lib_n1,
            hidden_ids: &hidden_ids,
            hidden_weight: None,
        };
        let llik = dec.impute_dense_nb(&log_theta, &dense, &full_kd)?;
        let _ = llik.mean_all()?.neg()?.backward()?;
        Ok(())
    });
    // 4. THE TRAINING STEP'S ENCODER: the dense read. Scores over every gene
    //    from the `[D]` query projection, a softmax over the gene axis, and one
    //    `[N, D]·[D, H]` gemm to pool. The visible mask is derived from the very
    //    `hidden_ids` block above, exactly as `masked_dense` derives it.
    let attn_q = Var::from_tensor(&Tensor::randn(0f32, 0.1, (1, H), &dev)?)?;
    let gate_nd = Tensor::rand(0.5f32, 3.0, (N, D), &dev)?;
    let visible_nd = {
        let dh = hidden_ids.dim(1)?;
        Tensor::ones((N, D), DType::F32, &dev)?.scatter(
            &hidden_ids,
            &Tensor::zeros((N, dh), DType::F32, &dev)?,
            1,
        )?
    };
    time(&dev, "dense pool over every gene (fwd)", || {
        let rq_d = query_over_features(&features, &attn_q)?;
        let scores =
            attention_scores_dense(&gate_nd, &rq_d, Some(&visible_nd), 1.0 / (H as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&scores, 1)?;
        let _ = pool_dense(&attn, &gate_nd, &features)?;
        Ok(())
    });
    time(&dev, "dense pool over every gene (fwd+bwd)", || {
        let rq_d = query_over_features(&features, &attn_q)?;
        let scores =
            attention_scores_dense(&gate_nd, &rq_d, Some(&visible_nd), 1.0 / (H as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&scores, 1)?;
        let pooled = pool_dense(&attn, &gate_nd, &features)?;
        let _ = pooled.sum_all()?.backward()?;
        Ok(())
    });

    println!("-- legacy: old-model scoring only, no training step runs these --");

    // 5. target_mask_nd scatter alone.
    time(&dev, "legacy: target_mask_nd scatter", || {
        let _ = target_mask_nd(&indices, &visible, D)?;
        Ok(())
    });

    // 6. Token gather the way the WINDOWED encoder read a slot block.
    time(&dev, "legacy: token gather [N,K,H] (fwd)", || {
        let e = rho
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((N, K, H))?;
        let _ = e.broadcast_mul(&gate.unsqueeze(2)?)?;
        Ok(())
    });
    time(&dev, "legacy: token gather [N,K,H] (fwd+bwd)", || {
        let e = rho
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((N, K, H))?;
        let c = e.broadcast_mul(&gate.unsqueeze(2)?)?;
        let _ = c.sum_all()?.backward()?;
        Ok(())
    });

    // 6b. The same pool, re-associated: ρq once, scores gathered from that
    //     vector, weights scattered onto [N, D], one gemm. Nothing [N, K, H].
    time(&dev, "legacy: scatter pool (fwd+bwd)", || {
        let rq_d = query_over_features(&features, &attn_q)?;
        let scores = attention_scores_from_vector(
            &gate,
            &indices,
            &rq_d,
            &visible,
            1.0 / (H as f64).sqrt(),
        )?;
        let attn = candle_nn::ops::softmax(&scores, 1)?;
        let pooled = pool_by_scatter(&attn, &gate, &indices, &features)?;
        let _ = pooled.sum_all()?.backward()?;
        Ok(())
    });
    // 7. Scatter of per-slot values onto [N, D] (the module view of the context).
    time(
        &dev,
        "legacy: scatter_add_cols [N,K]→[N,D] (fwd+bwd)",
        || {
            let v = Var::from_tensor(&Tensor::rand(-1f32, 1.0, (N, K), &dev)?)?;
            let nd = scatter_add_cols(&indices, &v, D)?;
            let _ = nd.sum_all()?.backward()?;
            Ok(())
        },
    );

    /////////////////////////////////////////////////////////////
    // The same gathers with row-parallel backward kernels      //
    /////////////////////////////////////////////////////////////

    let w_r = Var::from_tensor(&Tensor::randn(0f32, 0.1, (H, R), &dev)?)?;
    // 9. Keys the dense way: project the table, broadcast to [N, D, r], gather
    //    along the gene axis with [N, K, r] ids. Backward = scatter into
    //    [N, D, r], parallel over N·r.
    let idx3 = indices
        .unsqueeze(2)?
        .broadcast_as((N, K, R))?
        .contiguous()?;
    time(&dev, "keys via gather on broadcast [N,D,r] (fwd)", || {
        let rho_r = rho.matmul(&w_r)?;
        let big = rho_r.unsqueeze(0)?.broadcast_as((N, D, R))?.contiguous()?;
        let _ = big.gather(&idx3, 1)?.broadcast_mul(&gate.unsqueeze(2)?)?;
        Ok(())
    });
    time(
        &dev,
        "keys via gather on broadcast [N,D,r] (fwd+bwd)",
        || {
            let rho_r = rho.matmul(&w_r)?;
            let big = rho_r.unsqueeze(0)?.broadcast_as((N, D, R))?.contiguous()?;
            let keys = big.gather(&idx3, 1)?.broadcast_mul(&gate.unsqueeze(2)?)?;
            let _ = keys.sum_all()?.backward()?;
            Ok(())
        },
    );
    // 9b. Full-width tokens the same way ([N, D, H] broadcast).
    let idx3h = indices
        .unsqueeze(2)?
        .broadcast_as((N, K, H))?
        .contiguous()?;
    time(
        &dev,
        "tokens via gather on broadcast [N,D,H] (fwd+bwd)",
        || {
            let big = rho.unsqueeze(0)?.broadcast_as((N, D, H))?.contiguous()?;
            let tok = big.gather(&idx3h, 1)?.broadcast_mul(&gate.unsqueeze(2)?)?;
            let _ = tok.sum_all()?.backward()?;
            Ok(())
        },
    );

    // 10. The encoder pool as a dense [N, D] weight matrix times ρ:
    //     scores from a gathered [D] vector, weights scattered into [N, D]
    //     with a flattened index_add (its backward is an index_select), then
    //     one matmul. No per-slot rows of ρ are ever gathered.
    let q_pool = Var::from_tensor(&Tensor::randn(0f32, 0.1, (H, 1), &dev)?)?;
    let offsets = Tensor::arange(0u32, N as u32, &dev)?
        .affine(D as f64, 0.0)?
        .unsqueeze(1)?;
    let flat_idx = indices.broadcast_add(&offsets)?.reshape(N * K)?;
    time(&dev, "encoder pool as dense W[N,D]·ρ (fwd+bwd)", || {
        let u = rho.matmul(&q_pool)?.squeeze(1)?; // [D]
        let u_nd = u.unsqueeze(0)?.broadcast_as((N, D))?.contiguous()?;
        let s = u_nd.gather(&indices, 1)?.mul(&gate)?; // [N, K]
        let neg = visible.affine(-1.0, 1.0)?.affine(-1e9, 0.0)?;
        let a = candle_nn::ops::softmax(&(s + neg)?, 1)?;
        let w_nk = a.mul(&gate)?.reshape(N * K)?;
        let w_nd = Tensor::zeros(N * D, DType::F32, &dev)?
            .index_add(&flat_idx, &w_nk, 0)?
            .reshape((N, D))?;
        let pooled = w_nd.matmul(&rho)?; // [N, H]
        let _ = pooled.sum_all()?.backward()?;
        Ok(())
    });

    //////////////////////////////////////////////
    // gather_rows: the custom row-parallel op   //
    //////////////////////////////////////////////

    // Correctness on this device: gradient through gather_rows vs index_select.
    {
        let flat = indices.flatten_all()?;
        let w = Tensor::rand(-1f32, 1.0, (N * K, H), &dev)?;
        let t1 = Var::from_tensor(&rho)?;
        let g1 = (gather_rows(t1.as_tensor(), &flat)? * &w)?
            .sum_all()?
            .backward()?;
        let t2 = Var::from_tensor(&rho)?;
        let g2 = (t2.as_tensor().index_select(&flat, 0)? * &w)?
            .sum_all()?
            .backward()?;
        let diff = (g1.get(&t1).unwrap() - g2.get(&t2).unwrap())?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let scale = g2.get(&t2).unwrap().abs()?.max_all()?.to_scalar::<f32>()?;
        println!("gather_rows backward vs index_select backward: max |diff| = {diff:.3e} (max |grad| = {scale:.3e})");
    }
    // The scatter alone, against candle's index_add: the backward of every
    // gather, isolated from the elementwise passes around it.
    {
        use candle_util::fast_index::index_add_rows;
        let flat = indices.flatten_all()?;
        let src = Tensor::rand(-1f32, 1.0, (N * K, H), &dev)?;
        let zeros = Tensor::zeros((D, H), DType::F32, &dev)?;
        time(&dev, "index_add_rows [N·K, H] → [D, H] (alone)", || {
            let _ = index_add_rows(&zeros, &flat, &src)?;
            Ok(())
        });
        time(&dev, "candle index_add, same shapes (alone)", || {
            let _ = zeros.index_add(&flat, &src, 0)?;
            Ok(())
        });
    }
    time(&dev, "gather_rows [N·K, H] (fwd+bwd)", || {
        let e = gather_rows(&rho, &indices.flatten_all()?)?.reshape((N, K, H))?;
        let c = e.broadcast_mul(&gate.unsqueeze(2)?)?;
        let _ = c.sum_all()?.backward()?;
        Ok(())
    });
    time(
        &dev,
        "gather_rows on [D, r] table, [N·K] ids (fwd+bwd)",
        || {
            let rho_r = rho.matmul(&w_r)?;
            let k = gather_rows(&rho_r, &indices.flatten_all()?)?.reshape((N, K, R))?;
            let _ = k
                .broadcast_mul(&gate.unsqueeze(2)?)?
                .sum_all()?
                .backward()?;
            Ok(())
        },
    );
    Ok(())
}
