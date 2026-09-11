//! Tests for the dense masked-imputation heads: the `[N, D]` mixture rate
//! sums to one, the dense NB / multinomial heads match an element-wise
//! reference, and they agree with the indexed heads at the same positions.

use super::*;
use candle_core::{DType, Device};
use std::collections::HashMap;

const D: usize = 6;
const H: usize = 3;
const K: usize = 4;
const N: usize = 3;

fn dev() -> Device {
    Device::Cpu
}

fn rho() -> Tensor {
    let rows: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    Tensor::from_vec(rows, (D, H), &dev()).unwrap()
}

/// A free feature side holding exactly [`rho`], for the tests that want a
/// fixed table rather than a learned one.
fn fixed_features() -> std::sync::Arc<crate::feature_embedding::FeatureEmbedding> {
    crate::feature_embedding::FeatureEmbedding::fixed(rho())
}

fn alpha() -> Tensor {
    let rows: Vec<f32> = (0..K * H)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.25)
        .collect();
    Tensor::from_vec(rows, (K, H), &dev()).unwrap()
}

/// A decoder with deterministic `α`, `φ` and a non-uniform pinned background.
fn decoder() -> EmbeddedNbTopicDecoder {
    let log_phi: Vec<f32> = (0..D).map(|g| 0.2 * g as f32 - 0.5).collect();
    let log_pi: Vec<f32> = (0..D).map(|g| -((g + 1) as f32).ln()).collect();
    let mut ts = HashMap::new();
    ts.insert("dec.topic.embeddings".to_string(), alpha());
    ts.insert(
        "dec.log_phi".to_string(),
        Tensor::from_vec(log_phi, (1, D), &dev()).unwrap(),
    );
    ts.insert(
        "dec.log_pi".to_string(),
        Tensor::from_vec(log_pi, (1, D), &dev()).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev());
    EmbeddedNbTopicDecoder::new(K, fixed_features(), vb.pp("dec")).unwrap()
}

fn log_theta() -> Tensor {
    let logits: Vec<f32> = (0..N * K).map(|i| ((i * 3 % 7) as f32) * 0.6).collect();
    let t = Tensor::from_vec(logits, (N, K), &dev()).unwrap();
    ops::log_softmax(&t, 1).unwrap()
}

/// Counts with zeros in every row.
fn values() -> Tensor {
    #[rustfmt::skip]
    let v: Vec<f32> = vec![
        3.0, 0.0, 1.0, 0.0, 5.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0, 7.0,
        1.0, 1.0, 0.0, 4.0, 0.0, 0.0,
    ];
    Tensor::from_vec(v, (N, D), &dev()).unwrap()
}

/// 1 = scored. Scores zero-count genes and leaves some nonzero ones out.
fn mask() -> Tensor {
    #[rustfmt::skip]
    let m: Vec<f32> = vec![
        0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
    ];
    Tensor::from_vec(m, (N, D), &dev()).unwrap()
}

fn residual() -> Tensor {
    let r: Vec<f32> = (0..N * D)
        .map(|i| 0.5 + ((i * 5 % 9) as f32) * 0.1)
        .collect();
    Tensor::from_vec(r, (N, D), &dev()).unwrap()
}

fn lib() -> Tensor {
    Tensor::from_vec(vec![9.0f32, 10.0, 7.0], (N, 1), &dev()).unwrap()
}

fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_vec2().unwrap()
}

fn to_vec1(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn dense_mixture_rate_rows_sum_to_one() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let p = dec.mixture_rate_nd(&log_theta(), &full_kd).unwrap();
    assert_eq!(p.dims(), &[N, D]);
    for (n, row) in to_vec2(&p).iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row {n} sums to {s}");
    }
}

/// The dense heads compose `μ = residual · ℓ · p` and the mask exactly as an
/// element-by-element reference does. The reference calls the same NB element
/// function on `[1, 1]` tensors, so this pins the composition, not the lgamma.
#[test]
fn dense_heads_match_an_elementwise_reference() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let p = to_vec2(&dec.mixture_rate_nd(&log_theta(), &full_kd).unwrap());
    let (y, m, r) = (to_vec2(&values()), to_vec2(&mask()), to_vec2(&residual()));
    let l = to_vec1(&lib());
    let log_phi = to_vec1(dec.log_phi());

    let (values, residual, lib, mask) = (values(), residual(), lib(), mask());
    let target = MaskedDenseTarget {
        values: &values,
        residual: Some(&residual),
        lib: &lib,
        mask: &mask,
    };
    let nb = to_vec1(
        &dec.impute_dense_nb(&log_theta(), &target, &full_kd)
            .unwrap(),
    );
    let mn = to_vec1(
        &dec.impute_dense_multinomial(&log_theta(), &target, &full_kd)
            .unwrap(),
    );

    let one = |v: f32| Tensor::new(&[[v]], &dev()).unwrap();
    for n in 0..N {
        let mut nb_ref = 0f32;
        let mut mn_ref = 0f32;
        for g in 0..D {
            if m[n][g] == 0.0 {
                continue;
            }
            let mu = r[n][g] * l[n] * p[n][g];
            let elem = nb_log_likelihood_elem(&one(y[n][g]), &one(mu), &one(log_phi[g])).unwrap();
            nb_ref += to_vec1(&elem)[0];
            mn_ref += y[n][g] * (p[n][g] + 1e-20).ln();
        }
        assert!(
            (nb[n] - nb_ref).abs() < 1e-3,
            "row {n}: dense NB {} vs reference {nb_ref}",
            nb[n]
        );
        assert!(
            (mn[n] - mn_ref).abs() < 1e-4,
            "row {n}: dense multinomial {} vs reference {mn_ref}",
            mn[n]
        );
    }
}

/// Handing the indexed heads every gene of a row, in any order, must give the
/// dense heads' number: the two layouts score one likelihood.
#[test]
fn dense_and_indexed_heads_agree_on_the_same_positions() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let perms: Vec<Vec<u32>> = vec![
        vec![5, 0, 3, 1, 4, 2],
        vec![2, 4, 0, 5, 1, 3],
        vec![0, 1, 2, 3, 4, 5],
    ];
    let gather = |t: &Tensor| -> Tensor {
        let v = to_vec2(t);
        let mut g = Vec::with_capacity(N * D);
        for (n, p) in perms.iter().enumerate() {
            for &j in p {
                g.push(v[n][j as usize]);
            }
        }
        Tensor::from_vec(g, (N, D), &dev()).unwrap()
    };
    let indices = Tensor::from_vec(perms.concat(), (N, D), &dev()).unwrap();
    let (values, residual, lib, mask) = (values(), residual(), lib(), mask());
    let (vals_k, mask_k, res_k) = (gather(&values), gather(&mask), gather(&residual));

    let indexed = MaskedNbTarget {
        indices: &indices,
        residual: Some(&res_k),
        values: &vals_k,
        lib: &lib,
        mask: &mask_k,
    };
    let dense = MaskedDenseTarget {
        values: &values,
        residual: Some(&residual),
        lib: &lib,
        mask: &mask,
    };

    let nb_i = to_vec1(
        &dec.impute_masked_nb(&log_theta(), &indexed, &full_kd)
            .unwrap(),
    );
    let nb_d = to_vec1(&dec.impute_dense_nb(&log_theta(), &dense, &full_kd).unwrap());
    let mn_i = to_vec1(
        &dec.impute_masked_multinomial(&log_theta(), &indexed, &full_kd)
            .unwrap(),
    );
    let mn_d = to_vec1(
        &dec.impute_dense_multinomial(&log_theta(), &dense, &full_kd)
            .unwrap(),
    );
    for n in 0..N {
        assert!(
            (nb_i[n] - nb_d[n]).abs() < 1e-3,
            "row {n}: indexed NB {} vs dense {}",
            nb_i[n],
            nb_d[n]
        );
        assert!(
            (mn_i[n] - mn_d[n]).abs() < 1e-4,
            "row {n}: indexed multinomial {} vs dense {}",
            mn_i[n],
            mn_d[n]
        );
    }
}

////////////////////////////
// Module-collapsed heads   //
////////////////////////////

use super::{ModuleTarget, QueryTarget};
use crate::decoder::coarsening_map::CoarseningMap;
use crate::loss::nb_log_likelihood_elem;

/// Modules {0,1}, {2,3,4}, {5} over the six genes, shares from a mean
/// vector [2,1 | 1,1,2 | 4].
fn coarsening_map() -> CoarseningMap {
    CoarseningMap::new(
        &[0, 0, 1, 1, 1, 2],
        &[2.0 / 3.0, 1.0 / 3.0, 0.25, 0.25, 0.5, 1.0],
        &dev(),
    )
    .unwrap()
}

const M: usize = 3;

/// A decoder over the three modules with deterministic α, φ and background.
fn module_decoder() -> EmbeddedNbTopicDecoder {
    let log_phi: Vec<f32> = (0..M).map(|m| 0.3 * m as f32 - 0.4).collect();
    let log_pi: Vec<f32> = vec![
        (3.0f32 / 11.0).ln(),
        (4.0f32 / 11.0).ln(),
        (4.0f32 / 11.0).ln(),
    ];
    let mut ts = HashMap::new();
    ts.insert("mdec.topic.embeddings".to_string(), alpha());
    ts.insert(
        "mdec.log_phi".to_string(),
        Tensor::from_vec(log_phi, (1, M), &dev()).unwrap(),
    );
    ts.insert(
        "mdec.log_pi".to_string(),
        Tensor::from_vec(log_pi, (1, M), &dev()).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev());
    EmbeddedNbTopicDecoder::new_with_coarsening(
        K,
        fixed_features(),
        coarsening_map(),
        vb.pp("mdec"),
    )
    .unwrap()
}

#[test]
fn module_logits_carry_the_module_background_and_rows_sum_to_one() {
    let dec = module_decoder();
    assert_eq!(dec.dim_obs(), M);
    let full_km = dec.full_logits_kd().unwrap();
    assert_eq!(full_km.dims(), &[K, M]);
    // Host: (α − ᾱ)·ρ̄ᵀ + log π_m with ρ̄ the within-module mean.
    let a = alpha().to_vec2::<f32>().unwrap();
    let r = rho().to_vec2::<f32>().unwrap();
    let groups: [&[usize]; 3] = [&[0, 1], &[2, 3, 4], &[5]];
    let log_pi = [
        (3.0f32 / 11.0).ln(),
        (4.0f32 / 11.0).ln(),
        (4.0f32 / 11.0).ln(),
    ];
    let mean_a: Vec<f32> = (0..H)
        .map(|h| (0..K).map(|k| a[k][h]).sum::<f32>() / K as f32)
        .collect();
    let got = full_km.to_vec2::<f32>().unwrap();
    for k in 0..K {
        for (m, g) in groups.iter().enumerate() {
            let bar: Vec<f32> = (0..H)
                .map(|h| g.iter().map(|&i| r[i][h]).sum::<f32>() / g.len() as f32)
                .collect();
            let want: f32 = (0..H).map(|h| (a[k][h] - mean_a[h]) * bar[h]).sum::<f32>() + log_pi[m];
            assert!(
                (got[k][m] - want).abs() < 1e-5,
                "logit[{k}][{m}] {} vs {want}",
                got[k][m]
            );
        }
    }
    let beta = ops::softmax(&full_km, 1)
        .unwrap()
        .sum(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(beta.iter().all(|s| (s - 1.0).abs() < 1e-5));
}

/// Under the identity map the module scorer is the dense gene scorer, term
/// for term, for both likelihoods; the scored units are the mask.
#[test]
fn module_scorer_matches_the_dense_gene_scorer_under_the_identity_map() {
    let dec = decoder();
    assert!(dec.coarsening().is_identity());
    let full_kd = dec.full_logits_kd().unwrap();
    let (values, mask, lib) = (values(), mask(), lib());
    let visible = mask.affine(-1.0, 1.0).unwrap();
    let visible_counts = (&values * &visible).unwrap();
    let t = ModuleTarget {
        values: &values,
        visible_counts: &visible_counts,
        visible_share: &visible,
        residual: None,
        lib: &lib,
    };
    let dense = MaskedDenseTarget {
        values: &values,
        residual: None,
        lib: &lib,
        mask: &mask,
    };
    let (nb, units) = dec
        .score_unseen_modules_nb(&log_theta(), &t, &full_kd)
        .unwrap();
    let want = dec.impute_dense_nb(&log_theta(), &dense, &full_kd).unwrap();
    for (a, b) in to_vec1(&nb).iter().zip(to_vec1(&want)) {
        assert!((a - b).abs() < 1e-4, "NB module {a} vs dense {b}");
    }
    assert_eq!(to_vec1(&units), to_vec1(&mask.sum(1).unwrap()));
    let (mn, _) = dec
        .score_unseen_modules_multinomial(&log_theta(), &t, &full_kd)
        .unwrap();
    let want = dec
        .impute_dense_multinomial(&log_theta(), &dense, &full_kd)
        .unwrap();
    for (a, b) in to_vec1(&mn).iter().zip(to_vec1(&want)) {
        assert!((a - b).abs() < 1e-4, "multinomial module {a} vs dense {b}");
    }
}

/// The unseen target is the module total minus what the context saw, the
/// rate is scaled by the prior unseen share, and a module the context saw
/// completely is not scored.
#[test]
fn unseen_module_scores_match_a_host_reference_and_skip_full_modules() {
    let dec = module_decoder();
    let full_km = dec.full_logits_kd().unwrap();
    #[rustfmt::skip]
    let values: Vec<f32> = vec![
        9.0, 4.0, 2.0,
        1.0, 6.0, 0.0,
        5.0, 5.0, 5.0,
    ];
    #[rustfmt::skip]
    let vis_counts: Vec<f32> = vec![
        3.0, 0.0, 2.0,   // row 0 saw part of module 0, none of 1, all of 2
        0.0, 6.0, 0.0,
        5.0, 1.0, 0.0,
    ];
    #[rustfmt::skip]
    let vis_share: Vec<f32> = vec![
        1.0 / 3.0, 0.0, 1.0,
        0.0, 0.5, 0.0,
        1.0, 0.25, 0.0,
    ];
    let values_t = Tensor::from_vec(values.clone(), (N, M), &dev()).unwrap();
    let vc = Tensor::from_vec(vis_counts.clone(), (N, M), &dev()).unwrap();
    let vs = Tensor::from_vec(vis_share.clone(), (N, M), &dev()).unwrap();
    let lib = lib();
    let t = ModuleTarget {
        values: &values_t,
        visible_counts: &vc,
        visible_share: &vs,
        residual: None,
        lib: &lib,
    };
    let (llik, units) = dec
        .score_unseen_modules_nb(&log_theta(), &t, &full_km)
        .unwrap();
    // Host reference on the same rate.
    let rate = dec.mixture_rate_nd(&log_theta(), &full_km).unwrap();
    let unseen = (&values_t - &vc).unwrap();
    let s = vs.affine(-1.0, 1.0).unwrap();
    let mu = (rate * &s).unwrap().broadcast_mul(&lib).unwrap();
    let log_phi = dec.log_phi().broadcast_as((N, M)).unwrap();
    let elem = nb_log_likelihood_elem(&unseen, &mu, &log_phi).unwrap();
    let scored = s.gt(1e-6).unwrap().to_dtype(DType::F32).unwrap();
    let want = (elem * &scored).unwrap().sum(1).unwrap();
    for (a, b) in to_vec1(&llik).iter().zip(to_vec1(&want)) {
        assert!((a - b).abs() < 1e-4, "{a} vs host {b}");
    }
    assert_eq!(to_vec1(&units), vec![2.0, 3.0, 2.0]);
    // Perturb the fully visible modules' totals: nothing moves.
    let mut v2 = values;
    v2[2] += 7.0; // row 0, module 2
    v2[6] += 7.0; // row 2, module 0
    let values2 = Tensor::from_vec(v2, (N, M), &dev()).unwrap();
    let t2 = ModuleTarget {
        values: &values2,
        visible_counts: &vc,
        visible_share: &vs,
        residual: None,
        lib: &lib,
    };
    let (llik2, _) = dec
        .score_unseen_modules_nb(&log_theta(), &t2, &full_km)
        .unwrap();
    assert_eq!(to_vec1(&llik), to_vec1(&llik2));
}

/// A query gene's rate is its module's rate times its share times the
/// residual; with r = 0 it is the expanded dictionary's rate.
#[test]
fn gene_level_query_rate_is_module_rate_times_share_times_residual() {
    let dec = module_decoder();
    let full_km = dec.full_logits_kd().unwrap();
    let ids = Tensor::from_vec(vec![5u32, 0, 3, 1, 4, 2], (N, 2), &dev()).unwrap();
    let xq = Tensor::from_vec(vec![2.0f32, 0.0, 1.0, 4.0, 0.0, 3.0], (N, 2), &dev()).unwrap();
    let w = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 0.0, 1.0, 1.0], (N, 2), &dev()).unwrap();
    let r = Tensor::from_vec(vec![0.3f32, -0.2, 0.0, 0.5, 1.0, -1.0], (N, 2), &dev()).unwrap();
    let lib = lib();
    let q = QueryTarget {
        gene_ids: &ids,
        values: &xq,
        weight: &w,
        log_residual: &r,
        lib: &lib,
    };
    let got = to_vec1(&dec.score_queries_nb(&log_theta(), &q, &full_km).unwrap());
    // Host: μ = ℓ · rate_{m(g)} · π_{g|m} · exp(r), φ at the module.
    let rate = dec
        .mixture_rate_nd(&log_theta(), &full_km)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let f2c = [0usize, 0, 1, 1, 1, 2];
    let share = [2.0f32 / 3.0, 1.0 / 3.0, 0.25, 0.25, 0.5, 1.0];
    let idv = ids.to_vec2::<u32>().unwrap();
    let rv = r.to_vec2::<f32>().unwrap();
    let libv = to_vec1(&lib);
    let mut mu = vec![0f32; N * 2];
    let mut lp = vec![0f32; N * 2];
    let log_phi_m = to_vec1(dec.log_phi());
    for n in 0..N {
        for j in 0..2 {
            let g = idv[n][j] as usize;
            mu[n * 2 + j] = libv[n] * rate[n][f2c[g]] * share[g] * rv[n][j].exp();
            lp[n * 2 + j] = log_phi_m[f2c[g]];
        }
    }
    let mu_t = Tensor::from_vec(mu, (N, 2), &dev()).unwrap();
    let lp_t = Tensor::from_vec(lp, (N, 2), &dev()).unwrap();
    let want = to_vec1(
        &(nb_log_likelihood_elem(&xq, &mu_t, &lp_t).unwrap() * &w)
            .unwrap()
            .sum(1)
            .unwrap(),
    );
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "query llik {a} vs host {b}");
    }
}

/// The indexed head on a module decoder scores a gene at its module's rate
/// times its share — the expanded dictionary.
#[test]
fn indexed_head_at_genes_agrees_with_the_expanded_dictionary() {
    let dec = module_decoder();
    let full_km = dec.full_logits_kd().unwrap();
    let ids = Tensor::from_vec(vec![5u32, 0, 3, 1, 4, 2], (N, 2), &dev()).unwrap();
    let got = dec
        .mixture_rate_nk(&log_theta(), &ids, &full_km)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let rate = dec
        .mixture_rate_nd(&log_theta(), &full_km)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let f2c = [0usize, 0, 1, 1, 1, 2];
    let share = [2.0f32 / 3.0, 1.0 / 3.0, 0.25, 0.25, 0.5, 1.0];
    let idv = ids.to_vec2::<u32>().unwrap();
    for n in 0..N {
        for j in 0..2 {
            let g = idv[n][j] as usize;
            let want = rate[n][f2c[g]] * share[g];
            assert!(
                (got[n][j] - want).abs() < 1e-5,
                "rate[{n}][{j}] {} vs {want}",
                got[n][j]
            );
        }
    }
}

/// A per-module batch offset multiplies the NB mean: a uniform offset `c` is
/// the same as scaling the library by `c`.
#[test]
fn a_module_residual_scales_the_nb_mean() {
    let dec = module_decoder();
    let full_km = dec.full_logits_kd().unwrap();
    let values = Tensor::from_vec(
        vec![9.0f32, 4.0, 2.0, 1.0, 6.0, 0.0, 5.0, 5.0, 5.0],
        (N, M),
        &dev(),
    )
    .unwrap();
    let vc = Tensor::zeros((N, M), DType::F32, &dev()).unwrap();
    let vs = Tensor::from_vec(
        vec![0.2f32, 0.0, 0.5, 0.0, 0.5, 0.0, 0.1, 0.25, 0.0],
        (N, M),
        &dev(),
    )
    .unwrap();
    let lib = lib();
    let c = 1.7f64;
    let uniform = Tensor::full(c as f32, (N, M), &dev()).unwrap();
    let with_res = ModuleTarget {
        values: &values,
        visible_counts: &vc,
        visible_share: &vs,
        residual: Some(&uniform),
        lib: &lib,
    };
    let scaled = lib.affine(c, 0.0).unwrap();
    let with_lib = ModuleTarget {
        values: &values,
        visible_counts: &vc,
        visible_share: &vs,
        residual: None,
        lib: &scaled,
    };
    let (a, _) = dec
        .score_unseen_modules_nb(&log_theta(), &with_res, &full_km)
        .unwrap();
    let (b, _) = dec
        .score_unseen_modules_nb(&log_theta(), &with_lib, &full_km)
        .unwrap();
    for (x, y) in to_vec1(&a).iter().zip(to_vec1(&b)) {
        assert!((x - y).abs() < 1e-4, "residual {x} vs scaled library {y}");
    }
}

/// The decoder shares the encoder's feature side, and "shares" has to mean the
/// live one. With a free table that is automatic, because the table is a `Var`
/// and a held handle reads through to it. With a composed feature side the
/// handle is a computed value, so holding it freezes the feature rows at their
/// initialization while the parameters underneath move — the fit would train
/// its topics against an embedding that never updates.
#[test]
fn the_decoder_sees_the_feature_side_as_it_moves() {
    use crate::feature_embedding::{FeatureEmbedding, LOGITS_VAR_NAME, MU_VAR_NAME};
    let dev = Device::Cpu;
    let vm = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let (d, h, m, k) = (5usize, 3usize, 2usize, 2usize);
    let features = FeatureEmbedding::new(d, m, h, vb.pp("enc")).unwrap();
    let dec = EmbeddedNbTopicDecoder::new(k, std::sync::Arc::new(features), vb.pp("dec")).unwrap();

    // A flat membership gives every feature the same row, and a dictionary
    // that is uniform whatever `μ` holds — so separate the features first, or
    // the check cannot fail no matter how stale the read is.
    {
        let data = vm.data().lock().unwrap();
        let spread: Vec<f32> = (0..d * m)
            .map(|i| if i % m == i / m % m { 4.0 } else { 0.0 })
            .collect();
        data[&format!("enc.{LOGITS_VAR_NAME}")]
            .set(&Tensor::from_vec(spread, (d, m), &dev).unwrap())
            .unwrap();
    }
    let before: Vec<Vec<f32>> = dec.get_dictionary().unwrap().to_vec2().unwrap();
    assert!(
        before.iter().any(|r| r != &before[0]),
        "the fixture must separate the features, or staleness is undetectable"
    );

    // Move the dictionary the feature rows are composed from.
    {
        let data = vm.data().lock().unwrap();
        let mu = &data[&format!("enc.{MU_VAR_NAME}")];
        let bumped = (mu.as_tensor() * 3.0).unwrap();
        mu.set(&bumped).unwrap();
    }

    let after: Vec<Vec<f32>> = dec.get_dictionary().unwrap().to_vec2().unwrap();
    assert_ne!(
        before, after,
        "the decoder is reading a snapshot of the feature side, not the live one"
    );
}
