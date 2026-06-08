//! Synthetic-fixture sim-recovery test for the component-resolved,
//! region-keyed embedding (the rework's standing validation rule: must
//! recover planted structure on simulated data).
//!
//! There is no E2E test for the embed pipeline, so this drives the real
//! `model` + `train` + `sampling` stack on a hand-built `PseudobulkData`
//! whose pools encode a known generative truth, then asserts the
//! deviation gate recovered it.
//!
//! Planted world (cell axis only):
//!   * Cells split into group A (`0..n/2`) and group B (`n/2..n`).
//!   * Every gene is *expressed* in group A → its AGG (β anchor) and its
//!     region-0 m6A satellite both put mass on group-A cells.
//!   * Each gene's region-1 m6A satellite puts mass on group-B cells —
//!     the modification is *decoupled* from expression there.
//!
//! Two same-gene m6A components therefore sit in different regions with
//! different cell usage, so the only way to fit both is the additive
//! γ_{m,r,:} region offset. The swap-modality negative (which here swaps
//! region 0↔1 for the lone modifier modality) supplies the contrast.
//!
//! Assertions:
//!   (b) **redundancy null** — region-0 tracks expression, so its
//!       satellite embedding stays aligned with β_g (logdev ≈ 0).
//!   (c) **multi-site resolution** — region-1 leans toward group B far
//!       more than region-0 does; the two components are distinguishable.

use super::common::candle_core;
use candle_core::{Device, Tensor};
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::args::GemArgs;
use super::common::ComputeDevice;
use super::feature_table::FeatureTable;
use super::model::GemModel;
use super::pseudobulk::{AxisPools, PseudobulkData, StratumPool};
use super::region::RegionMap;
use super::train::train;

const N_GENES: usize = 4;
const N_CELLS: usize = 40;
const H: usize = 8;
const K: usize = 3;
const R: usize = 2;
const M6A: u32 = 1; // modality id (count = 0)

fn group_a() -> std::ops::Range<usize> {
    0..(N_CELLS / 2)
}
fn group_b() -> std::ops::Range<usize> {
    (N_CELLS / 2)..N_CELLS
}

/// Build one (gene, cell, region) pool with unit weights.
fn pool_from(entries: &[(u32, u32, u32)]) -> StratumPool {
    StratumPool {
        gene_ids: entries.iter().map(|e| e.0).collect(),
        axis_ids: entries.iter().map(|e| e.1).collect(),
        region_ids: entries.iter().map(|e| e.2).collect(),
        counts: vec![1.0; entries.len()],
        weights: vec![1.0; entries.len()],
    }
}

/// Synthetic feature names so `FeatureTable` assigns the gene/modality
/// ids and `measured` mask that the planted pools assume:
///   gene id = g (appearance order), modality id: count=0, m6A=1.
fn feature_names() -> Vec<Box<str>> {
    let mut names = Vec::new();
    for g in 0..N_GENES {
        names.push(format!("gene{g}/count/spliced").into_boxed_str());
        names.push(format!("gene{g}/m6A/0").into_boxed_str()); // region 0
        names.push(format!("gene{g}/m6A/1").into_boxed_str()); // region 1
    }
    names
}

fn build_pseudobulk() -> PseudobulkData {
    let mut agg = Vec::new();
    let mut count = Vec::new();
    let mut modifier = Vec::new();

    for g in 0..N_GENES as u32 {
        // β anchor + region-0 satellite live in group A (expression).
        for c in group_a() {
            agg.push((g, c as u32, 0));
            count.push((g, c as u32, 0));
            modifier.push((g, c as u32, 0)); // region 0
        }
        // region-1 satellite lives in group B (decoupled from expression).
        for c in group_b() {
            modifier.push((g, c as u32, 1)); // region 1
        }
    }

    let agg = pool_from(&agg);
    let count_comp = pool_from(&count);
    let modifier_pool = pool_from(&modifier);
    let modality_mass: f32 = modifier_pool.weights.iter().sum();

    // modifier_comp_per_modality: slot 0 (count) empty, slot 1 (m6A).
    let modifier_comp_per_modality = vec![pool_from(&[]), modifier_pool];
    let modality_total_mass = vec![0.0, modality_mass];

    let cell_pools = AxisPools {
        n_units: N_CELLS,
        agg,
        count_comp,
        modifier_comp_per_modality,
        modality_total_mass,
    };

    let gene_ubiquity =
        super::gene_weight::ubiquity_from_count_pool(&cell_pools.count_comp, N_GENES, N_CELLS);

    PseudobulkData {
        cell_to_pb_per_level: Vec::new(),
        cell_pools,
        pb_pools_per_level: Vec::new(),
        gene_fisher_weights: vec![1.0; N_GENES],
        gene_ubiquity,
    }
}

fn test_args() -> GemArgs {
    GemArgs {
        genes: vec!["".into()],
        dartseq: None,
        atoi: None,
        apa: None,
        dartseq_components: None,
        atoi_components: None,
        apa_components: None,
        batch_files: None,
        out: "".into(),
        embedding_dim: H,
        n_programs: K,
        n_regions: R,
        num_levels: 0,
        sort_dim: 10,
        knn_pb: 10,
        num_opt_iter: 1,
        proj_dim: 8,
        ignore_batch: true,
        no_cell_axis: false,
        feature_name_delim: '_',
        feature_name_exact: true,
        preload_data: true,
        epochs: 70,
        phase2_epochs: Some(0),
        batches_per_epoch: Some(4),
        batch_size: 96,
        learning_rate: 6e-2,
        z_l2: 1e-3,
        delta_l2: 5e-3, // δ shrinkage → keeps the exp gate from blowing up
        f_agg: 0.34,
        f_count: 0.0,
        tau: 1.0,
        tau_modality: 0.5,
        housekeeping_penalty: 0.0,
        resolve_topics: false,
        num_topics: None,
        max_k: 30,
        aa_iters: 50,
        aa_subsample: None,
        n_rand: 8,
        n_swap_gene_mode: 4,
        n_swap_modality: 6,
        seed: 7,
        device: ComputeDevice::Cpu,
        device_no: 0,
    }
}

/// Deterministically re-initialise the model's parameters from a seeded
/// RNG. candle's `Init::Randn` seeds from entropy, which would make this
/// training test flaky run-to-run; with the sampler also seeded this
/// pins the whole test. Biases stay at zero; everything else gets small
/// Gaussian noise. Vars are visited in sorted-name order so the draw
/// sequence is independent of `HashMap` iteration order.
fn seed_params(model: &GemModel, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let data = model.varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for name in names {
        let var = &data[name];
        let dims = var.shape().dims().to_vec();
        let n: usize = dims.iter().product();
        let values: Vec<f32> = if name.starts_with('b') {
            vec![0.0; n] // biases
        } else {
            (0..n)
                .map(|_| {
                    let x: f32 = StandardNormal.sample(&mut rng);
                    0.05_f32 * x
                })
                .collect()
        };
        let t = Tensor::from_vec(values, dims, &Device::Cpu).unwrap();
        var.set(&t).unwrap();
    }
}

fn row_vec(t: &Tensor) -> Vec<f32> {
    t.to_vec2::<f32>().unwrap().into_iter().next().unwrap()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = dot(a, a).sqrt();
    let nb = dot(b, b).sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot(a, b) / (na * nb)
    }
}

/// Satellite embedding e_f(g, m6A, region) for a single gene/region.
fn satellite_embed(model: &GemModel, g: u32, region: u32) -> Vec<f32> {
    let (e, _) = model
        .embed_and_bias_rows(&[g], &[g], &[M6A], &[region], &[g], &[M6A], &[false])
        .unwrap();
    row_vec(&e)
}

#[test]
fn recovers_region_resolved_deviation() {
    let dev = Device::Cpu;
    let table = FeatureTable::build(&feature_names(), &RegionMap::empty(R));
    assert_eq!(table.n_genes(), N_GENES);
    assert_eq!(table.n_regions, R);

    let pb = build_pseudobulk();
    let args = test_args();
    let mut model =
        GemModel::new(N_GENES, table.n_modalities(), K, R, H, N_CELLS, &[], &dev).unwrap();
    seed_params(&model, 20260530);

    let stop = Arc::new(AtomicBool::new(false));
    train(&args, &table, &pb, &mut model, &stop).unwrap();

    // Mean trained cell embedding per group.
    let e_cell = model.e_cell.to_vec2::<f32>().unwrap();
    let mean = |range: std::ops::Range<usize>| -> Vec<f32> {
        let mut acc = [0.0_f32; H];
        for c in range.clone() {
            for h in 0..H {
                acc[h] += e_cell[c][h];
            }
        }
        let n = range.len() as f32;
        acc.iter().map(|x| x / n).collect()
    };
    let mean_a = mean(group_a());
    let mean_b = mean(group_b());

    // Sanity: the two cell groups actually separated, else the test
    // proves nothing about region resolution.
    assert!(
        cosine(&mean_a, &mean_b) < 0.9,
        "cell groups did not separate (cos={:.3})",
        cosine(&mean_a, &mean_b)
    );

    // Per-gene, the deviation gate must route each region's satellite to
    // the cells where its modification mass actually sits:
    //   region 0 tracks expression (group A) → must prefer A;
    //   region 1 is decoupled (group B)      → must prefer B.
    // Dot-difference is scale-robust to the exp gate's magnitude.
    let mut r0_prefers_a = 0;
    let mut r1_prefers_b = 0;
    let mut resolves = 0;
    for g in 0..N_GENES as u32 {
        let e_r0 = satellite_embed(&model, g, 0);
        let e_r1 = satellite_embed(&model, g, 1);

        let r0_a = dot(&e_r0, &mean_a) - dot(&e_r0, &mean_b);
        let r1_b = dot(&e_r1, &mean_b) - dot(&e_r1, &mean_a);

        // (b) redundancy: region-0 satellite follows expression to A.
        if r0_a > 0.0 {
            r0_prefers_a += 1;
        }
        // (c-decoupled): region-1 satellite follows its modification to B.
        if r1_b > 0.0 {
            r1_prefers_b += 1;
        }
        // (c) multi-site resolution: the two same-gene components are
        // distinguishable — region 1 leans to B strictly more than
        // region 0 does.
        if r1_b > -r0_a {
            resolves += 1;
        }
    }

    assert_eq!(
        r0_prefers_a, N_GENES,
        "(b) region-0 (tracks expression) must prefer group A for every gene ({r0_prefers_a}/{N_GENES})"
    );
    assert_eq!(
        r1_prefers_b, N_GENES,
        "(c) region-1 (decoupled) must prefer group B for every gene ({r1_prefers_b}/{N_GENES})"
    );
    assert_eq!(
        resolves, N_GENES,
        "(c) two same-gene components in different regions must be distinguishable ({resolves}/{N_GENES})"
    );

    // AGG rows must be the pure β anchor (gate masked to exp(0)).
    let (e_agg, _) = model
        .embed_and_bias_rows(&[0], &[0], &[0], &[0], &[0], &[0], &[true])
        .unwrap();
    let beta0 = row_vec(&model.beta.narrow(0, 0, 1).unwrap());
    let e_agg = row_vec(&e_agg);
    for h in 0..H {
        assert!(
            (e_agg[h] - beta0[h]).abs() < 1e-5,
            "AGG embedding must equal β exactly"
        );
    }
}
