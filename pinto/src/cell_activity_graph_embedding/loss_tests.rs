use super::loss::cage_nce_loss_per_level;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use graph_embedding_util::loss::{CellChainBatch, NceObjective};
use graph_embedding_util::model::{JointEmbedModel, ModelArgs, ModelInit};
use nalgebra::DMatrix;

const N_CELLS: usize = 32;
const N_GENES: usize = 6;
const DIM: usize = 4;
const B: usize = 5;
const K: usize = 3;
const L: usize = 3;

/// `e_feat` is an INPUT to the loss — so these tests
/// drive it directly rather than through a gate or a `pip` mask.
fn model_with_e_feat(
    varmap: &VarMap,
    dev: &Device,
    e_feat: Option<&DMatrix<f32>>,
) -> JointEmbedModel {
    JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: N_GENES,
            n_cells: N_CELLS,
            embedding_dim: DIM,
            seed: 42,
        },
        &ModelInit {
            e_feat,
            e_cell: None,
            // Zero biases, so a dead gene embedding leaves the score at EXACTLY
            // the bias term and `pair_magnitude_reports_a_dead_gene_embedding`
            // can key off the pair term alone.
            b_feat: &[0.0_f32; N_GENES],
            b_cell: &[0.0_f32; N_CELLS],
        },
        varmap,
        dev,
    )
    .unwrap()
}

fn model(varmap: &VarMap, dev: &Device) -> JointEmbedModel {
    model_with_e_feat(varmap, dev, None)
}

/// One deterministic batch per gene. Cells are picked by a fixed stride so the
/// batches differ from each other without needing an RNG.
fn batches() -> (Vec<CellChainBatch>, Vec<u32>) {
    let mut out = Vec::new();
    for g in 0..N_GENES {
        let left: Vec<u32> = (0..B).map(|b| ((g * 7 + b * 3) % N_CELLS) as u32).collect();
        let right: Vec<u32> = (0..B)
            .map(|b| ((g * 7 + b * 3 + 1) % N_CELLS) as u32)
            .collect();
        let per_level_neg: Vec<Vec<u32>> = (0..L)
            .map(|l| {
                (0..B * K)
                    .map(|i| ((g * 5 + l * 11 + i * 2 + 17) % N_CELLS) as u32)
                    .collect()
            })
            .collect();
        out.push(CellChainBatch {
            left_cells: left,
            right_cells: right,
            per_level_neg,
            n_negatives: K,
        });
    }
    (out, (0..N_GENES as u32).collect())
}

#[test]
fn returns_one_loss_per_gene_and_level() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let m = model(&varmap, &dev);
    let (bs, ids) = batches();

    let out = cage_nce_loss_per_level(&m, bs, &ids, NceObjective::Logistic, &dev).unwrap();
    assert_eq!(out.per_level.dims(), &[N_GENES, L]);
    assert_eq!(
        out.mean_abs_pair.dims().len(),
        0,
        "pair magnitude is scalar"
    );
}

/// A zeroed gene embedding annihilates the gene direction, leaving the ungated
/// cell biases to explain the objective. That is the degenerate optimum the
/// squared-gate hazard used to lead to — measured under the old variational
/// gate: at a sparse init the pair term underflowed to exactly 0 by epoch 4 —
/// and `mean_abs_pair` is the diagnostic that must see it coming.
#[test]
fn pair_magnitude_reports_a_dead_gene_embedding() {
    let dev = Device::Cpu;

    let vm_live = VarMap::new();
    let live_m = model(&vm_live, &dev);
    let (bs, ids) = batches();
    let live = cage_nce_loss_per_level(&live_m, bs, &ids, NceObjective::Logistic, &dev).unwrap();
    let live_pair: f32 = live.mean_abs_pair.to_scalar().unwrap();
    assert!(
        live_pair > 0.0,
        "a live e_feat should leave a live pair term"
    );

    let vm_dead = VarMap::new();
    let zeros = DMatrix::<f32>::zeros(N_GENES, DIM);
    let dead_m = model_with_e_feat(&vm_dead, &dev, Some(&zeros));
    let (bs2, ids2) = batches();
    let dead = cage_nce_loss_per_level(&dead_m, bs2, &ids2, NceObjective::Logistic, &dev).unwrap();
    let dead_pair: f32 = dead.mean_abs_pair.to_scalar().unwrap();
    assert_eq!(dead_pair, 0.0, "a zeroed e_feat must kill the pair term");
    assert!(live_pair > dead_pair);
}

/// With NO gate installed, the pair term is exactly `e_feat[g] . (e_u * e_v)`.
///
/// This pins that `gathered_gate_weights` returns `None` on an ungated model and
/// the gather stays plain — i.e. installing the gate is what turns selection on,
/// and nothing else silently scales the gene rows. A stray factor would slip
/// past every other test in this file, because they only ever compare losses to
/// each other.
#[test]
fn score_is_exactly_the_raw_gene_embedding() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    // A gene embedding whose rows differ a lot, so a mask cannot coincidentally
    // reproduce the expected value.
    let e_feat = DMatrix::<f32>::from_fn(N_GENES, DIM, |g, d| {
        0.3 * (g as f32 + 1.0) - 0.17 * (d as f32)
    });
    let m = model_with_e_feat(&varmap, &dev, Some(&e_feat));

    let ef: Vec<Vec<f32>> = m.e_feat.to_vec2().unwrap();
    let ec: Vec<Vec<f32>> = m.e_cell.to_vec2().unwrap();

    let (bs, ids) = batches();
    // Hand-compute the mean |pair| over the same rows the loss builds.
    let mut acc = 0.0_f64;
    let mut n = 0_usize;
    for (cb, &gid) in bs.iter().zip(ids.iter()) {
        for b in 0..B {
            let u = cb.left_cells[b] as usize;
            let v = cb.right_cells[b] as usize;
            let s: f32 = (0..DIM)
                .map(|d| ef[gid as usize][d] * ec[u][d] * ec[v][d])
                .sum();
            acc += f64::from(s.abs());
            n += 1;
        }
    }
    let expected = (acc / n as f64) as f32;

    let out = cage_nce_loss_per_level(&m, bs, &ids, NceObjective::Logistic, &dev).unwrap();
    let got: f32 = out.mean_abs_pair.to_scalar().unwrap();
    assert!(
        (got - expected).abs() <= 1e-5 * expected.abs().max(1e-6),
        "pair term is not the raw e_feat score: got {got}, expected {expected} \
         — something is masking the gene rows"
    );
}

/// Gradient must reach the gene embedding.
#[test]
fn backward_reaches_the_gene_embedding() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let m = model(&varmap, &dev);
    let (bs, ids) = batches();

    let out = cage_nce_loss_per_level(&m, bs, &ids, NceObjective::Logistic, &dev).unwrap();
    let grads = out.per_level.sum_all().unwrap().backward().unwrap();

    let g = grads
        .get(&m.e_feat)
        .expect("no gradient reached the gene embedding");
    let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(mag > 0.0, "gradient reached e_feat but is all zero");
}

/// The installed `pip` reaches the score, and the per-epoch draw changes it.
///
/// Without this, cage could install a gate that `gathered_gate_weights` never
/// consulted and every other test here would still pass — which is exactly the
/// defect the forked loss had before it routed through geu's helper.
#[test]
fn installed_pip_gates_the_score_and_resampling_moves_it() {
    use graph_embedding_util::model::GateKind;
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let mut m = model(&varmap, &dev);

    let (bs, ids) = batches();
    let ungated: f32 = cage_nce_loss_per_level(&m, bs, &ids, NceObjective::Logistic, &dev)
        .unwrap()
        .mean_abs_pair
        .to_scalar()
        .unwrap();

    // Half the table off, half on: a draw must land strictly between.
    let mut p = vec![1.0_f32; N_GENES * DIM];
    for (i, v) in p.iter_mut().enumerate() {
        if i % DIM >= DIM / 2 {
            *v = 0.0;
        }
    }
    let pip = candle_util::candle_core::Tensor::from_vec(p, (N_GENES, DIM), &dev).unwrap();
    m.install_gate_pip(GateKind::Identity, &pip).unwrap();

    let (bs2, ids2) = batches();
    let gated: f32 = cage_nce_loss_per_level(&m, bs2, &ids2, NceObjective::Logistic, &dev)
        .unwrap()
        .mean_abs_pair
        .to_scalar()
        .unwrap();
    assert!(
        (gated - ungated).abs() > 1e-6,
        "installing pip did not change the score: {gated} vs {ungated}"
    );

    // A draw of a deterministic 0/1 pip reproduces it exactly, so the masked
    // score must match the frozen-pip score rather than drift.
    m.resample_gate_mask().unwrap();
    let (bs3, ids3) = batches();
    let drawn: f32 = cage_nce_loss_per_level(&m, bs3, &ids3, NceObjective::Logistic, &dev)
        .unwrap()
        .mean_abs_pair
        .to_scalar()
        .unwrap();
    assert!(
        (drawn - gated).abs() < 1e-5,
        "z ~ Bern(0/1) should reproduce the pip exactly: {drawn} vs {gated}"
    );

    m.clear_gate_mask();
}

/// `--nce-objective` must actually reach the loss. The two objectives differ in
/// how they normalize (`softmax` puts the positive and its negatives in one
/// distribution; `logistic` decides each pair alone), so on identical batches
/// they cannot agree — if they do, the argument is being dropped somewhere
/// between the flag and `loss.rs`.
#[test]
fn objective_selects_a_different_loss() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let m = model(&varmap, &dev);

    let mean_per_level = |obj| {
        let (bs, ids) = batches();
        cage_nce_loss_per_level(&m, bs, &ids, obj, &dev)
            .unwrap()
            .per_level
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };

    let logistic = mean_per_level(NceObjective::Logistic);
    let softmax = mean_per_level(NceObjective::Softmax);

    assert!(
        logistic.is_finite() && softmax.is_finite(),
        "both objectives must be finite: logistic {logistic}, softmax {softmax}"
    );
    assert!(
        (logistic - softmax).abs() > 1e-6,
        "objective did not reach the loss: logistic {logistic} == softmax {softmax}"
    );

    // Same objective twice is deterministic on a fixed model, so the difference
    // above is the objective and not batch noise.
    assert!((mean_per_level(NceObjective::Softmax) - softmax).abs() < 1e-6);
}
