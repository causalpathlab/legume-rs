//! The cell-side corruption of [`NceCorruption::Both`]: what it adds to the
//! feature-side loss, and that a row never competes with itself.

use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::VarMap;
use graph_embedding_util::loss::{
    nce_loss_identity, softmax_nce, EdgeBatch, NceCorruption, NceObjective,
};
use graph_embedding_util::model::{JointEmbedModel, ModelArgs, ModelInit};

const D: usize = 6;
const H: usize = 3;
const N: usize = 4;

fn model(vm: &VarMap, dev: &Device) -> JointEmbedModel {
    let e_feat = nalgebra::DMatrix::<f32>::from_fn(D, H, |f, k| ((f * 3 + k * 5) % 7) as f32 * 0.2 - 0.6);
    let e_cell = nalgebra::DMatrix::<f32>::from_fn(N, H, |c, k| ((c * 5 + k * 2) % 5) as f32 * 0.3 - 0.5);
    let b_feat: Vec<f32> = (0..D).map(|f| f as f32 * 0.1).collect();
    let b_cell: Vec<f32> = (0..N).map(|c| c as f32 * 0.05).collect();
    JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: D,
            n_cells: N,
            embedding_dim: H,
            seed: 1,
        },
        &ModelInit {
            e_feat: Some(&e_feat),
            e_cell: Some(&e_cell),
            b_feat: &b_feat,
            b_cell: &b_cell,
        },
        vm,
        dev,
    )
    .unwrap()
}

fn scalar(t: &Tensor) -> f32 {
    t.to_scalar::<f32>().unwrap()
}

fn batch() -> EdgeBatch {
    // Cell 1 is drawn twice (positives 1 and 3): those two rows must not see
    // each other as negatives.
    EdgeBatch {
        coarse_cells: vec![0, 1, 2, 1],
        fine_feats: vec![2, 0, 5, 3],
        neg_feats: vec![1, 4, 3, 5, 0, 1, 2, 4],
        n_negatives: 2,
    }
}

/// `Both` = `Feature` + the mean per-edge cell-side softmax term, computed
/// here by hand from the model's tables with the same-cell columns removed.
#[test]
fn both_adds_the_hand_computed_cell_side_term() {
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let m = model(&vm, &dev);
    let feat_only = scalar(&nce_loss_identity(&m, batch(), NceObjective::Softmax, NceCorruption::Feature, &dev).unwrap());
    let both = scalar(&nce_loss_identity(&m, batch(), NceObjective::Softmax, NceCorruption::Both, &dev).unwrap());

    let e_feat: Vec<f32> = m.e_feat.flatten_all().unwrap().to_vec1().unwrap();
    let e_cell: Vec<f32> = m.e_cell.flatten_all().unwrap().to_vec1().unwrap();
    let b_feat: Vec<f32> = m.b_feat.to_vec1().unwrap();
    let b_cell: Vec<f32> = m.b_cell.to_vec1().unwrap();
    let score = |f: usize, c: usize| -> f32 {
        (0..H).map(|k| e_feat[f * H + k] * e_cell[c * H + k]).sum::<f32>() + b_feat[f] + b_cell[c]
    };
    let b = batch();
    let mut cell_term = 0f32;
    for (i, (&ci, &fi)) in b.coarse_cells.iter().zip(&b.fine_feats).enumerate() {
        let pos = score(fi as usize, ci as usize);
        let negs: Vec<f32> = b
            .coarse_cells
            .iter()
            .enumerate()
            .filter(|&(j, &cj)| j != i && cj != ci)
            .map(|(_, &cj)| score(fi as usize, cj as usize))
            .collect();
        let pos_t = Tensor::from_vec(vec![pos], 1, &dev).unwrap();
        let neg_t = Tensor::from_vec(negs.clone(), (1, negs.len()), &dev).unwrap();
        cell_term += scalar(&softmax_nce(&pos_t, &[neg_t]).unwrap().squeeze(0).unwrap());
    }
    cell_term /= b.coarse_cells.len() as f32;
    assert!(cell_term > 0.0);
    assert!(
        (both - feat_only - cell_term).abs() < 1e-4,
        "both {both} − feature {feat_only} = {} ≠ hand-computed cell term {cell_term}",
        both - feat_only
    );
}

/// With every positive on the SAME cell there is no other cell to compete
/// with. Under the softmax objective the cell side then adds exactly nothing;
/// under the logistic one it adds the positive term once more — each side
/// carries its own `−log σ(pos)`, as in PBG — and nothing else.
#[test]
fn a_single_cell_batch_has_no_cell_negatives() {
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let m = model(&vm, &dev);
    let same = || EdgeBatch {
        coarse_cells: vec![2, 2, 2],
        fine_feats: vec![0, 3, 5],
        neg_feats: vec![1, 2, 4, 0, 1, 3],
        n_negatives: 2,
    };
    let softmax_a = scalar(&nce_loss_identity(&m, same(), NceObjective::Softmax, NceCorruption::Feature, &dev).unwrap());
    let softmax_b = scalar(&nce_loss_identity(&m, same(), NceObjective::Softmax, NceCorruption::Both, &dev).unwrap());
    assert!((softmax_a - softmax_b).abs() < 1e-5, "softmax: {softmax_a} vs {softmax_b}");

    let e_feat: Vec<f32> = m.e_feat.flatten_all().unwrap().to_vec1().unwrap();
    let e_cell: Vec<f32> = m.e_cell.flatten_all().unwrap().to_vec1().unwrap();
    let b_feat: Vec<f32> = m.b_feat.to_vec1().unwrap();
    let b_cell: Vec<f32> = m.b_cell.to_vec1().unwrap();
    let b = same();
    let pos_term: f32 = b
        .coarse_cells
        .iter()
        .zip(&b.fine_feats)
        .map(|(&c, &f)| {
            let s = (0..H).map(|k| e_feat[f as usize * H + k] * e_cell[c as usize * H + k]).sum::<f32>()
                + b_feat[f as usize]
                + b_cell[c as usize];
            (1.0 + (-s).exp()).ln() // −log σ(s)
        })
        .sum::<f32>()
        / b.coarse_cells.len() as f32;
    let logistic_a = scalar(&nce_loss_identity(&m, same(), NceObjective::Logistic, NceCorruption::Feature, &dev).unwrap());
    let logistic_b = scalar(&nce_loss_identity(&m, same(), NceObjective::Logistic, NceCorruption::Both, &dev).unwrap());
    assert!(
        (logistic_b - logistic_a - pos_term).abs() < 1e-4,
        "logistic: {logistic_b} − {logistic_a} ≠ {pos_term}"
    );
}
