//! What a coarse read must do, pinned before the encoder is rewired.
//!
//! The load-bearing test is the first one: at the identity grouping the coarse
//! path has to reproduce the per-gene gather-and-sum exactly. If it does, the
//! new encoder is the old one with a grouping applied, and every later
//! difference is the grouping rather than a rewrite.

use super::{coarse_profile, pool_groups};
use crate::decoder::coarsening_map::CoarseningMap;
use crate::fast_index::gather_rows;
use candle_core::{DType, Device, Tensor};

const D: usize = 6;
const H: usize = 3;
const C: usize = 3;

fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32).unwrap().to_vec2().unwrap()
}

/// Genes 0,1 -> group 0; genes 2,3 -> group 1; genes 4,5 -> group 2.
fn grouping(dev: &Device) -> CoarseningMap {
    let fine_to_coarse = [0usize, 0, 1, 1, 2, 2];
    let share = [0.5f32; D];
    CoarseningMap::new(&fine_to_coarse, &share, dev).unwrap()
}

fn group_table(dev: &Device) -> Tensor {
    let v: Vec<f32> = (0..C * H).map(|i| (i as f32 * 0.37).sin()).collect();
    Tensor::from_vec(v, (C, H), dev).unwrap()
}

/// Two cells, two slots each, both slots visible.
fn context(dev: &Device) -> (Tensor, Tensor, Tensor) {
    let indices = Tensor::from_vec(vec![0u32, 2, 3, 5], (2, 2), dev).unwrap();
    let gate = Tensor::from_vec(vec![1.5f32, 2.0, 0.5, 3.0], (2, 2), dev).unwrap();
    let visible = Tensor::from_vec(vec![1.0f32; 4], (2, 2), dev).unwrap();
    (indices, gate, visible)
}

/// THE equivalence test. With one gene per group the grouping is the identity,
/// so pooling the profile against a `[D, H]` table must equal gathering each
/// slot's row and adding it in, gated. Anything else means the new path is not
/// the old one with a grouping applied.
#[test]
fn the_identity_grouping_reproduces_the_per_slot_gather_and_sum() {
    let dev = Device::Cpu;
    let map = CoarseningMap::identity(D, &dev).unwrap();
    let table = {
        let v: Vec<f32> = (0..D * H).map(|i| (i as f32 * 0.21).cos()).collect();
        Tensor::from_vec(v, (D, H), &dev).unwrap()
    };
    let (indices, gate, visible) = context(&dev);

    let profile = coarse_profile(&indices, &gate, &visible, &map).unwrap();
    let got = to_vec2(&pool_groups(&profile, &table).unwrap());

    // The per-gene path, written out: gather, gate, sum over slots.
    let rows = gather_rows(&table, &indices.flatten_all().unwrap())
        .unwrap()
        .reshape((2, 2, H))
        .unwrap();
    let want = to_vec2(
        &rows
            .broadcast_mul(&gate.unsqueeze(2).unwrap())
            .unwrap()
            .sum(1)
            .unwrap(),
    );

    for (n, (a, b)) in got.iter().zip(&want).enumerate() {
        for (h, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "cell {n} dim {h}: coarse {x} vs per-gene {y}"
            );
        }
    }
}

/// A hidden slot must reach neither the profile nor the pool, or the encoder
/// reads the very count the decoder is asked to impute.
#[test]
fn a_masked_slot_contributes_to_no_group() {
    let dev = Device::Cpu;
    let map = grouping(&dev);
    let (indices, gate, _) = context(&dev);
    // Cell 0 hides its second slot (gene 2, group 1); cell 1 hides both.
    let visible = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (2, 2), &dev).unwrap();

    let profile = to_vec2(&coarse_profile(&indices, &gate, &visible, &map).unwrap());
    assert_eq!(profile[0].len(), C);
    // Cell 0: only gene 0 survives, in group 0, carrying its gate.
    assert!((profile[0][0] - 1.5).abs() < 1e-6, "{:?}", profile[0]);
    assert_eq!(
        profile[0][1], 0.0,
        "the hidden slot's group must stay empty"
    );
    assert_eq!(profile[0][2], 0.0);
    // Cell 1 saw nothing.
    assert!(
        profile[1].iter().all(|&x| x == 0.0),
        "a fully masked cell must read as empty: {:?}",
        profile[1]
    );
}

/// A fully masked cell must still produce a defined row rather than a NaN,
/// because a high mask fraction makes such cells ordinary.
#[test]
fn a_cell_with_no_visible_slot_pools_to_a_finite_row() {
    let dev = Device::Cpu;
    let map = grouping(&dev);
    let (indices, gate, _) = context(&dev);
    let visible = Tensor::zeros((2, 2), DType::F32, &dev).unwrap();

    let pooled = pool_groups(
        &coarse_profile(&indices, &gate, &visible, &map).unwrap(),
        &group_table(&dev),
    )
    .unwrap();
    assert_eq!(pooled.dims(), &[2, H]);
    for row in to_vec2(&pooled) {
        for v in row {
            assert!(v.is_finite(), "non-finite pooled value {v}");
            assert_eq!(v, 0.0, "an empty read must pool to zero, not to a bias");
        }
    }
}

/// Two slots of one cell landing in the same group must add, not overwrite.
/// This is what makes a group's level a sum over its observed members.
#[test]
fn slots_sharing_a_group_accumulate() {
    let dev = Device::Cpu;
    let map = grouping(&dev);
    // Both slots of cell 0 are genes of group 0.
    let indices = Tensor::from_vec(vec![0u32, 1, 4, 5], (2, 2), &dev).unwrap();
    let gate = Tensor::from_vec(vec![1.5f32, 2.5, 1.0, 1.0], (2, 2), &dev).unwrap();
    let visible = Tensor::from_vec(vec![1.0f32; 4], (2, 2), &dev).unwrap();

    let profile = to_vec2(&coarse_profile(&indices, &gate, &visible, &map).unwrap());
    assert!(
        (profile[0][0] - 4.0).abs() < 1e-6,
        "1.5 + 2.5: {:?}",
        profile[0]
    );
    assert_eq!(profile[0][1], 0.0);
    assert!((profile[1][2] - 2.0).abs() < 1e-6, "{:?}", profile[1]);
}

/// The grouping is a function on genes, so no gated mass is lost or double
/// counted: the profile's row total is the visible gated total.
#[test]
fn every_visible_gene_lands_in_exactly_one_group() {
    let dev = Device::Cpu;
    let map = grouping(&dev);
    let (indices, gate, visible) = context(&dev);

    let profile = to_vec2(&coarse_profile(&indices, &gate, &visible, &map).unwrap());
    let gates = to_vec2(&gate);
    for (n, row) in profile.iter().enumerate() {
        let got: f32 = row.iter().sum();
        let want: f32 = gates[n].iter().sum();
        assert!((got - want).abs() < 1e-5, "cell {n}: {got} vs {want}");
    }
}

/// One gene observed at unit gate must read as exactly its group's row. This is
/// the property that makes a gained gene usable with no new parameter: assign
/// it to a group and it inherits that group's trained vector.
#[test]
fn a_single_gene_reads_as_its_groups_row() {
    let dev = Device::Cpu;
    let map = grouping(&dev);
    let table = group_table(&dev);
    // One cell, one slot: gene 3, which grouping() puts in group 1.
    let indices = Tensor::from_vec(vec![3u32], (1, 1), &dev).unwrap();
    let gate = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();
    let visible = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();

    let pooled = to_vec2(
        &pool_groups(
            &coarse_profile(&indices, &gate, &visible, &map).unwrap(),
            &table,
        )
        .unwrap(),
    );
    let want = to_vec2(&table);
    for (h, (x, y)) in pooled[0].iter().zip(&want[1]).enumerate() {
        assert!((x - y).abs() < 1e-6, "dim {h}: {x} vs group 1's {y}");
    }
}

/// Two genes of the SAME group are indistinguishable to the encoder. That is
/// the cost of the design and it should be asserted rather than discovered:
/// whatever the encoder learns about one member it applies to the other.
#[test]
fn genes_of_one_group_are_indistinguishable_to_the_encoder() {
    let dev = Device::Cpu;
    let map = grouping(&dev);
    let table = group_table(&dev);
    let one = |g: u32| {
        let indices = Tensor::from_vec(vec![g], (1, 1), &dev).unwrap();
        let gate = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();
        let visible = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();
        to_vec2(
            &pool_groups(
                &coarse_profile(&indices, &gate, &visible, &map).unwrap(),
                &table,
            )
            .unwrap(),
        )
    };
    // Genes 2 and 3 share group 1.
    assert_eq!(one(2), one(3), "same group must give the same read");
    // Genes 2 and 4 do not.
    assert_ne!(one(2), one(4), "different groups must differ");
}
