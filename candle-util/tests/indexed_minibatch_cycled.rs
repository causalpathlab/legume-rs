// The probe in `train_masked` measures memory on a `minibatch_cycled`
// batch because training bootstrap-pads every batch to the full
// minibatch size: a probe batch truncated to the row count would
// under-measure the per-row cost. These tests pin the cycling contract.

use candle_util::candle_core::Device;
use candle_util::data::indexed::{IndexedInMemoryArgs, IndexedInMemoryData};
use nalgebra::DMatrix;

fn small_loader() -> IndexedInMemoryData {
    let data = DMatrix::<f32>::from_row_slice(
        4,
        6,
        &[
            0.1, 0.5, 0.3, 0.9, 0.2, 0.7, //
            0.8, 0.1, 0.6, 0.2, 0.9, 0.3, //
            0.3, 0.7, 0.1, 0.4, 0.6, 0.5, //
            0.2, 0.3, 0.8, 0.1, 0.5, 0.9, //
        ],
    );
    let w = vec![1.0f32; 6];
    IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
        input: &data,
        input_null: None,
        input_context_size: 3,
        input_shortlist_weights: &w,
        input_mean: None,
    })
    .unwrap()
}

#[test]
fn cycled_batch_has_exactly_n_rows_beyond_the_data() {
    let loader = small_loader();
    let mb = loader.minibatch_cycled(10, &Device::Cpu).unwrap();
    assert_eq!(mb.input_values.dims()[0], 10);
}

#[test]
fn cycled_rows_repeat_the_data_in_order() {
    let loader = small_loader();
    let mb = loader.minibatch_cycled(10, &Device::Cpu).unwrap();
    let vals = mb.input_values.to_vec2::<f32>().unwrap();
    // row i must equal row (i % 4): same values in the same slots
    for i in 0..10 {
        assert_eq!(vals[i], vals[i % 4], "row {i} != row {}", i % 4);
    }
}

#[test]
fn cycled_row_ids_repeat_the_data_in_order() {
    let loader = small_loader();
    let mb = loader.minibatch_cycled(10, &Device::Cpu).unwrap();
    let ids: Vec<u32> = mb.row_ids.to_vec1().unwrap();
    assert_eq!(ids, (0..10).map(|i| (i % 4) as u32).collect::<Vec<_>>());
}

#[test]
fn cycled_within_data_matches_ordered() {
    let loader = small_loader();
    let cycled = loader.minibatch_cycled(4, &Device::Cpu).unwrap();
    let ordered = loader.minibatch_ordered(0, 4, &Device::Cpu).unwrap();
    assert_eq!(
        cycled.input_values.to_vec2::<f32>().unwrap(),
        ordered.input_values.to_vec2::<f32>().unwrap()
    );
}
