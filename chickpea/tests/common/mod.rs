//! Fixtures shared by the integration tests.
#![allow(dead_code)]

use chickpea::common::Mat;
use genomic_data::coordinates::{GeneTss, PeakCoord};

/// A gene TSS on chromosome 1.
pub fn tss(pos: i64) -> Option<GeneTss> {
    Some(GeneTss {
        chr: "1".into(),
        tss: pos,
    })
}

/// A 500-bp peak on chromosome 1.
pub fn peak(start: i64) -> Option<PeakCoord> {
    Some(PeakCoord {
        chr: "1".into(),
        start,
        end: start + 500,
    })
}

/// A smooth positive signal over `s` samples: `max(sin(0.1 j), 0) + 0.05`.
pub fn sine_signal(s: usize) -> Vec<f32> {
    (0..s)
        .map(|j| ((j as f32) * 0.1).sin().max(0.0) + 0.05)
        .collect()
}

/// A `rows × cols` matrix with `f(row, col)` entries.
pub fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Mat {
    Mat::from_fn(rows, cols, f)
}
