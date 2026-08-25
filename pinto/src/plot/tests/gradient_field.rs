//! The gradient field's contracts: per-point binning and coherence from
//! hand-built vectors, the NaN-honesty of the value channel, the
//! count-weighted smoothing, and the coherence floor plus occupancy rule
//! that keep field lines evenly spaced and off incoherent ground.
//!
//! Geometry used throughout: a 100 x 100 frame at `target_bins = 4`
//! (also the clamp floor) gives a 4 x 4 grid of 25-px bins; bin (ix, iy)
//! covers `[25*ix, 25*(ix+1)) x [25*iy, 25*(iy+1))`.

use crate::plot::gradient_field::{build_gradient_field, field_lines, FieldLineArgs};
use plot_utils::rasterize::Extent;

const EXT: Extent = Extent { w: 100, h: 100 };

fn right() -> (f32, f32) {
    (1.0, 0.0)
}
fn left() -> (f32, f32) {
    (-1.0, 0.0)
}

/// Aligned cells in bin (0,1), opposed cells in bin (2,1): coherence
/// separates them, and the mean direction and value read from the bin.
#[test]
fn coherence_is_one_for_aligned_and_zero_for_opposed_cells() {
    let pts = vec![(10.0, 30.0), (15.0, 45.0), (60.0, 30.0), (65.0, 45.0)];
    let vecs = vec![right(), right(), right(), left()];
    let vals = vec![1.0f32, 3.0, 5.0, 7.0];
    let f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);

    let ((dx, dy), coh, val, c) = f.sample((12.0, 37.0)).expect("bin (0,1) populated");
    assert!((coh - 1.0).abs() < 1e-6, "aligned cells cohere, got {coh}");
    assert!(dx > 0.0 && dy.abs() < 1e-6, "mean direction points right");
    assert!((val - 2.0).abs() < 1e-6, "value mean over the bin's cells");
    assert!((c - 2.0).abs() < 1e-6, "two cells of weight");

    let (_, coh, _, _) = f.sample((62.0, 37.0)).expect("bin (2,1) populated");
    assert!(coh.abs() < 1e-6, "opposed cells cancel, got {coh}");
}

/// A NaN contribution must not dilute the value mean, and a bin whose
/// contributions are ALL NaN reports NaN, not a fabricated neutral 0.
#[test]
fn nan_values_neither_dilute_nor_fabricate() {
    let pts = vec![(10.0, 30.0), (15.0, 45.0), (60.0, 30.0), (65.0, 45.0)];
    let vecs = vec![right(), right(), right(), right()];
    let vals = vec![2.0f32, f32::NAN, f32::NAN, f32::NAN];
    let f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);

    let (_, _, val, _) = f.sample((12.0, 37.0)).unwrap();
    assert!(
        (val - 2.0).abs() < 1e-6,
        "one finite value of 2.0 must mean 2.0, not 1.0, got {val}"
    );
    let ((dx, _), _, val, _) = f.sample((62.0, 37.0)).unwrap();
    assert!(dx > 0.0, "direction is still measured");
    assert!(val.is_nan(), "no finite value ever arrived: NaN, not 0");
}

/// Smoothing spreads measured direction into neighbouring bins with its
/// weight attached, so an empty bin borrows direction WITH provenance
/// (weight < 1 cell), while a bin of cancelling cells stays incoherent:
/// the blur must not manufacture agreement out of disagreement.
#[test]
fn smoothing_borrows_with_weight_but_does_not_invent_coherence() {
    let pts = vec![
        (12.0, 37.0), // bin (0,1), rightward
        (87.0, 30.0), // bin (3,1), opposed pair, far from the borrower
        (87.0, 45.0),
    ];
    let vecs = vec![right(), right(), left()];
    let vals = vec![1.0f32, 1.0, 1.0];
    let mut f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);
    f.smooth(1);

    // The empty neighbour (1,1) borrowed the (0,1) cell's direction...
    let ((dx, _), coh, _, c) = f.sample((37.0, 37.0)).expect("borrowed weight");
    assert!(dx > 0.0 && coh > 0.9, "borrowed direction stays coherent");
    assert!(c < 1.0, "...but at sub-cell weight, so floors can see it");

    // ...while the opposed bin remains incoherent after smoothing.
    let (_, coh, _, _) = f.sample((87.0, 37.0)).unwrap();
    assert!(coh < 0.35, "smoothing must not invent coherence, got {coh}");
}

/// Field lines neither seed in nor continue into bins below the
/// coherence floor: where the cells disagree, the figure draws nothing.
/// (Bilinear sampling lets a line taper partway toward the incoherent
/// bin centre, so the assertion allows the interpolation zone.)
#[test]
fn field_lines_stop_at_the_coherence_floor() {
    // Columns 0-1 (x < 50): rightward cells everywhere. Columns 2-3:
    // opposed pairs, coherence 0.
    let mut pts = Vec::new();
    let mut vecs = Vec::new();
    for iy in 0..4 {
        let y = 25.0 * iy as f32 + 12.0;
        for ix in 0..2 {
            pts.push((25.0 * ix as f32 + 12.0, y));
            vecs.push(right());
        }
        for ix in 2..4 {
            pts.push((25.0 * ix as f32 + 8.0, y));
            vecs.push(right());
            pts.push((25.0 * ix as f32 + 16.0, y));
            vecs.push(left());
        }
    }
    let vals = vec![1.0f32; pts.len()];
    let f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);
    let fl = field_lines(
        &f,
        &FieldLineArgs {
            density: 1.0,
            min_length: 0.05,
            max_length: 4.0,
            min_coherence: 0.35,
            min_count: 0.5,
        },
    );
    assert!(!fl.lines.is_empty(), "coherent ground must carry lines");
    for line in &fl.lines {
        for &(x, _) in line {
            assert!(
                x <= 58.0,
                "a line reached deep into the incoherent half: x = {x}"
            );
        }
    }
}

/// The streamplot discard rule: a coherent island too small to carry a
/// line of `min_length` produces NO line at all (and releases its mask
/// cells), rather than littering the figure with stubs.
#[test]
fn short_lines_are_discarded_whole() {
    // One coherent bin only; everything else is empty.
    let pts = vec![(12.0, 37.0), (16.0, 40.0)];
    let vecs = vec![right(), right()];
    let vals = vec![1.0f32; 2];
    let f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);

    let strict = FieldLineArgs {
        density: 1.0,
        min_length: 1.0,
        max_length: 4.0,
        min_coherence: 0.35,
        min_count: 0.5,
    };
    let none = field_lines(&f, &strict);
    assert!(none.lines.is_empty(), "an island cannot carry a 1.0 line");

    let lenient = FieldLineArgs {
        min_length: 0.05,
        ..strict
    };
    let some = field_lines(&f, &lenient);
    assert!(
        !some.lines.is_empty(),
        "the same island passes a 0.05 floor"
    );
}

/// On a uniformly rightward field the mask spaces the lines: they are
/// few relative to the seed count, and long (spanning most of the
/// frame) rather than a pileup of fragments.
#[test]
fn mask_spacing_yields_few_long_lines() {
    let mut pts = Vec::new();
    for iy in 0..4 {
        for ix in 0..4 {
            pts.push((25.0 * ix as f32 + 12.0, 25.0 * iy as f32 + 12.0));
        }
    }
    let vecs = vec![right(); pts.len()];
    let vals = vec![1.0f32; pts.len()];
    let f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);
    // Explicit density so the mask geometry the assertion counts against
    // is pinned by the TEST, not by whatever the default happens to be.
    let fl = field_lines(
        &f,
        &FieldLineArgs {
            density: 1.0,
            ..FieldLineArgs::default()
        },
    );
    assert!(!fl.lines.is_empty());
    assert!(
        fl.lines.len() <= 34,
        "one line per mask row at most (30 rows + margin), got {}",
        fl.lines.len()
    );
    let longest = fl
        .lines
        .iter()
        .map(|l| {
            l.windows(2)
                .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
                .sum::<f32>()
        })
        .fold(0.0f32, f32::max);
    assert!(
        longest > 55.0,
        "lines should run most of the frame, longest = {longest}"
    );
}

/// One arrow per populated bin that clears both floors, none elsewhere.
#[test]
fn gridded_arrows_come_from_coherent_populated_bins_only() {
    let pts = vec![(10.0, 30.0), (15.0, 45.0), (60.0, 30.0), (65.0, 45.0)];
    let vecs = vec![right(), right(), right(), left()];
    let vals = vec![1.0f32; 4];
    let f = build_gradient_field(&pts, &vecs, &vals, EXT, 4);
    let (arrows, arrow_vals) = f.gridded_arrows(0.35, 1.0);
    assert_eq!(arrows.len(), 1, "only the coherent bin draws");
    assert_eq!(arrow_vals.len(), 1);
    let ((x0, _), (x1, _)) = arrows[0];
    assert!(x0 < 50.0 && x1 < 50.0, "and it is the left one");
}
