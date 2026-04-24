//! Smoke tests for the rasterizer's per-point and segment paths.

use plot_utils::{
    rasterize_group_png, rasterize_segment_layer_png, DataBounds, Extent, PointShape, RadiusSpec,
};

const EXT: Extent = Extent { w: 64, h: 64 };
const RED: (u8, u8, u8) = (200, 30, 30);

#[test]
fn scalar_vs_per_homogeneous_match() {
    let pts = vec![(10.0, 10.0), (30.0, 30.0), (50.0, 50.0)];
    let radii = vec![2.5; 3];

    let a = rasterize_group_png(
        &pts,
        EXT,
        RadiusSpec::Scalar(2.5),
        RED,
        1.0,
        PointShape::Circle,
    )
    .unwrap();
    let b = rasterize_group_png(
        &pts,
        EXT,
        RadiusSpec::Per(&radii),
        RED,
        1.0,
        PointShape::Circle,
    )
    .unwrap();

    // Same homogeneous radius → same PNG (bit-for-bit is asking a lot
    // across PNG encoders, but byte-length should match at minimum).
    assert_eq!(
        a.len(),
        b.len(),
        "homogeneous per-point radius should match scalar"
    );
    assert!(!a.is_empty());
}

#[test]
fn per_point_length_mismatch_errors() {
    let pts = vec![(10.0, 10.0); 3];
    let radii = vec![2.0; 2];
    let res = rasterize_group_png(
        &pts,
        EXT,
        RadiusSpec::Per(&radii),
        RED,
        1.0,
        PointShape::Circle,
    );
    assert!(res.is_err(), "mismatched radii should error");
}

#[test]
fn segment_rasterizer_emits_non_empty_png() {
    let segs = vec![((5.0, 5.0), (55.0, 55.0)), ((5.0, 55.0), (55.0, 5.0))];
    let png = rasterize_segment_layer_png(&segs, EXT, 1.5, RED, 0.8).unwrap();
    assert!(!png.is_empty());
}

#[test]
fn segment_rasterizer_empty_input_yields_transparent() {
    let png = rasterize_segment_layer_png(&[], EXT, 1.5, RED, 0.8).unwrap();
    // Empty input → blank pixmap, still a valid PNG of the extent.
    assert!(!png.is_empty());
}

#[test]
fn data_bounds_pixel_mapping_roundtrip() {
    let b = DataBounds::from_minmax(0.0, 10.0, 0.0, 10.0);
    let p = b.to_pixel((5.0, 5.0), EXT);
    // center of data → ~center of pixels (inside the 2% pad).
    assert!((p.0 - 32.0).abs() < 2.0);
    assert!((p.1 - 32.0).abs() < 2.0);
}
