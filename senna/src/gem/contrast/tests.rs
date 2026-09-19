use super::{contrast_rows, write_contrast_table};
use senna::embed_common::Mat;
use crate::gem::tracks::assign_tracks;
use matrix_util::parquet::read_parquet_string_columns_by_name;
use matrix_util::traits::IoOps;

fn names(rows: &[&str]) -> Vec<Box<str>> {
    rows.iter().map(|&s| s.into()).collect()
}

/// GENE1 carries both channels of `count` and both channels of `m6a`; GENE2
/// carries only `count/spliced` (no `unspliced`, no `m6a` rows at all), so it
/// must be skipped on every modality.
fn fixture() -> (crate::gem::tracks::TrackPlan, Vec<Box<str>>, Mat, Vec<f32>) {
    let axis = names(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
        "GENE1/m6a/methylated",
        "GENE1/m6a/unmethylated",
    ]);
    let plan = assign_tracks(&axis).expect("assign_tracks");
    // rows: 0 count/spliced(G1), 1 count/unspliced(G1), 2 count/spliced(G2),
    // 3 m6a/methylated(G1), 4 m6a/unmethylated(G1) — 2 columns (H).
    let rho = Mat::from_row_slice(
        5,
        2,
        &[
            1.0, 2.0, // row 0
            4.0, 6.0, // row 1
            10.0, 20.0, // row 2
            0.5, 1.5, // row 3
            2.5, 0.5, // row 4
        ],
    );
    let b_feat = vec![0.1, 0.3, 0.9, -0.2, 0.4];
    (plan, axis, rho, b_feat)
}

#[test]
fn contrasts_only_genes_with_both_channels_present() {
    let (plan, axis, rho, b_feat) = fixture();
    let table = contrast_rows(&plan, &axis, &rho, &b_feat);

    assert_eq!(
        table.rows,
        names(&["GENE1/count", "GENE1/m6a"]),
        "GENE2 has no unspliced or m6a row on either channel: skipped on both modalities"
    );
    assert_eq!(table.modality, names(&["count", "m6a"]));
    assert_eq!(table.gene, names(&["GENE1", "GENE1"]));

    // count: unspliced(row1) - spliced(row0) = [4-1, 6-2] = [3, 4]
    assert_eq!(
        table.delta.row(0).iter().copied().collect::<Vec<f32>>(),
        [3.0, 4.0]
    );
    // m6a: methylated(row3) - unmethylated(row4) = [0.5-2.5, 1.5-0.5] = [-2, 1]
    assert_eq!(
        table.delta.row(1).iter().copied().collect::<Vec<f32>>(),
        [-2.0, 1.0]
    );

    // count bias: b_feat[1] - b_feat[0] = 0.3 - 0.1
    assert!((table.bias[0] - 0.2).abs() < 1e-6, "{}", table.bias[0]);
    // m6a bias: b_feat[3] - b_feat[4] = -0.2 - 0.4
    assert!((table.bias[1] - (-0.6)).abs() < 1e-6, "{}", table.bias[1]);
}

#[test]
fn write_then_read_back_round_trips_the_string_and_numeric_columns() {
    let (plan, axis, rho, b_feat) = fixture();
    let table = contrast_rows(&plan, &axis, &rho, &b_feat);

    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    write_contrast_table(&prefix, &table).expect("write_contrast_table");

    // feature_contrast.parquet: key `feature`, then `modality`/`gene` (Str,
    // skipped by the numeric reader), then h0..h{H-1} (F32).
    let delta_path = format!("{prefix}.feature_contrast.parquet");
    let read = Mat::from_parquet_with_row_names(&delta_path, Some(0)).expect("read delta back");
    assert_eq!(read.rows, table.rows);
    assert_eq!(read.cols, vec![Box::from("h0"), Box::from("h1")]);
    assert_eq!(read.mat.nrows(), 2);
    assert!((read.mat[(0, 0)] - 3.0).abs() < 1e-5);
    assert!((read.mat[(0, 1)] - 4.0).abs() < 1e-5);
    assert!((read.mat[(1, 0)] - (-2.0)).abs() < 1e-5);
    assert!((read.mat[(1, 1)] - 1.0).abs() < 1e-5);

    let cols = read_parquet_string_columns_by_name(&delta_path, &["feature", "modality", "gene"])
        .expect("read string columns");
    assert_eq!(cols[0], table.rows);
    assert_eq!(cols[1], names(&["count", "m6a"]));
    assert_eq!(cols[2], names(&["GENE1", "GENE1"]));

    // feature_contrast_bias.parquet: key `feature`, `modality`, `gene`, `bias`.
    let bias_path = format!("{prefix}.feature_contrast_bias.parquet");
    let bias_cols =
        read_parquet_string_columns_by_name(&bias_path, &["feature", "modality", "gene"])
            .expect("read bias string columns");
    assert_eq!(bias_cols[0], table.rows);
    assert_eq!(bias_cols[1], names(&["count", "m6a"]));
    assert_eq!(bias_cols[2], names(&["GENE1", "GENE1"]));
    let bias_mat =
        Mat::from_parquet_with_row_names(&bias_path, Some(0)).expect("read bias numeric back");
    assert_eq!(bias_mat.cols, vec![Box::from("bias")]);
    assert!((bias_mat.mat[(0, 0)] - 0.2).abs() < 1e-5);
    assert!((bias_mat.mat[(1, 0)] - (-0.6)).abs() < 1e-5);
}
