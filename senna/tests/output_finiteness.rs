//! A diverged run must not be able to write an artifact that looks healthy.
//!
//! Regression for the `masked-vae` failure where `{out}.latent.parquet` was
//! written with the correct shape, row names and columns but every value
//! `NaN` — the divergence only surfaced downstream, in R.

use senna::embed_common::Mat;
use senna::output_helpers::{save_dictionary, save_latent, save_pb_gene};

fn names(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

fn tmp_prefix(tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!("senna_finite_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("run").to_string_lossy().into_owned()
}

#[test]
fn save_latent_rejects_nan_and_writes_clean() {
    let out = tmp_prefix("latent");
    let cells = names("cell", 4);

    let mut latent = Mat::from_element(4, 3, 0.5f32);
    save_latent(&out, &latent, &cells, None).expect("finite latent should write");
    assert!(std::path::Path::new(&format!("{out}.latent.parquet")).exists());

    // A single NaN is enough — the real failure was all-NaN, but a partially
    // corrupt latent is just as unusable downstream.
    latent[(2, 1)] = f32::NAN;
    let err = save_latent(&out, &latent, &cells, None)
        .expect_err("NaN latent must not reach disk")
        .to_string();
    assert!(err.contains("non-finite"), "unexpected message: {err}");
    assert!(
        err.contains("latent.parquet"),
        "should name the path: {err}"
    );

    latent[(2, 1)] = f32::INFINITY;
    assert!(save_latent(&out, &latent, &cells, None).is_err(), "Inf too");
}

#[test]
fn dictionary_and_pb_gene_reject_nan() {
    let out = tmp_prefix("dict");
    let genes = names("gene", 5);

    let mut dict = Mat::from_element(5, 2, 0.1f32);
    save_dictionary(&out, &dict, &genes).expect("finite dictionary should write");
    dict[(0, 0)] = f32::NAN;
    assert!(save_dictionary(&out, &dict, &genes).is_err());

    let mut pb = Mat::from_element(5, 3, 1.0f32);
    save_pb_gene(&out, &pb, &genes).expect("finite pb_gene should write");
    pb[(4, 2)] = f32::NAN;
    assert!(save_pb_gene(&out, &pb, &genes).is_err());
}
