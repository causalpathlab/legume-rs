//! Genes whose exposure effect cannot be estimated (no counts, or an
//! exposure level whose individuals all have zero counts) are left out:
//! NA contrast and NA p-value, never a number. Every other gene is tested.

use legume_numeric::matrix::traits::IoOps;
use std::io::Read;
use std::process::Command;

type Mat = nalgebra::DMatrix<f32>;

fn cocoa(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_cocoa"))
        .args(args)
        .output()
        .expect("run cocoa");
    assert!(
        out.status.success(),
        "cocoa {:?} failed:\n{}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn column(t: &legume_numeric::matrix::traits::MatWithNames<Mat>, name: &str) -> Vec<f32> {
    let j = t
        .cols
        .iter()
        .position(|c| c.as_ref() == name)
        .unwrap_or_else(|| panic!("no column {name}: {:?}", t.cols));
    t.mat.column(j).iter().cloned().collect()
}

#[test]
fn sparse_genes_get_na_and_the_rest_are_tested() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim");
    let sim = sim.to_str().unwrap();
    // few cells per individual: many genes see no counts in one arm
    cocoa(&[
        "simulate-one",
        "-r",
        "200",
        "-c",
        "240",
        "-a",
        "10",
        "-n",
        "2",
        "--n-samples-per-exposure",
        "4",
        "--gene-mean-sd",
        "2",
        "--rseed",
        "3",
        "-o",
        sim,
    ]);
    let bytes = std::fs::read(format!("{sim}.samples.gz")).unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .unwrap();
    std::fs::write(
        format!("{sim}.topic.txt"),
        text.lines().map(|_| "0\n").collect::<String>(),
    )
    .unwrap();

    cocoa(&[
        "diff",
        &format!("{sim}.zarr.zip"),
        "-i",
        &format!("{sim}.samples.gz"),
        "-e",
        &format!("{sim}.exposures.gz"),
        "-t",
        &format!("{sim}.topic.txt"),
        "--preload-data",
        "--n-permutations",
        "30",
        "-o",
        &format!("{sim}.out"),
    ]);

    let t = Mat::from_parquet(&format!("{sim}.out.perm.parquet")).expect("perm table");
    let (contrast, pvalue) = (column(&t, "contrast"), column(&t, "pvalue"));
    let left_out = contrast.iter().filter(|c| c.is_nan()).count();
    assert!(left_out > 0, "the sparse design should leave some genes out");
    assert!(left_out < contrast.len(), "every gene was left out");
    for (c, p) in contrast.iter().zip(&pvalue) {
        assert_eq!(c.is_nan(), p.is_nan(), "contrast {c} with p-value {p}");
        if !p.is_nan() {
            assert!((0.0..=1.0).contains(p), "p-value {p}");
        }
    }
}
