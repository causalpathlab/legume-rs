//! Stage 1 (pseudobulks and the cell-state baseline) must depend on cell
//! state only: individual-level confounders enter stage 2 alone. So the
//! unadjusted contrast of a run with a covariate file must equal the
//! contrast of the same run without one.

use legume_numeric::matrix::traits::IoOps;
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

/// Column `name` of a `*.contrast.parquet` table.
fn column(file: &str, name: &str) -> Vec<f32> {
    let t = Mat::from_parquet(file).expect("read contrast table");
    let j = t
        .cols
        .iter()
        .position(|c| c.as_ref() == name)
        .unwrap_or_else(|| panic!("no column {name} in {file}: {:?}", t.cols));
    t.mat.column(j).iter().cloned().collect()
}

#[test]
fn covariate_file_does_not_change_stage_one() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim");
    let sim = sim.to_str().unwrap();
    cocoa(&[
        "simulate-one",
        "-r",
        "120",
        "-c",
        "1200",
        "-a",
        "12",
        "-n",
        "2",
        "--n-samples-per-exposure",
        "6",
        "--n-covariates",
        "2",
        "--pve-covar-exposure",
        "0.6",
        "--pve-covar-gene",
        "0.3",
        "--rseed",
        "3",
        "-o",
        sim,
    ]);
    // one topic
    let samples = std::fs::read(format!("{sim}.samples.gz")).unwrap();
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&samples[..]), &mut text)
        .unwrap();
    let topic: String = text.lines().map(|_| "0\n").collect();
    std::fs::write(format!("{sim}.topic.txt"), topic).unwrap();

    let common = [
        format!("{sim}.zarr.zip"),
        "-i".into(),
        format!("{sim}.samples.gz"),
        "-e".into(),
        format!("{sim}.exposures.gz"),
        "-t".into(),
        format!("{sim}.topic.txt"),
        "--preload-data".into(),
    ];
    let run = |extra: &[String], out: &str| {
        let mut a: Vec<&str> = vec!["diff"];
        a.extend(common.iter().map(|s| s.as_str()));
        a.extend(extra.iter().map(|s| s.as_str()));
        a.extend(["-o", out]);
        cocoa(&a);
    };
    let plain = format!("{sim}.plain");
    let adjusted = format!("{sim}.adj");
    run(&[], &plain);
    run(
        &["--covariate-file".into(), format!("{sim}.conf.tsv.gz")],
        &adjusted,
    );

    let a = column(&format!("{plain}.contrast.parquet"), "contrast");
    let b = column(
        &format!("{adjusted}.contrast.parquet"),
        "contrast_unadjusted",
    );
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "the covariate file changed stage 1: max |contrast difference| = {worst}"
    );
}
