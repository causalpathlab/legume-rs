//! Stage 1 (which cells count, and their sums) must not depend on the
//! exposure labels: relabelling the individuals leaves the per-gene log mean
//! count, computed from the stage-1 sums, unchanged. A label-free stage 1 is
//! computed once and reused by every permutation draw.

use legume_numeric::matrix::traits::IoOps;
use std::io::{Read, Write};
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

fn column(file: &str, name: &str) -> Vec<f32> {
    let t = Mat::from_parquet(file).expect("read contrast table");
    let j = t
        .cols
        .iter()
        .position(|c| c.as_ref() == name)
        .expect("column");
    t.mat.column(j).iter().cloned().collect()
}

fn read_gz(path: &str) -> String {
    let bytes = std::fs::read(path).unwrap();
    let mut s = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut s)
        .unwrap();
    s
}

fn write_gz(path: &str, text: &str) {
    let f = std::fs::File::create(path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::default());
    gz.write_all(text.as_bytes()).unwrap();
    gz.finish().unwrap();
}

#[test]
fn relabelling_exposures_does_not_change_stage_one() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim");
    let sim = sim.to_str().unwrap();
    // exposure strongly shifts cell types, so cell types are unevenly shared
    cocoa(&[
        "simulate-collider",
        "-r",
        "80",
        "-c",
        "600",
        "-a",
        "8",
        "-t",
        "4",
        "-n",
        "2",
        "--n-samples-per-exposure",
        "4",
        "--n-covariates",
        "1",
        "--n-cell-covariates",
        "1",
        "--pve-exposure-celltype",
        "0.95",
        "--pve-cell-covar-celltype",
        "0.02",
        "--rseed",
        "11",
        "-o",
        sim,
    ]);

    // rotate the labels across individuals
    let rows: Vec<(String, String)> = read_gz(&format!("{sim}.exposures.gz"))
        .lines()
        .map(|l| {
            let mut f = l.split('\t');
            (f.next().unwrap().to_string(), f.next().unwrap().to_string())
        })
        .collect();
    let n = rows.len();
    let rotated: String = (0..n)
        .map(|i| format!("{}\t{}\n", rows[i].0, rows[(i + 1) % n].1))
        .collect();
    write_gz(&format!("{sim}.rotated.gz"), &rotated);

    let run = |exposures: &str, out: &str| {
        cocoa(&[
            "diff",
            &format!("{sim}.zarr.zip"),
            "-i",
            &format!("{sim}.samples.gz"),
            "-e",
            exposures,
            "-t",
            &format!("{sim}.celltypes.gz"),
            "--preload-data",
            "-o",
            out,
        ]);
    };
    run(&format!("{sim}.exposures.gz"), &format!("{sim}.a"));
    run(&format!("{sim}.rotated.gz"), &format!("{sim}.b"));

    let a = column(&format!("{sim}.a.contrast.parquet"), "log_mean");
    let b = column(&format!("{sim}.b.contrast.parquet"), "log_mean");
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-5,
        "stage 1 depends on the exposure labels: max |log mean difference| = {worst}"
    );
}
