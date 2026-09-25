//! Stage-1 pseudobulks must pool individuals: a pseudobulk that holds too
//! few individuals cannot separate its cell-state rate from their
//! multipliers, and is dropped. The run reports per topic how many
//! pseudobulks and cells were kept, and how many individuals each kept
//! pseudobulk links.

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
fn pseudobulks_pool_individuals_and_the_run_reports_it() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim");
    let sim = sim.to_str().unwrap();
    cocoa(&[
        "simulate-one",
        "-r",
        "100",
        "-c",
        "2400",
        "-a",
        "10",
        "-n",
        "2",
        "--n-samples-per-exposure",
        "6",
        "--rseed",
        "4",
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
        "-o",
        &format!("{sim}.out"),
    ]);

    let t = Mat::from_parquet(&format!("{sim}.out.stage1.parquet")).expect("stage-1 report");
    let median = column(&t, "median_individuals_per_pseudobulk")[0];
    let (kept, dropped) = (column(&t, "cells_kept")[0], column(&t, "cells_dropped")[0]);
    assert!(
        median >= 3.0,
        "median individuals per kept pseudobulk = {median}"
    );
    assert!(
        dropped < 0.5 * (kept + dropped),
        "dropped {dropped} of {} cells",
        kept + dropped
    );
}
