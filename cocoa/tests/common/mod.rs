//! A small deterministic dataset for the end-to-end tests, written directly
//! (no simulator): cell states with marker genes, individual multipliers,
//! two exposure levels alternating over individuals, and two genes whose
//! effect cannot be estimated (gene 0 has no counts, gene 1 has none at
//! level 1).
#![allow(dead_code)]

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use legume_numeric::matrix::traits::IoOps;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use std::io::{Read, Write};
use std::process::Command;

pub type Mat = nalgebra::DMatrix<f32>;

/// Genes with a planted non-estimable effect.
pub const GENE_NO_COUNTS: usize = 0;
pub const GENE_ZERO_AT_LEVEL_1: usize = 1;

pub struct Spec {
    pub n_genes: usize,
    pub n_indv: usize,
    pub n_states: usize,
    pub cells_per_indv: usize,
    /// exposure moves cells between states (level 1 into state 0)
    pub exposure_shifts_states: bool,
}

pub struct Fixture {
    _dir: tempfile::TempDir,
    pub prefix: String,
}

impl Fixture {
    pub fn data(&self) -> String {
        format!("{}.zarr", self.prefix)
    }
    pub fn samples(&self) -> String {
        format!("{}.samples.gz", self.prefix)
    }
    pub fn exposures(&self) -> String {
        format!("{}.exposures.gz", self.prefix)
    }
    /// every cell in one topic
    pub fn one_topic(&self) -> String {
        format!("{}.topic.txt", self.prefix)
    }
    /// cell states as topics
    pub fn states(&self) -> String {
        format!("{}.states.gz", self.prefix)
    }
    /// one covariate per individual
    pub fn covariates(&self) -> String {
        format!("{}.conf.tsv.gz", self.prefix)
    }
    pub fn out(&self, tag: &str) -> String {
        format!("{}.{}", self.prefix, tag)
    }
}

pub fn build(spec: &Spec) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("fx").to_str().unwrap().to_string();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);

    let level = |i: usize| i % 2;
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    let (mut samples, mut states) = (String::new(), String::new());
    let mut cell = 0u64;
    for i in 0..spec.n_indv {
        let x = level(i);
        for c in 0..spec.cells_per_indv {
            let s = if spec.exposure_shifts_states && x == 1 && c % 2 == 0 {
                0
            } else {
                c % spec.n_states
            };
            for g in 0..spec.n_genes {
                if g == GENE_NO_COUNTS || (g == GENE_ZERO_AT_LEVEL_1 && x == 1) {
                    continue;
                }
                let marker = if g % spec.n_states == s { 4.0 } else { 1.0 };
                let indv = 1.0 + 0.1 * ((i * 7 + g * 3) % 5) as f32;
                let effect = if g % 10 == 2 && x == 1 { 2.0 } else { 1.0 };
                let y = Poisson::new(marker * indv * effect)
                    .unwrap()
                    .sample(&mut rng);
                if y > 0.0 {
                    triplets.push((g as u64, cell, y));
                }
            }
            samples.push_str(&format!("{i}\n"));
            states.push_str(&format!("{s}\n"));
            cell += 1;
        }
    }
    let n_cells = cell as usize;

    let mut data = create_sparse_from_triplets(
        &triplets,
        (spec.n_genes, n_cells, triplets.len()),
        Some(&format!("{prefix}.zarr")),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    data.register_row_names_vec(
        &(0..spec.n_genes)
            .map(|g| format!("GENE{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    data.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    drop(data);

    write_gz(&format!("{prefix}.samples.gz"), &samples);
    write_gz(&format!("{prefix}.states.gz"), &states);
    std::fs::write(
        format!("{prefix}.topic.txt"),
        samples.lines().map(|_| "0\n").collect::<String>(),
    )
    .unwrap();
    let exposures: String = (0..spec.n_indv)
        .map(|i| format!("{i}\t{}\n", level(i)))
        .collect();
    write_gz(&format!("{prefix}.exposures.gz"), &exposures);
    let conf: String = (0..spec.n_indv)
        .map(|i| {
            format!(
                "{}\n",
                ((i * 5) % spec.n_indv) as f32 / spec.n_indv as f32 - 0.5
            )
        })
        .collect();
    write_gz(&format!("{prefix}.conf.tsv.gz"), &conf);

    Fixture { _dir: dir, prefix }
}

pub fn cocoa(args: &[&str]) {
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

/// `cocoa diff` on the fixture with the given topic file and extra args.
pub fn diff(fx: &Fixture, topics: &str, exposures: &str, extra: &[&str], out: &str) {
    let data = fx.data();
    let samples = fx.samples();
    let mut args = vec![
        "diff",
        &data,
        "-i",
        &samples,
        "-e",
        exposures,
        "-t",
        topics,
        "--preload-data",
    ];
    args.extend_from_slice(extra);
    args.extend(["-o", out]);
    cocoa(&args);
}

pub fn column(file: &str, name: &str) -> Vec<f32> {
    let t = Mat::from_parquet(file).unwrap_or_else(|e| panic!("{file}: {e}"));
    let j = t
        .cols
        .iter()
        .position(|c| c.as_ref() == name)
        .unwrap_or_else(|| panic!("no column {name}: {:?}", t.cols));
    t.mat.column(j).iter().cloned().collect()
}

pub fn read_gz(file: &str) -> String {
    let bytes = std::fs::read(file).unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .unwrap();
    text
}

pub fn write_gz(file: &str, text: &str) {
    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(file).unwrap(),
        flate2::Compression::default(),
    );
    enc.write_all(text.as_bytes()).unwrap();
    enc.finish().unwrap();
}
