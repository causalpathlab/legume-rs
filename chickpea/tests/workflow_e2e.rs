//! End-to-end workflow smoke test on synthetic pb matrices.

mod common;

use chickpea::common::Mat;
use chickpea::p2g::embed_ge::HierEmbedConfig;
use chickpea::p2g::link_map::LinkParams;
use chickpea::p2g::pb_levels::PbLevels;
use chickpea::p2g::workflow::*;
use common::{backend, barcodes, peak, tss};
use genomic_data::coordinates::{GeneTss, PeakCoord};
use legume_numeric::candle::candle_core::Device;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::path::Path;

struct Fixture {
    rna: Mat,
    atac: Mat,
    parent: Vec<usize>,
    gene_names: Vec<Box<str>>,
    peak_names: Vec<Box<str>>,
    gene_tss: Vec<Option<GeneTss>>,
    peak_coords: Vec<Option<PeakCoord>>,
}

/// Two genes and three peaks over 48 finest pbs in two smooth programs; the
/// third peak is noise. The pbs nest under 4 parents of 12.
fn fixture() -> Fixture {
    let s = 48usize;
    let mut rna = Mat::zeros(2, s);
    let mut atac = Mat::zeros(3, s);
    for j in 0..s {
        let z = (j as f32) * 0.15;
        let sig0 = z.sin().abs() + 0.1;
        let sig1 = z.cos().abs() + 0.1;
        rna[(0, j)] = sig0 * 10.0;
        rna[(1, j)] = sig1 * 10.0;
        atac[(0, j)] = sig0 * 8.0;
        atac[(1, j)] = sig1 * 8.0;
        atac[(2, j)] = ((j % 5) as f32) * 0.2;
    }
    Fixture {
        rna,
        atac,
        parent: (0..s).map(|j| j / 12).collect(),
        gene_names: vec!["G0".into(), "G1".into()],
        peak_names: vec![
            "chr1:100000-100500".into(),
            "chr1:200000-200500".into(),
            "chr1:150000-150500".into(),
        ],
        gene_tss: vec![tss(100_000), tss(200_000)],
        peak_coords: vec![peak(100_000), peak(200_000), peak(150_000)],
    }
}

/// A coarse level whose columns are the sums of their children.
fn sum_children(m: &Mat, parent: &[usize]) -> Mat {
    let n_parents = parent.iter().max().map_or(0, |&p| p + 1);
    let mut out = Mat::zeros(m.nrows(), n_parents);
    for (j, &p) in parent.iter().enumerate() {
        for r in 0..m.nrows() {
            out[(r, p)] += m[(r, j)];
        }
    }
    out
}

fn params() -> WorkflowParams {
    WorkflowParams {
        abc: LinkParams {
            cis_window: 500_000,
            max_cis: 50,
            min_weight: 0.1,
            ..LinkParams::default()
        },
        embed: HierEmbedConfig {
            dim: 8,
            epochs: 2,
            seed: 3,
            device: Device::Cpu,
            n_gene_modules: 4,
            n_peak_modules: 4,
            units_per_step: 16,
            modules_per_unit: 2,
            lr: 0.1,
            merge_every: 0,
            merge_cosine: 0.95,
        },
        min_cluster_samples: 5,
        target_clusters: Some(2),
    }
}

fn assert_outputs(out: &str) {
    assert!(Path::new(out).join("peaks.parquet").is_file());
    assert!(Path::new(out).join("clusters.parquet").is_file());
    assert!(
        Path::new(out)
            .join("peak_gene")
            .join("chr1.parquet")
            .is_file(),
        "expected peak_gene/chr1.parquet under {out}"
    );
    // Senna-style peak / gene / finest-pb embedding parquets (prefix = out dir path).
    for suffix in [
        "peak_embedding",
        "gene_embedding",
        "pb_embedding",
        "pb_tree_embedding",
    ] {
        assert!(
            Path::new(&format!("{out}.{suffix}.parquet")).is_file(),
            "expected {out}.{suffix}.parquet"
        );
    }
}

/// Two cells per finest pb: the even cell gets the floor half of each count,
/// the odd cell the rest, so cells sum back to their pb column.
fn split_cells(pb: &Mat) -> (Mat, Vec<usize>) {
    let n = pb.ncols();
    let mut cells = Mat::zeros(pb.nrows(), 2 * n);
    for j in 0..n {
        for r in 0..pb.nrows() {
            let v = pb[(r, j)];
            let lo = (v / 2.0).floor();
            cells[(r, 2 * j)] = lo;
            cells[(r, 2 * j + 1)] = v - lo;
        }
    }
    (cells, (0..2 * n).map(|c| c / 2).collect())
}

fn parquet_rows(path: &str) -> usize {
    let file = std::fs::File::open(path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    reader.metadata().file_metadata().num_rows() as usize
}

#[test]
fn e2e_with_cells_writes_the_cell_embedding_and_clusters_cells() {
    let f = fixture();
    let levels = PbLevels {
        rna: Some(vec![f.rna.clone(), sum_children(&f.rna, &f.parent)]),
        atac: vec![f.atac.clone(), sum_children(&f.atac, &f.parent)],
        parent: vec![f.parent.clone()],
    };
    let (rna_cells, cell_to_pb) = split_cells(&f.rna);
    let (atac_cells, _) = split_cells(&f.atac);
    let (rb, ab) = (backend(&rna_cells), backend(&atac_cells));
    let cells = CellInputs {
        backends: vec![&rb, &ab],
        barcodes: barcodes(rna_cells.ncols()),
        cell_to_pb: &cell_to_pb,
    };
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().to_string_lossy().into_owned();
    run_from_pseudobulk(
        &PbMultiome {
            levels: &levels,
            gene_activity: None,
            gene_tss: &f.gene_tss,
            peak_coords: &f.peak_coords,
            gene_names: &f.gene_names,
            peak_names: &f.peak_names,
        },
        &cells,
        &out,
        &params(),
    )
    .unwrap();
    assert_outputs(&out);
    let path = format!("{out}.cell_embedding.parquet");
    assert!(Path::new(&path).is_file(), "expected {path}");
    assert_eq!(parquet_rows(&path), rna_cells.ncols());
}

/// ATAC-only with cells: the surrogate gene axis has no per-cell counts, so
/// the cells are projected on the peak axis alone and the parquet still lands.
#[test]
fn e2e_atac_only_with_cells_projects_on_the_peak_axis_alone() {
    let f = fixture();
    let levels = PbLevels {
        rna: None,
        atac: vec![f.atac.clone(), sum_children(&f.atac, &f.parent)],
        parent: vec![f.parent.clone()],
    };
    let (atac_cells, cell_to_pb) = split_cells(&f.atac);
    let ab = backend(&atac_cells);
    let cells = CellInputs {
        backends: vec![&ab],
        barcodes: barcodes(atac_cells.ncols()),
        cell_to_pb: &cell_to_pb,
    };
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().to_string_lossy().into_owned();
    run_from_pseudobulk(
        &PbMultiome {
            levels: &levels,
            gene_activity: Some(&f.rna),
            gene_tss: &f.gene_tss,
            peak_coords: &f.peak_coords,
            gene_names: &f.gene_names,
            peak_names: &f.peak_names,
        },
        &cells,
        &out,
        &params(),
    )
    .unwrap();
    assert_outputs(&out);
    let path = format!("{out}.cell_embedding.parquet");
    assert_eq!(parquet_rows(&path), atac_cells.ncols());
}
