//! The per-axis cell projection through the public API: the driver over
//! groups, and the group streamer over sparse backends.

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use graph_embedding_util::fit::projection::{
    project_cells_axes, stream_cell_groups, AxesProjector, AxisDict, CellGroup,
};
use legume_numeric::candle::candle_core::Device;

fn dictionary(n_feat: usize, h: usize, scale: f32, shift: usize) -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0f32; n_feat * h];
    let mut b = vec![0f32; n_feat];
    for f in 0..n_feat {
        for k in 0..h {
            e[f * h + k] = (((((f + shift) * 7 + k * 13) % 11) as f32 / 11.0) - 0.5) * scale;
        }
        b[f] = ((((f + shift) * 5) % 7) as f32 / 7.0) - 0.3;
    }
    (e, b)
}

fn rates(e: &[f32], b: &[f32], h: usize, theta: &[f32], c: f32) -> (Vec<u32>, Vec<f32>) {
    let mut feats = Vec::new();
    let mut counts = Vec::new();
    for f in 0..b.len() {
        let ef = &e[f * h..(f + 1) * h];
        let s: f32 = ef.iter().zip(theta).map(|(a, t)| a * t).sum::<f32>() + b[f] + c;
        feats.push(f as u32);
        counts.push(s.exp());
    }
    (feats, counts)
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// Twelve planted cells in three programs over two axes.
struct Fixture {
    e0: Vec<f32>,
    b0: Vec<f32>,
    e1: Vec<f32>,
    b1: Vec<f32>,
    thetas: Vec<Vec<f32>>,
}

fn fixture(h: usize) -> Fixture {
    let (e0, b0) = dictionary(120, h, 0.5, 0);
    let (e1, b1) = dictionary(90, h, 0.5, 4);
    let programs = [
        vec![0.8f32, -0.6, 0.4, 0.2],
        vec![-0.5, 0.7, -0.2, 0.6],
        vec![0.2, 0.1, -0.7, -0.3],
    ];
    let thetas: Vec<Vec<f32>> = (0..12).map(|i| programs[i % 3].clone()).collect();
    Fixture {
        e0,
        b0,
        e1,
        b1,
        thetas,
    }
}

impl Fixture {
    fn dicts(&self) -> [AxisDict<'_>; 2] {
        [
            AxisDict {
                label: "a0",
                feat: &self.e0,
                b_feat: &self.b0,
            },
            AxisDict {
                label: "a1",
                feat: &self.e1,
                b_feat: &self.b1,
            },
        ]
    }

    fn groups(&self, h: usize, size: usize) -> Vec<CellGroup> {
        (0..self.thetas.len())
            .step_by(size)
            .map(|start| {
                let end = (start + size).min(self.thetas.len());
                CellGroup {
                    cells: (start as u32..end as u32).collect(),
                    axes: vec![
                        (start..end)
                            .map(|i| {
                                rates(
                                    &self.e0,
                                    &self.b0,
                                    h,
                                    &self.thetas[i],
                                    0.3 + 0.05 * i as f32,
                                )
                            })
                            .collect(),
                        (start..end)
                            .map(|i| {
                                rates(
                                    &self.e1,
                                    &self.b1,
                                    h,
                                    &self.thetas[i],
                                    1.4 - 0.05 * i as f32,
                                )
                            })
                            .collect(),
                    ],
                }
            })
            .collect()
    }
}

#[test]
fn the_gauge_mean_is_removed_across_groups_and_returned() {
    let h = 4;
    let f = fixture(h);
    let p = AxesProjector::new(&f.dicts(), h, 1e-3, &Device::Cpu).unwrap();
    let n = f.thetas.len();
    let out = project_cells_axes(&p, n, f.groups(h, 5).into_iter().map(Ok)).unwrap();
    assert_eq!(out.theta.len(), n * h);
    assert_eq!(out.intercepts.len(), 2);
    for k in 0..h {
        let mean: f32 = (0..n).map(|i| out.theta[i * h + k]).sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-4, "dim {k} mean {mean}");
    }
    for (i, want) in f.thetas.iter().enumerate() {
        let got: Vec<f32> = (0..h)
            .map(|k| out.theta[i * h + k] + out.theta_mean[k])
            .collect();
        assert!(cos(&got, want) > 0.97, "cell {i}");
    }
}

#[test]
fn group_size_does_not_change_the_answer() {
    let h = 4;
    let f = fixture(h);
    let p = AxesProjector::new(&f.dicts(), h, 1e-3, &Device::Cpu).unwrap();
    let one = project_cells_axes(&p, 12, f.groups(h, 12).into_iter().map(Ok)).unwrap();
    let many = project_cells_axes(&p, 12, f.groups(h, 5).into_iter().map(Ok)).unwrap();
    for (a, b) in one.theta.iter().zip(&many.theta) {
        assert!((a - b).abs() < 2e-2, "{a} vs {b}");
    }
}

#[test]
fn a_group_with_a_cell_id_past_n_cells_is_refused() {
    let h = 4;
    let f = fixture(h);
    let p = AxesProjector::new(&f.dicts(), h, 1e-3, &Device::Cpu).unwrap();
    assert!(project_cells_axes(&p, 6, f.groups(h, 12).into_iter().map(Ok)).is_err());
}

/// A backend of `rows × cols` with `value(r, c)` where nonzero.
fn backend(rows: usize, cols: usize, value: impl Fn(usize, usize) -> f32) -> SparseIoVec {
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = value(r, c);
            if v != 0.0 {
                triplets.push((r as u64, c as u64, v));
            }
        }
    }
    let shape = (rows, cols, triplets.len());
    let mut b = create_sparse_from_triplets(&triplets, shape, None, Some(&SparseIoBackend::Zarr))
        .expect("backend");
    b.register_row_names_vec(
        &(0..rows)
            .map(|r| format!("r{r}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..cols)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

#[test]
fn streamed_groups_reproduce_the_backends_columns() {
    let n_cells = 7;
    let rna = backend(5, n_cells, |r, c| {
        if (r + c).is_multiple_of(3) {
            (r * 10 + c) as f32 + 1.0
        } else {
            0.0
        }
    });
    let atac = backend(4, n_cells, |r, c| if (r * c) % 2 == 1 { 2.0 } else { 0.0 });
    let backends = [&rna, &atac];
    let groups: Vec<CellGroup> = stream_cell_groups(&backends, 3)
        .unwrap()
        .collect::<anyhow::Result<_>>()
        .unwrap();
    assert_eq!(
        groups.iter().map(|g| g.cells.len()).collect::<Vec<_>>(),
        vec![3, 3, 1]
    );
    let mut seen = 0u32;
    for g in &groups {
        assert_eq!(g.axes.len(), 2);
        for (i, &c) in g.cells.iter().enumerate() {
            assert_eq!(c, seen);
            seen += 1;
            let (f, n) = &g.axes[0][i];
            let want: Vec<(u32, f32)> = (0..5u32)
                .filter(|&r| (r as usize + c as usize).is_multiple_of(3))
                .map(|r| (r, (r as usize * 10 + c as usize) as f32 + 1.0))
                .collect();
            let got: Vec<(u32, f32)> = f.iter().copied().zip(n.iter().copied()).collect();
            assert_eq!(got, want, "cell {c} rna");
            let (f, _) = &g.axes[1][i];
            let want: Vec<u32> = (0..4u32)
                .filter(|&r| (r as usize * c as usize) % 2 == 1)
                .collect();
            assert_eq!(f, &want, "cell {c} atac");
        }
    }
}

#[test]
fn backends_with_different_column_counts_are_refused() {
    let a = backend(3, 4, |r, c| (r + c) as f32);
    let b = backend(3, 5, |r, c| (r + c) as f32);
    assert!(stream_cell_groups(&[&a, &b], 2).is_err());
}
