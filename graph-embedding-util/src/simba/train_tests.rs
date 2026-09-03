use super::*;
use crate::simba::batch::PaddedBatch;
use crate::simba::graph::EdgeList;
use crate::simba::{run_simba, SimbaConfig};
use candle_util::candle_core::{Device, Tensor};
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Deterministic small tables: cell r, dim d → sin(r + 0.3 d), gene likewise
/// with a cosine, so every dot product is a distinct nonzero number.
fn tables(n_cells: usize, n_genes: usize, d: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let cells = (0..n_cells)
        .map(|r| (0..d).map(|j| (r as f64 + 0.3 * j as f64).sin()).collect())
        .collect();
    let genes = (0..n_genes)
        .map(|r| {
            (0..d)
                .map(|j| (0.7 * r as f64 - 0.2 * j as f64).cos())
                .collect()
        })
        .collect();
    (cells, genes)
}

fn model_from(cells: &[Vec<f64>], genes: &[Vec<f64>], c: usize, dev: &Device) -> SimbaModel {
    let flat = |t: &[Vec<f64>]| -> Vec<f32> { t.iter().flatten().map(|&v| v as f32).collect() };
    let d = cells[0].len();
    let ec = Tensor::from_vec(flat(cells), (cells.len(), d), dev).unwrap();
    let eg = Tensor::from_vec(flat(genes), (genes.len(), d), dev).unwrap();
    SimbaModel::from_tables(&ec, &eg, c).unwrap()
}

/// A batch of `n_real` edges `(i, i)` at chunk size `c` with `u` uniform
/// negatives per chunk taken from a fixed cycle, weight `w`.
fn batch(n_real: usize, c: usize, u: usize, w: f32, n_cells: usize, n_genes: usize) -> PaddedBatch {
    let k = n_real.div_ceil(c);
    let p = k * c;
    let mut b = PaddedBatch {
        k,
        c,
        u,
        n_real,
        lhs: vec![0; p],
        rhs: vec![0; p],
        row_w: vec![0.0; p],
        col_valid: vec![0.0; p],
        uni_lhs: (0..k * u).map(|i| ((3 * i + 1) % n_cells) as u32).collect(),
        uni_rhs: (0..k * u).map(|i| ((5 * i + 2) % n_genes) as u32).collect(),
    };
    for i in 0..n_real {
        b.lhs[i] = (i % n_cells) as u32;
        b.rhs[i] = (i % n_genes) as u32;
        b.row_w[i] = w;
        b.col_valid[i] = 1.0;
    }
    b
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// PBG's loss for one batch, chunk by chunk in f64.
fn reference_loss(cells: &[Vec<f64>], genes: &[Vec<f64>], b: &PaddedBatch) -> f64 {
    let mut total = 0.0;
    for q in 0..b.k {
        let rows: Vec<usize> = (q * b.c..(q + 1) * b.c).filter(|&i| i < b.n_real).collect();
        let uni_l: Vec<&Vec<f64>> = (0..b.u)
            .map(|j| &cells[b.uni_lhs[q * b.u + j] as usize])
            .collect();
        let uni_r: Vec<&Vec<f64>> = (0..b.u)
            .map(|j| &genes[b.uni_rhs[q * b.u + j] as usize])
            .collect();
        for &i in &rows {
            let l = &cells[b.lhs[i] as usize];
            let r = &genes[b.rhs[i] as usize];
            let pos = dot(l, r);
            // rhs corruption: the chunk's other genes + uniform genes
            let mut cand = vec![pos];
            cand.extend(
                rows.iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dot(l, &genes[b.rhs[j] as usize])),
            );
            cand.extend(uni_r.iter().map(|g| dot(l, g)));
            let l_rhs = lse(&cand) - pos;
            let mut cand = vec![pos];
            cand.extend(
                rows.iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dot(&cells[b.lhs[j] as usize], r)),
            );
            cand.extend(uni_l.iter().map(|cc| dot(cc, r)));
            let l_lhs = lse(&cand) - pos;
            total += f64::from(b.row_w[i]) * (l_lhs + l_rhs);
        }
    }
    total
}

#[test]
fn the_masked_batch_negative_block_never_scores_a_positive_against_itself_or_a_pad_column() {
    let dev = Device::Cpu;
    let (cells, genes) = tables(6, 6, 4);
    let c = 4;
    let model = model_from(&cells, &genes, c, &dev);
    let b = batch(3, c, 2, 1.0, 6, 6); // one chunk, rows 0..3 real, row 3 is a pad
    let s = model.score_blocks(&b, &dev).unwrap();
    let rhs = s.rhs_bat.to_vec3::<f32>().unwrap();
    let lhs = s.lhs_bat.to_vec3::<f32>().unwrap();
    assert_eq!(rhs.len(), 1);
    for i in 0..c {
        for j in 0..c {
            let masked = i == j || j == 3;
            let want_rhs = dot(&cells[b.lhs[i] as usize], &genes[b.rhs[j] as usize]);
            let want_lhs = dot(&cells[b.lhs[j] as usize], &genes[b.rhs[i] as usize]);
            if masked {
                assert!(
                    f64::from(rhs[0][i][j]) < -1e8,
                    "rhs [{i},{j}] must be masked"
                );
                assert!(
                    f64::from(lhs[0][i][j]) < -1e8,
                    "lhs [{i},{j}] must be masked"
                );
            } else {
                assert!(
                    approx(f64::from(rhs[0][i][j]), want_rhs, 1e-5),
                    "rhs [{i},{j}]"
                );
                assert!(
                    approx(f64::from(lhs[0][i][j]), want_lhs, 1e-5),
                    "lhs [{i},{j}]"
                );
            }
        }
    }
    let pos = s.pos.to_vec2::<f32>().unwrap();
    assert!(approx(
        f64::from(pos[0][1]),
        dot(&cells[1], &genes[1]),
        1e-5
    ));
    let ru = s.rhs_uni.expect("uniform block").to_vec3::<f32>().unwrap();
    assert!(approx(
        f64::from(ru[0][2][1]),
        dot(&cells[2], &genes[b.uni_rhs[1] as usize]),
        1e-5
    ));
}

#[test]
fn fused_batch_loss_equals_a_per_chunk_reference_computed_in_f64() {
    let dev = Device::Cpu;
    let (cells, genes) = tables(9, 11, 5);
    let c = 3;
    let model = model_from(&cells, &genes, c, &dev);
    let b = batch(7, c, 2, 2.33, 9, 11); // 3 chunks, 2 pad rows
    let got = f64::from(
        model
            .batch_loss(&b, &dev)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
    );
    let want = reference_loss(&cells, &genes, &b);
    assert!(
        approx(got, want, 1e-4 * want.abs().max(1.0)),
        "fused {got} vs reference {want}"
    );
}

#[test]
fn padding_a_short_group_does_not_change_the_loss_of_its_real_edges() {
    let dev = Device::Cpu;
    let (cells, genes) = tables(8, 8, 3);
    let wide = model_from(&cells, &genes, 50, &dev);
    let tight = model_from(&cells, &genes, 7, &dev);
    let b_wide = batch(7, 50, 0, 1.0, 8, 8);
    let b_tight = batch(7, 7, 0, 1.0, 8, 8);
    let a = f64::from(
        wide.batch_loss(&b_wide, &dev)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
    );
    let b = f64::from(
        tight
            .batch_loss(&b_tight, &dev)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
    );
    assert!(
        approx(a, b, 1e-4 * b.abs().max(1.0)),
        "padded {a} vs exact {b}"
    );
}

#[test]
fn weight_decay_term_is_the_summed_squared_norm_of_both_tables() {
    let dev = Device::Cpu;
    let (cells, genes) = tables(5, 4, 3);
    let model = model_from(&cells, &genes, 2, &dev);
    let got = f64::from(model.frob_sq().unwrap().to_scalar::<f32>().unwrap());
    let want: f64 = cells.iter().chain(&genes).flatten().map(|v| v * v).sum();
    assert!(approx(got, want, 1e-4));
}

/// 14 edges over two levels on 5 cells × 4 genes.
fn small_edges() -> EdgeList {
    let pairs = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 0),
        (4, 0),
        (4, 2),
        (0, 2),
        (1, 3),
        (2, 0),
        (3, 1),
    ];
    EdgeList {
        n_cells: 5,
        n_genes: 4,
        cell: pairs.iter().map(|p| p.0).collect(),
        gene: pairs.iter().map(|p| p.1).collect(),
        level: (0..pairs.len()).map(|i| 1 + (i % 2) as u8).collect(),
    }
}

#[test]
fn eval_edges_are_held_out_from_training_and_an_eval_loss_is_reported_every_epoch() {
    let cfg = SimbaConfig {
        dim: 4,
        epochs: 3,
        batch_size: 4,
        num_batch_negs: 2,
        num_uniform_negs: 2,
        wd: Some(0.0),
        eval_fraction: 0.25,
        seed: 5,
        ..SimbaConfig::default()
    };
    let out = train(small_edges(), &cfg).unwrap();
    assert_eq!(out.n_eval_edges, 3, "int(14 · 0.25)");
    assert_eq!(out.n_train_edges, 11);
    assert_eq!(out.epochs.len(), 3);
    assert!(out
        .epochs
        .iter()
        .all(|e| e.eval_loss.is_some_and(f64::is_finite)));
    assert!(out
        .epochs
        .iter()
        .all(|e| e.train_loss.is_finite() && e.train_loss > 0.0));
    assert_eq!(out.e_cell.dims(), &[5, 4]);
    assert_eq!(out.e_gene.dims(), &[4, 4]);
    assert_eq!(out.relations.len(), 2);
    assert_eq!(out.wd, 0.0);

    let none = train(
        small_edges(),
        &SimbaConfig {
            eval_fraction: 0.0,
            ..cfg.clone()
        },
    )
    .unwrap();
    assert_eq!(none.n_eval_edges, 0);
    assert!(none.epochs.iter().all(|e| e.eval_loss.is_none()));
    // auto weight decay is reported when not pinned
    let auto = train(
        small_edges(),
        &SimbaConfig {
            wd: None,
            epochs: 1,
            ..cfg.clone()
        },
    )
    .unwrap();
    assert!(approx(auto.wd, crate::simba::auto_wd(14), 0.0));
    // wd_interval 1 draws the decay on every batch: 11 train edges in
    // single-relation batches of 4 is at least 3 batches.
    let every = train(
        small_edges(),
        &SimbaConfig {
            wd: Some(1e-4),
            wd_interval: 1,
            epochs: 1,
            ..cfg.clone()
        },
    )
    .unwrap();
    let hits = every.epochs[0].wd_hits;
    assert!((3..=11).contains(&hits), "wd hits {hits}");
}

/// Two planted cell groups: cells 0..20 express genes 0..10, cells 20..40
/// express genes 10..20; the other block is sparse and low.
fn planted_backend() -> SparseIoVec {
    let (n_cells, n_genes) = (40usize, 20usize);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        let grp = usize::from(c >= 20);
        for g in 0..n_genes {
            let own = usize::from(g >= 10) == grp;
            let x = if own {
                3 + (c + g) % 4
            } else if (c * 7 + g) % 5 == 0 {
                1
            } else {
                0
            };
            if x > 0 {
                triplets.push((g as u64, c as u64, x as f32));
            }
        }
    }
    let shape = (n_genes, n_cells, triplets.len());
    let mut b = create_sparse_from_triplets(&triplets, shape, None, Some(&SparseIoBackend::Zarr))
        .expect("backend");
    b.register_row_names_vec(
        &(0..n_genes)
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

#[test]
fn training_on_two_planted_cell_groups_separates_them_and_scores_markers_on_their_own_group() {
    let data = planted_backend();
    let hvg: Vec<usize> = (0..20).collect();
    let cfg = SimbaConfig {
        dim: 8,
        epochs: 30,
        batch_size: 100,
        num_batch_negs: 10,
        num_uniform_negs: 10,
        wd: Some(0.0),
        eval_fraction: 0.0,
        seed: 1,
        ..SimbaConfig::default()
    };
    let out = run_simba(&data, &hvg, &cfg).unwrap();
    assert_eq!(out.n_edges, out.n_train_edges);
    assert_eq!(out.level_counts.iter().sum::<usize>(), out.n_edges);
    assert_eq!(out.level_counts.len(), out.relations.len());
    let ec = out.e_cell.to_vec2::<f32>().unwrap();
    let eg = out.e_gene.to_vec2::<f32>().unwrap();
    let cos = |a: &[f32], b: &[f32]| {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        d / (na * nb).max(1e-12)
    };
    let (mut within, mut across, mut nw, mut na) = (0.0f32, 0.0f32, 0, 0);
    for i in 0..40 {
        for j in (i + 1)..40 {
            let s = cos(&ec[i], &ec[j]);
            if (i < 20) == (j < 20) {
                within += s;
                nw += 1;
            } else {
                across += s;
                na += 1;
            }
        }
    }
    let (within, across) = (within / nw as f32, across / na as f32);
    assert!(
        within > across + 0.2,
        "within-group cosine {within} vs across {across}"
    );
    // Every marker scores its own group's cells above the other group's.
    let mean_dot = |g: usize, range: std::ops::Range<usize>| -> f32 {
        range
            .clone()
            .map(|c| ec[c].iter().zip(&eg[g]).map(|(x, y)| x * y).sum::<f32>())
            .sum::<f32>()
            / range.len() as f32
    };
    for g in 0..20 {
        let (own, other) = if g < 10 {
            (0..20, 20..40)
        } else {
            (20..40, 0..20)
        };
        assert!(
            mean_dot(g, own) > mean_dot(g, other),
            "gene {g} scores its own group higher"
        );
    }
    let last = out.epochs.last().unwrap().train_loss;
    let first = out.epochs.first().unwrap().train_loss;
    assert!(last < first, "loss falls over training: {first} → {last}");
    // SIMBA's fixed-T co-embed puts each marker nearer its own group's centroid.
    let co = crate::postprocess::feature_coembedding_fixed_t(&out.e_cell, &out.e_gene, 0.5)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let centroid = |range: std::ops::Range<usize>| -> Vec<f32> {
        (0..8)
            .map(|h| range.clone().map(|c| ec[c][h]).sum::<f32>() / range.len() as f32)
            .collect()
    };
    let (ca, cb) = (centroid(0..20), centroid(20..40));
    let dist = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
    for (g, row) in co.iter().enumerate().take(20) {
        let (own, other) = if g < 10 { (&ca, &cb) } else { (&cb, &ca) };
        assert!(
            dist(row, own) < dist(row, other),
            "gene {g} co-embeds with its own group"
        );
    }
}
