use super::*;
use crate::simba::graph::EdgeList;
use crate::simba::{run_simba, SimbaConfig};
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
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
    // `int(7 · 0.25)` per level, the hold-out being per relation.
    assert_eq!(out.n_eval_edges, 2);
    assert_eq!(out.n_train_edges, 12);
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
    // wd_interval 1 draws the decay on every batch: 12 train edges in
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
    assert!((3..=12).contains(&hits), "wd hits {hits}");
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
