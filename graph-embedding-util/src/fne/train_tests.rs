use super::*;
use crate::fne::batch::PaddedBatch;
use crate::fne::graph::{NodeTypeTable, Relation, RelationTable, TypedEdgeList};
use crate::fne::FneConfig;
use candle_util::candle_core::{Device, Tensor};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Deterministic small tables: lhs r, dim d → sin(r + 0.3 d), rhs likewise
/// with a cosine, so every dot product is a distinct nonzero number.
fn tables(n_lhs: usize, n_rhs: usize, d: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let lhs = (0..n_lhs)
        .map(|r| (0..d).map(|j| (r as f64 + 0.3 * j as f64).sin()).collect())
        .collect();
    let rhs = (0..n_rhs)
        .map(|r| {
            (0..d)
                .map(|j| (0.7 * r as f64 - 0.2 * j as f64).cos())
                .collect()
        })
        .collect();
    (lhs, rhs)
}

fn to_tensor(t: &[Vec<f64>], dev: &Device) -> Tensor {
    let d = t[0].len();
    let flat: Vec<f32> = t.iter().flatten().map(|&v| v as f32).collect();
    Tensor::from_vec(flat, (t.len(), d), dev).unwrap()
}

/// Two node types stacked: lhs rows `0..n_lhs`, rhs rows after them.
fn model_from(lhs: &[Vec<f64>], rhs: &[Vec<f64>], c: usize, dev: &Device) -> FneModel {
    FneModel::from_type_tables(&[to_tensor(lhs, dev), to_tensor(rhs, dev)], c).unwrap()
}

/// A batch of `n_real` edges `(i, off + i)` at chunk size `c` with `u`
/// uniform negatives per chunk taken from a fixed cycle inside each type,
/// per-row weights `w[i]`.
fn batch(n_real: usize, c: usize, u: usize, w: &[f32], n_lhs: usize, n_rhs: usize) -> PaddedBatch {
    let k = n_real.div_ceil(c);
    let p = k * c;
    let off = n_lhs as u32;
    let mut b = PaddedBatch {
        k,
        c,
        u,
        n_real,
        rel: 0,
        lhs: vec![0; p],
        rhs: vec![0; p],
        row_w: vec![0.0; p],
        col_valid: vec![0.0; p],
        uni_lhs: (0..k * u).map(|i| ((3 * i + 1) % n_lhs) as u32).collect(),
        uni_rhs: (0..k * u)
            .map(|i| off + ((5 * i + 2) % n_rhs) as u32)
            .collect(),
    };
    for i in 0..n_real {
        b.lhs[i] = (i % n_lhs) as u32;
        b.rhs[i] = off + (i % n_rhs) as u32;
        b.row_w[i] = w[i % w.len()];
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

/// PBG's loss for one batch, chunk by chunk in f64, on the stacked table.
fn reference_loss(lhs_t: &[Vec<f64>], rhs_t: &[Vec<f64>], b: &PaddedBatch) -> f64 {
    let off = lhs_t.len();
    let row = |g: u32| -> &Vec<f64> {
        let g = g as usize;
        if g < off {
            &lhs_t[g]
        } else {
            &rhs_t[g - off]
        }
    };
    let mut total = 0.0;
    for q in 0..b.k {
        let rows: Vec<usize> = (q * b.c..(q + 1) * b.c).filter(|&i| i < b.n_real).collect();
        let uni_l: Vec<&Vec<f64>> = (0..b.u).map(|j| row(b.uni_lhs[q * b.u + j])).collect();
        let uni_r: Vec<&Vec<f64>> = (0..b.u).map(|j| row(b.uni_rhs[q * b.u + j])).collect();
        for &i in &rows {
            let l = row(b.lhs[i]);
            let r = row(b.rhs[i]);
            let pos = dot(l, r);
            let mut cand = vec![pos];
            cand.extend(
                rows.iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dot(l, row(b.rhs[j]))),
            );
            cand.extend(uni_r.iter().map(|g| dot(l, g)));
            let l_rhs = lse(&cand) - pos;
            let mut cand = vec![pos];
            cand.extend(
                rows.iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dot(row(b.lhs[j]), r)),
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
    let (lhs, rhs) = tables(6, 6, 4);
    let c = 4;
    let model = model_from(&lhs, &rhs, c, &dev);
    let b = batch(3, c, 2, &[1.0], 6, 6); // one chunk, rows 0..3 real, row 3 is a pad
    let s = model.score_blocks(&b, &dev).unwrap();
    let rb = s.rhs_bat.to_vec3::<f32>().unwrap();
    let lb = s.lhs_bat.to_vec3::<f32>().unwrap();
    assert_eq!(rb.len(), 1);
    // Pad rows point at global id 0, a row of the lhs block, so read every
    // id through the flat table the model gathers from.
    let row = |g: u32| -> &Vec<f64> {
        let g = g as usize;
        if g < 6 {
            &lhs[g]
        } else {
            &rhs[g - 6]
        }
    };
    for i in 0..c {
        for j in 0..c {
            let masked = i == j || j == 3;
            if masked {
                assert!(
                    f64::from(rb[0][i][j]) < -1e8,
                    "rhs [{i},{j}] must be masked"
                );
                assert!(
                    f64::from(lb[0][i][j]) < -1e8,
                    "lhs [{i},{j}] must be masked"
                );
            } else {
                let want_rhs = dot(row(b.lhs[i]), row(b.rhs[j]));
                let want_lhs = dot(row(b.lhs[j]), row(b.rhs[i]));
                assert!(
                    approx(f64::from(rb[0][i][j]), want_rhs, 1e-5),
                    "rhs [{i},{j}]"
                );
                assert!(
                    approx(f64::from(lb[0][i][j]), want_lhs, 1e-5),
                    "lhs [{i},{j}]"
                );
            }
        }
    }
    let pos = s.pos.to_vec2::<f32>().unwrap();
    assert!(approx(f64::from(pos[0][1]), dot(&lhs[1], &rhs[1]), 1e-5));
    let ru = s.rhs_uni.expect("uniform block").to_vec3::<f32>().unwrap();
    assert!(approx(
        f64::from(ru[0][2][1]),
        dot(&lhs[2], &rhs[(b.uni_rhs[1] - 6) as usize]),
        1e-5
    ));
}

#[test]
fn fused_batch_loss_equals_a_per_chunk_reference_computed_in_f64() {
    let dev = Device::Cpu;
    let (lhs, rhs) = tables(9, 11, 5);
    let c = 3;
    let model = model_from(&lhs, &rhs, c, &dev);
    // 3 chunks, 2 pad rows, per-row weights that differ (relation × edge).
    let b = batch(7, c, 2, &[2.33, 0.4, 1.7], 9, 11);
    let got = f64::from(
        model
            .batch_loss(&b, &dev)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
    );
    let want = reference_loss(&lhs, &rhs, &b);
    assert!(
        approx(got, want, 1e-4 * want.abs().max(1.0)),
        "fused {got} vs reference {want}"
    );
}

#[test]
fn padding_a_short_group_does_not_change_the_loss_of_its_real_edges() {
    let dev = Device::Cpu;
    let (lhs, rhs) = tables(8, 8, 3);
    let wide = model_from(&lhs, &rhs, 50, &dev);
    let tight = model_from(&lhs, &rhs, 7, &dev);
    let b_wide = batch(7, 50, 0, &[1.0], 8, 8);
    let b_tight = batch(7, 7, 0, &[1.0], 8, 8);
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
fn weight_decay_term_is_the_summed_squared_norm_of_the_flat_table() {
    let dev = Device::Cpu;
    let (lhs, rhs) = tables(5, 4, 3);
    let model = model_from(&lhs, &rhs, 2, &dev);
    let got = f64::from(model.frob_sq().unwrap().to_scalar::<f32>().unwrap());
    let want: f64 = lhs.iter().chain(&rhs).flatten().map(|v| v * v).sum();
    assert!(approx(got, want, 1e-4));
}

#[test]
fn a_two_type_table_reproduces_simbas_seeded_init_exactly() {
    let dev = Device::Cpu;
    let t = NodeTypeTable::new(&[("e_cell", 7), ("e_gene", 5)]).unwrap();
    let ours = FneModel::new(&t, 6, 3, 42, &dev).unwrap();
    let simba = crate::simba::train::SimbaModel::new(7, 5, 6, 3, 42, &dev).unwrap();
    let e = ours.e.as_tensor().to_vec2::<f32>().unwrap();
    let ec = simba.e_cell.as_tensor().to_vec2::<f32>().unwrap();
    let eg = simba.e_gene.as_tensor().to_vec2::<f32>().unwrap();
    assert_eq!(e.len(), 12);
    assert_eq!(&e[..7], &ec[..], "cell block bit-identical");
    assert_eq!(&e[7..], &eg[..], "gene block bit-identical");
}

#[test]
fn the_hold_out_split_never_empties_a_relation_and_honours_the_floor() {
    assert_eq!(split_relation(0, 0.05, 1), (0, 0));
    assert_eq!(
        split_relation(1, 0.05, 1),
        (1, 0),
        "one edge stays in training"
    );
    assert_eq!(split_relation(4, 0.5, 1), (2, 2));
    assert_eq!(split_relation(4, 0.05, 1), (3, 1), "floor of one");
    assert_eq!(split_relation(4, 0.05, 3), (1, 3), "floor capped at n − 1");
    assert_eq!(split_relation(100, 0.05, 1), (95, 5));
    assert_eq!(
        split_relation(100, 0.0, 1),
        (100, 0),
        "fraction 0 holds out nothing"
    );
    assert_eq!(
        split_relation(100, 0.9, 1),
        (50, 50),
        "fraction clamped at one half"
    );
}

/// Genes 0..10, two cell types, two terms. Genes 0..5 mark type A and sit
/// in term X; genes 5..10 mark type B and sit in term Y; a PPI ring
/// within each gene group; one weak cross edge.
fn planted_graph() -> (TypedEdgeList, NodeTypeTable, RelationTable) {
    let t = NodeTypeTable::new(&[("gene", 10), ("cell_type", 2), ("term", 2)]).unwrap();
    let rels = RelationTable::new(
        vec![
            Relation {
                name: "ppi".into(),
                lhs_type: 0,
                rhs_type: 0,
                weight: 1.0,
                undirected: true,
            },
            Relation {
                name: "marker".into(),
                lhs_type: 0,
                rhs_type: 1,
                weight: 1.0,
                undirected: false,
            },
            Relation {
                name: "go".into(),
                lhs_type: 0,
                rhs_type: 2,
                weight: 1.0,
                undirected: false,
            },
        ],
        &t,
    )
    .unwrap();
    let mut e = TypedEdgeList::default();
    let mut w = Vec::new();
    let mut push = |l: u32, r: u32, rel: u16, wt: f32| {
        e.lhs.push(l);
        e.rhs.push(r);
        e.rel.push(rel);
        w.push(wt);
    };
    for grp in 0..2u32 {
        let genes: Vec<u32> = (grp * 5..grp * 5 + 5).collect();
        for i in 0..5 {
            for j in (i + 1)..5 {
                push(genes[i], genes[j], 0, 1.0);
            }
            push(genes[i], t.global(1, grp), 1, 1.0);
            push(genes[i], t.global(2, grp), 2, 1.0);
        }
    }
    push(4, 5, 0, 0.1);
    e.weight = Some(w);
    (e, t, rels)
}

#[test]
fn training_a_three_type_graph_places_each_gene_group_with_its_own_type_and_term() {
    let (edges, t, rels) = planted_graph();
    let n_edges = edges.len();
    let cfg = FneConfig {
        dim: 8,
        epochs: 60,
        batch_size: 16,
        num_batch_negs: 4,
        num_uniform_negs: 4,
        wd: Some(0.0),
        eval_fraction: 0.0,
        seed: 1,
        ..FneConfig::default()
    };
    let out = train(edges, t, rels, &cfg).unwrap();
    assert_eq!(out.n_edges, n_edges);
    assert_eq!(out.n_train_edges, n_edges);
    assert_eq!(out.n_eval_edges, 0);
    assert_eq!(out.per_relation.len(), 3);
    assert_eq!(out.per_relation[0].n_edges, 21);
    assert_eq!(out.per_relation[1].n_edges, 10);
    assert_eq!(out.embedding.dims(), &[14, 8]);
    let e = out.embedding.to_vec2::<f32>().unwrap();
    let dot = |a: usize, b: usize| -> f32 { e[a].iter().zip(&e[b]).map(|(x, y)| x * y).sum() };
    let ty = |g: usize| out.node_types.global(1, g as u32) as usize;
    let term = |g: usize| out.node_types.global(2, g as u32) as usize;
    for g in 0..10usize {
        let own = usize::from(g >= 5);
        assert!(
            dot(g, ty(own)) > dot(g, ty(1 - own)),
            "gene {g} scores its own cell type higher"
        );
        assert!(
            dot(g, term(own)) > dot(g, term(1 - own)),
            "gene {g} scores its own term higher"
        );
    }
    let first = out.epochs.first().unwrap().train_loss;
    let last = out.epochs.last().unwrap().train_loss;
    assert!(last < first, "loss falls over training: {first} → {last}");
}

#[test]
fn eval_edges_are_held_out_per_relation_and_an_eval_loss_is_reported_every_epoch() {
    let (edges, t, rels) = planted_graph();
    let cfg = FneConfig {
        dim: 4,
        epochs: 3,
        batch_size: 8,
        num_batch_negs: 2,
        num_uniform_negs: 2,
        wd: Some(0.0),
        eval_fraction: 0.25,
        eval_min_per_relation: 1,
        seed: 5,
        ..FneConfig::default()
    };
    let out = train(edges.clone(), t.clone(), rels.clone(), &cfg).unwrap();
    // 21 ppi → 5 eval; 10 marker → 2; 10 go → 2.
    assert_eq!(
        out.per_relation
            .iter()
            .map(|s| s.n_eval)
            .collect::<Vec<_>>(),
        vec![5, 2, 2]
    );
    assert_eq!(out.n_eval_edges, 9);
    assert_eq!(out.n_train_edges, 32);
    for s in &out.per_relation {
        assert!(s.train_loss.is_finite() && s.train_loss > 0.0);
        assert!(s.eval_loss.is_some_and(|e| e.is_finite() && e > 0.0));
    }
    // The overall eval loss is the edge-weighted mean of the relations'.
    let last = out.epochs.last().unwrap().eval_loss.unwrap();
    let weighted: f64 = out
        .per_relation
        .iter()
        .map(|s| s.eval_loss.unwrap() * s.n_eval as f64)
        .sum::<f64>()
        / 9.0;
    assert!((last - weighted).abs() < 1e-4 * last.abs().max(1.0));
    assert!(out
        .epochs
        .iter()
        .all(|e| e.eval_loss.is_some_and(f64::is_finite)));
    assert!(out
        .epochs
        .iter()
        .all(|e| e.train_loss.is_finite() && e.train_loss > 0.0));
    assert_eq!(out.wd, 0.0);

    // A tiny relation keeps at least one training edge under a large fraction.
    let mut tiny = TypedEdgeList::default();
    tiny.lhs.push(0);
    tiny.rhs.push(1);
    tiny.rel.push(0);
    for i in 0..10u32 {
        tiny.lhs.push(i);
        tiny.rhs.push(t.global(1, i % 2));
        tiny.rel.push(1);
    }
    let out = train(
        tiny,
        t.clone(),
        rels.clone(),
        &FneConfig {
            eval_fraction: 0.5,
            epochs: 1,
            ..cfg.clone()
        },
    )
    .unwrap();
    let counts: Vec<(usize, usize, usize)> = out
        .per_relation
        .iter()
        .map(|s| (s.n_edges, s.n_train, s.n_eval))
        .collect();
    assert_eq!(counts, vec![(1, 1, 0), (10, 5, 5), (0, 0, 0)]);
    assert!(out.per_relation[0].train_loss.is_finite());
    assert_eq!(out.per_relation[0].eval_loss, None, "nothing held out");
    assert!(out.per_relation[1].eval_loss.is_some_and(f64::is_finite));
    assert!(out.per_relation[2].train_loss.is_nan(), "no edges at all");
    assert_eq!(out.per_relation[2].eval_loss, None);

    let none = train(
        edges.clone(),
        t.clone(),
        rels.clone(),
        &FneConfig {
            eval_fraction: 0.0,
            ..cfg.clone()
        },
    )
    .unwrap();
    assert_eq!(none.n_eval_edges, 0);
    assert!(none.epochs.iter().all(|e| e.eval_loss.is_none()));
    // auto weight decay is reported when not pinned
    let auto = train(
        edges.clone(),
        t.clone(),
        rels.clone(),
        &FneConfig {
            wd: None,
            epochs: 1,
            ..cfg.clone()
        },
    )
    .unwrap();
    assert!(approx(auto.wd, auto_wd(41), 0.0));
    // wd_interval 1 draws the decay on every batch.
    let every = train(
        edges,
        t,
        rels,
        &FneConfig {
            wd: Some(1e-4),
            wd_interval: 1,
            epochs: 1,
            ..cfg.clone()
        },
    )
    .unwrap();
    let hits = every.epochs[0].wd_hits;
    assert!((4..=32).contains(&hits), "wd hits {hits}");
}

#[test]
fn a_repeated_relation_trains_more_and_its_stats_say_so() {
    let (edges, t, rels) = planted_graph();
    let base = FneConfig {
        dim: 8,
        epochs: 5,
        batch_size: 8,
        num_batch_negs: 4,
        num_uniform_negs: 4,
        wd: Some(0.0),
        eval_fraction: 0.0,
        seed: 3,
        ..FneConfig::default()
    };
    let once = train(edges.clone(), t.clone(), rels.clone(), &base).unwrap();
    let rep = train(
        edges,
        t,
        rels,
        &FneConfig {
            relation_repeats: vec![1, 8, 1],
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(once.per_relation[1].repeat, 1);
    assert_eq!(rep.per_relation[1].repeat, 8);
    assert_eq!(
        rep.per_relation[1].n_train, once.per_relation[1].n_train,
        "repeats do not change the edge counts"
    );
    assert!(
        rep.per_relation[1].train_loss < once.per_relation[1].train_loss,
        "eight passes over the marker relation fit it better: {} vs {}",
        rep.per_relation[1].train_loss,
        once.per_relation[1].train_loss
    );
    assert!(rep.per_relation[1].train_loss.is_finite());
}

#[test]
fn the_same_seed_gives_the_same_table() {
    let (edges, t, rels) = planted_graph();
    let cfg = FneConfig {
        dim: 4,
        epochs: 2,
        batch_size: 8,
        num_batch_negs: 2,
        num_uniform_negs: 2,
        eval_fraction: 0.1,
        seed: 9,
        ..FneConfig::default()
    };
    let a = train(edges.clone(), t.clone(), rels.clone(), &cfg).unwrap();
    let b = train(edges, t, rels, &cfg).unwrap();
    assert_eq!(
        a.embedding.to_vec2::<f32>().unwrap(),
        b.embedding.to_vec2::<f32>().unwrap()
    );
}
