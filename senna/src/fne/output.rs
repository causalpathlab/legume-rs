//! The `fne` artifacts: one embedding table over every node, the node
//! types beside it, the relations, and the per-epoch losses.

use super::graph::TypedGraph;
use graph_embedding_util as ge;
use graph_embedding_util::fne::FneOutput;
use log::info;
use matrix_util::parquet::{write_named_table, Column};

pub(crate) fn write_outputs(
    out: &FneOutput,
    graph: &TypedGraph,
    prefix: &str,
) -> anyhow::Result<()> {
    let n = out.embedding.dim(0)?;
    let h = out.embedding.dim(1)?;
    anyhow::ensure!(
        n == graph.node_names.len(),
        "fne: {n} embedding rows for {} named nodes",
        graph.node_names.len()
    );

    // feature_embedding.parquet — [N, H], row column `feature`, every type.
    ge::save_embedding(
        &format!("{prefix}.feature_embedding.parquet"),
        &out.embedding,
        &graph.node_names,
        "feature",
    )?;

    // feature_types.parquet — the node type of every row, same order.
    write_named_table(
        &format!("{prefix}.feature_types.parquet"),
        "feature",
        &graph.node_names,
        &[(Box::from("type"), Column::Str(&graph.node_types))],
    )?;

    // relations.parquet — one row per relation.
    let rel_names: Vec<Box<str>> = out.relations.iter().map(|r| r.name.clone()).collect();
    let lhs: Vec<Box<str>> = out
        .relations
        .iter()
        .map(|r| Box::from(out.node_types.name(r.lhs_type as usize)))
        .collect();
    let rhs: Vec<Box<str>> = out
        .relations
        .iter()
        .map(|r| Box::from(out.node_types.name(r.rhs_type as usize)))
        .collect();
    let weight: Vec<f32> = out.relations.iter().map(|r| r.weight).collect();
    let as_i32 = |f: fn(&ge::fne::RelationStats) -> usize| -> Vec<i32> {
        out.per_relation.iter().map(|s| f(s) as i32).collect()
    };
    let n_edges = as_i32(|s| s.n_edges);
    let n_train = as_i32(|s| s.n_train);
    let n_eval = as_i32(|s| s.n_eval);
    let train_loss: Vec<f32> = out
        .per_relation
        .iter()
        .map(|s| s.train_loss as f32)
        .collect();
    let eval_loss: Vec<f32> = out
        .per_relation
        .iter()
        .map(|s| s.eval_loss.map_or(f32::NAN, |v| v as f32))
        .collect();
    write_named_table(
        &format!("{prefix}.relations.parquet"),
        "relation",
        &rel_names,
        &[
            (Box::from("lhs_type"), Column::Str(&lhs)),
            (Box::from("rhs_type"), Column::Str(&rhs)),
            (Box::from("weight"), Column::F32(&weight)),
            (Box::from("n_edges"), Column::I32(&n_edges)),
            (Box::from("n_train"), Column::I32(&n_train)),
            (Box::from("n_eval"), Column::I32(&n_eval)),
            (Box::from("train_loss"), Column::F32(&train_loss)),
            (Box::from("eval_loss"), Column::F32(&eval_loss)),
        ],
    )?;

    // log_likelihood.parquet — per-epoch losses (eval is NaN when nothing
    // was held out).
    let epoch_names: Vec<Box<str>> = out
        .epochs
        .iter()
        .map(|e| e.epoch.to_string().into_boxed_str())
        .collect();
    let train_loss: Vec<f32> = out.epochs.iter().map(|e| e.train_loss as f32).collect();
    let eval_loss: Vec<f32> = out
        .epochs
        .iter()
        .map(|e| e.eval_loss.map_or(f32::NAN, |v| v as f32))
        .collect();
    let wd_hits: Vec<i32> = out.epochs.iter().map(|e| e.wd_hits as i32).collect();
    write_named_table(
        &format!("{prefix}.log_likelihood.parquet"),
        "epoch",
        &epoch_names,
        &[
            (Box::from("train_loss"), Column::F32(&train_loss)),
            (Box::from("eval_loss"), Column::F32(&eval_loss)),
            (Box::from("wd_hits"), Column::I32(&wd_hits)),
        ],
    )?;

    info!("Saved {n} features × {h} dims to {prefix}.feature_embedding.parquet");
    Ok(())
}
