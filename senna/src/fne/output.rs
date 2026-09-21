//! The `fne` artifacts: one embedding table over every node, the node
//! types beside it, the relations, and the per-epoch losses.

use super::graph::TypedGraph;
use graph_embedding_util as ge;
use graph_embedding_util::fne::FneOutput;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use std::io::Write;

/// `feature <TAB> type <TAB> name <TAB> text`, one row per node carrying
/// any text; tabs and newlines inside the text are flattened to spaces so
/// the file stays one row per node. An empty name or text is an empty
/// field.
pub(crate) fn write_text_export(graph: &TypedGraph, path: &str) -> anyhow::Result<()> {
    let flat = |s: &str| s.split(['\t', '\n', '\r']).collect::<Vec<_>>().join(" ");
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "feature\ttype\tname\ttext")?;
    for (id, text) in &graph.texts {
        let i = *id as usize;
        writeln!(
            w,
            "{}\t{}\t{}\t{}",
            graph.node_names[i],
            graph.node_types[i],
            text.name.as_deref().map_or(String::new(), flat),
            text.text.as_deref().map_or(String::new(), flat),
        )?;
    }
    w.flush()?;
    info!(
        "fne: wrote text for {} of {} nodes to {path}",
        graph.texts.len(),
        graph.node_names.len()
    );
    Ok(())
}

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
    data_beans::aux::feature_types::write_feature_types(
        prefix,
        &graph.node_names,
        &graph.node_types,
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
    let polarity: Vec<Box<str>> = out
        .relations
        .iter()
        .map(|r| Box::from(r.polarity.as_str()))
        .collect();
    let as_i32 = |f: fn(&ge::fne::RelationStats) -> usize| -> Vec<i32> {
        out.per_relation.iter().map(|s| f(s) as i32).collect()
    };
    let n_edges = as_i32(|s| s.n_edges);
    let n_train = as_i32(|s| s.n_train);
    let n_eval = as_i32(|s| s.n_eval);
    let repeat = as_i32(|s| s.repeat);
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
            (Box::from("polarity"), Column::Str(&polarity)),
            (Box::from("weight"), Column::F32(&weight)),
            (Box::from("n_edges"), Column::I32(&n_edges)),
            (Box::from("n_train"), Column::I32(&n_train)),
            (Box::from("n_eval"), Column::I32(&n_eval)),
            (Box::from("repeat"), Column::I32(&repeat)),
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
