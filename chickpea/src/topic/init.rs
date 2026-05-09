//! Warm-start the indexed encoder's per-feature embeddings from a
//! `chickpea embed-graph` parquet. Names missing from the parquet
//! keep their Kaiming init, so partial overlap is fine.

use crate::common::*;
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::VarMap;
use matrix_util::traits::IoOps;
use rustc_hash::FxHashMap;

pub fn init_feature_embeddings_from_parquet(
    varmap: &VarMap,
    e_feat_path: &str,
    encoder_prefix: &str,
    gene_names: &[Box<str>],
    peak_names: &[Box<str>],
    embedding_dim: usize,
    dev: &Device,
) -> anyhow::Result<()> {
    info!("Warm-starting encoder feature embeddings from {e_feat_path}");
    let loaded = nalgebra::DMatrix::<f32>::from_parquet(e_feat_path)?;
    let h = loaded.cols.len();
    if h != embedding_dim {
        anyhow::bail!(
            "embedding_dim mismatch: {e_feat_path} has H={h}, fit-topic --embedding-dim={embedding_dim}"
        );
    }

    let name_to_row: FxHashMap<&str, usize> = loaded
        .rows
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();

    let (gene_t, gene_hits) = build_seeded_tensor(
        varmap,
        &loaded.mat,
        &name_to_row,
        gene_names,
        encoder_prefix,
        "gene_expert",
        dev,
    )?;
    let (peak_t, peak_hits) = build_seeded_tensor(
        varmap,
        &loaded.mat,
        &name_to_row,
        peak_names,
        encoder_prefix,
        "atac_expert",
        dev,
    )?;

    set_var(
        varmap,
        &format!("{encoder_prefix}.gene_expert.feature.embeddings"),
        &gene_t,
    )?;
    set_var(
        varmap,
        &format!("{encoder_prefix}.atac_expert.feature.embeddings"),
        &peak_t,
    )?;

    info!(
        "Seeded gene_expert: {}/{} rows; atac_expert: {}/{} rows",
        gene_hits,
        gene_names.len(),
        peak_hits,
        peak_names.len()
    );
    Ok(())
}

fn build_seeded_tensor(
    varmap: &VarMap,
    parquet_mat: &nalgebra::DMatrix<f32>,
    name_to_row: &FxHashMap<&str, usize>,
    target_names: &[Box<str>],
    encoder_prefix: &str,
    expert: &str,
    dev: &Device,
) -> anyhow::Result<(Tensor, usize)> {
    let h = parquet_mat.ncols();
    let n = target_names.len();
    let key = format!("{encoder_prefix}.{expert}.feature.embeddings");

    let existing_var = {
        let data = varmap.data().lock().expect("VarMap lock");
        data.get(&key)
            .ok_or_else(|| anyhow::anyhow!("var '{key}' not found in VarMap"))?
            .as_tensor()
            .clone()
    };
    let mut buf: Vec<f32> = existing_var.flatten_all()?.to_vec1()?;

    let mut hits = 0usize;
    for (i, name) in target_names.iter().enumerate() {
        if let Some(&row) = name_to_row.get(name.as_ref()) {
            for c in 0..h {
                buf[i * h + c] = parquet_mat[(row, c)];
            }
            hits += 1;
        }
    }
    Ok((Tensor::from_vec(buf, (n, h), dev)?, hits))
}

fn set_var(varmap: &VarMap, name: &str, t: &Tensor) -> anyhow::Result<()> {
    let data = varmap.data().lock().expect("VarMap lock");
    let var = data
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("var '{name}' not found in VarMap"))?;
    var.set(t)?;
    Ok(())
}
