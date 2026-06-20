//! Post-training inference + writers for `pinto lc-etm`.
//!
//! Output convention mirrors `pinto lc` for the shared files
//! (`.link_community.parquet`, `.propensity.parquet`, `.gene_community.parquet`)
//! so `pinto plot` and downstream consumers work unchanged. Three
//! additional ETM-only artefacts are emitted: `.latent.parquet` (per-edge
//! soft π_e), `.gene_embeddings.parquet` (ρ), `.community_embeddings.parquet` (α).

use anyhow::Context;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::data::top_k_indices_weighted;
use candle_util::decoder::EmbeddedTopicDecoder;
use candle_util::encoder::IndexedEmbeddingEncoder;
use candle_util::traits::*;
use graph_embedding_util::embedding_col_names;
use log::info;
use matrix_util::traits::{ConvertMatOps, IoOps};
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::link_community::profiles::shannon_entropy_rows;

type Mat = DMatrix<f32>;

const INFERENCE_BATCH: usize = 4096;

/// Inputs to [`run_inference_and_write`].
pub struct InferenceArgs<'a> {
    pub encoder: &'a IndexedEmbeddingEncoder,
    pub decoder: &'a EmbeddedTopicDecoder,
    pub edge_profiles: &'a Mat,
    pub edges: &'a [(usize, usize)],
    pub n_cells: usize,
    pub n_communities: usize,
    pub cell_names: &'a [Box<str>],
    pub gene_names: &'a [Box<str>],
    pub shortlist_weights: &'a [f32],
    pub context_size: usize,
    pub dev: &'a Device,
    pub out_prefix: &'a str,
}

/// Run the trained encoder + decoder over `edge_profiles` and write all
/// downstream outputs.
pub fn run_inference_and_write(args: InferenceArgs) -> anyhow::Result<()> {
    let n_edges = args.edge_profiles.nrows();
    let n_genes = args.edge_profiles.ncols();
    anyhow::ensure!(
        args.edges.len() == n_edges,
        "edges ({}) and edge_profiles rows ({}) must match",
        args.edges.len(),
        n_edges
    );
    anyhow::ensure!(
        args.shortlist_weights.len() == n_genes,
        "shortlist_weights length {} != n_genes {}",
        args.shortlist_weights.len(),
        n_genes
    );

    info!(
        "Inference: {} edges × {} communities (batch {})",
        n_edges, args.n_communities, INFERENCE_BATCH
    );

    let pi_ek = infer_edge_pi(
        args.encoder,
        args.edge_profiles,
        args.shortlist_weights,
        args.context_size,
        args.dev,
    )?;
    let propensity = aggregate_propensity(&pi_ek, args.edges, args.n_cells);

    // `get_dictionary()` returns log_β (log_softmax over genes); the
    // melted parquet's `mean` column convention is probability, so exp first.
    let beta_gk = host_from_tensor(&args.decoder.get_dictionary()?.exp()?)?;
    let rho_gh = host_from_tensor(args.decoder.feature_embeddings())?;
    let alpha_kh = host_from_tensor(args.decoder.topic_embeddings())?;

    let hard_z: Vec<usize> = (0..n_edges).map(|e| argmax_row(&pi_ek, e)).collect();
    crate::link_community::outputs::write_link_communities(
        &format!("{}.link_community.parquet", args.out_prefix),
        args.edges,
        &hard_z,
        args.cell_names,
    )?;

    write_propensity(
        &propensity,
        args.cell_names,
        args.n_communities,
        args.out_prefix,
    )?;
    write_gene_community_melted(&beta_gk, args.gene_names, args.out_prefix)?;
    write_latent(&pi_ek, args.edges, args.cell_names, args.out_prefix)?;
    write_embedding_table(
        &rho_gh,
        args.gene_names,
        "gene",
        "gene_embeddings",
        args.out_prefix,
    )?;
    let community_names = community_col_names(alpha_kh.nrows());
    write_embedding_table(
        &alpha_kh,
        &community_names,
        "community",
        "community_embeddings",
        args.out_prefix,
    )?;

    Ok(())
}

fn host_from_tensor(t: &Tensor) -> anyhow::Result<Mat> {
    let host = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    Mat::from_tensor(&host).context("tensor → DMatrix")
}

fn argmax_row(mat: &Mat, row: usize) -> usize {
    let mut best_k = 0;
    let mut best_v = f32::NEG_INFINITY;
    for k in 0..mat.ncols() {
        let v = mat[(row, k)];
        if v > best_v {
            best_v = v;
            best_k = k;
        }
    }
    best_k
}

fn infer_edge_pi(
    encoder: &IndexedEmbeddingEncoder,
    edge_profiles: &Mat,
    shortlist_weights: &[f32],
    context_size: usize,
    dev: &Device,
) -> anyhow::Result<Mat> {
    let n_edges = edge_profiles.nrows();
    let n_genes = edge_profiles.ncols();
    let k = context_size.min(n_genes);
    let n_topics = encoder.dim_latent();

    let mut pi_ek = Mat::zeros(n_edges, n_topics);

    for start in (0..n_edges).step_by(INFERENCE_BATCH) {
        let end = (start + INFERENCE_BATCH).min(n_edges);
        let nb = end - start;

        // Build the per-edge row, score against shortlist, and pack into
        // dense top-K slots in one parallel pass — no Vec<Vec<f32>> in
        // between. Each thread allocates one G-length row buffer, reused
        // implicitly via the collect.
        let topk: Vec<(Vec<u32>, Vec<f32>)> = (start..end)
            .into_par_iter()
            .map(|e| {
                let row: Vec<f32> = (0..n_genes).map(|g| edge_profiles[(e, g)]).collect();
                top_k_indices_weighted(&row, shortlist_weights, k)
            })
            .collect();

        let mut indices_flat = vec![0u32; nb * k];
        let mut values_flat = vec![0f32; nb * k];
        for (i, (idx, val)) in topk.iter().enumerate() {
            let len = idx.len().min(k);
            indices_flat[i * k..i * k + len].copy_from_slice(&idx[..len]);
            values_flat[i * k..i * k + len].copy_from_slice(&val[..len]);
        }
        let indices = Tensor::from_vec(indices_flat, (nb, k), dev)?;
        let values = Tensor::from_vec(values_flat, (nb, k), dev)?;

        let (log_z_nk, _) =
            encoder.forward_indexed_t(&indices, &values, None, None, None, false)?;
        let pi_nk: Vec<Vec<f32>> = log_z_nk.exp()?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        for (i, row) in pi_nk.into_iter().enumerate() {
            for (kk, v) in row.into_iter().enumerate() {
                pi_ek[(start + i, kk)] = v;
            }
        }
    }

    Ok(pi_ek)
}

fn aggregate_propensity(pi_ek: &Mat, edges: &[(usize, usize)], n_cells: usize) -> Mat {
    let n_topics = pi_ek.ncols();
    let mut accum = Mat::zeros(n_cells, n_topics);
    let mut degree = vec![0f32; n_cells];

    for (e, &(i, j)) in edges.iter().enumerate() {
        for k in 0..n_topics {
            let p = pi_ek[(e, k)];
            accum[(i, k)] += p;
            accum[(j, k)] += p;
        }
        degree[i] += 1.0;
        degree[j] += 1.0;
    }

    for i in 0..n_cells {
        let d = degree[i].max(1.0);
        for k in 0..n_topics {
            accum[(i, k)] /= d;
        }
    }
    accum
}

fn community_col_names(k: usize) -> Vec<Box<str>> {
    (0..k).map(|c| format!("C{c}").into_boxed_str()).collect()
}

fn write_propensity(
    propensity: &Mat,
    cell_names: &[Box<str>],
    n_communities: usize,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let n_cells = propensity.nrows();
    let entropy = shannon_entropy_rows(propensity);

    let mut combined = Mat::zeros(n_cells, n_communities + 1);
    for i in 0..n_cells {
        for k in 0..n_communities {
            combined[(i, k)] = propensity[(i, k)];
        }
        combined[(i, n_communities)] = entropy[i];
    }

    let mut col_names = community_col_names(n_communities);
    col_names.push("entropy".into());

    combined.to_parquet_with_names(
        &format!("{}.propensity.parquet", out_prefix),
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;
    Ok(())
}

/// Melted `(gene, community, mean)` form that
/// `pinto::plot::load::read_gene_community` expects.
fn write_gene_community_melted(
    beta_gk: &Mat,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let g = beta_gk.nrows();
    let k = beta_gk.ncols();
    let n_rows = g * k;
    let community_names = community_col_names(k);

    let mut gene_col: Vec<Box<str>> = Vec::with_capacity(n_rows);
    let mut community_col: Vec<Box<str>> = Vec::with_capacity(n_rows);
    let mut mean_col: Vec<f32> = Vec::with_capacity(n_rows);
    for gi in 0..g {
        for ki in 0..k {
            gene_col.push(gene_names[gi].clone());
            community_col.push(community_names[ki].clone());
            mean_col.push(beta_gk[(gi, ki)]);
        }
    }

    let col_names: Vec<Box<str>> = vec!["gene".into(), "community".into(), "mean".into()];
    let col_types = vec![
        ParquetType::BYTE_ARRAY,
        ParquetType::BYTE_ARRAY,
        ParquetType::FLOAT,
    ];

    let writer = ParquetWriter::new(
        &format!("{}.gene_community.parquet", out_prefix),
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("row"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_string_column(&mut row_group, &gene_col)?;
    parquet_add_string_column(&mut row_group, &community_col)?;
    parquet_add_numeric_column(&mut row_group, &mean_col)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

fn write_latent(
    pi_ek: &Mat,
    edges: &[(usize, usize)],
    cell_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let col_names = community_col_names(pi_ek.ncols());
    let row_names: Vec<Box<str>> = edges
        .iter()
        .map(|&(i, j)| format!("{}:{}", cell_names[i], cell_names[j]).into_boxed_str())
        .collect();
    pi_ek.to_parquet_with_names(
        &format!("{}.latent.parquet", out_prefix),
        (Some(&row_names), Some("edge")),
        Some(&col_names),
    )?;
    Ok(())
}

/// Generic `[R × H]` embedding table writer. Used for both
/// `gene_embeddings` (ρ) and `community_embeddings` (α).
fn write_embedding_table(
    mat: &Mat,
    row_names: &[Box<str>],
    row_axis: &str,
    suffix: &str,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let col_names = embedding_col_names(mat.ncols());
    mat.to_parquet_with_names(
        &format!("{}.{}.parquet", out_prefix, suffix),
        (Some(row_names), Some(row_axis)),
        Some(&col_names),
    )?;
    Ok(())
}
