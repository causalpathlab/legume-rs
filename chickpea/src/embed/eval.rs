//! Output writers for trained joint multiome embeddings.
//!
//! Writes gzipped TSVs for the unified `E_feat` (genes ∪ peaks),
//! `E_cell`, plus bias vectors. The feature TSV includes a `modality`
//! column tagging each row as `gene` or `peak` so downstream scripts
//! can split the matrix as needed.

use crate::embed::data::Modality;
use crate::embed::model::JointEmbedModel;
use candle_util::candle_core::Tensor;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Save a `[N, H]` embedding matrix to a gzipped TSV with row labels.
fn save_embedding_tsv(path: &str, table: &Tensor, row_names: &[Box<str>]) -> anyhow::Result<()> {
    let n = table.dim(0)?;
    let h = table.dim(1)?;
    if row_names.len() != n {
        anyhow::bail!(
            "row_names ({}) does not match embedding rows ({})",
            row_names.len(),
            n
        );
    }
    let f = File::create(path)?;
    let gz = GzEncoder::new(f, Compression::default());
    let mut w = BufWriter::new(gz);

    // Header: name, h0, h1, ..., h{H-1}
    write!(w, "name")?;
    for j in 0..h {
        write!(w, "\th{}", j)?;
    }
    writeln!(w)?;

    let data: Vec<f32> = table.flatten_all()?.to_vec1()?;
    for i in 0..n {
        write!(w, "{}", row_names[i])?;
        for j in 0..h {
            write!(w, "\t{:.6}", data[i * h + j])?;
        }
        writeln!(w)?;
    }
    w.flush()?;
    Ok(())
}

/// Save a `[N]` bias vector to a TSV with row labels.
fn save_bias_tsv(path: &str, bias: &Tensor, row_names: &[Box<str>]) -> anyhow::Result<()> {
    let data: Vec<f32> = bias.flatten_all()?.to_vec1()?;
    if data.len() != row_names.len() {
        anyhow::bail!(
            "bias len ({}) does not match row_names ({})",
            data.len(),
            row_names.len(),
        );
    }
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "name\tbias")?;
    for (i, name) in row_names.iter().enumerate() {
        writeln!(w, "{}\t{:.6}", name, data[i])?;
    }
    w.flush()?;
    Ok(())
}

pub struct OutputContext<'a> {
    pub feature_names: &'a [Box<str>],
    pub feature_modality: &'a [Modality],
    pub barcodes: &'a [Box<str>],
}

pub fn save_outputs(
    model: &JointEmbedModel,
    ctx: &OutputContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    save_feature_embedding_tsv(
        &format!("{}.e_feat.tsv.gz", out_prefix),
        &model.e_feat,
        ctx.feature_names,
        ctx.feature_modality,
    )?;
    save_embedding_tsv(
        &format!("{}.e_cell.tsv.gz", out_prefix),
        &model.e_cell,
        ctx.barcodes,
    )?;
    save_feature_bias_tsv(
        &format!("{}.b_feat.tsv", out_prefix),
        &model.b_feat,
        ctx.feature_names,
        ctx.feature_modality,
    )?;
    save_bias_tsv(
        &format!("{}.b_cell.tsv", out_prefix),
        &model.b_cell,
        ctx.barcodes,
    )?;
    Ok(())
}

/// Save unified feature embedding `[D, H]` with a `modality` column
/// tagging each row as `gene` or `peak`.
fn save_feature_embedding_tsv(
    path: &str,
    table: &Tensor,
    names: &[Box<str>],
    modality: &[Modality],
) -> anyhow::Result<()> {
    let n = table.dim(0)?;
    let h = table.dim(1)?;
    if names.len() != n || modality.len() != n {
        anyhow::bail!(
            "names/modality lengths ({}/{}) ≠ embedding rows ({})",
            names.len(),
            modality.len(),
            n
        );
    }
    let f = File::create(path)?;
    let gz = GzEncoder::new(f, Compression::default());
    let mut w = BufWriter::new(gz);

    write!(w, "name\tmodality")?;
    for j in 0..h {
        write!(w, "\th{}", j)?;
    }
    writeln!(w)?;

    let data: Vec<f32> = table.flatten_all()?.to_vec1()?;
    for i in 0..n {
        let m = match modality[i] {
            Modality::Gene => "gene",
            Modality::Peak => "peak",
        };
        write!(w, "{}\t{}", names[i], m)?;
        for j in 0..h {
            write!(w, "\t{:.6}", data[i * h + j])?;
        }
        writeln!(w)?;
    }
    w.flush()?;
    Ok(())
}

fn save_feature_bias_tsv(
    path: &str,
    bias: &Tensor,
    names: &[Box<str>],
    modality: &[Modality],
) -> anyhow::Result<()> {
    let data: Vec<f32> = bias.flatten_all()?.to_vec1()?;
    if data.len() != names.len() {
        anyhow::bail!("bias len {} ≠ names {}", data.len(), names.len());
    }
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "name\tmodality\tbias")?;
    for (i, name) in names.iter().enumerate() {
        let m = match modality[i] {
            Modality::Gene => "gene",
            Modality::Peak => "peak",
        };
        writeln!(w, "{}\t{}\t{:.6}", name, m, data[i])?;
    }
    w.flush()?;
    Ok(())
}
