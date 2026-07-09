//! Output writers for `senna deconvolve`.

use super::gibbs::DeconvResult;
use crate::embed_common::{axis_id_names, Mat};
use anyhow::{Context, Result};
use log::info;
use matrix_util::traits::IoOps;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Metadata recorded in the run summary JSON.
pub struct RunMeta<'a> {
    pub from: &'a str,
    pub markers: &'a str,
    pub kind: String,
    pub exact: bool,
    pub warmup: usize,
    pub draws: usize,
    pub bulk_files: &'a [Box<str>],
}

pub fn write_outputs(
    out: &str,
    sample_names: &[Box<str>],
    gene_names: &[Box<str>],
    sample_z: &Mat,
    result: &DeconvResult,
    meta: &RunMeta,
) -> Result<()> {
    let ct = &result.celltype_names;
    let h_names = axis_id_names("h", sample_z.ncols());

    // Fractions (posterior mean, wide) + credible-interval long form.
    write_wide_tsv(
        &format!("{out}.fractions.tsv"),
        "sample",
        sample_names,
        ct,
        &result.fractions_mean,
    )?;
    write_fraction_ci(&format!("{out}.fractions_ci.tsv"), sample_names, ct, result)?;
    write_wide_tsv(
        &format!("{out}.abundance.tsv"),
        "sample",
        sample_names,
        ct,
        &result.abundance_mean,
    )?;

    // Projected bulk + posterior anchors (embedding space).
    sample_z.to_parquet_with_names(
        &format!("{out}.sample_embedding.parquet"),
        (Some(sample_names), Some("sample")),
        Some(&h_names),
    )?;
    result.anchors_post.to_parquet_with_names(
        &format!("{out}.anchors.parquet"),
        (Some(ct), Some("celltype")),
        Some(&h_names),
    )?;

    // Per-cell-type expression stacks (samples × genes), DE-ready.
    let expr_dir = format!("{out}.expression");
    std::fs::create_dir_all(&expr_dir)
        .with_context(|| format!("creating expression dir {expr_dir}"))?;
    for (c, mat) in result.expression.iter().enumerate() {
        let sg = mat.transpose(); // D×S → S×D
        let fname = format!("{expr_dir}/{}.parquet", sanitize(&ct[c]));
        sg.to_parquet_with_names(
            &fname,
            (Some(sample_names), Some("sample")),
            Some(gene_names),
        )?;
    }

    // Per-sample QC.
    write_residual(&format!("{out}.residual.tsv"), sample_names, result)?;
    write_manifest(out, meta)?;

    info!(
        "senna deconvolve: wrote {out}.{{fractions,fractions_ci,abundance,residual}}.tsv, \
         {out}.{{sample_embedding,anchors}}.parquet, {out}.expression/*.parquet"
    );
    Ok(())
}

/// Wide `sample × celltype` TSV of a matrix.
fn write_wide_tsv(
    path: &str,
    corner: &str,
    rows: &[Box<str>],
    cols: &[Box<str>],
    mat: &Mat,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {path}"))?);
    write!(w, "{corner}")?;
    for c in cols {
        write!(w, "\t{c}")?;
    }
    writeln!(w)?;
    for (i, name) in rows.iter().enumerate() {
        write!(w, "{name}")?;
        for j in 0..cols.len() {
            write!(w, "\t{:.6}", mat[(i, j)])?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Long-form fraction credible intervals.
fn write_fraction_ci(
    path: &str,
    samples: &[Box<str>],
    cols: &[Box<str>],
    result: &DeconvResult,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {path}"))?);
    writeln!(w, "sample\tcelltype\tmean\tsd\tq2.5\tq97.5")?;
    for (i, s) in samples.iter().enumerate() {
        for (j, ctn) in cols.iter().enumerate() {
            writeln!(
                w,
                "{s}\t{ctn}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                result.fractions_mean[(i, j)],
                result.fractions_sd[(i, j)],
                result.fractions_lo[(i, j)],
                result.fractions_hi[(i, j)],
            )?;
        }
    }
    Ok(())
}

fn write_residual(path: &str, samples: &[Box<str>], result: &DeconvResult) -> Result<()> {
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {path}"))?);
    writeln!(w, "sample\ttotal_counts\tdeviance\tpearson")?;
    for (i, s) in samples.iter().enumerate() {
        let r = &result.residual[i];
        writeln!(
            w,
            "{s}\t{:.1}\t{:.4}\t{:.4}",
            r.total, r.deviance, r.pearson
        )?;
    }
    Ok(())
}

fn write_manifest(out: &str, meta: &RunMeta) -> Result<()> {
    let path = format!("{out}.deconvolve.json");
    let json = serde_json::json!({
        "version": 1,
        "kind": "deconvolve",
        "from": meta.from,
        "markers": meta.markers,
        "source_kind": meta.kind,
        "projection_exact": meta.exact,
        "warmup": meta.warmup,
        "draws": meta.draws,
        "bulk": meta.bulk_files.iter().map(std::string::ToString::to_string).collect::<Vec<_>>(),
        "outputs": {
            "fractions": format!("{out}.fractions.tsv"),
            "fractions_ci": format!("{out}.fractions_ci.tsv"),
            "abundance": format!("{out}.abundance.tsv"),
            "sample_embedding": format!("{out}.sample_embedding.parquet"),
            "anchors": format!("{out}.anchors.parquet"),
            "expression_dir": format!("{out}.expression"),
            "residual": format!("{out}.residual.tsv"),
        },
    });
    std::fs::write(&path, serde_json::to_string_pretty(&json)?)
        .with_context(|| format!("writing {path}"))?;
    Ok(())
}

/// Filesystem-safe cell-type label for per-type filenames.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
