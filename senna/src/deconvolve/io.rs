//! Output writers for `senna deconvolve`.

use super::result::DeconvResult;
use crate::embed_common::{axis_id_names, Mat};
use anyhow::{Context, Result};
use log::info;
use matrix_util::traits::IoOps;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Metadata recorded in the run summary JSON.
pub struct RunMeta<'a> {
    pub from: &'a str,
    pub markers: Option<&'a str>,
    pub kind: String,
    /// Which reference built the component profiles.
    pub reference: &'a str,
    /// Number of reference components `R` (cell types, or archetypes), pooled
    /// across chains.
    pub n_components: usize,
    /// Chains pooled into the posterior; >1 means the reference partition was
    /// averaged over rather than conditioned on.
    pub n_chains: usize,
    /// `cell` or `mrna` — what the reported fractions are shares of.
    pub fraction_units: &'a str,
    pub warmup: usize,
    pub draws: usize,
    pub bulk_files: &'a [Box<str>],
    /// Whether a trace was requested; the manifest must not point at a file
    /// that was never created.
    pub traced: bool,
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
        (
            Some(&result.anchor_names),
            Some(result.anchor_axis.corner()),
        ),
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

    // Per-sample QC + sampler diagnostics.
    write_residual(&format!("{out}.residual.tsv"), sample_names, result)?;
    let converged_written =
        write_convergence(&format!("{out}.convergence.tsv"), sample_names, ct, result)?;
    write_manifest(out, meta, converged_written)?;

    info!(
        "senna deconvolve: wrote {out}.{{fractions,fractions_ci,abundance,residual}}.tsv, \
         {out}.{{sample_embedding,anchors}}.parquet, {out}.expression/*.parquet"
    );
    Ok(())
}

/// Wide `sample × celltype` TSV of a matrix.
pub(super) fn write_wide_tsv(
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
    w.flush().with_context(|| format!("flushing {path}"))
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
    w.flush().with_context(|| format!("flushing {path}"))
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
    w.flush().with_context(|| format!("flushing {path}"))
}

/// Split-R̂ and effective sample size per reported scalar.
///
/// Single-chain split-R̂ catches a drifting or unconverged chain, but not
/// multimodality — for that, rerun with a different `--seed` and compare.
fn write_convergence(
    path: &str,
    samples: &[Box<str>],
    cols: &[Box<str>],
    result: &DeconvResult,
) -> Result<bool> {
    if result.convergence.is_empty() {
        return Ok(false);
    }
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {path}"))?);
    writeln!(w, "sample\tcelltype\tsplit_rhat\tess")?;
    for cv in &result.convergence {
        writeln!(
            w,
            "{}\t{}\t{:.4}\t{:.1}",
            samples[cv.sample], cols[cv.celltype], cv.rhat, cv.ess
        )?;
    }
    let worst = result
        .convergence
        .iter()
        .fold(0f32, |acc, cv| acc.max(cv.rhat));
    if worst > 1.05 {
        // With pooled chains the chains hold *different references*, so a large
        // R̂ is the partitions disagreeing, not a chain failing to settle.
        // Telling the user to raise --warmup would be advice that cannot work.
        if result.n_chains > 1 {
            log::warn!(
                "deconvolve: worst between-chain R̂ is {worst:.3} over {} pooled references — \
                 the reference partition moves the answer more than sampling noise does. The \
                 pooled interval reflects that; treat a single-partition run as overconfident.",
                result.n_chains
            );
        } else {
            log::warn!(
                "deconvolve: worst split-R̂ is {worst:.3} (> 1.05) — the chain has not settled; \
                 raise --warmup or --draws"
            );
        }
    }
    Ok(true)
}

fn write_manifest(out: &str, meta: &RunMeta, converged_written: bool) -> Result<()> {
    let path = format!("{out}.deconvolve.json");
    let json = serde_json::json!({
        "version": 1,
        "kind": "deconvolve",
        "from": meta.from,
        "markers": meta.markers,
        "source_kind": meta.kind,
        "reference": meta.reference,
        "n_components": meta.n_components,
        "n_chains": meta.n_chains,
        "fraction_units": meta.fraction_units,
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
            "convergence": converged_written.then(|| format!("{out}.convergence.tsv")),
            "trace": meta.traced.then(|| format!("{out}.trace.tsv.gz")),
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
