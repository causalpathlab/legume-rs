//! `senna embed-diag`: effective rank and common-mode readout of a run's
//! embedding tables.
//!
//! Every mechanism aimed at "the embedding uses a handful of its directions" is
//! trying to move one number — the participation ratio — and until this existed
//! that number was computed off-repo, by hand, per run. This reads it off the
//! manifest so an A/B is one command per arm and the same arithmetic for every
//! arm. It measures; it decides nothing.

use crate::run_manifest;
use clap::Args;
use matrix_util::embedding_geometry::{embedding_geometry, EmbeddingGeometry};
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

#[derive(Args, Debug)]
pub struct EmbedDiagArgs {
    #[arg(help = "Run manifest ({out}.senna.json) or the run's --out prefix")]
    pub(crate) from: Box<str>,
}

/// The tables a manifest may record that have the `[units × h]` shape the
/// geometry is defined on. Order is the report order.
const TABLES: [&str; 3] = ["cell_embedding", "feature_loading", "module_dictionary"];

pub fn embed_diag(args: &EmbedDiagArgs) -> anyhow::Result<()> {
    let rows = collect_geometry(&args.from)?;
    print_report(&rows);
    Ok(())
}

/// Measure every recorded table of `from`, in [`TABLES`] order. Errors when the
/// manifest records none of them — a run with nothing to measure is a wrong
/// `--from`, not an empty report.
pub(crate) fn collect_geometry(
    from: &str,
) -> anyhow::Result<Vec<(&'static str, EmbeddingGeometry)>> {
    let (manifest, dir) = run_manifest::load_for(from)?;
    let out = &manifest.outputs;
    let recorded = [
        out.cell_embedding.as_deref(),
        out.feature_loading.as_deref(),
        out.module_dictionary.as_deref(),
    ];

    let mut rows = Vec::new();
    for (name, rel) in TABLES.iter().zip(recorded) {
        let Some(rel) = rel else { continue };
        let path = run_manifest::resolve(&dir, rel);
        let path = path.to_string_lossy();
        let table = DMatrix::<f32>::from_parquet(&path)
            .map_err(|e| anyhow::anyhow!("{name} at {path}: {e}"))?;
        rows.push((*name, embedding_geometry(&table.mat)));
    }
    anyhow::ensure!(
        !rows.is_empty(),
        "{from}: the manifest records none of {}",
        TABLES.join(" / ")
    );
    Ok(rows)
}

/// A fixed-width table on stdout: this is a report, not a log line.
fn print_report(rows: &[(&str, EmbeddingGeometry)]) {
    println!(
        "{:<18} {:>8} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "table", "rows", "h", "pr_raw", "pr_ctr", "pair_cos", "mode_cos", "max_corr", "max_vif"
    );
    for (name, g) in rows {
        println!(
            "{:<18} {:>8} {:>5} {:>9.2} {:>9.2} {:>9.3} {:>9.3} {:>9.3} {:>9.2}",
            name,
            g.n_rows,
            g.h,
            g.eff_rank_raw,
            g.eff_rank_centered,
            g.mean_pairwise_cos,
            g.common_mode_cos,
            g.max_abs_corr,
            g.max_vif
        );
    }
    println!();
    println!(
        "pr_raw / pr_ctr: participation ratio of the raw / column-centred Gram, in [1, h].\n\
         Read it as VARIANCE CONCENTRATION, not useful dimensionality: a low value says\n\
         few directions carry the variance, not that the rest are noise.\n\
         pr_raw far below pr_ctr is a mean offset (mode_cos near 1), not a collapse.\n\
         pair_cos: signed mean cosine over distinct row pairs (a balanced cloud reads −1/(n−1))."
    );
}

#[cfg(test)]
mod tests;
