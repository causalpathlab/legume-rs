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

pub fn embed_diag(args: &EmbedDiagArgs) -> anyhow::Result<()> {
    let rows = collect_geometry(&args.from)?;
    print_report(&rows);
    Ok(())
}

/// A digest of a file's bytes, used to recognise ONE table recorded under two
/// manifest slots. `bge --skip-etm` writes ρ to `{out}.feature_loading.parquet`
/// **and** `{out}.dictionary.parquet` — two distinct files with identical
/// content — so comparing paths, canonical or not, does not catch it, and the
/// report shows one table as two independent findings that happen to agree.
fn content_key(path: &std::path::Path) -> std::io::Result<u64> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::fs::read(path)?.hash(&mut h);
    Ok(h.finish())
}

/// Measure every geometry table the manifest records, in report order. Errors
/// when it records none — a run with nothing to measure is a wrong `--from`,
/// not an empty report.
///
/// WHICH slots are measurable is [`run_manifest::RunOutputs::geometry_tables`]'s
/// call, not this command's; all that is decided here is that a table is
/// measured once per distinct file.
pub(crate) fn collect_geometry(
    from: &str,
) -> anyhow::Result<Vec<(&'static str, EmbeddingGeometry)>> {
    let (manifest, dir) = run_manifest::load_for(from)?;

    let mut rows = Vec::new();
    let mut seen: Vec<u64> = Vec::new();
    for (name, rel) in manifest.outputs.geometry_tables() {
        let path = run_manifest::resolve(&dir, rel);
        let key =
            content_key(&path).map_err(|e| anyhow::anyhow!("{name} at {}: {e}", path.display()))?;
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let path = path.to_string_lossy();
        let table = DMatrix::<f32>::from_parquet(&path)
            .map_err(|e| anyhow::anyhow!("{name} at {path}: {e}"))?;
        rows.push((name, embedding_geometry(&table.mat)));
    }
    anyhow::ensure!(
        !rows.is_empty(),
        "{from}: the manifest records none of {}",
        run_manifest::GEOMETRY_TABLE_SLOTS.join(" / ")
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
