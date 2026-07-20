//! Loading gem's **gene embedding** for marker annotation — the co-embedded
//! `{out}.feature_embedding.parquet`, not the `{out}.beta_feature_embedding.parquet` β.
//!
//! # Why not β
//!
//! Marker annotation is a **Euclidean nearest-centroid** call: it builds a prototype from a
//! type's marker genes and hands each cell to the closest one. That is only meaningful if the
//! genes and the cells live in the *same metric space*. β and θ do not.
//!
//! gem's likelihood couples them through an **inner product** (`⟨β_g, θ_c⟩`), which fixes their
//! relative *directions* but says nothing about their relative *scale* — the model is free to
//! make β small and θ large, and it does. Measured on a 15,315-cell cord-blood run:
//!
//! | table | rows | mean ‖row‖ | median ‖row‖ |
//! |---|---|---|---|
//! | `beta_feature_embedding` | 34,189 | 0.21 | **0.00** |
//! | `feature_embedding` | 3,695 | 3.11 | 3.05 |
//! | `cell_embedding` | 15,315 | 30.07 | 26.77 |
//!
//! Two things follow, and both are fatal for a distance-based call.
//!
//! 1. **The metric degenerates.** With cells ~140× longer than β rows,
//!    `‖x − c‖² = ‖x‖² − 2⟨x,c⟩ + ‖c‖²` loses its `‖c‖²` term entirely and `‖x‖²` is constant
//!    across types, so `argmin_t ‖x − c_t‖²` collapses to `argmax_t ⟨x, c_t⟩` — an *unnormalized*
//!    inner product in which a centroid's **norm** is a free parameter that decides the winner
//!    almost regardless of its direction. Measured: the rank correlation between a type's
//!    centroid norm and the share of cells it captures is **+0.93**, and the five types with the
//!    most scattered (i.e. least meaningful) panels captured **97.7%** of all cells.
//! 2. **Half of β is not there.** Its median row norm is **zero**: most genes were never trained
//!    and their post-hoc projection failed its null test, so they contribute nothing to a
//!    centroid while still counting as "matched" markers.
//!
//! `feature_embedding` is the model's own feature vectors — the ones actually fitted — and it is
//! what `pinto annotate` and `senna annotate-by-projection` have always used. `faba annotate` was
//! the odd one out.
//!
//! # The modality split
//!
//! gem's feature embedding is keyed by **feature row**, not by gene:
//! `ENSG00000000971_CFH/count/spliced`, `.../count/unspliced` — a spliced *and* an unspliced row
//! per gene. A marker panel names genes, so matching it against the raw table would silently pull
//! **both** rows into the same centroid and average the mature identity together with the nascent
//! one. [`load_gene_embedding`] therefore selects a single modality and strips the suffix back to
//! the gene key, which is what the marker matcher expects.

use anyhow::{Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{IoOps, MatWithNames};

/// Which feature rows to keep out of gem's feature embedding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Modality {
    /// `{gene}/count/spliced` — the mature identity program.
    Spliced,
    /// `{gene}/count/unspliced` — the nascent program.
    Unspliced,
}

impl Modality {
    fn suffix(self) -> &'static str {
        match self {
            Self::Spliced => "/count/spliced",
            Self::Unspliced => "/count/unspliced",
        }
    }
}

/// Read `{prefix}.feature_embedding.parquet`, keep only `modality`'s rows, and re-key them by
/// gene so a marker panel can match them.
///
/// Errors when the modality selects nothing — that is a real misconfiguration (a spliced-only gem
/// run has no unspliced rows) and silently annotating against an empty gene set would be worse.
pub fn load_gene_embedding(prefix: &str, modality: Modality) -> Result<MatWithNames<DMatrix<f32>>> {
    let path = format!("{prefix}.feature_embedding.parquet");
    let feat = DMatrix::<f32>::from_parquet(&path)
        .with_context(|| format!("reading gene embedding {path}"))?;

    let suffix = modality.suffix();
    let keep: Vec<usize> = feat
        .rows
        .iter()
        .enumerate()
        .filter(|(_, name)| name.ends_with(suffix))
        .map(|(i, _)| i)
        .collect();
    anyhow::ensure!(
        !keep.is_empty(),
        "{path} has no `{suffix}` feature rows (found {} rows, e.g. `{}`). A spliced-only \
         `faba gem` run has no unspliced program to annotate.",
        feat.rows.len(),
        feat.rows.first().map_or("", |s| s.as_ref())
    );

    let rows: Vec<Box<str>> = keep
        .iter()
        .map(|&i| {
            let name = feat.rows[i].as_ref();
            Box::from(name.strip_suffix(suffix).unwrap_or(name))
        })
        .collect();
    let mat = feat.mat.select_rows(&keep);
    info!(
        "gene embedding: {} of {} feature rows are `{suffix}` → {} genes [{} × {}]",
        keep.len(),
        feat.rows.len(),
        rows.len(),
        mat.nrows(),
        mat.ncols()
    );

    Ok(MatWithNames {
        rows,
        cols: feat.cols,
        mat,
    })
}
