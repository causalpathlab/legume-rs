//! Resolve a per-gene embedding + reconstruction offset from an upstream run
//! manifest, abstracting over the two supported feature-embedding sources.
//!
//! **`senna bge` is the only supported source**, with or without `--skip-etm`.
//! It provides three things:
//!
//! - `feature_loading.parquet` — the per-gene loading ρ on the model axis, i.e.
//!   what multiplies the cell embedding in the Poisson log-rate. Recorded on both
//!   paths. Older runs kept ρ only under `--skip-etm`, where it borrowed the
//!   `dictionary` slot, so that legacy layout is still read as a fallback — and
//!   there the slot must be checked by CONTENT (a β has gene-log-simplex columns),
//!   never by which cell-side slots are populated, since `latent` and
//!   `cell_embedding` are interchangeable and have swapped roles between versions.
//! - `feature_embedding.parquet` — the co-embedding (genes on the cell manifold),
//!   whose marker-weighted centroids become the cell-type anchors.
//! - `feature_bias.parquet` — the per-gene offset `a_g`.
//!
//! The persisted ρ is the GATED snapshot (`materialize_e_feat` bakes the feature
//! gate into `e_feat`), i.e. already the loading that enters the Poisson rate —
//! do not correct for the gate separately.
//!
//! Topic-family runs are rejected up front; see [`EmbeddingSource::load`].

use crate::embed_common::{DVec, Mat};
use crate::run_manifest::{self, ArtifactScale, RunKind, RunManifest};
use anyhow::{Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{IoOps, MatWithNames};
use std::path::Path;

/// Everything the deconvolution needs from the upstream embedding run.
pub struct EmbeddingSource {
    /// `D×H` embedding used for the Poisson projection and the reconstruction
    /// `μ_{g,c} = exp(ρ_g·t_c + a_g)`.
    pub rho: Mat,
    /// `D×H` embedding whose marker-weighted centroids define the anchors:
    /// bge's co-embedding, i.e. genes placed on the cell manifold.
    pub anchor_emb: Mat,
    /// `D` per-gene log-offset `a_g` in the Poisson rate.
    pub gene_offset: DVec,
    /// `D` gene names (row order of `rho`).
    pub feature_names: Vec<Box<str>>,
    /// Embedding dimension `H`.
    pub h: usize,
    /// Source run kind (for logging).
    pub kind: RunKind,
}

impl EmbeddingSource {
    pub fn load(from: &str) -> Result<Self> {
        let (manifest, dir) = RunManifest::load(Path::new(from))?;
        info!(
            "deconvolve: loaded manifest ({from}): kind={}",
            manifest.kind
        );
        let resolve = |rel: &str| -> String {
            run_manifest::resolve(&dir, rel)
                .to_string_lossy()
                .into_owned()
        };

        match manifest.kind {
            RunKind::Bge => Self::from_bge(&manifest, &dir, &resolve),
            // Topic-family sources are DISABLED: benchmarked at Pearson 0.08
            // (noise) vs 0.99 for `bge --skip-etm` on identical data. Their ρ
            // pairs with the topic embeddings α under a softmax head, not with
            // cell-space positions under a Poisson rate, so the projection and
            // the `exp(ρ·t + a)` reconstruction both use the wrong likelihood.
            // Failing loudly beats returning plausible-looking noise.
            RunKind::Topic | RunKind::Itopic | RunKind::MaskedVae | RunKind::JointTopic => {
                anyhow::bail!(
                    "deconvolve: topic-family runs (`{}`) are not supported — the embedding-projection \
                     reference is invalid under a softmax-ETM head (benchmarked at r=0.08). Use \
                     `senna bge`.\n\nNote: this run already carries a better reference than \
                     the one deconvolve reconstructs — `dictionary_empirical.parquet` is a \
                     full-resolution per-topic gene simplex, and `dispersion.parquet` a per-gene NB \
                     dispersion. Consuming those directly is the planned rework.",
                    manifest.kind
                )
            }
            other => {
                anyhow::bail!("deconvolve: unsupported source kind `{other}` — use `senna bge`")
            }
        }
    }

    /// `bge`: the per-gene loading ρ + the co-embedding that grounds the anchors.
    ///
    /// Resolution is delegated to [`run_manifest::resolve_feature_loading_for`],
    /// the single place that knows where ρ can live and that verifies each
    /// candidate's scale. Duplicating that probe here is what let the same bug
    /// recur in three consumers.
    fn from_bge(m: &RunManifest, dir: &Path, resolve: &impl Fn(&str) -> String) -> Result<Self> {
        let coembed_rel = m.outputs.feature_embedding.as_deref().ok_or_else(|| {
            anyhow::anyhow!("bge manifest has no `outputs.feature_embedding` (co-embed anchors)")
        })?;
        let coembed = load_mat(&resolve(coembed_rel), "co-embedding")?;

        let (rho_path, bias_path) = run_manifest::resolve_feature_loading_for(m, dir)?;
        let rho = load_mat(&rho_path, "per-gene loading ρ")?;
        ArtifactScale::ensure(&rho.mat, ArtifactScale::Signed, &rho_path)?;

        // The Poisson rate needs `a_g`; without it μ would silently lose its
        // per-gene offset, so this is required rather than defaulted to zero.
        let bias_path = bias_path.ok_or_else(|| {
            anyhow::anyhow!("no feature_bias.parquet beside `{rho_path}` — deconvolve needs the per-gene offset a_g")
        })?;
        let gene_offset = load_mat(&bias_path, "feature_bias")?
            .mat
            .column(0)
            .into_owned();

        Self::assemble(rho, coembed.mat, gene_offset, RunKind::Bge)
    }

    fn assemble(
        rho: MatWithNames<Mat>,
        anchor_mat: Mat,
        gene_offset: DVec,
        kind: RunKind,
    ) -> Result<Self> {
        let h = rho.mat.ncols();
        anyhow::ensure!(
            anchor_mat.ncols() == h,
            "deconvolve: anchor embedding H={} != ρ H={h}",
            anchor_mat.ncols()
        );
        anyhow::ensure!(
            anchor_mat.nrows() == rho.mat.nrows(),
            "deconvolve: anchor genes ({}) != ρ genes ({})",
            anchor_mat.nrows(),
            rho.mat.nrows()
        );
        anyhow::ensure!(
            gene_offset.len() == rho.mat.nrows(),
            "deconvolve: gene offset ({}) != ρ genes ({})",
            gene_offset.len(),
            rho.mat.nrows()
        );
        info!(
            "deconvolve: ρ [{} genes × {h}], {kind} source",
            rho.mat.nrows()
        );
        Ok(Self {
            rho: rho.mat,
            anchor_emb: anchor_mat,
            gene_offset,
            feature_names: rho.rows,
            h,
            kind,
        })
    }
}

/// Read a matrix parquet with a descriptive error context.
fn load_mat(path: &str, what: &str) -> Result<MatWithNames<Mat>> {
    DMatrix::<f32>::from_parquet(path).with_context(|| format!("reading {what} {path}"))
}
