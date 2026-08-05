//! Resolve a per-gene embedding + reconstruction offset from an upstream run
//! manifest, abstracting over the two supported feature-embedding sources.
//!
//! - **`bge --skip-etm`** (exact): the raw Poisson ρ = `e_feat` is persisted as
//!   `dictionary.parquet`; the co-embedding (genes on the cell manifold) is
//!   `feature_embedding.parquet` and grounds the marker anchors; the per-gene
//!   Poisson offset is `feature_bias.parquet`. An ETM-resolved run is rejected by
//!   inspecting the dictionary's CONTENT (β columns are gene log-simplexes) rather
//!   than which manifest slots are populated — `latent` and `cell_embedding` are
//!   interchangeable cell-side slots whose usage has flipped between bge versions,
//!   so neither identifies the layout. The persisted ρ is the GATED snapshot
//!   (`materialize_e_feat` bakes the feature gate into `e_feat`), i.e. the loading
//!   that actually enters the Poisson rate, so the reconstruction stays consistent
//!   with the model — do not correct for the gate separately.
//! - **`masked-topic`** (approximate): `feature_embedding.parquet` IS the raw
//!   learned ρ `[D,H]` (used for both projection and anchors); ρ was trained
//!   under a softmax-ETM head, so projecting it through the Poisson solver is a
//!   transfer approximation. The gene offset is the log topic-averaged gene
//!   marginal from the β `dictionary.parquet`.

use crate::embed_common::{DVec, Mat};
use crate::run_manifest::{self, RunKind, RunManifest};
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
    /// `D×H` embedding whose marker-weighted centroids define the anchors
    /// (co-embedding for bge; identical to `rho` for masked-topic).
    pub anchor_emb: Mat,
    /// `D` per-gene log-offset `a_g` in the Poisson rate.
    pub gene_offset: DVec,
    /// `D` gene names (row order of `rho`).
    pub feature_names: Vec<Box<str>>,
    /// Embedding dimension `H`.
    pub h: usize,
    /// Source run kind (for logging).
    pub kind: RunKind,
    /// True when the projection geometry is exact (bge) vs. approximate (masked-topic).
    pub exact: bool,
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
            RunKind::Bge => Self::from_bge(&manifest, &resolve),
            // masked-vae included: this path reads β + ρ only, and its β is the
            // same gene-simplex the other topic-family kinds produce. The
            // Gaussian latent never enters here.
            RunKind::Topic | RunKind::Itopic | RunKind::MaskedVae | RunKind::JointTopic => {
                Self::from_masked_topic(&manifest, &resolve)
            }
            other => anyhow::bail!(
                "deconvolve: unsupported source kind `{other}` — use `senna bge --skip-etm` \
                 (exact) or `senna masked-topic` (approximate)"
            ),
        }
    }

    /// `bge --skip-etm`: dictionary = raw ρ, feature_embedding = co-embed anchors.
    fn from_bge(m: &RunManifest, resolve: &impl Fn(&str) -> String) -> Result<Self> {
        let dict_rel =
            m.outputs.dictionary.as_deref().ok_or_else(|| {
                anyhow::anyhow!("bge manifest has no `outputs.dictionary` (raw ρ)")
            })?;
        let coembed_rel = m.outputs.feature_embedding.as_deref().ok_or_else(|| {
            anyhow::anyhow!("bge manifest has no `outputs.feature_embedding` (co-embed anchors)")
        })?;

        let dict_path = resolve(dict_rel);
        let rho = load_mat(&dict_path, "raw ρ (dictionary)")?;
        let coembed = load_mat(&resolve(coembed_rel), "co-embedding")?;
        // Discriminate on CONTENT, not on which manifest fields are populated:
        // `latent` / `cell_embedding` are interchangeable cell-side slots whose
        // usage has flipped between bge versions, so neither identifies the
        // layout. β is `log_softmax` over genes, so each of its columns
        // exponentiates to 1; a raw embedding ρ never does.
        anyhow::ensure!(
            !is_log_simplex_columns(&rho.mat),
            "deconvolve: `{dict_path}` holds an ETM β (its columns are gene simplexes), not the \
             raw Poisson ρ. Re-run `senna bge --skip-etm` so the raw ρ is persisted."
        );
        // feature_bias sits beside the dictionary; it is not recorded in the manifest.
        let bias_path = sibling(&dict_path, ".dictionary.parquet", ".feature_bias.parquet")?;
        let gene_offset = load_mat(&bias_path, "feature_bias")?
            .mat
            .column(0)
            .into_owned();

        Self::assemble(rho, coembed.mat, gene_offset, RunKind::Bge, true)
    }

    /// `masked-topic`: feature_embedding = raw ρ (also the anchor space); the
    /// per-gene offset is the log topic-averaged marginal of β.
    fn from_masked_topic(m: &RunManifest, resolve: &impl Fn(&str) -> String) -> Result<Self> {
        let feat_rel = m.outputs.feature_embedding.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "manifest has no `outputs.feature_embedding` — a plain `topic` run has no per-gene \
                 embedding ρ. Train with `senna masked-topic` (or use `bge --skip-etm`)."
            )
        })?;
        let rho = load_mat(&resolve(feat_rel), "feature embedding ρ")?;

        // Offset a_g = ln(mean_k β[g,k]): the gene marginal under uniform topics.
        let dict_rel = m.outputs.dictionary.as_deref().ok_or_else(|| {
            anyhow::anyhow!("masked-topic manifest has no `outputs.dictionary` (β) for the offset")
        })?;
        let beta = load_mat(&resolve(dict_rel), "β dictionary")?;
        anyhow::ensure!(
            beta.mat.nrows() == rho.mat.nrows(),
            "masked-topic: β genes ({}) != ρ genes ({})",
            beta.mat.nrows(),
            rho.mat.nrows()
        );
        let k = beta.mat.ncols().max(1) as f32;
        let gene_offset = DVec::from_iterator(
            beta.mat.nrows(),
            beta.mat.row_iter().map(|r| (r.sum() / k + 1e-8).ln()),
        );

        // Anchors share ρ's space; clone the matrix for the (identical) anchor role.
        let anchor_mat = rho.mat.clone();
        Self::assemble(rho, anchor_mat, gene_offset, RunKind::Topic, false)
    }

    fn assemble(
        rho: MatWithNames<Mat>,
        anchor_mat: Mat,
        gene_offset: DVec,
        kind: RunKind,
        exact: bool,
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
            "deconvolve: ρ [{} genes × {h}], {} source{}",
            rho.mat.nrows(),
            kind,
            if exact {
                ""
            } else {
                " (approximate projection)"
            }
        );
        Ok(Self {
            rho: rho.mat,
            anchor_emb: anchor_mat,
            gene_offset,
            feature_names: rho.rows,
            h,
            kind,
            exact,
        })
    }
}

/// Read a matrix parquet with a descriptive error context.
fn load_mat(path: &str, what: &str) -> Result<MatWithNames<Mat>> {
    DMatrix::<f32>::from_parquet(path).with_context(|| format!("reading {what} {path}"))
}

/// True when every column is a log-simplex over rows (`Σ_g exp(x) ≈ 1`) — the
/// signature of an ETM β dictionary (`log_softmax` over genes), which a raw
/// embedding ρ never satisfies.
pub(super) fn is_log_simplex_columns(m: &Mat) -> bool {
    m.ncols() > 0
        && (0..m.ncols()).all(|k| {
            let sum: f64 = m.column(k).iter().map(|&v| f64::from(v).exp()).sum();
            (sum - 1.0).abs() < 1e-2
        })
}

/// Derive a sibling artifact path by swapping a known suffix on the last path
/// component (e.g. `foo.dictionary.parquet` → `foo.feature_bias.parquet`).
fn sibling(path: &str, from_suffix: &str, to_suffix: &str) -> Result<String> {
    path.strip_suffix(from_suffix)
        .map(|stem| format!("{stem}{to_suffix}"))
        .ok_or_else(|| anyhow::anyhow!("expected `{from_suffix}` suffix on `{path}`"))
}
