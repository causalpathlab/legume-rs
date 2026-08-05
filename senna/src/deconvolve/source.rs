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
//! Topic-family runs are rejected; see [`EmbeddingSource::from_masked_topic`].

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
    /// True when the projection geometry is exact. Always true today (bge is the
    /// only supported source); retained for a future source whose geometry is not.
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

    /// `bge`: raw ρ + the co-embedding that grounds the marker anchors.
    ///
    /// ρ comes from `outputs.feature_loading` when the run recorded it (any bge run,
    /// with or without `--skip-etm`). Runs written before that field existed only
    /// kept ρ under `--skip-etm`, where it borrowed the `dictionary` slot — hence
    /// the fallback, which must then verify the slot really holds ρ and not β.
    fn from_bge(m: &RunManifest, resolve: &impl Fn(&str) -> String) -> Result<Self> {
        let coembed_rel = m.outputs.feature_embedding.as_deref().ok_or_else(|| {
            anyhow::anyhow!("bge manifest has no `outputs.feature_embedding` (co-embed anchors)")
        })?;
        let coembed = load_mat(&resolve(coembed_rel), "co-embedding")?;

        let (rho, rho_path) = match m.outputs.feature_loading.as_deref() {
            Some(rel) => {
                let p = resolve(rel);
                (load_mat(&p, "raw ρ")?, p)
            }
            None => {
                let dict_rel = m.outputs.dictionary.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "bge manifest records neither `outputs.feature_loading` nor `outputs.dictionary`"
                    )
                })?;
                let p = resolve(dict_rel);
                let rho = load_mat(&p, "raw ρ (legacy dictionary slot)")?;
                // Legacy layout only: discriminate on CONTENT, never on which
                // cell-side slots are populated (`latent` / `cell_embedding` are
                // interchangeable and have swapped roles between bge versions).
                // β is `log_softmax` over genes, so each of its columns
                // exponentiates to 1; a raw embedding ρ never does.
                anyhow::ensure!(
                    !is_log_simplex_columns(&rho.mat),
                    "deconvolve: `{p}` holds an ETM β (its columns are gene simplexes), not the \
                     raw Poisson ρ, and this run predates `outputs.feature_loading`. Re-run \
                     `senna bge` (any recent build records ρ on both paths)."
                );
                (rho, p)
            }
        };

        // feature_bias sits beside ρ; it is not recorded in the manifest.
        let bias_path = rho_path
            .strip_suffix(".feature_loading.parquet")
            .or_else(|| rho_path.strip_suffix(".dictionary.parquet"))
            .map(|stem| format!("{stem}.feature_bias.parquet"))
            .ok_or_else(|| anyhow::anyhow!("cannot derive feature_bias path from `{rho_path}`"))?;
        let gene_offset = load_mat(&bias_path, "feature_bias")?
            .mat
            .column(0)
            .into_owned();

        Self::assemble(rho, coembed.mat, gene_offset, RunKind::Bge, true)
    }

    /// `masked-topic` / topic-family. **Disabled**: benchmarked at Pearson 0.08
    /// (noise) against a `bge` run's 0.99 on identical data.
    ///
    /// This is not a mild approximation. A topic model's ρ pairs with the TOPIC
    /// embeddings α under a softmax head (`β = log_softmax_d(ρ·αᵀ)`); it does not
    /// pair with cell-space positions under a Poisson rate, so both the
    /// `project_cells` bulk projection and the `exp(ρ·t + a)` reconstruction are
    /// applying the wrong likelihood.
    ///
    /// The right rework is not to fix the projection but to skip it: for a topic
    /// run each β column already IS a per-type gene simplex (`Σ_g exp(β) = 1`) —
    /// precisely BayesPrism's normalized reference — so the reference should be
    /// read straight off β, with markers used only to map topics onto cell types.
    /// Kept behind this guard rather than deleted so that rework has a home.
    #[allow(dead_code)]
    fn from_masked_topic(m: &RunManifest, resolve: &impl Fn(&str) -> String) -> Result<Self> {
        let feat_rel = m.outputs.feature_embedding.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "manifest has no `outputs.feature_embedding` — a plain `topic` run has no per-gene \
                 embedding ρ. Use `senna bge`."
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
        // Topic-family runs store LOG β (`log_softmax` over genes); a few paths
        // store β directly. Detect which, then take the gene marginal under
        // uniform topics — in log space that is
        // `logsumexp_k(logβ[g,k]) − ln K`, NOT `ln(mean_k logβ)` (whose argument
        // is negative, which silently yields NaN and a degenerate reference).
        let k = beta.mat.ncols().max(1) as f32;
        let gene_offset = if is_log_simplex_columns(&beta.mat) {
            DVec::from_iterator(
                beta.mat.nrows(),
                beta.mat.row_iter().map(|r| {
                    let mx = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let se: f32 = r.iter().map(|&v| (v - mx).exp()).sum();
                    mx + se.max(1e-30).ln() - k.ln()
                }),
            )
        } else {
            DVec::from_iterator(
                beta.mat.nrows(),
                beta.mat.row_iter().map(|r| (r.sum() / k + 1e-8).ln()),
            )
        };
        anyhow::ensure!(
            gene_offset.iter().all(|v| v.is_finite()),
            "deconvolve: masked-topic gene offset is not finite — the β dictionary at `{}` is \
             neither log-simplex nor probability-scaled",
            resolve(dict_rel)
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
