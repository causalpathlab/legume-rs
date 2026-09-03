//! Resolve the gene axis and the archetype inputs from an upstream run manifest.
//!
//! **`senna bge` is the only supported source**, with or without `--skip-etm`.
//! What is taken from it:
//!
//! - the gene axis and embedding width, from `feature_loading.parquet` — the
//!   per-gene loading ρ. Older runs kept ρ only under `--skip-etm`, where it
//!   borrowed the `dictionary` slot, so that legacy layout is read as a
//!   fallback. There the slot must be checked by CONTENT (a β has
//!   gene-log-simplex columns), never by which cell-side slots are populated,
//!   since `latent` and `cell_embedding` are interchangeable and have swapped
//!   roles between versions.
//! - the cell embedding, which the archetypes are clustered in.
//! - the input counts and the cell annotation, which the profiles are built from.
//!
//! Only ρ's row names and width are retained — the reference is measured from
//! the counts rather than reconstructed — but its values are still scale-checked,
//! so a manifest pointing at a topic dictionary is caught rather than silently
//! supplying the wrong gene axis.
//!
//! Topic-family runs are rejected up front; see [`EmbeddingSource::load`].

use crate::embed_common::Mat;
use crate::run_manifest::{self, ArtifactScale, RunKind, RunManifest};
use anyhow::{Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{IoOps, MatWithNames};
use std::path::Path;

/// Everything the deconvolution needs from the upstream embedding run.
pub struct EmbeddingSource {
    /// `D` gene names, defining the panel the bulk is aligned to.
    pub feature_names: Vec<Box<str>>,
    /// Embedding dimension `H`, used to pick the cell embedding.
    pub h: usize,
    /// Source run kind (for logging).
    pub kind: RunKind,
    /// Candidate `N×H` cell-embedding paths from the manifest, most specific
    /// first. The archetypes are clustered in the first whose width matches `H`;
    /// older runs recorded Z under `latent` instead of `cell_embedding`.
    pub cell_embedding_paths: Vec<String>,
    /// Count matrices the source run was trained on — the archetype profile
    /// source when `--sc-data` is not given.
    pub data_files: Vec<Box<str>>,
    /// Cell annotation recorded by the manifest, if any.
    pub annotation_path: Option<String>,
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

        let mut built = match manifest.kind {
            RunKind::Bge => Self::from_bge(&manifest, &dir),
            RunKind::Simba => anyhow::bail!(
                "deconvolve: `simba` runs are not supported (no gene bias, no projector); use `senna bge`."
            ),
            // Topic-family sources are not accepted. The archetype reference
            // needs only a gene axis, a cell embedding, the counts and an
            // annotation, all of which a topic run has, so this is a matter of
            // the path being unverified rather than a structural bar. Failing
            // loudly beats returning plausible-looking numbers.
            RunKind::Topic | RunKind::Itopic | RunKind::MaskedVae | RunKind::JointTopic => {
                anyhow::bail!(
                    "deconvolve: topic-family runs (`{}`) are not supported; use `senna bge`.\n\n\
                     The archetype reference needs a gene axis, a cell embedding, the counts and \
                     an annotation, which a topic run does have — this path is simply unverified. \
                     Such a run also carries `dictionary_empirical.parquet`, a per-topic gene \
                     simplex, and `dispersion.parquet`, a per-gene NB dispersion; consuming those \
                     directly is the planned rework.",
                    manifest.kind
                )
            }
            other => {
                anyhow::bail!("deconvolve: unsupported source kind `{other}` — use `senna bge`")
            }
        }?;

        built.cell_embedding_paths = manifest
            .outputs
            .cell_embedding
            .as_deref()
            .into_iter()
            .chain(manifest.outputs.latent.as_deref())
            .map(&resolve)
            .collect();
        built.data_files = manifest
            .data
            .input
            .iter()
            .map(|f| resolve(f).into_boxed_str())
            .collect();
        built.annotation_path = manifest.annotate.annotation.as_deref().map(&resolve);
        Ok(built)
    }

    /// `bge`: the gene axis and the embedding width.
    ///
    /// Resolution is delegated to [`run_manifest::resolve_feature_loading_for`],
    /// the single place that knows where ρ can live and that verifies each
    /// candidate's scale. Duplicating that probe here is what let the same bug
    /// recur in three consumers.
    fn from_bge(m: &RunManifest, dir: &Path) -> Result<Self> {
        let (rho_path, _) = run_manifest::resolve_feature_loading_for(m, dir)?;
        let rho = load_mat(&rho_path, "per-gene loading ρ")?;
        ArtifactScale::ensure(&rho.mat, ArtifactScale::Signed, &rho_path)?;
        Self::assemble(rho, RunKind::Bge)
    }

    /// Keep the gene axis and the width; the values are not needed.
    fn assemble(rho: MatWithNames<Mat>, kind: RunKind) -> Result<Self> {
        let h = rho.mat.ncols();
        info!(
            "deconvolve: gene axis [{} genes], embedding H={h}, {kind} source",
            rho.mat.nrows()
        );
        Ok(Self {
            feature_names: rho.rows,
            h,
            kind,
            cell_embedding_paths: Vec::new(),
            data_files: Vec::new(),
            annotation_path: None,
        })
    }
}

/// Read a matrix parquet with a descriptive error context.
fn load_mat(path: &str, what: &str) -> Result<MatWithNames<Mat>> {
    DMatrix::<f32>::from_parquet(path).with_context(|| format!("reading {what} {path}"))
}
