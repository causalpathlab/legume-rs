//! Marker-based cell-type calls attached to the trajectory's nodes.

use anyhow::{Context, Result};
use log::info;

use graph_embedding_util::type_annotation::{
    annotate_with_communities, CommunityCalls, InputEmbeddings, MarkerBootstrapConfig, Regroup,
    TermOraConfig,
};
use matrix_util::branching::Branching;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::principal_graph::kmeans_centroids_seeded;

use super::write::*;

/// Inputs for [`annotate_trajectory`] — bundled to keep the fan-in a struct.
pub(super) struct AnnotateTrajArgs<'a> {
    pub(super) prefix: &'a str,
    pub(super) out: &'a str,
    pub(super) markers: &'a str,
    /// Raw θ `[N × H]` — the same latent space `faba annotate` scores in.
    pub(super) raw_theta: &'a DMatrix<f32>,
    pub(super) cell_names: &'a [Box<str>],
    /// Per-cell MST-node id (the k-means `labels`) — the annotation clustering.
    pub(super) labels: &'a [usize],
    /// Number of MST nodes.
    pub(super) k: usize,
    pub(super) num_perm: usize,
    pub(super) obo: Option<&'a str>,
    pub(super) label_cl: Option<&'a str>,
    /// Stability bootstrap over the marker panel + the k-means grouping (`None` ⇒ point estimate).
    pub(super) bootstrap: Option<MarkerBootstrapConfig>,
    /// θ `[N × H]`, kept so a replicate can re-run k-means on it under a fresh seed.
    pub(super) theta: &'a DMatrix<f32>,
    pub(super) kmeans_iter: usize,
    pub(super) seed: u64,
}

/// Name each trajectory node by cell type: run the `faba annotate` term-ORA core over the
/// MST-node grouping (raw θ vs the gem β dictionary), giving every node a permutation-
/// calibrated call. Writes `{out}.lineage_annot.*` and returns the per-node
/// [`CommunityCalls`]. Run BEFORE root selection (it doesn't depend on the root) so
/// `--root-type` can pick the root from these calls; the caller writes
/// `{out}.trajectory_annotation.parquet` afterwards via [`write_trajectory_annotation`].
pub(super) fn compute_node_calls(a: &AnnotateTrajArgs) -> Result<CommunityCalls> {
    // The co-embedded feature vectors, not β — see `faba::gem::marker_embedding` for why a
    // Euclidean nearest-centroid call against β is not a well-posed question.
    let beta = faba::gem::marker_embedding::load_gene_embedding(
        a.prefix,
        faba::gem::marker_embedding::Modality::Spliced,
    )?;
    let cfg = TermOraConfig {
        n_perm: a.num_perm,
        // `--seed` drives the whole fit; it should drive the annotation's randomness too. It was
        // silently falling through to `TermOraConfig`'s own default of 42, so varying `--seed`
        // moved the centroids but left the permutation null (and now the bootstrap) untouched.
        min_markers: 3,
        seed: a.seed,
        obo: a.obo.map(str::to_owned),
        label_cl: a.label_cl.map(str::to_owned),
        panel_perm: 0,
        support_perm: 0,
        bootstrap: a.bootstrap.clone(),
        ..TermOraConfig::default()
    };
    let input = InputEmbeddings {
        feature_emb: &beta.mat,
        gene_names: &beta.rows,
        cell_emb: a.raw_theta,
        cell_names: a.cell_names,
    };

    // One replicate's grouping: the same k-means, reseeded.
    //
    // The trajectory's own nodes stay put — they are the structure, and `--seed` is meant to
    // reproduce them. This redraws the grouping *inside* the annotation bootstrap only, which is
    // what gives the resampling something to disagree about. Holding the partition fixed and
    // resampling the panel alone is close to a no-op: a node's argmax does not flip because a
    // few markers were redrawn, so every call comes back with support ≈ 1 and nothing abstains.
    // k-means++ from a fresh seed lands on genuinely different nodes, and a label that survives
    // *that* is a label worth printing on a trajectory.
    let regroup = |seed: u64| -> Result<Vec<usize>> {
        let (_, labels) = kmeans_centroids_seeded(a.theta, a.k, a.kmeans_iter, seed);
        Ok(labels)
    };
    let regroup: Option<&Regroup<'_>> = a.bootstrap.as_ref().map(|_| &regroup as &Regroup<'_>);

    annotate_with_communities(
        &input,
        a.markers,
        &format!("{}.lineage_annot", a.out),
        true, // IDF-weight markers, as `faba annotate` does by default
        a.labels,
        a.k,
        regroup,
        &cfg,
    )
}

/// Cross the per-node calls with the rooted forest → the labeled trajectory: one row per
/// node — `role` (root | terminal | internal), `cell_type`, `confidence`. Terminals are
/// derived from the rooted children (a node with no children), not from the orientation,
/// so abstained edges cannot misclassify a leaf.
pub(super) fn write_trajectory_annotation(
    calls: &CommunityCalls,
    br: &Branching,
    path: &str,
) -> Result<()> {
    let k = br.parent.len();
    let mut has_child = vec![false; k];
    for p in br.parent.iter().flatten() {
        if *p < k {
            has_child[*p] = true;
        }
    }
    let node_names = numbered("node_", k);
    let roles: Vec<Box<str>> = (0..k)
        .map(|node| {
            if br.parent[node].is_none() {
                Box::from("root")
            } else if !has_child[node] {
                Box::from("terminal")
            } else {
                Box::from("internal")
            }
        })
        .collect();
    write_named_table(
        path,
        "node",
        &node_names,
        &[
            (Box::from("role"), Column::Str(&roles)),
            (Box::from("cell_type"), Column::Str(&calls.labels)),
            (Box::from("confidence"), Column::F32(&calls.confidence)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("wrote {path} ({k} nodes; {} root(s))", br.roots.len());
    Ok(())
}

/////////////////////
// PHATE 2D layout //
/////////////////////
