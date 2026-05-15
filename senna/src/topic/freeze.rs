//! Helper for the `--freeze-feature-embedding` flow shared by
//! `senna {indexed-topic, cell-embedded-topic}`.
//!
//! Two-phase: the senna data loader applies a feature mask BEFORE we
//! know the post-load gene order, but we need the post-load gene order
//! to materialize the frozen `ρ` matrix in the right row order. So:
//!
//! 1. [`FrozenFeatureSpec::resolve_from_prefix`] probes the on-disk
//!    layout (gbe-style `dictionary.parquet` + `feature_bias.parquet`,
//!    or topic-style `feature_embedding.parquet` alone) and caches the
//!    paths.
//! 2. [`FrozenFeatureSpec::mask_fn`] returns a feature-mask closure for
//!    `load_and_collapse`. The closure lazy-loads the source dictionary's
//!    canonical-name set on first invocation.
//! 3. After loading completes, [`FrozenFeatureSpec::materialize`] reads
//!    the source dictionary again to build the `[|keep|, H]` host
//!    matrix in the post-mask gene order (which equals the order
//!    `data_vec.row_names()` returns after masking).

use auxiliary_data::feature_names::FeatureNameKind;
use auxiliary_data::frozen_features::{
    load_frozen_feature_host, FrozenFeatureHost, FrozenLoadArgs,
};
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

pub struct FrozenFeatureSpec {
    pub dictionary_path: String,
    pub bias_path: Option<String>,
    pub name_kind: FeatureNameKind,
    /// Cached source canonical-name set, populated lazily on first
    /// `mask_fn` invocation. `Rc<RefCell<...>>` so the closure can share
    /// the cache with the owning spec without taking `&mut self`.
    source_canon: Rc<RefCell<Option<FxHashSet<Box<str>>>>>,
}

impl FrozenFeatureSpec {
    /// Probe `{prefix}.dictionary.parquet` + `{prefix}.feature_bias.parquet`
    /// (gbe layout); fall back to `{prefix}.feature_embedding.parquet`
    /// (topic / cell-embedded-topic layout, bias defaults to zero).
    /// Errors if neither layout is present.
    pub fn resolve_from_prefix(prefix: &str, name_kind: FeatureNameKind) -> anyhow::Result<Self> {
        let gbe_dict = format!("{prefix}.dictionary.parquet");
        let gbe_bias = format!("{prefix}.feature_bias.parquet");
        let topic_dict = format!("{prefix}.feature_embedding.parquet");

        let (dictionary_path, bias_path) = if Path::new(&gbe_dict).exists() {
            let bias = if Path::new(&gbe_bias).exists() {
                Some(gbe_bias)
            } else {
                log::warn!(
                    "{} found but {} missing — loading dictionary only, bias defaults to zero",
                    gbe_dict,
                    gbe_bias
                );
                None
            };
            (gbe_dict, bias)
        } else if Path::new(&topic_dict).exists() {
            log::info!(
                "Frozen feature side: loading topic-style {} (bias = 0)",
                topic_dict
            );
            (topic_dict, None)
        } else {
            anyhow::bail!(
                "--freeze-feature-embedding {prefix}: neither {prefix}.dictionary.parquet \
                 nor {prefix}.feature_embedding.parquet found"
            );
        };
        Ok(Self {
            dictionary_path,
            bias_path,
            name_kind,
            source_canon: Rc::new(RefCell::new(None)),
        })
    }

    /// Build a feature-mask closure suitable for `topic::common::FeatureMaskFn`.
    /// Lazy-loads the source dict's row-name canonical set on first
    /// invocation; subsequent calls reuse the cache.
    pub fn mask_fn(&self) -> Box<crate::topic::common::FeatureMaskFn> {
        let kind = self.name_kind.clone();
        let canon_cell = self.source_canon.clone();
        let dict_path = self.dictionary_path.clone();
        Box::new(move |row_names: &[Box<str>]| -> anyhow::Result<Vec<bool>> {
            if canon_cell.borrow().is_none() {
                let dict = <DMatrix<f32> as IoOps>::from_parquet(&dict_path)?;
                let mut set: FxHashSet<Box<str>> = FxHashSet::default();
                for name in dict.rows.iter() {
                    set.insert(kind.canonicalize(name));
                }
                log::info!(
                    "Frozen feature side: {} canonical names loaded from {}",
                    set.len(),
                    dict_path
                );
                *canon_cell.borrow_mut() = Some(set);
            }
            let borrow = canon_cell.borrow();
            let set = borrow.as_ref().unwrap();
            Ok(row_names
                .iter()
                .map(|n| set.contains(&kind.canonicalize(n)))
                .collect())
        })
    }

    /// After `load_and_collapse`, build the host-side `[|keep|, H]`
    /// matrix and bias vector reordered to follow `post_load_gene_names`.
    /// The source-axis intersection is total in the well-formed case
    /// (every surviving gene was kept precisely because it had a source
    /// row), so `keep_target_indices` should equal `0..post_load_gene_names.len()`.
    pub fn materialize(
        &self,
        post_load_gene_names: &[Box<str>],
    ) -> anyhow::Result<FrozenFeatureHost> {
        load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &self.dictionary_path,
            bias_path: self.bias_path.as_deref(),
            target_feature_names: post_load_gene_names,
            name_kind: self.name_kind.clone(),
        })
    }
}
