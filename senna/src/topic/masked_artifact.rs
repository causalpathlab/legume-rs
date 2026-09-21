//! What a masked model **is** on disk — declared once, here.
//!
//! Before this module the definition was implicit and scattered: each consumer re-derived
//! how to open a model, and nothing connected them. A file added to training and forgotten
//! elsewhere produced a model that loaded fine in `predict` and failed somewhere else, much
//! later, with an unrelated-looking error.
//!
//! The fix is a single **declaration** ([`REQUIRED`]) plus a reader that refuses anything
//! incomplete ([`MaskedModel::open`]). Any writer that drifts from the declaration is caught
//! at the next open, by name, instead of degrading silently.
//!
//! **When `probe` grows to cover more families,** each family
//! gets its own module of this shape — their artifacts genuinely differ (bge has no
//! checkpoint or metadata at all; dense `topic` has `fisher_weights` and no feature
//! embedding), so a shared trait would be inventing commonality that is not there. What
//! *is* common is the pattern, and its one reusable piece is [`require_files`]; move
//! that to a neutral home when the second family lands, not before.

use crate::predict::{score_masked_backend, MaskedScoreArgs, MaskedScored};
use crate::topic::eval::QueryNameOpts;
use crate::topic::model_metadata::{
    load_feature_mean, load_shortlist_weights, masked_head_from_model_type, TopicModelMetadata,
};
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::vae::masked_topic::LatentHead;

/// Every file a masked model must have for *all* of its consumers to work.
///
/// `predict`/`probe` need the safetensors, the metadata, and the three per-gene
/// parquets; `--freeze-feature-embedding` needs the feature embedding. Anything that
/// writes a masked model writes all of these — see [`write_masked_model`], and the test
/// at the bottom of this file which pins the list against a real training run.
///
/// Deliberately **not** listed: `dictionary_empirical`, `pb_gene`, `pb_latent`,
/// `latent`, `cell_proj`. Those come from the pseudobulk collapse, which only training
/// performs. Consumers
/// fall back (`plot-topic` does `dictionary_empirical.or(dictionary)`), which is the
/// same fallback `vae` and `joint-topic` already rely on.
pub const REQUIRED: &[&str] = &[
    "model.json",
    "safetensors",
    "dictionary.parquet",
    "feature_embedding.parquet",
    "feature_mean.parquet",
];

/// Required IN ADDITION, and only for a model that recorded a context window.
///
/// Those weights ranked the top-K such an encoder read, so the model cannot be
/// scored without them. Nothing this build trains has a window, and nothing
/// writes the file — requiring it unconditionally would make every new model
/// unopenable.
pub const REQUIRED_WINDOWED: &[&str] = &["shortlist_weights.parquet"];

/// Assert `{prefix}.{suffix}` exists for every required suffix, naming all the missing
/// ones at once.
///
/// Family-agnostic: `bge`, dense `topic` and `vae` artifact modules will want exactly
/// this when `probe`/`update` reach them. Move it somewhere neutral at that point.
pub(crate) fn require_files(prefix: &str, required: &[&str], what: &str) -> anyhow::Result<()> {
    let missing: Vec<String> = required
        .iter()
        .map(|suffix| format!("{prefix}.{suffix}"))
        .filter(|path| !std::path::Path::new(path).exists())
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{prefix} is not a complete {what} — missing {}. Every writer must emit the full \
         set; a model missing these loads in some subcommands and fails in others.",
        missing.join(", ")
    );
    Ok(())
}

/// An opened, validated masked model: its metadata, latent head, and gene axis.
///
/// Construction is the single gate every consumer passes through, so the "is this a
/// masked model, and is it complete?" question is answered in one place instead of at
/// each call site.
pub struct MaskedModel<'a> {
    pub prefix: &'a str,
    pub metadata: TopicModelMetadata,
    pub head: LatentHead,
    pub feature_mean: Vec<f32>,
    /// The shortlist an OLD model was fitted with; `None` for a window-free
    /// model, which has nothing to rank.
    pub shortlist: Option<Vec<f32>>,
}

impl<'a> MaskedModel<'a> {
    /// Open and validate. Fails with the missing filenames rather than letting a
    /// half-written model surface as a confusing error later.
    pub fn open(prefix: &'a str) -> anyhow::Result<Self> {
        require_files(prefix, REQUIRED, "masked model")?;

        let metadata = TopicModelMetadata::load(prefix)?;
        let head = masked_head_from_model_type(&metadata.model_type).ok_or_else(|| {
            anyhow::anyhow!(
                "{prefix} is a '{}' model; masked models only (masked-topic/-vae/-sbp)",
                metadata.model_type
            )
        })?;
        let (_gene_names, feature_mean) = load_feature_mean(prefix)?;
        let shortlist = match metadata.enc_context_size {
            None => None,
            Some(_) => {
                require_files(
                    prefix,
                    REQUIRED_WINDOWED,
                    "masked model with a context window",
                )?;
                Some(load_shortlist_weights(prefix)?.1)
            }
        };

        Ok(Self {
            prefix,
            metadata,
            head,
            feature_mean,
            shortlist,
        })
    }

    /// Encoder-only scoring on `files`, with the exact-name matching that `probe` and
    /// `update` both want (`predict` keeps its own user-configurable naming).
    ///
    /// `need_llik` is the caller's: the per-cell predictive score costs a second full
    /// pass over every column plus a dense `[D, minibatch]` reconstruction per block, so
    /// callers that want only `z_nk` should pass `false`.
    pub fn score(
        &self,
        files: &[Box<str>],
        preload: bool,
        minibatch_size: usize,
        need_llik: bool,
        dev: &Device,
    ) -> anyhow::Result<MaskedScored> {
        let qopts = QueryNameOpts::default();
        score_masked_backend(MaskedScoreArgs {
            model: self.prefix,
            data_files: files,
            batch_files: None,
            // `probe` takes no batch files, so there is never a δ to iterate.
            block_size: None,
            delta_iters: 0,
            // No δ. probe compares a calibration arm against a query arm, and δ is
            // fitted from whichever arm it is given — it would absorb exactly the
            // compositional novelty probe is looking for, and only on the arm with
            // more than one inferred batch. Both arms must be scored by the same
            // frozen model.
            estimate_batch_delta: false,
            preload,
            minibatch_size,
            query_name_opts: &qopts,
            metadata: &self.metadata,
            head: self.head,
            need_llik,
            dev,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write just enough of a masked model for [`MaskedModel::open`] to get as
    /// far as the file check: a real `model.json` and a real
    /// `feature_mean.parquet`, the rest present but empty.
    fn skeleton(prefix: &str, enc_context_size: Option<usize>, with_shortlist: bool) {
        let genes: Vec<Box<str>> = (0..4).map(|i| format!("G{i}").into()).collect();
        let m = TopicModelMetadata {
            model_type: crate::topic::model_metadata::MODEL_TYPE_MASKED_VAE.into(),
            decoder_types: vec!["nb".into()],
            decoder_weights: vec![1.0],
            n_features_encoder: 4,
            n_features_full: 4,
            n_topics: 2,
            encoder_hidden: vec![8],
            num_levels: 1,
            level_decoder_dims: vec![4],
            adj_method: "residual".into(),
            has_coarsening: false,
            embedding_dim: Some(4),
            enc_context_size,
            theta_mean: None,
            n_train_cells: None,
            n_gene_modules: None,
            query_rank: None,
        };
        m.save(prefix).unwrap();
        crate::topic::model_metadata::save_feature_mean(&[1.0, 1.0, 1.0, 1.0], &genes, prefix)
            .unwrap();
        for suffix in [
            "safetensors",
            "dictionary.parquet",
            "feature_embedding.parquet",
        ] {
            std::fs::write(format!("{prefix}.{suffix}"), b"").unwrap();
        }
        if with_shortlist {
            use legume_numeric::matrix::traits::IoOps;
            let cols: Vec<Box<str>> = vec!["weight".into()];
            nalgebra::DMatrix::<f32>::from_column_slice(4, 1, &[1.0, 1.0, 1.0, 1.0])
                .to_parquet_with_names(
                    &format!("{prefix}.shortlist_weights.parquet"),
                    (Some(&genes), Some("gene")),
                    Some(&cols),
                )
                .unwrap();
        }
    }

    /// A window-free model has no shortlist: nothing ranked a context window,
    /// because there is none. Requiring the file would make every model this
    /// build writes unopenable by `predict` and `probe`.
    #[test]
    fn a_window_free_model_needs_no_shortlist_weights() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("dense").to_string_lossy().into_owned();
        skeleton(&prefix, None, false);
        let m = MaskedModel::open(&prefix).expect("a window-free model is complete without it");
        assert!(m.shortlist.is_none(), "no window, no shortlist");
        assert!(m.metadata.enc_context_size.is_none());
    }

    /// An OLD model still needs its shortlist: the window it recorded is scored
    /// by those weights, and reading it without them would be a different model.
    #[test]
    fn a_windowed_model_still_requires_its_shortlist_weights() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("windowed").to_string_lossy().into_owned();
        skeleton(&prefix, Some(512), false);
        let msg = match MaskedModel::open(&prefix) {
            Ok(_) => panic!("a windowed model without its weights is incomplete"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("shortlist_weights.parquet"),
            "the error must name the missing file; got: {msg}"
        );

        skeleton(&prefix, Some(512), true);
        let m = MaskedModel::open(&prefix).expect("with the weights it opens");
        assert_eq!(m.shortlist.as_deref().map(<[f32]>::len), Some(4));
    }

    #[test]
    fn open_names_every_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let prefix = dir.path().join("nothing_here");
        let prefix = prefix.to_str().expect("utf8");

        // `MaskedModel` is not `Debug` (it carries the whole gene axis), so match on the
        // Result rather than `expect_err`.
        let msg = match MaskedModel::open(prefix) {
            Ok(_) => panic!("an empty prefix is not a model"),
            Err(e) => e.to_string(),
        };
        for suffix in REQUIRED {
            assert!(
                msg.contains(suffix),
                "the error must name {suffix} so the writer that dropped it is obvious; got: {msg}"
            );
        }
    }
}
