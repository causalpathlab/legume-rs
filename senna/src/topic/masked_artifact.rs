//! What a masked model **is** on disk — declared once, here.
//!
//! Before this module the definition was implicit and scattered: `masked_topic`'s
//! output phase wrote one set of files, `update` wrote another, `probe` and `update`
//! each re-derived how to open one, and nothing connected them. A file added to
//! training and forgotten in `update` produced a child that loaded fine in `predict`
//! and failed somewhere else, much later, with an unrelated-looking error.
//!
//! The fix is not a single writer — training's writes are legitimately interleaved with
//! its holdout eval and device move, so forcing them together would change behaviour.
//! It is a single **declaration** ([`REQUIRED`]) plus a reader that refuses anything
//! incomplete ([`MaskedModel::open`]). Any writer that drifts from the declaration is
//! caught at the next open, by name, instead of degrading silently.
//!
//! **When `probe`/`update` grow to cover `bge`, dense `topic` and `vae`,** each family
//! gets its own module of this shape — their artifacts genuinely differ (bge has no
//! checkpoint or metadata at all; dense `topic` has `fisher_weights` and no feature
//! embedding), so a shared trait would be inventing commonality that is not there. What
//! *is* common is the pattern, and its one reusable piece is [`require_files`]; move
//! that to a neutral home when the second family lands, not before.

use crate::predict::{score_masked_backend, MaskedScoreArgs, MaskedScored};
use crate::topic::eval::QueryNameOpts;
use crate::topic::model_metadata::{
    load_feature_mean, load_shortlist_weights, masked_head_from_model_type, save_feature_mean,
    save_parameters, save_shortlist_weights, TopicModelMetadata,
};
use crate::topic::train_masked::{write_feature_embedding, write_masked_dictionary};
use candle_util::vae::masked_topic::LatentHead;

/// Every file a masked model must have for *all* of its consumers to work.
///
/// `predict`/`probe` need the safetensors, the metadata, and the three per-gene
/// parquets; `--freeze-feature-embedding` needs the feature embedding. Anything that
/// writes a masked model writes all of these — see [`write_masked_model`], and the test
/// at the bottom of this file which pins the list against a real training run.
///
/// Deliberately **not** listed: `dictionary_empirical`, `pb_gene`, `pb_latent`,
/// `latent`, `cell_proj`. Those come from the pseudobulk collapse, which only training
/// performs — `update` refits a dictionary and has no cell→pseudobulk map. Consumers
/// fall back (`plot-topic` does `dictionary_empirical.or(dictionary)`), which is the
/// same fallback `vae` and `joint-topic` already rely on.
pub const REQUIRED: &[&str] = &[
    "model.json",
    "safetensors",
    "dictionary.parquet",
    "feature_embedding.parquet",
    "shortlist_weights.parquet",
    "feature_mean.parquet",
];

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
    /// The training gene axis, from `feature_mean.parquet` (which returns it alongside
    /// the values, so no separate dictionary read).
    pub gene_names: Vec<Box<str>>,
    pub feature_mean: Vec<f32>,
    pub shortlist: Vec<f32>,
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
        let (gene_names, feature_mean) = load_feature_mean(prefix)?;
        let (_, shortlist) = load_shortlist_weights(prefix)?;

        Ok(Self {
            prefix,
            metadata,
            head,
            gene_names,
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
    ) -> anyhow::Result<MaskedScored> {
        use crate::masked_topic::FeatureNameKindArg;
        let qopts = QueryNameOpts {
            kind: FeatureNameKindArg::Exact.resolve_or_gene(),
            suffix_delim: None,
            keep_suffix: None,
        };
        score_masked_backend(MaskedScoreArgs {
            model: self.prefix,
            data_files: files,
            batch_files: None,
            preload,
            minibatch_size,
            query_name_opts: &qopts,
            metadata: &self.metadata,
            head: self.head,
            need_llik,
        })
    }
}

/// Write every file in [`REQUIRED`], in one call.
///
/// `update` uses this directly. Training cannot — its writes are interleaved with the
/// holdout eval and the CPU move, and reordering them would change behaviour — so it
/// writes the same set inline and the test below pins that the two agree.
pub struct WriteArgs<'a> {
    pub out: &'a str,
    pub decoder: &'a candle_util::decoder::EmbeddedNbTopicDecoder,
    pub parameters: &'a candle_util::candle_nn::VarMap,
    pub metadata: &'a TopicModelMetadata,
    pub gene_names: &'a [Box<str>],
    pub feature_mean: &'a [f32],
    pub shortlist: &'a [f32],
}

pub fn write_masked_model(a: WriteArgs<'_>) -> anyhow::Result<()> {
    // The dictionary is `log_softmax_d(α·ρᵀ)` and `probe`/`predict` score against the
    // PARQUET, not the safetensors — so it must be regenerated from the current α or the
    // written model silently scores as whatever it was before.
    write_masked_dictionary(a.decoder, a.gene_names, a.out)?;
    // ρ is unchanged by an α refit, but the artifact has to be self-contained for
    // `--freeze-feature-embedding` / `--init-feature-embedding` to resolve against it.
    write_feature_embedding(a.decoder.feature_embeddings(), a.gene_names, a.out)?;
    save_shortlist_weights(a.shortlist, a.gene_names, a.out)?;
    save_feature_mean(a.feature_mean, a.gene_names, a.out)?;
    save_parameters(a.parameters, a.out)?;
    a.metadata.save(a.out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn required_set_is_what_the_writer_emits() {
        // Pins the declaration against `write_masked_model`'s body. If a file is added
        // to one and not the other, this fails rather than shipping a model that loads
        // in some subcommands and not others.
        let written = [
            "dictionary.parquet",
            "feature_embedding.parquet",
            "shortlist_weights.parquet",
            "feature_mean.parquet",
            "safetensors",
            "model.json",
        ];
        let mut a: Vec<&str> = REQUIRED.to_vec();
        let mut b: Vec<&str> = written.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(
            a, b,
            "REQUIRED and write_masked_model disagree — update both together"
        );
    }
}
