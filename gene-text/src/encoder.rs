//! The text encoder: any BERT-family model from the Hugging Face Hub (or a
//! local directory with `config.json`, `tokenizer.json` and
//! `model.safetensors`), run through candle. A document comes back as one
//! pooled vector and the last-layer vector of every real token with its
//! byte span, so the caller can align tokens to its own words.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use log::info;
use std::path::PathBuf;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// Attention-masked mean over the tokens (the sentence-transformers
    /// recipe).
    Mean,
    /// The `[CLS]` vector.
    Cls,
}

/// Where the model comes from.
pub struct ModelSpec {
    pub model_id: String,
    pub revision: String,
    /// A directory holding the three files; skips the Hub.
    pub local_dir: Option<PathBuf>,
}

struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
}

fn resolve_files(spec: &ModelSpec) -> Result<ModelFiles> {
    if let Some(dir) = &spec.local_dir {
        let f = |name: &str| -> Result<PathBuf> {
            let p = dir.join(name);
            anyhow::ensure!(p.exists(), "{}: missing {name}", dir.display());
            Ok(p)
        };
        return Ok(ModelFiles {
            config: f("config.json")?,
            tokenizer: f("tokenizer.json")?,
            weights: f("model.safetensors")?,
        });
    }
    use hf_hub::{api::sync::Api, Repo, RepoType};
    let api = Api::new().context("Hugging Face Hub client")?;
    let repo = api.repo(Repo::with_revision(
        spec.model_id.clone(),
        RepoType::Model,
        spec.revision.clone(),
    ));
    info!(
        "fetching {}@{} from the Hub (cached under ~/.cache/huggingface)",
        spec.model_id, spec.revision
    );
    Ok(ModelFiles {
        config: repo.get("config.json")?,
        tokenizer: repo.get("tokenizer.json")?,
        weights: repo.get("model.safetensors")?,
    })
}

/// One real token of an encoded document.
pub struct TokenVec {
    pub start: usize,
    pub end: usize,
    pub vec: Vec<f32>,
}

/// One encoded document.
pub struct DocEncoding {
    /// L2-normalised pooled vector.
    pub pooled: Vec<f32>,
    pub tokens: Vec<TokenVec>,
}

pub struct Encoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pooling: Pooling,
    hidden: usize,
}

impl Encoder {
    pub fn load(
        spec: &ModelSpec,
        max_tokens: usize,
        pooling: Pooling,
        device: Device,
    ) -> Result<Self> {
        let files = resolve_files(spec)?;
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&files.config)
                .with_context(|| format!("reading {}", files.config.display()))?,
        )
        .context("parsing the model config")?;
        let hidden = config.hidden_size;
        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_tokens,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        // SAFETY: the safetensors file is memory-mapped read-only for the
        // lifetime of the loaded weights; it is not modified while mapped.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&files.weights], DType::F32, &device)? };
        let model = BertModel::load(vb, &config).context("loading the BERT weights")?;
        info!(
            "encoder: {} (hidden {hidden}, max {max_tokens} tokens, {:?} pooling) on {:?}",
            spec.local_dir
                .as_deref()
                .map_or(spec.model_id.clone(), |d| d.display().to_string()),
            pooling,
            device
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            pooling,
            hidden,
        })
    }

    #[must_use]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Encode a batch of texts.
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<DocEncoding>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let b = encodings.len();
        let l = encodings.first().map_or(0, |e| e.get_ids().len());
        let mut ids: Vec<u32> = Vec::with_capacity(b * l);
        let mut mask: Vec<u32> = Vec::with_capacity(b * l);
        for e in &encodings {
            ids.extend_from_slice(e.get_ids());
            mask.extend_from_slice(e.get_attention_mask());
        }
        let ids = Tensor::from_vec(ids, (b, l), &self.device)?;
        let mask = Tensor::from_vec(mask, (b, l), &self.device)?;
        let type_ids = ids.zeros_like()?;
        let hidden = self.model.forward(&ids, &type_ids, Some(&mask))?; // [b, l, h]
        let maskf = mask.to_dtype(DType::F32)?.unsqueeze(2)?; // [b, l, 1]
        let pooled = match self.pooling {
            Pooling::Mean => {
                let s = hidden.broadcast_mul(&maskf)?.sum(1)?; // [b, h]
                let n = maskf.sum(1)?.clamp(1.0, f64::MAX)?; // [b, 1]
                s.broadcast_div(&n)?
            }
            Pooling::Cls => hidden.narrow(1, 0, 1)?.squeeze(1)?,
        };
        let norm = pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        let pooled = pooled.broadcast_div(&norm)?.to_vec2::<f32>()?;
        let hidden = hidden.to_vec3::<f32>()?;
        let mut out = Vec::with_capacity(b);
        for (i, e) in encodings.iter().enumerate() {
            let offsets = e.get_offsets();
            let m = e.get_attention_mask();
            let special = e.get_special_tokens_mask();
            let mut tokens = Vec::with_capacity(l);
            for t in 0..l {
                if m[t] == 0 || special.get(t).copied().unwrap_or(0) == 1 {
                    continue;
                }
                let (s, en) = offsets[t];
                if en <= s {
                    continue;
                }
                tokens.push(TokenVec {
                    start: s,
                    end: en,
                    vec: hidden[i][t].clone(),
                });
            }
            out.push(DocEncoding {
                pooled: pooled[i].clone(),
                tokens,
            });
        }
        Ok(out)
    }
}
