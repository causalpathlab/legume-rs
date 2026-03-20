#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/////////////////////////////////
// Delta Joint Topic Decoder   //
// Shared base + chain deltas  //
/////////////////////////////////

pub struct DeltaTopicDecoder {
    n_modalities: usize,
    n_features: usize,
    n_features_vec: Vec<usize>, // vec![n_features; n_modalities] for dim_obs()
    n_topics: usize,
    base: SoftmaxLinear,
    deltas: Vec<Tensor>,
}

impl DeltaTopicDecoder {
    /// Create a delta topic decoder with shared base + cumulative chain deltas.
    ///
    /// * `base.logits` [K, D] — shared base dictionary logits
    /// * `base.logit_bias` [1, D] — shared bias
    /// * `delta.{m}` [K, D] — delta logits for modality m (m = 1..M-1), init to zero
    pub fn new(
        n_modalities: usize,
        n_features: usize,
        n_topics: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let base = log_softmax_linear(n_topics, n_features, vs.pp("base"))?;

        let deltas = (1..n_modalities)
            .map(|m| {
                vs.get_with_hints(
                    (n_topics, n_features),
                    &format!("delta.{}", m),
                    candle_nn::init::ZERO,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            n_modalities,
            n_features,
            n_features_vec: vec![n_features; n_modalities],
            n_topics,
            base,
            deltas,
        })
    }

    /// Compute all effective log-dictionaries in a single O(M) pass.
    /// Returns M tensors [K, D] in log-softmax space.
    fn all_effective_log_betas(&self) -> Result<Vec<Tensor>> {
        let mut logits = self.base.raw_biased_logits_kd()?;
        let rank = logits.rank();
        let mut result = Vec::with_capacity(self.n_modalities);
        result.push(ops::log_softmax(&logits, rank - 1)?);
        for delta in &self.deltas {
            logits = (&logits + delta)?;
            result.push(ops::log_softmax(&logits, rank - 1)?);
        }
        Ok(result)
    }

    /// Get the base dictionary (log-softmax of base logits), transposed to [D, K]
    pub fn get_base_dictionary(&self) -> Result<Tensor> {
        self.base.weight_dk()
    }

    /// Get the raw delta logit tensors (one per non-reference modality)
    pub fn get_deltas(&self) -> &[Tensor] {
        &self.deltas
    }
}

impl JointDecoderModuleT for DeltaTopicDecoder {
    fn get_dictionary(&self) -> Result<Vec<Tensor>> {
        self.all_effective_log_betas()?
            .into_iter()
            .map(|t| t.transpose(0, 1))
            .collect()
    }

    fn forward(&self, z_nk: &Tensor) -> Result<Vec<Tensor>> {
        self.all_effective_log_betas()?
            .iter()
            .map(|log_beta| logsumexp_forward(z_nk, log_beta)?.exp())
            .collect()
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd_vec: &[Tensor],
        _llik: &LlikFn,
    ) -> Result<(Vec<Tensor>, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let log_recon_vec: Vec<Tensor> = self
            .all_effective_log_betas()?
            .iter()
            .map(|log_beta| logsumexp_forward(z_nk, log_beta))
            .collect::<Result<Vec<_>>>()?;

        joint_multinomial_llik(log_recon_vec, x_nd_vec)
    }

    fn dim_obs(&self) -> &[usize] {
        &self.n_features_vec
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
