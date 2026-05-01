use candle_util::candle_nn;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use serde::{Deserialize, Serialize};

/// Canonical `model_type` strings for `TopicModelMetadata`. Use these in place
/// of inline `"topic"` / `"indexed_topic"` literals so a typo doesn't silently
/// break warm-start or predict dispatch.
pub const MODEL_TYPE_TOPIC: &str = "topic";
pub const MODEL_TYPE_INDEXED: &str = "indexed_topic";

/// Metadata needed to reconstruct a trained topic model for inference.
#[derive(Serialize, Deserialize)]
pub struct TopicModelMetadata {
    /// Model variant: `topic`, `indexed_topic`, `joint_topic`
    pub model_type: Box<str>,
    /// Decoder types used (e.g. `["multinom"]`, `["multinom", "vmf"]`)
    pub decoder_types: Vec<Box<str>>,
    /// Per-decoder loss weights (sum to 1.0)
    pub decoder_weights: Vec<f64>,
    /// Feature dimension at encoder level (`D_coarse` or `D_full`)
    pub n_features_encoder: usize,
    /// Full feature dimension before coarsening
    pub n_features_full: usize,
    /// Number of topics (K)
    pub n_topics: usize,
    /// Encoder hidden layer sizes
    pub encoder_hidden: Vec<usize>,
    /// Number of multi-level coarsening levels
    pub num_levels: usize,
    /// Feature dimension per decoder level
    pub level_decoder_dims: Vec<usize>,
    /// Adjustment method: "batch" or "residual"
    pub adj_method: Box<str>,
    /// Whether feature coarsening was used
    pub has_coarsening: bool,
    /// [D, H] feature embedding width (indexed_topic only)
    #[serde(default)]
    pub embedding_dim: Option<usize>,
    /// Top-K shortlist size at encoder (indexed_topic only)
    #[serde(default)]
    pub enc_context_size: Option<usize>,
    /// Top-K shortlist size at decoder (indexed_topic only)
    #[serde(default)]
    pub dec_context_size: Option<usize>,
    /// Mean training topic proportions θ̄ ∈ ℝ^K. Used at predict time as the
    /// mixture weights for the training-implied gene marginal in
    /// per-batch δ estimation. Falls back to uniform 1/K when absent.
    #[serde(default)]
    pub theta_mean: Option<Vec<f32>>,
}

impl TopicModelMetadata {
    pub fn save(&self, prefix: &str) -> anyhow::Result<()> {
        let path = format!("{prefix}.model.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        log::info!("Saved model metadata to {path}");
        Ok(())
    }

    pub fn load(prefix: &str) -> anyhow::Result<Self> {
        let path = format!("{prefix}.model.json");
        let json = std::fs::read_to_string(&path)?;
        let metadata: Self = serde_json::from_str(&json)?;
        Ok(metadata)
    }

    /// Compute `θ̄_train` from the training-time `[N, K]` log θ matrix and
    /// re-save the metadata. Idempotent — call after the post-training eval
    /// pass returns `z_nk`.
    pub fn populate_theta_mean_and_save(
        &mut self,
        log_z_nk: &nalgebra::DMatrix<f32>,
        prefix: &str,
    ) -> anyhow::Result<()> {
        let n = log_z_nk.nrows() as f32;
        if n <= 0.0 {
            return Ok(());
        }
        let k = log_z_nk.ncols();
        let theta_mean: Vec<f32> = (0..k)
            .map(|kk| log_z_nk.column(kk).iter().map(|&x| x.exp()).sum::<f32>() / n)
            .collect();
        self.theta_mean = Some(theta_mean);
        self.save(prefix)
    }
}

/// Save feature coarsening to JSON alongside the model.
pub fn save_coarsening(coarsening: &FeatureCoarsening, prefix: &str) -> anyhow::Result<()> {
    let path = format!("{prefix}.coarsening.json");
    let json = serde_json::to_string(coarsening)?;
    std::fs::write(&path, json)?;
    log::info!("Saved feature coarsening to {path}");
    Ok(())
}

/// Load feature coarsening from JSON.
pub fn load_coarsening(prefix: &str) -> anyhow::Result<Option<FeatureCoarsening>> {
    let path = format!("{prefix}.coarsening.json");
    match std::fs::read_to_string(&path) {
        Ok(json) => {
            let fc: FeatureCoarsening = serde_json::from_str(&json)?;
            Ok(Some(fc))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Load dictionary parquet, returning gene names and the beta matrix [D × K].
pub fn load_dictionary(prefix: &str) -> anyhow::Result<(Vec<Box<str>>, nalgebra::DMatrix<f32>)> {
    use matrix_util::traits::IoOps;
    let path = format!("{prefix}.dictionary.parquet");
    let result = nalgebra::DMatrix::<f32>::from_parquet_with_row_names(&path, Some(0))?;
    log::info!(
        "Loaded dictionary: {} genes × {} topics from {}",
        result.rows.len(),
        result.mat.ncols(),
        path
    );
    Ok((result.rows, result.mat))
}

/// Save `VarMap` parameters as safetensors.
pub fn save_parameters(parameters: &candle_nn::VarMap, prefix: &str) -> anyhow::Result<()> {
    let path = format!("{prefix}.safetensors");
    parameters.save(&path)?;
    log::info!("Saved model parameters to {path}");
    Ok(())
}

/// Save NB-Fisher shortlist weights for indexed-topic prediction.
pub fn save_shortlist_weights(
    weights: &[f32],
    gene_names: &[Box<str>],
    prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::traits::IoOps;
    let path = format!("{prefix}.shortlist_weights.parquet");
    let mat = nalgebra::DMatrix::<f32>::from_column_slice(weights.len(), 1, weights);
    let cols: Vec<Box<str>> = vec!["weight".into()];
    mat.to_parquet_with_names(&path, (Some(gene_names), Some("gene")), Some(&cols))?;
    log::info!("Saved shortlist weights to {path}");
    Ok(())
}

/// Load NB-Fisher shortlist weights from disk; returns (gene_names, weights).
pub fn load_shortlist_weights(prefix: &str) -> anyhow::Result<(Vec<Box<str>>, Vec<f32>)> {
    use matrix_util::traits::IoOps;
    let path = format!("{prefix}.shortlist_weights.parquet");
    let result = nalgebra::DMatrix::<f32>::from_parquet_with_row_names(&path, Some(0))?;
    anyhow::ensure!(
        result.mat.ncols() >= 1,
        "shortlist_weights parquet missing column at {path}"
    );
    let weights: Vec<f32> = result.mat.column(0).iter().copied().collect();
    log::info!("Loaded {} shortlist weights from {path}", weights.len());
    Ok((result.rows, weights))
}

/// Load per-gene NB dispersion φ from `{prefix}.dispersion.parquet`.
/// Returns `None` if the file doesn't exist (e.g. multinomial-only training run).
pub fn load_dispersion(prefix: &str) -> anyhow::Result<Option<Vec<f32>>> {
    use matrix_util::traits::IoOps;
    let path = format!("{prefix}.dispersion.parquet");
    if !std::path::Path::new(&path).exists() {
        return Ok(None);
    }
    let result = nalgebra::DMatrix::<f32>::from_parquet_with_row_names(&path, Some(0))?;
    anyhow::ensure!(
        result.mat.ncols() >= 1,
        "dispersion parquet missing column at {path}"
    );
    let phi: Vec<f32> = result.mat.column(0).iter().copied().collect();
    log::info!("Loaded {} dispersion values from {path}", phi.len());
    Ok(Some(phi))
}
