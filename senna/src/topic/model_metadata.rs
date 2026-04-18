use candle_util::candle_nn;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use serde::{Deserialize, Serialize};

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
