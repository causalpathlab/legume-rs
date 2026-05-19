use candle_util::candle_core;
use candle_util::candle_nn;
use candle_util::data::GraphCsr;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use serde::{Deserialize, Serialize};

/// Canonical `model_type` strings for `TopicModelMetadata`. Use these in place
/// of inline `"topic"` / `"indexed_topic"` literals so a typo doesn't silently
/// break warm-start or predict dispatch.
pub const MODEL_TYPE_TOPIC: &str = "topic";
// Bumped from "indexed_topic" to "indexed_topic_packed" with the
// packed top-K cutover: old safetensors+model.json files fail load
// here with a clean model-type mismatch instead of silently loading
// against an incompatible runtime.
pub const MODEL_TYPE_INDEXED: &str = "indexed_topic_packed";

/// Metadata needed to reconstruct a trained topic model for inference.
#[derive(Serialize, Deserialize)]
pub struct TopicModelMetadata {
    /// Model variant: `topic`, `indexed_topic_packed`, `joint_topic`
    pub model_type: Box<str>,
    /// Decoder types used (e.g. `["multinom"]`, `["multinom", "nb"]`)
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
    /// Number of directed CSR entries (`= 2 × undirected edges` after
    /// symmetrisation) for the feature-feature graph baked into the
    /// model safetensors. `Some(n)` ⇒ the indexed encoder has a GAT
    /// block and the safetensors carries `graph.row_ptr / col_idx /
    /// values` of length `n_features + 1`, `n`, and `n`. `None` ⇒
    /// legacy sum-pool mode.
    #[serde(default, alias = "n_graph_edges")]
    pub n_graph_edges: Option<usize>,
}

impl TopicModelMetadata {
    /// `true` iff the model carries a feature-feature graph.
    pub fn has_feature_graph(&self) -> bool {
        self.n_graph_edges.is_some_and(|n| n > 0)
    }

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

const GRAPH_ROW_PTR: &str = "graph.row_ptr";
const GRAPH_COL_IDX: &str = "graph.col_idx";
const GRAPH_VALUES: &str = "graph.values";

/// Register Vars for the feature graph in `parameters`, then overwrite
/// them with the actual CSR contents. Called once at training time after
/// the trainable-parameter optimization is finished, so the optimizer
/// never touches these Vars. Indices are stored as f32 (with a precision
/// guard against >2^24 indices).
pub fn save_feature_graph_into_varmap(
    parameters: &mut candle_nn::VarMap,
    dev: &candle_core::Device,
    graph: &GraphCsr,
) -> anyhow::Result<()> {
    use candle_core::{DType, Tensor};
    let max_row_ptr = *graph.row_ptr.last().unwrap_or(&0);
    let max_col = graph.col_idx.iter().copied().max().unwrap_or(0);
    let max_idx = max_row_ptr.max(max_col);
    anyhow::ensure!(
        max_idx < (1u32 << 24),
        "feature-graph index {max_idx} exceeds the f32-mantissa-safe range 2^24; \
         cannot persist as f32 safetensors. Drop edges or shrink the gene axis."
    );

    let row_ptr_t = Tensor::from_vec(
        graph.row_ptr.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        (graph.row_ptr.len(),),
        dev,
    )?;
    let col_idx_t = Tensor::from_vec(
        graph.col_idx.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        (graph.col_idx.len(),),
        dev,
    )?;
    let values_t = Tensor::from_vec(graph.values.clone(), (graph.values.len(),), dev)?;

    let _ = parameters.get(
        (graph.row_ptr.len(),),
        GRAPH_ROW_PTR,
        candle_nn::Init::Const(0.0),
        DType::F32,
        dev,
    )?;
    let _ = parameters.get(
        (graph.col_idx.len(),),
        GRAPH_COL_IDX,
        candle_nn::Init::Const(0.0),
        DType::F32,
        dev,
    )?;
    let _ = parameters.get(
        (graph.values.len(),),
        GRAPH_VALUES,
        candle_nn::Init::Const(0.0),
        DType::F32,
        dev,
    )?;

    parameters.set_one(GRAPH_ROW_PTR, &row_ptr_t)?;
    parameters.set_one(GRAPH_COL_IDX, &col_idx_t)?;
    parameters.set_one(GRAPH_VALUES, &values_t)?;
    log::info!(
        "Persisted feature graph ({} CSR entries) into model safetensors",
        graph.col_idx.len()
    );
    Ok(())
}

/// Pre-allocate feature-graph Vars in `parameters` so `parameters.load(...)`
/// can write into them. Call this *before* loading the safetensors file at
/// predict / impute time, only when `metadata.has_feature_graph` is true.
pub fn allocate_feature_graph_vars(
    parameters: &candle_nn::VarMap,
    dev: &candle_core::Device,
    n_features: usize,
    n_edges: usize,
) -> anyhow::Result<()> {
    use candle_core::DType;
    let _ = parameters.get(
        (n_features + 1,),
        GRAPH_ROW_PTR,
        candle_nn::Init::Const(0.0),
        DType::F32,
        dev,
    )?;
    let _ = parameters.get(
        (n_edges,),
        GRAPH_COL_IDX,
        candle_nn::Init::Const(0.0),
        DType::F32,
        dev,
    )?;
    let _ = parameters.get(
        (n_edges,),
        GRAPH_VALUES,
        candle_nn::Init::Const(0.0),
        DType::F32,
        dev,
    )?;
    Ok(())
}

/// Reconstruct a `GraphCsr` from the feature-graph Vars previously
/// populated by `parameters.load(...)`. Pairs with
/// [`allocate_feature_graph_vars`].
pub fn read_feature_graph_from_varmap(
    parameters: &candle_nn::VarMap,
    n_features: usize,
) -> anyhow::Result<GraphCsr> {
    let data = parameters.data().lock().expect("VarMap lock");
    let get = |name: &str| -> anyhow::Result<Vec<f32>> {
        let var = data
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name} in VarMap"))?;
        Ok(var.as_tensor().to_vec1::<f32>()?)
    };
    let row_ptr_f32 = get(GRAPH_ROW_PTR)?;
    let col_idx_f32 = get(GRAPH_COL_IDX)?;
    let values = get(GRAPH_VALUES)?;
    anyhow::ensure!(
        row_ptr_f32.len() == n_features + 1,
        "graph.row_ptr length {} != n_features + 1 = {}",
        row_ptr_f32.len(),
        n_features + 1,
    );
    let row_ptr: Vec<u32> = row_ptr_f32.into_iter().map(|x| x as u32).collect();
    let col_idx: Vec<u32> = col_idx_f32.into_iter().map(|x| x as u32).collect();
    Ok(GraphCsr {
        n_features,
        row_ptr,
        col_idx,
        values,
    })
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

/// Save per-gene mean expression rate `μ_d` used by the indexed encoder.
///
/// `μ_d` = per-gene mean across pseudobulks at the finest level. The
/// indexed encoder gathers `μ_d` at each cell's top-K positions and
/// composes it with the per-cell batch null as a multiplicative
/// count-rate divisor before Anscombe stabilization — joint
/// correction for `E[y] = batch_effect · gene_mean · biological_deviation`,
/// leaving the cell's biological deviation as the encoder input.
pub fn save_feature_mean(
    feature_mean: &[f32],
    gene_names: &[Box<str>],
    prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::traits::IoOps;
    let path = format!("{prefix}.feature_mean.parquet");
    let mat = nalgebra::DMatrix::<f32>::from_column_slice(feature_mean.len(), 1, feature_mean);
    let cols: Vec<Box<str>> = vec!["mean".into()];
    mat.to_parquet_with_names(&path, (Some(gene_names), Some("gene")), Some(&cols))?;
    log::info!("Saved feature mean to {path}");
    Ok(())
}

/// Load per-gene mean expression rate; returns (gene_names, μ_d).
pub fn load_feature_mean(prefix: &str) -> anyhow::Result<(Vec<Box<str>>, Vec<f32>)> {
    use matrix_util::traits::IoOps;
    let path = format!("{prefix}.feature_mean.parquet");
    let result = nalgebra::DMatrix::<f32>::from_parquet_with_row_names(&path, Some(0))?;
    anyhow::ensure!(
        result.mat.ncols() >= 1,
        "feature_mean parquet missing column at {path}"
    );
    let feature_mean: Vec<f32> = result.mat.column(0).iter().copied().collect();
    log::info!("Loaded {} feature means from {path}", feature_mean.len());
    Ok((result.rows, feature_mean))
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
