use clap::Args;

use crate::gem::common::ComputeDevice;

/// Per-gene likelihood for the masked imputation loss.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum LikelihoodArg {
    /// Negative binomial — over-dispersed counts, library-scaled, learnable φ.
    Nb,
    /// Multinomial — depth-invariant composition per track. The default: real
    /// cohorts differ in depth between samples, and modelling absolute counts
    /// lets the deepest pseudobulks dominate the fit.
    #[default]
    Multinomial,
}

impl LikelihoodArg {
    #[must_use]
    pub fn to_lib(self) -> candle_util::vae::masked_gem::GemLikelihood {
        use candle_util::vae::masked_gem::GemLikelihood as L;
        match self {
            LikelihoodArg::Nb => L::Nb,
            LikelihoodArg::Multinomial => L::Multinomial,
        }
    }
}

/// Runtime knobs.
#[derive(Args, Debug, Clone)]
pub struct RuntimeArgs {
    #[arg(
        long = "no-preload-data",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Preload all sparse column data into memory before training",
        long_help = "Preload all sparse column data into memory before training.\n\
                     On by default — much faster than repeated disk reads on typical SSDs.\n\
                     Pass `--no-preload-data` to stream instead (only for data that will not fit in RAM)."
    )]
    pub preload_data: bool,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = ComputeDevice::Cuda,
        value_enum,
        help = "Compute device",
        long_help = "Compute device. `cuda` / `metal` require the matching cargo feature\n\
                     (`cargo install faba --features cuda`);\n\
                     a binary built without it falls back to `cpu` with a warning."
    )]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    #[arg(
        long,
        alias = "max-threads",
        default_value_t = 16,
        help = "CPU threads (default 16; 0 = all available)"
    )]
    pub threads: usize,
}

/// CLI arguments for `faba gem-encoder`.
///
/// The latent is a deterministic softmax simplex. Two alternatives were
/// implemented and removed after measurement, both recorded at
/// [`candle_util::vae::masked_gem::velocity_dim`] and here: a Gaussian/KL head
/// (effective rank 1.03 at `kl = 1.0`, and the splice-ratio check degraded
/// monotonically with KL weight), and a stick-breaking simplex (effective rank
/// 1.33 against softmax's 3.14 on the same six-sample fit). The masked objective
/// is the regularizer; neither addition earned its place.
#[derive(Args, Debug, Clone)]
pub struct GemEncoderArgs {
    #[arg(
        value_name = "GENES",
        value_delimiter = ',',
        help = "Gene count matrix prefix(es), space- or comma-separated",
        long_help = "Gene-level count matrices, given positionally —\n\
                     space-separated, so shell globs work: `faba gem-encoder out/*_genes.zarr.zip`.\n\
                     Commas are also accepted.\n\
                     Rows must follow `{gene_key}/count/{spliced|unspliced}`;\n\
                     the unspliced rows are REQUIRED (they are the model's base — see --delta-l2).\n\
                     Multiple files are merged under Union column alignment (cells joined by barcode),\n\
                     with each file's barcodes tagged by its sample id to keep samples distinct."
    )]
    pub genes_pos: Vec<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Optional batch labels (one file, one label per unified cell)"
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n\
                     {out}.latent.parquet                    cell × K LOG THETA (theta = exp of these)\n\
                     {out}.latent_mature.parquet             cell × K log theta fitted to MATURE alone\n\
                     {out}.latent_nascent.parquet            cell × K log theta fitted to NASCENT alone\n\
                     {out}.cell_embedding.parquet            cell × H (theta·alpha) for faba annotate\n\
                     {out}.velocity.parquet                  cell × H velocity, SAME space as cell_embedding\n\
                     {out}.velocity_factor.parquet           cell × K velocity = exp(nascent) - exp(mature)\n\
                     {out}.dictionary.parquet                gene × K MATURE program\n\
                     {out}.dictionary_nascent.parquet        gene × K nascent program\n\
                     {out}.feature_embedding.parquet         2*gene × H CO-EMBEDDED onto the cell manifold,\n\
                     .                                       /count/spliced + /count/unspliced rows;\n\
                     .                                       this is what faba annotate reads\n\
                     {out}.raw_feature_embedding.parquet     gene × H raw spliced program (rho + delta)\n\
                     {out}.delta_feature_embedding.parquet   gene × H splice-ratio offset\n\
                     {out}.splice_ratio_qc.parquet           per-gene model vs observed ratio\n\
                     {out}.dispersion.parquet                gene × 2 NB dispersion\n\
                     {out}.log_likelihood.parquet            per-epoch training trace\n\
                     {out}.safetensors + {out}.model.json    weights + architecture\n\
                     \n\
                     NOTE the per-cell tables (latent, cell_embedding, velocity,\n\
                     velocity_factor, latent_mature, latent_nascent) may contain FEWER ROWS\n\
                     than the input: cell QC drops\n\
                     failing cells from the OUTPUTS (never from training). Join downstream\n\
                     tables by the cell/barcode column, never by row position. --no-qc keeps\n\
                     every cell; --qc-report writes the per-cell keep/drop table."
    )]
    pub out: Box<str>,

    ///////////
    // model //
    ///////////
    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent factors K"
    )]
    pub n_latent: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Per-gene embedding dimension H (must be >= K)"
    )]
    pub embedding_dim: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Inject reparameterized Gaussian noise on the latent during training (no KL)",
        long_help = "Sample the latent as z = mu + sigma*eps during training, with NO KL term —\n\
                     a noisy autoencoder rather than a variational one.\n\
                     \n\
                     A full Gaussian + KL head was implemented and removed:\n\
                     at kl=1.0 the latent collapsed to effective rank 1.03 (a one-dimensional curve),\n\
                     at kl=0.1 it reached 2.14, both far below the softmax head.\n\
                     This flag keeps the half of that head which might still help — the sampling —\n\
                     and drops the half that did the damage — the prior-pulling KL.\n\
                     \n\
                     Inference stays deterministic either way, so the velocity passes are unaffected."
    )]
    pub latent_noise: bool,

    #[arg(
        long,
        short = 'e',
        value_delimiter = ',',
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes after the per-track pool (comma-separated)",
        long_help = "Encoder hidden layer sizes after the per-track pool, comma-separated.\n\
                     The last entry is the trunk output width the latent head reads;\n\
                     everything before it is hidden.\n\
                     \n\
                     Input is 2 * --embedding-dim (the two pooled tracks concatenated),\n\
                     so the default is 256 -> 128 -> 1024 -> 128 -> topics.\n\
                     \n\
                     The wide middle matches `senna topic`, whose default is the same\n\
                     128,1024,128 and which is the sibling this encoder is modelled on.\n\
                     The narrow 128,128 it used before was not chosen on evidence."
    )]
    pub encoder_layers: Vec<usize>,

    #[arg(
        long,
        default_value_t = 512,
        help = "Encoder context window: top-K GENES per cell",
        long_help = "Each cell keeps its top-K genes, ranked on the pooled (spliced + unspliced) score,\n\
                     so a gene's two tracks are always selected together.\n\
                     Smaller K is faster; larger K sees more of each cell."
    )]
    pub context_size: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = LikelihoodArg::Multinomial,
        help = "Masked-loss likelihood: multinomial (depth-invariant, default) or nb (over-dispersed counts)",
        long_help = "multinomial — models each track's COMPOSITION over genes; no library size, no dispersion.\n\
                     nb          — models absolute counts, library-scaled, with a learnable per-gene dispersion.\n\
                     \n\
                     Multinomial is the default because sample depth varies a lot in practice\n\
                     (in one six-sample cohort the inputs spanned 552 to 7195 cells),\n\
                     and under NB the deepest pseudobulks dominate the objective.\n\
                     Across 3 replicates per arm on that cohort, multinomial gave a richer latent\n\
                     on every metric (effective rank 5.9 vs 4.8, factors used 14.3 vs 9.7,\n\
                     median max-factor weight 0.26 vs 0.19).\n\
                     \n\
                     Note the margin is suggestive, not decisive at n=3 — run-to-run spread is large\n\
                     (NB effective rank ranged 3.8-6.2 across identical configs).\n\
                     Prefer nb when depth is uniform and absolute abundance is what you want modelled."
    )]
    pub likelihood: LikelihoodArg,

    /////////////
    // masking //
    /////////////
    #[arg(
        long,
        default_value_t = 0.15,
        help = "Per-gene hide probability for the masked-imputation loss",
        long_help = "Per-gene hide probability. ONE Bernoulli draw per gene, SHARED by both\n\
                     splice tracks, so hiding a gene hides it wholly and both its tracks are\n\
                     scored there.\n\
                     \n\
                     The shared draw is what gives delta a monopoly. Both tracks are predicted\n\
                     from ONE theta, so the only thing that can make them differ is delta.\n\
                     Cross-modal masking (hiding a whole track) was tried and removed: it hands\n\
                     the encoder two different inputs and therefore a competing LATENT delta,\n\
                     which it takes — measured ||dz|| at 1.43x the latent's own spread, with\n\
                     canonical lineage markers falling from rank ~200 to ~33,400 of 34,179.\n\
                     \n\
                     0.15 is the masked-language-modelling convention: BERT's rate, and the\n\
                     rate Geneformer masks transcriptomes at. Inherited from text rather than\n\
                     derived from transcriptomics — a defensible default, not a tuned one."
    )]
    pub mask_fraction: f64,

    ////////////////////
    // regularization //
    ////////////////////
    #[arg(
        long = "delta-l2",
        default_value_t = 0.0,
        help = "Ridge on the splice-ratio offset delta (0 = off, the default; typical 0.1–10)",
        long_help = "L2 ridge on the per-gene splice-ratio offset delta.\n\
                     \n\
                     The model is parameterized nascent-first: the unspliced embedding is rho,\n\
                     and the spliced one is rho + delta, so delta is the steady-state splice-ratio offset.\n\
                     It is log(splicing / DEGRADATION), not a splicing rate:\n\
                     a gene scores high either by splicing fast or by having stable mature mRNA,\n\
                     and this model cannot tell those apart.\n\
                     Under this parameterization <alpha_t, delta_g> is the steady-state\n\
                     log(splicing / degradation) ratio of the RNA-velocity ODE,\n\
                     made factor-resolved and low-rank.\n\
                     \n\
                     The ridge's null, delta = 0, is exactly 'mature composition equals nascent composition',\n\
                     i.e. no differential processing.\n\
                     Too large and the two dictionaries coincide and there is no velocity signal at all.\n\
                     \n\
                     The default is OFF, because delta is already constrained by construction:\n\
                     it is a per-gene embedding in R^H contracted with the topic embedding alpha,\n\
                     so <alpha_t, delta_g> is rank-H and never a free gene-by-topic matrix.\n\
                     The ridge is a second constraint layered on that one,\n\
                     and measurement does not support paying for it\n\
                     (3 wt libraries, 8791 cells, 20 topics, 300 epochs; marker rank is the median\n\
                     of 10 canonical markers by probability within their best topic, of 33609 genes):\n\
                     \n\
                       --delta-l2     splice r     marker rank, beta / delta     |delta|\n\
                       0 (default)       0.387             239 / 124              0.88\n\
                       1                 0.398             276 / 204              0.54\n\
                       1e4               0.206             374 / 495              0.03\n\
                     \n\
                     Crushing delta costs the splice ratio AND the dictionary, both tracks.\n\
                     Between 0 and 1 the gap is small and points both ways:\n\
                     0 recovers markers better, 1 is marginally better on the splice ratio.\n\
                     \n\
                     Note that with the ridge off, |delta| grows slowly and had NOT flattened\n\
                     by epoch 300, so it is not a quantity to read convergence from.\n\
                     Raise this only if you have specific reason to think delta is fitting noise\n\
                     on the sparse unspliced track — that was long assumed to be the dominant\n\
                     failure mode, and on this data it was not observed."
    )]
    pub delta_l2: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 penalty on the gene embedding rho (0 = off)"
    )]
    pub feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Uniform smoothing of the factor proportions during training",
        long_help = "theta_smooth = (1-a) theta + a/K.\n\
                     Keeps every factor on the gradient path and prevents dead factors.\n\
                     Typical 0.01–0.2; 0 disables."
    )]
    pub topic_smoothing: f64,

    //////////////
    // collapse //
    //////////////
    #[arg(
        long,
        default_value_t = 3,
        help = "Number of pseudobulk collapse levels (coarse→fine)"
    )]
    pub num_levels: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Bits of the projection sketch used to hash cells into the finest pseudobulks"
    )]
    pub sort_dim: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "kNN neighbours for cross-batch pseudobulk matching during collapse"
    )]
    pub knn_pb: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Optimization iterations for the pseudobulk collapse/refine"
    )]
    pub num_opt_iter: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Random-projection dimension for the sketch that drives collapse"
    )]
    pub proj_dim: usize,

    #[arg(
        long = "n-hvg",
        default_value_t = 5000,
        help = "Rank the top-N variable GENES to build the pseudobulk sketch (0 = use all)",
        long_help = "Gene-level highly-variable-gene ranking for the RANDOM PROJECTION only.\n\
                     \n\
                     The top-N most variable genes (spliced + unspliced pooled) get weight 1 in\n\
                     the projection that drives the pseudobulk partition; the rest get 0 and are\n\
                     excluded from the projection geometry. The partition is then built where\n\
                     the structure is, on variable genes, instead of being diluted by ~30k\n\
                     near-constant ones.\n\
                     \n\
                     It does NOT restrict the model. Training, the dictionaries and every\n\
                     output still cover every gene. An earlier version masked the rows off the\n\
                     matrix outright, which is the wrong scope for a partitioning heuristic —\n\
                     a marker gene that missed the cut was not down-weighted downstream, it was\n\
                     ABSENT from dictionary.parquet, and `faba annotate` would score that cell\n\
                     type on whatever fraction of its panel survived and still return a\n\
                     confident-looking call.\n\
                     \n\
                     0 uses every gene in the first random projection too."
    )]
    pub n_hvg: usize,

    #[arg(long, help = "Drop batch labels (treat all cells as one batch)")]
    pub ignore_batch: bool,

    #[arg(
        long = "no-batch-adjust",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Disable batch adjustment (ON by default)",
        long_help = "Batch adjustment, ON by default. Pass `--no-batch-adjust` to disable.\n\
                     \n\
                     When on, the model is trained as a triple:\n\
                       encoder input  = mu_observed  (the batch-MIXED counts)\n\
                       encoder null   = mu_residual  (the per-batch offset, PER TRACK)\n\
                       decoder target = mu_adjusted  (the batch-FREE counts)\n\
                     \n\
                     So the encoder is given the batch signal as information rather than\n\
                     having a correction imposed on it, while the decoder is scored against\n\
                     a target with no batch effect in it — which is what leaves the latent\n\
                     no gradient reward for carrying batch.\n\
                     At inference the encoder gets the same pair: the cell's observed counts\n\
                     plus its pseudobulk's residual, so training and evaluation agree.\n\
                     \n\
                     CHECK WHAT YOUR BATCHES ARE.\n\
                     Batch is resolved in three tiers: --batch-files, then an embedded\n\
                     `@`-tag in the cell names, then the file name. With several inputs and no\n\
                     --batch-files, each file's cells are tagged `@<sample>` and that tag\n\
                     becomes the batch — so on rep{1,2,3}_{wt,mut} the batches are the SIX\n\
                     samples, wt/mut among them, and adjustment removes the wt-vs-mut contrast\n\
                     along with donor effects. If that contrast is the biology, pass\n\
                     --batch-files with the labels you mean (e.g. replicate), or\n\
                     --no-batch-adjust.\n\
                     \n\
                     The pseudobulk collapse is batch-aware regardless of this flag."
    )]
    pub batch_adjust: bool,

    #[arg(
        long,
        default_value = "",
        help = "Strip this suffix from each input basename to form its sample id"
    )]
    pub genes_sample_strip: Box<str>,

    //////////////
    // training //
    //////////////
    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
    pub epochs: usize,

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    pub minibatch_size: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 1e-2,
        help = "AdamW learning rate"
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (0 = off, i.e. plain Adam)"
    )]
    pub weight_decay: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off)"
    )]
    pub grad_clip: f32,

    /////////////
    // cell QC //
    /////////////
    /// Cell QC, applied as an OUTPUT FILTER only — see the note on `--out`.
    #[command(flatten)]
    pub qc: data_beans::qc_lib::QcArgs,

    #[command(flatten)]
    pub runtime: RuntimeArgs,
}

impl GemEncoderArgs {
    /// The gene matrices to load.
    pub fn genes(&self) -> anyhow::Result<&[Box<str>]> {
        anyhow::ensure!(
            !self.genes_pos.is_empty(),
            "no gene matrices given — pass them positionally, \
             e.g. `faba gem-encoder out/*_genes.zarr.zip -o out/gme`"
        );
        Ok(&self.genes_pos)
    }

    /// Validate everything cheap before any I/O.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (0.0..1.0).contains(&self.mask_fraction),
            "--mask-fraction must be in [0, 1), got {}",
            self.mask_fraction
        );
        self.genes()?;
        anyhow::ensure!(self.n_latent > 0, "-t/--n-latent must be > 0");
        anyhow::ensure!(
            self.embedding_dim >= self.n_latent,
            "--embedding-dim ({}) must be >= --n-latent ({}): the dictionary \
             beta = softmax(alpha . rho^T) has rank at most H, so at most {} \
             independent factors can be represented",
            self.embedding_dim,
            self.n_latent,
            self.embedding_dim,
        );
        anyhow::ensure!(
            !self.encoder_layers.is_empty(),
            "--encoder-layers must name at least one width"
        );
        anyhow::ensure!(self.context_size > 0, "--context-size must be > 0");
        anyhow::ensure!(self.num_levels > 0, "--num-levels must be > 0");
        Ok(())
    }
}

#[cfg(test)]
#[path = "args_tests.rs"]
mod tests;
