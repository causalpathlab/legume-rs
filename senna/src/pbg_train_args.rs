//! The PyTorch-BigGraph training knobs shared by every command that runs the
//! [`graph_embedding_util::fne`] engine (`senna simba`, `senna fne`): one clap
//! struct, so the flags, their defaults (PBG's own) and their help read the
//! same everywhere. The embedding dimension stays with each command, whose
//! published default differs.

use crate::embed_common::*;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub struct PbgTrainArgs {
    #[arg(
        long,
        short = 'i',
        default_value_t = 10,
        help = "Training epochs (PBG: 10)"
    )]
    pub(crate) epochs: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.1,
        help = "Row-wise Adagrad learning rate (PBG: 0.1)"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Edges per batch (PBG: 1000)",
        long_help = "Edges per batch. Every batch holds ONE relation,\n\
                     drawn with probability proportional to that relation's remaining edges.\n\
                     One optimizer step per batch."
    )]
    pub(crate) batch_size: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Batch negatives, i.e. the chunk size (PBG: 50)",
        long_help = "A batch is cut into chunks of this many positives.\n\
                     Within a chunk every other positive's endpoints are negatives.\n\
                     A positive never competes with itself."
    )]
    pub(crate) num_batch_negs: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Uniform negatives per chunk (PBG: 50)",
        long_help = "Random nodes drawn per chunk and shared by its positives,\n\
                     inside the relation's own node types on each side.\n\
                     Both sides are corrupted."
    )]
    pub(crate) num_uniform_negs: usize,

    #[arg(
        long,
        help = "Weight decay; omit for SIMBA's automatic value",
        long_help = "L2 weight decay on the node table.\n\
                     Omit it for SIMBA's `auto_wd`, which scales a reference value by the edge count.\n\
                     Pass 0 to disable."
    )]
    pub(crate) weight_decay: Option<f64>,

    #[arg(
        long,
        default_value_t = 50,
        help = "Draw the weight decay with probability 1/N per batch (PBG: 50)"
    )]
    pub(crate) wd_interval: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Fraction of each relation's edges held out for the eval loss (PBG: 0.05)",
        long_help = "Edges never trained on, scored with the same loss after every epoch.\n\
                     Held out per relation and drawn once; PBG re-draws a global share each epoch.\n\
                     Pass 0 to train on every edge."
    )]
    pub(crate) eval_fraction: f64,

    #[arg(long, default_value_t = 1, help = "Random seed")]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,
}
