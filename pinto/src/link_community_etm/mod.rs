//! Link community via embedded topic model (`pinto lc-etm`).
//!
//! Treats each cell-cell edge as a "document" with token counts
//! `y_e = x_i + x_j`, then fits the senna-style indexed topic ETM using
//! [`candle_util::vae::masked_topic`]. The K topics are the link
//! communities; β = softmax(α · ρᵀ) gives the per-community gene rates;
//! the encoder produces per-edge soft posteriors π_e ∈ Δ^K.

pub mod args;
pub mod data;
pub mod fit;
pub mod post;

pub use args::SrtLinkCommunityEtmArgs;
pub use fit::fit_srt_link_community_etm;
