mod fit_layout_common;
mod fit_layout_phate;
mod fit_layout_tsne;
mod fit_layout_umap;
pub mod plot;
pub(crate) mod viz_prep;

pub use fit_layout_phate::*;
pub use fit_layout_tsne::*;
pub use fit_layout_umap::*;
pub use plot::scatter::*;
pub use plot::topic::*;
