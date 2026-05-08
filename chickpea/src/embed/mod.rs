//! Joint multiome graph-embedding (SIMBA-inspired, sparse, count-NCE).
//!
//! Discriminative joint embedding of genes, peaks, and cells in a single
//! shared `H`-dim space. Trained via count-noise-contrastive estimation
//! (Gutmann & Hyvärinen 2010; mechanically a GloVe-style log-bilinear
//! count factorization with NB-Fisher feature weights for housekeeping
//! downweighting) on sketch-coarsened pseudobulk pseudographs.
//!
//! Two relations: `(gene, cell)` from RNA counts and `(peak, cell)` from
//! ATAC counts. Cell barcode is the primary key for `E_cell` — paired
//! barcodes share an embedding row by construction; unpaired cells get
//! their own rows and align implicitly via shared `E_gene` / `E_peak`.
//!
//! Sparsity is preserved end-to-end: triplet streams from `SparseIoVec`
//! are sampled directly; no dense pseudobulk is materialized.

pub mod coarsen;
pub mod data;
pub mod eval;
pub mod fit;
pub mod loss;
pub mod model;
pub mod training;
