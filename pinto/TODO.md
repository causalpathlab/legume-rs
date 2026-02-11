# TODO

## Research / Modeling

- [ ] Explicit gene-gene (feature-feature) interaction model
  - Currently only cell-cell interactions are modeled in shared latent space
  - Want to capture feature-feature interactions explicitly with sparse structure
  - Expand: build both cell-cell and gene-gene interaction graphs
  - Then collapse/aggregate both into pseudobulk summaries for efficient model fitting
  - Extends existing SVD and topic pipelines

## Features / Enhancements

- [ ] Model checkpoint save/load (currently only parquet parameter output)
- [ ] Streaming/chunked reading for large datasets

## Testing

- [ ] Unit tests for batch effect estimation (`srt_estimate_batch_effects.rs`)
  - Test with single batch (no-op) and multiple batches
  - Requires mock `SparseIoVec` data
- [ ] Gamma inference and KNN tests belong in `matrix-param` and `matrix-util` respectively
- [ ] Integration tests for end-to-end SVD and topic pipelines (require sample data)

## Done

- [x] Remove `#![allow(dead_code)]` in `srt_common.rs` — cleaned up unused items
- [x] Remove commented-out `// #![allow(dead_code)]` in `srt_random_projection.rs`
- [x] Remove `#[allow(dead_code)]` from `srt_gene_pairs.rs` and `srt_gene_graph.rs`
- [x] Replace `.unwrap()` with proper error handling in `srt_common.rs` and `srt_input.rs`
- [x] Delete `src/old/fit_srt_topic.rs`
- [x] Delete empty `src/lib.rs`
- [x] Remove unused dependencies (`candle-util`, `flate2`, `dashmap`, `num_cpus`)
- [x] Fix all clippy warnings
- [x] Add progress bars / stage logging to SVD pipelines
- [x] Input validation: KNN k vs dataset size, projection dim, coordinate ranges
- [x] README for the pinto crate
- [x] Usage examples / sample data references
- [x] Unit tests for `fit_srt_gene_network.rs` helpers (argsort, spectral embed, UMAP weights)
- [x] Unit tests for `srt_gene_pairs.rs` (elbow detection, gene-pair deltas)
- [x] Unit tests for `srt_input.rs` (batch coordinate appending)
- [x] Mutex+Arc parallelism — confirmed necessary and idiomatic for rayon
