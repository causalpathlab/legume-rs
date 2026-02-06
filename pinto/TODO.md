# TODO

## Research / Modeling

- [ ] Explicit gene-gene (feature-feature) interaction model
  - Currently only cell-cell interactions are modeled in shared latent space
  - Want to capture feature-feature interactions explicitly with sparse structure
  - Expand: build both cell-cell and gene-gene interaction graphs
  - Then collapse/aggregate both into pseudobulk summaries for efficient model fitting
  - Extends existing SVD and topic pipelines

## Code Quality

- [ ] Remove `#![allow(dead_code)]` in `srt_common.rs` — clean up unused items instead
- [ ] Audit `#[allow(unused)]` in `srt_collapse_pairs.rs` (struct and impl block) — remove what's truly dead
- [ ] Remove commented-out `// #![allow(dead_code)]` in `srt_random_projection.rs`
- [ ] Replace `.unwrap()` with proper error handling in `srt_common.rs:133` and `srt_input.rs:257`

## Features / Enhancements

- [ ] Add progress bars to SVD pipeline (topic pipeline already has them)
- [ ] Input validation: KNN k vs dataset size, projection dim, coordinate ranges
- [ ] Model checkpoint save/load (currently only parquet parameter output)
- [ ] Streaming/chunked reading for large datasets
- [ ] Expose library API via `lib.rs` (currently empty)

## Testing

- [ ] Unit tests for core statistical ops: Gamma inference, random projection, KNN
- [ ] Unit tests for data I/O: parquet reading, coordinate alignment
- [ ] Unit tests for batch effect estimation
- [ ] Integration tests for end-to-end SVD and topic pipelines

## Documentation

- [ ] README for the pinto crate
- [ ] Usage examples / sample data references

## Cleanup

- [ ] Decide on `src/old/fit_srt_topic.rs` — delete or archive elsewhere
- [ ] Evaluate lock-based parallelism (Mutex+Arc) — consider reduce patterns where possible
