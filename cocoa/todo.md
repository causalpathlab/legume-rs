
* [x] parallelize permutation — replaced sequential loop with rayon par_iter; pre-generate shuffled exposures for reproducibility, parallel replay + estimation, sequential reduction

* [x] reuse matrix-util as much as possible — column ops (column_mut += column) replace manual gene loops in collapse_cocoa_data.rs

* [x] check inefficient routines in stat.rs — replaced row_iter_mut + component_mul_assign with column-wise scale_mut; eliminated Vec allocation in hot weight computation loop (two-pass: sum then accumulate)

* [ ] any other ideas to speed up?
  - selective calibration in matrix-param (only compute posterior_mean, skip sd/log variants when unused)
  - parallelize estimate_parameters across topics (par_iter over k)
  - fuse reset_stat + add_stat in update_stat to avoid double matrix traversal

