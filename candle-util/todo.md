small test

* [x] ~~for susie, on the variational side, we can simplify with real single effect and variance~~
  - Attempted scalar (L,k) mean and shared (L,k) variance — both degrade SGVB recovery.
    Per-variable (L,p,k) params needed for gradient-based symmetry breaking in SGVB.
    CAVI can do scalar effects via closed-form Bayes factors, but SGVB cannot.

big implementation

* [x] implement spike-and-slab for sgvb — `SpikeSlabVar` with independent Bernoulli gates, flat only
* [ ] multi-level spike-and-slab — not feasible: soft-collapsing correlated variables cancels signal

refactor

* [x] consistent api for different variational — `ComponentVariational` for all SuSiE variants
