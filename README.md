# `legume-rs`: Library for Expedited Genomics data analysis with Unified Matching steps in Rust ecosystem 

- previously: ASAP (Analysis of Single-cell data matrix by Approximate Pseudo-bulk projection) 

The goal of ASAP framework:

1. Cell topic annotation should be reasonably fast enough to be integrated with a typical analysis pipeline.

2. The whole algorithm should be accessible to users with modest computing resources (e.g., low memory footprint).

3. Batch adjustment and follow-up analysis should render intuitive (causal) interpretations.


## Installation

```sh
cargo install --path data-beans # utility functions for sparse data management

cargo install --path senna # single-cell embedding with nearest neighbourhood adjustment

cargo install --path pinto # probably interacting cell pairs

cargo install --path cocoa-diff # counterfactual confounder adjustment
```

