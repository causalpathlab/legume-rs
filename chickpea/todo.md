## Multi-Resolution Cascade SuSiE for Peak-Gene Linking

X (ATAC peaks, 100k-1M) -> Y (RNA genes)

### Architecture

- Sequential coarse-to-fine cascade with per-level SuSiE
- Feature coarsening (D peaks → d modules) + VariantTree hierarchy over modules
- Each tree level: aggregate X at that resolution, run SuSiE, prune low-PIP branches
- Final level: expand to individual peaks, run SuSiE with parent PIP as prior
- Per-gene tasks, parallelized with rayon

### Done

1. `CascadeTask`, `CascadeResult`, `CascadeParams` structs
2. `aggregate_modules_by_tree_level()` — sum module pseudobulk by tree grouping
3. `build_child_prior()` — cascade PIP from parent to children (hard mask + soft prior)
4. `run_cascade()` — the level-by-level loop
5. `run_susie_level_gaussian()` — CAVI SuSiE wrapper per level
6. `run_susie_level_sgvb()` — SGVB SuSiE wrapper per level
7. `simulate_link_data()` — simulation for testing
8. Prior weights support in `cavi_susie` (candle-util)
9. All 8 tests passing

### Future

- Multi-gene blocks: group genes by cis-window overlap + expression correlation
- Shared vs independent α across genes in a block
- Better coarsening that accounts for signal (not purely data-driven)
