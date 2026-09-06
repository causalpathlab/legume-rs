# Fine-grained immune states inside the 10-bit pseudobulk tree

## Context

Goal: a model trained on the whole Granja BMMC donor whose cell latent separates CD4 naive from CD4 memory, with the default 10-bit sketch (about 1000 finest pseudobulks, 3 levels) and no T-only retraining. The user fixed the sketch at 10 bits and asked for a careful assessment of each idea before building.

Measured on 2026-09-06 (memory `senna_granja_granularity_findings.md`):

- CD45RA/CD45RO are splice isoforms of PTPRC, invisible in 3' counts. The proxy program is ~160 genes at 2-4 fold, 2.5% of UMI mass, none binary.
- The pseudobulk tree is blind to the split at every level: purity 0.68/0.71/0.74 (128/511/1006 pbs) vs label-shuffle 0.63/0.67/0.70. Finest co-members share the label 59%; five bge neighbours share it 83%.
- Mechanism: finest leaves are sign bits of the top 10 rSVD components of a random count sketch; those are lineage mass, a 2.5% program never flips a bit. Pseudobulk profiles carry the contrast only as a slope (r = 0.83 with memory fraction), never as vertices.
- The topic trainer trains only on pb rows. bge injects 16 cells per pb and projects every cell. `senna vae` on the same blind pbs reaches bge's probe (0.89) but not its dominance (2-cluster ARI 0.15 vs 0.50). `rest` is refuted (linear in frozen theta, probe capped at 0.78). `masked-vae` posterior-collapses at beta = 1.
- More marginal bits (13-bit sketch): 2-cluster ARI 0.32, probe 0.81, broad ARI 0.45 -> 0.27. 12-bit: probe 0.84, ARI 0.02. Not the lever.
- Noise: topic family 0.08 ARI; bge 0.02 ARI / 0.01 probe.

Reference bar (bge, 3 seeds): 2-cluster ARI 0.50, probe 0.88. Topic today: ARI 0.00-0.17, probe 0.78-0.79.

Fixed scorer: `score_granja.py` (session scratchpad `bmm/`). Readouts: CD4 naive/memory 2-cluster k-means ARI (dominance), 5-fold linear probe (information), broad/fine ARI (lineage must not drop), >= 3 reps for any topic-family claim, never argmax theta.

## Code facts (read-only exploration, file:line)

### pb tree (data-beans-alg `collapse_data`)
- Sketch: log1p counts, column-normalised, seeded Gaussian basis, batch-centred, z-scored, clamped (`random_projection.rs:180-184, 377-407`). Hash `binary_sort_columns` (`random_projection.rs:535-563`): rSVD of the sketch, bit k = sign of component k, bit 0 = most dominant.
- Levels for sort_dim 10 / 3 levels = `[10, 9, 7]` bits (`refine.rs:718-733`; `DEFAULT_COARSEST_SORT_DIM = 7`, `stats.rs:772`). Coarse identity = LOW bits (`initial_per_level_from_hash`, `refine.rs:68-88`). The bits subdividing a coarse node are the HIGH components 7..9, read per pb-sample by `build_reproject_offsets` (`refine.rs:97-121`) and consumed by `refine_multilevel::refine_assignments` (`refine_multilevel.rs:262-275`), which accepts any bounded per-pb offset.
- Single batch: `refine_or_identity` (`refine.rs:123-146`) short-circuits, so on this donor the leaves are exactly the hash labels. DC-Poisson refine only runs with >= 2 batches, and its fine-level candidate sets are already within-node.
- Hook point: `collapse_columns_multilevel_with_hierarchy` (`mod.rs:556-561`) and `collapse_columns_multilevel_vec` (`mod.rs:900-906`), the line `let fine_codes = binary_sort_columns(proj_kn, kk)?;` before `assign_groups`.
- In scope at the hook: batch per column (`data_vec.get_batch_membership`), counts per node (`read_columns_csc`, preloaded), column multiplicity (carried reference/bulk columns). Diagnostic block for a tree log line: `refine.rs:346-379`.
- Reusable: `AdjustByDivisionOp` / `adjust_by_poisson_ratio` (`matrix-util/src/dmatrix_util.rs:226,245-270`), `rsvd` (`dmatrix_rsvd.rs:186-214`, seeded), `dc_poisson::Profiles`, `DcPoissonStats::from_profiles` (`dc_poisson.rs:333`), `nb_dispersion::DispersionTrend::{fit, fisher_weight}` (`nb_dispersion.rs:58-147`), `pb_reference::cell_counts_from` (`senna/src/pb_reference.rs:108-130`). No Pearson residual function exists.
- `--from` partition inheritance (`mod.rs:619-825`) builds pb-samples from fresh marginal codes and modal-votes the inherited labels (`:694-701`); nesting is only `debug_assert`ed. Viz re-hashes the cached projection for layout only (`fit_layout_common.rs:926-935, 1212`), never reads the training tree.
- `MultilevelParams` struct-literal sites that must gain the new field: `senna/src/topic/common.rs:596`, `senna/src/joint_topic.rs:254`, `senna/src/svd/fit.rs:161`, `senna/src/svd/fit_joint.rs:138`, `senna/src/gem_encoder/load.rs:184`, `graph-embedding-util/src/fit/setup.rs:63`, `cocoa/src/randomly_partition_data.rs:156`, tests in `data-beans-alg/tests/{anchored_batches,panel_observability,bulk_batches,pooled_frame}.rs`.
- Stale help text: `--num-levels` says "4 to sort-dim"; the floor is 7 (`senna/src/refine_weighting.rs:199-202`).

### Topic-family trainers (candle-util)
- Dense trainer `candle-util/src/vae/topic.rs:141-189`: per epoch, per level, per minibatch: encoder -> decoder -> `loss = (kl - llik).mean()` -> AdamW step per level per batch. No cross-level tensor, no row ids in `MinibatchData` (`data/loader.rs:7-12`). Shared by `senna topic` and `senna vae` (`vae.rs:319-347`).
- Rows are Gamma posterior RATE samples `[n_pb, D]` drawn once per run (`topic/common.rs:224-241`); pb size not carried; NB decoders use the row-sum as depth. Gamma sufficient statistics ARE retained because senna calibrates with `CalibrateTarget::All` (`stats.rs:468`, `topic/common.rs:602`); only a plane accessor on `GammaMatrix` is missing.
- Parent map fine pb -> coarse pb derivable from `cell_to_pb_per_level` (`topic/common.rs:267`, finest-last; `PipelineCtx.cell_to_pb_finest`, `topic/cmd.rs:741`).
- Masked trainer `candle-util/src/vae/masked_topic.rs:489-631`: minibatches carry `mb.base.row_ids`; index vector is arbitrary and draws are keyed by source row (`masked_epoch.rs:276,293`), so grouping siblings is free. Decoder scores unseen genes through `ModuleTarget` (`decoder/masked_etm.rs:73-86`) from one call site (`masked_topic.rs:543-550`).
- Anchor prior (`--anchor-penalty`, default 1.0): CE toward the pseudobulk empirical dictionary, a lineage-mass prior.
- Log bug: "Level i/N: X samples" prints the feature count (`topic.rs:88-96, 225-236`).

### bge fit (graph-embedding-util)
- Phase 1 = one composite over `[cell axis (<= 16 cells per pb at every level)] + pb axes`; phase 2 = per-cell unshrunk Poisson SGD against the frozen feature side for every cell (`fit/projection/cells.rs:86`).
- Negatives corrupt the FEATURE only: cell axis uniform over expressed genes (`loss/feat.rs:101-122`); score = `e_feat.e_cell + b_feat + b_cell` (`model/score.rs:40,48`); `softmax_nce`/`logistic_nce` accept several negative blocks (`loss/mod.rs:132,167`).
- Tree ids reach `axes::build_axis_data` (`fit/mod.rs:150`) only for subsampling; `CompositeTrainContext.cell_to_pb_per_level` exists, unread (`training.rs:220-226`).
- Precedents: by-parent sibling pools + rejection draw (`loss/cell.rs:42-51,178-213`, `loss/chain.rs:112-184`), opt-in per-epoch pools with fallback counting (`loss/modules.rs:261-290,377-393`), two-sided corruption in SIMBA (`simba/train.rs:133-180`), `batched_matvec` (`candle-util/src/batched_dot.rs:40`).

## Assessment of each idea

Verdicts: DO (fixes a measured cause, cheap, testable), LATER (sound, depends on an earlier piece), DROP.

### 1. Residual high bits with per-split left/right contrasts — DO, first
- **Mechanism vs evidence.** The three subdividing bits are components 7..9 of a mass sketch. Replacing them per coarse node with recursive within-node residual bisections attacks exactly the measured cause, keeps the 128 lineage nodes and the ~1000 leaves, and leaves every consumer's shapes unchanged. The 13-bit run showed the simplex spends a topic on the fine axis once rows are purer; this gets that without the lineage cost because the 7 low bits are untouched.
- **Left vs right (user's requirement).** Each internal split owns a contrast: the residual loading vector, the post-split weighted log fold change per gene, per-batch side counts, a strength verdict against the Marchenko-Pastur edge, and a two-group Poisson log-likelihood ratio. On this donor the CD4 T nodes should show a split whose contrast is S100A4/ANXA1/IL32/KLRB1 vs CCR7/SELL/LEF1.
- **Label-free gate before training.** Per node: variance explained by the residual leaves over the random expectation (L-1)/(n-1), computed for BOTH the residual bits and the old marginal bits against the same residual, so every run logs its own A/B. Labelled dev check on this donor: pure-pseudobulk excess over a label shuffle (today 18 vs 12 memory-majority at the finest level) and the majority-vote ARI ceiling (today 0.23).
- **Cost.** ~900 exact rank-2 eigenproblems on <= 50-cell blocks, sub-second with rayon; landmark path above 2048 cells per node.
- **Risks.** Detectability at n ~ 47 (rank-1 spike must exceed sigma sqrt(p/n)): gene floor, top-2000 genes by residual variance, robust sigma, clipping; deeper splits will often sit below the edge on small data and are flagged. Multi-batch: per-(node, batch) profiles, pooled SVD, per-batch side counts. `--from`: fix the pb-sample construction so inherited leaves are reproduced exactly. Viz path unaffected.

### 2. SVD + Nystrom for the tree — folded into 1
- The learned residual basis per node is the SVD; Nystrom is the large-N landmark extension. A second round from a trained model's factor-by-cell matrix is an iteration (tree -> model -> factors -> tree), not needed for the first result, and must never touch the coarse bits (reverted proj_kn inheritance).

### 3. Hierarchical split likelihood — LATER (masked first, dense later)
- **Identity.** With child mass m_cg = n_c l_c p_cg and parent mass defined as the sibling sum, the leaf likelihood equals the root term plus per-parent split terms S_P = sum_c y_cg log(m_cg / M_Pg), invariant to any per-gene factor shared by siblings. If levels were nested, today's independent sum is 3 L_root + 2 S_coarse + 1 S_finest: the finest split is one sixth of the objective. Unit weights reduce to "finest level only" (`--num-levels 1`, a no-code A/B). Any w_finest > 1 needs an explicit split term.
- **Formulation.** Sibling-sum on the child level's own rows: Y_c = n_c x rate, parent = sum over siblings in the minibatch, mass n_c l_c p_c with p_c the decoder simplex (every dense decoder already returns it).
- **Shared core.** `candle-util/src/loss.rs`: `sibling_split_llik(y_nd, log_mass_nd, log_w_n1, segment_n, n_segments) -> SplitScores { model, null, saturated }` via `fast_index::{index_add_rows, gather_rows}`; `loader_util::grouped_minibatch_indices`; a `LevelTree { parent_of_row, n_parents, row_size }`; senna `pb_trees_from_membership` with a HARD nesting check (the `--from` modal vote can break nesting; fix with `project_to_refinement` after the vote).
- **Plan A, masked (~150 lines + tests).** `IndexedTrainConfig.tree`, `MaskedTrainOpts { split_weight, split_only }`, `MaskedMinibatch.segment`, grouped index in `begin_epoch` (`masked_epoch.rs:242-276`), split on UNSEEN counts with mass n_c l_c (theta beta)_m (1 - visible share) after `masked_topic.rs:543-550`; CLI `--tree-split-weight` (default 0), `--tree-split-only` (multinomial head only).
- **Plan B, dense (~400-500 lines).** `MinibatchData.row_ids/segment`, `InMemoryData::shuffle_grouped_on_device`, `TrainConfig { tree, split_weight, split_only }`, split at `topic.rs:162`; threaded `PreparedData.cell_to_pb_per_level -> PipelineCtx.pb_trees -> senna TrainConfig -> candle TrainConfig`; `senna vae` shares it. +20-30% decoder time on CPU, negligible on CUDA.
- **Readout.** Per depth per epoch: model, null (children carry the parent profile, differ by mass), saturated (y / sibling sum); `captured = (model - null) / (saturated - null)` at the finest depth says whether the tree exposes within-node structure and whether the model learned it.
- **Why later.** Split terms carry nothing until the tree separates fine states within nodes. Test `--anchor-penalty 0` alongside, since that prior opposes small within-node programs.
- **Tests.** Flat tree gives model = 0; one segment reproduces the chain rule to 1e-4; per-gene shift invariance; planted uneven split orders true > flat > swapped, saturated >= model >= null.

### 4. Sibling negatives in bge — LATER, premise first
- **Premise check (no new code).** bge's dominance may come from phase 2 (unshrunk per-cell fit) rather than negatives. Test: score the VAE latent after the same per-cell projection, or bge with phase-2 weight decay.
- **If built.** A cell-corrupted negative block: by-parent sibling pool over `cell_to_pb_per_level[0]`, threaded like `module_pools`, scored with `batched_matvec`, passed as a second block to the NCE; pools restricted to phase-1 active cells; rho in {0, 0.25, 0.5}. Gate: 2-cluster ARI up, probe and broad ARI unchanged.

### 5. Simplex plus deviation head — LATER
- Gives the topic family the VAE's reach with named fold-change programs. Revisit only if the topic probe still sits below 0.85 after 1 and 3.

### 6. Direction readout (within-cluster ICA / sparse PCA on the latent) — DO in parallel, no training
- Makes bge and the VAE usable for fine types now; off the critical path.

### Dropped (do not re-propose)
More marginal bits, sketch rescaling, coarsening changes, context re-ranking, likelihood re-weighting, gene-level NCE decoder, gene-pair tree negatives, `rest` for this question, `masked-vae` at beta = 1.

## Recommended approach: step 1 in detail (residual bits)

Design decisions:
- Rewrite bits `node_dim..finest_dim` (7..9) of every cell's code BEFORE `assign_groups`, through a private helper `finest_codes(data_vec, proj_kn, level_dims, params)` used by both entry points in `collapse_data/mod.rs`. Low bits preserved exactly; everything downstream unchanged.
- Recursive bisection per node: top residual component -> left/right; recompute residuals in each child against the child's own (per-batch) profile; repeat to depth `finest_dim - node_dim`. Bit `node_dim + d - 1` = side at depth d, so per-level masking yields exactly the depth-k partition and any stopped split still nests.
- Residual: Pearson, `(x - n_c lambda) / sqrt(n_c lambda)`, 0 where lambda = 0, clipped at +-sqrt(n); genes kept if node count >= 5, at most 2000 by residual variance; sigma-hat = median mean-square residual over genes.
- Components: n <= 2048 exact Gram eigen (`nalgebra::SymmetricEigen`) -> u1, s1, s2, v1 oriented so the largest |loading| is positive (deterministic left/right). Above: seeded landmarks (512), fit the whole depth on landmarks, one streaming pass assigns the rest via `bits_from_tree`.
- Edge: `s1 > (1 + margin) sigma (sqrt n + sqrt p)`, margin 0.10. Below edge: `Keep` (default; split applied, flagged) or `Stop` (leaf).
- Per split: loadings, top +-20 contrast genes with post-split weighted lfc (`DispersionTrend::fisher_weight`), two-group Poisson LLR (cross-checked against `DcPoissonStats::from_profiles`), VE ratio, per-batch side counts, `passes_edge`, `applied`. Per node: VE ratio residual vs marginal.
- Params: `MultilevelParams.residual_bits: Option<ResidualBitsParams>` (`None` in `new()` and all non-senna sites). senna `CollapseArgs`: `--fine-bits residual|marginal` (default residual), hidden `--pb-split-edge-margin`, `--pb-split-below-edge`, `--pb-residual-genes`; `--fine-bits marginal` reproduces today's tree bit for bit.
- Output: `MultilevelCollapseOut.pb_tree: Option<PbTree>` -> `PreparedData.pb_tree` -> `{out}.pb_tree.json` written beside `cell_to_pb.parquet` by topic/vae/masked-topic (gene ids mapped to names, full loadings dropped), registered in the manifest (`RunOutputs.pb_tree`).
- Log line (in `rewrite_high_bits` and the collapse-structure block): nodes x depth, splits applied, above-edge counts per depth, median s1/edge, median VE ratio residual vs marginal.
- `--from`: build pb-samples from the inherited finest labels instead of fresh marginal codes so votes are unanimous; `pb_tree` is None on that path.
- Follow-up PR (multi-batch only): `MoveGuard::accept_move_to` default method, a `ContrastGuard` that vetoes sibling moves against the split's own contrast, re-anchoring of pb-samples that changed node after the coarsest sweep.

Files:
- new `data-beans-alg/src/collapse_data/residual_bits.rs` + `residual_bits_tests.rs` (registered in `collapse_data/mod.rs:37-56`)
- `collapse_data/mod.rs` (hook :556-561 and :900-906; `MultilevelParams` :64-129; `MultilevelCollapseOut` :58-61; `--from` :650-701)
- `collapse_data/refine.rs` (`RefineCollectCtx` :228-252; diagnostic :346-379; output :480-495)
- `senna/src/refine_weighting.rs` (`CollapseArgs` :120-242), `senna/src/topic/common.rs` (`PreparedData` :254-273, `LoadCollapseArgs` :496-541, params :596-611), `senna/src/postprocess/viz_prep.rs` (writer beside `write_cell_to_pb` :143-183), `senna/src/run_manifest.rs` (:675-702, :1331-1335, :1408-1410), call sites `topic/cmd.rs:617-627`, `vae.rs:436-445`, `masked_topic.rs:1490-1499`
- the `MultilevelParams` literal sites listed above

Tests (TDD, failing first, sibling `*_tests.rs`, `////` headers, no dataset names or numbers in source):
1. Pearson residual closed form (clipping, gene floor, lambda = 0).
2. Planted program recovered: one node, 60 cells, 400 genes, 24 program genes at 3-fold, Poisson draws; sign agreement >= 0.9 modulo flip; passes edge.
3. Null node stays below edge; `Stop` yields bits 0 / not applied; `Keep` applies and flags.
4. `rewrite_high_bits` preserves low bits, bounds high bits, one record per node, no-op when depth is 0.
5. Landmark path agrees with exact (>= 0.9).
6. End-to-end on a synthetic `SparseIoVec` (8 lineages x 2 fine states x 40 cells): marginal bits agree with the fine state < 0.7, residual bits > 0.9, per node.
7. Two-group LLR matches `DcPoissonStats`; VE ratio is 1 under a random partition.
8. `build_reproject_offsets` stays bounded with rewritten codes; refine hierarchy test reused with residual-style offsets.

Sequencing: skeleton + failing tests 1-4,7 -> residual/eigen/edge/LLR/VE -> recursion + `rewrite_high_bits` -> end-to-end test 6 -> hooks, `pb_tree` threading, log line -> landmark path -> senna CLI, writer, manifest, literal sites -> `--from` fix -> (later PR) refine guard.

## Verification

1. `cargo test -p data-beans-alg -p senna` green; `cargo fmt`; clippy `-D warnings`.
2. This donor, `--fine-bits marginal` vs default: identical `cell_to_pb.parquet` for marginal (byte-identity guard on the hash path).
3. Tree gate before training: the log line's VE ratio residual > marginal; labelled check with the scorer's metadata: pure-pseudobulk excess over shuffle and majority-vote ARI ceiling at the finest level (today 0.23). The CD4 T node splits' top contrast genes should be the measured program.
4. `senna topic` x 3 reps and `senna vae` x 3 reps, default 10-bit sketch, CUDA, `--preload-data`, scored with `score_granja.py`: report probe, 2-cluster ARI, broad/fine ARI against today's bands (topic 0.78-0.79 / 0.00-0.17 / 0.39-0.45; vae 0.89 / 0.15 / 0.39-0.42; bge bar 0.88 / 0.50).
5. Same runs with `--num-levels 1` (finest only, the unit chain rule) as the free A/B for idea 3.
6. Record outcomes in memory `senna_granja_granularity_findings.md`.

## After step 1
- Step 3 (masked split target) only if the tree's finest-depth structure is real (above-edge splits in the T nodes) and the topic probe is still below the bar.
- Step 4 premise test for bge dominance; sibling negatives only if negatives, not phase 2, explain it.
- Step 6 direction readout independently.

## Step 1 outcome (2026-09-06, branch ypp/residual-pb-bits)

Tree gate on this donor, default 10-bit sketch, residual high bits (128 nodes x depth 3, 687 splits, all applied):
- `--fine-bits marginal` reproduces the previous run's `cell_to_pb.parquet` exactly (byte-identity guard).
- Above the noise edge by depth: 124/127, 185/229, 187/331. Variance-explained ratio, median over nodes: residual 2.79 vs marginal 1.03.
- CD4 naive/memory, labelled: finest-level purity 0.878 vs shuffle 0.670 (marginal tree 0.740 vs 0.702); memory-majority pseudobulks 70 vs 9 shuffled (marginal 18 vs 12); majority-vote ARI ceiling 0.568 (marginal 0.228); middle level 0.514 (marginal 0.177); coarse level unchanged at 0.118.
- CD4-dominant node splits name GZMK, S100A4, ANXA1, KLRB1, CCL5, IL32 on one side and CCR7, SELL, LTB with the translation genes EEF1A1/TPT1 on the other; bit-1 agreement with the label 0.83-0.94 in the purer nodes, lower where a node also holds NK/CD8 cells and the first split takes the cytotoxic axis.
- Pseudobulk-level topic proportions on pure CD4 pseudobulks: probe 0.88-0.92 vs chance 0.54 with the residual tree (110 pure pbs), 0.84 vs chance 0.74 with the marginal tree (43 pure pbs). The decoder learns the axis from the rows now.

Training readouts (current main binary, senna 0.14.1; the previous session's numbers came from a 0.16.0 branch binary, so only same-binary comparisons count):
| run | broad ARI | fine ARI | CD4 n/m k-means | probe |
|---|---|---|---|---|
| topic, marginal tree | 0.277 | 0.164 | 0.00 | 0.746 |
| topic, residual tree x3 | 0.31-0.33 | 0.17-0.19 | 0.00-0.07 | 0.72-0.78 |
| vae, residual tree x3 | 0.41-0.43 | 0.27-0.29 | 0.11-0.15 | 0.907-0.910 |

Reading: the tree does what it was built to do and the topic decoder picks the axis up at pseudobulk level, but the cell latent does not move because the encoder input is the coarsened axis whose mega-bucket hides the program (the earlier finding). The decisive next check is residual tree + `--max-coarse-features 0` (full encoder axis; the old tree gave probe 0.863 there). Dominance (k-means) is untouched by the tree, as the objective analysis predicted; that is the split-likelihood step.

Full-axis check (residual tree + `--max-coarse-features 0`, one run): broad ARI 0.236, CD4 n/m k-means 0.058, probe 0.852. The encoder can read the program once the axis is uncoarsened (0.75 -> 0.85), lineage ARI pays as before, dominance does not appear. The marginal control was stopped as unnecessary: the tree's gain is on the rows and the decoder side; the cell-level topic readout is set by the encoder input and the objective.

## Tree refinement passes (2026-09-06, uncommitted on ypp/residual-pb-bits)

Framing: the marginal hash is the initial tree; each pass fixes a measured defect and is scored by the collapse alone (`senna topic -i 1`, seconds), label-free (variance-explained ratio, above-edge counts) and labelled (lineage purity and majority-vote ARI per level; CD4 naive/memory purity and ceiling). Names to apply at cleanup: pass 1 = rebin (`RebinParams`, `--pb-rebin-sweeps`, `rebin.rs`), pass 2 = resplit (`--pb-resplit residual|marginal`, `resplit.rs`).

| tree | levels (pbs) | broad purity coarse / finest | fine-type ARI coarse / finest | CD4 purity finest | CD4 ceiling finest |
|---|---|---|---|---|---|
| marginal hash | 128 / 511 / 1006 | 0.527 / 0.605 | 0.335 / 0.399 | 0.740 | 0.228 |
| + resplit (residual high bits) | 128 / 484 / 815 | 0.527 / 0.820 | 0.335 / 0.623 | 0.878 | 0.568 |
| + rebin (DC-Poisson re-sort of cells among top nodes, bits re-packed: 10 nodes, depth 6) | 69 / 237 / 419 | 0.881 / 0.910 | 0.706 / 0.745 | 0.920 | 0.704 |
| + stop below the noise edge | 65 / 180 / 248 | 0.881 / 0.907 | 0.706 / 0.735 | 0.920 | 0.704 |

Findings:
- rebin with all same-batch groups as candidates converges in ~3 sweeps and collapses the 128 marginal nodes to the ~10 lineages; the freed bits must go to the bisection (re-pack), otherwise the tree has 69 leaves. With re-pack the coarsest level is purer than the old finest level and naive vs memory separates at depth 1-2 of the T node.
- stop-below-edge removes 40% of the leaves at no purity cost: depth 5-6 splits on this donor are noise (more cells would fill them; the edge verdict is the label-free version of that statement).
- Sweep count (3 vs 10) does not matter. Pass 3 (leaf sharpening by likelihood on the split's contrast genes) not yet tested; the remaining finest impurity is ~9% broad.

Open before firming: multi-batch behaviour (rebin confines moves to a batch; resplit profiles per batch; interplay with the BBKNN refine untested, this donor is single-batch); scale (all-groups candidates are O(N x K); switch to sketch-kNN candidates for large N); one training readout with the final tree (lineage ARI must not regress; probe expected to move only with the encoder axis).

## Frontier driver + repaired rsvd (2026-09-06, uncommitted)

The recursion is replaced by a frontier: every leaf proposes its split in parallel, proposals are applied in decreasing two-group likelihood ratio until the next level's leaf target (2^dim per level) is met, and the nested levels are packed into prefix codes whose widths replace the level dims downstream. Kernels (profiles, residual build, scores, side sums, variance ratio) are rayon-parallel; the leading component comes from matrix-util's `rsvd`, which was REPAIRED: pivoted-LU basis without its permutation -> thin QR half-steps; oversampled basis kept through the projection. Planted-spike test in `matrix-util/src/dmatrix_rsvd_tests.rs`.

Gate at equal leaf counts (targets 128/512/1024), repaired rsvd everywhere:
| tree | leaves | broad purity coarse / finest | fine ARI finest | CD4 purity finest | CD4 ceiling finest |
|---|---|---|---|---|---|
| marginal hash | 128/502/941 | 0.652 / 0.726 | 0.501 | 0.780 | 0.310 |
| residual growth from marginal nodes | 128/512/1010 | 0.652 / 0.848 | 0.637 | 0.881 | 0.580 |
| reassign cells + residual growth, floor 8 | 128/512/1024 | 0.893 / 0.914 | 0.743 | 0.913 | 0.681 |
| same, floor 4 | 128/512/1024 | 0.892 / 0.918 | 0.735 | 0.905 | 0.653 |

Firm design: hash -> reassign cells (3 sweeps, all same-batch nodes as candidates; kNN candidates for scale) -> grow to the leaf targets with floor 8, edge verdicts recorded, Keep policy. Cleanup still to do: names (reassign_cells / no resplit), CLI (`--pb-tree marginal|refined` style switch instead of `--fine-bits` + hidden knobs), artifact naming of roots vs coarsest groups, matrix-util version bump, remove `repack_low_bits` remnants, multi-batch test, memory-bounded rebin for large N.

## Final tree, training readouts (2026-09-06, same binary, repaired rsvd, `--pb-tree refined` default)

| run | broad ARI | fine ARI | CD4 n/m 2-means ARI | probe |
|---|---|---|---|---|
| topic, `--pb-tree marginal` | 0.412 | 0.270 | 0.029 | 0.779 |
| topic, refined tree | 0.383 | 0.222 | 0.225 | 0.845 |
| vae, refined tree | 0.469 | 0.302 | 0.076 | 0.919 |

The refined tree moves the topic cell latent with the coarsened encoder input unchanged: naive vs memory becomes a visible gradient inside the T island (UMAP `rb/umap_refined_tree.png`), where the marginal hash gives a fully mixed island. Lineage ARI within the topic family's 0.08 noise (single runs). Progress bars: reassign cells inherits the DC-Poisson sweep bar; the tree growth has a leaf-count bar toward the finest target with the level in the message.

## Residual pseudobulk rows (branch ypp/residual-pb-rows, uncommitted, 2026-09-06)

`senna topic --pb-residual-rows`: a copy of the finest pseudobulks as their excess over the parent (floored at zero, rescaled to the child's mass; empty rows dropped), added as an extra training level on the finest axis ahead of the collapsed levels (own decoder slot; finest stays last). 782 of 1024 finest pseudobulks carry excess on this donor.

| run (single) | broad ARI | fine ARI | CD4 n/m 2-means | probe |
|---|---|---|---|---|
| tree only, default coarsening | 0.383 | 0.222 | 0.225 | 0.845 |
| tree + residual rows, default coarsening | 0.352 | 0.182 | 0.157 | 0.802 |
| tree only, full axis | 0.223 | 0.142 | 0.061 | 0.895 |
| tree + residual rows, full axis | 0.229 | 0.169 | 0.040 | 0.831 |

REFUTED (augmentation form): the excess rows lower the probe by 0.04-0.06 and the dominance readout with either encoder axis. Code removed, branch deleted. Not tried: the consistent form (cells residualised against their root at inference too, topics = within-lineage programs); the augmentation result says the train/inference mismatch is what hurts, so that form would be a different model, not a flag. Side finding: tree + full axis reaches probe 0.895 (the bge bar) at a lineage cost (broad 0.22), so the encoder's gene axis is the remaining lever: coarsen genes by residual co-variation (the gene tree) so the program stays visible without the full-axis cost.

