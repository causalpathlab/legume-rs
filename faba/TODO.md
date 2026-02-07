# TODO: DartSeq Site-Expression Correlation Algorithm

## Goal

New `dartseq-site-expr` subcommand: correlate gene expression counts with site beta values within genes, per cell type. Filter unreliable betas from low-expression genes. Flexible modes (filter/correlate/both) and granularity (per-cell/per-cell-type).

## Files to Create

- `src/site_expr_corr.rs` — Pearson/Spearman correlation, `GeneCorrelation` struct, Parquet output
- `src/run_site_expr.rs` — `SiteExprArgs`, `AnalysisMode`, `Granularity`, `CorrMethod` enums, `run_site_expr()` orchestration

## Files to Modify

- `src/run_dartseq_count.rs` — Extract `SiteCallingParams` struct (shared site-calling fields), flatten into `DartSeqCountArgs`, make `find_all_methylated_sites`, `gather_m6a_stats`, `collect_gene_m6a_stats`, `estimate_m6a_stat` `pub(crate)`
- `src/main.rs` — Register `DartseqSiteExpr` subcommand
- `src/lib.rs` — Export `site_expr_corr` module

## Algorithm (3 Phases)

1. **Site Calling** — reuse `find_all_methylated_sites()` → `DashMap<GeneId, Vec<MethylatedSite>>`
2. **Joint Collection** (parallel by gene) — `count_reads_per_gene()` for expression + `collect_gene_m6a_stats()` for betas, join by cell barcode, filter by `--min-gene-count`
3. **Output** — per-cell-type: dense Parquet with correlation stats; per-cell: filtered sparse matrix + correlation summary

## Key CLI Flags

- `--mode filter|correlate|both`
- `--granularity per-cell|per-cell-type`
- `--min-gene-count 5` — filter threshold
- `--min-cells-for-corr 10` — min cells for correlation
- `--corr-method pearson|spearman`
- Site-calling params via `#[command(flatten)]` from `SiteCallingParams`

## Key Reuse

- `gene_count.rs:22` — `count_reads_per_gene()` (already pub)
- `run_dartseq_count.rs:628,1116` — site calling + per-cell beta (make pub(crate))
- `data/cell_membership.rs` — `CellMembership::from_file()`, `.cell_types()`, `.matches_celltype()`
- `common.rs` — `format_data_triplets()`, `ToBackend`
- `statrs` crate — `StudentsT` for correlation p-values
