# `faba` design & methods notes

## Methods — describes code that exists

| doc | what it is |
|---|---|
| [`profiling-methods.md`](profiling-methods.md) | **BAM → per-cell features.** DART-seq m6A, A-to-I editing, alternative polyadenylation, gene counts and cell calling, SNP genotyping — the test, the null and the thresholds for each. Also lists where the code and its own `--help` text disagree. |

---

The annotation and lineage write-ups moved to [`senna/docs`](../../senna/docs/README.md) with the
subcommands they describe. faba's product is the per-cell feature matrices; everything that reads
one and fits a model is `senna`.
