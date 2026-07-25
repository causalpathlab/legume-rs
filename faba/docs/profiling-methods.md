# Feature profiling in `faba` — methods

How `faba` turns alignment files into per-cell feature matrices: m6A methylation, A-to-I editing,
alternative polyadenylation, gene counts, read depth, and SNP genotypes.

Every default given here is the shipped default, and every claim was read off the code rather
than off the help text (where the two disagreed, §9 says so). References are collected in §10.

---

## 1. What is shared by every modality

**Reading the BAM.** Reads are dropped if the duplicate flag is set. The pileup-based modalities
(m6A, A-to-I, SNP) additionally require `MAPQ ≥ --min-mapping-quality` (20), drop secondary and
supplementary alignments, drop paired reads that are not properly paired, and require each
individual base to have `Phred ≥ --min-base-quality` (20). **`faba genes` and `faba depth` do
not apply those filters** — they take every non-duplicate read with a gene tag. That asymmetry is
deliberate (counting wants sensitivity, variant calling wants specificity) but it is worth stating
in a write-up rather than leaving for a reader to discover.

**Counting molecules, not reads.** Reads carrying the same UMI (`--umi-tag`, `UB`) are collapsed
to one observation per cell per gene, in the manner of UMI-tools [7]. `--no-umi-dedup` turns this
off. This is *separate from*, and on top of, the duplicate-flag filter.

**Which cells are real.** Every modality inherits one cell set, called by `faba genes` (§5).

**Output.** Sparse matrices in Zarr (default, zipped) or HDF5, with feature rows keyed
`{gene}/{modality}/{channel}` — e.g. `{gene}/m6a/methylated`. Site-level rows carry the position:
`{gene}/m6a/{chr}:{pos}/methylated`.

---

## 2. `dartseq` — m6A methylation

DART-seq [1, 2] fuses the cytidine deaminase APOBEC1 to the m6A reader YTH, so cytidines *next to*
a methylated adenosine get edited C→U and show up as C→T in the reads. The catalytically-dead
YTH mutant is the control.

**Where to look.** Only at reference-validated DART motifs: `RAC` on the plus strand (R = A or G,
then A, then the C that gets edited) and the reverse-complement `GTY` on the minus strand.

**The test.** At each motif C, `faba` compares the C→T rate in the signal sample against the rate
at *the same base* in the pooled control. The null is not "no conversion" — it is **"this base
converts at the same rate in signal and control."** That framing is what makes the control do real
work: a germline C/T variant converts equally in both arms, so it fails the test rather than
having to be masked out.

The 2×2 table (signal/control × converted/unconverted) is tested one-sided:

- **Fisher's exact test** [3] when any cell of the table has fewer than 5 reads, or total coverage
  is under 100;
- otherwise a **beta-binomial likelihood-ratio test** with a shared overdispersion ρ
  (`--m6a-contrast-overdispersion`, 0.02), testing `H₀: p_signal = p_control`. The one-sided
  p-value is ½·P(χ²₁ ≥ D).

**Null-cell QC (de-dilution).** Before discovery, a fast pre-pass tallies each cell's
conversions at reference motif positions and drops the cells that edit no more than the
catalytically-dead control does. These are not bad cells — droplet calling, gene
complexity and mitochondrial fraction all pass them — the reporter simply did not work
in them, and every existing QC stage is expression-based and therefore blind to that.
Leaving them in contributes coverage without signal: measured on DART data, **90.9% of
the cells covering MYC convert nothing while carrying 74.5% of its coverage**, which is
enough to bury the gene entirely.

This is QC, not a hypothesis test, so it has no significance level. The cut is placed
where the *discarded* population stops being distinguishable from the control —
`--cell-scan-tolerance` (default **1.0**) is how much editing the discarded pool may
still show, as a multiple of the control — and 1.0 is the *data-driven* point: cut
exactly where the discarded pool's rate equals the control's, so nothing demonstrably
real is thrown away. Nothing is tuned. Raising it cuts deeper and concentrates the kept
pool; the logged `dropped/control` reports the cost. Alternatively
`--cell-scan-control-tail` places the cut on the control's own scale ("keep cells
editing more than 98% of depth-matched control cells"). Both are quantile/ratio rules on
an empirical reference — **not** tests: no p-values and no multiplicity anywhere in cell
QC. Every run logs its operating point in *both* units (`dropped/control 1.00; cut sits
at control p95`), because the correspondence is data-dependent: tolerance 1.2 mapped to
p95 on one panel and p98 on another, so read the reported percentile rather than
assuming a fixed equivalence.

Under `-v`, a side-by-side **logit-scale** histogram of the per-cell conversion rate
(signal vs control) is printed. Logit because the rates are ~1e-4 to 1e-2 and a linear
axis collapses them into the leftmost bins. It shows what the method rests on: the
signal distribution is the control's own mode *plus* a long upper tail where the control
has essentially no mass. The signal arm alone is **not** bimodal — the separation lives
in the comparison, which is why the control library is required rather than optional.
`--quantify-competent-only` extends the filter to the m6A output matrices as well.
It is **off by default and scoped to m6A alone**: faba's rule is that every modality
inherits one cell set, so restricting the m6A matrices makes their columns a subset of
the gene/apa/atoi matrices and cross-modality joins will drop cells. Left off, the
matrices carry every QC-passing cell — measured whole-genome, ~60% of those are null
cells, so a per-cell methylation rate read straight off the default matrix runs low.
The per-cell audit carries a `kept` column, so the same filtering can be done
downstream instead. The control matrices are never restricted either way.

The filter applies to the **signal arm only** — selecting control cells on apparent
activity would select on background and inflate the null. There is no off switch:
discovering on cells where the reporter never worked is not an alternative analysis,
just a diluted one. The scan no-ops on its own when there is no control arm to
calibrate against (A-to-I, or m6A run without `--control-bam`). A per-cell audit goes
to `{output}_m6a_cell_qc.tsv.gz`, with `scored` separating "assessed and rejected" from
"too little coverage to assess". `faba dartseq` and `faba all` share the same knobs, so
the two paths cannot drift. Measured end to end on chr19 + MYC: 131 → 235 selected
sites, and MYC is called only once the null cells are dropped.

**Putative sites vs the test.** A site is a *putative candidate* on the sequencing pattern alone:
the RAC/GTY motif plus observed WT C→U at/above the signal floors — signal coverage ≥
`--min-coverage` (10) and signal conversions ≥ `--min-conversion` (5). Everything else is the *test*
that decides selected vs unselected, applied after discovery: control coverage ≥
`--edit-control-min-coverage` (3) and an absolute excess `≥ --m6a-min-delta` (0.02) on the raw rates
`a/n`, then the FDR. A putative site that misses any of these is *recorded* (not dropped) in
`m6a_sites_unselected.parquet` with a `reason` (`low_control` / `delta` / `fdr`), so every candidate
is accounted for.

There is no control-fold gate. A Bullseye-style `p_WT / p_MUT ≥ 1.25` guard was measured on three
DART replicates and found inert: 94–99% of putative sites passed it and removing it changed the
selected-site count by exactly zero, because a site clearing a 0.02 absolute excess off a ~0.1%
control rate already clears 1.25×. The delta guard subsumes it.

**Multiple testing.** Set by `--m6a-test-level`. Under `site` (the default) each putative motif C is
tested and Benjamini–Hochberg [4]-corrected across all sites genome-wide. Under `gene` (opt-in) each
gene's putative C's are pooled into one WT-vs-MUT 2×2, the effect-size guards run on the pooled
counts, and BH runs *across genes* — but pooling dilutes a single strong site among the gene's weaker
C's, so on real chr19 data it calls *fewer*, not more (a strong 6%-vs-0% site drops below the delta
floor once pooled); a selected gene's result, with its reason, goes to `m6a_genes.parquet`. Either
way `-q/--fdr` (0.1) is a lenient **target FDR**, not a per-feature threshold; `--fdr 1.0` disables
it entirely, leaving the coverage + delta gates as the field-standard filter.

**Discovery is pooled, not stratified.** When cells are grouped by expression (§8), discovery scans
each gene's pooled WT marginal over the grouped cells — a putative site needs only the motif and
observed C→U, and the pooled marginal detects everything any single group would, with the full WT
evidence per site (so the gene-level pooled 2×2 is not under-counted). The grouping
is driven by expression and never by the editing signal, which is what keeps the test honest.

**Quantification.** A second pass counts, per cell and per site, converted and unconverted reads.
Sites seen in fewer than `--site-min-cells` (10) cells are dropped — this is the **reproducibility**
control, the single-cell analogue of the field's replicate-concordance requirement (scDART-seq keeps
a site only if seen in ≥ 10 cells). **Only cells with at least one converted read at a site
contribute a row** — worth stating, because it means the zeros in the matrix are structural, not
observed.

**Requires a control.** The command errors out without `--control-bam`. m6A cannot be told apart
from genomic C/T variation without one.

---

## 3. `atoi` — A-to-I RNA editing

Inosine is read as guanosine, so editing shows up as A→G (or T→C on the minus strand) against the
reference. There is **no control arm**: every reference A is a candidate.

**The test.** For a site with `n` reads and `k` alt reads, the null is that all `k` are sequencing
noise: `k ~ BetaBinomial(n, α, β)` with mean `ε = --error-rate` (0.01) and intra-site correlation
`ρ = --overdispersion` (0.1). The p-value is the upper tail `P(K ≥ k)`. With ρ = 0 this degenerates
to a plain binomial. This is the single-condition test used by SAILOR [5] and JACUSA2 [6].

Gates before testing: `n ≥ --min-coverage` (**5** — note this differs from dartseq's 10) and
`k ≥ --min-conversion` (**3**). Then Benjamini–Hochberg [4] across all sites, keeping `q ≤ --fdr`
(0.05).

Quantification is as in §2, with channels `edited` / `unedited`.

---

## 4. `apa` — alternative polyadenylation

**Finding poly(A) sites.** Reads whose soft-clipped tail is a run of A (plus strand) or T (minus
strand) mark a cleavage site: at least `--polya-min-tail-length` (10) A/T bases with at most
`--polya-max-non-a-or-t` (3) mismatches. Internal priming is filtered out — if the genome around
the putative site is already A/T-rich (`≥ --polya-internal-prime-count` (7) A/T in a
`--polya-internal-prime-window` (10) bp window), the site is discarded, because the poly(A) tail
may be genomic rather than transcribed.

**The model (SCAPE [8]).** `faba` implements the SCAPE mixture: each read fragment `(x, l, r)` is
generated from one of `K` poly(A) sites, with the fragment's 3′ end distributed around the site,
the poly(A) tail length `s` uniform, and a **uniform noise component** absorbing fragments that
belong to no site. Site positions and widths are held fixed; only the mixing weights are fit, by
EM. **`K` is chosen by BIC** — candidates are added greedily, refit, and the lowest-BIC model
kept, stopping after two consecutive increases. Nearby sites are then merged if BIC improves.
Cells are assigned to components by hard argmax and UMI-deduplicated.

**The fast default path.** When only PDUI is wanted (the default: `--no-pdui` off, `--mixture`
off), the EM is skipped entirely. Read 3′-ends are clustered by **recursive bisection at the
largest gap** that still leaves `≥ --min-coverage` (10) reads on each side, gaps below
`--merge-distance` (50 bp) are not split, and the top two clusters are kept — provided the
runner-up carries at least 2% of the dominant cluster's mass. Fragments are then assigned to the
nearer of the two.

**PDUI** is only defined for genes with exactly two active sites. The matrix stores **counts** in
two channels, `{gene}/apa/proximal` and `{gene}/apa/distal`; the ratio
`PDUI = distal / (distal + proximal)` is left for the analyst to form, so that coverage is not
thrown away.

**There is no significance test in the APA path.** Sites are selected by BIC (or by the mass rule
on the fast path). No p-values, no FDR.

---

## 5. `genes` — gene counts, and cell calling

Counts reads per gene, splice-aware by default. A read is called **unspliced** if any aligned
block falls outside every annotated exon; otherwise **spliced** (the alevin-fry "S+A" convention
[9]). Three matrices per batch: total, spliced, unspliced. These are what `faba gem` consumes.

**This command *does* call cells.** The default `--cell-filter` is `empty-drops`, and the cell set
it produces is inherited by every other modality. It is the union of two rules:

1. **OrdMag knee** [10] — take the barcode at rank `expected_cells × 0.01` among the top
   `--expected-cells` (3000) barcodes, and keep everything with at least 10% of its count.
2. **EmptyDrops** [11] — for barcodes below the knee but above `--cell-min-umis` (500), estimate
   the ambient RNA profile from very low-count barcodes by Simple Good–Turing smoothing [12],
   score each candidate's likelihood under that ambient multinomial, build a null by Monte-Carlo
   (`--cell-sims`, 10000), and keep barcodes that are significantly *unlike* ambient at
   Benjamini–Hochberg `q < --cell-fdr` (0.01).

`--cell-filter nnz` opts out, leaving only the non-zero-count floors — that gives an unfiltered
superset of a CellRanger filtered matrix.

**Mitochondrial QC.** The per-cell mitochondrial fraction is always reported. Cells above a cutoff
are dropped, where the cutoff is `--max-mito-frac` if given, and otherwise a **data-driven elbow**
(the point of maximum perpendicular distance from the chord of the sorted MT% curve). MT genes are
excluded from the matrix unless `--keep-mito`.

Cell calling looks at spliced counts and at all biotypes; the *quantified* gene set is then
narrowed by `--gene-type` and mitochondrial exclusion. Genes are kept if seen in
`≥ --row-nnz-cutoff` (10) cells; cells if they carry `≥ --column-nnz-cutoff` (10) genes.

---

## 6. `snp` — variant discovery and genotyping

**Discovery.** At each pileup position with `depth ≥ --min-coverage` (10), take the most frequent
non-reference allele and require `≥ --min-alt-count` (3) reads and `≥ --min-alt-freq` (0.1) of the
depth. Known sites can be force-called from a VCF/BCF/parquet with `--known-snps`.

**Genotyping.** Three genotype likelihoods (`RR`, `RA`, `AA`) are computed either from per-base
quality scores, in the manner of Li's framework [13] (the default), or — with a constant error
rate — from the binomial pileup model used by cellSNP-lite and Vartrix [14]:
`P(D|RR) = Binom(k; n, ε)`, `P(D|RA) = Binom(k; n, ½)`, `P(D|AA) = Binom(k; n, 1−ε)`.

The call is the **maximum a posteriori** genotype under priors `P(het) = 0.001`,
`P(hom-alt) = 0.0001` (fixed, not exposed as flags), and the confidence is
`GQ = −10·log₁₀(1 − P(best))`. A site is a no-call if `depth < --min-depth` (5) or
`GQ < --min-gq` (20). **There is no multiple-testing correction in the SNP path** — GQ is the only
confidence gate.

In single-cell mode `faba` writes two same-shaped matrices, `{batch}_snp_alt` and
`{batch}_snp_depth`, so a per-cell B-allele fraction is `alt / depth`.

**One asymmetry to know about.** The SNP *mask* used to protect RNA-editing sites from being
thrown away as variants applies a VAF filter (`--snp-mask-min-vaf`, 0.35) **only inside
`faba all`**. Standalone `faba snp` builds its mask without it.

---

## 7. `depth`, `pwm`, `pileup`, `metagene` — descriptive routines

None of these fit a model or produce a p-value.

- **`depth`** bins the genome at `--resolution-kb` and counts, per cell, the **number of reads
  overlapping each bin** (via an interval tree) — not per-base coverage.
- **`pwm`** collects base counts in a ± `--window` (10) bp window around called sites, reverse-
  complementing minus-strand sites. The output is a base-frequency matrix, not a log-odds PWM.
- **`pileup`** renders one gene's sites as an ASCII histogram, or (with `--gtf`/`--bam`) a faceted
  Miami plot: sites above, gene model in the middle, read depth below, one panel per cell type.
- **`metagene`** maps each site to its 5′UTR / CDS / 3′UTR and bins its relative, strand-aware
  position within that feature.

---

## 8. `all` — the full pipeline

The steps run in this order, and each one's output constrains the next:

```
SNP  →  genes  →  [expression grouping]  →  ATOI  →  m6A  →  APA
```

- **SNP** runs first, in bulk mode, and produces the variant mask. It is not fatal if it fails.
- **genes** calls cells and picks the expressed gene set. **Every downstream modality inherits
  both** — this is what makes the modalities directly comparable, since they share a cell axis.
- **The expression grouping** (`--cluster-resolution`, default 0.5, on) builds one shared cell
  grouping by random projection → randomised SVD → kNN graph → **Leiden** [15]. It is used to
  stratify m6A and A-to-I *discovery*, for the reason given in §2. It is **not** cell typing, and
  should not be reported as such.
- **ATOI** runs masked by the SNP mask, and produces the editing mask.
- **m6A** runs masked by the editing mask (a C→T at an edited site is not methylation). It is
  **skipped, not failed**, if no `--control-bam` is given. The SNP mask is *not* applied by
  default, because the WT-vs-MUT contrast already rejects germline variants.
- **APA** runs last, because the SCAPE EM is the expensive step.

The pipeline deliberately relaxes the per-modality count floors (`--gene-min-cells`,
`--cell-min-genes` default to 0 here, versus 10 standalone) so that the cell and gene axes are set
once, by the gene-counting step, and not silently re-filtered by each modality afterwards.

---

## 9. Where the code and its own help text disagree

Found by reading both. These are documentation bugs, not method bugs, but they will mislead anyone
writing this up from `--help` alone:

| flag / text | says | actually |
|---|---|---|
| `faba all` `about` | `SNP → genes → ATOI → APA → m6A` | runs `SNP → genes → ATOI → m6A → APA` |
| `--mixture-max-k` (m6A, A-to-I) | "max components to test **via BIC**" | m6A/A-to-I call components from smoothed-density **modes**; `max_k` is only a cap. BIC genuinely selects `K` **in APA only** |
| `--intron-buffer` (genes) | "reads within this buffer are discarded" | declared, but **no consumer** — the splice classifier uses a strict "any base outside an exon" rule |
| `--valid-genes` help | `{batch}_genes_kept.tsv.gz` | the file written is the pooled `genes_kept.tsv.gz` |

---

## 10. References

1. Meyer KD. *DART-seq: an antibody-free method for global m⁶A detection.* Nat Methods 16,
   1275–1280 (2019).
2. Tegowski M, Flamand MN, Meyer KD. *scDART-seq reveals distinct m⁶A signatures and mRNA
   methylation heterogeneity in single cells.* Mol Cell 82, 868–878 (2022).
3. Fisher RA. *On the interpretation of χ² from contingency tables, and the calculation of P.*
   J R Stat Soc 85, 87–94 (1922).
4. Benjamini Y, Hochberg Y. *Controlling the false discovery rate.* J R Stat Soc B 57, 289–300
   (1995).
5. Deffit SN, et al. *The C. elegans neural editome reveals an ADAR target mRNA required for
   proper chemotaxis.* eLife 6, e28625 (2017). [SAILOR]
6. Piechotta M, et al. *JACUSA2: a framework for the RNA-seq-based detection of RNA modifications.*
   BMC Bioinformatics 23, 438 (2022).
7. Smith T, Heger A, Sudbery I. *UMI-tools: modeling sequencing errors in Unique Molecular
   Identifiers.* Genome Res 27, 491–499 (2017).
8. Zhou R, et al. *SCAPE: a mixture model revealing single-cell polyadenylation diversity.*
   Nucleic Acids Res 50, e66 (2022).
9. He D, et al. *Alevin-fry unlocks rapid, accurate and memory-frugal quantification of single-cell
   RNA-seq data.* Nat Methods 19, 316–322 (2022).
10. Zheng GXY, et al. *Massively parallel digital transcriptional profiling of single cells.*
    Nat Commun 8, 14049 (2017). [OrdMag / Cell Ranger]
11. Lun ATL, et al. *EmptyDrops: distinguishing cells from empty droplets in droplet-based
    single-cell RNA sequencing data.* Genome Biol 20, 63 (2019).
12. Gale WA, Sampson G. *Good-Turing frequency estimation without tears.* J Quant Linguist 2,
    217–237 (1995).
13. Li H. *A statistical framework for SNP calling, mutation discovery, association mapping and
    population genetical parameter estimation from sequencing data.* Bioinformatics 27, 2987–2993
    (2011).
14. Huang X, Huang Y. *cellsnp-lite: an efficient tool for genotyping single cells.* Bioinformatics
    37, 4569–4571 (2021).
15. Traag VA, Waltman L, van Eck NJ. *From Louvain to Leiden: guaranteeing well-connected
    communities.* Sci Rep 9, 5233 (2019).
