# Feature profiling in `faba` — methods

How `faba` turns alignment files into per-cell feature matrices: m6A methylation, A-to-I editing,
alternative polyadenylation, gene counts, read depth, and SNP genotypes.

Every default given here is the shipped default, and every claim was read off the code rather
than off the help text (where the two disagreed, §10 says so). References are collected in §11.

---

## 1. What is shared by every modality

**Reading the BAM.** Reads are dropped if the duplicate flag is set. The pileup-based modalities
(m6A, A-to-I, SNP) additionally require `MAPQ ≥ --min-mapping-quality` (20), drop secondary and
supplementary alignments, drop paired reads that are not properly paired, and require each
individual base to have `Phred ≥ --min-base-quality` (20). **`faba count` and `faba depth` do
not apply those filters** — they take every non-duplicate read with a gene tag. That asymmetry is
deliberate (counting wants sensitivity, variant calling wants specificity) but it is worth stating
in a write-up rather than leaving for a reader to discover.

**Counting molecules, not reads.** Reads carrying the same UMI (`--umi-tag`, `UB`) are collapsed
to one observation per cell per gene, in the manner of UMI-tools [7]. `--no-umi-dedup` turns this
off. This is *separate from*, and on top of, the duplicate-flag filter.

**Which cells are real.** Every modality inherits one cell set, called by `faba count` (§5).

**Nothing is thresholded by the producers.** Every subcommand in §2–§6 writes every called cell,
every gene with a count, and every putative site with its statistics. The only floors are the
candidacy floors that keep an all-zero row from existing. p-values, odds ratios, edit ratios,
cells-per-site and nnz cutoffs are `faba qc` flags (§8), applied to stored columns, and
`faba qc-report` shows what each one keeps before it is applied.

**Output.** Sparse matrices in Zarr (default, zipped) or HDF5, with feature rows keyed
`{gene}/{modality}/{channel}` — e.g. `{gene}/m6a/methylated`. Site-level rows carry the position:
`{gene}/m6a/{chr}:{pos}/methylated`.

### 1.1 The gene model: merged features, spliced coordinates

Anything that asks *where in a transcript* a position falls — APA's poly(A) positions (§4),
`rel_pos`, the methylation mixture — needs a gene model, and there are two ways to get one wrong.
Both were live in `faba` until they were measured, so the reasoning is recorded here rather than
left in a commit. **The metagene no longer uses this model — see §1.2** — but it is what the
defects below were found on, and it still governs everything else.

**Do not use union spans.** `genomic_data` used to offer a `build_union_gene_model` that collapsed
every record of a feature type for a gene into a single `min(start)..max(stop)` interval. That is
not the feature; it is the *reach* of the feature. It has since been deleted — nothing called it
once the metagene moved to per-transcript models (§1.2) — but the measurement is why, and the same
mistake is available to anyone who reaches for `min`/`max` over a gene's records: the union CDS
span covers most of the gene, introns included, overlaps the union UTR span in most genes, and the
union 3′UTR span runs far longer than its spliced length.

Under a gene-level model, features had to be tested in the order 5′UTR → CDS → 3′UTR, so an
oversized CDS span *claimed* 3′UTR sites. Because those sites sit at the far end of the span, they
binned to the **last CDS bin**. That was the origin of a terminal-bin spike that looked like biology
and was not. (The metagene has since moved to per-transcript regions, which are disjoint, so it
needs no priority order at all — §1.2. The measurement stands as how the defect was found, and the
ordering problem is still live for anything that classifies a position against merged features.)

The tell that it was never library 3′ bias: the spike was **worst in CDS**, whereas 3′ coverage
pileup, adapter read-through, internal priming and 3′-end base quality all predict a spike in the
*3′UTR*. A second, independent argument applies to the m6A calls specifically — they are WT-vs-MUT
contrasts, and the MUT arm shares the library chemistry, so anything symmetric between the arms is
already cancelled before the metagene sees it.

**Measure position along the spliced feature.** A relative position taken along a genomic span
lets introns consume transcript coordinate. Position runs along the concatenated intervals,
mirrored on the reverse strand, so `bin = rel * nbins / spliced_len`. That correction applies on
both models: the metagene now takes it along one transcript's regions, everything else along the
gene's merged ones.

**Verification.** After the fix, an independent transcript-level classification agrees with the
gene-level model up to isoform convention (a site that is CDS in one isoform and 5′UTR in another,
where that model tested 5′UTR first, or a site inside an overlapping gene's CDS). What survives is
the expected biology: CDS climbs into the stop codon and the 3′UTR is front-loaded just past it.

**The same defect reached APA.** `apa` built its 3′UTR regions from the same union model, which
matters more there than anywhere else: APA's whole estimand is *position within the 3′UTR*, so a
region running through introns and CDS corrupts the estimate rather than just the plot, and a
large share of admitted regions had a span overlapping the gene's own CDS. 3′UTRs are now merged
annotated exons, `--min-utr-length` gates on the **spliced** length, and
a read is charged only its exonic bases — so a read lying in an intron of the 3′UTR contributes
nothing, and one spanning an intron is credited its spliced length, not its genomic length.

**Cross-isoform CDS/3′UTR overlap is real, and is kept.** After the fix, a share of regions still
have a 3′UTR exon overlapping some CDS record. **This is biology, not a residual artifact, and it
must not be "corrected" away.** No transcript in the annotation has a `UTR` record overlapping a
`CDS` record *of the same transcript* — the annotation is internally consistent within a
transcript. The overlap exists only *across* isoforms, which is exactly what alternative last exons
and alternative stop usage produce: the same genomic base is genuinely coding in one isoform and
3′UTR in another, and both records are true.

So that share is a **statement about the transcriptome**, not a quality metric to minimise. A
gene-level merged model reports that base as 3′UTR because for some isoform it is one, and reads
there are informative about 3′-end usage — which is what APA estimates. The model is therefore
**gene-level, merged across isoforms** — stated plainly rather than implying a per-transcript
resolution it does not have.

### 1.2 The metagene is per-transcript; the rest of `faba` is not

`metagene` (§7) is the one exception to everything above. It elects **one transcript per gene** —
the longest spliced — and places each site on that transcript's own 5′UTR / CDS / 3′UTR, which are
disjoint by construction. It does so because its job is to be held against a *published* profile:
scDART-seq's metagene was made with MetaPlotR [19], and a shape difference is only informative if
the procedure is the same one. **Fidelity to that procedure beats improving on it.**

Nothing else moved. APA still measures 3′-end usage on the merged model, `rel_pos` is still the
offset along merged exons, and the methylation mixture still shares that axis. Their estimands are
about *any-isoform* exonic evidence, where merging is the right answer rather than the wrong one.

**The cost is not small.** Electing one transcript leaves more sites unassigned than the merged
model does, and the 5′UTR track collapses: almost all of the merged model's 5′UTR sites lie in
regions that are 5′UTR only in non-canonical isoforms (alternative first exons and TSS). CDS and
3′UTR are robust. **Neither 5′UTR count should carry weight**, and a swing of that size on a small
track is the reason to state which model produced a figure.

The effect on the profile is not subtle: under the merged model the 3′UTR's terminal bin dominates
the plot while the stop-codon bin barely registers; per transcript it is the other way round. The
two models support opposite readings of the same sites.

**Deviations from MetaPlotR**, all deliberate, all documented where they are implemented
(`genomic-data/src/transcript.rs`, `faba/src/site_analysis/metagene.rs`):

1. *Isoform election.* Its `visualize_metagenes.R` writes `dist[duplicated(gene_name), ]`, which
   keeps rows two through N — dropping every single-isoform gene outright and keeping all but the
   shortest of the rest. Its README variant de-duplicates an *unsorted* table and so elects by file
   order. Neither matches what either document says it does; we implement the stated intent, and
   break ties on `transcript_id` because `read_gff_record_vec` collects through `par_bridge` and
   record order is not reproducible. Running their script on our own `--dist-measures` output shows
   the gap directly in the scale factors.
2. *Site-weighted medians.* Bin widths come from median region sizes over the **assigned sites**,
   not the transcript set — `visualize_metagenes.R` computes them from `dist`, one row per site. The
   two readings differ materially in the 3′UTR, and the transcript-weighted one would draw it too
   narrow while still looking entirely plausible.
3. *`--include-non-coding`* has no MetaPlotR counterpart. That track sits on its own [0,1] axis and
   its density is normalised within itself.

**`rel_pos` is a transcript coordinate.** The site parquet's `rel_pos` is the strand-aware offset
along the gene's **merged exons** — introns consume none of it. It was an offset from the gene
start in genomic space, which for a typical human gene says more about intron content than about
where in the mRNA a site sits. The column is **nullable**, and is null for an intronic site: such a
site has no transcript coordinate, and substituting the nearest exon edge would put a value there
that no reader could distinguish from a real one.

The exon model is the same shape used above — gene-level, merged across isoforms, so a base exonic
in *any* isoform is exonic here.

The **methylation mixture** uses the same coordinate. Its position covariate and the `gene_length`
that normalises it must sit on one axis, and both were genomic: consistently wrong together, so the
fit was valid but the axis was mostly intron. Moving both to spliced shrinks the covariate's range
by the gene's intron content, and fitted components then sit toward the 3′ end of the mature
transcript, matching the stop-codon enrichment in §7. Under the old axis that fraction was not
interpretable. Sites with no transcript position are dropped from the fit rather than nudged onto
the nearest exon, and the count is logged.

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

The 2×2 table (signal/control × converted/unconverted) is tested one-sided by a single test:
**Fisher's exact test** [3], conditioning on all four margins, returning `P(signal converted ≥
observed)`.

There used to be a second branch — an overdispersed beta-binomial likelihood-ratio test, taken once
every cell of the 2×2 was deep enough. Both tests were individually correct; the *dispatch* was
not. Two different nulls met at a count threshold, so the p-value jumped discontinuously across
it: one extra converted read in the control, or doubling the coverage at a fixed effect, could
move a site across orders of magnitude. A statistic that is not monotone in its own evidence
cannot rank sites. The exact branch was kept because a catalytically-dead control converts next
to nothing at most sites, so the exact test was already the common path. The cost, stated
plainly: overdispersion is now unmodelled rather than applied to the deep minority.

**Null-cell QC (de-dilution).** Before discovery, a fast pre-pass tallies each cell's
conversions at reference motif positions and drops the cells that edit no more than the
catalytically-dead control does. These are not bad cells — droplet calling, gene
complexity and mitochondrial fraction all pass them — the reporter simply did not work
in them, and every existing QC stage is expression-based and therefore blind to that.
Leaving them in contributes coverage without signal, which is enough to bury a focally
methylated gene entirely when most of the cells covering it never convert.

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
at control p95`), because the correspondence is data-dependent and differs between
panels, so read the reported percentile rather than assuming a fixed equivalence.

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
matrices carry every QC-passing cell — a large share of those are null
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
the two paths cannot drift.

**Putative sites, and no test.** A site is a *putative candidate* on the sequencing pattern alone:
the RAC/GTY motif, at least `--min-conversion` (1) converted signal read, and total coverage (signal
+ control) ≥ `--min-coverage` (1). That is the whole producer-side decision. Every putative site is
written to `m6a_sites.parquet` with its statistics — `pv`, `log_odds`, `log_odds_se`, `coverage`,
`converted`, `control_coverage`, `control_converted` and the raw base counts — quantified into the
`_site` matrix, and pooled into the gene-level matrix. There is no p-value cutoff, no odds-ratio
guard, no control-coverage floor and no mask here: those are `faba qc` flags (`--site-max-pv`,
`--site-min-log-odds`, `--site-min-coverage`, `--site-min-cells`, …; §8) applied to stored columns,
so a different cut is a rerun of `qc`, never of a BAM scan. The floors default to 1 because the only
site they exclude is one with no converted read at all, i.e. an empty row.

What follows is the reasoning behind the `qc` knobs. It was written when the guard and the cutoff
lived in the producer; the argument is the same, only the flag that carries it moved.

`--site-min-log-odds` uses the **raw** cross-product `ln((a_w·u_m)/(u_w·a_m))`, with no continuity correction, so
a control that never converts reads `+∞` and passes. That is the common case, not an edge case: a
catalytically-dead control converts nothing, or next to nothing, at most sites. Correcting the guard
would invert its meaning: with
`a_m = 0` a Haldane-corrected guard passes only when the WT odds exceed `0.5/(n_MUT + 0.5)`, an
implied **minimum WT rate** of 12.5% at `n_MUT = 3` and 25% at `n_MUT = 1` — which is precisely the
pathology the odds ratio was brought in to remove. Worked case: `(a_w, u_w, a_m, u_m) = (30, 4970,
0, 3)` is a 0.6% WT site whose control converts 0 of 3 reads. Raw, that is `+∞`; corrected, −3.148,
i.e. a claim that the control converts 23× *more*, off three reads. Both agree the site is unproven
(Fisher p = 0.982) — but only the raw guard lets it be dropped as `pvalue` ("no evidence") instead
of `log_odds` ("no effect").

The default `1e-4` is not tuned; it means "direction only". On integer counts the smallest odds ratio
above 1 a table can express is `1 + 1/(u_w·a_m)`, which exceeds 1e-4 for every table with `u_w·a_m`
below 10⁴ — essentially all of them. So the threshold falls in the gap between "exactly 1" and the
next expressible value, and anything from about 1e-6 to 1e-4 behaves identically. A genomic C/T
variant converts equally in both arms, so its two cross-products are the same float and its log odds
is exactly `0.0`: rejected at any positive floor, at any depth. That is the one job this guard has.

**Why a ratio and not a difference.** `--m6a-min-delta`, an absolute `p_WT − p_MUT` floor, was
removed because it measured the wrong thing in three separate ways.

It was on the wrong *scale*. The Fisher exact test's null is `OR = 1`, which is multiplicative; a
difference is additive. Guard and test were not measuring the same quantity.

It barely consulted the *control*. DART background is small next to any real signal rate, so
`p_WT − p_MUT ≈ p_WT`: a flag documented as an effect-size guard behaves as a minimum-WT-rate
filter, tracking the WT conversion rate rather than the odds ratio. It rejects deep, clean sites
with large odds ratios and can rank a gene's strongest site, by odds ratio, last.

And it was denominated in a unit the method itself rescales. Null-cell QC leaves `a_w` alone and
shrinks `u_w`, so it multiplies the WT *odds* by `1/f` — and `f` differs per gene (see above). One
fixed additive threshold therefore meant something different
at every gene. A log-odds threshold absorbs that rescaling as a constant shift.

**Do not read an effect size off a low-abundance site.** This is the sharpest form of the argument.
A rate difference is bounded by the larger arm's rate and quantized by depth, so it reports *depth*,
not effect: one converted read of five scores `delta = 0.20`, while a genuinely strong site at 40 of
744 scores 0.053. The thin site's delta is four times larger and means nothing. Two sites with
identical `1/5` WT arms score the same 0.20 whether the control has 300 reads or 20.

The site parquets therefore carry `log_odds` and `log_odds_se` beside `pv`. The standard error is
Woolf's [15], `sqrt(Σ 1/n_ij)` on cells corrected by the Haldane–Anscombe +0.5 [16, 17, 18] — applied
to the *reported* estimate only, never to the guard, for the reason given above. It is dominated by
the **smallest cell**
rather than by either library's depth — so it is large exactly where the evidence is thin, which is
how "do not trust this effect size" becomes a number instead of a warning. On the two `1/5` sites
above it reads 1.71 either way, but the Wald lower bounds separate them at 2.50 and −0.20.

Two caveats, stated so they do not arrive as bug reports. With `a_m = 0` the corrected control cell
is 0.5, so the SE is floored near `√2 ≈ 1.41` regardless of depth: at sites with no control conversion it flags
uncertainty without *ranking* it, and ranking there is `pv`'s job. And because the reported estimate
is corrected while the guard is not, the two can disagree in sign at a nearly-empty control — by
design, as the worked case above shows. `log_odds − 1.96·log_odds_se` is a Wald lower bound if one
is wanted; it is deliberately neither a column nor a filter, because to a normal approximation it is
the same one-sided test `pv` already reports exactly.

There is no separate control-fold gate, because the odds-ratio guard *is* one. A Bullseye-style
`p_WT / p_MUT ≥ 1.25` guard is the same family — on rates rather than odds, with the threshold at
1.25 rather than ~1.0 — and `faba qc --site-min-fold` exposes it, beside `--site-min-edit-ratio` and
`--site-min-converted`, so a Bullseye-style call set can be reproduced from the same producer
output. `faba qc-report` shows what any of them keeps on the data at hand.

**The two arms are not symmetric, and resampling cannot fix it.** The WT arm is de-diluted (restricted
to editing-competent cells) while the control is not, so the arms rest on different cell counts. That
asymmetry is real but it is not a *variance* problem, and matching sample sizes would not help.

The Woolf SE is dominated by the smallest cell, and at a typical site that cell is `a_m` — the
control's *converted* count, a few reads at most — not either library's depth: the deep unconverted
control cell contributes `1/u_m ≈ 0`. Subsampling the control down to WT depth therefore discards
the term that was already free and only widens the SE.

It would also add variance without removing bias. Simple random subsampling is unbiased for a
proportion, so the resampled control rate has the same expectation; it only inflates the variance by
`n_MUT/n_WT`, levelling the more precise arm down to the less precise one. Worse here specifically:
with only a handful of control conversions at most sites, subsampling turns one into none most of the time,
converting a measured background into a structural zero. And the closed form is what such a bootstrap
converges to anyway — the bootstrap variance of a proportion is `p(1−p)/n`, whose delta-method logit
SE is `1/a + 1/u`, and summing the arms gives Woolf exactly.

Ambient contamination of the control is likewise not a bias: ambient soup in a MUT library is MUT
RNA, so its conversion rate matches the cells it leaked from and the mixture collapses back.

What does survive is **selection, not contamination**. Competent cells are chosen by their conversion
at the same motif positions discovery then tests — the motif rule is deliberately shared between the
cell scan and the sifter — so `p_WT` is conditioned on an outcome-dependent event and `p_MUT` is not.
No work on the control arm fixes that. The contrast is coherent under one **stated assumption**: that
the control arm is homogeneous, with no editing-competent subpopulation. If every control cell shares
one background rate, conditioning on competence within it is vacuous, and comparing selected-WT to
all-MUT is a legitimate contrast — "the rate among editing cells" against "the one rate there is".
That premise is what a catalytically-dead control *is*, and it is also why control cells must never
be competence-filtered: selecting them on apparent activity would select on background. A second,
weaker assumption backs it: competence is scored over every motif position, so any single site
contributes ~1/n of its own selection evidence — negligible at a typical site, largest at exactly the
strong sites that drove the ranking.

Both are testable rather than merely asserted. The cell scan already fits a beta-binomial null from
the control arm, so running the competence call *on* the control and looking for a competent tail
tests the first directly. If the control is homogeneous, there is nothing to correct.

**Multiple testing: there is none.** Each putative motif C carries its own marginal p-value, and
`faba qc --site-max-pv` (**0.05**) keeps it or not on that value alone. No Benjamini–Hochberg and
no q-values: `m6a_sites.parquet` carries a single `pv` column and nothing beside it.

A marginal cutoff is **not** scale-free the way an FDR threshold is. "q ≤ 0.05" means the same thing
at 300 tests and at 300,000; "p ≤ 0.05" admits `0.05 × m` null sites, so its meaning moves with `m`
and the flag alone cannot tell you what you bought. `faba qc-report` therefore prints, for every
cutoff on the grid, what it keeps (§8); calibrating the cutoff is the user's call.

This is deliberate. BH [4] controls the FDR under independence or *positive regression dependence*,
and neighbouring candidate C's have neither: they are covered by the **same reads**, so their 2×2s
share cells and depth, and a read converted at one site is evidence *against* its unconverted
neighbour — the dependence is not even reliably positive. Under arbitrary dependence the valid
procedure is Benjamini–Yekutieli, which divides α by `Σ 1/i ≈ ln m`: an order-of-magnitude penalty
at the tens of thousands of putative sites of one library, i.e. calling almost nothing. Between "BY and call nothing" and
"stop claiming FDR control", claiming BH's guarantee while its assumption fails was the one
indefensible option. Two further measurements point the same way: BH was running on p-values that are
not uniform under H₀ (a site only exists once it clears `--min-conversion`, so the null tail is
truncated away), and on the post-guard subset. That second objection is *stronger* now than it was:
the subset is filtered by `--m6a-min-log-odds`, which is monotone in a monotone transform of the
Fisher statistic itself rather than merely correlated with it as the old delta guard was, so the
conditioning deflates every q by roughly #eligible/#putative and is even harder to defend. This
matches the field: Bullseye and scDART call sites by thresholds plus control fold and replicate
reproducibility, not by a genome-wide FDR. `--site-max-pv 1` disables the cutoff entirely.

**The unit is the site, never the gene.** A gene-level mode used to pool every putative C in a gene
into one 2×2 and test that. It is gone. faba's whole m6A method rests on **de-dilution** — cells that
never edit are removed *before* discovery, because leaving them in buries real signal in a
denominator of non-signal (see the null-cell QC above). Pooling a gene re-introduces exactly that
failure one level up: it averages a focal methylation site against the gene's non-methylated
positions, with no corresponding de-dilution step. The decisive objection is that a gene-level verdict
cannot say **which** C carries the mark, and that holds under any guard.

Under the retired delta guard the dilution surfaced as false negatives: a gene with a few strong
sites among many quiet ones pooled to a delta under the floor, so the gene was rejected at the effect
guard and every site inherited that verdict, while per site the strong ones are called. Under
the odds-ratio guard a pooled 2×2 of that shape clears the floor comfortably, so pooling now fails in
the opposite direction, by false **attribution** — one verdict inherited by every C in the gene, most
of which carry nothing. The direction of the error moved; the argument for the site as the unit did
not. Gene pooling is dilution-honest by construction and dilution-*blind* in practice, precisely on
the focally methylated genes the assay exists to find. `m6a_genes.parquet` / `atoi_genes.parquet`
are no longer produced, nor is `m6a_sites_unselected.parquet`: there is one site table per
modality, and `faba qc` writes what it drops to `m6a_sites_dropped.parquet` with a `reason`. The
`gene_pv` and `qvalue` columns are gone from the site parquets — `qvalue` had become a byte-for-byte
copy of `pv`, which is a lying name, not a schema.

**Discovery is pooled, never stratified.** Discovery scans each gene's pooled WT marginal over
every cell that passed QC. A putative site needs only the motif and observed C→U, and the pooled
marginal detects everything any single subset of cells would, with the full WT evidence per site,
so no site's 2×2 is under-counted. Supplying `--cell-membership` restricts *which* cells enter that
marginal, but the labels themselves reach only the second pass — they never split the test.

This is a deliberate reversal. Earlier versions auto-grouped cells by expression (random projection
→ randomised SVD → kNN → Leiden, behind `--cluster-resolution`) and stratified discovery on those
groups, on the theory that an edit confined to a sub-population is diluted by pooling. It was
removed, for three reasons. Mechanically, per-group scanning emits one site per (position, group)
and, after dedup, leaves each site's 2×2 holding a single group's reads — it *lost* evidence rather
than concentrating it. Statistically, calling a site when any of K groups clears the guard is a max
over K correlated tests, and this pipeline runs one exact test per site with no multiplicity
correction, relying on cross-replicate reproducibility instead — precisely the rule a
max-over-strata scan defeats. Biologically, the premise was wrong: conversion rate at a DART site is
set by catalytic competence (does this cell express functional APOBEC1–YTH) and by the site's m6A
occupancy, and neither is the axis a whole-transcriptome embedding partitions on. The dilution was
real but it was competence-structured, not cell-type-structured, which is why the null-cell QC
above recovers it — 18 → 66 reproducible sites — and the expression grouping did not.

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

Candidacy floors: `n ≥ --min-coverage` (**5**) and `k ≥ --min-conversion` (**3**), the same in
`faba atoi` and `faba all`. They are higher than m6A's 1 / 1 because A-to-I has no motif anchor: at
1 / 1 every reference A with a single mismatching read would be a site. No p-value cutoff is applied
here either — every putative site is written with its `pv`, and `faba qc --site-max-pv` decides, as a
marginal cutoff with no multiplicity correction and no gene-level pooling, for the reasons given in
§2. The argument is if anything stronger here: A-to-I has no control arm and no de-dilution
pre-pass, so a pooled gene test would average edited positions against unedited ones with nothing
to undo it. No SNP mask is applied: a germline A/G variant is a site with an edit ratio near 0.5 or
1, which `--site-max-edit-ratio` can drop, and the SNP call set is there to join against.

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

**The 3′UTR is a spliced model** — merged annotated exons, not a `min(start)..max(stop)` span, with
`--min-utr-length` gating on the spliced length and every UTR-relative coordinate (`alpha`, the
fragment start `x`, the fragment length `l`, the poly(A) position) measured along the exons. A read
inside a 3′UTR intron is not counted; a read spanning one is charged its spliced length. §1.1 gives
the measurements and the residual cross-isoform overlap this model still carries. Two consequences
worth knowing when comparing against older output: `genomic_start`/`genomic_stop` shift by 1 bp,
because the alpha→genomic map is now the exact inverse of the genomic→alpha map and previously was
not; and a fragment clipped at the UTR boundary is one base longer, because covered bases are now
counted rather than differenced.

---

## 5. `count` — gene counts, and cell calling

Counts reads per gene, splice-aware by default. A read is called **unspliced** if any aligned
block falls outside every annotated exon; otherwise **spliced** (the alevin-fry "S+A" convention
[9]). One matrix per batch, `{batch}_count`, with both tracks as rows
(`{gene}/count/{spliced|unspliced}`; sum a gene's two rows for its total). These are what
`senna gem` consumes.

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
`≥ --row-nnz-cutoff` (1) cells; cells if they carry `≥ --column-nnz-cutoff` (1) genes — i.e. only
empty rows and columns are dropped. An opinionated floor belongs to `faba qc` (§8).

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

**The call set and the allele-frequency track are different objects.** `snp_sites.parquet` (and
the matching VCF) is the call set: genotype, GQ, rsid, pooled allele counts. In single-cell mode
`faba` also writes one per-cell matrix, `{batch}_baf`, and it carries none of that — only two read
counts per cell per locus, on channels `alt` and `depth`, so a per-cell B-allele fraction is
`alt / depth`. It is named for what it measures rather than for the step that chose its positions,
because reading it as "the SNP output" invites treating a per-cell count as a per-cell genotype.

Two properties of that matrix are worth stating, since both differ from every other faba modality:

- **Rows are keyed on the locus, not on a gene** — `{chr}:{pos}/baf/{alt|depth}`. A variant is a
  coordinate; it does not belong to the gene whose region happened to fetch its reads. Keying rows
  by gene, as earlier versions did, gave a variant inside two overlapping genes two different row
  names and counted its reads twice. Genes are still required (`-g/--gff`) because they are how
  pass 2 finds reads to fetch, and each locus is now assigned exactly one owner gene so it is
  scanned once.
- **The two channels nest rather than partition.** Everywhere else a unit's channels are exclusive
  and sum to coverage (methylated + unmethylated, spliced + unspliced). Here `alt ≤ depth`, so BAF
  is `alt / depth` and summing the two channels is meaningless.

**The call set is an output, not a mask.** No other modality consumes it: the editing and APA
steps used to drop sites at called variants, and that was removed, for speed and because the
m6A contrast already rejects a germline variant on its own (equal conversion in both arms, §2).
Join `snp_sites.parquet` against a site table downstream if a variant overlap matters.

---

## 7. `depth`, `pwm`, `pileup`, `metagene` — descriptive routines

None of these fit a model or produce a p-value.

- **`depth`** bins the genome at `--resolution-kb` and counts, per cell, the **number of reads
  overlapping each bin** (via an interval tree) — not per-base coverage.
- **`pwm`** collects base counts in a ± `--window` (10) bp window around called sites, reverse-
  complementing minus-strand sites. The output is a base-frequency matrix, not a log-odds PWM.
- **`pileup`** renders one gene's sites as an ASCII histogram, or (with `--gtf`/`--bam`) a faceted
  Miami plot: sites above, gene model in the middle, read depth below, one panel per cell type.
- **`metagene`** follows MetaPlotR [19], so its output can be held against published m6A profiles.
  Each site is placed on **one elected transcript** — the longest spliced per gene, or every coding
  isoform under `--isoforms all` — and given that transcript's coordinate: 5′UTR in [0,1), CDS in
  [1,2), 3′UTR in [2,3). Within a transcript the three regions are disjoint, so nothing needs a
  priority order and no union span can claim a neighbour's sites. Position runs along the
  **spliced** region. §1.2 has the model, its measured cost, and the three places we follow
  MetaPlotR's stated procedure rather than its published code.
  Bins split between the regions in proportion to each region's **median** spliced length over the
  assigned sites, so they depend on the sites as well as the annotation — compare profile *shapes*
  between runs, not bar widths. A maximum would be one gene's: titin's merged CDS is 114,586 nt
  against a median of 1,347.
  The TSV keeps `#feature`, `genomic_bin` and `count`, then appends `bin_start`, `bin_end`, `frac`
  and `density` on MetaPlotR's rescaled axis, where the CDS keeps width 1 and each UTR is drawn at
  its median size relative to the CDS.
  `--dist-measures` writes MetaPlotR's own per-site table — its fourteen columns in its order, then
  `strand` and `rescaled_location` — so `visualize_metagenes.R` runs on faba output with only its
  input path changed. Note `coord` there is **1-based**, matching the `end` field of the 0-based BED
  MetaPlotR reads, whereas the site parquet stores 0-based. Its `utr3_st` column is the signed
  spliced distance from the stop codon, which is what its feature-distance plot is drawn on.
  **Counts are raw.** A bin is also taller where more of its positions were deep enough to test,
  which on a 3′-biased library means the last bins of the 3′UTR; a terminal peak is not evidence of
  enrichment on its own.

---

## 8. `qc` and `qc-report` — the one place faba thresholds anything

`faba qc` reads a faba output directory and writes a **new** fileset to `-o`, never in place. The
producers are inclusive by design (§1), so this is where every opinionated decision lives, and
`faba qc-report` puts the data behind the decision.

**Order of operations, per batch.**

1. **Cells** are decided once, on `{batch}_count`: a non-zero floor (`-c/--column-nnz-cutoff`, or
   the BIC-guarded 2-means suggestion under `--auto-cutoff`, as `data-beans squeeze`), then the
   data-beans MAD-outlier cell QC on detected genes and total counts (`--qc-mads` 5,
   `--qc-min-cell-nnz` 2; `--no-cell-qc` skips it). The same keep set is applied to every matrix
   of the batch, so the modalities stay column-aligned. The kept barcodes go to
   `{batch}_cells.tsv.gz` and the per-cell verdicts to `{batch}_cell_qc_report.tsv`.
2. **Sites** are decided on the parquet columns, in this order: `--site-min-coverage` (3, on
   signal + control reads), `--site-min-converted` (1), `--site-min-edit-ratio` /
   `--site-max-edit-ratio` (0 / 1, on `converted / coverage`), then for m6A `--site-min-fold` (1,
   signal rate over control rate; a clean control passes) and `--site-min-log-odds` (1e-4, the
   raw cross-product, §2), then `--site-max-pv` (0.05), then `--site-min-cells` (10): the number
   of kept cells carrying a converted read at the site, read off the `_site` matrices and pooled
   over batches. The kept rows are written under the same name; the dropped rows go to
   `{modality}_sites_dropped.parquet` with a `reason` column naming the first check that failed.
3. **`_site` matrices** keep both channels of every kept site, over the kept cells.
4. **Gene-level `{batch}_m6a` / `{batch}_atoi` are re-pooled** from the filtered site matrix
   (`{gene}/{modality}/{chr}:{pos}/{channel}` rows summed to `{gene}/{modality}/{channel}`), so
   gene level cannot disagree with the site cut. The producer's pooled matrix, which sums every
   putative site, is replaced.
5. **Every other matrix** (`_count`, `_apa`, `_*_mixture`, `_baf`, `_depth`) takes the cell keep
   set and `-r/--row-nnz-cutoff` (0 = drop only empty rows; `--auto-cutoff` suggests one).
6. Everything else is copied through, and `qc_summary.tsv` lists every written matrix with its
   shape before and after.

**`qc-report`.** Sweeps each `qc` criterion over a grid, one at a time with the others off, and
writes `{prefix}.qc_report.parquet` (`modality, criterion, unit, threshold, n_kept, n_genes`) plus
an ASCII chart on stderr — one panel per (modality, criterion), a row per threshold, `*` bars
scaled to the panel's largest count, in the style of `faba metagene`. The grids are dense at the
permissive end, because that is where the trade-off is decided. Editing modalities print a
−log10(p) histogram of every putative site first (rows with criterion `neglog10_pv_hist`, one per
bin), then sweep `max_pv`, `min_log_odds`, `min_fold`, `min_coverage`, `min_converted`,
`min_edit_ratio` and `min_cells`; `count` and `apa` sweep `min_cells` and `min_counts` per gene
unit, and `count` also `min_genes_per_cell`. No error rate is estimated: how a marginal p-value or
any other column should be calibrated is left to the user, with the table as the evidence.

---

## 9. `all` — the full pipeline

The steps run in this order, and each one's output constrains the next:

```
SNP  →  count  →  [depth]  →  ATOI  →  m6A  →  APA
```

Modalities do not mask one another, and no step applies a p-value or effect-size cutoff; the
pipeline's output is the inclusive fileset `faba qc` (§8) cuts.

- **SNP** runs first, in per-cell mode, and writes the call set and BAF matrices. Nothing downstream
  reads it. It is not fatal if it fails.
- **count** calls cells and picks the expressed gene set. **Every downstream modality inherits
  both** — this is what makes the modalities directly comparable, since they share a cell axis.
- **depth** (`--depth-resolution-kb`, opt-in) writes `{batch}_depth`, binned per-cell read depth.
  It is independent of every other step and nothing downstream consumes it, so it can fail
  without costing anything that follows. It sits directly after gene
  counting for one reason only — that is where the called-cell axis exists, and sharing it keeps
  the depth matrix's columns identical to every other modality's.
- **ATOI** writes every putative site (§3). Discovery is in bulk, over the cells step 1 called —
  as it is for m6A, so the two cannot disagree about which cells were compared. There is no cell
  grouping step; `--cluster-resolution` and the Leiden grouping behind it were removed, for the
  reasons in §2.
- **m6A** writes every putative site (§2). It is **skipped, not failed**, if no `--control-bam` is
  given.
- **APA** runs last, because the SCAPE EM is the expensive step.

The per-modality floors are the same in `all` and standalone: `--gene-min-cells` 1 and
`--cell-min-genes` 1 drop only empty rows and columns, so the cell and gene axes are set once, by
the count step, and every modality carries them unchanged to `faba qc`. `--site-min-cells` (1) and
the editing floors are likewise shared by const with the standalone commands.

---

## 10. Where the code and its own help text disagree

Found by reading both. These are documentation bugs, not method bugs, but they will mislead anyone
writing this up from `--help` alone. The rest of what this section used to list has since been
fixed in the help text itself, so only the live discrepancy is kept here:

| flag / text | says | actually |
|---|---|---|
| `--mixture-max-k` (m6A, A-to-I) | "max components to test **via BIC**" | m6A/A-to-I call components from smoothed-density **modes**, then truncate to `max_k` (`editing/mixture.rs`), so it is a plain cap and never a selection criterion. BIC genuinely selects `K` **in APA only**. The CLI help now says so; `fit_gene_mixture`'s own rustdoc still makes the stale claim |

---

## 11. References

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
15. Woolf B. *On estimating the relation between blood group and disease.* Ann Hum Genet 19,
    251–253 (1955). [the log-odds standard error]
16. Haldane JBS. *The estimation and significance of the logarithm of a ratio of frequencies.*
    Ann Hum Genet 20, 309–311 (1956). doi:10.1111/j.1469-1809.1955.tb01285.x [the +0.5 correction]
17. Anscombe FJ. *On estimating binomial response relations.* Biometrika 43, 461–464 (1956).
18. Gart JJ, Zweifel JR. *On the bias of various estimators of the logit and its variance with
    application to quantal bioassay.* Biometrika 54, 181–187 (1967).
19. Olarerin-George AO, Jaffrey SR. *MetaPlotR: a Perl/R pipeline for plotting metagenes of
    nucleotide modifications and other transcriptomic sites.* Bioinformatics 33, 1563–1564 (2017).
    doi:10.1093/bioinformatics/btx002 [the metagene convention, §1.2 and §7]
