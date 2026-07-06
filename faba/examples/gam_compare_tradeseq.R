#!/usr/bin/env Rscript
# Compare our binomial spline GAM against tradeSeq::associationTest directly.
#
# tradeSeq is built for many genes sharing one cell population, where per-cell library
# size is a stable cross-gene normalization. The generator emits exactly that: a gene
# panel over one shared pseudotime, symmetric ± slopes keeping library size ~flat. We
# fit the whole panel with one fitGAM and compare its per-gene associationTest p-values
# to ours. (tradeSeq models NB counts with a per-cell offset — coverage n is constant
# per gene here, so the NB trend test answers the same question as our binomial one.)
#
# Generate the input TSVs first (from faba):
#   GAM_COMPARE_DIR=<dir> cargo test -p faba --bin faba \
#     assoc::gam::tests::dump_gam_compare_data -- --ignored --nocapture
# Usage: Rscript gam_compare_tradeseq.R <dir>

suppressMessages(library(tradeSeq))
dir   <- commandArgs(trailingOnly = TRUE)[1]
cells <- read.delim(file.path(dir, "gam_cells.tsv"))
ours  <- read.delim(file.path(dir, "gam_ours.tsv"))

# Wide gene × cell count matrix + shared per-cell pseudotime.
ct     <- xtabs(k ~ gene + cell, data = cells)
counts <- matrix(as.integer(ct), nrow = nrow(ct), ncol = ncol(ct),
                 dimnames = list(rownames(ct), colnames(ct)))
tt  <- cells[!duplicated(cells$cell), c("cell", "t")]
tt  <- tt[order(tt$cell), "t"]
pt  <- matrix(tt, ncol = 1)
cw  <- matrix(1, nrow = ncol(counts), ncol = 1)

sce <- fitGAM(counts = counts, pseudotime = pt, cellWeights = cw, nknots = 5, verbose = FALSE)
at  <- associationTest(sce)
ts  <- data.frame(gene = rownames(at), p_tradeseq = at$pvalue)

df  <- merge(ours, ts, by = "gene")
nlp <- function(p) -log10(pmax(p, 1e-16))
ok  <- is.finite(df$p_tradeseq)
cat(sprintf("\ntradeSeq associationTest vs ours — genes: %d (usable %d)\n", nrow(df), sum(ok)))
cat(sprintf("Spearman -log10(p_tradeseq) vs -log10(p_quasi): %.3f\n",
            cor(nlp(df$p_tradeseq[ok]), nlp(df$p_quasi[ok]), method = "spearman")))
cat(sprintf("Spearman -log10(p_tradeseq) vs -log10(p_binom): %.3f\n\n",
            cor(nlp(df$p_tradeseq[ok]), nlp(df$p_binom[ok]), method = "spearman")))
o <- df[order(df$cov, df$slope), c("gene","slope","cov","p_tradeseq","p_binom","p_quasi","effect")]
o[,c("p_tradeseq","p_binom","p_quasi")] <- signif(o[,c("p_tradeseq","p_binom","p_quasi")], 3)
print(o, row.names = FALSE)
