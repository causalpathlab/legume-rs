#!/usr/bin/env Rscript
# Compare our binomial spline GAM p-values against mgcv (tradeSeq's GAM engine).
# mgcv::gam with a penalized cubic-regression spline + binomial family is exactly the
# reference implementation, minus tradeSeq's NB-family/offset choice for counts.
#
# Generate the input TSVs first (from faba):
#   GAM_COMPARE_DIR=<dir> cargo test -p faba --bin faba \
#     assoc::gam::tests::dump_gam_compare_data -- --ignored --nocapture
# Usage: Rscript gam_compare.R <dir>   (dir holds gam_cells.tsv, gam_ours.tsv)

suppressMessages(library(mgcv))
dir   <- commandArgs(trailingOnly = TRUE)[1]
cells <- read.delim(file.path(dir, "gam_cells.tsv"))
ours  <- read.delim(file.path(dir, "gam_ours.tsv"))

genes  <- unique(cells$gene)
mgcv_p <- sapply(genes, function(g) {
  d <- cells[cells$gene == g, ]
  fit <- tryCatch(
    gam(cbind(k, n - k) ~ s(t, bs = "cr", k = 5), family = binomial, data = d),
    error = function(e) NULL)
  if (is.null(fit)) return(NA_real_)
  summary(fit)$s.table[1, "p-value"]
})
mg <- data.frame(gene = genes, p_mgcv = as.numeric(mgcv_p))

df  <- merge(ours, mg, by = "gene")
nlp <- function(p) -log10(pmax(p, 1e-16))

cat(sprintf("\nmgcv binomial GAM vs ours — genes: %d\n", nrow(df)))
cat(sprintf("Spearman  -log10(p_mgcv) vs -log10(p_quasi):  %.3f\n",
            cor(nlp(df$p_mgcv), nlp(df$p_quasi), method = "spearman")))
cat(sprintf("Spearman  -log10(p_mgcv) vs -log10(p_binom):  %.3f\n\n",
            cor(nlp(df$p_mgcv), nlp(df$p_binom), method = "spearman")))
o <- df[order(df$cov, df$slope), c("gene","slope","cov","p_mgcv","p_binom","p_quasi","effect")]
o[,c("p_mgcv","p_binom","p_quasi")] <- signif(o[,c("p_mgcv","p_binom","p_quasi")], 3)
print(o, row.names = FALSE)
