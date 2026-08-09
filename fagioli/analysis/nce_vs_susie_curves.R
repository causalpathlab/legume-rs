#!/usr/bin/env Rscript
##
## ROC and precision-recall curves: NCE embedding against per-trait SuSiE-RSS.
##
## Every number reported so far has been an AUC on a BALANCED subset -- the
## positive class against one chosen negative class, ignoring everything else.
## The real problem is ~32 causal variants in 900, so precision-recall over the
## full set is the honest picture and ROC is the flattering one. Both are drawn
## so the gap between them is visible.
##
## Two tasks, because they are different questions:
##
##   detection      causal (pleiotropic + trait-specific) against everything
##   specificity    trait-specific against pleiotropic, among causal variants
##
## Variants are pooled across seeds; the NCE fit is not reproducible run to run
## (candle's CPU init cannot be seeded), so a single seed's curve would be
## noise on top of noise.
##
## Reads the fixture written by
##   cargo test -p fagioli --release --test export_three_class -- --ignored
##
## Run: Rscript fagioli/analysis/nce_vs_susie_curves.R

suppressPackageStartupMessages({
  library(arrow)
  library(susieR)
})

EXPORT <- local({
  for (p in c("fagioli/target/three_class_export", "target/three_class_export",
              "../target/three_class_export")) {
    if (dir.exists(p)) return(p)
  }
  stop("fixture not found -- run: cargo test -p fagioli --release ",
       "--test export_three_class -- --ignored")
})
OUTFILE <- file.path(dirname(EXPORT), "nce_vs_susie_curves.png")
SEEDS <- c(20250808, 111, 222)
N_IND <- 800
SNPS_PER_BLOCK <- 150
N_BLOCKS <- 6

read_mat <- function(path) {
  d <- as.data.frame(read_parquet(path))
  as.matrix(d[, setdiff(names(d), "snp_id"), drop = FALSE])
}
read_ld <- function(path) { r <- read_mat(path); dimnames(r) <- NULL; r }

participation_ratio <- function(x) {
  s2 <- sum(x^2); s4 <- sum(x^4)
  if (s4 <= 0) return(NA_real_)
  s2^2 / s4
}

## ---- curves ---------------------------------------------------------------

## Both curves from one pass. `score` high = predicted positive.
curves <- function(score, label) {
  keep <- !is.na(score)
  score <- score[keep]; label <- label[keep]
  o <- order(score, decreasing = TRUE)
  lab <- label[o]
  tp <- cumsum(lab); fp <- cumsum(1 - lab)
  p <- sum(lab); n <- sum(1 - lab)
  list(
    fpr = c(0, fp / n), tpr = c(0, tp / p),
    recall = tp / p, precision = tp / (tp + fp),
    baseline = p / (p + n),
    auroc = sum(diff(c(0, fp / n)) * (head(c(0, tp / p), -1) + tail(c(0, tp / p), -1)) / 2),
    ## AUPRC by the interpolation-free step rule, which does not flatter.
    auprc = sum(diff(c(0, tp / p)) * (tp / (tp + fp)))
  )
}

## ---- gather ---------------------------------------------------------------

acc <- list()
for (seed in SEEDS) {
  sid <- format(seed, scientific = FALSE)
  z <- read_mat(file.path(EXPORT, sprintf("z_%s.parquet", sid)))
  cls <- as.integer(read_mat(file.path(EXPORT, sprintf("labels_%s.parquet", sid)))[, 1])
  nce <- read_mat(file.path(EXPORT, sprintf("nce_%s.parquet", sid)))
  m <- nrow(z); tt <- ncol(z)

  pip <- matrix(0, m, tt); eff <- matrix(0, m, tt)
  for (b in seq_len(N_BLOCKS)) {
    idx <- ((b - 1) * SNPS_PER_BLOCK + 1):(b * SNPS_PER_BLOCK)
    R <- read_ld(file.path(EXPORT, sprintf("ld_%s_block%d.parquet", sid, b - 1)))
    for (t in seq_len(tt)) {
      fit <- tryCatch(susie_rss(z = z[idx, t], R = R, n = N_IND, L = 5, verbose = FALSE),
                      error = function(e) NULL)
      if (is.null(fit)) stop("susie fit failed at seed ", sid, " block ", b, " trait ", t)
      pip[idx, t] <- susieR::susie_get_pip(fit)
      eff[idx, t] <- coef(fit)[-1]          # already integrates over inclusion
    }
  }
  eff[rowSums(abs(eff)) == 0, ] <- NA_real_

  acc[[length(acc) + 1]] <- data.frame(
    class      = cls,
    z_norm     = sqrt(rowSums(z^2)),
    z_pr       = apply(z, 1, participation_ratio),
    nce_norm   = nce[, "u_norm"],
    nce_pr     = nce[, "pr_fitted"],
    susie_pip  = rowSums(pip),
    susie_pr   = apply(eff, 1, participation_ratio)
  )
  cat(sprintf("seed %-9s done\n", sid))
}
d <- do.call(rbind, acc)

## ---- two tasks ------------------------------------------------------------
## Detection: causal against everything. High score = causal.
det <- list(
  "NCE  ||u_g||"   = list(s = d$nce_norm,  l = as.integer(d$class > 0)),
  "SuSiE  sum PIP" = list(s = d$susie_pip, l = as.integer(d$class > 0)),
  "raw  ||z_g||"   = list(s = d$z_norm,    l = as.integer(d$class > 0))
)
## Specificity: trait-specific against pleiotropic, among causal only.
cz <- d[d$class > 0, ]
spec <- list(
  "NCE  PR(V u_g)" = list(s = -cz$nce_pr,  l = as.integer(cz$class == 2)),
  "SuSiE  PR(eff)" = list(s = -cz$susie_pr, l = as.integer(cz$class == 2)),
  "raw  PR(z_g)"   = list(s = -cz$z_pr,    l = as.integer(cz$class == 2))
)

COL <- c("#B4413C", "#2E6F9E", "#3E8E5A")

panel <- function(arms, kind, title) {
  cs <- lapply(arms, function(a) curves(a$s, a$l))
  if (kind == "roc") {
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "false positive rate",
         ylab = "true positive rate", main = title, las = 1)
    abline(0, 1, col = "grey75", lty = 2)
    for (i in seq_along(cs)) lines(cs[[i]]$fpr, cs[[i]]$tpr, col = COL[i], lwd = 2.2)
    legend("bottomright", bty = "n", lwd = 2.2, col = COL[seq_along(cs)],
           legend = sprintf("%s  (AUROC %.3f)", names(arms), sapply(cs, `[[`, "auroc")))
  } else {
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "recall", ylab = "precision",
         main = title, las = 1)
    abline(h = cs[[1]]$baseline, col = "grey75", lty = 2)
    for (i in seq_along(cs)) lines(cs[[i]]$recall, cs[[i]]$precision, col = COL[i], lwd = 2.2)
    legend("bottomleft", bty = "n", lwd = 2.2, col = COL[seq_along(cs)],
           legend = sprintf("%s  (AUPRC %.3f)", names(arms), sapply(cs, `[[`, "auprc")))
    mtext(sprintf("dashed: chance = %.3f", cs[[1]]$baseline), side = 1, line = 3.6, cex = 0.7)
  }
  invisible(cs)
}

png(OUTFILE, width = 1500, height = 1450, res = 150)
op <- par(mfrow = c(2, 2), mar = c(5, 4.5, 3.5, 1.2), cex.main = 1.0)
r1 <- panel(det,  "roc", "Detection: causal vs rest")
r2 <- panel(det,  "pr",  "Detection: causal vs rest")
r3 <- panel(spec, "roc", "Specificity: trait-specific vs pleiotropic")
r4 <- panel(spec, "pr",  "Specificity: trait-specific vs pleiotropic")
par(op)
invisible(dev.off())

cat("\n== detection: causal vs rest ==\n")
for (i in seq_along(det))
  cat(sprintf("  %-16s AUROC %.3f   AUPRC %.3f\n", names(det)[i], r1[[i]]$auroc, r2[[i]]$auprc))
cat(sprintf("  chance AUPRC = %.3f\n", r2[[1]]$baseline))

cat("\n== specificity: trait-specific vs pleiotropic (causal only) ==\n")
for (i in seq_along(spec))
  cat(sprintf("  %-16s AUROC %.3f   AUPRC %.3f\n", names(spec)[i], r3[[i]]$auroc, r4[[i]]$auprc))
cat(sprintf("  chance AUPRC = %.3f\n", r4[[1]]$baseline))

cat(sprintf("\nwrote %s\n", OUTFILE))
