#!/usr/bin/env Rscript
## Does the drift probe pick up novelty, and does it ignore covariate shift?
##
## Reads the per-cell fit score that `senna predict` now emits directly
## ({out}.predictive.parquet, column `llik_per_count`). Higher = better fit;
## novelty = lower.
##
##   A = in-distribution control (must NOT be flagged)
##   B = covariate shift        (must NOT be flagged as novel)
##   C = held-out topic         (must be flagged)
##
## Checks: AUC(A vs C) ~ 1, AUC(A vs B) ~ 0.5, and a titration detection floor
## obtained by mixing the A and C per-cell scores.
suppressMessages(library(arrow))

W <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(W)) stop("usage: analyze.R <workdir>")

fit_of <- function(tag) {
  d <- as.data.frame(read_parquet(file.path(W, paste0("pred_", tag, ".predictive.parquet"))))
  d$llik_per_count
}
sA <- fit_of("A")
sB <- fit_of("B")
sC <- fit_of("C")

## AUC via the rank (Mann-Whitney) identity; novelty score = -fit
auc <- function(indist, novel) {
  s <- c(-indist, -novel)
  y <- c(rep(0L, length(indist)), rep(1L, length(novel)))
  r <- rank(s)
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  (sum(r[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

thr <- quantile(sA, 0.05) # null threshold from the in-distribution control
below <- function(x) mean(x < thr)

fracs <- c(0, .05, .10, .25, .50, 1)
T <- 400L
set.seed(0)
titr <- sapply(fracs, function(f) {
  nn <- round(f * T)
  draw <- c(
    if (nn > 0) sample(sC, nn, TRUE),
    if (T - nn > 0) sample(sA, T - nn, TRUE)
  )
  below(draw)
})

cat(sprintf("AUC(A vs C novel)     = %.4f   (want ~1.0)\n", auc(sA, sC)))
cat(sprintf("AUC(A vs B covariate) = %.4f   (want ~0.5)\n", auc(sA, sB)))
cat(sprintf("flag rate (< A p05):  A=%.3f  B=%.3f  C=%.3f\n", below(sA), below(sB), below(sC)))
cat("\nper-batch fit (higher = better fit; novel lowest):\n")
for (nm in c("A", "B", "C")) {
  x <- get(paste0("s", nm))
  cat(sprintf(
    "  %s  n=%5d  mean=%+.3f  p05=%+.3f  p50=%+.3f  p95=%+.3f\n",
    nm, length(x), mean(x), quantile(x, .05), median(x), quantile(x, .95)
  ))
}
cat("\ntitration (novel fraction -> flag rate):\n")
for (i in seq_along(fracs)) cat(sprintf("  f=%.2f  flag=%.3f\n", fracs[i], titr[i]))

## ---- two-panel figure ----
col <- c(A = "#2b8cbe", B = "#41ab5d", C = "#e6550d")
for (open_dev in list(
  function() pdf(file.path(W, "probe_result.pdf"), 10, 4.2),
  function() png(file.path(W, "probe_result.png"), 1000, 420)
)) {
  open_dev()
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  dA <- density(sA)
  dB <- density(sB)
  dC <- density(sC)
  plot(dA,
    col = col["A"], lwd = 2, xlim = range(sA, sB, sC), ylim = c(0, max(dA$y, dB$y, dC$y)),
    main = "per-cell fit by batch", xlab = "log-lik / count", ylab = "density"
  )
  lines(dB, col = col["B"], lwd = 2)
  lines(dC, col = col["C"], lwd = 2)
  abline(v = thr, lty = 2)
  legend("topleft", c("A in-dist", "B covariate", "C novel", "A p05 null"),
    col = c(col, "black"), lwd = c(2, 2, 2, 1), lty = c(1, 1, 1, 2), bty = "n", cex = .9
  )

  plot(fracs, titr,
    type = "b", pch = 19, col = col["C"], ylim = c(0, 1),
    main = "titration: detection floor", xlab = "novel fraction in batch",
    ylab = "flag rate (< A p05)"
  )
  abline(0.05, 0.95, lty = 3, col = "gray50")
  grid()
  dev.off()
}
cat("\nsaved probe_result.{pdf,png}\n")
