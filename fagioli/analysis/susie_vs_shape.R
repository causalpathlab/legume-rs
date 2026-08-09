#!/usr/bin/env Rscript
##
## Per-trait SuSiE-RSS against the LD-free shape statistic.
##
## The two arms differ in exactly what is being claimed:
##
##   PR(z_g)      raw marginal rows, no LD anywhere, locus resolution
##   susie_rss    in-sample LD, one fit per (block, trait), variant resolution
##
## susieR gets the *in-sample* LD from the same genotypes, so there is no panel
## mismatch to pay for -- the baseline at its strongest.
##
## Reads the fixture written by
##   cargo test -p fagioli --release --test export_three_class -- --ignored
##
## Run: Rscript fagioli/analysis/susie_vs_shape.R

suppressPackageStartupMessages({
  library(arrow)
  library(susieR)
})

## cargo runs tests with cwd = the package directory, so the export lands under
## fagioli/target regardless of where this script is invoked from.
EXPORT <- local({
  for (p in c("fagioli/target/three_class_export", "target/three_class_export",
              "../target/three_class_export")) {
    if (dir.exists(p)) return(p)
  }
  stop("fixture not found -- run: cargo test -p fagioli --release ",
       "--test export_three_class -- --ignored")
})
SEEDS <- c(20250808, 111, 222)
N_IND <- 800
SNPS_PER_BLOCK <- 150
N_BLOCKS <- 6
N_HAP <- 6

## ---- statistics -----------------------------------------------------------

## Effective number of nonzero coordinates. PR(c * e_t) = 1; for an isotropic
## Gaussian of length T it tends to (T + 2) / 3.
participation_ratio <- function(x) {
  s2 <- sum(x^2)
  s4 <- sum(x^4)
  if (s4 <= 0) return(0)
  s2^2 / s4
}

## Mann-Whitney AUC of `score` ranking `pos` above `neg`.
auc_pair <- function(score, pos, neg) {
  s <- c(score[pos], score[neg])
  lab <- c(rep(1L, length(pos)), rep(0L, length(neg)))
  r <- rank(s)
  n1 <- sum(lab == 1L); n0 <- sum(lab == 0L)
  if (n1 == 0 || n0 == 0) return(0.5)
  (sum(r[lab == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

## Variants in LD with any of `seed_set`: same block and same haplotype index,
## which is how the fixture builds correlation.
ld_partners_of <- function(seed_set, m) {
  key <- function(g) paste(g %/% SNPS_PER_BLOCK, (g %% SNPS_PER_BLOCK) %% N_HAP)
  keys <- unique(key(seed_set - 1L))
  which(key(seq_len(m) - 1L) %in% keys)
}

read_mat <- function(path) {
  d <- as.data.frame(read_parquet(path))
  as.matrix(d[, setdiff(names(d), "snp_id"), drop = FALSE])
}

## The exported LD is f32, so the round trip leaves it asymmetric in the last
## bit and susieR warns. Symmetrize rather than let it do so silently.
read_ld <- function(path) {
  r <- read_mat(path)
  (r + t(r)) / 2
}

## ---- per seed -------------------------------------------------------------

rows <- list()
for (seed in SEEDS) {
  z <- read_mat(file.path(EXPORT, sprintf("z_%s.parquet", format(seed, scientific = FALSE))))
  cls <- as.integer(read_mat(file.path(EXPORT, sprintf("labels_%s.parquet",
                                                       format(seed, scientific = FALSE))))[, 1])
  m <- nrow(z); tt <- ncol(z)
  pleio <- which(cls == 1L); spec <- which(cls == 2L); nul <- which(cls == 0L)

  ## Arm 1: shape on raw rows. No LD read at all.
  pr_raw <- apply(z, 1, participation_ratio)

  ## Arm 2: per-trait SuSiE-RSS, one fit per (block, trait).
  pip <- matrix(0, m, tt)
  eff <- matrix(0, m, tt)
  in_cs <- matrix(FALSE, m, tt)
  for (b in seq_len(N_BLOCKS)) {
    idx <- ((b - 1) * SNPS_PER_BLOCK + 1):(b * SNPS_PER_BLOCK)
    R <- read_ld(file.path(EXPORT, sprintf("ld_%s_block%d.parquet",
                                            format(seed, scientific = FALSE), b - 1)))
    for (t in seq_len(tt)) {
      fit <- tryCatch(
        susie_rss(z = z[idx, t], R = R, n = N_IND, L = 5, verbose = FALSE),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      pip[idx, t] <- susieR::susie_get_pip(fit)
      eff[idx, t] <- coef(fit)[-1]
      cs <- susieR::susie_get_cs(fit, Xcorr = R)$cs
      if (length(cs)) in_cs[idx[unique(unlist(cs))], t] <- TRUE
    }
  }

  ## A variant's trait profile under SuSiE: its posterior effect across traits,
  ## gated by inclusion. PR of that is the same shape question, asked of a
  ## method that resolved LD first.
  prof <- pip * eff
  pr_susie <- apply(prof, 1, participation_ratio)
  susie_any <- rowSums(in_cs) > 0

  ## SuSiE selects CAUSAL variants, not trait-specific ones, so its FDP has to
  ## be scored against the causal class. Scoring it against `spec` would be
  ## marking it wrong for succeeding at its own task.
  sel <- which(susie_any)
  causal <- c(pleio, spec)
  fdp_v <- if (length(sel)) mean(!(sel %in% causal)) else NA_real_
  tagged <- ld_partners_of(causal, m)
  fdp_l <- if (length(sel)) mean(!(sel %in% tagged)) else NA_real_

  rows[[length(rows) + 1]] <- data.frame(
    seed = seed,
    pr_raw_pl_sp = auc_pair(pr_raw, pleio, spec),
    pr_susie_pl_sp = auc_pair(pr_susie, pleio, spec),
    pip_pl_sp = auc_pair(rowSums(pip), pleio, spec),
    pr_raw_pl_null = auc_pair(pr_raw, pleio, nul),
    pip_pl_null = auc_pair(rowSums(pip), pleio, nul),
    n_cs = sum(susie_any),
    fdp_variant = fdp_v,
    fdp_locus = fdp_l
  )
  cat(sprintf("seed %-9s done: %d variants in a credible set\n", seed, sum(susie_any)))
}

res <- do.call(rbind, rows)

cat("\n== pleiotropic vs trait-specific (AUC) ==\n")
print(round(res[, c("seed", "pr_raw_pl_sp", "pr_susie_pl_sp", "pip_pl_sp")], 3),
      row.names = FALSE)
cat(sprintf("\nmean:  PR(z) raw %.3f   PR(susie profile) %.3f   sum PIP %.3f\n",
            mean(res$pr_raw_pl_sp), mean(res$pr_susie_pl_sp), mean(res$pip_pl_sp)))

cat("\n== detection, pleiotropic vs null (AUC) ==\n")
cat(sprintf("PR(z) raw %.3f   sum PIP %.3f\n",
            mean(res$pr_raw_pl_null), mean(res$pip_pl_null)))

cat("\n== variant resolution: SuSiE credible sets, scored against CAUSAL ==\n")
print(round(res[, c("seed", "n_cs", "fdp_variant", "fdp_locus")], 3), row.names = FALSE)
cat(sprintf("\nmean FDP: variant-level %.3f, locus-level %.3f\n",
            mean(res$fdp_variant, na.rm = TRUE), mean(res$fdp_locus, na.rm = TRUE)))
cat("\nNot comparable to the rotation arm's 0.80/0.03: that arm selects for\n",
    "trait-SPECIFICITY, this one for causality. Different targets, different FDPs.\n", sep = "")
