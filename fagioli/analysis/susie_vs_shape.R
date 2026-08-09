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
  keep <- !is.na(s); s <- s[keep]; lab <- lab[keep]
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

## susieR warns that Xcorr "is not symmetric" on every fit. The values are
## bit-exactly symmetric -- the culprit is dimnames: read_mat drops the snp_id
## column, so colnames are set while rownames are NULL, and isSymmetric()
## compares attributes. Stripping them silences it at the source; averaging with
## the transpose (an earlier attempt) is a no-op and does not.
read_ld <- function(path) {
  r <- read_mat(path)
  dimnames(r) <- NULL
  r
}

## ---- per seed -------------------------------------------------------------

rows <- list()
for (seed in SEEDS) {
  sid <- format(seed, scientific = FALSE)
  z <- read_mat(file.path(EXPORT, sprintf("z_%s.parquet", sid)))
  cls <- as.integer(read_mat(file.path(EXPORT, sprintf("labels_%s.parquet", sid)))[, 1])
  m <- nrow(z); tt <- ncol(z)
  pleio <- which(cls == 1L); spec <- which(cls == 2L); nul <- which(cls == 0L)

  ## Arm 1: shape on raw rows. No LD read at all.
  pr_raw <- apply(z, 1, participation_ratio)

  ## Arm 2: per-trait SuSiE-RSS, one fit per (block, trait).
  pip <- matrix(0, m, tt)
  eff <- matrix(0, m, tt)
  in_cs <- logical(m)
  n_skip <- 0L
  for (b in seq_len(N_BLOCKS)) {
    idx <- ((b - 1) * SNPS_PER_BLOCK + 1):(b * SNPS_PER_BLOCK)
    R <- read_ld(file.path(EXPORT, sprintf("ld_%s_block%d.parquet", sid, b - 1)))
    for (t in seq_len(tt)) {
      fit <- tryCatch(
        susie_rss(z = z[idx, t], R = R, n = N_IND, L = 5, verbose = FALSE),
        error = function(e) NULL
      )
      ## A skipped fit leaves 150 rows of this trait at zero, which silently
      ## removes a coordinate from every profile in the block. Count them.
      if (is.null(fit)) { n_skip <- n_skip + 1; next }
      pip[idx, t] <- susieR::susie_get_pip(fit)
      eff[idx, t] <- coef(fit)[-1]
      cs <- susieR::susie_get_cs(fit, Xcorr = R)$cs
      if (length(cs)) in_cs[idx[unique(unlist(cs))]] <- TRUE
    }
  }

  ## A variant's trait profile under SuSiE is its posterior effect across
  ## traits. `coef.susie` returns colSums(alpha * mu), which ALREADY integrates
  ## over inclusion -- multiplying by susie_get_pip() again would apply the
  ## inclusion weight twice and artificially sharpen each row toward its
  ## highest-PIP trait, which is exactly what PR measures.
  prof <- eff
  pr_susie <- apply(prof, 1, participation_ratio)
  ## A variant SuSiE never included has an all-zero profile. PR = 0 there would
  ## rank it as MAXIMALLY trait-specific, scoring non-detection as evidence of
  ## specificity and pushing this arm's AUC down. Drop those instead.
  pr_susie[rowSums(abs(prof)) == 0] <- NA_real_
  susie_any <- in_cs

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
  cat(sprintf("seed %-9s done: %d variants in a credible set, %d fits skipped\n",
              sid, sum(susie_any), n_skip))
  stopifnot(n_skip == 0L)
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
cat("\nThis is a coverage property of credible sets, not FDR control -- susieR\n",
    "makes no FDR claim, so read it as the realized FDP on one fixture. It is\n",
    "also not comparable to the rotation arm's FDP*: that arm selects for\n",
    "trait-SPECIFICITY, this one for causality. Different targets.\n", sep = "")
