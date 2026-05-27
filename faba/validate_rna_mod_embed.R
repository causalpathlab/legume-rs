#!/usr/bin/env Rscript
# Recovery validation for `faba rna-mod-embed`.
#
# Compares rmodem's CP-factored outputs against `data-beans-sim faba`'s
# ground truth:
#
#   1. z_g recovery — Procrustes-align rmodem's z to sim's z_true on
#      measured genes, then per-program Pearson.
#   2. Q vs A recovery — per-modality, per-program signature strength;
#      after the same Procrustes alignment, compare ||Q[k, m, :]|| to
#      |A_true[m, k]|.
#   3. Measured-mask alignment — does rmodem's measured mask match the
#      simulator's substrate_mask?
#   4. Manifest smoke — every path declared in `*.faba.json` exists.
#
# Usage:
#   Rscript validate_rna_mod_embed.R <sim_prefix> <run_prefix>
#
# Example:
#   Rscript validate_rna_mod_embed.R /tmp/rmodem_test/sim /tmp/rmodem_test/run

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: validate_rna_mod_embed.R <sim_prefix> <run_prefix>")
}
sim_prefix <- args[1]
run_prefix <- args[2]

cat(sprintf("# faba rna-mod-embed validation\n"))
cat(sprintf("# sim:  %s\n", sim_prefix))
cat(sprintf("# run:  %s\n", run_prefix))

# ---- helpers --------------------------------------------------------

read_named <- function(path, name_col) {
  df <- as.data.frame(arrow::read_parquet(path))
  stopifnot(name_col %in% names(df))
  names_v <- df[[name_col]]
  mat <- as.matrix(df[, setdiff(names(df), name_col), drop = FALSE])
  rownames(mat) <- names_v
  storage.mode(mat) <- "double"
  mat
}

procrustes_align <- function(X, Y) {
  # Returns X %*% R minimising ||X R - Y||_F (rotation, no scale/shift).
  # Both X, Y must be [n, k] with the same n.
  stopifnot(nrow(X) == nrow(Y))
  S <- svd(t(X) %*% Y)
  R <- S$u %*% t(S$v)
  list(aligned = X %*% R, rotation = R)
}

# ---- 1. Manifest smoke ---------------------------------------------

manifest_path <- paste0(run_prefix, ".faba.json")
cat(sprintf("\n[1] Manifest smoke (%s)\n", manifest_path))
mf <- jsonlite::fromJSON(manifest_path)
required <- c("gene_embedding", "gene_program_loadings", "program_signatures",
              "modality_axis", "measured_mask")
missing_paths <- character()
for (k in required) {
  p <- mf[[k]]
  if (is.null(p) || !file.exists(p)) {
    missing_paths <- c(missing_paths, sprintf("%s -> %s", k, p))
  }
}
if (length(missing_paths) > 0) {
  cat("  FAIL — missing artefacts:\n")
  cat(paste("   -", missing_paths, collapse = "\n"), "\n")
} else {
  cat(sprintf("  OK — kind='%s', H=%d, K=%d, M=%d\n",
              mf$kind, mf$embedding_dim, mf$n_programs, mf$n_modalities))
}

# ---- Load ground truth + predictions -------------------------------

z_true_raw  <- read_named(paste0(sim_prefix, ".gene_program_loadings.parquet"), "gene")
A_true_raw  <- read_named(paste0(sim_prefix, ".program_writer_editor.parquet"), "modality")
phi_raw     <- read_named(paste0(sim_prefix, ".substrate_mask.parquet"), "gene")

rho   <- read_named(paste0(run_prefix, ".gene_embedding.parquet"), "gene")
z_hat <- read_named(paste0(run_prefix, ".gene_program_loadings.parquet"), "gene")
meas  <- read_named(paste0(run_prefix, ".measured_mask.parquet"), "gene")

# Align gene axis (intersection by name).
common_genes <- intersect(rownames(z_hat), rownames(z_true_raw))
z_true <- z_true_raw[common_genes, , drop = FALSE]
z_hat  <- z_hat[common_genes, , drop = FALSE]
phi    <- phi_raw[common_genes, , drop = FALSE]
meas   <- meas[common_genes, , drop = FALSE]

cat(sprintf("\n  ground-truth: G=%d, K_topic=%d, M=%d\n",
            nrow(z_true), ncol(z_true), nrow(A_true_raw)))
cat(sprintf("  predictions:  G=%d, K=%d, H=%d\n",
            nrow(z_hat), ncol(z_hat), ncol(rho)))

# ---- 2. z recovery via Procrustes + signed-permutation match -------

cat("\n[2] z recovery\n")
# z is identifiable up to a *signed permutation* of program axes (you
# can permute programs across (z, Q) and negate any program axis
# without changing predictions). Procrustes gives an orthogonal
# rotation which captures permutation but only handles per-axis sign
# flips collectively (det = ±1). So we report both metrics: Procrustes
# (lower bound) and signed-permutation Hungarian (matches the true
# identifiability class).
#
# Restrict to genes with ≥1 measured modifier modality — that's where
# the modifier-comp loss actually feeds z gradient.
any_measured <- rowSums(phi) >= 1
z_t <- z_true[any_measured, , drop = FALSE]
z_p <- z_hat[any_measured,  , drop = FALSE]
K <- min(ncol(z_t), ncol(z_p))
z_t <- z_t[, 1:K, drop = FALSE]
z_p <- z_p[, 1:K, drop = FALSE]

# (a) Procrustes baseline.
al  <- procrustes_align(z_p, z_t)
z_p_al <- al$aligned
proc_cor <- sapply(seq_len(K), function(k) cor(z_p_al[, k], z_t[, k]))
cat(sprintf("  Procrustes-aligned per-program Pearson: mean=%.3f, min=%.3f, max=%.3f\n",
            mean(proc_cor), min(proc_cor), max(proc_cor)))

# (b) Signed-permutation match (greedy Hungarian on |cor|).
cor_mat <- matrix(0.0, K, K)
for (i in seq_len(K)) for (j in seq_len(K)) {
  s_i <- sd(z_p[, i]); s_j <- sd(z_t[, j])
  cor_mat[i, j] <- if (s_i > 0 && s_j > 0) cor(z_p[, i], z_t[, j]) else 0
}
# Greedy: pick the largest |r| cell repeatedly; remove that row/col.
used_p <- logical(K); used_t <- logical(K)
matched <- data.frame(prog_pred = integer(), prog_true = integer(),
                      sign = integer(), r = double())
for (step in seq_len(K)) {
  mask <- outer(!used_p, !used_t, "&")
  scores <- abs(cor_mat)
  scores[!mask] <- -Inf
  if (all(is.infinite(scores))) break
  flat <- which.max(scores)
  i <- ((flat - 1L) %% K) + 1L
  j <- ((flat - 1L) %/% K) + 1L
  used_p[i] <- TRUE; used_t[j] <- TRUE
  r <- cor_mat[i, j]
  matched <- rbind(matched, data.frame(prog_pred = i, prog_true = j,
                                       sign = sign(r), r = r))
}
matched <- matched[order(matched$prog_true), ]
cat(sprintf("  Signed-permutation match (|cor|): mean=%.3f, min=%.3f, max=%.3f\n",
            mean(abs(matched$r)), min(abs(matched$r)), max(abs(matched$r))))
cat("  matched programs (true → pred [sign]):\n")
for (row in seq_len(nrow(matched))) {
  cat(sprintf("    true %d ← pred %d [%+d]   r=%+.3f\n",
              matched$prog_true[row], matched$prog_pred[row],
              matched$sign[row], matched$r[row]))
}

# ---- 3. Q recovery ---------------------------------------------------

cat("\n[3] Q (program signatures) recovery\n")
# Read long-format program_signatures and reshape to [K, M, H].
ps <- as.data.frame(arrow::read_parquet(paste0(run_prefix, ".program_signatures.parquet")))
# Row column is "program_modality" = "program_<k>/<modality_name>".
stopifnot("program_modality" %in% names(ps))
pm <- strsplit(ps$program_modality, "/", fixed = TRUE)
prog <- vapply(pm, `[`, "", 1L)
modn <- vapply(pm, `[`, "", 2L)
prog_levels <- unique(prog); modn_levels <- unique(modn)
H <- ncol(ps) - 1L
Q <- array(0.0, dim = c(length(prog_levels), length(modn_levels), H),
           dimnames = list(prog_levels, modn_levels, sprintf("dim_%d", 0:(H - 1L))))
for (i in seq_len(nrow(ps))) {
  ki <- match(prog[i], prog_levels)
  mi <- match(modn[i], modn_levels)
  Q[ki, mi, ] <- as.numeric(ps[i, setdiff(names(ps), "program_modality")])
}

# Per-(k, m) signature strength = ||Q[k, m, :]||₂.
S_pred <- apply(Q, c(1, 2), function(v) sqrt(sum(v * v)))
rownames(S_pred) <- prog_levels
colnames(S_pred) <- modn_levels
cat("  ||Q[k, m, :]||₂ predicted (rows=program, cols=modality):\n")
print(round(S_pred, 3))

# A_true is [M × K_topic]. Make it [K × M] for direct comparison.
# After Procrustes rotation R = al$rotation (K_pred × K_true), the
# predicted programs are reindexed by R. Apply R to A_true:
#   A_aligned[k_pred, m] = sum_{k_true} R[k_pred, k_true] * A_true[m, k_true]
A_t <- A_true_raw[, 1:K, drop = FALSE]      # [M × K]
A_t_for_pred <- t(A_t %*% t(al$rotation))   # [K × M]
S_true <- abs(A_t_for_pred[1:K, , drop = FALSE])
# Restrict to modifier modalities (drop "count" which has no z·Q
# contribution in the sim).
mod_keep <- intersect(modn_levels, c("m6A", "A2I", "pA"))
sim_mod_match <- intersect(rownames(A_true_raw), mod_keep)
if (length(sim_mod_match) > 0L) {
  cat(sprintf("\n  Signed correlation ||Q[k, m, :]|| vs aligned |A_true[m, k]|:\n"))
  for (m in sim_mod_match) {
    pred_col <- which(modn_levels == m)
    true_row <- which(rownames(A_true_raw) == m)
    if (length(pred_col) == 0L || length(true_row) == 0L) next
    a_v <- abs(as.numeric(A_t_for_pred[, true_row]))[1:K]
    p_v <- S_pred[, pred_col][1:K]
    r <- if (sd(p_v) > 0 && sd(a_v) > 0) cor(p_v, a_v) else NA_real_
    cat(sprintf("    modality %-5s  r=%.3f   (pred=%s, true=%s)\n",
                m, r, paste(round(p_v, 2), collapse=","),
                paste(round(a_v, 2), collapse=",")))
  }
}

# ---- 4. Measured-mask alignment --------------------------------------

cat("\n[4] Measured-mask alignment\n")
# rmodem's measured[m] = "any modifier-comp row for (g, m)".
# sim's substrate_mask[m] = "φ_{g,m} = 1".
shared_mods <- intersect(colnames(meas), colnames(phi))
for (m in shared_mods) {
  if (m == "count") next
  agreement <- mean(meas[, m] == phi[, m])
  cat(sprintf("  modality %-5s  agreement=%.1f%%  (rmodem: %d genes; sim φ: %d genes)\n",
              m, 100 * agreement, sum(meas[, m]), sum(phi[, m])))
}

cat("\n# done\n")
