#!/usr/bin/env Rscript
## Split the reference backend (sim.zarr, topics 0..4) into reftrain / A / B.
##
## `data-beans-sim topic --holdout-topics 5` already routed the topic-5 cells into
## sim.holdout.zarr (used directly as batch C, the novelty). Here we only carve the
## reference file:
##
##   reftrain : 80% of reference cells   -> train the K=5 reference model
##   A        : the remaining 20%        -> in-distribution control (must NOT flag)
##   B        : covariate shift          -> reference cells resampled to enrich topic 2
##
## Output files hold comma-joined COLUMN POSITIONS into sim.zarr (0-indexed), the form
## `data-beans subset-columns -i` consumes. sim.zarr's column order is the ascending
## global id of reference-eligible cells, so position p <-> that cell.
suppressMessages(library(arrow))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("usage: partition.R <sim-prefix> <outdir>")
sim_prefix <- args[1]
outdir <- args[2]

HELDOUT_TOPIC <- 5L # 0-indexed
ENRICH_TOPIC <- 2L
set.seed(0)

tbl <- as.data.frame(read_parquet(paste0(sim_prefix, ".prop.parquet")))
name_col <- if ("cell" %in% names(tbl)) "cell" else names(tbl)[1]
theta <- as.matrix(tbl[, setdiff(names(tbl), name_col), drop = FALSE])
dom <- max.col(theta, ties.method = "first") - 1L # 0-indexed dominant topic

## sim.zarr column order = ascending global id of reference-eligible cells
ref_global <- sort(which(dom < HELDOUT_TOPIC) - 1L)
dom_pos <- dom[ref_global + 1L] # dominant topic at each sim.zarr position
npos <- length(ref_global)

perm <- sample.int(npos) - 1L # 0-indexed positions
cut <- floor(0.8 * npos)
reftrain <- sort(perm[seq_len(cut)])
holdout <- perm[(cut + 1L):npos]
A <- sort(holdout)

## B: covariate shift -> enrich topic 2 to ~50% among the held-out positions
t2 <- holdout[dom_pos[holdout + 1L] == ENRICH_TOPIC]
other <- holdout[dom_pos[holdout + 1L] != ENRICH_TOPIC]
B <- sort(c(t2, sample(other, min(length(other), length(t2)))))

write_idx <- function(tag, ix) {
  writeLines(paste(ix, collapse = ","), file.path(outdir, paste0("idx_", tag, ".txt")))
}
write_idx("reftrain", reftrain)
write_idx("A", A)
write_idx("B", B)

cat(sprintf("cells=%d  topics=%d  held-out topic=%d\n", nrow(theta), ncol(theta), HELDOUT_TOPIC))
cat(sprintf("reference cells=%d  novel cells=%d\n", npos, nrow(theta) - npos))
cat(sprintf("reftrain=%d  A=%d  B=%d\n", length(reftrain), length(A), length(B)))
cat(sprintf(
  "topic-%d share:  A=%.3f  B=%.3f  (B is the covariate shift)\n",
  ENRICH_TOPIC,
  mean(dom_pos[A + 1L] == ENRICH_TOPIC),
  mean(dom_pos[B + 1L] == ENRICH_TOPIC)
))
