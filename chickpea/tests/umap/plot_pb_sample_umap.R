#!/usr/bin/env Rscript
# UMAP of pb-sample embeddings from chickpea workflow (uwot + ggplot2).
#
#   cargo test -p chickpea -- dump_pb_embeds_for_umap
#   Rscript chickpea/tests/umap/plot_pb_sample_umap.R [in_dir] [out_png]

suppressPackageStartupMessages({
  library(uwot)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

# Locate this script's directory when run via Rscript.
script_dir <- tryCatch(
  {
    cmd <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", cmd[grep("^--file=", cmd)])
    dirname(normalizePath(f))
  },
  error = function(e) getwd()
)

in_dir <- if (length(args) >= 1) {
  args[[1]]
} else {
  file.path(script_dir, "out")
}

out_png <- if (length(args) >= 2) {
  args[[2]]
} else {
  file.path(in_dir, "pb_sample_umap.png")
}

emb_path <- file.path(in_dir, "pb_sample_embedding.tsv")
lab_path <- file.path(in_dir, "pb_sample_cluster.tsv")
if (!file.exists(emb_path) || !file.exists(lab_path)) {
  stop(
    "missing embedding TSVs under ", in_dir,
    "\nrun: cargo test -p chickpea -- dump_pb_embeds_for_umap"
  )
}

emb <- read.delim(emb_path, check.names = FALSE)
lab <- read.delim(lab_path, check.names = FALSE)
X <- as.matrix(emb[, setdiff(names(emb), "sample"), drop = FALSE])
storage.mode(X) <- "double"
y <- lab$cluster[match(emb$sample, lab$sample)]

set.seed(42)
Z <- uwot::umap(
  X,
  n_neighbors = 15,
  min_dist = 0.3,
  metric = "cosine",
  n_epochs = 200,
  verbose = FALSE
)

df <- data.frame(
  UMAP1 = Z[, 1],
  UMAP2 = Z[, 2],
  cluster = factor(y)
)

p <- ggplot(df, aes(UMAP1, UMAP2, color = cluster)) +
  geom_point(size = 2.2, alpha = 0.85) +
  coord_equal() +
  labs(
    title = "UMAP of pb-sample embeddings",
    subtitle = "gene FNE embeds x RNA (synthetic two-program data)",
    color = "cluster"
  ) +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank())

dir.create(dirname(out_png), recursive = TRUE, showWarnings = FALSE)
ggsave(out_png, p, width = 5.5, height = 4.5, dpi = 140)
message("wrote ", normalizePath(out_png))
