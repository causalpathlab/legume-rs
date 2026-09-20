# Simulation harnesses for the group model's gene-wise dispersion prior.
#
#   make sim-recovery      recover a planted dispersion trend end to end
#   make sim-calibration   permutation null rate on a confounded null
#
# Both write under $(OUT); numbers are printed, not stored in the repo.

BIN ?= ../target/release/cocoa
OUT ?= /tmp/cocoa-sim
SIM_ARGS ?= -r 1000 -c 6000 -a 0 -n 2 --n-samples-per-exposure 8 --n-covariates 2 \
            --pve-covar-exposure 0.3 --gene-mean-sd 1 \
            --indv-dispersion-trend 0.5,-0.5 --indv-dispersion-sd 0.5 --rseed 5
N_PERM ?= 100

.PHONY: sim-recovery sim-calibration

$(OUT):
	mkdir -p $(OUT)

# No individual-level covariate effect here: the group model has no term for
# it, so it would land in phi and blur the planted trend.
sim-recovery: | $(OUT)
	$(BIN) simulate-one $(SIM_ARGS) --pve-covar-gene 0 -o $(OUT)/recov
	gunzip -c $(OUT)/recov.samples.gz | sed 's/.*/0/' > $(OUT)/recov.topic.txt
	$(BIN) diff $(OUT)/recov.zarr.zip -i $(OUT)/recov.samples.gz \
	    -e $(OUT)/recov.exposures.gz -t $(OUT)/recov.topic.txt -o $(OUT)/recov.diff
	Rscript -e 'suppressMessages(library(arrow)); \
	  a <- as.data.frame(read_parquet("$(OUT)/recov.diff.contrast.parquet")); \
	  t <- read.table("$(OUT)/recov.dispersion.tsv.gz", header = TRUE); \
	  t <- t[match(as.integer(a$$gene), t$$gene), ]; \
	  cat(sprintf("pearson(log phi_hat, log phi_true) = %.3f  rmse = %.3f\n", \
	    cor(log(a$$dispersion), log(t$$phi)), sqrt(mean((log(a$$dispersion) - log(t$$phi))^2))))'

# Confounded null (individual covariate drives both exposure and expression);
# the fraction of genes below 0.05 should sit near the nominal rate.
sim-calibration: | $(OUT)
	$(BIN) simulate-one $(SIM_ARGS) --pve-covar-gene 0.3 -o $(OUT)/calib
	gunzip -c $(OUT)/calib.samples.gz | sed 's/.*/0/' > $(OUT)/calib.topic.txt
	$(BIN) diff $(OUT)/calib.zarr.zip -i $(OUT)/calib.samples.gz \
	    -e $(OUT)/calib.exposures.gz -t $(OUT)/calib.topic.txt \
	    --n-permutations $(N_PERM) -o $(OUT)/calib.diff
	Rscript -e 'suppressMessages(library(arrow)); \
	  p <- as.data.frame(read_parquet("$(OUT)/calib.diff.perm.parquet")); \
	  cat(sprintf("genes with p < 0.05: %.3f  p < 0.01: %.3f\n", \
	    mean(p$$pvalue < 0.05), mean(p$$pvalue < 0.01)))'
