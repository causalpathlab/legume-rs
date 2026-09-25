# Simulation harness.
#
#   make sim-calibration   permutation null rate on a confounded null
#
# Writes under $(OUT); numbers are printed, not stored in the repo.
# Needs the release binary and R with the arrow package.

BIN ?= ../target/release/cocoa
OUT ?= /tmp/cocoa-sim
SIM_ARGS ?= -r 1000 -c 6000 -a 0 -n 2 --n-samples-per-exposure 8 --n-covariates 2 \
            --pve-covar-exposure 0.3 --gene-mean-sd 1 \
            --indv-dispersion-trend 0.5,-0.5 --indv-dispersion-sd 0.5 --rseed 5
N_PERM ?= 100
R_PRELUDE = suppressMessages(library(arrow))

.PHONY: sim-calibration

$(OUT):
	mkdir -p $(OUT)

# Simulate, fabricate a single-topic file, run diff. Per-target variables
# supply the check's settings.
$(OUT)/%.diff.contrast.parquet: | $(OUT)
	$(BIN) simulate-one $(SIM_ARGS) $(SIM_EXTRA) -o $(OUT)/$*
	gunzip -c $(OUT)/$*.samples.gz | sed 's/.*/0/' > $(OUT)/$*.topic.txt
	$(BIN) diff $(OUT)/$*.zarr.zip -i $(OUT)/$*.samples.gz -e $(OUT)/$*.exposures.gz \
	    -t $(OUT)/$*.topic.txt $(DIFF_EXTRA) -o $(OUT)/$*.diff

# Confounded null (the individual covariate drives both exposure and
# expression); the fraction of genes below 0.05 should sit near the nominal rate.
sim-calibration: SIM_EXTRA = --pve-covar-gene 0.3
sim-calibration: DIFF_EXTRA = --n-permutations $(N_PERM)
sim-calibration: $(OUT)/calib.diff.contrast.parquet
	Rscript -e '$(R_PRELUDE); \
	  p <- as.data.frame(read_parquet("$(OUT)/calib.diff.perm.parquet")); \
	  cat(sprintf("genes with p < 0.05: %.3f  p < 0.01: %.3f\n", \
	    mean(p$$pvalue < 0.05), mean(p$$pvalue < 0.01)))'
