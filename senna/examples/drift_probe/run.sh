#!/usr/bin/env bash
# Drift-probe hold-out experiment: can `senna probe` pick up a topic the
# reference model was never trained on, without firing on a covariate shift?
#
#   usage:  ./run.sh [workdir]
#   env:    BIN=<dir with data-beans, data-beans-sim, senna>   EPOCHS=<n>
#
# Build first:  cargo build --release -p senna -p data-beans -p data-beans-sim
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
BIN="${BIN:-$ROOT/target/release}"
WORK="${1:-${WORK:-$PWD/drift_probe_run}}"
EPOCHS="${EPOCHS:-250}"
mkdir -p "$WORK"

echo "== 1. simulate K=6 topics; route the topic-5 cells to a second backend =="
# --holdout-topics guarantees the reference model never trains on topic 5.
"$BIN/data-beans-sim" topic -r 2000 -c 6000 -f 6 \
  --pve-topic 0.9 --depth 1000 --beta-scale 1.0 --batches 1 --rseed 42 \
  --holdout-topics 5 -o "$WORK/sim" --backend zarr

echo "== 2. split the reference file into reftrain / A / B =="
Rscript "$HERE/partition.R" "$WORK/sim" "$WORK"

echo "== 3. materialize the three reference backends =="
for tag in reftrain A B; do
  "$BIN/data-beans" subset-columns "$WORK/sim.zarr" \
    -i "$(cat "$WORK/idx_${tag}.txt")" -o "$WORK/${tag}.zarr"
done

echo "== 4. train the K=5 reference (topics 0-4 only) =="
"$BIN/senna" masked-topic "$WORK/reftrain.zarr" -t 5 -i "$EPOCHS" -o "$WORK/refmodel"

echo "== 5. score the held-out batches (emits {out}.predictive.parquet) =="
"$BIN/senna" predict --model "$WORK/refmodel" --out "$WORK/pred_A" "$WORK/A.zarr"
"$BIN/senna" predict --model "$WORK/refmodel" --out "$WORK/pred_B" "$WORK/B.zarr"
"$BIN/senna" predict --model "$WORK/refmodel" --out "$WORK/pred_C" "$WORK/sim.holdout.zarr"

echo "== 6. probe verdicts + counterfactual axes (A calibrates the null) =="
for q in "A:$WORK/A.zarr" "B:$WORK/B.zarr" "C:$WORK/sim.holdout.zarr"; do
  "$BIN/senna" -v probe --model "$WORK/refmodel" --calibration "$WORK/A.zarr" \
    "${q#*:}" -o "$WORK/probe_${q%%:*}" --influence 2>&1 |
    grep -E "counterfactual:|probe verdict:" || true
done

echo "== 7. analysis + figure =="
Rscript "$HERE/analyze.R" "$WORK"

echo
echo "outputs in $WORK  (probe_*.probe.json, probe_result.pdf/png)"
