#!/usr/bin/env bash
# Figures/tables for both experiments, then the cross-experiment comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ROOT="${OUTPUT_ROOT:-results}"
for e in distinct_pairs all_pairs; do
  [ -d "$ROOT/$e" ] && python -m decompositionality.cli plot --experiment "$e" --output-root "$ROOT"
done
python -m decompositionality.cli compare --results-root "$ROOT"
