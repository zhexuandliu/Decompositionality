#!/usr/bin/env bash
# Full benchmark run (train + evaluate) of ONE model: scripts/run_full_model.sh <experiment> <model_id>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m decompositionality.cli train --experiment "$1" --model "$2" --output-root "${OUTPUT_ROOT:-results}" "${@:3}"
