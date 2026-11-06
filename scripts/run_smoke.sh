#!/usr/bin/env bash
# Tiny end-to-end run with the FULL architectures (integration check, not results).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
EXP="${1:-distinct_pairs}"
python -m decompositionality.cli smoke --experiment "$EXP" --output-root "${OUTPUT_ROOT:-results}" "${@:2}"
