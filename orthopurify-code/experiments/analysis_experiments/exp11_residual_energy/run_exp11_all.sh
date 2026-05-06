#!/bin/bash
# exp11: residual energy and weight-similarity analyses.
# CPU-only; run from anywhere inside orthopurify-code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

bash "$SCRIPT_DIR/run_exp11_residual.sh"
bash "$SCRIPT_DIR/run_exp11_weight_sim.sh"
