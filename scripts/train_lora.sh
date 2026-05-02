#!/usr/bin/env bash
# train_lora.sh — convenience wrapper: calls train.sh with TRAIN_TYPE=use_lora
#
# Usage:
#   bash scripts/train_lora.sh <GPU> <MODEL> <DATASET> <PATCH_TYPE> <PATCH_LOC> <ATTACK_TYPE> <NAME> [PR] [EPOCH]
#
# Identical to:
#   bash scripts/train.sh <GPU> <MODEL> use_lora <DATASET> <PATCH_TYPE> <PATCH_LOC> <ATTACK_TYPE> <NAME> [PR] [EPOCH]
#
# All env-var overrides from train.sh are supported (LR, LOSS, DS_CONFIG, etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec bash "${SCRIPT_DIR}/train.sh" \
    "$1" "$2" use_lora "$3" "$4" "$5" "$6" "$7" "${8:-0.5}" "${9:-2}"
