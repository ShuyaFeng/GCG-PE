#!/usr/bin/env bash
# Full pipeline driver. Run from the pipeline/ directory.
set -euo pipefail

CONFIG="${1:-configs/full.yaml}"
DEVICE="${2:-cuda}"

echo "=== Smoke test first ==="
python run.py --config configs/smoke.yaml --device "${DEVICE}"

echo "=== Full run: ${CONFIG} ==="
python run.py --config "${CONFIG}" --device "${DEVICE}" \
  --stages data train extract transfer validate features tables

echo "=== Tables written to runs/*/tables/ ==="
