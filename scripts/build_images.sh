#!/usr/bin/env bash
# Build all three benchmark Docker images.
# Run from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_ROOT="$REPO_ROOT/benchmark"

cd "$BENCH_ROOT"

echo "[build] clinical-agent:latest"
docker build -t clinical-agent:latest -f agent/Dockerfile .

echo "[build] zero-shot-baseline:latest"
docker build -t zero-shot-baseline:latest -f baselines/zero_shot/Dockerfile .

echo "[build] no-tools-baseline:latest"
docker build -t no-tools-baseline:latest -f baselines/no_tools/Dockerfile .

echo "[build] done."
docker images | grep -E "(clinical-agent|zero-shot-baseline|no-tools-baseline)"
