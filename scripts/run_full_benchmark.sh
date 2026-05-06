#!/usr/bin/env bash
# Run the full benchmark for all three systems.
#   - 3 systems: agent (ours), zero-shot baseline, no-tools baseline
#   - 10 cases per system
#   - 3 trials per case
# Total: 90 container invocations.
#
# Assumes images are already built (run scripts/build_images.sh first)
# and OPENAI_API_KEY is exported.
#
# Run from repo root:
#   source .venv/bin/activate
#   bash scripts/run_full_benchmark.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set" >&2
    exit 1
fi

export KMP_DUPLICATE_LIB_OK=TRUE

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="benchmark/results/final_${TIMESTAMP}"
mkdir -p "$OUT_BASE"

run_system() {
    local tag="$1"
    local image="$2"
    local outdir="$OUT_BASE/$tag"
    echo
    echo "=========================================="
    echo "[run_full_benchmark] System: $tag  (image: $image)"
    echo "=========================================="
    python benchmark/harness/harness.py \
        --image "$image" \
        --trials 3 \
        --output-dir "$outdir" \
        --save-predictions \
        --run-tag "$tag" \
        --timeout 180 \
        2>&1 | tee "$outdir.log"
}

run_system agent       clinical-agent:latest
run_system zs          zero-shot-baseline:latest
run_system nt          no-tools-baseline:latest

echo
echo "[run_full_benchmark] All runs complete. Results under: $OUT_BASE"
ls -la "$OUT_BASE"
