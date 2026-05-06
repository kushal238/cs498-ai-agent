# Change Summary

## Final Agent Paper Package

This package prepares the repository for submission alongside the agent paper.

### Added

- Complete Dockerized ClinicalAgent implementation.
- Zero-shot and no-tools baseline agents.
- Host-side benchmark harness for isolated evaluation.
- Clinical BERTScore scoring for free-text stages.
- Full benchmark orchestration script for the three-system paper run.
- Re-scoring script for saved predictions.
- Aggregation script for comparison tables and stage-breakdown figure generation.
- Final paper run outputs under `benchmark/results/final_20260505_172124/`.

### Cleaned

- Removed stale pre-final benchmark outputs from `benchmark/results/`.
- Removed duplicate root-level `stage_breakdown.png`; the packaged figure now lives
  with the final run outputs.
- Added ignore rules for future generated benchmark outputs.
- Added a benchmark `.dockerignore` so Docker build contexts exclude cases,
  ground truths, tests, and results.

### Documentation

- Rewrote the top-level README with setup, dependencies, agent run commands,
  experiment reproduction steps, expected outputs, final results location, and
  evaluation metrics.
- Added `benchmark/agent/README.md` to document the agent implementation,
  runtime dependencies, run commands, reproduction steps, and evaluation scripts.
