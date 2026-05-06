# CS498 Clinical Workflow Agent

This repository contains the complete agent implementation and benchmark package for
the clinical workflow agent paper. The agent takes synthetic patient dialogue and
chart notes, runs a six-stage clinical pipeline, and returns a SOAP-style report.
A host-side benchmark harness runs the agent in Docker and scores the output against
ground truth that never enters the container.

The package includes:

- Agent source code in `benchmark/runner/` and `benchmark/agent/`
- Baseline agents in `benchmark/baselines/`
- Synthetic benchmark cases and ground truths in `benchmark/cases/` and `benchmark/ground_truths/`
- Host-side scoring and evaluation code in `benchmark/harness/` and `benchmark/shared/scoring/`
- Experiment scripts in `scripts/`
- Final paper run outputs in `benchmark/results/final_20260505_172124/`

## Current Agent

The production container entrypoint is `benchmark/agent/agent_main.py`. It runs
`ClinicalAgent` from `benchmark/runner/agent.py`, which uses:

- Conditional planning in `benchmark/runner/planner.py`
- Per-stage execution, retries, validation, and fallbacks in `benchmark/runner/executor.py`
- Shared working memory and execution logs in `benchmark/runner/state.py`
- Stage implementations in `benchmark/runner/stage_*.py`
- OpenAI structured outputs via `benchmark/runner/llm_client.py`
- PubMed, RxNorm, and OpenFDA helper tools in `benchmark/shared/tools/`

The six evaluated stages are:

1. Transcription cleanup
2. Clinical summarization
3. Differential diagnosis
4. Medication normalization
5. Drug interaction checking
6. Final SOAP report generation

## Setup

Requirements:

- Python 3.11 recommended
- Docker
- OpenAI API key
- Bash-compatible shell for the scripts in `scripts/` (Git Bash, WSL, macOS, or Linux)

Install host-side dependencies:

```bash
python -m pip install -r benchmark/requirements.txt
```

Set your OpenAI API key before running agents or full benchmark experiments:

```bash
export OPENAI_API_KEY="<your-openai-api-key>"
```

PowerShell equivalent:

```powershell
$env:OPENAI_API_KEY="<your-openai-api-key>"
```

The agent Docker image uses the smaller dependency file
`benchmark/agent/requirements.txt`. Host-only scoring dependencies such as
BERTScore, sentence-transformers, and pytest stay in `benchmark/requirements.txt`.

## Build Docker Images

Build all evaluated systems from the repository root:

```bash
bash scripts/build_images.sh
```

This creates:

- `clinical-agent:latest` - full tool-augmented agent
- `zero-shot-baseline:latest` - one-shot GPT baseline
- `no-tools-baseline:latest` - six-stage pipeline without external tools

Manual build commands:

```bash
cd benchmark
docker build -t clinical-agent:latest -f agent/Dockerfile .
docker build -t zero-shot-baseline:latest -f baselines/zero_shot/Dockerfile .
docker build -t no-tools-baseline:latest -f baselines/no_tools/Dockerfile .
```

## Run The Agent

The benchmark harness is the easiest way to run the agent on one or more cases.
It pipes each `input.json` into the selected Docker image, captures the JSON
prediction, scores it, and writes CSV outputs.

Run the full agent on all cases:

```bash
python benchmark/harness/harness.py \
  --image clinical-agent:latest \
  --timeout 180 \
  --save-predictions
```

Run a single case:

```bash
python benchmark/harness/harness.py \
  --cases-dir benchmark/cases/case_02 \
  --image clinical-agent:latest \
  --timeout 180 \
  --save-predictions
```

Run a baseline:

```bash
python benchmark/harness/harness.py --image zero-shot-baseline:latest --timeout 180
python benchmark/harness/harness.py --image no-tools-baseline:latest --timeout 180
```

By default, outputs are written to `benchmark/results/`:

- `run_<timestamp>_raw.csv` - one row per case, trial, stage, and metric
- `run_<timestamp>_summary.csv` - mean and standard deviation across trials
- `predictions/*.json` - saved raw model outputs when `--save-predictions` is used

## Reproduce Experiments

The final paper run compares three systems across 10 cases with 3 trials per
case, for 90 total container invocations.

From the repository root:

```bash
bash scripts/build_images.sh
bash scripts/run_full_benchmark.sh
```

The full script writes a timestamped directory:

```text
benchmark/results/final_<YYYYMMDD_HHMMSS>/
```

To recompute scores from saved predictions:

```bash
python scripts/rescore_predictions.py benchmark/results/final_<YYYYMMDD_HHMMSS>
```

To regenerate the comparison table and stage-breakdown figure:

```bash
python scripts/aggregate_results.py benchmark/results/final_<YYYYMMDD_HHMMSS>
```

Expected generated files:

- `comparison_summary.csv`
- `comparison_summary.md`
- `stage_breakdown.png`
- `<system>/run_<system>_<timestamp>_raw.csv`
- `<system>/run_<system>_<timestamp>_summary.csv`
- `<system>/run_<system>_<timestamp>_rescored_raw.csv`
- `<system>/run_<system>_<timestamp>_rescored_summary.csv`
- `<system>/predictions/*.json`

## Final Results Included

The packaged final run is:

```text
benchmark/results/final_20260505_172124/
```

Important files:

- `comparison_summary.csv` - wide table used for paper numbers
- `comparison_summary.md` - readable summary table
- `stage_breakdown.png` - figure generated from final results
- `agent/`, `zs/`, `nt/` - per-system raw outputs, summaries, rescored summaries, and predictions

Headline result: the tool-augmented agent reaches perfect F1 on drug interaction
checking in the final run and is strongest on differential-diagnosis ranking
nDCG, while final SOAP report scores vary by section.

## Evaluation Metrics

- Transcription cleanup: ROUGE and clinical BERTScore
- Clinical summarization: ROUGE and clinical BERTScore
- Differential diagnosis: semantic Concept F1 and nDCG
- Medication normalization: Concept F1 over ingredients
- Drug interaction checking: Concept F1 over drug pairs and recommendation ROUGE-L
- Final SOAP report generation: ROUGE-L and clinical BERTScore per SOAP section

BERTScore uses `emilyalsentzer/Bio_ClinicalBERT` by default. It can be changed with:

```bash
export BERTSCORE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
export BERTSCORE_LAYER="9"
```

## Tests

Run unit tests without live external APIs:

```bash
pytest benchmark/tests/ -m "not integration" -v
```

Run integration tests that hit live PubMed, RxNorm, or OpenFDA endpoints:

```bash
pytest benchmark/tests/ -m integration -v
```

Run the lightweight BERTScore sanity checks:

```bash
python scripts/sanity_bertscore.py
python scripts/test_bertscore_long.py
```

The first BERTScore run downloads the selected Hugging Face model.

## Change Summary

This final package includes:

- Full ClinicalAgent source code and baseline implementations
- Host-side benchmark harness and scoring code
- Clinical BERTScore as a supplemental semantic metric for free-text stages
- Docker build script for all evaluated systems
- Full benchmark orchestration script for the 3-system, 10-case, 3-trial run
- Re-scoring script for saved predictions
- Aggregation script for final comparison tables and figures
- Final paper results under `benchmark/results/final_20260505_172124/`

## Data Policy

The benchmark cases in this repository are synthetic or de-identified. Do not
commit real patient data, private notes, API keys, or local runtime state.
