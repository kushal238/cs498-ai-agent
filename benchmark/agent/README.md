# Agent Implementation README

This directory contains the Docker entrypoint and runtime dependencies for the
ClinicalAgent submitted with the agent paper. The implementation itself is split
across `benchmark/agent/`, `benchmark/runner/`, and `benchmark/shared/`.

## Agent Source Layout

```text
benchmark/
  agent/
    agent_main.py          Container entrypoint
    Dockerfile             Runtime image definition
    requirements.txt       Runtime Python dependencies
  runner/
    agent.py               ClinicalAgent orchestration loop
    planner.py             Stage plan construction
    executor.py            Stage execution, retries, validation, fallbacks
    validator.py           JSON-schema validation for stage outputs
    state.py               Working memory, scratchpad, execution log
    llm_client.py          OpenAI structured-output client
    stage_transcription.py Stage 1 implementation
    stage_summarization.py Stage 2 implementation
    stage_diagnosis.py     Stage 3 implementation
    stage_medications.py   Stage 4 implementation
    stage_interactions.py  Stage 5 implementation
    stage_report.py        Stage 6 implementation
  shared/
    schemas/               Input and output schemas
    tools/                 PubMed, RxNorm, and OpenFDA helper clients
```

The container image copies only runtime code and shared utilities. Ground truths,
benchmark results, tests, and host-side scoring code stay outside the agent
container.

## Dependencies

Runtime dependencies are listed in:

```text
benchmark/agent/requirements.txt
```

Host-side benchmark and evaluation dependencies are listed separately in:

```text
benchmark/requirements.txt
```

Install host dependencies from the repository root:

```bash
python -m pip install -r benchmark/requirements.txt
```

The agent requires an OpenAI API key:

```bash
export OPENAI_API_KEY="<your-openai-api-key>"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="<your-openai-api-key>"
```

Optional external resource:

- `NCBI_API_KEY` increases PubMed rate limits for diagnosis-citation lookup.

## Build The Agent

From the repository root:

```bash
cd benchmark
docker build -t clinical-agent:latest -f agent/Dockerfile .
```

To build all evaluated systems, including baselines:

```bash
bash scripts/build_images.sh
```

## Run The Agent On The Benchmark

From the repository root:

```bash
python benchmark/harness/harness.py \
  --image clinical-agent:latest \
  --timeout 180 \
  --save-predictions
```

This runs every case under `benchmark/cases/`, passes each case input to the
Dockerized agent, and writes results under `benchmark/results/`.

Expected outputs:

- `run_<timestamp>_raw.csv`
- `run_<timestamp>_summary.csv`
- `predictions/run_<timestamp>_<case_id>_trial<n>.json`

## Reproduce The Paper Experiments

The paper comparison evaluates three systems:

- `clinical-agent:latest`
- `zero-shot-baseline:latest`
- `no-tools-baseline:latest`

Run the full experiment:

```bash
bash scripts/build_images.sh
bash scripts/run_full_benchmark.sh
```

Re-score saved predictions with the current metrics:

```bash
python scripts/rescore_predictions.py benchmark/results/final_<timestamp>
```

Generate comparison tables and figures:

```bash
python scripts/aggregate_results.py benchmark/results/final_<timestamp>
```

The included final run is stored at:

```text
benchmark/results/final_20260505_172124/
```

## Evaluation Scripts

- `benchmark/harness/harness.py` runs a selected Docker image on benchmark cases
  and writes raw/summary score CSVs.
- `scripts/run_full_benchmark.sh` runs the full three-system paper experiment.
- `scripts/rescore_predictions.py` re-scores saved prediction JSON files.
- `scripts/aggregate_results.py` generates comparison tables and the
  stage-breakdown figure.
- `scripts/sanity_bertscore.py` checks the clinical BERTScore setup.
- `scripts/test_bertscore_long.py` checks long-text BERTScore truncation.

## Metrics

- Transcription cleanup: ROUGE and clinical BERTScore
- Clinical summarization: ROUGE and clinical BERTScore
- Differential diagnosis: semantic Concept F1 and nDCG
- Medication normalization: Concept F1
- Drug interaction checking: Concept F1 and recommendation ROUGE-L
- Final SOAP report: ROUGE-L and clinical BERTScore by SOAP section

## Data And Resources

Inputs are in `benchmark/cases/`. Expected outputs are in
`benchmark/ground_truths/` and are used only by the host-side evaluation harness.
The agent container receives only the case input JSON.

Do not commit real patient data, API keys, private notes, account identifiers,
local runtime state, or machine-specific configuration.
