# CS498 Clinical Workflow AI Benchmark

This repo benchmarks an agentic AI system that takes raw patient dialogue and chart notes and produces a SOAP clinical report. The agent runs inside a Docker container and goes through six sequential stages — transcription cleanup, clinical summarization, differential diagnosis, medication normalization, drug interaction checking, and final report generation. A host-side harness scores the output against ground truth without the agent ever seeing the answer key.

The benchmark is modeled after SWE-bench. We run three comparison agents (baselines) against the main tool-augmented pipeline to isolate what's actually doing the heavy lifting.

---

## Setup

**Clone and install dependencies on the host (required for the harness and tests):**

```powershell
pip install -r benchmark/requirements.txt
```

**Set your OpenAI API key** (all agents use GPT-4o):

```powershell
# PowerShell
$env:OPENAI_API_KEY="sk-..."

# bash/zsh
export OPENAI_API_KEY="sk-..."
```

---

## Docker Images

There are four Docker images — one for each agent. All use the same build context (`benchmark/`).

**Main agent** (full pipeline + RxNorm, PubMed, OpenFDA):
```powershell
docker build -t clinical-agent:latest -f benchmark/agent/Dockerfile benchmark/
```

**Zero-shot baseline** (single GPT-4o prompt, no pipeline, no tools):
```powershell
docker build -t clinical-zero-shot:latest -f benchmark/baselines/zero_shot/Dockerfile benchmark/
```

**No-tools baseline** (same 6-stage pipeline, LLM only — no API calls):
```powershell
docker build -t clinical-no-tools:latest -f benchmark/baselines/no_tools/Dockerfile benchmark/
```

---

## Running the Harness

The harness is the main entry point. It finds all cases in `benchmark/cases/`, runs the agent container on each one, scores against `benchmark/ground_truths/`, prints a results table, and writes CSVs to `benchmark/results/`.

**Basic run (main agent, all cases):**
```powershell
python benchmark/harness/harness.py
```

**Run with a specific image:**
```powershell
python benchmark/harness/harness.py --image clinical-zero-shot:latest
```

**Run 3 trials per case and save raw predictions:**
```powershell
python benchmark/harness/harness.py --trials 3 --save-predictions
```

**Full example — comparing all three agents:**
```powershell
# Main agent
python benchmark/harness/harness.py --image clinical-agent:latest --trials 3 --output-dir benchmark/results --save-predictions --timeout 300

# Zero-shot
python benchmark/harness/harness.py --image clinical-zero-shot:latest --trials 3 --output-dir benchmark/results --timeout 300

# No-tools
python benchmark/harness/harness.py --image clinical-no-tools:latest --trials 3 --output-dir benchmark/results --timeout 300
```

Results land in timestamped CSVs in `benchmark/results/`. Each run produces two files:
- `run_YYYYMMDD_HHMMSS_raw.csv` — one row per (case, trial, stage, metric)
- `run_YYYYMMDD_HHMMSS_summary.csv` — mean and stddev across trials

### All harness flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--image` | `clinical-agent:latest` | Which Docker image to run |
| `--cases-dir` | `benchmark/cases/` | Where to look for case folders |
| `--trials` | `1` | How many times to run each case (scores are averaged) |
| `--output-dir` | `benchmark/results/` | Where to write CSV results |
| `--save-predictions` | off | Saves each agent's raw JSON output to `output-dir/predictions/` — useful for debugging |
| `--timeout` | `120` | Seconds to wait per container before giving up. Increase to `300` if hitting timeouts (each case makes several API calls) |
| `--build` | off | Rebuilds the Docker image before running |
| `--network` | `bridge` | Docker network mode. Use `none` for locked-down final evaluation (blocks all outbound traffic — note this will break the tool-augmented agent's API calls) |

---

## The Three Baselines

All baselines use the same GPT-4o model. The only variable is the architecture.

### 1. Zero-shot (`clinical-zero-shot:latest`)
Sends a single prompt with all patient data and asks GPT-4o to produce all six pipeline outputs in one shot. No stages, no tools. Tests whether pipeline decomposition adds anything at all.

### 2. No-tools pipeline (`clinical-no-tools:latest`)
Runs the same six stages as the main agent but skips every external API call. Stages 3, 4, and 5 use GPT-4o's internal medical knowledge instead of PubMed, RxNorm, and OpenFDA. All `rxnorm_id` and `pmid` fields come back as `null`. Tests whether the grounded tool calls (not just the pipeline structure) are what's driving performance.

### 3. Main agent (`clinical-agent:latest`)
The full system. Six stages with real API calls to PubMed for DDx citations, NIH RxNav for medication normalization, and OpenFDA for drug interaction checking.

---

## Running Tests

Unit tests (no network, no Docker required):
```powershell
pytest benchmark/tests/test_harness.py benchmark/tests/test_baselines.py benchmark/tests/test_scoring.py -v
```

Integration tests (hit real APIs — needs internet):
```powershell
pytest benchmark/tests/test_tools.py -m integration -v
```

All tests:
```powershell
pytest benchmark/tests/ -v
```

---

## Adding a New Benchmark Case

1. Copy the template folder:
   ```powershell
   cp -r benchmark/cases/case_01_template benchmark/cases/case_XX
   ```

2. Edit `benchmark/cases/case_XX/input.json` with the patient data. The `case_id` field must match the folder name.

3. Edit `benchmark/cases/case_XX/metadata.json` with the case title, description, and difficulty.

4. Create `benchmark/ground_truths/case_XX.json` with the expected outputs for all six stages. Use the existing ground truth files (e.g. `case_02.json`) as a reference for the format.

5. Validate the input against the schema:
   ```powershell
   python -c "
   import json, jsonschema, pathlib
   schema = json.loads(pathlib.Path('benchmark/shared/schemas/input_schema.json').read_text())
   data = json.loads(pathlib.Path('benchmark/cases/case_XX/input.json').read_text())
   jsonschema.validate(data, schema)
   print('Valid')
   "
   ```

6. Run the harness on just the new case to check it loads cleanly:
   ```powershell
   python benchmark/harness/harness.py --cases-dir benchmark/cases/case_XX --timeout 300
   ```

---

## Folder Structure

```
benchmark/
├── cases/                    ← one folder per benchmark case (input + metadata)
├── ground_truths/            ← answer keys, host-side only
├── shared/
│   ├── schemas/              ← JSON Schema files for input, ground truth, metadata
│   ├── tools/                ← PubMed, RxNorm, OpenFDA API wrappers
│   └── scoring/              ← ROUGE, Concept F1, nDCG
├── runner/
│   ├── langgraph_runner.py   ← main pipeline (6 LangGraph nodes)
│   └── llm.py                ← shared GPT-4o client
├── agent/
│   ├── agent_main.py         ← container entrypoint for main agent
│   └── Dockerfile
├── baselines/
│   ├── zero_shot/            ← single-prompt baseline
│   └── no_tools/             ← pipeline without API calls
├── harness/
│   └── harness.py            ← host-side orchestrator
├── tests/                    ← unit and integration tests
├── results/                  ← CSV outputs from harness runs (gitignored except .gitkeep)
└── requirements.txt
```

---

## Environment Variables

| Variable | Required | Where it's used |
|----------|----------|-----------------|
| `OPENAI_API_KEY` | Yes | All agents — passed from host into container by the harness |
| `NCBI_API_KEY` | No | PubMed — increases rate limit from 3 to 10 req/s. Get one at ncbi.nlm.nih.gov/account |
| `BENCHMARK_ROOT` | No | Set automatically to `/app` inside containers. Override on host if running outside the repo root |
