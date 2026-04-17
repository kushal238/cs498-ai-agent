# Clinical Workflow AI Benchmark

A SWE-bench-style benchmark for evaluating clinical AI agents. An agent receives a raw patient dialogue and chart notes, runs them through a six-stage pipeline, and returns a physician-ready SOAP report — all inside an isolated Docker container. The harness scores the output on the host without ever exposing the answer key to the agent.

**Agent runtime (current):** The container entrypoint (`agent/agent_main.py`) runs **`ClinicalAgent`** (`runner/agent.py`) — a small agentic loop with a **conditional plan** (`runner/planner.py`), **per-stage execution** with retries and JSON-schema validation (`runner/executor.py`, `runner/validator.py`), and **memory** for outputs, reasoning, and an execution log (`runner/state.py`). Each clinical stage lives in **`runner/stage_*.py`** and calls the **OpenAI SDK** via **`runner/llm_client.py`** (Pydantic structured outputs). The older **`langgraph_runner.py`** stub remains for tests / reference but is not the container’s primary path.

---

## Pipeline stages

| # | Stage | Input | Output | Metric |
|---|-------|-------|--------|--------|
| 1 | Transcription cleanup | Raw dialogue | Cleaned transcript | ROUGE |
| 2 | Clinical summarization | Transcript + chart notes | Clinical summary | ROUGE |
| 3 | Differential diagnosis | Summary | Ranked DDx list (PubMed-backed) | Concept F1 + nDCG |
| 4 | Medication normalization | Medication list | RxNorm-mapped medications | Concept F1 |
| 5 | Drug-drug interaction check | Normalized medications | Interaction list (OpenFDA) | Concept F1 |
| 6 | Final report generation | All prior outputs | SOAP-format report | ROUGE-L per section |

---

## Folder structure

```
benchmark/
├── cases/
│   └── case_01_template/      ← reference case (copy to add new cases)
│       ├── input.json         ← patient data (validated against input_schema.json)
│       └── metadata.json      ← case metadata
├── ground_truths/             ← answer keys (host-side only, never enter container)
│   └── case_01_template.json
├── shared/
│   ├── schemas/
│   │   ├── input_schema.json
│   │   ├── ground_truth_schema.json
│   │   └── metadata_schema.json
│   ├── tools/
│   │   ├── pubmed.py          ← NCBI E-utilities wrapper
│   │   └── rxnorm.py          ← NIH RxNav + OpenFDA wrapper
│   └── scoring/
│       ├── rouge_score.py     ← ROUGE scoring
│       ├── concept_f1.py      ← Concept-level F1 (embedding-based for DDx)
│       ├── ndcg.py            ← nDCG for ranked DDx (embedding-based)
│       └── embeddings.py      ← shared sentence-transformer model + cosine similarity
├── runner/
│   ├── agent.py               ← ClinicalAgent: plan → execute → memory loop
│   ├── planner.py             ← stage order + skip med stages if no medications
│   ├── executor.py            ← retries, validation, scratchpad, fallbacks
│   ├── state.py               ← AgentState, plan, working_memory, scratchpad, log
│   ├── validator.py           ← per-stage output JSON Schema checks
│   ├── llm_client.py          ← OpenAI + Pydantic structured chat
│   ├── stage_transcription.py … stage_report.py  ← one module per pipeline stage
│   └── langgraph_runner.py    ← legacy stub pipeline (tests / reference)
├── agent/
│   ├── agent_main.py          ← container entrypoint
│   ├── Dockerfile             ← agent image (no ground_truths/, no harness/)
│   └── requirements.txt       ← minimal container deps (no scoring libs)
├── harness/
│   └── harness.py             ← host-side orchestrator (scores, never enters container)
├── tests/
│   ├── test_scoring.py        ← unit tests for ROUGE / F1 / nDCG
│   ├── test_pipeline.py       ← unit tests for schema validation + stub nodes
│   ├── test_harness.py        ← unit tests for case discovery + score_case()
│   └── test_tools.py          ← integration tests for PubMed + RxNorm (needs network)
└── requirements.txt           ← host-side deps (harness, scoring, sentence-transformers, tests)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r benchmark/requirements.txt
```

### 2. Run unit tests (no network, no Docker)

```bash
pytest benchmark/tests/ -m "not integration" -v
```

### 3. Build the agent Docker image

```bash
docker build -t clinical-agent:latest -f benchmark/agent/Dockerfile benchmark/
```

### 4. Run the full harness

```bash
# Run all cases and print a results table
python benchmark/harness/harness.py

# Build image and run in one step
python benchmark/harness/harness.py --build

# Save each case’s raw JSON prediction (transcript, summary, DDx, SOAP, debug log)
python benchmark/harness/harness.py --save-predictions

# Custom results directory (CSVs + optional predictions/ subdirectory)
python benchmark/harness/harness.py --output-dir path/to/results

# Options
python benchmark/harness/harness.py --help
```

**Results layout:** By default the harness writes **`benchmark/results/run_<timestamp>_raw.csv`** and **`_summary.csv`**. With **`--save-predictions`**, it also writes **`benchmark/results/predictions/run_<timestamp>_<case_id>_trial<n>.json`** (full stdout payload per case, including optional **`_debug_execution_log`**).

The harness discovers **every** subdirectory of `benchmark/cases/` that contains **`input.json`**. Set **`OPENAI_API_KEY`** on the host before running so the harness can inject it into the container.

---

## How to implement or change the agent

The benchmark defines the **stage interface**; production logic lives in **`benchmark/runner/stage_*.py`**. Each stage’s **`run(context)`** returns a dict with **`reasoning`**, **`confidence`**, and **`output`** (the slice validated by `validator.py` and merged into memory). Use **`llm_client.chat`** or **`llm_client.chat_structured`** for model calls; use **`shared/tools/pubmed.py`** and **`rxnorm.py`** where appropriate (real API calls when networked).

To register different behavior without editing `agent.py`, tests may assign **`executor.STAGE_MAP`** before importing **`ClinicalAgent`**.

The six **`node_*`** functions in **`langgraph_runner.py`** are a **legacy** linear stub; prefer **`stage_*.py`** + **`ClinicalAgent`** for the Docker agent.

---

## How to add a new benchmark case

1. **Copy the template:**
   ```bash
   cp -r benchmark/cases/case_01_template benchmark/cases/case_XX_your_name
   ```

2. **Edit `input.json`** — set a unique `case_id`, fill in all fields, validate:
   ```bash
   python -c "
   import json, jsonschema, pathlib
   schema = json.loads(pathlib.Path('benchmark/shared/schemas/input_schema.json').read_text())
   data   = json.loads(pathlib.Path('benchmark/cases/case_XX_your_name/input.json').read_text())
   jsonschema.validate(data, schema)
   print('Valid!')
   "
   ```

3. **Create `benchmark/ground_truths/case_XX_your_name.json`** with the expected outputs for all six stages.

4. **Run the harness** to confirm the case loads and scores without errors:
   ```bash
   python benchmark/harness/harness.py --cases-dir benchmark/cases/case_XX_your_name
   ```

---

## Running integration tests (live API calls)

```bash
# Requires internet access — hits NCBI PubMed and NIH RxNav
pytest benchmark/tests/test_tools.py -m integration -v
```

Set `NCBI_API_KEY` to increase PubMed rate limits from 3 → 10 req/s:
```bash
export NCBI_API_KEY=your_key_here   # get one at https://www.ncbi.nlm.nih.gov/account/
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Required by all agents. Passed from host into container by the harness |
| `NCBI_API_KEY` | No | PubMed rate limit: 3 req/s without, 10 req/s with. Get one at ncbi.nlm.nih.gov/account |
| `BENCHMARK_ROOT` | No | Override benchmark root path (set automatically to `/app` in container) |
| `SCORING_EMBED_MODEL` | No | HuggingFace model for DDx semantic scoring. Default: `pritamdeka/S-PubMedBert-MS-MARCO` (~400 MB, downloaded automatically). Set to `all-MiniLM-L6-v2` for a lighter option |
| `SCORING_EMBED_THRESHOLD` | No | Cosine similarity cutoff for condition matching. Default: `0.90`. Lower = more partial credit for paraphrases |

---

## Data source policy

**Do not commit real patient data.** All cases must use `data_source: "synthetic"` or be de-identified per HIPAA Safe Harbor. MIMIC-IV data requires PhysioNet credentialing — see https://physionet.org/content/mimiciv/
