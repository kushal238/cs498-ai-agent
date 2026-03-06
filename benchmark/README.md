# Clinical Workflow AI Benchmark

A SWE-bench-style benchmark for evaluating clinical AI agents. An agent receives a raw patient dialogue and chart notes, runs them through a six-stage pipeline, and returns a physician-ready SOAP report — all inside an isolated Docker container. The harness scores the output on the host without ever exposing the answer key to the agent.

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
│       ├── concept_f1.py      ← Concept-level F1
│       └── ndcg.py            ← nDCG for ranked DDx
├── runner/
│   └── langgraph_runner.py   ← LangGraph pipeline stub nodes + run_pipeline()
├── agent/
│   ├── agent_main.py          ← container entrypoint
│   └── Dockerfile             ← agent image (no ground_truths/, no harness/)
├── harness/
│   └── harness.py             ← host-side orchestrator (scores, never enters container)
├── tests/
│   ├── test_scoring.py        ← unit tests for ROUGE / F1 / nDCG
│   ├── test_pipeline.py       ← unit tests for schema validation + stub nodes
│   ├── test_harness.py        ← unit tests for case discovery + score_case()
│   └── test_tools.py          ← integration tests for PubMed + RxNorm (needs network)
└── requirements.txt
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

# Options
python benchmark/harness/harness.py --help
```

Expected output (stubs, no agent implemented):

```
[Harness] Found 1 case(s): ['case_01_template']
[Harness] Running case: case_01_template

========================================================================
BENCHMARK RESULTS
========================================================================

Case: case_01_template
----------------------------------------------------
  transcription_cleanup      rouge1=0.000  rouge2=0.000  rougeL=0.000
  clinical_summarization     rouge1=0.000  rouge2=0.000  rougeL=0.000
  differential_diagnosis     precision=0.000  recall=0.000  f1=0.000  ndcg=0.000
  medication_normalization   precision=0.000  recall=0.000  f1=0.000
  drug_interaction_check     precision=0.000  recall=0.000  f1=0.000
  final_report_generation    subjective_rougeL=0.000  ...
```

---

## How to implement an agent

The benchmark defines the interface; you write the agent. Implement the six node functions in `benchmark/runner/langgraph_runner.py`:

```python
def node_transcription_cleanup(state: dict) -> dict:
    # state["patient_transcript"] → return {"transcription_cleaned": "..."}

def node_clinical_summarization(state: dict) -> dict:
    # state["transcription_cleaned"], state["chart_notes"] → return {"clinical_summary": "..."}

def node_differential_diagnosis(state: dict) -> dict:
    # use find_supporting_citations() from shared/tools/pubmed.py
    # return {"differential_diagnosis": [{"condition": ..., "pmid": ..., "rationale": ...}]}

def node_medication_normalization(state: dict) -> dict:
    # use normalize_medication_list() from shared/tools/rxnorm.py
    # return {"normalized_medications": [{"original": ..., "rxnorm_id": ..., "ingredient": ...}]}

def node_drug_interaction_check(state: dict) -> dict:
    # use check_interactions() from shared/tools/rxnorm.py
    # return {"drug_interactions": [{"drug_a": ..., "drug_b": ..., "severity": ..., "recommendation": ...}]}

def node_final_report_generation(state: dict) -> dict:
    # return {"final_report": {"subjective": ..., "objective": ..., "assessment": ..., "plan": ...}}
```

The tools in `shared/tools/` are available inside the container and make real API calls to PubMed and NIH RxNav.

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
| `NCBI_API_KEY` | No | PubMed rate limit: 3 req/s without, 10 req/s with |
| `BENCHMARK_ROOT` | No | Override benchmark root path (set automatically in container) |

---

## Data source policy

**Do not commit real patient data.** All cases must use `data_source: "synthetic"` or be de-identified per HIPAA Safe Harbor. MIMIC-IV data requires PhysioNet credentialing — see https://physionet.org/content/mimiciv/
