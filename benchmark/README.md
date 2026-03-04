# Clinical Workflow AI Benchmark

A benchmark scaffold for evaluating an agentic AI system that takes a
patient dialogue + chart notes as input and produces a physician-ready
clinical report as output.

> **Status**: Scaffold only — workflow nodes and scoring functions are stubs.
> See the TODO comments throughout the codebase for what to implement next.

---

## What this repo is

This project benchmarks a six-stage clinical AI pipeline:

| # | Stage | Input | Output |
|---|-------|-------|--------|
| 1 | Transcription cleanup | Raw dialogue | Cleaned transcript |
| 2 | Clinical summarization | Cleaned transcript + chart notes | Clinical summary |
| 3 | Differential diagnosis | Summary | Ranked DDx list (PubMed-backed) |
| 4 | Medication normalization | Medication list | RxNorm-mapped medications |
| 5 | Drug-drug interaction check | Normalized medications | Interaction list (NIH RxNav) |
| 6 | Final report generation | All prior outputs | SOAP-format report |

There are 10 benchmark cases (only `case_01_template` is populated). Each
case runs the full pipeline and is scored against a human-curated ground truth.

---

## Folder structure

```
benchmark/
├── cases/
│   └── case_01_template/      ← reference case (copy to add new cases)
│       ├── input.json         ← patient data (validated against input_schema.json)
│       ├── ground_truth.json  ← expected outputs (fill in TODOs)
│       └── metadata.json      ← case metadata
├── shared/
│   ├── schemas/
│   │   ├── input_schema.json
│   │   ├── ground_truth_schema.json
│   │   └── metadata_schema.json
│   ├── tools/
│   │   ├── rxnorm.py          ← NIH RxNav API wrapper (stub)
│   │   └── pubmed.py          ← NCBI E-utilities wrapper (stub)
│   ├── scoring/
│   │   ├── rouge_score.py     ← ROUGE scoring (stub)
│   │   ├── concept_f1.py      ← Concept-level F1 (stub)
│   │   └── ndcg.py            ← nDCG for ranked DDx (stub)
│   └── mock_data/
│       └── synthetic_patient_01.json
├── runner/
│   ├── langgraph_runner.py    ← LangGraph pipeline runner (stub nodes)
│   └── evaluate.py            ← Batch evaluator + results table
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## How to add a new benchmark case

1. **Copy the template:**
   ```bash
   cp -r benchmark/cases/case_01_template benchmark/cases/case_XX_your_name
   ```

2. **Edit `input.json`:**
   - Set a unique `case_id` (e.g. `"case_02_diabetes_polyp"`)
   - Fill in `data_source`, `difficulty`, `patient_history`, `patient_transcript`,
     `chart_notes`, and `medication_list`
   - Validate against the schema:
     ```bash
     python -c "
     import json, jsonschema, pathlib
     schema = json.loads(pathlib.Path('benchmark/shared/schemas/input_schema.json').read_text())
     data   = json.loads(pathlib.Path('benchmark/cases/case_XX_your_name/input.json').read_text())
     jsonschema.validate(data, schema)
     print('Valid!')
     "
     ```

3. **Fill in `ground_truth.json`:**
   - Replace every `"TODO: ..."` string with the real expected output
   - Replace every `null` with the correct typed value
   - Keep `case_id` consistent with `input.json`

4. **Fill in `metadata.json`:**
   - Set `title`, `description`, `workflow_stages_tested`, `created_by`, and `notes`

5. **Run the reference case** (see below) to confirm the runner loads your case
   without schema errors.

---

## How to run the reference case

**With Python directly:**

```bash
# From the repo root
cd /path/to/cs498-ai-agent

pip install langgraph langchain jsonschema requests rouge-score scikit-learn numpy pytest

python benchmark/runner/langgraph_runner.py benchmark/cases/case_01_template
```

Expected output (until workflow nodes are implemented):

```
[Runner] Loaded and validated case: case_01_template
[Stage 1] transcription_cleanup — NOT IMPLEMENTED
[Stage 2] clinical_summarization — NOT IMPLEMENTED
[Stage 3] differential_diagnosis — NOT IMPLEMENTED
[Stage 4] medication_normalization — NOT IMPLEMENTED
[Stage 5] drug_interaction_check — NOT IMPLEMENTED
[Stage 6] final_report_generation — NOT IMPLEMENTED

[Runner] Pipeline complete for case: case_01_template
```

**With Docker:**

```bash
cd benchmark/docker
docker compose up --build
```

---

## How to run all cases with evaluate.py

```bash
python benchmark/runner/evaluate.py
# or specify a custom cases directory:
python benchmark/runner/evaluate.py --cases-dir benchmark/cases
```

This will discover every folder under `benchmark/cases/` that contains
`input.json`, run the pipeline, score against `ground_truth.json`, and
print a results table.

---

## Data source notes

| Source | Description | Access |
|--------|-------------|--------|
| `synthetic` | Fully synthetic, no real patients | Open |
| `agbonnet` | Agbonnet et al. anonymized dataset | Check paper for terms |
| `mimic_iv` | MIMIC-IV clinical notes | **Requires PhysioNet credentialing** — see https://physionet.org/content/mimiciv/ |

**Do not commit any real patient data.** All cases in this repo must use
`data_source: "synthetic"` or be de-identified in accordance with HIPAA
Safe Harbor / MIMIC data use agreements.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NCBI_API_KEY` | No | Increases PubMed rate limit from 3 to 10 req/s. Get one at https://www.ncbi.nlm.nih.gov/account/ |

---

## Running tests

```bash
pytest benchmark/
```

(No tests are written yet — add them under `benchmark/tests/` as you
implement workflow nodes and scoring functions.)
