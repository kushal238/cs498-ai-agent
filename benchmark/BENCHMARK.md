# Clinical Workflow AI Benchmark

A SWE-bench-style benchmark for evaluating clinical AI agents end-to-end. An agent receives a raw patient dialogue plus chart notes, runs a six-stage clinical reasoning pipeline, and emits a physician-ready SOAP report. The benchmark harness runs the agent inside an isolated Docker container, captures its JSON output, and scores it on the host against a held-out answer key. The agent never sees the ground truth.

This document describes only the benchmark itself — the case format, scoring code, harness, and contract any agent must satisfy. It does **not** describe the reference agent in this repo (see `benchmark/agent/` for that). Anyone with a Docker-packaged clinical agent that conforms to the I/O contract below can run this benchmark.

---

## Pipeline stages and metrics

| # | Stage                       | Input                              | Output                         | Metric                         |
|---|-----------------------------|------------------------------------|--------------------------------|--------------------------------|
| 1 | Transcription cleanup       | Raw dialogue                       | Cleaned transcript             | ROUGE-1/2/L + BERTScore (F1)   |
| 2 | Clinical summarization      | Transcript + chart notes           | Clinical summary               | ROUGE-1/2/L + BERTScore (F1)   |
| 3 | Differential diagnosis      | Summary                            | Ranked DDx list (PubMed-backed)| Concept F1 (semantic) + nDCG   |
| 4 | Medication normalization    | Medication list                    | RxNorm-mapped medications      | Concept F1                     |
| 5 | Drug-drug interaction check | Normalized medications             | Interaction list (severity + recommendation) | Concept F1 on pairs + ROUGE-L on recommendations |
| 6 | Final report generation     | All prior outputs                  | SOAP-format report             | ROUGE-L + BERTScore F1 per section |

- ROUGE uses `rouge-score` (Google Research). See `shared/scoring/rouge_score.py`.
- BERTScore uses `emilyalsentzer/Bio_ClinicalBERT` for clinical-domain semantic similarity. See `shared/scoring/bertscore.py`.
- Concept F1 for DDx uses embedding cosine similarity (`pritamdeka/S-PubMedBert-MS-MARCO`, threshold 0.90 by default) to credit clinically equivalent paraphrases. Med and DDI stages use exact token-bag matching against canonical RxNorm names. See `shared/scoring/concept_f1.py`.
- nDCG measures ranking quality of the DDx list against the ground-truth ranking. See `shared/scoring/ndcg.py`.

---

## Folder layout (benchmark-only)

```
benchmark/
├── cases/                         ← one folder per case, each with input.json + metadata.json
│   ├── case_01_template/          ← reference template — copy to add new cases
│   └── case_02 … case_10/
├── ground_truths/                 ← answer keys (host-side only, NEVER enter the container)
│   └── case_<id>.json
├── shared/
│   ├── schemas/
│   │   ├── input_schema.json      ← what cases must look like
│   │   ├── ground_truth_schema.json ← what your agent must emit
│   │   └── metadata_schema.json
│   ├── tools/                     ← optional helpers your agent MAY use
│   │   ├── pubmed.py              ← NCBI E-utilities wrapper
│   │   ├── rxnorm.py              ← NIH RxNav wrapper
│   │   └── fda.py                 ← OpenFDA wrapper
│   └── scoring/
│       ├── rouge_score.py         ← ROUGE
│       ├── bertscore.py           ← Bio_ClinicalBERT BERTScore
│       ├── concept_f1.py          ← concept-level F1 (semantic for DDx)
│       ├── ndcg.py                ← nDCG for ranked DDx
│       └── embeddings.py          ← shared sentence-transformer model
├── harness/
│   └── harness.py                 ← host-side orchestrator (case discovery, docker run, scoring, CSVs)
├── tests/                         ← unit + integration tests
└── requirements.txt               ← host-side deps (harness, scoring, tests)
```

The `agent/` and `baselines/` directories are **examples** of agents conforming to the contract; they are not part of the benchmark spec.

---

## Required external resources

| Resource | Required? | Why | How to get |
|----------|-----------|-----|------------|
| Docker | Yes | The harness builds and runs every agent in an isolated container | https://docs.docker.com/get-docker/ |
| Python 3.11+ | Yes | Runs the harness and scoring code on the host | https://www.python.org/downloads/ |
| `OPENAI_API_KEY` | Only if your agent uses OpenAI | Passed from host into the container by the harness | https://platform.openai.com/api-keys |
| `NCBI_API_KEY` | No | Raises PubMed rate limit from 3 to 10 req/s; only matters if your agent calls PubMed | https://www.ncbi.nlm.nih.gov/account/ |
| Internet access from container | If your agent calls external APIs (PubMed, RxNav, OpenFDA, OpenAI) | Use `--network=bridge` (default). For locked-down evaluation use `--network=none` | n/a |
| HuggingFace model downloads | First scoring run only | Auto-downloaded by `sentence-transformers` and `bert-score` (~400 MB + ~440 MB). Cached in `~/.cache/huggingface/` | n/a |

> **Disk and memory:** the BERTScore and DDx encoders together need ~1 GB on disk and around 2 GB of RAM at score time. Set `KMP_DUPLICATE_LIB_OK=TRUE` on macOS to avoid OpenMP runtime conflicts.

---

## Setup (host machine)

```bash
# 1. Clone or copy this repo, then from the repo root:
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install host-side dependencies (harness + scoring + tests):
pip install -r benchmark/requirements.txt

# 3. macOS only — avoid OpenMP duplicate-library aborts during BERTScore:
export KMP_DUPLICATE_LIB_OK=TRUE

# 4. If your agent uses OpenAI, export your key:
export OPENAI_API_KEY="sk-..."
```

Confirm the install with the unit tests (no network, no Docker required):

```bash
pytest benchmark/tests/ -m "not integration" -v
```

Optionally run the integration tests, which hit live PubMed and RxNav:

```bash
pytest benchmark/tests/test_tools.py -m integration -v
```

---

## The agent contract

The benchmark is agent-agnostic. To plug in your own clinical agent, package it as a Docker image that follows this contract:

### 1. Input — read JSON from stdin

When the container starts, the harness pipes a JSON object on stdin matching `shared/schemas/input_schema.json`. Required top-level keys:

- `case_id` (string)
- `data_source` (`"synthetic" | "agbonnet" | "mimic_iv"`)
- `patient_transcript` (string, raw doctor-patient dialogue)
- `chart_notes` (string, raw clinical notes)
- `medication_list` (array of strings)
- `patient_history` (object with `age`, `sex`, `chief_complaint`, `known_conditions`, `known_allergies`)
- `difficulty` (`"simple" | "moderate" | "complex"`)

See `benchmark/cases/case_02/input.json` for a worked example.

### 2. Output — print one JSON object to stdout

The container must write **a single JSON object on the last non-empty line of stdout** matching `shared/schemas/ground_truth_schema.json`. All debugging output must go to stderr. Required keys:

- case_id (string, must match the input case_id)
- transcription_cleaned (string)
- `clinical_summary` (string)
- `differential_diagnosis` (array of `{condition, pmid, rationale}`)
- `normalized_medications` (array of `{original, rxnorm_id, ingredient}`)
- `drug_interactions` (array of `{drug_a, drug_b, severity, recommendation}` — `severity` ∈ `minor | moderate | major | contraindicated | unknown`)
- `final_report` (object with `subjective`, `objective`, `assessment`, `plan`)

Anything else on stdout is tolerated as long as the **last** non-empty line is the prediction JSON. If parsing fails, the harness records the trial as failed.

### 3. Container constraints

- Must accept input on stdin (the harness invokes `docker run --rm --network=<mode> -i <image>` and pipes JSON in).
- Must respect `OPENAI_API_KEY` if your agent uses OpenAI — the harness passes it through with `-e OPENAI_API_KEY=$OPENAI_API_KEY`.
- Must finish within `--timeout` seconds (default 120, often raise to 180–300 for tool-using agents).
- Must **not** bake `benchmark/ground_truths/` or `benchmark/harness/` into the image. The reference Dockerfile (`benchmark/agent/Dockerfile`) shows the safe COPY pattern.

### 4. Building your image

> **Shortcut for the bundled reference images.** If you just want the agent + baseline images this repo ships with (`clinical-agent:latest`, `zero-shot-baseline:latest`, `no-tools-baseline:latest`), run `bash scripts/build_images.sh` from the repo root and skip the manual command below. The script `cd`s into `benchmark/` and runs the same `docker build` invocations described here, one per image, then lists the resulting tags.

Build context should be `benchmark/` so your image can `COPY shared/` for tool wrappers and (if you want them) the I/O schemas:

```bash
docker build -t my-clinical-agent:latest -f path/to/your/Dockerfile benchmark/
```

---

## Running an evaluation

> **Shortcut for the full sweep.** If you want the same 3-system × 10-case × 3-trial run this repo's paper used, run `bash scripts/run_full_benchmark.sh` from the repo root (after `bash scripts/build_images.sh` and `export OPENAI_API_KEY=...`). It invokes the harness three times — once each for `clinical-agent:latest`, `zero-shot-baseline:latest`, and `no-tools-baseline:latest` — with `--trials 3 --save-predictions --timeout 180`, tags each run, and writes everything under `benchmark/results/final_<timestamp>/<system>/`. The manual commands below are what that script runs under the hood; use them when you want to evaluate a custom agent or a single case.

The harness lives at `benchmark/harness/harness.py`. From the repo root:

```bash
# Single trial across all cases, default image name:
python benchmark/harness/harness.py --image my-clinical-agent:latest

# 3 trials per case, save raw JSON predictions, longer timeout:
python benchmark/harness/harness.py \
    --image my-clinical-agent:latest \
    --trials 3 \
    --timeout 300 \
    --save-predictions \
    --output-dir benchmark/results/my_run

# Run only one case while you iterate:
python benchmark/harness/harness.py \
    --image my-clinical-agent:latest \
    --cases-dir benchmark/cases/case_02 \
    --timeout 300

# Locked-down evaluation (no outbound network from the container):
python benchmark/harness/harness.py \
    --image my-clinical-agent:latest \
    --network none
```

### All harness flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | `clinical-agent:latest` | Docker image tag to run for each case |
| `--cases-dir` | `benchmark/cases/` | Where to look for case folders (any subdir containing `input.json`) |
| `--trials` | `1` | How many independent runs per case (means + stddevs are written to the summary CSV) |
| `--timeout` | `120` | Seconds before a stuck container is killed and marked as a failed trial |
| `--output-dir` | `benchmark/results/` | Where to write `*_raw.csv`, `*_summary.csv`, and (if requested) `predictions/` |
| `--save-predictions` | off | Persist each trial's full JSON prediction to `output-dir/predictions/` for later rescoring |
| `--build` | off | `docker build` the image before running (uses `benchmark/agent/Dockerfile` — change it if you point at your own image) |
| `--network` | `bridge` | Container network mode. Use `none` to block all outbound traffic for a hermetic eval |
| `--run-tag` | empty | Optional tag prepended to the run id (e.g. `agent`, `zs`); useful when running multiple systems into the same `--output-dir` |

### Comparing multiple agents in one go

`scripts/run_full_benchmark.sh` is a convenience wrapper that runs the same case set against three image tags (3 trials each) and dumps everything under `benchmark/results/final_<timestamp>/<system>/`. Adapt the image names in that script if you want to compare your own agent against the reference one.

---

## Interpreting results

Each run writes two CSVs into `--output-dir`:

- `run_<timestamp>_raw.csv`
  Schema: `run_id, case_id, trial, stage, metric, value`. One row per (case × trial × stage × metric). This is the source of truth — averaging happens downstream.
- `run_<timestamp>_summary.csv`
  Schema: `run_id, case_id, stage, metric, mean, stddev`. Mean and stddev computed across the `--trials` runs of each case. Use this for per-case reporting.

If you passed `--save-predictions`, you also get `predictions/run_<timestamp>_<case_id>_trial<n>.json` — the full JSON your container emitted, useful for spot-checking failures or rescoring later without re-running the agent.

### Stage-by-stage interpretation

- **Stage 1 / 2 / 6 (free text)** — read `rougeL` and `bertscore_f1` together. ROUGE-L rewards lexical overlap; BERTScore F1 rewards semantic similarity even when the agent paraphrases. Both range 0–1; higher is better. A high BERTScore F1 with low ROUGE-L usually means the agent rephrased correctly. The reverse usually means the agent copied surface forms but missed clinical meaning.
- **Stage 3 (differential diagnosis)** — `f1` reflects whether the right conditions were named (semantically); `ndcg` reflects whether they were ranked correctly. A model that names every plausible condition gets high recall but may have low nDCG if the top-1 is wrong.
- **Stage 4 (medication normalization)** — `f1` over RxNorm ingredient names. Token-bag matching, so spelling and word order matter. Brand vs generic mismatches will cost recall.
- **Stage 5 (drug interactions)** — `f1` is over (drug_a, drug_b) pairs (order-insensitive). `recommendation_rougeL` averages ROUGE-L between predicted and reference recommendation text **for matched pairs**. A pasted FDA label table may achieve high pair F1 but very low recommendation ROUGE-L.
- **Stage 6 (final report)** — four metrics per SOAP section (`subjective_rougeL`, `subjective_bertscore_f1`, etc.). Aggregating to a single "report score" is fine for headline numbers; treat it as the mean of the four sections.

### Aggregating across runs

For three-system comparisons (e.g. ours vs. zero-shot vs. no-tools) the repo includes a small post-processor:

```bash
# Re-score saved predictions with the current scoring code (handy if you
# bumped scoring after the run, or BERTScore didn't run in-process):
python scripts/rescore_predictions.py benchmark/results/final_<timestamp>

# Build the wide comparison CSV, markdown table, and stage-breakdown bar chart:
python scripts/aggregate_results.py benchmark/results/final_<timestamp>
```

`aggregate_results.py` produces `comparison_summary.csv`, `comparison_summary.md`, and `stage_breakdown.png` — the latter is the per-stage bar chart used in the paper.

### Sanity checks

- A case row with all metrics at exactly 0 usually means the agent's container failed (parse error, timeout, or non-zero exit). Check stderr in the harness output and the `predictions/` JSON.
- Identical scores across trials with `--trials > 1` mean either your agent is fully deterministic (e.g. `temperature=0`) or the trials all collapsed to the same failure path.
- A `recommendation_rougeL` of `0.0` with non-zero pair F1 is normal when the agent identifies the right pair but writes a free-text rationale far from the ground-truth phrasing.

---

## Adding a new case

1. Copy the template:
   ```bash
   cp -r benchmark/cases/case_01_template benchmark/cases/case_XX_yourname
   ```
2. Edit `case_XX_yourname/input.json`. Set `case_id` to match the folder name. Validate against the schema:
   ```bash
   python -c "
   import json, jsonschema, pathlib
   schema = json.loads(pathlib.Path('benchmark/shared/schemas/input_schema.json').read_text())
   data   = json.loads(pathlib.Path('benchmark/cases/case_XX_yourname/input.json').read_text())
   jsonschema.validate(data, schema)
   print('Valid')
   "
   ```
3. Create `benchmark/ground_truths/case_XX_yourname.json` with the expected outputs for all six stages, matching `shared/schemas/ground_truth_schema.json`.
4. Run the harness on just that case to confirm scoring works end-to-end:
   ```bash
   python benchmark/harness/harness.py \
       --image my-clinical-agent:latest \
       --cases-dir benchmark/cases/case_XX_yourname \
       --timeout 300
   ```

> **Data policy.** Do not commit real patient data. All cases must be `data_source: "synthetic"`, `data_source: "agbonnet"`, or de-identified per HIPAA Safe Harbor. MIMIC-IV (`data_source: "mimic_iv"`) requires PhysioNet credentialing; do not check raw MIMIC text into git. See https://physionet.org/content/mimiciv/.

---

## Optional environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | unset | Forwarded to the container by the harness if set on the host |
| `NCBI_API_KEY` | unset | PubMed rate limit lift (3 → 10 req/s) for tool wrappers in `shared/tools/pubmed.py` |
| `BENCHMARK_ROOT` | repo `benchmark/` | Override benchmark root path for harness/scoring imports (set automatically to `/app` inside the reference container) |
| `SCORING_EMBED_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | Encoder for DDx semantic concept F1. Set to `all-MiniLM-L6-v2` for a smaller alternative |
| `SCORING_EMBED_THRESHOLD` | `0.90` | Cosine threshold for DDx condition matching. Lower = more partial credit for paraphrases |
| `BERTSCORE_MODEL` | `emilyalsentzer/Bio_ClinicalBERT` | Encoder for free-text BERTScore. Override to swap in another clinical encoder (e.g. PubMedBERT) |
| `BERTSCORE_LAYER` | `9` | Hidden-state layer used by BERTScore |
| `KMP_DUPLICATE_LIB_OK` | unset | Set to `TRUE` on macOS to avoid OpenMP runtime conflicts during BERTScore |

---

## Reproducing a published number

If you want to compare your agent against numbers already in this repo:

1. Build your agent image with a unique tag (e.g. `my-agent:v1`).
2. Run the full sweep with `scripts/run_full_benchmark.sh`, editing the `run_system` lines to point at your tag (and removing the tags you do not want to re-run).
3. Re-score and aggregate:
   ```bash
   python scripts/rescore_predictions.py benchmark/results/final_<timestamp>
   python scripts/aggregate_results.py   benchmark/results/final_<timestamp>
   ```
4. Compare `comparison_summary.md` against the version under the timestamped run from this repo.

The benchmark itself is fully deterministic given fixed cases and ground truths — any variance you see across reruns comes from your agent (model sampling, network responses, retries), not from scoring.
