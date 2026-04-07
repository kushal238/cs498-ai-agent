# Agent container for the clinical AI benchmark.
#
# Build context: benchmark/ directory (one level above this file)
# Build command (from benchmark/):
#   docker build -t clinical-agent:latest -f agent/Dockerfile .
#
# This image intentionally excludes:
#   - ground_truths/   (answer keys — harness-side only)
#   - harness/         (host orchestrator)
#   - cases/           (raw case files — input is injected via INPUT_JSON env var)

FROM python:3.11-slim

LABEL description="Clinical AI benchmark agent container"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY agent/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only agent code and shared utilities
COPY agent/agent_main.py        /app/agent_main.py
COPY runner/langgraph_runner.py /app/langgraph_runner.py
COPY runner/llm.py              /app/llm.py
COPY shared/                    /app/shared/

# Schema path resolution inside the container
ENV BENCHMARK_ROOT=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "agent_main.py"]
