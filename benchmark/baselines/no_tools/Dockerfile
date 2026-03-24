FROM python:3.11-slim

LABEL description="No-tools pipeline ablation — 6-stage pipeline, LLM-only, no API calls"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY baselines/no_tools/no_tools_runner.py /app/no_tools_runner.py
COPY baselines/no_tools/no_tools_main.py   /app/no_tools_main.py
COPY runner/llm.py                          /app/llm.py
COPY shared/schemas/                        /app/shared/schemas/

ENV BENCHMARK_ROOT=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "no_tools_main.py"]
