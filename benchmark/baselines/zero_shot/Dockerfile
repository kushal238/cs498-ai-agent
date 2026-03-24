FROM python:3.11-slim

LABEL description="Zero-shot single-prompt baseline — one GPT-4o call, no pipeline, no tools"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY baselines/zero_shot/zero_shot_main.py /app/zero_shot_main.py
COPY runner/llm.py                          /app/llm.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "zero_shot_main.py"]
